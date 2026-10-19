import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from .backbone import ResNet_8_2, CovNextV2_nano, CovNext_dinov3, DINOv3B, DINOv3B_resnet, HighResFormerplusplus_v3, HighResFormerplusplus_v3_withfpn
from .backbone import HighResFormerplusplus_structural_extractor,HighResFormerplusplus_v4
from .utils.position_encoding import PositionEncodingSine
from .xoftr_module import LocalFeatureTransformer, FineProcess, CoarseMatching, FineSubMatching
from src.jamma.utils.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from src.jamma.mamba_module import JointMamba
from kornia.utils import create_meshgrid

class XoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

class XoFTR_Light(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== 写死的配置参数 ==========
        self.W_f = 5
        self.W_m = 3  # fine窗口大小
        
        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True
        
        # Fine层配置
        self.fine_d_model = 128
        self.fine_inference = False
        self.fine_dsmax_temperature = 0.1
        self.fine_thr = 0.1
        

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device
        
        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:], 
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        
        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape
        
        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:]
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    
    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0//scale, W0//scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0), 'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)

       
    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载权重"""
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_dino(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        # self.backbone = EfficientDINOv3Backbone(
        #         model_type=config['dino']['model_type'],
        #         freeze=config['dino']['freeze']
        #     )
        self.backbone = DINOv3B(config['resnet'], layer_strategy="balanced")
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_dino_resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        # self.backbone = EfficientDINOv3Backbone(
        #         model_type=config['dino']['model_type'],
        #         freeze=config['dino']['freeze']
        #     )
        self.backbone = DINOv3B(config['resnet'], layer_strategy="balanced")
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_dino_resnet_panmatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        # self.backbone = EfficientDINOv3Backbone(
        #         model_type=config['dino']['model_type'],
        #         freeze=config['dino']['freeze']
        #     )
        self.backbone = HighResFormerplusplus_v3(features=64, output_corr_chans=[128, 196, 256, 392])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        # print(f"feat_c0:{feat_c0.shape}")
        # print(f"feat_m0:{feat_m0.shape}")
        # print(f"feat_f0:{feat_f0.shape}")
        # def normalize_feat(feat):
        #     # 层归一化
        #     feat = F.layer_norm(feat, feat.shape[1:])
        #     # 或者简单的缩放
        #     # feat = feat * 0.1
        #     return feat
        
        # feat_c0 = normalize_feat(feat_c0)
        # feat_c1 = normalize_feat(feat_c1)
        # feat_m0 = normalize_feat(feat_m0)
        # feat_m1 = normalize_feat(feat_m1)
        # feat_f0 = normalize_feat(feat_f0)
        # feat_f1 = normalize_feat(feat_f1)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # print(f"feat_c0:{feat_c0.shape}")
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_dino_resnet_panmatch_light(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小
        
        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3(features=128, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device
        
        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:], 
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        
        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape
        
        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:]
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    
    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0//scale, W0//scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0), 'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_dino_resnet_panmatch_light_withfpn(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小
        
        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3_withfpn(features=96, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device
        
        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:], 
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f,saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape
        
        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    
    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0//scale, W0//scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0), 'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_Light_convnextv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== 写死的配置参数 ==========
        self.W_f = 5
        self.W_m = 3  # fine窗口大小
        
        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True
        
        # Fine层配置
        self.fine_d_model = 128
        self.fine_inference = False
        self.fine_dsmax_temperature = 0.1
        self.fine_thr = 0.1
        

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = CovNextV2_nano()
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device
        
        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:], 
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        # if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
        #     feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
        #     (feat_c0, feat_c1) = feats_c.split(data['bs'])
        #     (feat_m0, feat_m1) = feats_m.split(data['bs'])
        #     (feat_f0, feat_f1) = feats_f.split(data['bs'])
        # else:  # handle different input shapes
        #     feat_c0, feat_m0, feat_f0 = self.backbone(image0)
        #     feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        self.backbone(data)
        feat_c0, feat_c1 = data['feat_c0'], data['feat_c1']
        feat_m0, feat_m1 = data['feat_m0'], data['feat_m1']
        feat_f0, feat_f1 = data['feat_f0'], data['feat_f1']
        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape
        
        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:]
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_c0'].transpose(1, 2), data['feat_c1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module       
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data) 

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
    
    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0//scale, W0//scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0), 'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_c0'].flatten(2, 3), data['feat_c1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_c0': desc0,
            'feat_c1': desc1,
        })

        # print(data)

        self.joint_mamba(data)

       
    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载权重"""
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_convnext_dino(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = CovNext_dinov3()
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching= FineSubMatching(config)


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2,3], keepdim=True)
        image0_std = data['image0'].std(dim=[2,3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2,3], keepdim=True)
        image1_std = data['image1'].std(dim=[2,3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        # if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
        #     feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
        #     (feat_c0, feat_c1) = feats_c.split(data['bs'])
        #     (feat_m0, feat_m1) = feats_m.split(data['bs'])
        #     (feat_f0, feat_f1) = feats_f.split(data['bs'])
        # else:  # handle different input shapes
        #     feat_c0, feat_m0, feat_f0 = self.backbone(image0)
        #     feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        self.backbone(data)
        feat_c0, feat_c1 = data['feat_c0'], data['feat_c1']
        feat_m0, feat_m1 = data['feat_m0'], data['feat_m1']
        feat_f0, feat_f1 = data['feat_f0'], data['feat_f1']
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class XoFTR_dino_resnet_panmatch_xoftr(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3_withfpn(features=96, output_corr_chans=[128, 196, 256, 392])

        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        # 关键点编码器
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)


        # ========== 2. 粗匹配 ==========
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)



    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class XoFTR_convnextv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ========== 写死的配置参数 ==========
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Fine层配置
        self.fine_d_model = 128
        self.fine_inference = False
        self.fine_dsmax_temperature = 0.1
        self.fine_thr = 0.1

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = CovNextV2_nano()
        # 关键点编码器
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        # 关键点编码器
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        # if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
        #     feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
        #     (feat_c0, feat_c1) = feats_c.split(data['bs'])
        #     (feat_m0, feat_m1) = feats_m.split(data['bs'])
        #     (feat_f0, feat_f1) = feats_f.split(data['bs'])
        # else:  # handle different input shapes
        #     feat_c0, feat_m0, feat_f0 = self.backbone(image0)
        #     feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        self.backbone(data)
        feat_c0, feat_c1 = data['feat_c0'], data['feat_c1']
        feat_m0, feat_m1 = data['feat_m0'], data['feat_m1']
        feat_f0, feat_f1 = data['feat_f0'], data['feat_f1']
        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:]
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        # ========== 2. 粗匹配 ==========
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载权重"""
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class XoFTR_dino_resnet_panmatch_without_fpn(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3(features=96, output_corr_chans=[128, 196, 256, 392])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:]
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # ========== 2. 粗匹配 ==========
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class GeoSAM_without_RopE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3_withfpn(features=96, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        # self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0 // scale, W0 // scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0),
                           'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0
        desc1 = desc1
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class GeoSAM_reverse_GGSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v4(features=96, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0 // scale, W0 // scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0),
                           'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)


class GeoSAM_ablation_structure(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_structural_extractor(features=96, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0 // scale, W0 // scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0),
                           'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)




# class GeoSAM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Misc
#         self.config = config
#         self.W_f = 5
#         self.W_m = 3  # fine窗口大小
#
#         # Coarse层配置
#         self.coarse_d_model = 256
#         self.coarse_thr = 0.2
#         self.coarse_border_rm = 2
#         self.coarse_dsmax_temperature = 0.1
#         self.coarse_train_percent = 0.2
#         self.coarse_train_pad_num_gt_min = 200
#         self.coarse_inference = False
#         self.coarse_use_sm = True
#
#         # Modules
#         # self.backbone = ResNet_8_2(config['resnet'])
#         self.backbone = HighResFormerplusplus_v3_withfpn(features=96, output_corr_chans=[128, 196, 256, 392])
#         # 关键点编码器
#         self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
#         # Mamba交互层
#         self.joint_mamba = JointMamba(
#             self.coarse_d_model, 4,
#             rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
#             profiler=None
#         )
#         self.coarse_matching = CoarseMatching(config['match_coarse'])
#         self.fine_process = FineProcess(config)
#         self.fine_matching = FineSubMatching(config)
#
#     def forward(self, data):
#         """完整的前向过程"""
#         # 获取设备信息
#         device = data['image0'].device
#
#         # ========== 1. 图像预处理和特征提取 ==========
#         bs = data['image0'].size(0)
#         data.update({
#             'bs': bs,
#             'hw0_i': data['image0'].shape[2:],
#             'hw1_i': data['image1'].shape[2:],
#             'W_f': self.W_f,
#             'W_m': self.W_m,
#             'fine_window_size': self.W_f
#         })
#
#         eps = 1e-6
#
#         image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
#         image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
#         image0 = (data['image0'] - image0_mean) / (image0_std + eps)
#
#         image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
#         image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
#         image1 = (data['image1'] - image1_mean) / (image1_std + eps)
#
#         if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
#             feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
#             (feat_c0, feat_c1) = feats_c.split(data['bs'])
#             (feat_m0, feat_m1) = feats_m.split(data['bs'])
#             (feat_f0, feat_f1) = feats_f.split(data['bs'])
#             # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
#             (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
#         else:  # handle different input shapes
#             # 【修改】用 4 个变量接住返回值
#             feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
#             feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
#             # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
#             # feat_c1, feat_m1, feat_f1 = self.backbone(image1)
#
#         # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
#
#         # 记录维度信息
#         B, C, H8, W8 = feat_c0.shape
#         _, _, H4, W4 = feat_m0.shape
#         _, _, H2, W2 = feat_f0.shape
#
#         data.update({
#             'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
#             'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
#             'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
#             'c': C,
#             'h_8': H8, 'w_8': W8,
#             'h_4': H4, 'w_4': W4,
#             'h_2': H2, 'w_2': W2,
#             'feat_8_0': feat_c0,
#             'feat_8_1': feat_c1,
#             'hw0_i': data['image0'].shape[2:],
#             'hw1_i': data['image1'].shape[2:],
#             # 【新增】把切分好的掩码存入 data 字典！
#             # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
#             'mask_8_0': mask_8_0,
#             'mask_8_1': mask_8_1
#         })
#         feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
#         # ========== 2. 粗匹配 ==========
#         self.coarse_match_preprocess(data)
#         feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)
#
#         mask_c0 = mask_c1 = None  # mask is useful in training
#         if 'mask0' in data:
#             mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
#         # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
#
#         # 3. match coarse-level
#         self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
#
#         # 4. fine-level matching module
#         feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
#                                                            feat_m0, feat_m1,
#                                                            feat_c0, feat_c1,
#                                                            feat_c0_pre, feat_c1_pre,
#                                                            data)
#
#         # 5. match fine-level and sub-pixel refinement
#         self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
#
#     def coarse_match_preprocess(self, data):
#         B0, _, H0, W0 = data['image0'].shape
#         scale = 8
#         h0_8, w0_8 = H0 // scale, W0 // scale
#         device0 = data['image0'].device
#         grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0),
#                            'h w t->(h w) t')] * B0  # kpt_xy
#         grid_8 = torch.stack(grid0, 0)
#
#         # print(grid_8.shape)
#         data.update({
#             'grid_8': grid_8,
#         })
#         desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
#         # print(f"coarse_match desc0 shape1: {desc0.shape}")
#         # print(f"coarse_match desc1 shape1: {desc1.shape}")
#         kpts0, kpts1 = data['grid_8'], data['grid_8']
#         # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
#         # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
#         # Keypoint normalization.
#         kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
#         kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
#         # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
#         # print(f"coarse_match kpts1 shape2: {kpts1.shape}")
#
#         kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
#         # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
#         # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
#         # desc3 = self.kenc(kpts0)
#         # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
#         # desc4 = self.kenc(kpts1)
#         # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
#         desc0 = desc0 + self.kenc(kpts0)
#         desc1 = desc1 + self.kenc(kpts1)
#         # print(f"coarse_match desc0 shape2: {desc0.shape}")
#         data.update({
#             'feat_8_0': desc0,
#             'feat_8_1': desc1,
#         })
#
#         # print(data)
#
#         self.joint_mamba(data)

class GeoSAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.W_f = 5
        self.W_m = 3  # fine窗口大小

        # Coarse层配置
        self.coarse_d_model = 256
        self.coarse_thr = 0.2
        self.coarse_border_rm = 2
        self.coarse_dsmax_temperature = 0.1
        self.coarse_train_percent = 0.2
        self.coarse_train_pad_num_gt_min = 200
        self.coarse_inference = False
        self.coarse_use_sm = True

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3_withfpn(features=96, output_corr_chans=[128, 196, 256, 392])
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """完整的前向过程"""
        # 获取设备信息
        device = data['image0'].device

        # ========== 1. 图像预处理和特征提取 ==========
        bs = data['image0'].size(0)
        data.update({
            'bs': bs,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'W_f': self.W_f,
            'W_m': self.W_m,
            'fine_window_size': self.W_f
        })

        eps = 1e-6

        image0_mean = data['image0'].mean(dim=[2, 3], keepdim=True)
        image0_std = data['image0'].std(dim=[2, 3], keepdim=True)
        image0 = (data['image0'] - image0_mean) / (image0_std + eps)

        image1_mean = data['image1'].mean(dim=[2, 3], keepdim=True)
        image1_std = data['image1'].std(dim=[2, 3], keepdim=True)
        image1 = (data['image1'] - image1_mean) / (image1_std + eps)

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f, saliency_mask = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
            # 【新增】将掩码也沿着 batch 维度切分为图 0 和图 1 的掩码
            (mask_8_0, mask_8_1) = saliency_mask.split(data['bs'])
        else:  # handle different input shapes
            # 【修改】用 4 个变量接住返回值
            feat_c0, feat_m0, feat_f0, mask_8_0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1, mask_8_1 = self.backbone(image1)
            # feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            # feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")

        # 记录维度信息
        B, C, H8, W8 = feat_c0.shape
        _, _, H4, W4 = feat_m0.shape
        _, _, H2, W2 = feat_f0.shape

        data.update({
            'hw0_c': (H8, W8), 'hw1_c': (H8, W8),
            'hw0_m': (H4, W4), 'hw1_m': (H4, W4),
            'hw0_f': (H2, W2), 'hw1_f': (H2, W2),
            'c': C,
            'h_8': H8, 'w_8': W8,
            'h_4': H4, 'w_4': W4,
            'h_2': H2, 'w_2': W2,
            'feat_8_0': feat_c0,
            'feat_8_1': feat_c1,
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            # 【新增】把切分好的掩码存入 data 字典！
            # 这样后面的 JointMamba 就能直接从 data 里拿到它们了
            'mask_8_0': mask_8_0,
            'mask_8_1': mask_8_1
        })
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()
        # ========== 2. 粗匹配 ==========
        self.coarse_match_preprocess(data)
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(feat_f0, feat_f1,
                                                           feat_m0, feat_m1,
                                                           feat_c0, feat_c1,
                                                           feat_c0_pre, feat_c1_pre,
                                                           data)

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def coarse_match_preprocess(self, data):
        B0, _, H0, W0 = data['image0'].shape
        scale = 8
        h0_8, w0_8 = H0 // scale, W0 // scale
        device0 = data['image0'].device
        grid0 = [rearrange((create_meshgrid(h0_8, w0_8, False, device0) * scale).squeeze(0),
                           'h w t->(h w) t')] * B0  # kpt_xy
        grid_8 = torch.stack(grid0, 0)

        # print(grid_8.shape)
        data.update({
            'grid_8': grid_8,
        })
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        # print(f"coarse_match desc0 shape1: {desc0.shape}")
        # print(f"coarse_match desc1 shape1: {desc1.shape}")
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # print(f"coarse_match kpts0 shape1: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape1: {kpts1.shape}")
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape[-2:])
        # print(f"coarse_match kpts0 shape2: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape2: {kpts1.shape}")

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        # print(f"coarse_match kpts0 shape3: {kpts0.shape}")
        # print(f"coarse_match kpts1 shape3: {kpts1.shape}")
        # desc3 = self.kenc(kpts0)
        # print(f"coarse_match self.kenc(kpts0) shape: {desc3.shape}")
        # desc4 = self.kenc(kpts1)
        # print(f"coarse_match self.kenc(kpts1) shape: {desc4.shape}")
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        # print(f"coarse_match desc0 shape2: {desc0.shape}")
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        # print(data)

        self.joint_mamba(data)

