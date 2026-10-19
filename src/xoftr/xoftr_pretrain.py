import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from .backbone import ResNet_8_2, DINOv3B, CovNextV2_nano, CovNext_dinov3, HighResFormerplusplus_v3
from .utils.position_encoding import PositionEncodingSine
from .xoftr_module import LocalFeatureTransformer, FineProcess
from src.jamma.utils.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from src.jamma.mamba_module import JointMamba
from src.jamma.matching_module import CoarseMatching, FineSubMatching
from kornia.utils import create_meshgrid


class XoFTR_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.patch_size = config["pretrain_patch_size"]

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.fine_process = FineProcess(config)
        self.mask_token_f = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][2], 1, 1))
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
    
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)

    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat
        

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

        image0 = data["image0_norm"] if "image0_norm" in data else data["image0"]
        image1 = data["image1_norm"] if "image1_norm" in data else data["image1"]

        mask0 = mask1 = None  # mask fro madded images
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # mask input images
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        data.update({"masked_image0":image0.clone().detach().cpu(),
                     "masked_image1":image1.clone().detach().cpu()})

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # mask output layers of backbone and replace with trainable token
        feat_c0 = self.mask_layer(feat_c0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask0,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask1,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_m0 = self.mask_layer(feat_m0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask0,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask1,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_f0 = self.mask_layer(feat_f0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask0,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask1,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching module
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. Coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})


        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class XoFTR_Pretrain_Light(nn.Module):
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
        
        # MAE配置
        self.patch_size = config.get("pretrain_patch_size", 64)
        
        # ========== 模型组件 ==========
        # 初始化backbone
        self.backbone = ResNet_8_2(config['resnet'])
        
        # 获取维度
        block_dims = config['resnet']["block_dims"]
        self.dim_f = block_dims[0]
        self.dim_m = block_dims[1]
        self.dim_c = block_dims[2]
        
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
        self.fine_process = FineProcess(config)
        
        # 掩码标记
        self.mask_token_f = nn.Parameter(torch.zeros(1, block_dims[0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, block_dims[1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, block_dims[2], 1, 1))
        
        # 初始化
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)
        
        self.act = nn.GELU()
    
    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat

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

        image0 = data.get("image0_norm", data["image0"])
        image1 = data.get("image1_norm", data["image1"])
        
        # 调试：确保输入是单通道
        # print(f"DEBUG: Before mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        if image0.shape[1] != 1:
            print(f"ERROR: image0 has {image0.shape[1]} channels, expected 1")
            raise ValueError(f"Input image0 has {image0.shape[1]} channels, expected 1")
        
        mask0 = mask1 = None
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # 掩码输入图像
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        
        # print(f"DEBUG: After mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: After mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        # 保存掩码后的图像用于可视化
        data.update({
            "masked_image0": image0.clone().detach().cpu(),
            "masked_image1": image1.clone().detach().cpu()
        })
        
        # 分别提取特征 - 添加更多调试信息
        # print(f"DEBUG: Before backbone - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before backbone - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        
        # 掩码特征层
        feat_c0 = self.mask_layer(feat_c0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask0,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask1,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        
        feat_m0 = self.mask_layer(feat_m0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask0,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask1,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        
        feat_f0 = self.mask_layer(feat_f0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask0,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask1,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        
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

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        # feat_c0, feat_c1 = feat_c0.flatten(2, 3).transpose(1, 2), feat_c1.flatten(2, 3).transpose(1, 2)

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})
        
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)
        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})
    
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
    
class XoFTR_Pretrain_dino(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.patch_size = config["pretrain_patch_size"]

        # Modules
        # self.backbone = ResNet_8_2(config['resnet'])
        # self.backbone = EfficientDINOv3Backbone(config['dinov3'])
        self.backbone = DINOv3B(config['resnet'], layer_strategy="balanced")
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.fine_process = FineProcess(config)
        self.mask_token_f = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][2], 1, 1))
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
    
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)

    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat
        

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

        image0 = data["image0_norm"] if "image0_norm" in data else data["image0"]
        image1 = data["image1_norm"] if "image1_norm" in data else data["image1"]

        mask0 = mask1 = None  # mask fro madded images
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # mask input images
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        data.update({"masked_image0":image0.clone().detach().cpu(),
                     "masked_image1":image1.clone().detach().cpu()})

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # mask output layers of backbone and replace with trainable token
        feat_c0 = self.mask_layer(feat_c0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask0,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask1,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_m0 = self.mask_layer(feat_m0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask0,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask1,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_f0 = self.mask_layer(feat_f0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask0,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask1,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching module
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. Coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})


        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
    
class XoFTR_Pretrain_Light_convnextv2(nn.Module):
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
        
        # MAE配置
        self.patch_size = config.get("pretrain_patch_size", 64)
        
        # ========== 模型组件 ==========
        # 初始化backbone
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = CovNextV2_nano()
        
        # 获取维度
        block_dims = config['resnet']["block_dims"]
        self.dim_f = block_dims[0]
        self.dim_m = block_dims[1]
        self.dim_c = block_dims[2]
        
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
        self.fine_process = FineProcess(config)
        
        # 掩码标记
        self.mask_token_f = nn.Parameter(torch.zeros(1, block_dims[0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, block_dims[1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, block_dims[2], 1, 1))
        
        # 初始化
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)
        
        self.act = nn.GELU()
    
    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat

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

        image0 = data.get("image0_norm", data["image0"])
        image1 = data.get("image1_norm", data["image1"])
        
        # 调试：确保输入是单通道
        # print(f"DEBUG: Before mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        if image0.shape[1] != 1:
            print(f"ERROR: image0 has {image0.shape[1]} channels, expected 1")
            raise ValueError(f"Input image0 has {image0.shape[1]} channels, expected 1")
        
        mask0 = mask1 = None
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # 掩码输入图像
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        
        # print(f"DEBUG: After mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: After mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        # 保存掩码后的图像用于可视化
        data.update({
            "masked_image0": image0.clone().detach().cpu(),
            "masked_image1": image1.clone().detach().cpu()
        })
        
        # 分别提取特征 - 添加更多调试信息
        # print(f"DEBUG: Before backbone - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before backbone - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
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
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        
        # 掩码特征层
        feat_c0 = self.mask_layer(feat_c0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask0,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask1,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        
        feat_m0 = self.mask_layer(feat_m0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask0,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask1,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        
        feat_f0 = self.mask_layer(feat_f0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask0,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask1,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        
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

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        # feat_c0, feat_c1 = feat_c0.flatten(2, 3).transpose(1, 2), feat_c1.flatten(2, 3).transpose(1, 2)

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})
        
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)
        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})
    
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
    
class XoFTR_Pretrain_dino_resnet_panmatch_light(nn.Module):
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
        # MAE配置
        self.patch_size = config.get("pretrain_patch_size", 64)
        
        # ========== 模型组件 ==========
        # 初始化backbone
        # self.backbone = ResNet_8_2(config['resnet'])
        self.backbone = HighResFormerplusplus_v3(features=96, output_corr_chans=[128, 196, 256, 392])
        
        # 获取维度
        block_dims = config['resnet']["block_dims"]
        self.dim_f = block_dims[0]
        self.dim_m = block_dims[1]
        self.dim_c = block_dims[2]
        
        # 关键点编码器
        self.kenc = KeypointEncoder_wo_score(self.coarse_d_model, [32, 64, 128, self.coarse_d_model])
        
        # Mamba交互层
        self.joint_mamba = JointMamba(
            self.coarse_d_model, 4,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            profiler=None
        )
        
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
        self.fine_process = FineProcess(config)
        
        # 掩码标记
        self.mask_token_f = nn.Parameter(torch.zeros(1, block_dims[0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, block_dims[1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, block_dims[2], 1, 1))
        
        # 初始化
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)
        
        self.act = nn.GELU()
    
    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat

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

        image0 = data.get("image0_norm", data["image0"])
        image1 = data.get("image1_norm", data["image1"])
        
        # 调试：确保输入是单通道
        # print(f"DEBUG: Before mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        if image0.shape[1] != 1:
            print(f"ERROR: image0 has {image0.shape[1]} channels, expected 1")
            raise ValueError(f"Input image0 has {image0.shape[1]} channels, expected 1")
        
        mask0 = mask1 = None
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # 掩码输入图像
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        
        # print(f"DEBUG: After mask_layer - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: After mask_layer - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        # 保存掩码后的图像用于可视化
        data.update({
            "masked_image0": image0.clone().detach().cpu(),
            "masked_image1": image1.clone().detach().cpu()
        })
        
        # 分别提取特征 - 添加更多调试信息
        # print(f"DEBUG: Before backbone - image0 shape: {image0.shape}, dtype: {image0.dtype}")
        # print(f"DEBUG: Before backbone - image1 shape: {image1.shape}, dtype: {image1.dtype}")
        
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # print(f"DEBUG: After backbone - feat_c0 shape: {feat_c0.shape}")
        
        # 掩码特征层
        feat_c0 = self.mask_layer(feat_c0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask0,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 8,
                                 mask=mask1,
                                 mask_scale=1,
                                 mask_token=self.mask_token_c)
        
        feat_m0 = self.mask_layer(feat_m0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask0,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 4,
                                 mask=mask1,
                                 mask_scale=2,
                                 mask_token=self.mask_token_m)
        
        feat_f0 = self.mask_layer(feat_f0,
                                 data["mae_mask0"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask0,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                 data["mae_mask1"],
                                 mae_mask_scale=self.patch_size // 2,
                                 mask=mask1,
                                 mask_scale=4,
                                 mask_token=self.mask_token_f)
        
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

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        # feat_c0, feat_c1 = feat_c0.flatten(2, 3).transpose(1, 2), feat_c1.flatten(2, 3).transpose(1, 2)

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})
        
        feat_c0, feat_c1 = data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2)
        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})
    
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