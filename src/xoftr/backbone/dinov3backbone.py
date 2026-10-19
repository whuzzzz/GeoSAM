# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms

# class EfficientDINOv3Backbone(nn.Module):
#     """
#     高效版本，共享基础计算
#     """
    
#     def __init__(self, model_type='dinov3_vitb16', freeze=True):
#         super().__init__()
        
#         # self.model = torch.hub.load('facebookresearch/dinov3', model_type)
#         self.model = torch.hub.load(
#             repo_or_dir='third_party/dinov3',
#             model=model_type,
#             source="local",
#             trust_repo=True
#         )
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#         self.feat_dim = self.model.embed_dim
#         self.patch_size = self.model.patch_size
        
#         # 共享的基础特征增强
#         self.base_enhancer = nn.Sequential(
#             nn.Conv2d(self.feat_dim, 512, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
        
#         # 三个分支（只做上采样和最后降维）
#         self.branch_1_8 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(512, 256, 1)
#         )
        
#         self.branch_1_4 = nn.Sequential(
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
#             nn.Conv2d(512, 196, 1)
#         )
        
#         self.branch_1_2 = nn.Sequential(
#             nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
#             nn.Conv2d(512, 128, 1)
#         )
        
#         # 轻量级后处理
#         self.post_256 = nn.Conv2d(256, 256, 3, padding=1)
#         self.post_196 = nn.Conv2d(196, 196, 3, padding=1)
#         self.post_128 = nn.Conv2d(128, 128, 3, padding=1)
        
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
    
#     def forward(self, x):
#         # 输入处理
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         x = self.normalize(x)
        
#         # 提取DINOv3特征
#         with torch.no_grad():
#             base_feat = self.model.get_intermediate_layers(
#                 x, n=11, reshape=True, norm=True
#             )[0]
        
#         # 基础特征增强
#         enhanced = self.base_enhancer(base_feat)
        
#         # 三个分支
#         feat_1_8 = self.post_256(self.branch_1_8(enhanced))
#         feat_1_4 = self.post_196(self.branch_1_4(enhanced))
#         feat_1_2 = self.post_128(self.branch_1_2(enhanced))
        
#         return feat_1_8, feat_1_4, feat_1_2
    
# # class EfficientDINOv3Backbone(nn.Module):
# #     """
# #     直接替换ResNet_8_2的DINO版本
# #     保持相同的输入输出接口
# #     """
    
# #     def __init__(self, config, model_type='dinov3_vitb16', freeze=True):
# #         super().__init__()
        
# #         # 保持与ResNet相同的config参数
# #         self.config = config
        
# #         # 加载DINOv3模型并冻结
# #         self.dino_model = torch.hub.load(
# #             repo_or_dir='third_party/dinov3',
# #             model=model_type,
# #             source="local",
# #             trust_repo=True
# #         )
        
# #         # 冻结DINO所有参数
# #         for param in self.dino_model.parameters():
# #             param.requires_grad = False
# #         self.dino_model.eval()
        
# #         # 获取DINO特征维度
# #         self.dino_feat_dim = self.dino_model.embed_dim
        
# #         # 可训练的适配层（替换ResNet的卷积层）
# #         # 注意：DINO输出是1/16分辨率，我们需要调整到1/8、1/4、1/2
        
# #         # 1. 适配层：将DINO特征映射到目标维度
# #         target_dim = config['block_dims'][2]  # 使用ResNet配置中的最后一维（256）
# #         self.adapter = nn.Sequential(
# #             nn.Conv2d(self.dino_feat_dim, target_dim * 2, 1, bias=False),
# #             nn.BatchNorm2d(target_dim * 2),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(target_dim * 2, target_dim, 3, padding=1, bias=False),
# #             nn.BatchNorm2d(target_dim),
# #             nn.ReLU(inplace=True)
# #         )
        
# #         # 2. 上采样层：生成多分辨率特征
# #         # 1/8特征（DINO输出约1/16，需要上采样2x）
# #         self.upsample_to_1_8 = nn.Sequential(
# #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
# #             nn.Conv2d(target_dim, target_dim, 3, padding=1, bias=False),
# #             nn.BatchNorm2d(target_dim),
# #             nn.ReLU(inplace=True)
# #         )
        
# #         # 1/4特征（再上采样2x）
# #         self.upsample_to_1_4 = nn.Sequential(
# #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
# #             nn.Conv2d(target_dim, config['block_dims'][1], 3, padding=1, bias=False),  # 196
# #             nn.BatchNorm2d(config['block_dims'][1]),
# #             nn.ReLU(inplace=True)
# #         )
        
# #         # 1/2特征（再上采样2x）
# #         self.upsample_to_1_2 = nn.Sequential(
# #             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
# #             nn.Conv2d(config['block_dims'][1], config['block_dims'][0], 3, padding=1, bias=False),  # 128
# #             nn.BatchNorm2d(config['block_dims'][0]),
# #             nn.ReLU(inplace=True)
# #         )
        
# #         # 输出卷积（与ResNet的layer3_outconv对应）
# #         self.layer3_outconv = nn.Conv2d(target_dim, target_dim, 1, bias=False)
        
# #         # 初始化可训练参数（与ResNet相同的初始化方式）
# #         self._init_weights()
        
# #         # 图像标准化参数
# #         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
# #         self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
# #     def _init_weights(self):
# #         """使用与ResNet相同的权重初始化"""
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)
    
# #     def preprocess(self, x):
# #         """预处理输入图像"""
# #         # 灰度转RGB（如果输入是单通道）
# #         if x.shape[1] == 1:
# #             x = x.repeat(1, 3, 1, 1)
        
# #         # ImageNet标准化
# #         x = (x - self.mean) / self.std
# #         return x
    
# #     def extract_dino_features(self, x):
# #         """提取DINO特征（冻结部分）"""
# #         with torch.no_grad():
# #             # 获取最后一层特征
# #             features = self.dino_model.get_intermediate_layers(
# #                 x,
# #                 n=11,
# #                 reshape=True,  # 输出形状为[B, C, H, W]
# #                 norm=True      # L2归一化
# #             )[0]
# #         return features
    
# #     def forward(self, x):
# #         """
# #         前向传播
# #         输入: x [B, 1, H, W] 灰度图像
# #         输出: (x3_out, x2, x1) 对应ResNet的输出
# #             - x3_out: 1/8分辨率特征
# #             - x2: 1/4分辨率特征  
# #             - x1: 1/2分辨率特征
# #         """
# #         # 1. 预处理
# #         x_rgb = self.preprocess(x)  # [B, 3, H, W]
        
# #         # 2. 提取DINO特征（冻结，无梯度）
# #         # DINO输出分辨率：H/16 × W/16
# #         dino_feat = self.extract_dino_features(x_rgb)  # [B, 768, H/16, W/16]
        
# #         # 3. 通过适配层（可训练，有梯度）
# #         adapted = self.adapter(dino_feat)  # [B, 256, H/16, W/16]
        
# #         # 4. 生成1/8特征（上采样2x）
# #         feat_1_16 = adapted
# #         feat_1_8 = self.upsample_to_1_8(feat_1_16)  # [B, 256, H/8, W/8]
# #         x3_out = self.layer3_outconv(feat_1_8)      # 对应ResNet的x3_out
        
# #         # 5. 生成1/4特征（再上采样2x）
# #         feat_1_4 = self.upsample_to_1_4(feat_1_8)   # [B, 128, H/4, W/4]
# #         x2 = feat_1_4                               # 对应ResNet的x2
        
# #         # 6. 生成1/2特征（再上采样2x）
# #         feat_1_2 = self.upsample_to_1_2(feat_1_4)   # [B, 64, H/2, W/2]
# #         x1 = feat_1_2                               # 对应ResNet的x1
        
# #         return x3_out, x2, x1

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, List, Tuple, Union
import torch
from einops import rearrange
from torch import nn
import torch.nn as nn
from functools import partial
import torchvision.models as models
from torch.nn import functional as F


Normalizer = Callable[[torch.Tensor], torch.Tensor]
DescriptorName = str
    
# ImageNet标准化
def imagenet(img: torch.Tensor) -> torch.Tensor:
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    return (img - imagenet_mean[None, :, None, None]) / imagenet_std[
        None, :, None, None
    ]
    
# # 默认设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def wrap_with_normalize(
    forward: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
):
    def wrapped_forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        with (
            torch.autocast(device.type, torch.bfloat16, enabled=enable_amp),
            torch.set_grad_enabled(not frozen),
        ):
            if self.training and frozen:
                self.eval()
            B, C, H, W = img.shape
            assert C == 3, f"Image must have 3 channels, but got shape {img.shape=}"
            img_n = normalizer(img)
            H = H // patch_size
            W = W // patch_size
            raw_outs = forward(img_n)
            maybe_feat_normalizer = (
                F.normalize if normalize_feats else lambda x, dim=-1: x
            )
            return [
                maybe_feat_normalizer(
                    rearrange(x, "B (H W) D -> B H W D", H=H, W=W), dim=-1
                )
                for x in raw_outs
            ]

    return wrapped_forward


def wrap_model(
    model: nn.Module,
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
    func: Any,
):
    if enable_amp and frozen:  # if training we want params in fp32
        model = model.to(torch.bfloat16)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.frozen = frozen
    type(model).forward = wrap_with_normalize(
        func,
        normalizer=normalizer,
        patch_size=patch_size,
        enable_amp=enable_amp,
        frozen=frozen,
        normalize_feats=normalize_feats,
    )
    return model


def _get_layers(layers, model):
    return [b if b > 0 else len(model.blocks) + b for b in layers]

@dataclass(frozen=True)
class DescriptorConfig:
    # name: DescriptorName = "dinov3_vitb16"  # 默认改为vitb16
    name: DescriptorName = "dinov3_vitl16"
    enable_amp: bool = True
    frozen: bool = True
    normalize_feats: bool = False
    # layer_idx: list[int] = field(
    #     default_factory=lambda: [2, 5, 8, 11]  # 改为vitb16的推荐层
    # )
    layer_idx: list[int] = field(
        default_factory=lambda: [4, 11, 17, 23]  # 改为vitb16的推荐层
    )
    weights_path: str | None = None

class DescriptorFactory:
    """描述符工厂类，更清晰的设计"""
    
    @staticmethod
    def create(config: DescriptorConfig) -> nn.Module:
        partial_wrap = partial(
            wrap_model,
            enable_amp=config.enable_amp,
            frozen=config.frozen,
            normalize_feats=config.normalize_feats,
        )
        match config.name:
            case "dinov3_vitb16":
                normalizer = imagenet
                # 修复：使用正确的模型名称和版本
                dinov3_vitb16: nn.Module = torch.hub.load(
                    repo_or_dir='third_party/dinov3',
                    model="dinov3_vitb16",
                    source="local",
                    weights='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
                    skip_validation=True,
                ).to(device)
                layers = _get_layers(config.layer_idx, dinov3_vitb16)
                # ✅ 修复：使用正确的变量名 dinov3_vitb16
                return partial_wrap(
                    dinov3_vitb16,  # 这里修复了！
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(dinov3_vitb16.get_intermediate_layers, n=layers),
                )
            # case "dinov3_convnext_base":
            #     normalizer = imagenet
            #     # 修复：使用正确的模型名称和版本
            #     dinov3_convnext_base: nn.Module = torch.hub.load(
            #         repo_or_dir='third_party/dinov3',
            #         model="dinov3_convnext_base",
            #         source="local",
            #         weights='weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth', 
            #         skip_validation=True,
            #     ).to(device)
            #     layers = _get_layers(config.layer_idx, dinov3_convnext_base)
            #     # ✅ 修复：使用正确的变量名 dinov3_vitb16
            #     return partial_wrap(
            #         dinov3_convnext_base,  # 这里修复了！
            #         normalizer=normalizer,
            #         patch_size=16,
            #         func=partial(dinov3_convnext_base.get_intermediate_layers, n=layers),
            #     )
            case "dinov3_vitl16":
                normalizer = imagenet
                # 修复：使用正确的模型名称和版本
                dinov3_vitl16: nn.Module = torch.hub.load(
                    repo_or_dir='third_party/dinov3',
                    model="dinov3_vitl16",
                    source="local",
                    weights='weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth', 
                    skip_validation=True,
                ).to(device)
                layers = _get_layers(config.layer_idx, dinov3_vitl16)
                # ✅ 修复：使用正确的变量名 dinov3_vitb16
                return partial_wrap(
                    dinov3_vitl16,  # 这里修复了！
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(dinov3_vitl16.get_intermediate_layers, n=layers),
                )
            case _:
                raise ValueError(f"Unknown model name: {config.name}")
            
class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        feature_only: bool = True,
        down_ratio: int = 1,
        align_corners: bool = True,
    ) -> None:
        super(DPTHead, self).__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.align_corners = align_corners
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1,
                head_features_1 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    conv2_in_channels,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        tokens: torch.Tensor | List[torch.Tensor],
        *,
        img_A: torch.Tensor | None = None,
        img_B: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(tokens, torch.Tensor):
            B, H, W, D = tokens.shape
        else:
            B, H, W, D = tokens[0].shape
        return self._forward_impl(tokens, H, W)

    def _forward_impl(
        self,
        aggregated_tokens_list_or_tokens: List[torch.Tensor] | torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            assert not isinstance(aggregated_tokens_list_or_tokens, torch.Tensor), (
                "aggregated_tokens_list_or_tokens should be a list of tensors"
            )
            aggregated_tokens_list = aggregated_tokens_list_or_tokens
            if len(aggregated_tokens_list) != len(self.out_channels):
                assert len(self.out_channels) % len(aggregated_tokens_list) == 0, (
                    "out_channels must be a multiple of intermediate_layer_idx"
                )
                factor = len(self.out_channels) // len(aggregated_tokens_list)
                aggregated_tokens_list = [
                    x
                    for xs in [
                        [aggregated_tokens_list[i]] * factor
                        for i in range(len(aggregated_tokens_list))
                    ]
                    for x in xs
                ]

            B = aggregated_tokens_list[0].shape[0]

            H, W = patch_h * self.patch_size, patch_w * self.patch_size

            out = []

            for dpt_idx in range(len(self.out_channels)):
                x = aggregated_tokens_list[dpt_idx]

                x = x.reshape(B, -1, x.shape[-1])

                x = self.norm(x)

                x = x.permute(0, 2, 1).reshape(
                    (x.shape[0], x.shape[-1], patch_h, patch_w)
                )

                x = self.projects[dpt_idx](x)
                # if self.pos_embed:
                #     x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)

                out.append(x)

            # Fuse features from multiple layers.
            out = self.scratch_forward(out)
            # Interpolate fused output to match target image resolution.
            out = custom_interpolate(
                out,
                (
                    int(patch_h * self.patch_size / self.down_ratio),
                    int(patch_w * self.patch_size / self.down_ratio),
                ),
                mode="bilinear",
                align_corners=self.align_corners,
            )

            # if self.pos_embed:
            #     out = self._apply_pos_embed(out, W, H)

            if self.feature_only:
                return out.view(B, *out.shape[1:])
        # float it for precision
        out = out.float()
        out = self.scratch.output_conv2(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(
    features: int,
    size: int = None,
    has_residual: bool = True,
    groups: int = 1,
    align_corners: bool = True,
) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=align_corners,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=self.groups,
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(
                features, activation, bn, groups=self.groups
            )

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(
            features, activation, bn, groups=self.groups
        )

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk, size=size, mode=mode, align_corners=align_corners
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(
            x, size=size, mode=mode, align_corners=align_corners
        )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class DINOv3B(nn.Module):
    """针对dinov3_vitb16优化的版本 - 可以直接使用"""
    
    def __init__(self, config=None, layer_strategy="balanced"):
        super().__init__()     
        # 根据策略选择层
        layer_strategies = {
            "balanced": [2, 5, 8, 11],      # 平衡型
            "detail": [2, 5, 9],         # 细节型
            "semantic": [5, 8, 11],      # 语义型
            "uniform": [0, 4, 8],        # 均匀型
            "deep": [6, 9, 11],          # 深层型
        }
        
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        self.in_planes = initial_dim
        
        # self.layer_idx = layer_strategies.get(layer_strategy, [2, 5, 8, 11])
        # print(f"使用策略 '{layer_strategy}': 提取层 {layer_idx}")
        
        # self.dino_model = torch.hub.load(
        #     repo_or_dir='third_party/dinov3',
        #     model="dinov3_vitb16",
        #     source="local",
        #     weights='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
        #     skip_validation=True
        # )
        # self.dino_model = torch.hub.load(
        #     repo_or_dir='third_party/dinov3',
        #     model="dinov3_vitb16",
        #     source="local",
        #     trust_repo=True
        # )
        
        # # 冻结DINO参数
        # for param in self.dino_model.parameters():
        #     param.requires_grad = False
        # # self.dino_model.eval()
        
        # # dinov3_vitb16的特征维度是768
        # self.dino_dim = 768
        cfg = DescriptorConfig()  # 使用默认配置
        self.f = DescriptorFactory.create(cfg)  # 创建模型
        # self.head = DPTHead(
        #     dim_in=768,
        #     features=256,
        #     down_ratio=2,
        #     feature_only=True,
        # )
        self.head = DPTHead(
            dim_in=1024,
            features=256,
            down_ratio=2,
            feature_only=True,
        )
        self.dim_align = nn.Sequential(
            nn.Conv2d(256, block_dims[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])

        # 标准化（用于预处理输入图像）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self._init_weights()

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        输入: x [B, 1, H, W] 灰度图像
        输出: (feat_1_8, feat_1_4, feat_1_2) 三个分辨率的特征
        """
        # 1. 预处理：灰度转RGB + 标准化
        # B, C, H, W = x.shape
        # if C == 1:
        #     x = x.repeat(1, 3, 1, 1)
        # x = (x - self.mean) / self.std
        
        # # 2. 提取DINO特征
        # # 注意：Descriptor返回的是 [B, H, W, C] 格式的列表
        # features = self.dino_model.get_intermediate_layers(
        #     x,
        #     n=self.layer_idx,
        #     reshape=False,  # 保持 [B, N, D] 格式
        #     norm=True       # L2归一化
        # )
        # patch_h = H // 16
        # patch_w = W // 16
        # spatial_features = []
        # for feat in features:
        #     # feat: [B, N, D], N = patch_h * patch_w
        #     feat = feat.reshape(B, patch_h, patch_w, -1)
        #     spatial_features.append(feat)
        # x0 = self.head(spatial_features)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        spatial_features = self.f(x)
        x0 = self.head(spatial_features)
        x0 = self.dim_align(x0)
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        x3_out = self.layer3_outconv(x3)

        return x3_out, x2, x1

# class DINOv3B(nn.Module):
#     def __init__(self, config=None, layer_strategy="balanced"):
#         super().__init__()
        
#         self.output_dims = [256, 196, 128]  # [256, 196, 128]
        
#         # 层选择策略
#         layer_strategies = {
#             "balanced": [3, 6, 11],      # 平衡型
#             "detail": [2, 5, 9],         # 细节型
#             "semantic": [5, 8, 11],      # 语义型
#         }
        
#         self.layer_idx = layer_strategies.get(layer_strategy, [3, 6, 11])
#         # print(f"使用策略 '{layer_strategy}': 提取层 {self.layer_idx}")
        
#         self.dino_model = torch.hub.load(
#             repo_or_dir='third_party/dinov3',
#             model="dinov3_vitb16",
#             source="local",
#             weights='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
#             skip_validation=True
#         )
        
#         # 冻结DINO参数
#         for param in self.dino_model.parameters():
#             param.requires_grad = False
#         self.dino_model.eval()
        
#         # DINO特征维度
#         self.dino_dim = 768
        
#         # 1. Projects: 1x1卷积直接投影到目标维度
#         self.projects = nn.ModuleList([
#             nn.Conv2d(
#                 in_channels=self.dino_dim,
#                 out_channels=out_dim,  # 直接到目标维度
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             ) for out_dim in self.output_dims  # [256, 196, 128]
#         ])
        
#         # ✅ 删除复杂的upsample_blocks，后面直接使用F.interpolate
        
#         # 标准化
#         self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
#         # 初始化
#         self._init_weights()
    
#     def _init_weights(self):
#         """初始化权重"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
#     def preprocess(self, x):
#         """预处理：灰度转RGB + ImageNet标准化"""
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         return (x - self.imagenet_mean) / self.imagenet_std
    
#     def extract_dino_features(self, x):
#         """提取DINO特征"""
#         features = self.dino_model.get_intermediate_layers(
#             x,
#             n=self.layer_idx,
#             reshape=False,  # 保持 [B, N, D] 格式
#             norm=True       # L2归一化
#         )
#         return features
    
#     def forward(self, x):
#         """
#         前向传播
#         输入: x [B, 1, H, W] 灰度图像
#         输出: (feat_1_8, feat_1_4, feat_1_2) 维度分别为[256, 196, 128]
#         """
#         # 1. 预处理
#         x_rgb = self.preprocess(x)
        
#         # 2. 提取DINO特征
#         dino_features = self.extract_dino_features(x_rgb)
        
#         # 3. 计算patch尺寸
#         B, _, H, W = x.shape
#         patch_h = H // 16  # DINO patch size = 16
#         patch_w = W // 16
        
#         # 4. 将DINO特征转换为图像格式 [B, D, H, W]
#         img_features = []
#         for feat in dino_features:
#             # feat形状: [B, N, D] 其中 N = patch_h * patch_w
#             B, N, D = feat.shape
#             feat_img = feat.permute(0, 2, 1).reshape((B, D, patch_h, patch_w))
#             img_features.append(feat_img)
        
#         # 5. Projects: 1x1卷积直接投影到目标维度
#         projected = []
#         for i, (project, feat) in enumerate(zip(self.projects, img_features)):
#             projected.append(project(feat))  # 保持各自维度
        
#         # 6. ✅ 最简单的上采样：直接使用F.interpolate
#         # 计算目标尺寸
#         target_1_8 = (patch_h * 2, patch_w * 2)  # 1/8分辨率
#         target_1_4 = (patch_h * 4, patch_w * 4)  # 1/4分辨率
#         target_1_2 = (patch_h * 8, patch_w * 8)  # 1/2分辨率
        
#         # feat_1_8: [256, 40, 40] → 上采样到80×80
#         feat_1_8 = F.interpolate(
#             projected[0], 
#             size=target_1_8, 
#             mode='bilinear', 
#             align_corners=True
#         )
        
#         # feat_1_4: [196, 40, 40] → 上采样到160×160
#         feat_1_4 = F.interpolate(
#             projected[1],
#             size=target_1_4,
#             mode='bilinear',
#             align_corners=True
#         )
        
#         # feat_1_2: [128, 40, 40] → 上采样到320×320
#         feat_1_2 = F.interpolate(
#             projected[2],
#             size=target_1_2,
#             mode='bilinear',
#             align_corners=True
#         )
        
#         return feat_1_8, feat_1_4, feat_1_2