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
        default_factory=lambda: [4, 11, 17, 23]  # 改为vitl16的推荐层
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
        # features: int = 192,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        # out_channels: List[int] = [192, 384, 768, 768],
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

# class ChannelAtt(nn.Module):
#     def __init__(self, in_channels, ratio=4):
#         super(ChannelAtt, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpaCNN(nn.Module):
#     def __init__(self, in_channels, SK_size = 3, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.bn2 = nn.BatchNorm2d(in_channels)

#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         return F.relu(Y)

# class SpatialCNNAtt(nn.Module):
#     def __init__(self,in_channels = 64, SK_size = 3, kernel_size=3):
#         super(SpatialCNNAtt, self).__init__()
#         self.scnn = SpaCNN(in_channels=in_channels, SK_size=SK_size)
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.scnn(x)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class SAFFM(nn.Module):
#     def __init__(self, channel=64, SK_size = 3):
#         super(SAFFM, self).__init__()
#         self.spaCnnAtt = SpatialCNNAtt(in_channels=channel, SK_size=SK_size)
#         self.chaAtt = ChannelAtt(in_channels=channel)
#         self.conv1 = nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, stride=1)
#         self.bn1 = nn.BatchNorm2d(channel)
#     def forward(self, x ,y):
#         f = self.chaAtt(x) * self.spaCnnAtt(y)
#         z = torch.cat([f*x, f*y], dim=1)
#         return F.relu(self.bn1(self.conv1(z)))

class CrossDomainAlignmentBlock(nn.Module):
    def __init__(self, dim):
        super(CrossDomainAlignmentBlock, self).__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim)
        self.w_q = nn.Parameter(torch.Tensor(dim, dim))
        self.w_k = nn.Parameter(torch.Tensor(dim, dim))
        self.w_v = nn.Parameter(torch.Tensor(dim, dim))
        self.alpha = nn.Parameter(torch.Tensor(dim))
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim)
        )

        # 初始化权重
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.ones_(self.alpha)

    def forward(self, geometric_features, semantic_priors):
        # geometric_features: [N, C]，局部几何特征
        # semantic_priors: [N, C]，语义先验特征
        [b,c,h,w]=geometric_features.shape
        geometric_features=geometric_features.flatten(2).permute(0,2,1)
        semantic_priors = semantic_priors.flatten(2).permute(0,2,1)

        # 层归一化
        norm_geo = self.layer_norm(geometric_features)
        norm_sem = self.layer_norm(semantic_priors)
        semantic_priors = self.MLP(semantic_priors)
        geometric_features = self.MLP(geometric_features)

        # 线性变换
        Q = torch.matmul(norm_sem, self.w_q)  #1*3600*256
        K = torch.matmul(norm_geo, self.w_k)  #1*3600*256

        # 特征相关性计算
        A = torch.matmul(Q.transpose(1, 2), K) / self.alpha.unsqueeze(0)
        A = F.softmax(A, dim=-1)

        # xx=torch.matmul(self.layer_norm(geometric_features), self.w_v).permute(0,2,1)

        # 特征对齐
        aligned_semantic = torch.matmul(A, torch.matmul(self.layer_norm(geometric_features), self.w_v).permute(0,2,1))

        return aligned_semantic.view(b, c, h, w)

class CrossAttentionFusion_zhu(nn.Module):  #交叉注意力融合（Cross-Attention Fusion）方案
    def __init__(self, in_channels=256):
        super().__init__()
        # Dinov2特征预处理（通道对齐+上采样）
        # self.dinov2_preprocess = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=1)  # 通道压缩
        # )
        self.CrossDomainAlignmentBlock=CrossDomainAlignmentBlock(in_channels)


        # 注意力参数
        self.query_proj = nn.Conv2d(in_channels, in_channels, 1)  # FPN作为query
        self.key_proj = nn.Conv2d(in_channels, in_channels, 1)  # Dinov2作为key
        self.value_proj = nn.Conv2d(in_channels, in_channels, 1)  # Dinov2作为value

        # 初始化
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0)

    def forward(self, fpn_feat, dinov2_feat):
        """
        Args:
            fpn_feat:   B×256×45×80
            dinov2_feat: B×384×20×40
        Returns:
            fused_feat: B×256×45×80
        """
        # 预处理Dinov2特征
        # dinov2_feat = self.dinov2_preprocess(dinov2_feat)  # B×256×45×80
        dinov2_feat=self.CrossDomainAlignmentBlock(fpn_feat,dinov2_feat)
        dinov2=dinov2_feat
        # fpn_feat=self.MLP(fpn_feat)
        # dinov2 = self.MLP(dinov2_feat)


        # 生成Q/K/V
        Q = self.query_proj(fpn_feat).flatten(2)  # B×256×(45*80)
        K = self.key_proj(dinov2).flatten(2)  # B×256×(45*80)
        V = self.value_proj(dinov2).flatten(2)  # B×256×(45*80)

        # 注意力计算
        attn = torch.softmax(Q.transpose(1, 2) @ K / torch.sqrt(torch.tensor(K.size(1))), dim=-1)  # B×(45*80)×(45*80)
        fused = (attn @ V.transpose(1, 2)).transpose(1, 2)  # B×256×(45*80)
        # fused=self.FFN(fused)

        # 残差连接
        return fpn_feat + fused.view_as(fpn_feat)

class DINOv3B_resnet(nn.Module):
    """针对dinov3_vitb16优化的版本 - 可以直接使用"""
    
    def __init__(self, config=None, layer_strategy="balanced"):
        super().__init__()     
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        self.in_planes = initial_dim
        self.CrossAttentionFusion_zhu = CrossAttentionFusion_zhu()

        self.conv1_resnet = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_resnet = nn.BatchNorm2d(initial_dim)
        self.relu_resnet = nn.ReLU(inplace=True)

        self.layer1_resnet = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_resnet = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_resnet = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        self.layer3_outconv_resnet = conv1x1(block_dims[2], block_dims[2])
        
        # # dinov3_vitb16的特征维度是768
        # self.dino_dim = 768
        cfg = DescriptorConfig()  # 使用默认配置
        self.f = DescriptorFactory.create(cfg)  # 创建模型
        # self.head = DPTHead(
        #     dim_in=768,
        #     features=256,
        #     down_ratio=8,
        #     feature_only=True,
        # )
        self.head = DPTHead(
            dim_in=1024,
            features=256,
            down_ratio=8,
            feature_only=True,
        )
        
        # self.dim_align = nn.Sequential(
        #     nn.Conv2d(256, block_dims[2], kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(block_dims[2]),
        #     nn.ReLU(inplace=True)
        # )
        # self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        # self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        # self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        # 标准化（用于预处理输入图像）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # self.saffm_1_8 = SAFFM(channel=block_dims[2], SK_size=3)
        # self.saffm_1_4 = SAFFM(channel=block_dims[1], SK_size=3)
        # self.saffm_1_2 = SAFFM(channel=block_dims[0], SK_size=3)
        
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
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        spatial_features = self.f(x)
        x3_dino = self.head(spatial_features)
        # x3_dino = F.interpolate(x3_attention, scale_factor=2., mode='bilinear', align_corners=True)
        # x0_dino = self.dim_align(x0_dino)
        # x1_dino = self.layer1(x0_dino)  # 1/2
        # x2_dino = self.layer2(x1_dino)  # 1/4
        # x3_dino = self.layer3(x2_dino)  # 1/8

        # x3_out_dino = self.layer3_outconv(x3_dino)       

        x0_resnet = self.relu_resnet(self.bn1_resnet(self.conv1_resnet(x)))
        x1_resnet = self.layer1_resnet(x0_resnet)  # 1/2
        x2_resnet = self.layer2_resnet(x1_resnet)  # 1/4
        x3_resnet = self.layer3_resnet(x2_resnet)  # 1/8

        x3_out_resnet = self.layer3_outconv_resnet(x3_resnet)
        
        x3_attention = self.CrossAttentionFusion_zhu(x3_out_resnet, x3_dino)

        x2_attention = F.interpolate(x3_attention, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2_resnet)
        x2_out = self.layer2_outconv2(x2_out+x2_attention)

        x1_attention = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1_resnet)
        x1_out = self.layer1_outconv2(x1_out+x1_attention)
        return x3_attention, x2_out, x1_out
