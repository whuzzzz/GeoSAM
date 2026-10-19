from __future__ import print_function

from modulefinder import Module

import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.data
import types
import math
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from einops.einops import rearrange
from .linear_attention import Attention, crop_feature, pad_feature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# class ConvXBlock(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch

#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """

#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x

class ConvXv2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    # def __init__(self, dim, drop_path=0.):
    #     super().__init__()
    #     self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
    #     self.norm = LayerNorm(dim, eps=1e-6)
    #     self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
    #     self.act = nn.GELU()
    #     # self.grn = GRN(2 * dim)
    #     self.pwconv2 = nn.Linear(2 * dim, dim)
    #     self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # def forward(self, x):
    #     input = x
    #     x = self.dwconv(x)
    #     x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    #     x = self.norm(x)
    #     x = self.pwconv1(x)
    #     x = self.act(x)
    #     # x = self.grn(x)
    #     x = self.pwconv2(x)
    #     x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    #     x = input + self.drop_path(x)
    #     return x
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# class HighResGuideNet(nn.Module):
#     # 根据 ConvNeXt 进行改写
#     def __init__(self, features, depths=[2,2,2,2]):
#         super().__init__()
#         # 特征尺度为 [1/2, 1/4, 1/8, 1/16]
#         self.firstconv = nn.Sequential(
#             nn.Conv2d(3*2**2, features, kernel_size=7, stride=1, padding=3, dilation=1),
#             LayerNorm(features, eps=1e-6, data_format="channels_first"),
#         )
#         self.stages = nn.ModuleList()
#         downsample_list = [False, True, True, True]
#         self.scales = [2, 4, 8, 16]
#         for i in range(len(depths)):
#             stage = self._make_layer(ConvXBlock, inplanes=features, planes=features, blocks=depths[i], downsample=downsample_list[i])  # base = 1/8
#             self.stages.append(stage)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, inplanes, planes, blocks, downsample=False):
#         layers = []
#         if downsample:
#             downsample_layer = nn.Sequential(
#                 LayerNorm(inplanes, eps=1e-6, data_format="channels_first"),
#                 nn.Conv2d(inplanes, planes, kernel_size=2, stride=2),
#             )
#             layers.append(downsample_layer)
#         for i in range(0, blocks):
#             layers.append(block(planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         b, _, h, w = x.shape
#         x = F.unfold(x, kernel_size=2, stride=2, padding=0).reshape(b, -1, h // 2, w // 2).contiguous()
#         x = self.firstconv(x)
#         output = []
#         for i in range(len(self.stages)):
#             x = self.stages[i](x)
#             output.append(x)
#         return output

class HighResGuideNet(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, features, depths=[2,2,2,2], drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, features, kernel_size=3, stride=2, padding=1),  # 2倍下采样
            LayerNorm(features, eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(features, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(features, features, kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvXv2Block(dim=features, drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # b, _, h, w = x.shape
        # x = F.unfold(x, kernel_size=2, stride=2, padding=0).reshape(b, -1, h // 2, w // 2).contiguous()
        output = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            output.append(x)
        return output

def forward_vit(pretrained, x):
    """
    pretrained: including model & post-process
    x: input in image shape: b, 3, h, w
    """
    b, c, h, w = x.shape

    # ViT 前向推理
    intermediates = pretrained.model.forward_flex(
        x,
        hooks=pretrained.hooks,
        aggregation=pretrained.aggregation
    )

    # 计算patch数量
    patch_size = pretrained.patch_size
    h_patches = h // patch_size
    w_patches = w // patch_size
    expected_patches = h_patches * w_patches
    
    # 创建unflatten
    unflatten = nn.Unflatten(
        2,
        torch.Size([h_patches, w_patches]),
    )
    post_intermediates = []
    for feat in intermediates:
        total_tokens_i = feat.shape[1]
        if total_tokens_i > expected_patches:
            patch_feat = feat[:, total_tokens_i - expected_patches:, :]
        else:
            patch_feat = feat
        
        # 确保token数量正确
        if patch_feat.shape[1] != expected_patches:
            if patch_feat.shape[1] > expected_patches:
                patch_feat = patch_feat[:, :expected_patches, :]
            else:
                padding = expected_patches - patch_feat.shape[1]
                patch_feat = F.pad(patch_feat, (0, 0, 0, padding))
        
        # 转置并unflatten
        post_feat = patch_feat.transpose(1, 2)
        post_feat = unflatten(post_feat)
        post_intermediates.append(post_feat.contiguous())

    return post_intermediates

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

@torch.no_grad()
def forward_flex(self, x, hooks=[2,5,8,11], aggregation="sparse"):
    b, c, h, w = x.shape
    # print(f"[DEBUG] forward_flex: Input shape: {x.shape}, type: {type(x)}")

    # 对输入进行 position embedding (适应任意方形图像)
    x = self.prepare_tokens_with_masks(x, masks=None)
    # print(f"[DEBUG] prepare_tokens_with_masks returned: {x.shape}, type: {type(x)}")
    x = x[0]
    intermediates = []
    accumulator = 0
    num_accumulated = 0
    for i, blk in enumerate(self.blocks):
        # if not isinstance(x, torch.Tensor):
        #     x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
        x = blk(x)
        if aggregation == "dense":
            accumulator = accumulator + x
            num_accumulated += 1
        if i in hooks:
            if aggregation == "dense":
                x_ = accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
            else:
                x_ = x
            intermediates.append(x_)

    # x = self.norm(x)

    return intermediates

def _resize_pos_embed(self, posemb, gs_h, gs_w):
    # 使用 dinov2 的小改进
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_old**2 == gs_h*gs_w and gs_h == gs_w:     # 和训练情况一致, 无需插值
        return posemb

    h_scale = (gs_h + 0.1) / math.sqrt(len(posemb_grid))
    w_scale = (gs_w + 0.1) / math.sqrt(len(posemb_grid))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        scale_factor=(h_scale, w_scale),
        mode="bicubic"
    )
    assert gs_h == posemb_grid.shape[-2] and gs_w == posemb_grid.shape[-1]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

def _make_vit_b16_backbone(
    model,
    psize=[16, 16],                     # patch size
    hooks=[2, 5, 8, 11],                # 在 hocks 的位置定义输出
    vit_features=768,                   # transformer blocks 的 dim
    use_readout="ignore",               # 如何处理 [cls] 与其他 token 的关系
    aggregation="sparse",               # 如何处理额外的 ViT Tokens
    start_index=1,                      # [cls] 起始位置
    enable_attention_hooks=False,
    fast_attn=False
):
    pretrained = nn.Module()

    # 将模型传给 pretrained 模型容器, 并插入输出层
    pretrained.model = model

    # # 将 model 中的 attention 更换为 flash attn
    # if fast_attn: use_memeffattn_in_model(pretrained.model)

    # 更新后的 dinov2 引入 registers 来解决 features 中的高频噪声. 设计兼容性判断语句
    if getattr(model, "register_tokens", None) is not None:
        num_register_tokens = model.register_tokens.shape[1]
    else:
        num_register_tokens = 0

    # 将 token 转为 FPN 特征
    pretrained.act_postprocess = nn.Sequential(
        Slice(start_index+num_register_tokens),                                     # ignore [cls] token
        Transpose(1, 2),                                                            # BNC -> BCN
    )   # layer1, 1/4

    # 参数设置
    # pretrained.model.start_index = start_index
    # pretrained.model.patch_size = psize
    pretrained.vit_features = [vit_features]*4
    pretrained.hooks = hooks
    pretrained.patch_size = psize[0]
    pretrained.aggregation = aggregation

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained

def _make_pretrained_dinov3_vitb16_518(
    use_readout="ignore",
    hooks=None,
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=False
):
    # 加载模型 backbone
    # repo_path = 'third_party/dinov3'
    model = torch.hub.load(
        repo_or_dir='third_party/dinov3',
        model="dinov3_vitb16",
        source="local",
        weights='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
        skip_validation=True,
    )

    # 加载 depth anything 权重 (DAM 和 DINOv2 同 encoder)
    # DINOv3_weights = torch.load('weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', map_location="cpu")
    # model.load_state_dict(DINOv3_weights, strict=True)

    return _make_vit_b16_backbone(
        model,
        psize=[model.patch_size] * 2,
        hooks=hooks,
        vit_features=model.embed_dim,  # vit transformer 的通道数
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
        aggregation=aggregation,
        fast_attn=fast_attn
    )

#######################comatch#######################
class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1)  # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1)  # [W, 1]

        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[
            3]  # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W

        div_term = torch.exp(torch.arange(0, d_model // 4, 1).float() * (-math.log(10000.0) / (d_model // 4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, :x.size(1), :x.size(2), :]) + (
                    self.rotate_half(x) * self.sin[:, :x.size(1), :x.size(2), :])

    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False,
                                   groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1,
                                           stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, x_matchability_score=None,
                source_matchability_score=None, name=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature

        if source_matchability_score == None:
            query, source = self.norm1(self.aggregate(x).permute(0, 2, 3, 1)), self.norm1(
                self.max_pool(source).permute(0, 2, 3, 1))  # [N, H, W, C]
        else:
            pooled_source_matchability_score = self.max_pool(source_matchability_score).permute(0, 2, 3,
                                                                                                1)  # [N,1 H, W]->[N,H, W, 1]

            source_matchability_score_unfold = F.unfold(source_matchability_score,
                                                        kernel_size=(self.agg_size1, self.agg_size1),
                                                        stride=self.agg_size1)
            source_matchability_score_unfold = torch.softmax(source_matchability_score_unfold, dim=1)  # [N, ww, L]

            source_unfold = F.unfold(source, kernel_size=(self.agg_size1, self.agg_size1),
                                     stride=self.agg_size1)  # [N, wwC, L]
            source_unfold = rearrange(source_unfold, 'n (c ww) l -> n c ww l',
                                      ww=self.agg_size1 ** 2)  # # [N, C, ww, L]
            source_unfold = torch.sum(source_unfold * source_matchability_score_unfold.unsqueeze(1), dim=2)  # [N, C, L]
            weighted_source = rearrange(source_unfold, 'n c (h w) -> n c h w', h=H1 // self.agg_size1,
                                        w=W1 // self.agg_size1)
            query, source = self.norm1(self.aggregate(x * x_matchability_score).permute(0, 2, 3, 1)), self.norm1(
                weighted_source.permute(0, 2, 3, 1))  # [N, H, W, C]

        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        if source_matchability_score != None:
            value = value * pooled_source_matchability_score

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)

        m = self.merge(m.reshape(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0)  # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear',
                                                align_corners=False)  # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1)).permute(0, 2, 3, 1)  # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2)  # [N, C, H0, W0]

        return x + m

class guide_upsample(nn.Module):
    """Using original features with attention mechanism to guide the low resolution features for upsampling"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge  = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, low_feats, guide_feats):
        # 首先将图像转为序列
        b, _, H, W = guide_feats.shape
        _, _, h, w = low_feats.shape
        guide_feats = guide_feats.permute(0, 2, 3, 1).reshape(b, H*W, -1)   # (B,C,H,W) -> (B,HW,C)
        low_feats = low_feats.permute(0, 2, 3, 1).reshape(b, h*w, -1)       # (B,C,h,w) -> (B,hw,C)

        # 对 q,k,v 分别进行线性映射
        query = self.q_proj(guide_feats).reshape(b, H*W, 1, self.nhead, self.dim)    # (B,HW,C) -> (B,HW,1,n,c)

        # key & value 进行邻域聚合与上采样
        key = self.k_proj(low_feats).reshape(b, h, w, -1).permute(0, 3, 1, 2)                               # 还原低分辨率图像, (B,C,h,w)
        # key = F.pad(key.float(), pad=(1,1,1,1), mode="replicate")                                           # 转为 float32 后进行padding, 保持图像边缘的数据均值与整体相近, 进而避免上采样后的特征图周围有一圈数据的均值显著小于图像中心
        key = F.pad(key, pad=(1,1,1,1), mode="replicate")
        key = F.unfold(key, kernel_size=3, padding=0, stride=1, dilation=1).reshape(b, -1, h, w)     # 邻域聚合, (B,C*9,h,w)
        key = F.interpolate(key, size=(H, W), mode="nearest").reshape(b, -1, 9, H, W)                       # 最近邻插值, (B,C,9,H,W)
        key = key.permute(0, 3, 4, 2, 1).reshape(b, H*W, 9, self.nhead, self.dim)                           # (B,H*W,N,n,c)

        value = self.v_proj(low_feats).reshape(b, h, w, -1).permute(0, 3, 1, 2)                                 # 还原低分辨率图像
        # value = F.pad(value.float(), pad=(1, 1, 1, 1), mode="replicate")
        value = F.pad(value, pad=(1, 1, 1, 1), mode="replicate")
        value = F.unfold(value, kernel_size=3, padding=0, stride=1, dilation=1).reshape(b, -1, h, w)     # 邻域聚合
        value = F.interpolate(value, size=(H, W), mode="nearest").reshape(b, -1, 9, H, W)                       # 最近邻插值
        value = value.permute(0, 3, 4, 2, 1).reshape(b, H*W, 9, self.nhead, self.dim)                           # (B,H*W,N,n,c)

        # 计算注意力
        query = query.permute(0, 1, 3, 2, 4)        # (B,HW,n,1,c)
        key = key.permute(0, 1, 3, 4, 2)            # (B,HW,n,c,N)
        value = value.permute(0, 1, 3, 2, 4)        # (B,HW,n,N,c)
        # 为了避免半精度数值溢出, 改用数值友好的方式实现 attn
        delta = self.dim ** 0.5
        score = F.softmax(torch.matmul(query / delta, key / delta) * delta, dim=-1)     # (B,HW,n,1,N)
        message = torch.matmul(score, value).squeeze(-2)                            # (B,HW,n,c)
        message = self.merge(message.reshape(b, H*W, self.nhead * self.dim))

        # TODO: 补充归一化手段, 注意 InstanceNorm1d 输入/输出为 (N,C,L)
        message = self.norm1(message).permute(0, 2, 1)

        return message.reshape(b, -1, H, W).contiguous()

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# # =====================================================================
# # 1. 核心制导模块：空间 FiLM 0417
# # =====================================================================
# class SpatialFiLMFusion(nn.Module):
#     def __init__(self, target_dim, cond_dim, mid_dim=None):
#         super().__init__()
#         self.target_dim = target_dim
#         mid_dim = mid_dim if mid_dim is not None else max(target_dim // 2, 32)
#
#         self.modulator = nn.Sequential(
#             nn.Conv2d(cond_dim, mid_dim, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(mid_dim, target_dim * 2, kernel_size=1)
#         )
#
#         # 【零初始化护城河】：保证 Epoch 0 时，网络行为等同于无融合，稳住基线！
#         nn.init.constant_(self.modulator[-1].weight, 0)
#         nn.init.constant_(self.modulator[-1].bias, 0)
#
#     def forward(self, target_feat, cond_feat):
#         # target_feat: 底层传来的高分辨率、高频结构特征 (res)
#         # cond_feat: 顶层传下来的低分辨率、强语义特征 (out)
#
#         if target_feat.shape[2:] != cond_feat.shape[2:]:
#             cond_feat = F.interpolate(cond_feat, size=target_feat.shape[2:], mode='nearest')
#
#         stats = self.modulator(cond_feat)
#         gamma_offset, beta = stats.chunk(2, dim=1)
#
#         # gamma 控制特征的放大/压制，beta 隐式地补充全局语义信息
#         gamma = 1.0 + gamma_offset
#         fused_feat = target_feat * gamma + beta
#         return fused_feat


# # =====================================================================
# # 2. 升级版 FuseBlock：替换原来的 torch.cat 0417
# # =====================================================================
# class FuseBlock(nn.Module):
#     def __init__(self, features):
#         super(FuseBlock, self).__init__()
#         self.conv_ups = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
#
#         # 【核心改动】：用 FiLM 模块替换掉原来的 conv_cat
#         self.film_fusion = SpatialFiLMFusion(target_dim=features, cond_dim=features)
#
#         self.conv_out = ConvXv2Block(features)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)
#
#     def forward(self, out, res):
#         # out: 上一层的语义输出; res: 本层的跳跃连接高频特征
#
#         # 1. 尺度对齐与平滑 (保留你原本的处理)
#         _, _, h, w = res.shape
#         out_up = F.interpolate(out, size=(h, w), mode="nearest")
#         out_up = self.conv_ups(out_up)
#
#         # 2. 【核心调制】：用上采样后的语义(out_up)去精确调制边缘细节(res)
#         output = self.film_fusion(target_feat=res, cond_feat=out_up)
#
#         # 3. 最终平滑输出
#         output = self.conv_out(output)
#         return output



class FuseBlock(nn.Module):
    def __init__(self, features):
        super(FuseBlock, self).__init__()
        self.conv_ups = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, stride=1, padding=1),
            LayerNorm(features, eps=1e-6, data_format="channels_first"),
        )
        self.conv_out = ConvXv2Block(features)
        # self.conv_out = conv1x1(features, features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, out, res):
        # 尺度对齐
        _, _, h, w = res.shape
        out = F.interpolate(out, size=(h, w), mode="nearest")
        out = self.conv_ups(out)

        output = self.conv_cat(torch.cat([res, out], dim=1))
        output = self.conv_out(output)
        return output

# class fpn_decoder(nn.Module):
#     def __init__(self, features, num_res_blocks=[1,1,1,1]):
#         super().__init__()
#         """fpn 的输入为 guided scaled features, 均经过 LayerNorm 处理, 因此使用 Norm"""
#         self.features = features
#
#         self.deepconv = ConvXv2Block(features)
#         self.fuseconv = nn.ModuleList()
#         self.iresconv = nn.ModuleList()
#         for i in range(len(num_res_blocks)):
#             self.fuseconv.append(FuseBlock(features))
#             self.iresconv.append(self._make_fuse_layers(num_res_blocks[i]))
# #
#     def _make_fuse_layers(self, num_res):
#         layers = []
#         for i in range(1, num_res):
#             layers.append(ConvXv2Block(self.features))
#         return nn.Sequential(*layers)
#
#     def forward(self, x, rem_list):
#
#         # 注意, 所有输入均为 norm 后的结果, 此外, 该 decoder 保证所有输出都是 norm 后的结果
#         x = self.deepconv(x)
#         out = []
#         for i in range(len(rem_list)):
#             x = self.fuseconv[i](x, rem_list[i])
#             x = self.iresconv[i](x)
#             out.append(x)
#         return out


class fpn_decoder(nn.Module):
    def __init__(self, features, num_res_blocks=[1, 1, 1, 1]):
        super().__init__()
        self.features = features
        self.deepconv = ConvXv2Block(features)
        self.fuseconv = nn.ModuleList()
        self.iresconv = nn.ModuleList()
        for i in range(len(num_res_blocks)):
            self.fuseconv.append(FuseBlock(features))
            self.iresconv.append(self._make_fuse_layers(num_res_blocks[i]))

        # [新增改动]：增加一个结构显著性预测头 (Saliency Head)
        self.saliency_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 2, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid 确保输出的是 0~1 的权重
        )
        # # ================== 【新增：拯救冷启动的定海神针】 ==================
        # # 强行将最后一层卷积 (索引为 2) 的偏置设为 2.0
        # # 这样 Epoch 0 时的初始 Mask 值大约是 Sigmoid(2.0) ≈ 0.88
        # # 网络初期会采取“温和放行”策略，防止预训练特征被随机噪声破坏
        # nn.init.constant_(self.saliency_head[2].bias, 1.0)
        # # ====================================================================

    def _make_fuse_layers(self, num_res):
        layers = []
        for i in range(1, num_res):
            layers.append(ConvXv2Block(self.features))
        return nn.Sequential(*layers)


    def forward(self, x, rem_list):
        x = self.deepconv(x)
        out = []
        for i in range(len(rem_list)):
            x = self.fuseconv[i](x, rem_list[i])
            x = self.iresconv[i](x)
            out.append(x)

        # [新增改动]：假设送给 JamMa 的是 1/8 分辨率 (通常对应 out[2])
        # 用这个尺度的特征生成掩码
        saliency_mask = self.saliency_head(out[1])

        # 将掩码和特征一起返回
        return out, saliency_mask

def _make_encoder(
    backbone,
    hooks=None,
    use_readout="ignore",
    enable_attention_hooks=False,
    aggregation="sparse",
    fast_attn=False,
):
    if backbone == "dinov3_vitb16":
        pretrained = _make_pretrained_dinov3_vitb16_518(
            hooks=hooks,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks,
            aggregation=aggregation,
            fast_attn=fast_attn,
        )
    
    else:
        # TODO: 增加 ViT-g, ViT-s
        print(f"Backbone '{backbone}' not implemented")
        assert False

    return pretrained


class ModalityAgnosticStructureExtractor(nn.Module):
    """
    Structure-extractor ablation module.

    extractor_type:
        "none":
            No structural preprocessing. Return the original image.

        "fixed_laplacian":
            InstanceNorm + fixed depthwise Laplacian-like filtering + ReLU.

        "sobel":
            InstanceNorm + fixed depthwise Sobel gradient magnitude.

        "learnable_laplacian":
            InstanceNorm + learnable depthwise convolution initialized with
            a Laplacian-like kernel + ReLU. This is the proposed setting.
    """

    def __init__(
            self,
            in_channels=3,
            extractor_type="learnable_laplacian",
    ):
        super().__init__()

        valid_types = {
            "none",
            "fixed_laplacian",
            "sobel",
            "learnable_laplacian",
        }
        if extractor_type not in valid_types:
            raise ValueError(
                f"Unsupported extractor_type={extractor_type}. "
                f"Expected one of {sorted(valid_types)}."
            )

        self.extractor_type = extractor_type

        # Keep the same style-normalization operation for all explicit
        # structural extractors. The "none" variant bypasses it completely.
        self.style_norm = nn.InstanceNorm2d(
            in_channels,
            affine=True,
        )

        if extractor_type == "learnable_laplacian":
            self.edge_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            )
            self._init_laplacian_like(
                self.edge_conv.weight
            )

        elif extractor_type == "fixed_laplacian":
            kernel = torch.tensor(
                [
                    [-1.0 / 8.0, -1.0 / 8.0, -1.0 / 8.0],
                    [-1.0 / 8.0, 1.0, -1.0 / 8.0],
                    [-1.0 / 8.0, -1.0 / 8.0, -1.0 / 8.0],
                ],
                dtype=torch.float32,
            )
            kernel = kernel.view(1, 1, 3, 3)
            kernel = kernel.repeat(
                in_channels,
                1,
                1,
                1,
            )
            self.register_buffer(
                "laplacian_kernel",
                kernel,
                persistent=False,
            )

        elif extractor_type == "sobel":
            sobel_x = torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            sobel_y = torch.tensor(
                [
                    [-1.0, -2.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            )

            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(
                in_channels,
                1,
                1,
                1,
            )
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(
                in_channels,
                1,
                1,
                1,
            )

            self.register_buffer(
                "sobel_x_kernel",
                sobel_x,
                persistent=False,
            )
            self.register_buffer(
                "sobel_y_kernel",
                sobel_y,
                persistent=False,
            )

    @staticmethod
    def _init_laplacian_like(weight):
        with torch.no_grad():
            weight.fill_(-1.0 / 8.0)
            weight[:, :, 1, 1] = 1.0

    def forward(self, x):
        if self.extractor_type == "none":
            return x

        x_norm = self.style_norm(x)

        if self.extractor_type == "learnable_laplacian":
            x_structure = self.edge_conv(x_norm)
            return F.relu(x_structure)

        if self.extractor_type == "fixed_laplacian":
            x_structure = F.conv2d(
                x_norm,
                self.laplacian_kernel.to(
                    dtype=x_norm.dtype
                ),
                padding=1,
                groups=x_norm.shape[1],
            )
            return F.relu(x_structure)

        if self.extractor_type == "sobel":
            grad_x = F.conv2d(
                x_norm,
                self.sobel_x_kernel.to(
                    dtype=x_norm.dtype
                ),
                padding=1,
                groups=x_norm.shape[1],
            )
            grad_y = F.conv2d(
                x_norm,
                self.sobel_y_kernel.to(
                    dtype=x_norm.dtype
                ),
                padding=1,
                groups=x_norm.shape[1],
            )

            # Numerically stable gradient magnitude.
            magnitude = torch.sqrt(
                grad_x.square()
                + grad_y.square()
                + 1e-6
            )
            return magnitude

        raise RuntimeError(
            f"Unexpected extractor type: "
            f"{self.extractor_type}"
        )


class HighResFormerplusplus_v3_block(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求.
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
       更新: 使用分离式的 encoder, 确保 corr 和 ctex 的信息不互相干扰. 同时能够有效利用不同模型的信息: DINOv2 对匹配的感知能力更强, DAM和SAM对context的感知能力更强
    """
    def __init__(
        self,
        backbone="dinov3_vitb16",
        features=256,                  
        output_chans=[128,196,256,392],   # corr feature channels
        aggregation="sparse",               # 特征聚合方式, (sparse, dense)
        structure_extractor="learnable_laplacian",
    ):
        super().__init__()

        # 实例化视觉大模型
        vit_backbone = _make_encoder(
            backbone,
            hooks=[2, 5, 8, 11],
            aggregation=aggregation,
            use_readout="ignore",
            enable_attention_hooks=False,
        )
        # vit_backbone = _make_pretrained_dinov3_vitb16_518(
        #     hooks=[2, 5, 8, 11],
        #     use_readout="ignore",
        #     enable_attention_hooks=False,
        #     aggregation="sparse",
        #     fast_attn=True,
        # )
        for p in vit_backbone.model.parameters():
            p.requires_grad = False
        vit_backbone.eval()
        vit_features = vit_backbone.vit_features        # 特征维度
        self.forward_vit = forward_vit        # 前馈 function
        self.backbone = [vit_backbone]                  # 隐藏 backbone 参数

        # 对 ViT 的特征进行压缩处理, 使用 LayerNorm, 不改变 channel 间的相对大小, 且维持半精度数值稳定, 实验验证非常有效, 且提点
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Conv2d(vit_features[i], features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=features),
        ) for i in range(4)])

        # # 注意: 如果你的红外图是单通道的，这里需要做一个简单的通道对齐，或者统一转为3通道处理
        self.structure_extractor = structure_extractor
        self.masg_extractor = ModalityAgnosticStructureExtractor(
            in_channels=3,
            extractor_type=structure_extractor,
        )

        # 构造 CNN 引导多尺度特征
        self.cnn_encoder = HighResGuideNet(features, depths=[2,2,2,2])     # 最大降采样到 1/16
        self.scales = [2, 4, 8, 16]

        # # 构造引导上采样模块, 输出为 norm 后的结果
        # self.resize = nn.ModuleList([guide_upsample(d_model=features, nhead=8) for _ in range(4)])
        # 构造引导上采样模块 (替换为 AG_RoPE_EncoderLayer)
        # 采用阶梯式 agg_size0，保证所有层的 Attention 计算复杂度一致，极限省显存！
        agg_sizes = [8, 4, 2, 1]
        self.resize = nn.ModuleList([
            AG_RoPE_EncoderLayer(
                d_model=features,
                nhead=6,
                agg_size0=agg_sizes[i],  # 对高分辨率的 CNN Guide 进行智能动态降维
                agg_size1=1,  # DINO 的层本来就只有 1/14，无需再降，设为 1
                no_flash=True,
                rope=False,  # 开启二维旋转位置编码，拉满空间感知力！
                npe = (840, 840, 840, 840),
                fp32 = False,
            ) for i in range(4)
        ])


        # 多尺度融合
        self.cnn_decoder = fpn_decoder(features=features, num_res_blocks=[1,1,1,1])

        # self.proj = nn.ModuleList([nn.Sequential(
        #     ConvXv2Block(features),
        #     nn.Conv2d(features, output_chans[i], kernel_size=3, stride=1, padding=1, bias=False),
        # ) for i in range(len(output_chans) - 1)])
        self.light_proj = nn.ModuleList([
            conv1x1(features, output_chans[i])  # 1x1卷积，没有ConvXBlock
            for i in range(len(output_chans)-1)
        ])

    def train(self, mode=True):
        super().train(mode)
        self.backbone[0].eval()

    def get_vit_input(self, x, max_downsample_ratio=16):
        _, _, H, W = x.shape
        flag = H % max_downsample_ratio == 0 and W % max_downsample_ratio == 0
        if self.training and not flag:
            vit_H = math.ceil(H / max_downsample_ratio) * max_downsample_ratio
            vit_W = math.ceil(W / max_downsample_ratio) * max_downsample_ratio
            x_vit = F.interpolate(x, size=(vit_H, vit_W), mode="bilinear", align_corners=True)
            return x_vit.detach()
        return x

    def encoder(self, x):
        _, _, H, W = x.shape
        # print(f"self.backbone:{self.backbone}")
        # print(f"self.backbone[0]:{self.backbone[0]}")
        # print(f"self.backbone[0].patch_size:{self.backbone[0].patch_size}")

        # 域不变特征, 注意应当尽可能保证训练期间使用尽可能多的 PE 信息
        with torch.no_grad():
            x_vit = self.get_vit_input(x, max_downsample_ratio=self.backbone[0].patch_size)
            self.backbone[0] = self.backbone[0].to(x_vit.device).to(x_vit.dtype)
            intermediates = self.forward_vit(self.backbone[0], x_vit)  # 1/14
        layer_1, layer_2, layer_3, layer_4 = intermediates

        # 特征压缩, 尺度不变
        layer_1 = self.linear[0](layer_1)
        layer_2 = self.linear[1](layer_2)
        layer_3 = self.linear[2](layer_3)
        layer_4 = self.linear[3](layer_4)

        # 2. [核心改动] 在送入 CNN 提取 Guide 之前，进行强制的外观剥离！
        # 此时 x_structure 已经变成了一张类似于“线稿/边缘图”的纯结构张量

        x_structure = self.masg_extractor(x)  ####20250526
        guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_structure)
        # # 引导特征
        # x_cnn = x
        # guide_1, guide_2, guide_3, guide_4 = self.cnn_encoder(x_cnn)
        # print(guide_1.shape)
        # ================= [核心重构区] =================
        # 构造特征金字塔: multi-layers -> multi-scale
        # 注意: 必须显式指定 x=guide (高频主导), source=layer (低频语义)

        layer_1 = self.resize[0](x=guide_1, source=layer_1)  # 输出 1/2 分辨率的高清特征
        layer_2 = self.resize[1](x=guide_2, source=layer_2)  # 输出 1/4 分辨率的高清特征
        layer_3 = self.resize[2](x=guide_3, source=layer_3)  # 输出 1/8 分辨率的高清特征
        layer_4 = self.resize[3](x=guide_4, source=layer_4)  # 输出 1/16 分辨率的高清特征
        # ================================================
        # # 构造特征金字塔: multi-layers -> multi-scale
        # layer_1 = self.resize[0](layer_1, guide_1)  # 1/2
        # layer_2 = self.resize[1](layer_2, guide_2)  # 1/4
        # layer_3 = self.resize[2](layer_3, guide_3)  # 1/8
        # layer_4 = self.resize[3](layer_4, guide_4)  # 1/16

        return layer_4, layer_3, layer_2, layer_1

    # def decoder(self, layer_4, layer_3, layer_2, layer_1, img):
    #
    #     # input = layer_4
    #     # layer_4, layer_3, layer_2, layer_1 = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])
    #     # 【修改点 1】：正确接收返回的 out 列表和 saliency_mask
    #     feat_in = layer_4
    #     out, saliency_mask = self.cnn_decoder(input, rem_list=[layer_4, layer_3, layer_2, layer_1])
    #
    #     # 将 out 列表解包为 4 个特征
    #     layer_4, layer_3, layer_2, layer_1 = out
    #     return layer_4, layer_3, layer_2, layer_1, saliency_mask

    def decoder(self, layer_4, layer_3, layer_2, layer_1, img):
        # 【极其重要】：强制把变量名改为 feat_in，彻底消灭 input！
        feat_in = layer_4

        # 将 feat_in 传给 cnn_decoder
        out, saliency_mask = self.cnn_decoder(feat_in, rem_list=[layer_4, layer_3, layer_2, layer_1])

        # 解包 out 列表
        layer_4, layer_3, layer_2, layer_1 = out

        # 返回所有特征和掩码
        return layer_4, layer_3, layer_2, layer_1, saliency_mask

    def forward(self, x):
        # # obtain multi-scale decoded features
        x_layer_4, x_layer_3, x_layer_2, x_layer_1 = self.encoder(x)
        y_layer_4, y_layer_3, y_layer_2, y_layer_1,saliency_mask = self.decoder(x_layer_4, x_layer_3, x_layer_2, x_layer_1, x)

        # # 计算 corr features
        # # out_fpn = []
        # # for i in range(len(self.proj)):
        # #     out_fpn.append(self.proj[i](eval('y_layer_' + str(i + 1))))

        # # return out_fpn
        out_fpn = []
        for i in range(len(self.light_proj)):
            # 跳过ConvXBlock，只使用简单的卷积
            feat = eval('y_layer_' + str(i + 1))
            out_fpn.append(self.light_proj[i](feat))  # 使用轻量级投影
            # 【修改点 3】：把 saliency_mask 塞进输出列表的最后，一起返回给 xoftr
        out_fpn.append(saliency_mask)

        return out_fpn


class HighResFormerplusplus_structural_extractor(nn.Module):
    """
       1. 自设计的网络统一采用 LayerNorm, 保证跟 ViT 的常见范式对齐
       2. Foundation Model provides all-purpose features
       3. Guided Features provide position-sensitive information
       4. Multi-scale fusion module aggregates the multi-layer representations for local-global correspondence
       5. Multi-scale reconstruction loss for details retrival
       6. output feature context alignment for context consistent
       7. 猜测: cross constrain 需要作用于构建 cost volume 的特征上才有效, 否则模型在无纹理区域的预测效果不好
       8. 新增: aggregation = ['sparse', 'dense'], 用于将多层 tokens 聚合并输出
       9. 新增: norm(x), 完全利用预训练模型的所有信息
       a. remove the occlusion mask in L_cos, L_recon, and remove the right view constrains for simple.
       b. 统一多头输出为: corr_head, feat_head, recon_head, context_head
       高分辨率的特征输出, 配合 patch embedding 来适配不同网络的输入需求.
       根据 baseline 制定 GuideNet, 使用 baseline 的 encoder 提取 guiders, 并要求预先定义所需要的分辨率. 取消 cross loss
    """
    def __init__(
            self,
            features=128,
            output_corr_chans=[128,196,256,256],  # corr feature channels
    ):
        super().__init__()
        self.feature = HighResFormerplusplus_v3_block(
            features=features,
            output_chans=output_corr_chans,
            structure_extractor="none",
        )

    def forward(self, im0):
        if im0.shape[1] == 1:
            im0 = im0.repeat(1, 3, 1, 1)
        features = self.feature(im0)
        x1 = features[0]  # 1/2
        x2 = features[1]  # 1/4
        x3_out = features[2]  # 1/8

        # 【提取掩码】：把它从 features 列表的最后一位拿出来
        saliency_mask = features[3]

        return x3_out, x2, x1,saliency_mask
