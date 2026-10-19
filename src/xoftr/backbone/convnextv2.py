# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model

# class Block(nn.Module):
#     """ ConvNeXtV2 Block.

#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#     """

#     def __init__(self, dim, drop_path=0.):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.grn = GRN(4 * dim)
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.grn(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x


# class ConvNeXtV2(nn.Module):
#     """ ConvNeXt V2

#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """

#     def __init__(self, in_chans=3, num_classes=1000,
#                  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
#                  drop_path_rate=0., head_init_scale=1.
#                  ):
#         super().__init__()
#         self.depths = depths
#         self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                 nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
        
#         self.downsample_2x = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )

#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)

#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)

#     def forward_features(self, x):
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#         return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

#     def forward_features_8(self, x):
#         feat = {}
#         x = self.downsample_layers[0](x)
#         x = self.stages[0](x)
#         feat[4] = x
#         x = self.downsample_layers[1](x)
#         x = self.stages[1](x)
#         feat[8] = x
#         return feat  # global average pooling, (N, C, H, W) -> (N, C)

#     def forward_features_2_4_8(self, x):
#         """一次性返回2、4、8倍下采样特征"""
#         feat = {}
        
#         # 2倍特征
#         x2 = self.downsample_2x(x)
#         x2 = self.stages[0](x2)
#         feat[2] = x2
        
#         # 4倍特征
#         x4 = self.downsample_layers[0](x)
#         x4 = self.stages[0](x4)
#         feat[4] = x4
        
#         # 8倍特征
#         x8 = self.downsample_layers[1](x4)
#         x8 = self.stages[1](x8)
#         feat[8] = x8
        
#         return feat

#     def forward_features_4(self, x):
#         feat = {}
#         x = self.downsample_layers[0](x)
#         x = self.stages[0](x)
#         feat[4] = x
#         return feat  # global average pooling, (N, C, H, W) -> (N, C)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

# class LayerNorm(nn.Module):
#     """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """

#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)

#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x


# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """

#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x

# def convnextv2_atto(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
#     return model


# def convnextv2_femto(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
#     return model


# def convnext_pico(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
#     return model


# def convnextv2_nano(**kwargs):
#     model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
#     return model


# def convnextv2_tiny(**kwargs):
#     model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     return model


# def convnextv2_base(**kwargs):
#     model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     return model


# def convnextv2_large(**kwargs):
#     model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     return model


# def convnextv2_huge(**kwargs):
#     model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
#     return model

# class CovNextV2_nano(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = convnextv2_nano()
#         self.cnn.norm = None
#         self.cnn.head = None
#         self.cnn.downsample_layers[2] = None
#         self.cnn.downsample_layers[3] = None
#         self.cnn.stages[2] = None
#         self.cnn.stages[3] = None

#         state_dict = torch.hub.load_state_dict_from_url(
#             'https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt',
#             file_name='convnextv2_nano_pretrain.ckpt')
#         self.cnn.load_state_dict(state_dict, strict=False)

#         self.lin_2 = nn.Conv2d(80, 128, 1)
#         self.lin_4 = nn.Conv2d(80, 196, 1)
#         self.lin_8 = nn.Conv2d(160, 256, 1)
#         # self._init_projection_layers()

#     # def _init_projection_layers(self):
#     #     """初始化投影层权重"""
#     #     for m in [self.lin_2, self.lin_4, self.lin_8]:
#     #         if isinstance(m, nn.Conv2d):
#     #             # 对 lin_2 使用近似恒等初始化
#     #             if m.in_channels == 128 and m.out_channels == 128:
#     #                 nn.init.eye_(m.weight.view(128, 128, 1, 1))
#     #             else:
#     #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
#     #             if m.bias is not None:
#     #                 nn.init.constant_(m.bias, 0)

#     def forward(self, data):
#         B, C, H, W = data['image0'].shape
#         if C == 1:
#             # 如果是灰度图，复制为3通道
#             x0 = data['image0'].repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
#             x1 = data['image1'].repeat(1, 3, 1, 1)
#             # print(f"灰度图已转换为3通道: {x0.shape}")
#         else:
#             x0 = data['image0']
#             x1 = data['image1']
#         # x = torch.cat([data['image0'], data['image1']], 0)
#         x = torch.cat([x0, x1], 0)
#         feature_pyramid = self.cnn.forward_features_2_4_8(x)                      
#         feat_2_0, feat_2_1 = self.lin_2(feature_pyramid[2]).split(B)
#         feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)
#         feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
#         data.update({
#             'feat_c0': feat_8_0,
#             'feat_c1': feat_8_1,
#             'feat_m0': feat_4_0,
#             'feat_m1': feat_4_1,
#             'feat_f0': feat_2_0,
#             'feat_f1': feat_2_1
#         })


        
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),  # 2倍下采样
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.downsample_2x = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_8(self, x):
        feat = {}
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feat[4] = x
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        feat[8] = x
        return feat  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_2x(self, x):
        """提取2倍、4倍、8倍下采样特征"""
        feat = {}
        
        # Stem: 2倍下采样
        x = self.downsample_layers[0](x)  # 2倍下采样
        x = self.stages[0](x)             # stage1处理
        feat[2] = x                       # 存储2倍特征
        
        # 后续阶段
        x = self.downsample_layers[1](x)  # 再2倍 → 4倍
        x = self.stages[1](x)
        feat[4] = x
        
        x = self.downsample_layers[2](x)  # 再2倍 → 8倍  
        x = self.stages[2](x)
        feat[8] = x
        
        return feat

    def forward_features_4(self, x):
        feat = {}
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feat[4] = x
        return feat  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
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

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

class CovNextV2_nano(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cnn = convnextv2_nano()
        self.cnn = convnextv2_femto()
        # self.cnn = convnext_pico()
        self.cnn.norm = None
        self.cnn.head = None
        # self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        # self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        # state_dict = torch.hub.load_state_dict_from_url(
        #     'https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt',
        #     file_name='convnextv2_nano_pretrain.ckpt')
        # self.cnn.load_state_dict(state_dict, strict=False)

        # self.lin_2 = nn.Conv2d(64, 128, 1)
        # self.lin_4 = nn.Conv2d(128, 196, 1)
        # self.lin_8 = nn.Conv2d(256, 256, 1)
        self.lin_2 = nn.Conv2d(48, 128, 1)
        self.lin_4 = nn.Conv2d(96, 196, 1)
        self.lin_8 = nn.Conv2d(192, 256, 1)
        self._init_projection_layers()

    def _init_projection_layers(self):
        """初始化投影层权重"""
        import torch.nn.init as init
    
        # 方法1: 使用trunc_normal_初始化（与ConvNeXt一致）
        for name, m in [('lin_2', self.lin_2), ('lin_4', self.lin_4), ('lin_8', self.lin_8)]:
            if isinstance(m, nn.Conv2d):
                if name == 'lin_2':
                    # 对于lin_2，使用小标准差初始化，近似恒等
                    init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-0.02, b=0.02)
                else:
                    # 对于lin_4和lin_8，使用正常初始化
                    init.trunc_normal_(m.weight, std=0.02)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, data):
        B, C, H, W = data['image0'].shape
        if C == 1:
            # 如果是灰度图，复制为3通道
            x0 = data['image0'].repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
            x1 = data['image1'].repeat(1, 3, 1, 1)
            # print(f"灰度图已转换为3通道: {x0.shape}")
        else:
            x0 = data['image0']
            x1 = data['image1']
        # x = torch.cat([data['image0'], data['image1']], 0)
        x = torch.cat([x0, x1], 0)
        feature_pyramid = self.cnn.forward_features_2x(x)                      
        feat_2_0, feat_2_1 = self.lin_2(feature_pyramid[2]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        # feat_pyramid0 = self.cnn.forward_features_2x(x0)
        # feat_2_0 = self.lin_2(feat_pyramid0[2])
        # feat_4_0 = self.lin_4(feat_pyramid0[4])
        # feat_8_0 = self.lin_8(feat_pyramid0[8])
        
        # # 处理图像1
        # feat_pyramid1 = self.cnn.forward_features_2x(x1)
        # feat_2_1 = self.lin_2(feat_pyramid1[2])
        # feat_4_1 = self.lin_4(feat_pyramid1[4])
        # feat_8_1 = self.lin_8(feat_pyramid1[8])
        data.update({
            'feat_c0': feat_8_0,
            'feat_c1': feat_8_1,
            'feat_m0': feat_4_0,
            'feat_m1': feat_4_1,
            'feat_f0': feat_2_0,
            'feat_f1': feat_2_1
        })


        
