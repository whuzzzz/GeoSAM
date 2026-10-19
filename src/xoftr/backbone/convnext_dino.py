import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # ====== 新增的2倍下采样分支（将重新训练） ======
        self.extra_stem_2x = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),  # 2倍下采样
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),  # 额外的卷积增强表征
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # 2倍特征的处理块（比原始简单）
        self.extra_stage_2x = nn.Sequential(
            *[Block(dim=dims[0], drop_path=0.0, layer_scale_init_value=layer_scale_init_value) 
              for _ in range(2)]  # 只用2个block，减少计算量
        )

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # 特别初始化新增部分
        self._init_extra_layers()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_extra_layers(self):
        """初始化新增部分的权重"""
        def init_extra_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.01)  # 更小的std，更接近恒等变换
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.extra_stem_2x.apply(init_extra_weights)
        self.extra_stage_2x.apply(init_extra_weights)
    
    def freeze_pretrained_part(self):
        """冻结预训练部分（原始ConvNeXt）"""
        # 冻结所有原始层的参数
        for param in self.downsample_layers.parameters():
            param.requires_grad = False
        for param in self.stages.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False
        
        # 新增部分保持可训练
        for param in self.extra_stem_2x.parameters():
            param.requires_grad = True
        for param in self.extra_stage_2x.parameters():
            param.requires_grad = True

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    
    def forward_features_4(self, x):
        feat = {}
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feat[4] = x
        return feat  # global average pooling, (N, C, H, W) -> (N, C)
    
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

    def forward_features_all(self, x):
        """提取所有尺度的特征：2x, 4x, 8x"""
        feat = {}
        
        # ====== 2倍下采样特征（新增，可训练） ======
        x_2 = self.extra_stem_2x(x)
        x_2 = self.extra_stage_2x(x_2)
        feat[2] = x_2
        
        # ====== 4倍下采样特征（预训练，固定） ======
        x_4 = self.downsample_layers[0](x)  # 4倍下采样
        x_4 = self.stages[0](x_4)
        feat[4] = x_4
        
        # ====== 8倍下采样特征（预训练，固定） ======
        x_8 = self.downsample_layers[1](x_4)  # 再2倍 → 8倍
        x_8 = self.stages[1](x_8)
        feat[8] = x_8
        
        return feat

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

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
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
    #     url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])
        model_dinov3 = torch.hub.load(
            repo_or_dir='third_party/dinov3',
            model='dinov3_convnext_base',
            source="local",
            weights='weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'
        )
        # 获取 DINOv3 模型的状态字典
        state_dict = model_dinov3.state_dict()
        
        # 将权重加载到您的模型中
        # 注意：这里可能需要一些键名的映射，因为两个模型的结构可能不完全相同
        model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

class CovNext_dinov3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = convnext_base(pretrained=True)
        
        # # 冻结所有参数
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        # state_dict = torch.hub.load_state_dict_from_url(
        #     'https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt',
        #     file_name='convnextv2_nano_pretrain.ckpt')
        # self.cnn.load_state_dict(state_dict, strict=True)
        # self.model = torch.hub.load(
        #     repo_or_dir='third_party/dinov3',
        #     model='dinov3_convnext_base',
        #     source="local",
        #     trust_repo=True
        # )
        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.lin_2 = nn.Conv2d(128, 128, 1)
        self.lin_4 = nn.Conv2d(128, 196, 1)
        self.lin_8 = nn.Conv2d(256, 256, 1)
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
        feature_pyramid = self.cnn.forward_features_all(x)                      
        feat_2_0, feat_2_1 = self.lin_2(feature_pyramid[2]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        data.update({
            'feat_c0': feat_8_0,
            'feat_c1': feat_8_1,
            'feat_m0': feat_4_0,
            'feat_m1': feat_4_1,
            'feat_f0': feat_2_0,
            'feat_f1': feat_2_1
        })

        
