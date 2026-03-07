from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
# from fvcore.nn import FlopCountAnalysis, parameter_count_table    # 计算参数量和运算量
# from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
class BSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class DBSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 dilation, stride=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise_dilated
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class R_MDB123(nn.Module):
    """MDC(r=1-2-3): 与当前 DW5x5 版一致（b1,b2,b3 的膨胀率为 1/2/3；b4 为 1×1）。"""
    def __init__(self, channels: int, residual_init: float = 0.1):
        super().__init__()
        assert channels >= 4
        base = channels // 4
        self.split_c = (base, base, base, channels - 3 * base)

        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

        self.b1 = DBSConv(self.split_c[0], self.split_c[0], kernel_size=3, padding=1, dilation=1)
        self.b2 = DBSConv(self.split_c[1], self.split_c[1], kernel_size=3, padding=2, dilation=2)
        self.b3 = DBSConv(self.split_c[2], self.split_c[2], kernel_size=3, padding=3, dilation=3)
        self.b4 = nn.Conv2d(self.split_c[3], self.split_c[3], kernel_size=1)

        self.refine_bs = BSConv(channels, channels, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.in_proj(x))
        x1, x2, x3, x4 = torch.split(x, self.split_c, dim=1)
        o1 = self.act(self.b1(x1))
        o2 = self.act(self.b2(x2))
        o3 = self.act(self.b3(x3))
        o4 = self.act(self.b4(x4))
        merged = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.act(self.refine_bs(merged))
        return residual + out * self.res_scale

class R_MDB111(nn.Module):
    """MDC(r=1-1-1): 将前三分支的膨胀率统一为 1；b4 为 1×1。"""
    def __init__(self, channels: int, residual_init: float = 0.1):
        super().__init__()
        assert channels >= 4
        base = channels // 4
        self.split_c = (base, base, base, channels - 3 * base)

        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

        self.b1 = DBSConv(self.split_c[0], self.split_c[0], kernel_size=3, padding=1, dilation=1)
        self.b2 = DBSConv(self.split_c[1], self.split_c[1], kernel_size=3, padding=1, dilation=1)
        self.b3 = DBSConv(self.split_c[2], self.split_c[2], kernel_size=3, padding=1, dilation=1)
        self.b4 = nn.Conv2d(self.split_c[3], self.split_c[3], kernel_size=1)

        self.refine_bs = BSConv(channels, channels, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.in_proj(x))
        x1, x2, x3, x4 = torch.split(x, self.split_c, dim=1)
        o1 = self.act(self.b1(x1))
        o2 = self.act(self.b2(x2))
        o3 = self.act(self.b3(x3))
        o4 = self.act(self.b4(x4))
        merged = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.act(self.refine_bs(merged))
        return residual + out * self.res_scale

class R_MDB357(nn.Module):
    """W BSConv(k=3-5-7): 将前三分支改为 BSConv，核大小分别为 3/5/7；b4 为 1×1。"""
    def __init__(self, channels: int, residual_init: float = 0.1):
        super().__init__()
        assert channels >= 4
        base = channels // 4
        self.split_c = (base, base, base, channels - 3 * base)

        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

        self.b1 = BSConv(self.split_c[0], self.split_c[0], kernel_size=3, padding=1)
        self.b2 = BSConv(self.split_c[1], self.split_c[1], kernel_size=5, padding=2)
        self.b3 = BSConv(self.split_c[2], self.split_c[2], kernel_size=7, padding=3)
        self.b4 = nn.Conv2d(self.split_c[3], self.split_c[3], kernel_size=1)

        self.refine_bs = BSConv(channels, channels, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.in_proj(x))
        x1, x2, x3, x4 = torch.split(x, self.split_c, dim=1)
        o1 = self.act(self.b1(x1))
        o2 = self.act(self.b2(x2))
        o3 = self.act(self.b3(x3))
        o4 = self.act(self.b4(x4))
        merged = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.act(self.refine_bs(merged))
        return residual + out * self.res_scale

class R_MDB333(nn.Module):
    """W BSConv(k=3-3-3): 将前三分支改为 BSConv，核大小均为 3；b4 为 1×1。"""
    def __init__(self, channels: int, residual_init: float = 0.1):
        super().__init__()
        assert channels >= 4
        base = channels // 4
        self.split_c = (base, base, base, channels - 3 * base)

        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

        self.b1 = BSConv(self.split_c[0], self.split_c[0], kernel_size=3, padding=1)
        self.b2 = BSConv(self.split_c[1], self.split_c[1], kernel_size=3, padding=1)
        self.b3 = BSConv(self.split_c[2], self.split_c[2], kernel_size=3, padding=1)
        self.b4 = nn.Conv2d(self.split_c[3], self.split_c[3], kernel_size=1)

        self.refine_bs = BSConv(channels, channels, kernel_size=3, padding=1)
        self.res_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.in_proj(x))
        x1, x2, x3, x4 = torch.split(x, self.split_c, dim=1)
        o1 = self.act(self.b1(x1))
        o2 = self.act(self.b2(x2))
        o3 = self.act(self.b3(x3))
        o4 = self.act(self.b4(x4))
        merged = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.act(self.refine_bs(merged))
        return residual + out * self.res_scale


class SMDA(nn.Module):
    """SMDA - 轻量多分支注意力模块 (现改为四分支)

    更新后特点:
    - 四分支并行: (dilation=1,2,3) + 1×1 纯通道映射
    - 前三分支: DBSConv + BSConv 组合（局部/中程/远程）
    - 第四分支: 单纯 1×1 卷积 (强调纯通道重标定)
    - 拼接后 1×1 融合, 上采样 + shortcut 残差注意力生成
    - 输出为逐元素调制后的特征
    """

    def __init__(self, n_feats, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats * shrink)

        # === 头部：1×1标准卷积，通道压缩 ===
        self.head = nn.Conv2d(n_feats, f, 1)

        # === 四分支并行架构 ===
        # 分支1: dilation=1 (局部特征)
        self.dilation1_dbsconv = DBSConv(f, f, kernel_size=3, padding=1, dilation=1)
        self.dilation1_bsconv = BSConv(f, f, kernel_size=3, padding=1)  # 保持3x3

        # 分支2: dilation=2 (中程特征)
        self.dilation2_dbsconv = DBSConv(f, f, kernel_size=3, padding=2, dilation=2)
        self.dilation2_bsconv = BSConv(f, f, kernel_size=3, padding=1)  # 保持3x3

        # self.dilation2_bsconv = BSConv(f, f, kernel_size=5, padding=2)  # 恢复为5x5，匹配ckpt

        # 分支3: dilation=3 (远程特征)
        self.dilation3_dbsconv = DBSConv(f, f, kernel_size=3, padding=3, dilation=3)
        # self.dilation3_bsconv = BSConv(f, f, kernel_size=7, padding=3)  # 恢复为7x7，匹配ckpt
        self.dilation3_bsconv = BSConv(f, f, kernel_size=3, padding=1)  # 保持3x3

        # 分支4: 纯 1×1 通道映射 (不做膨胀/空间卷积，突出通道重标定)
        self.branch4_conv = nn.Conv2d(f, f, kernel_size=1)
        # === 分支融合：拼接+1×1融合 (输入通道 f*4) ===
        self.fusion = nn.Conv2d(f * 4, f, 1)

        # === 跳跃连接：1×1投影 ===
        self.shortcut = nn.Conv2d(f, f, 1)

        # === 尾部：1×1标准卷积，通道恢复 ===
        self.tail = nn.Conv2d(f, n_feats, 1)

        self.scale = scale

    def forward(self, x):
        """前向传播"""
        # 1. 头部通道压缩
        compressed = self.head(x)  # n_feats → f

        # 2. 可扩展最大池化
        downsampled = F.max_pool2d(compressed,
                                   kernel_size=self.scale * 2 + 1,
                                   stride=self.scale)

        # 3. 四分支并行处理
        branch1 = self.dilation1_bsconv(self.dilation1_dbsconv(downsampled))      # 局部
        branch2 = self.dilation2_bsconv(self.dilation2_dbsconv(downsampled))      # 中程
        branch3 = self.dilation3_bsconv(self.dilation3_dbsconv(downsampled))      # 远程
        branch4 = self.branch4_conv(downsampled)                                  # 纯通道

        # 4. 分支融合 (f*4 -> f)
        concatenated = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        fused = self.fusion(concatenated)

        # 5. 上采样恢复空间维度
        upsampled = F.interpolate(fused,
                                  size=(x.size(2), x.size(3)),
                                  mode='bilinear',
                                  align_corners=False)

        # 6. 跳跃连接
        shortcut = self.shortcut(compressed)
        enhanced = upsampled + shortcut

        # 7. 尾部通道恢复
        attention = self.tail(enhanced)  # f → n_feats

        attention = torch.sigmoid(attention)
        # 9. 特征调制输出
        return x * attention

class ChannelAttention(nn.Module):
    """通道注意力（CA），类似 RCAN/CBAM 的通道注意力实现"""
    def __init__(self, n_feats: int, reduction: int = 16):
        super().__init__()
        mid = max(1, n_feats // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(n_feats, mid, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, n_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.conv1(w))
        w = torch.sigmoid(self.conv2(w))
        return x * w

class SpatialAttention(nn.Module):
    """空间注意力（SA），类似 CBAM 的空间注意力实现"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        w = torch.sigmoid(self.conv(s))
        return x * w

class ESA(nn.Module):
    """Enhanced Spatial Attention（ESA）- 轻量实现，常用于小型 SR 网络"""
    def __init__(self, n_feats: int, shrink: float = 0.5, pool_ks: int = 7, pool_stride: int = 3):
        super().__init__()
        f = max(1, int(n_feats * shrink))
        self.reduce = nn.Conv2d(n_feats, f, 1)
        self.conv_f = nn.Conv2d(f, f, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_ks, stride=pool_stride, padding=pool_ks // 2)
        self.conv_g = nn.Conv2d(f, f, 3, padding=1)
        self.expand = nn.Conv2d(f, n_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = x
        y = self.reduce(x)
        y = self.conv_f(y)
        y = self.pool(y)
        y = self.conv_g(y)
        y = F.interpolate(y, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        y = self.expand(y)
        a = torch.sigmoid(y)
        return idn * a

class PDAB(nn.Module):
    """
    PDAB：每次特征蒸馏采用的卷积由 conv_mode 控制

        conv_mode 可选：
            - "dw5x5" / "mdc_r123" -> R_MDB123
            - "mdc_r111"           -> R_MDB111
            - "bsconv_357"        -> R_MDB357
            - "bsconv_333"        -> R_MDB333
    """

    def __init__(self,
                 num_feat,
                 attn_shrink=0.25,
                 attentionScale=2,
                 conv_mode: str = "mdc_r123",
                 attention_type: str = "SMDA"):
        super(PDAB, self).__init__()

        self.act = nn.GELU()
        self.attention_type = attention_type.upper() if isinstance(attention_type, str) else attention_type

        # 选择卷积骨干
        if conv_mode == "mdc_r123":
            Block = R_MDB123
        elif conv_mode == "mdc_r111":
            Block = R_MDB111
        elif conv_mode == "bsconv_357":
            Block = R_MDB357
        elif conv_mode == "bsconv_333":
            Block = R_MDB333
        else:
            raise ValueError(f"Unknown conv_mode: {conv_mode}")

        # 三阶段精炼模块
        self.stage1 = Block(num_feat)
        self.stage2 = Block(num_feat)
        self.stage3 = Block(num_feat)

        # 三阶段蒸馏（先蒸馏后精炼）
        self.distill1 = nn.Conv2d(num_feat, num_feat // 2, 1)
        self.distill2 = nn.Conv2d(num_feat, num_feat // 2, 1)
        self.distill3 = nn.Conv2d(num_feat, num_feat // 2, 1)

        # 尾部精炼与聚合
        self.tail_refine = BSConv(num_feat, num_feat // 2, kernel_size=3, padding=1)
        self.merge = nn.Conv2d(num_feat // 2 * 4, num_feat, 1)

        # 注意力选择
        if self.attention_type == "SMDA":
            self.attn = SMDA(num_feat, shrink=attn_shrink, scale=attentionScale)
        elif self.attention_type == "ESA":
            self.attn = ESA(num_feat, shrink=attn_shrink)
        elif self.attention_type == "CA":
            self.attn = ChannelAttention(num_feat)
        elif self.attention_type == "SA":
            self.attn = SpatialAttention(kernel_size=7)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

    def forward(self, x):
        # 第一阶段：input 先蒸馏，再精炼
        d1 = self.act(self.distill1(x))
        s1 = self.act(self.stage1(x))

        # 第二阶段：基于 s1
        d2 = self.act(self.distill2(s1))
        s2 = self.act(self.stage2(s1))

        # 第三阶段：基于 s2
        d3 = self.act(self.distill3(s2))
        s3 = self.act(self.stage3(s2))

        # 尾部 + 聚合
        tail = self.act(self.tail_refine(s3))
        out = torch.cat([d1, d2, d3, tail], dim=1)
        out_fused = self.merge(out)

        # 注意力 + 残差
        out_fused = self.attn(out_fused)
        out_fused = out_fused + x
        return out_fused


class PDAB_NoAttn(nn.Module):
    """
    PDAB_NoAttn: PDAB 的无注意力版本 (用于 PDFE_only 消融实验)
    仅保留卷积蒸馏结构 (PDFE)，移除 SMDA 注意力模块。
    """

    def __init__(self,
                 num_feat,
                 attn_shrink=0.25,
                 attentionScale=2,
                 conv_mode: str = "mdc_r123"):
        super(PDAB_NoAttn, self).__init__()

        self.act = nn.GELU()

        # 选择卷积骨干 (与 PDAB 保持一致)
        if conv_mode == "mdc_r123":
            Block = R_MDB123
        elif conv_mode == "mdc_r111":
            Block = R_MDB111
        elif conv_mode == "bsconv_357":
            Block = R_MDB357
        elif conv_mode == "bsconv_333":
            Block = R_MDB333
        else:
            raise ValueError(f"Unknown conv_mode: {conv_mode}")

        # 三阶段精炼模块
        self.stage1 = Block(num_feat)
        self.stage2 = Block(num_feat)
        self.stage3 = Block(num_feat)

        # 三阶段蒸馏
        self.distill1 = nn.Conv2d(num_feat, num_feat // 2, 1)
        self.distill2 = nn.Conv2d(num_feat, num_feat // 2, 1)
        self.distill3 = nn.Conv2d(num_feat, num_feat // 2, 1)

        # 尾部精炼与聚合
        self.tail_refine = BSConv(num_feat, num_feat // 2, kernel_size=3, padding=1)
        self.merge = nn.Conv2d(num_feat // 2 * 4, num_feat, 1)

    def forward(self, input):
        # 第一阶段
        d1 = self.act(self.distill1(input))
        s1 = self.act(self.stage1(input))

        # 第二阶段
        d2 = self.act(self.distill2(s1))
        s2 = self.act(self.stage2(s1))

        # 第三阶段
        d3 = self.act(self.distill3(s2))
        s3 = self.act(self.stage3(s2))

        # 尾部 + 聚合
        tail = self.act(self.tail_refine(s3))
        out = torch.cat([d1, d2, d3, tail], dim=1)
        out_fused = self.merge(out)

        # 无注意力，直接残差
        return out_fused + input


class ESAOnlyBlock(nn.Module):  # 仅保留注意力, 用于参数/算力消融
    """仅包含注意力与残差, 用于 esa_only 变体参数统计

    该模块不执行蒸馏和多阶段卷积, 只计算注意力自身参数占比.
    支持 CA/SA/ESA/SMDA 四种注意力。
    """
    def __init__(self, num_feat, attn_shrink=0.25, attentionScale=2, attention_type: str = "SMDA"):
        super().__init__()
        attn_type = attention_type.upper() if isinstance(attention_type, str) else attention_type
        if attn_type == "SMDA":
            self.attn = SMDA(num_feat, shrink=attn_shrink, scale=attentionScale)
        elif attn_type == "ESA":
            self.attn = ESA(num_feat, shrink=attn_shrink)
        elif attn_type == "CA":
            self.attn = ChannelAttention(num_feat)
        elif attn_type == "SA":
            self.attn = SpatialAttention(kernel_size=7)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

    def forward(self, x):
        return self.attn(x) + x

@ARCH_REGISTRY.register()
class PMDAN(nn.Module):
    def __init__(self, in_channels, num_feat, num_block,
                 out_channels, upscale,
                 conv_mode: str = "mdc_r123",
                 variant: str = "baseline",
                 attention_type: str = "SMDA"):
        super(PMDAN, self).__init__()

        self.conv_mode = conv_mode
        self.variant = variant
        self.attention_type = attention_type

        if variant not in {"baseline", "ebfb_only", "esa_only"}:
            raise ValueError(f"Unknown variant: {variant}. Expected one of ['baseline','ebfb_only','esa_only']")

        self.BSConv_first = BSConv(in_channels * 4, num_feat, kernel_size=3, padding=1)
        # 根据 variant 选择 Block 类型
        def make_block(scale):
            if self.variant == "baseline":
                return PDAB(num_feat=num_feat, attentionScale=scale, conv_mode=conv_mode,
                             attention_type=self.attention_type)
            elif self.variant == "ebfb_only":  # 只保留 EBFB (卷积蒸馏结构), 移除注意力
                return PDAB_NoAttn(num_feat=num_feat, attentionScale=scale, conv_mode=conv_mode)
            else:  # esa_only 只保留注意力, 去除蒸馏卷积
                return ESAOnlyBlock(num_feat=num_feat, attentionScale=scale, attention_type=self.attention_type)

        self.B1 = make_block(2)
        self.B2 = make_block(2)
        self.B3 = make_block(2)
        self.B4 = make_block(2)
        self.B5 = make_block(3)
        self.B6 = make_block(3)
        self.B7 = make_block(3)
        self.B8 = make_block(4)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2_conv = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        # self.c2_bsconv = BSConv(num_feat, num_feat, kernel_size=3, padding=1)
        self.upsampler = Upsamplers.PixelShuffleDirect(
            scale=upscale, num_feat=num_feat, num_out_ch=out_channels)

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.BSConv_first(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4,
                           out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_lr = self.c2_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output



if __name__ == "__main__":
#     from thop import profile
#
#     # 基于注意力类型的消融：CA / SA / ESA / SMDA
#     x = torch.randn(1, 3, 320, 180)
#     attn_types = ["CA", "SA", "ESA", "SMDA"]
#     for attn in attn_types:
#         net = DFDAN_v5(3, 56, 8, 3, 4, variant="baseline", attention_type=attn)
#         net.eval()
#         flops, params = profile(net, (x,))
#         print(f"=== Attention: {attn} ===")
#         print(f"Params: {params/1e3:.2f}K")
#         print(f"FLOPs:  {flops/1e9:.3f}G")

    # if __name__ == "__main__":
        from thop import profile

        x = torch.randn(1, 3, 320, 180)

        modes = [ "mdc_r123", "mdc_r111", "bsconv_357", "bsconv_333"]

        for mode in modes:
            net = PMDAN(in_channels=3,
                           num_feat=56,
                           num_block=8,
                           out_channels=3,
                           upscale=4,
                           conv_mode=mode)
            net.eval()
            flops, params = profile(net, (x,))
            print(f"=== conv_mode = {mode} ===")
            print("Params: {:.3f}K".format(params / 1e3))
            print("FLOPs:  {:.4f}G (64x64)".format(flops / 1e9))

        # === PDAB / PDFE_only / SMDA_only 消融 ===
        print("\n=== PDAB / PDFE_only / SMDA_only Ablation ===")
        ablations = [
            ("PDAB", "baseline"),        # 完整块（PDFE+SMDA）
            ("PDFE_only", "ebfb_only"),  # 仅卷积蒸馏/特征提取（无注意力）
            ("SMDA_only", "esa_only"),   # 仅注意力（无卷积蒸馏）
        ]
        for name, variant in ablations:
            net = PMDAN(in_channels=3,
                           num_feat=56,
                           num_block=8,
                           out_channels=3,
                           upscale=4,
                           conv_mode="mdc_r123",
                           variant=variant,
                           attention_type="SMDA")
            net.eval()
            flops, params = profile(net, (x,))
            print(f"{name}: Params={params/1e3:.2f}K, Multi-Adds={flops/1e9:.3f}G")
