import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


class CFTFuseBlock(nn.Module):
    """
    CodeFormer 风格的 CFT 融合块。
    将 res1（来自 LatentCodec）以 SFT 仿射变换注入 VAE decoder 特征。

    Args:
        res_ch:  res1 的通道数（来自 LatentCodec）
        dec_ch:  注入点 decoder 特征的通道数
    """
    def __init__(self, res_ch: int, dec_ch: int):
        super().__init__()
        # 把 res1 投影到 dec_ch，与 dec_feat 对齐后拼接
        self.res_proj = nn.Conv2d(res_ch, dec_ch, kernel_size=1)
        # 联合编码 → 预测仿射参数
        self.encode_enc = ResBlock(2 * dec_ch, dec_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
        )

    def forward(self, res_feat: torch.Tensor, dec_feat: torch.Tensor, w: float = 1.0):
        # 1. 空间对齐（res1 尺寸一般小于 decoder 当前尺寸）
        if res_feat.shape[-2:] != dec_feat.shape[-2:]:
            res_feat = F.interpolate(
                res_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        # 2. 通道投影
        res_feat = self.res_proj(res_feat)
        # 3. 联合编码
        enc_feat = self.encode_enc(torch.cat([res_feat, dec_feat], dim=1))
        # 4. 预测 scale / shift
        alpha = self.scale(enc_feat)   # α
        beta  = self.shift(enc_feat)   # β
        # 5. 残差注入：out = dec + w*(dec*α + β)
        return dec_feat + w * (dec_feat * alpha + beta)