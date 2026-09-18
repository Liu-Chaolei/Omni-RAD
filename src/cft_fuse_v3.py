import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class LambdaAdaptiveW(nn.Module):
    """
    从损失函数码率权重 λ 预测全局融合权重 w ∈ [0, 1]。
    λ 大（高码率）→ w ≈ 1（偏 res1），λ 小（低码率）→ w ≈ 0（偏 x_diff）。
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, lmbda: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lmbda: scalar tensor or shape [B] / [B, 1]
        Returns:
            w: shape [B, 1], values in [0, 1]
        """
        if lmbda.dim() == 0:
            lmbda = lmbda.unsqueeze(0).unsqueeze(0)  # [] → [1, 1]
        elif lmbda.dim() == 1:
            lmbda = lmbda.unsqueeze(-1)               # [B] → [B, 1]
        return self.mlp(lmbda)


class SpatialFuseAttention(nn.Module):
    """
    带差异先验的空间注意力模块。
    输入 res_proj 和 dec_feat（已对齐通道），额外计算 |res_proj - dec_feat| 作为
    纹理/平坦区域的归纳偏置，输出 [B, 1, H, W] 的空间权重图。

    平坦区域差异小 → 高值（偏 res1 / PSNR），纹理区域差异大 → 低值（偏 x_diff / 感知）。
    """
    def __init__(self, dec_ch: int):
        super().__init__()
        mid = dec_ch // 4
        self.net = nn.Sequential(
            nn.Conv2d(3 * dec_ch, mid, 1),        # 1×1 降维
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid, mid, 3, padding=1),     # 3×3 空间混合
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid, 1, 1),                  # → [B, 1, H, W]
            nn.Sigmoid(),
        )

    def forward(self, res_proj: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(res_proj - dec_feat)
        return self.net(torch.cat([res_proj, dec_feat, diff], dim=1))


class CFTFuseBlock(nn.Module):
    """
    CFT 融合块（v3）：SFT 仿射变换 + 空间注意力。
    将 res1 以 SFT 注入 VAE decoder 特征，融合强度由
    全局 w（来自 LambdaAdaptiveW）× 空间 attn_map 共同控制。
    """
    def __init__(self, res_ch: int, dec_ch: int):
        super().__init__()
        self.res_proj = nn.Conv2d(res_ch, dec_ch, kernel_size=1)
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
        self.spatial_attn = SpatialFuseAttention(dec_ch)

    def _inner_forward(self, res_feat: torch.Tensor, dec_feat: torch.Tensor, w_tensor: torch.Tensor):
        # 1. 空间对齐
        if res_feat.shape[-2:] != dec_feat.shape[-2:]:
            res_feat = F.interpolate(
                res_feat, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        # 2. 通道投影
        res_proj = self.res_proj(res_feat)
        # 3. 联合编码
        enc_feat = self.encode_enc(torch.cat([res_proj, dec_feat], dim=1))
        # 4. 预测 scale / shift
        alpha = self.scale(enc_feat)
        beta = self.shift(enc_feat)
        # 5. 空间注意力
        attn_map = self.spatial_attn(res_proj, dec_feat)  # [B, 1, H, W]
        # 6. 全局 w × 空间 attn_map
        combined_w = w_tensor * attn_map
        # 7. 残差注入
        return dec_feat + combined_w * (dec_feat * alpha + beta)

    def forward(self, res_feat: torch.Tensor, dec_feat: torch.Tensor, w: float = 1.0):
        if isinstance(w, torch.Tensor) and w.dim() >= 1:
            w_tensor = w.view(-1, 1, 1, 1)  # [B, 1] → [B, 1, 1, 1]
        else:
            w_tensor = torch.tensor(w, device=dec_feat.device, dtype=dec_feat.dtype)
        if self.training:
            return checkpoint(self._inner_forward, res_feat, dec_feat, w_tensor, use_reentrant=False)
        return self._inner_forward(res_feat, dec_feat, w_tensor)
