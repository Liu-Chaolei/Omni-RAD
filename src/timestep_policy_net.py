"""timestep_policy_net.py — Learned timestep policy for StableCodec.

Provides:
  - TimestepPolicyNet: lightweight MLP predicting delta_T relative to T_snr
  - build_timestep_features: assembles 13-dim decoder-side feature vector
  - SNRTimestepFeature: extracts T_snr and snr_compress from entropy scales
    (downgraded from DynamicTimestepModule — no longer decides final timestep)

Key design: PolicyNet predicts a small correction delta_T to T_snr, not absolute T.
This prevents the network from collapsing to low timesteps (which gives better
reconstruction but breaks the codec's rate-distortion tradeoff).
"""

import math
import torch
import torch.nn as nn


class SNRTimestepFeature(nn.Module):
    """Extract SNR-based timestep feature from entropy model scales.

    Produces T_snr and snr_compress as weak features for PolicyNet input.
    Does NOT decide the final UNet timestep.
    """

    def __init__(self, alphas_cumprod: torch.Tensor, t_min: int = 870, t_max: int = 999):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        ac = alphas_cumprod.float()
        self.register_buffer("alphas_cumprod", ac)
        snr_schedule = ac / (1.0 - ac)
        self.register_buffer("snr_schedule", snr_schedule)

    def forward(self, scales_all: torch.Tensor):
        """
        Args:
            scales_all: [B, 320, H, W] summed scales from checkerboard context.
        Returns:
            T_snr: [B] float, SNR-derived timestep (baseline for PolicyNet)
            snr_compress: [B] float, compression SNR value
        """
        sigma_quant = 1.0 / 12.0
        signal_var = scales_all.mean(dim=[1, 2, 3]) ** 2
        snr_compress = signal_var / sigma_quant
        T_raw = torch.searchsorted(-self.snr_schedule, -snr_compress)
        T_snr = self.t_min + (self.t_max - self.t_min) * (T_raw.float() / 999.0)
        T_snr = T_snr.clamp(self.t_min, self.t_max)
        return T_snr, snr_compress


def build_timestep_features(
    lmbda: torch.Tensor,
    bpp: torch.Tensor,
    scales_all: torch.Tensor,
    y_hat: torch.Tensor,
    sample: torch.Tensor,
    res1: torch.Tensor,
    T_snr: torch.Tensor,
    snr_compress: torch.Tensor,
) -> torch.Tensor:
    """Assemble 13-dim decoder-side feature vector for PolicyNet.

    All inputs are detached where needed by the caller. Features are chosen
    to be available at decode time (no dependence on original image).

    Args:
        lmbda: [B] per-image lambda values
        bpp: [B] actual bits-per-pixel (from rate module, detached)
        scales_all: [B, 320, H, W] entropy model scales
        y_hat: [B, 320, H', W'] quantized latent
        sample: [B, 256, H, W] lq_latent_hat[:, :256] (UNet input)
        res1: [B, 256, H, W] AuxDecoder residual
        T_snr: [B] SNR-derived timestep (baseline)
        snr_compress: [B] compression SNR

    Returns:
        features: [B, 13] normalized feature vector
    """
    eps = 1e-8
    B = lmbda.shape[0] if lmbda.dim() > 0 else scales_all.shape[0]
    device = scales_all.device

    # Ensure all scalar inputs are expanded to [B]
    def _to_B(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            return t.expand(B)
        return t

    lmbda = _to_B(lmbda)
    bpp = _to_B(bpp)
    T_snr = _to_B(T_snr)
    snr_compress = _to_B(snr_compress)

    scales_flat = scales_all.flatten(2)
    scales_mean = scales_flat.mean(dim=[1, 2])
    scales_std = scales_flat.std(dim=[1, 2])
    scales_p90 = torch.quantile(
        scales_flat.float(), 0.9, dim=2
    ).mean(dim=1)

    y_hat_flat = y_hat.flatten(1)
    y_hat_energy = y_hat_flat.pow(2).mean(dim=1)
    y_hat_std = y_hat_flat.std(dim=1)

    sample_flat = sample.flatten(1)
    sample_energy = sample_flat.pow(2).mean(dim=1)
    sample_std = sample_flat.std(dim=1)

    res1_flat = res1.flatten(1)
    res1_norm = res1_flat.abs().mean(dim=1)
    res1_energy = res1_flat.pow(2).mean(dim=1)

    features = torch.stack([
        torch.log(lmbda + eps),          # 0: log(lambda)
        bpp,                              # 1: actual bpp
        T_snr / 999.0,                    # 2: normalized T_snr
        snr_compress,                     # 3: compression SNR
        scales_mean,                      # 4: scales mean
        scales_std,                       # 5: scales std
        scales_p90,                       # 6: scales 90th percentile
        y_hat_energy,                     # 7: y_hat energy
        y_hat_std,                        # 8: y_hat std
        sample_energy,                    # 9: sample energy
        sample_std,                       # 10: sample std
        res1_norm,                        # 11: res1 L1 norm
        res1_energy,                      # 12: res1 energy
    ], dim=1)  # [B, 13]

    return features


POLICY_FEATURE_DIM = 13


class TimestepPolicyNet(nn.Module):
    """Learned timestep policy: predicts delta_T relative to T_snr baseline.

    Key design: T_pred = T_snr + delta_T, where delta_T is bounded to a small range.
    This prevents the network from collapsing to low timesteps (which gives better
    reconstruction but breaks the codec's rate-distortion tradeoff).

    The network predicts a small correction to the SNR-derived baseline,
    allowing it to fine-tune the timestep based on other decoder-side features
    while staying anchored to the physically-motivated T_snr.
    """

    def __init__(self, in_dim: int = POLICY_FEATURE_DIM, hidden: int = 128,
                 t_min: float = 870.0, t_max: float = 999.0,
                 delta_max: float = 30.0):
        """
        Args:
            in_dim: input feature dimension
            hidden: hidden layer size
            t_min: minimum allowed timestep (for clamping)
            t_max: maximum allowed timestep (for clamping)
            delta_max: maximum allowed |delta_T| from T_snr (default 30 steps)
        """
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.delta_max = delta_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor, T_snr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, in_dim] from build_timestep_features()
            T_snr: [B] SNR-derived baseline timestep
        Returns:
            T_pred: [B] float, T_snr + delta_T, clamped to [t_min, t_max]
        """
        u = self.net(features).squeeze(-1)  # [B]
        # delta_T ∈ [-delta_max, +delta_max] via tanh
        delta_T = self.delta_max * torch.tanh(u)
        # T_pred = T_snr + delta_T, clamped to valid range
        T_pred = T_snr + delta_T
        T_pred = T_pred.clamp(self.t_min, self.t_max)
        return T_pred
