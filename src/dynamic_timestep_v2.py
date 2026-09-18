"""Optimized dynamic-timestep module for variable-rate StableCodec.

Drop-in replacement for ``DynamicTimestepModule`` in ``latent_codec_variable2_step.py``.
This module is intentionally kept in a *separate* file so the original module
remains untouched.

Improvements over the baseline (and the experimental evidence behind each one):

P0 — inference-only fixes (no retraining required)
==================================================
1. Data-driven ``sigma_quant``: replaces the constant ``1/12`` with the actual
   per-sample quantization variance ``Var(y - y_hat)`` when ``y`` and ``y_hat``
   are provided. Motivated by the observation that the ``1/12`` approximation
   breaks at low bitrate (high λ), where per-λ Pearson collapsed from 0.61
   (λ=9) to 0.37 (λ=16).

2. Soft saturation in place of hard ``clamp(t_max)``: a smooth exponential
   schedule keeps the high-λ tail from being clipped to a single value.
   Motivated by the cluster of yellow points on the right edge of the
   T_calc-vs-T_opt scatter plot.

3. Optional isotonic calibration: at inference, an offline-fit isotonic
   regression mapping from T_calc to T_opt can be applied. Since
   Pearson(0.45) ≈ Spearman(0.48), the relationship is monotonic but not
   linear — a monotonic recalibration is the cheapest way to align them.

P1 — robustness (still inference-only)
======================================
4. Asymmetric safety bias: distortion-vs-T curves are steeper on the right
   side (≈2.5× the left slope), so a small negative bias trades a tiny
   left-side cost for a big right-side gain in worst-case scenarios.

5. Low-λ fallback: at λ ≲ 1.5 the distortion-vs-T curve is almost flat
   (per-λ Pearson 0.17 at λ=1), so dynamic T mostly contributes noise.
   Smoothly blend toward a fixed T at low λ.

P2 — capacity (requires retraining / fine-tuning)
=================================================
6. Learned ΔT head conditioned on codec-internal statistics + film_embed.
   ANOVA showed η²(image)=44.5% vs η²(λ)=28.0%; pixel-level complexity
   metrics (Laplacian, entropy, JPEG bpp) had |ρ|<0.08, so the image effect
   is mediated by the codec's latent representation rather than raw image
   content. A small head that ingests scale statistics + film_embed is the
   natural way to capture it.

Usage
-----
The module is API-compatible with the original ``DynamicTimestepModule``
when called with a single ``scales_all`` argument. All other inputs are
keyword-only and optional, so existing call sites keep working::

    self.dynamic_timestep = DynamicTimestepModuleV2(
        alphas_cumprod=scheduler.alphas_cumprod,
        t_min=800, t_max=999,
        soft_saturation=True,
        calibration_path="checkpoints/T_calibration.pt",  # optional
        use_delta_head=False,                              # set True after fine-tune
        safety_bias=2.0,
        lambda_floor=1.5,
        fallback_T=870.0,
    )

    T_star = self.dynamic_timestep(
        scales_all,
        y=y, y_hat=y_hat,           # enables data-driven sigma_quant
        film_embed=film_embed,      # required iff use_delta_head=True
        lmbda=lmbda,                # enables low-λ fallback
    )

The companion script ``calibrate_timestep.py`` produces the calibration
checkpoint from ``per_image.csv``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DeltaTHead(nn.Module):
    """Learned ΔT head conditioned on codec-internal statistics + film_embed.

    Outputs a bounded correction in ``[-max_delta, +max_delta]`` (timesteps).
    """

    def __init__(
        self,
        film_dim: int,
        hidden: int = 64,
        max_delta: float = 30.0,
    ) -> None:
        super().__init__()
        self.max_delta = max_delta
        # 4 statistics from scales_all + film_embed
        in_dim = 4 + film_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Initialize the last layer to ~0 so the head starts as a no-op,
        # letting the SNR formula carry inference until training kicks in.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        scales_all: torch.Tensor,
        film_embed: torch.Tensor,
    ) -> torch.Tensor:
        s = scales_all
        feats = torch.stack(
            [
                s.mean(dim=[1, 2, 3]),
                s.std(dim=[1, 2, 3]),
                s.amax(dim=[1, 2, 3]),
                s.amin(dim=[1, 2, 3]),
            ],
            dim=-1,
        )  # [B, 4]
        x = torch.cat([feats, film_embed], dim=-1)
        return self.max_delta * torch.tanh(self.net(x).squeeze(-1))


class DynamicTimestepModuleV2(nn.Module):
    """Optimized dynamic-timestep module.

    Args:
        alphas_cumprod: 1-D tensor of length T (typically 1000) from
            ``DDPMScheduler.alphas_cumprod``.
        t_min: lower bound of the output T* range.
        t_max: upper bound of the output T* range.
        soft_saturation: if True, replace the linear ``T_min + (T_max-T_min) * t/999``
            mapping with a smoother exponential schedule that does not clip
            the tail. Recommended for high-λ regimes.
        soft_sat_alpha: steepness of the soft saturation (only used when
            ``soft_saturation=True``). Smaller = closer to linear.
        calibration_path: optional path to a torch checkpoint produced by
            ``calibrate_timestep.py``. If provided, applies an isotonic
            recalibration ``T_calc → T_opt`` at inference time.
        use_delta_head: enable the learned ΔT head (P2). Requires
            ``film_embed`` to be passed at forward.
        delta_head_film_dim: width of the FiLM embedding (must match the
            codec's ``film_embed``).
        delta_head_hidden: hidden width of the ΔT MLP.
        delta_head_max: maximum absolute correction in timesteps.
        safety_bias: subtract this many timesteps from T* before clamp;
            exploits the asymmetric distortion-vs-T curve.
        lambda_floor: λ value below which the output is smoothly blended
            toward ``fallback_T``. Set to 0 to disable.
        fallback_T: fixed T used as the low-λ anchor.
    """

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        t_min: int = 800,
        t_max: int = 999,
        *,
        soft_saturation: bool = True,
        soft_sat_alpha: float = 3.0,
        calibration_path: Optional[str] = None,
        use_delta_head: bool = False,
        delta_head_film_dim: int = 64,
        delta_head_hidden: int = 64,
        delta_head_max: float = 30.0,
        safety_bias: float = 0.0,
        lambda_floor: float = 0.0,
        fallback_T: float = 870.0,
    ) -> None:
        super().__init__()
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.soft_saturation = bool(soft_saturation)
        self.soft_sat_alpha = float(soft_sat_alpha)
        self.safety_bias = float(safety_bias)
        self.lambda_floor = float(lambda_floor)
        self.fallback_T = float(fallback_T)

        ac = alphas_cumprod.float()
        self.register_buffer("alphas_cumprod", ac)
        snr_schedule = ac / (1.0 - ac).clamp_min(1e-12)
        self.register_buffer("snr_schedule", snr_schedule)
        # The schedule length determines the divisor used by the linear/soft
        # rescale; usually 999 for a 1000-step DDPM.
        self.schedule_len = int(snr_schedule.numel() - 1)

        if calibration_path is not None:
            ckpt = torch.load(calibration_path, map_location="cpu")
            self.register_buffer("cal_xs", ckpt["xs"].float())
            self.register_buffer("cal_ys", ckpt["ys"].float())
            self.use_calibration = True
        else:
            self.use_calibration = False

        self.use_delta_head = bool(use_delta_head)
        if self.use_delta_head:
            self.delta_head = DeltaTHead(
                film_dim=delta_head_film_dim,
                hidden=delta_head_hidden,
                max_delta=delta_head_max,
            )
        else:
            self.delta_head = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _compute_sigma_quant(
        self,
        scales_all: torch.Tensor,
        y: Optional[torch.Tensor],
        y_hat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return per-sample quantization variance (P0 fix #1)."""
        B = scales_all.size(0)
        if y is not None and y_hat is not None:
            sigma = (y - y_hat).var(dim=[1, 2, 3]).clamp_min(1e-8)
        else:
            sigma = scales_all.new_full((B,), 1.0 / 12.0)
        return sigma

    def _rescale_to_range(self, T_raw: torch.Tensor) -> torch.Tensor:
        """Map searchsorted-derived T_raw into [t_min, t_max] (P0 fix #2)."""
        T_norm = T_raw.float() / float(self.schedule_len)
        if self.soft_saturation:
            mapped = 1.0 - torch.exp(-self.soft_sat_alpha * T_norm)
            # Re-normalize so that T_norm=1 still yields t_max exactly.
            denom = 1.0 - torch.exp(
                torch.tensor(-self.soft_sat_alpha, device=T_raw.device)
            )
            mapped = mapped / denom
        else:
            mapped = T_norm
        return self.t_min + (self.t_max - self.t_min) * mapped

    def _apply_calibration(self, T_star: torch.Tensor) -> torch.Tensor:
        """Apply isotonic calibration via lookup (P0 fix #3)."""
        idx = torch.searchsorted(self.cal_xs, T_star)
        idx = idx.clamp(0, self.cal_xs.numel() - 1)
        return self.cal_ys[idx]

    def _apply_low_lambda_fallback(
        self,
        T_star: torch.Tensor,
        lmbda: torch.Tensor,
    ) -> torch.Tensor:
        """Blend toward a fixed T at low λ (P1 fix #6)."""
        # Sigmoid centered at ``lambda_floor`` in log-space.
        log_lam = torch.log(lmbda.clamp_min(1e-6))
        center = torch.log(torch.tensor(self.lambda_floor, device=lmbda.device))
        weight = torch.sigmoid(2.0 * (log_lam - center))  # → 1 above floor
        fixed = T_star.new_full(T_star.shape, self.fallback_T)
        return weight * T_star + (1.0 - weight) * fixed

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        scales_all: torch.Tensor,
        *,
        y: Optional[torch.Tensor] = None,
        y_hat: Optional[torch.Tensor] = None,
        film_embed: Optional[torch.Tensor] = None,
        lmbda: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the per-sample dynamic timestep.

        Args:
            scales_all: ``[B, C, H, W]`` summed scales from the 4-pass
                checkerboard context model.
            y, y_hat: optional ``[B, C, H, W]`` tensors. When both are
                provided, ``sigma_quant`` is estimated from
                ``Var(y - y_hat)`` per sample.
            film_embed: ``[B, film_dim]``; required when
                ``use_delta_head=True``.
            lmbda: ``[B]``; required when ``lambda_floor > 0``.

        Returns:
            T_star: ``[B]`` float tensor in ``[t_min, t_max]``.
        """
        sigma_quant = self._compute_sigma_quant(scales_all, y, y_hat)
        signal_var = scales_all.mean(dim=[1, 2, 3]) ** 2
        snr_compress = signal_var / sigma_quant

        T_raw = torch.searchsorted(-self.snr_schedule, -snr_compress)
        T_star = self._rescale_to_range(T_raw)

        if self.use_delta_head:
            if film_embed is None:
                raise ValueError("film_embed is required when use_delta_head=True")
            T_star = T_star + self.delta_head(scales_all, film_embed)

        if self.use_calibration:
            # Calibration is fit on uncorrected T_calc, so apply before bias.
            T_star = self._apply_calibration(T_star)

        if self.safety_bias != 0.0:
            T_star = T_star - self.safety_bias

        if self.lambda_floor > 0.0:
            if lmbda is None:
                raise ValueError("lmbda is required when lambda_floor > 0")
            T_star = self._apply_low_lambda_fallback(T_star, lmbda)

        return T_star.clamp(self.t_min, self.t_max)
