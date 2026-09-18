"""experiment_oracle_timestep.py — Path A of "Optimal-Timestep Learning Plan".

Implements Section 5 (路径 A：真实最优时间步分析) of
``最优时间步学习实验方案.md``: per-(image, λ) Oracle-T sweep, multi-oracle
resolution (PSNR / MS-SSIM / LPIPS / DISTS / combined), Section 5.4 feature
extraction, and per-feature Pearson / Spearman correlation reporting.

Outputs three CSVs under ``--out_dir``:
    * oracle_timestep_sweep.csv     per (image, λ, T) row with metrics
    * oracle_timestep_summary.csv   per (image, λ) row with oracles + features
    * correlation_table.csv         feature-vs-T_oracle correlations

Run:
    CUDA_VISIBLE_DEVICES=0 python src/experiment_oracle_timestep.py \
        --base_config ./configs/base.yaml \
        --test_config ./configs/test.yaml \
        --out_dir     ./results/oracle_timestep \
        --num_lambdas 6
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from torch_ema import ExponentialMovingAverage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("oracle_timestep")

DEFAULT_T_MIN: int = 800
DEFAULT_T_MAX: int = 999
DEFAULT_T_STEP: int = 10

DEFAULT_LAMBDA_MIN: float = 0.2
DEFAULT_LAMBDA_MAX: float = 128.0

# Combined-oracle scoring weights: w1 * norm(DISTS) + w2 * norm(LPIPS) - w3 * norm(PSNR)
DEFAULT_SCORE_WEIGHTS: Tuple[float, float, float] = (0.5, 0.5, 0.3)

ORACLE_METRIC_CHOICES: Tuple[str, ...] = ("lpips", "dists", "psnr", "combined")

EPS: float = 1e-8

WEIGHT_PATH = ''

def _build_t_candidates(t_min: int, t_max: int, t_step: int) -> List[int]:
    """``range(t_min, t_max, t_step)`` with the upper bound always included."""
    cand = list(range(int(t_min), int(t_max), max(1, int(t_step))))
    if not cand or cand[-1] != int(t_max):
        cand.append(int(t_max))
    return cand


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpConfig:
    """Frozen experiment configuration."""

    base_config: str
    test_config: str
    out_dir: str
    codec_path: Optional[str] = None
    num_lambdas: int = 6
    lambda_min: float = DEFAULT_LAMBDA_MIN
    lambda_max: float = DEFAULT_LAMBDA_MAX
    t_min: int = DEFAULT_T_MIN
    t_max: int = DEFAULT_T_MAX
    t_step: int = DEFAULT_T_STEP
    t_ref: int = 999
    oracle_metric: str = "lpips"
    score_weights: Tuple[float, float, float] = DEFAULT_SCORE_WEIGHTS
    max_images: Optional[int] = None
    device: str = "cuda:0"
    seed: int = 42
    make_plots: bool = False


@dataclass
class SweepRow:
    """One row of ``oracle_timestep_sweep.csv`` (per image x lambda x T)."""

    image_id: str
    lmbda: float
    actual_bpp: float
    T: int
    psnr: float
    ms_ssim: float
    lpips: float
    dists: float
    mse: float
    rate_loss: float


@dataclass
class SummaryRow:
    """One row of ``oracle_timestep_summary.csv`` (per image x lambda)."""

    image_id: str
    lmbda: float
    actual_bpp: float
    T_snr: int
    T_oracle_psnr: int
    T_oracle_ms_ssim: int
    T_oracle_lpips: int
    T_oracle_dists: int
    T_oracle_main: int
    best_score: float
    score_at_T_snr: float
    score_gap: float
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file from disk."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive shallow-merge for nested config dicts (later wins)."""
    out: Dict[str, Any] = {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge_dicts(out[k], v)
            else:
                out[k] = v
    return out


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation that returns NaN if either side is degenerate."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or xa.std() < 1e-12 or ya.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation via double argsort."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(xa))
    ry = np.argsort(np.argsort(ya))
    return safe_corr(rx, ry)


def preprocess_image(path: str, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    """Load an image, scale to ``[-1, 1]`` and right/bottom-pad to a multiple of 64."""
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [0,1]
    tensor = tensor * 2.0 - 1.0
    _, _, h, w = tensor.shape
    stride_h, stride_w = 64, 64
    pad_h = (math.ceil(h / stride_h) * stride_h) - h
    pad_w = (math.ceil(w / stride_w) * stride_w) - w
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w


def normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D array to ``[0, 1]``; constant arrays map to zeros."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Codec forward (with scales_all capture) and per-T decode
# ---------------------------------------------------------------------------


class _ScalesCapture:
    """Context manager that monkey-patches ``codec.dynamic_timestep.forward`` to record scales_all."""

    def __init__(self, dynamic_timestep_module: Any) -> None:
        self._mod = dynamic_timestep_module
        self._orig: Optional[Callable[..., Any]] = None
        self.scales_all: Optional[torch.Tensor] = None

    def __enter__(self) -> "_ScalesCapture":
        self._orig = self._mod.forward
        capture = self

        def _patched_forward(scales_all: torch.Tensor) -> torch.Tensor:
            capture.scales_all = scales_all.detach()
            return capture._orig(scales_all)  # type: ignore[misc]

        self._mod.forward = _patched_forward  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._orig is not None:
            self._mod.forward = self._orig  # type: ignore[assignment]


def codec_forward(
    net: Any,
    img_padded: torch.Tensor,
    ori_h: int,
    ori_w: int,
    lmbda_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """Run codec/encoder once and capture scales_all + likelihoods + film_embed."""
    latent2 = net.aux_codec((img_padded + 1.0) / 2.0).detach()
    lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor

    with _ScalesCapture(net.codec.dynamic_timestep) as cap:
        lq_latent_hat, rate_out, res1, T_calc = net.codec(
            lq_latent, latent2, ori_h, ori_w, lmbda_tensor
        )

    film_embed = net.codec.film_embed(lmbda_tensor)
    return {
        "lq_latent_hat": lq_latent_hat,
        "res1": res1,
        "T_calc": T_calc,
        "scales_all": cap.scales_all,
        "rate_out": rate_out,
        "film_embed": film_embed,
        "lq_latent": lq_latent,
        "latent2": latent2,
    }


def decode_at_timestep(
    net: Any,
    ctx: Dict[str, Any],
    t_value: int,
    pos_caption_enc: torch.Tensor,
    return_internals: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run UNet -> _batched_ddpm_step -> VAE.decode at a fixed timestep.

    When ``return_internals`` is True, also returns the cached ``model_pred``,
    ``x0_pred`` (post-DDPM step but pre-res1) and ``x_denoised`` for the Section
    5.4.5 UNet-response features.
    """
    device = ctx["lq_latent_hat"].device
    B = ctx["lq_latent_hat"].shape[0]

    # Adaptive UNet LoRA scaling driven by lambda-FiLM embedding.
    delta_s = net.unet_lora_proj(ctx["film_embed"])
    lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)

    t_long = torch.full((B,), int(t_value), dtype=torch.long, device=device)

    # Repeat caption embedding to match batch size if necessary.
    if pos_caption_enc.shape[0] != B:
        pos_caption_enc = pos_caption_enc.expand(B, -1, -1)

    model_pred = (
        net.unet(ctx["lq_latent_hat"], t_long, encoder_hidden_states=pos_caption_enc).sample
        * lora_scale
    )

    x0_pred = net._batched_ddpm_step(model_pred, t_long, ctx["lq_latent_hat"][:, :256])
    x_denoised = x0_pred + ctx["res1"]
    output = net.vae.decode(x_denoised / net.vae.config.scaling_factor).sample.clamp(-1, 1)

    internals: Dict[str, torch.Tensor] = {}
    if return_internals:
        internals = {
            "model_pred": model_pred.detach(),
            "x0_pred": x0_pred.detach(),
            "x_denoised": x_denoised.detach(),
        }
    return output, internals


# ---------------------------------------------------------------------------
# Section 5.4 feature extraction
# ---------------------------------------------------------------------------


def _percentile(t: torch.Tensor, q: float) -> float:
    """Numerically stable percentile that handles small/empty tensors."""
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t.flatten().float(), q).item())


def _spatial_gradient_energy(latent: torch.Tensor) -> float:
    """Mean squared first-order finite differences across H and W."""
    dh = latent[..., 1:, :] - latent[..., :-1, :]
    dw = latent[..., :, 1:] - latent[..., :, :-1]
    return float((dh.pow(2).mean() + dw.pow(2).mean()).item())


def _laplacian_energy(latent: torch.Tensor) -> float:
    """3x3 Laplacian energy averaged over channels and batch."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=latent.device,
        dtype=latent.dtype,
    ).view(1, 1, 3, 3)
    C = latent.shape[1]
    weight = kernel.expand(C, 1, 3, 3)
    lap = F.conv2d(latent, weight, padding=1, groups=C)
    return float(lap.pow(2).mean().item())


def _entropy_features(rate_out: Any) -> Dict[str, float]:
    """Statistics of -log p(y_hat) (NLL per element) from the gaussian conditional."""
    out: Dict[str, float] = {
        "mean_nll": float("nan"),
        "std_nll": float("nan"),
        "nll_p90": float("nan"),
        "bits_per_symbol": float("nan"),
    }
    likelihoods = getattr(rate_out, "quantized_latent_likelihoods", None)
    if likelihoods is None:
        likelihoods = getattr(rate_out, "latent_likelihoods", None)
    if likelihoods is None:
        return out
    nll = -torch.log(likelihoods.clamp_min(1e-12))
    out["mean_nll"] = float(nll.mean().item())
    out["std_nll"] = float(nll.std().item())
    out["nll_p90"] = _percentile(nll, 0.9)
    out["bits_per_symbol"] = float((nll / math.log(2.0)).mean().item())
    return out


def collect_timestep_features(
    *,
    lmbda: float,
    actual_bpp: float,
    target_bpp: Optional[float],
    scales_all: Optional[torch.Tensor],
    rate_out: Any,
    lq_latent_hat: torch.Tensor,
    res1: torch.Tensor,
    sample_ref: torch.Tensor,
    x0_pred_ref: torch.Tensor,
    T_snr: int,
) -> Dict[str, float]:
    """Build the Section 5.4 feature dictionary for one (image, lambda)."""
    feats: Dict[str, float] = {}

    # 5.4.1 rate / control
    feats["lambda"] = float(lmbda)
    feats["log_lambda"] = float(math.log(max(lmbda, EPS)))
    feats["actual_bpp"] = float(actual_bpp)
    if target_bpp is not None:
        feats["target_bpp"] = float(target_bpp)
        feats["bpp_gap"] = float(actual_bpp - target_bpp)

    # 5.4.2 scales / SNR
    if scales_all is not None:
        s = scales_all.float()
        feats["scales_mean"] = float(s.mean().item())
        feats["scales_std"] = float(s.std().item())
        feats["scales_min"] = float(s.min().item())
        feats["scales_max"] = float(s.max().item())
        feats["scales_p10"] = _percentile(s, 0.1)
        feats["scales_p50"] = _percentile(s, 0.5)
        feats["scales_p90"] = _percentile(s, 0.9)
        feats["SNR_compress"] = float((s.mean() ** 2 / (1.0 / 12.0)).item())
    feats["T_snr"] = float(T_snr)

    # 5.4.3 entropy NLL
    feats.update(_entropy_features(rate_out))

    # 5.4.4 latent content complexity (sample = lq_latent_hat[:, :256])
    sample = sample_ref.float()
    feats["latent_mean_abs"] = float(sample.abs().mean().item())
    feats["latent_std"] = float(sample.std().item())
    feats["latent_energy"] = float(sample.pow(2).mean().item())
    chan_energy = sample.pow(2).mean(dim=[0, 2, 3])
    feats["latent_channel_energy_std"] = float(chan_energy.std().item())
    feats["latent_spatial_gradient"] = _spatial_gradient_energy(sample)
    feats["latent_laplacian_energy"] = _laplacian_energy(sample)

    # 5.4.5 AuxDecoder + UNet response (T_ref features)
    res = res1.float()
    feats["res1_norm"] = float(res.abs().mean().item())
    feats["res1_energy"] = float(res.pow(2).mean().item())
    sample_energy = float(sample.pow(2).mean().item())
    feats["sample_energy"] = sample_energy
    delta = (x0_pred_ref.float() - sample)
    delta_energy = float(delta.pow(2).mean().item())
    feats["delta_norm"] = float(delta.abs().mean().item())
    feats["delta_energy"] = delta_energy
    cos = F.cosine_similarity(sample.flatten(1), x0_pred_ref.float().flatten(1), dim=1)
    feats["cos_sample_x0"] = float(cos.mean().item())
    feats["residual_ratio"] = float(delta_energy / (sample_energy + EPS))

    return feats


# ---------------------------------------------------------------------------
# Metric evaluators (PSNR / MS-SSIM / LPIPS / DISTS) via pyiqa
# ---------------------------------------------------------------------------


@dataclass
class MetricBundle:
    """Bundle of pyiqa metric callables initialised once per device."""

    psnr: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ms_ssim: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    lpips: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dists: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_metric_bundle(device: torch.device) -> MetricBundle:
    """Construct full-reference IQA metrics on the target device."""
    import pyiqa  # imported lazily to avoid heavy import at module load

    return MetricBundle(
        psnr=pyiqa.create_metric("psnr", device=device, as_loss=False),
        ms_ssim=pyiqa.create_metric("ms_ssim", device=device, as_loss=False),
        lpips=pyiqa.create_metric("lpips", device=device, as_loss=False, pretrained_model_path=WEIGHT_PATH+'LPIPS_v0.1_alex-df73285e.pth'),
        dists=pyiqa.create_metric("dists", device=device, as_loss=False, pretrained_model_path=WEIGHT_PATH+'DISTS_weights-f5e65c96.pth'),
    )


def _to_iqa_input(x: torch.Tensor) -> torch.Tensor:
    """pyiqa expects float in [0, 1]. Inputs are [-1, 1]."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def evaluate_metrics(
    bundle: MetricBundle,
    rec: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, float]:
    """Run all four metrics and return a flat dict of scalars."""
    rec_iqa = _to_iqa_input(rec)
    gt_iqa = _to_iqa_input(gt)
    with torch.no_grad():
        psnr_v = float(bundle.psnr(rec_iqa, gt_iqa).item())
        ms_v = float(bundle.ms_ssim(rec_iqa, gt_iqa).item())
        lpips_v = float(bundle.lpips(rec_iqa, gt_iqa).item())
        dists_v = float(bundle.dists(rec_iqa, gt_iqa).item())
        mse_v = float(F.mse_loss(rec_iqa, gt_iqa).item())
    return {
        "psnr": psnr_v,
        "ms_ssim": ms_v,
        "lpips": lpips_v,
        "dists": dists_v,
        "mse": mse_v,
    }


# ---------------------------------------------------------------------------
# Oracle resolution
# ---------------------------------------------------------------------------


def compute_combined_score(
    psnr_arr: np.ndarray,
    lpips_arr: np.ndarray,
    dists_arr: np.ndarray,
    weights: Tuple[float, float, float] = DEFAULT_SCORE_WEIGHTS,
) -> np.ndarray:
    """Combined oracle score per Section 5.2 (lower is better).

    score(T) = w1 * norm(DISTS) + w2 * norm(LPIPS) - w3 * norm(PSNR)
    """
    w1, w2, w3 = weights
    return w1 * normalize01(dists_arr) + w2 * normalize01(lpips_arr) - w3 * normalize01(psnr_arr)


def resolve_oracles(
    t_candidates: Sequence[int],
    metric_rows: Sequence[Dict[str, float]],
    score_weights: Tuple[float, float, float],
    main_metric: str,
    T_snr: int,
) -> Dict[str, Any]:
    """Compute multi-oracle T values + combined score statistics."""
    psnr_arr = np.array([r["psnr"] for r in metric_rows], dtype=np.float64)
    ms_arr = np.array([r["ms_ssim"] for r in metric_rows], dtype=np.float64)
    lpips_arr = np.array([r["lpips"] for r in metric_rows], dtype=np.float64)
    dists_arr = np.array([r["dists"] for r in metric_rows], dtype=np.float64)
    t_arr = np.array(list(t_candidates), dtype=np.int64)

    score = compute_combined_score(psnr_arr, lpips_arr, dists_arr, weights=score_weights)

    T_oracle_psnr = int(t_arr[int(np.argmax(psnr_arr))])
    T_oracle_ms = int(t_arr[int(np.argmax(ms_arr))])
    T_oracle_lpips = int(t_arr[int(np.argmin(lpips_arr))])
    T_oracle_dists = int(t_arr[int(np.argmin(dists_arr))])

    main_lookup = {
        "psnr": T_oracle_psnr,
        "ms_ssim": T_oracle_ms,
        "lpips": T_oracle_lpips,
        "dists": T_oracle_dists,
        "combined": int(t_arr[int(np.argmin(score))]),
    }
    if main_metric not in main_lookup:
        raise ValueError(f"Unknown oracle metric: {main_metric}")
    T_oracle_main = main_lookup[main_metric]

    best_score = float(np.min(score))
    # T_snr may not exist exactly in the candidate grid; pick the closest.
    snr_idx = int(np.argmin(np.abs(t_arr - int(T_snr))))
    score_at_T_snr = float(score[snr_idx])
    score_gap = score_at_T_snr - best_score

    return {
        "T_oracle_psnr": T_oracle_psnr,
        "T_oracle_ms": T_oracle_ms,
        "T_oracle_lpips": T_oracle_lpips,
        "T_oracle_dists": T_oracle_dists,
        "T_oracle_main": T_oracle_main,
        "best_score": best_score,
        "score_at_T_snr": score_at_T_snr,
        "score_gap": float(score_gap),
        "score": score,
    }


# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------


SWEEP_FIELDS: Tuple[str, ...] = (
    "image_id",
    "lambda",
    "actual_bpp",
    "T",
    "PSNR",
    "MS-SSIM",
    "LPIPS",
    "DISTS",
    "MSE",
    "rate_loss",
)


SUMMARY_BASE_FIELDS: Tuple[str, ...] = (
    "image_id",
    "lambda",
    "actual_bpp",
    "T_snr",
    "T_oracle_PSNR",
    "T_oracle_MS_SSIM",
    "T_oracle_LPIPS",
    "T_oracle_DISTS",
    "T_oracle_main",
    "best_score",
    "score_at_T_snr",
    "score_gap",
)


CORR_FIELDS: Tuple[str, ...] = (
    "feature",
    "pearson_corr",
    "spearman_corr",
    "group",
    "n_samples",
)


class _RollingCsv:
    """CSV writer that flushes on every append; supports lazy header writing.

    When ``lazy=True``, fieldnames are inferred from the first row's keys
    and the header is written on the first ``write`` call. ``base_fields``
    are guaranteed to come first in the header (extra keys appended in the
    order they appear in the first row).
    """

    def __init__(
        self,
        path: str,
        fieldnames: Optional[Sequence[str]] = None,
        *,
        lazy: bool = False,
        base_fields: Optional[Sequence[str]] = None,
    ) -> None:
        self._path = path
        self._lazy = lazy
        self._base_fields: List[str] = list(base_fields) if base_fields else []
        self._fieldnames: List[str] = list(fieldnames) if fieldnames else []
        self._fh = open(path, "w", newline="")
        self._writer: Optional[csv.DictWriter] = None
        if not lazy:
            assert fieldnames is not None, "fieldnames required when lazy=False"
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._fh.flush()

    def _init_writer_from_row(self, row: Dict[str, Any]) -> None:
        keys = list(row.keys())
        ordered = list(self._base_fields)
        for k in keys:
            if k not in ordered:
                ordered.append(k)
        self._fieldnames = ordered
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._fh.flush()

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._init_writer_from_row(row)
        assert self._writer is not None
        clean = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


# ---------------------------------------------------------------------------
# Per-(image, lambda) sweep
# ---------------------------------------------------------------------------


def run_sweep_for_pair(
    *,
    net: Any,
    bundle: MetricBundle,
    cfg: ExpConfig,
    image_id: str,
    img_padded: torch.Tensor,
    ori_h: int,
    ori_w: int,
    lmbda: float,
    pos_caption_enc: torch.Tensor,
    sweep_writer: "_RollingCsv",
    summary_writer: "_RollingCsv",
    t_candidates: Sequence[int],
) -> Optional[SummaryRow]:
    """Sweep over candidate timesteps for a single (image, lambda)."""
    device = img_padded.device
    lmbda_tensor = torch.full((img_padded.shape[0],), float(lmbda), device=device)

    with torch.no_grad():
        ctx = codec_forward(net, img_padded, ori_h, ori_w, lmbda_tensor)

    rate_out = ctx["rate_out"]
    actual_bpp = float(rate_out.quantized_total_bpp.detach().mean().item())
    rate_loss_v = float(rate_out.rate_loss.detach().mean().item())
    T_snr = int(ctx["T_calc"].detach().float().mean().item())

    metric_rows: List[Dict[str, float]] = []
    x0_pred_ref: Optional[torch.Tensor] = None

    with torch.no_grad():
        for t_value in t_candidates:
            need_internals = (int(t_value) == int(cfg.t_ref))
            output, internals = decode_at_timestep(
                net, ctx, int(t_value), pos_caption_enc, return_internals=need_internals
            )
            output = output[..., :ori_h, :ori_w]
            gt = img_padded[..., :ori_h, :ori_w]
            metrics = evaluate_metrics(bundle, output, gt)
            metric_rows.append(metrics)
            if need_internals:
                x0_pred_ref = internals["x0_pred"]

            sweep_writer.write({
                "image_id": image_id,
                "lambda": float(lmbda),
                "actual_bpp": actual_bpp,
                "T": int(t_value),
                "PSNR": metrics["psnr"],
                "MS-SSIM": metrics["ms_ssim"],
                "LPIPS": metrics["lpips"],
                "DISTS": metrics["dists"],
                "MSE": metrics["mse"],
                "rate_loss": rate_loss_v,
            })

    return _finalise_summary(
        net=net, ctx=ctx, cfg=cfg, image_id=image_id, lmbda=lmbda,
        actual_bpp=actual_bpp, T_snr=T_snr, t_candidates=t_candidates,
        metric_rows=metric_rows, x0_pred_ref=x0_pred_ref,
        pos_caption_enc=pos_caption_enc, summary_writer=summary_writer,
    )


def _finalise_summary(
    *,
    net: Any,
    ctx: Dict[str, Any],
    cfg: ExpConfig,
    image_id: str,
    lmbda: float,
    actual_bpp: float,
    T_snr: int,
    t_candidates: Sequence[int],
    metric_rows: Sequence[Dict[str, float]],
    x0_pred_ref: Optional[torch.Tensor],
    pos_caption_enc: torch.Tensor,
    summary_writer: "_RollingCsv",
) -> SummaryRow:
    """Resolve oracles, build features, write summary CSV row."""
    if x0_pred_ref is None:
        with torch.no_grad():
            _, internals = decode_at_timestep(
                net, ctx, int(cfg.t_ref), pos_caption_enc, return_internals=True
            )
        x0_pred_ref = internals["x0_pred"]

    oracles = resolve_oracles(
        t_candidates=t_candidates,
        metric_rows=metric_rows,
        score_weights=cfg.score_weights,
        main_metric=cfg.oracle_metric,
        T_snr=T_snr,
    )

    sample_ref = ctx["lq_latent_hat"][:, :256].detach()
    feats = collect_timestep_features(
        lmbda=lmbda,
        actual_bpp=actual_bpp,
        target_bpp=None,
        scales_all=ctx["scales_all"],
        rate_out=ctx["rate_out"],
        lq_latent_hat=ctx["lq_latent_hat"].detach(),
        res1=ctx["res1"].detach(),
        sample_ref=sample_ref,
        x0_pred_ref=x0_pred_ref,
        T_snr=T_snr,
    )

    base_row: Dict[str, Any] = {
        "image_id": image_id,
        "lambda": float(lmbda),
        "actual_bpp": actual_bpp,
        "T_snr": int(T_snr),
        "T_oracle_PSNR": int(oracles["T_oracle_psnr"]),
        "T_oracle_MS_SSIM": int(oracles["T_oracle_ms"]),
        "T_oracle_LPIPS": int(oracles["T_oracle_lpips"]),
        "T_oracle_DISTS": int(oracles["T_oracle_dists"]),
        "T_oracle_main": int(oracles["T_oracle_main"]),
        "best_score": float(oracles["best_score"]),
        "score_at_T_snr": float(oracles["score_at_T_snr"]),
        "score_gap": float(oracles["score_gap"]),
    }
    merged = {**base_row, **feats}
    summary_writer.write(merged)

    return SummaryRow(
        image_id=image_id,
        lmbda=float(lmbda),
        actual_bpp=actual_bpp,
        T_snr=int(T_snr),
        T_oracle_psnr=int(oracles["T_oracle_psnr"]),
        T_oracle_ms_ssim=int(oracles["T_oracle_ms"]),
        T_oracle_lpips=int(oracles["T_oracle_lpips"]),
        T_oracle_dists=int(oracles["T_oracle_dists"]),
        T_oracle_main=int(oracles["T_oracle_main"]),
        best_score=float(oracles["best_score"]),
        score_at_T_snr=float(oracles["score_at_T_snr"]),
        score_gap=float(oracles["score_gap"]),
        features=feats,
    )


# ---------------------------------------------------------------------------
# Grouped correlation analysis
# ---------------------------------------------------------------------------


DEFAULT_BPP_BINS: Tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1, float("inf"))
DEFAULT_COMPLEXITY_BINS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, float("inf"))


def _bin_label(value: float, edges: Sequence[float]) -> str:
    """Return a string like 'bin[0.01,0.025)' for ``value``."""
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})"
    return f">={float(edges[-1]):g}"


def _lambda_bin(lmbda: float) -> str:
    """Coarse log-spaced lambda bins."""
    if lmbda < 1.0:
        return "lambda<1"
    if lmbda < 4.0:
        return "lambda<4"
    if lmbda < 16.0:
        return "lambda<16"
    return "lambda>=16"


def _correlation_for_group(
    feature_name: str,
    feature_values: Sequence[float],
    targets: Sequence[float],
) -> Tuple[float, float, int]:
    """Return (pearson, spearman, n_samples) for a single (feature, group) pair."""
    if len(feature_values) != len(targets):
        raise ValueError("feature_values and targets length mismatch")
    if len(feature_values) < 3:
        return float("nan"), float("nan"), len(feature_values)
    f = np.asarray(feature_values, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    mask = np.isfinite(f) & np.isfinite(t)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan"), int(mask.sum())
    f, t = f[mask], t[mask]
    return safe_corr(f, t), spearman_corr(f, t), int(f.shape[0])


def _iter_groups(rows: Sequence[SummaryRow]) -> Dict[str, List[int]]:
    """Build group_name -> list of row indices."""
    groups: Dict[str, List[int]] = {"all": list(range(len(rows)))}
    for i, r in enumerate(rows):
        groups.setdefault(_lambda_bin(float(r.lmbda)), []).append(i)
        bpp_lbl = "bpp" + _bin_label(float(r.actual_bpp), DEFAULT_BPP_BINS)
        groups.setdefault(bpp_lbl, []).append(i)
        latent_grad = float(r.features.get("latent_spatial_gradient", float("nan")))
        if math.isfinite(latent_grad):
            cmplx_lbl = "complexity" + _bin_label(latent_grad, DEFAULT_COMPLEXITY_BINS)
            groups.setdefault(cmplx_lbl, []).append(i)
    return groups


def _target_for_metric(row: SummaryRow, metric: str) -> float:
    """Pick the oracle target T for the requested metric flavor."""
    metric_l = metric.lower()
    if metric_l in ("psnr",):
        return float(row.T_oracle_psnr)
    if metric_l in ("ms_ssim", "ms-ssim", "ms"):
        return float(row.T_oracle_ms_ssim)
    if metric_l in ("lpips",):
        return float(row.T_oracle_lpips)
    if metric_l in ("dists",):
        return float(row.T_oracle_dists)
    return float(row.T_oracle_main)


def write_correlation_table(
    rows: Sequence[SummaryRow],
    out_path: str,
    *,
    main_metric: str = "lpips",
) -> None:
    """Compute per-(feature, group) Pearson/Spearman correlations vs T_oracle.

    Targets correlate every feature against ``T_oracle_main``; the row also
    carries a tag column ``oracle_metric`` for clarity.
    """
    if not rows:
        LOGGER.warning("write_correlation_table called with empty rows")
        return
    feature_names: List[str] = sorted({k for r in rows for k in r.features.keys()})
    extras: List[Tuple[str, callable]] = [
        ("__T_snr", lambda r: float(r.T_snr)),
        ("__actual_bpp", lambda r: float(r.actual_bpp)),
        ("__lambda", lambda r: float(r.lmbda)),
    ]
    groups = _iter_groups(rows)
    targets_main = [_target_for_metric(r, main_metric) for r in rows]

    writer = _RollingCsv(out_path, fieldnames=tuple(CORR_FIELDS) + ("oracle_metric",))
    try:
        for feat in feature_names + [k for k, _ in extras]:
            getter = next((g for k, g in extras if k == feat), None)
            if getter is None:
                values = [float(r.features.get(feat, float("nan"))) for r in rows]
            else:
                values = [getter(r) for r in rows]
            for grp_name, idxs in groups.items():
                if len(idxs) < 3:
                    continue
                sub_v = [values[i] for i in idxs]
                sub_t = [targets_main[i] for i in idxs]
                p, s, n = _correlation_for_group(feat, sub_v, sub_t)
                writer.write({
                    "feature": feat,
                    "pearson_corr": p,
                    "spearman_corr": s,
                    "group": grp_name,
                    "n_samples": n,
                    "oracle_metric": main_metric,
                })
    finally:
        writer.close()


def _ensure_matplotlib():
    """Lazy-import matplotlib with the Agg backend so headless runs work."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433 — intentional lazy import

    return plt


def _scatter_plot(
    xs: List[float],
    ys: List[float],
    out_path: str,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    if len(xs) == 0:
        LOGGER.warning("scatter plot skipped (empty data): %s", out_path)
        return
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=120)
    ax.scatter(xs, ys, s=10, alpha=0.55, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_oracle_plots(
    rows: Sequence[SummaryRow],
    out_dir: str,
    *,
    main_metric: str = "lpips",
) -> None:
    """Write Section 11.3 sanity-check figures into ``out_dir``."""
    if len(rows) == 0:
        LOGGER.warning("no summary rows; skipping plots")
        return
    os.makedirs(out_dir, exist_ok=True)
    targets = [_target_for_metric(r, main_metric) for r in rows]

    _scatter_plot(
        [float(r.actual_bpp) for r in rows],
        targets,
        os.path.join(out_dir, "T_oracle_vs_bpp.png"),
        xlabel="actual bpp",
        ylabel=f"T_oracle ({main_metric})",
        title="Oracle timestep vs actual bpp",
    )
    _scatter_plot(
        [float(r.T_snr) for r in rows],
        targets,
        os.path.join(out_dir, "T_oracle_vs_T_snr.png"),
        xlabel="T_snr (handcrafted)",
        ylabel=f"T_oracle ({main_metric})",
        title="Oracle timestep vs handcrafted T_snr",
    )
    _plot_oracle_distribution_by_lambda(rows, out_dir, main_metric=main_metric)


def _plot_oracle_distribution_by_lambda(
    rows: Sequence[SummaryRow],
    out_dir: str,
    *,
    main_metric: str = "lpips",
) -> None:
    plt = _ensure_matplotlib()
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        label = _lambda_bin(float(row.lmbda))
        grouped.setdefault(label, []).append(_target_for_metric(row, main_metric))
    if not grouped:
        LOGGER.warning("no lambda bins available; skipping distribution plot")
        return
    labels = sorted(grouped.keys())
    data = [grouped[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=120)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel("lambda bin")
    ax.set_ylabel(f"T_oracle ({main_metric})")
    ax.set_title("Oracle timestep distribution by lambda bin")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "T_oracle_distribution_by_lambda.png")
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image iteration + model loading
# ---------------------------------------------------------------------------


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def iter_validation_images(
    base_config: Dict[str, object],
    max_images: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    """Yield (image_id, image_path) for test images, sorted by name."""

    root = base_config.get("test_dataset")
    if not root:
        raise ValueError("base_config['test_dataset'] is required")
    test_dir = Path(str(root))
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    files = sorted(
        p
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if max_images is not None and max_images > 0:
        files = files[:max_images]

    for path in files:
        yield path.stem, str(path)


def _load_partial_state_dict(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    name: str,
) -> None:
    """Load checkpoint with shape-safe filtering (skip mismatched keys)."""
    own = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = torch.nn.Module.load_state_dict(module, filtered, strict=False) or ([], [])
    if skipped:
        LOGGER.info("[%s] skipped %d shape-mismatched keys", name, len(skipped))
    if missing:
        LOGGER.info("[%s] missing keys: %d", name, len(missing))
    if unexpected:
        LOGGER.info("[%s] unexpected keys: %d", name, len(unexpected))


def load_model(
    cfg: "ExpConfig",
    train_config: Dict[str, Any],
    logger: logging.Logger,
) -> Any:
    """Replicate the model-construction path from trainv1_variable3_debug_step.py."""

    from StableCodec_variable2_step import StableCodec  # noqa: WPS433

    model_cfg = train_config["model"]
    sd_path = model_cfg["sd_path"]

    print(f"  [load_model] Constructing StableCodec (sd_path={sd_path})...", flush=True)
    net = StableCodec(sd_path=sd_path, config=model_cfg, logger=logger)
    print(f"  [load_model] Moving to device={cfg.device}...", flush=True)
    net = net.to(cfg.device)
    net.eval()
    print(f"  [load_model] Model on device.", flush=True)

    if train_config.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            LOGGER.warning("xformers requested but not available")

    ckpt_path = cfg.codec_path or model_cfg.get("codec_path")
    if not ckpt_path:
        raise ValueError("codec_path must be provided via --codec_path or test config")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print(f"  [load_model] Loading checkpoint: {ckpt_path}...", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    print(f"  [load_model] Checkpoint loaded, keys: {list(sd.keys())}", flush=True)

    if "state_dict_codec" in sd:
        _load_partial_state_dict(net.codec, sd["state_dict_codec"], "codec")
    if "state_dict_unet" in sd:
        _load_partial_state_dict(net.unet, sd["state_dict_unet"], "unet")
    if "state_dict_vae" in sd:
        _load_partial_state_dict(net.vae, sd["state_dict_vae"], "vae")
    print(f"  [load_model] State dicts loaded.", flush=True)

    if train_config.get("save_ema", False) and "ema_state_dict" in sd:
        try:
            ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
            ema_net.load_state_dict(sd["ema_state_dict"])
            ema_net.copy_to(net.parameters())
            LOGGER.info("Applied EMA weights to model")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("EMA load failed: %s", exc)

    # pos_caption_enc is already set during StableCodec.__init__ (tokenizer
    # and text_encoder are deleted after that). No need to call set_prompt again.
    if not hasattr(net, "pos_caption_enc") or net.pos_caption_enc is None:
        pos_prompt = train_config.get("model", {}).get("pos_prompt", "")
        if hasattr(net, "tokenizer"):
            net.set_prompt(pos_prompt)
        else:
            raise RuntimeError("pos_caption_enc not set and tokenizer already deleted")

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"  [load_model] Done.", flush=True)
    return net


def run_experiment(cfg: ExpConfig) -> None:
    """Orchestrate Section 5 (Path A) Oracle-T sweep, summary, and correlation analysis."""
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("Loading configs...", flush=True)
    base_config = load_yaml(cfg.base_config)
    test_config = load_yaml(cfg.test_config)
    train_config = merge_dicts(base_config, test_config)

    print(f"  base_config: {cfg.base_config}", flush=True)
    print(f"  test_config: {cfg.test_config}", flush=True)

    print("Loading model (this may take a few minutes)...", flush=True)
    net = load_model(cfg, train_config, LOGGER)
    print("Model loaded.", flush=True)

    device = torch.device(cfg.device)
    print("Building metrics...", flush=True)
    bundle = build_metric_bundle(device)
    print("Metrics ready.", flush=True)

    sweep_csv = os.path.join(cfg.out_dir, "oracle_timestep_sweep.csv")
    summary_csv = os.path.join(cfg.out_dir, "oracle_timestep_summary.csv")

    sweep_writer = _RollingCsv(sweep_csv, fieldnames=list(SWEEP_FIELDS))
    summary_writer = _RollingCsv(
        summary_csv,
        lazy=True,
        base_fields=list(SUMMARY_BASE_FIELDS),
    )

    t_candidates = _build_t_candidates(cfg.t_min, cfg.t_max, cfg.t_step)
    LOGGER.info("Sweeping over %d candidate timesteps: %s", len(t_candidates), t_candidates)

    if cfg.num_lambdas <= 1:
        lambda_grid = np.array([cfg.lambda_min], dtype=np.float64)
    else:
        lambda_grid = np.geomspace(cfg.lambda_min, cfg.lambda_max, cfg.num_lambdas)
    LOGGER.info("Lambda grid (%d points): %s", lambda_grid.size, lambda_grid.tolist())

    pos_caption_enc = getattr(net, "pos_caption_enc", None)
    if pos_caption_enc is None:
        raise RuntimeError("net.pos_caption_enc is not initialised; check load_model/set_prompt path")

    summary_rows: List[SummaryRow] = []

    image_iter = iter_validation_images(base_config, max_images=cfg.max_images)
    image_records = list(image_iter)
    if not image_records:
        raise RuntimeError("No validation images found; check base_config['train_dataset']/valid")
    LOGGER.info("Total validation images to process: %d", len(image_records))

    for img_idx, (image_id, img_path) in enumerate(image_records):
        try:
            img_padded, ori_h, ori_w = preprocess_image(img_path, device)
        except Exception as exc:  # pragma: no cover - I/O failure path
            LOGGER.warning("Skipping %s due to preprocessing error: %s", img_path, exc)
            continue
        print(
            f"[{img_idx + 1}/{len(image_records)}] {image_id} "
            f"({ori_h}x{ori_w}) x {len(lambda_grid)} lambdas x {len(t_candidates)} T ...",
            flush=True,
        )

        for lmbda in lambda_grid:
            lmbda_value = float(lmbda)
            try:
                summary_row = run_sweep_for_pair(
                    net=net,
                    bundle=bundle,
                    cfg=cfg,
                    image_id=image_id,
                    img_padded=img_padded,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    lmbda=lmbda_value,
                    pos_caption_enc=pos_caption_enc,
                    sweep_writer=sweep_writer,
                    summary_writer=summary_writer,
                    t_candidates=t_candidates,
                )
                print(f"    lambda={lmbda_value:.3f} done (bpp={summary_row.actual_bpp:.4f}, "
                      f"T_oracle={summary_row.T_oracle_main})", flush=True)
            except Exception as exc:  # pragma: no cover - safeguard per-pair failure
                print(f"    lambda={lmbda_value:.3f} FAILED: {exc}", flush=True)
                LOGGER.exception(
                    "Sweep failed for image_id=%s lambda=%.4f: %s", image_id, lmbda_value, exc
                )
                continue

            if summary_row is not None:
                summary_rows.append(summary_row)

    sweep_writer.close()
    summary_writer.close()

    if not summary_rows:
        LOGGER.warning("No summary rows produced; skipping correlation table and plots")
        return

    corr_csv = os.path.join(cfg.out_dir, "correlation_table.csv")
    write_correlation_table(summary_rows, corr_csv, main_metric=cfg.oracle_metric)
    LOGGER.info("Wrote correlation table to %s", corr_csv)

    if cfg.make_plots:
        try:
            write_oracle_plots(summary_rows, cfg.out_dir)
            LOGGER.info("Wrote oracle plots to %s", cfg.out_dir)
        except Exception as exc:  # pragma: no cover - plotting is optional
            LOGGER.warning("Plot generation failed: %s", exc)


def _parse_args() -> ExpConfig:
    """Parse CLI arguments into a frozen :class:`ExpConfig`."""
    parser = argparse.ArgumentParser(
        description="Oracle-T sweep & correlation analysis (Section 5, Path A).",
    )
    parser.add_argument("--base_config", type=str, required=True,
                        help="Path to base YAML config (provides dataset paths).")
    parser.add_argument("--test_config", type=str, required=True,
                        help="Path to test/stage YAML config (provides model/test settings).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for CSV and plot artifacts.")
    parser.add_argument("--codec_path", type=str, default=None,
                        help="Optional checkpoint path; overrides train_config.model.codec_path.")
    parser.add_argument("--num_lambdas", type=int, default=6,
                        help="Number of lambda points to sweep (geomspace).")
    parser.add_argument("--lambda_min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda_max", type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument("--t_min", type=int, default=DEFAULT_T_MIN)
    parser.add_argument("--t_max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--t_step", type=int, default=DEFAULT_T_STEP)
    parser.add_argument("--t_ref", type=int, default=DEFAULT_T_MAX,
                        help="Reference timestep used for UNet response features.")
    parser.add_argument("--oracle_metric", type=str, default="lpips",
                        choices=("psnr", "ms_ssim", "lpips", "dists", "combined"),
                        help="Metric used to define T_oracle_main.")
    parser.add_argument("--score_weights", type=float, nargs=3,
                        default=list(DEFAULT_SCORE_WEIGHTS),
                        metavar=("W_DISTS", "W_LPIPS", "W_PSNR"),
                        help="Combined-score weights (DISTS, LPIPS, -PSNR).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Cap on number of validation images (debug).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_plots", action="store_true",
                        help="If set, write oracle scatter plots alongside CSVs.")

    args = parser.parse_args()
    return ExpConfig(
        base_config=args.base_config,
        test_config=args.test_config,
        out_dir=args.out_dir,
        codec_path=args.codec_path,
        num_lambdas=args.num_lambdas,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        t_ref=args.t_ref,
        oracle_metric=args.oracle_metric,
        score_weights=tuple(args.score_weights),
        max_images=args.max_images,
        device=args.device,
        seed=args.seed,
        make_plots=args.make_plots,
    )


def main() -> None:
    """Entry point: configure logging, parse args, run experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = _parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
