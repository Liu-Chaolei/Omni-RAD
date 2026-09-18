"""Offline calibration: fit an isotonic regression T_calc → T_opt.

Consumes the per-image CSV produced by ``experiment_timesteps.py`` and
writes a checkpoint that ``DynamicTimestepModuleV2`` can load via the
``calibration_path`` argument.

The calibration is monotonic-only (no slope changes), which is the right
inductive bias because we already know Pearson(0.45) ≈ Spearman(0.48):
the relationship is monotonic but not linear, so a monotonic recalibration
is the cheapest correction we can apply.

Usage:
    python src/calibrate_timestep.py \
        --per_image_csv  results/experiment_timesteps/per_image.csv \
        --out_path       checkpoints/T_calibration.pt \
        --t_min          800 \
        --t_max          999

The resulting ``.pt`` file contains two 1-D tensors ``xs`` and ``ys`` such
that ``ys[searchsorted(xs, T_calc)] ≈ T_opt``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators algorithm (weighted, non-decreasing fit)."""
    n = len(y)
    vals = y.astype(np.float64).copy()
    weights = w.astype(np.float64).copy()
    sizes = np.ones(n, dtype=np.int64)
    idx = 0
    stack_v: list[float] = []
    stack_w: list[float] = []
    stack_s: list[int] = []
    for i in range(n):
        v, ww, sz = vals[i], weights[i], sizes[i]
        while stack_v and stack_v[-1] >= v:
            pv = stack_v.pop()
            pw = stack_w.pop()
            ps = stack_s.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww = pw + ww
            sz = ps + sz
        stack_v.append(v)
        stack_w.append(ww)
        stack_s.append(sz)
    out = np.empty(n, dtype=np.float64)
    j = 0
    for v, sz in zip(stack_v, stack_s):
        out[j:j + sz] = v
        j += sz
    return out


def fit_isotonic(
    t_calc: np.ndarray,
    t_opt: np.ndarray,
    t_min: int,
    t_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an isotonic (monotone non-decreasing) regression on T_calc → T_opt
    and sample it on the integer T grid ``[t_min, t_max]``.

    Tries ``sklearn.isotonic.IsotonicRegression`` first; falls back to a
    pure-numpy PAV implementation when scikit-learn is unavailable.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        iso = IsotonicRegression(
            y_min=float(t_min),
            y_max=float(t_max),
            out_of_bounds="clip",
        )
        iso.fit(t_calc, t_opt)
        xs = np.arange(t_min, t_max + 1, dtype=np.float64)
        ys = iso.predict(xs)
        return xs, ys
    except ImportError:
        pass

    # Pure-numpy fallback: average ties, run PAV, then linearly interpolate.
    order = np.argsort(t_calc, kind="stable")
    xs_sorted = t_calc[order]
    ys_sorted = t_opt[order]

    uniq_x, inverse = np.unique(xs_sorted, return_inverse=True)
    sums = np.zeros_like(uniq_x, dtype=np.float64)
    counts = np.zeros_like(uniq_x, dtype=np.float64)
    np.add.at(sums, inverse, ys_sorted)
    np.add.at(counts, inverse, 1.0)
    means = sums / counts

    fitted = _pav(means, counts)
    fitted = np.clip(fitted, float(t_min), float(t_max))

    xs = np.arange(t_min, t_max + 1, dtype=np.float64)
    ys = np.interp(xs, uniq_x, fitted, left=fitted[0], right=fitted[-1])
    return xs, ys


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.per_image_csv)
    if "T_calc" not in df.columns or "T_opt" not in df.columns:
        raise ValueError(
            f"{args.per_image_csv} must contain T_calc and T_opt columns"
        )

    t_calc = df["T_calc"].to_numpy(dtype=np.float64)
    t_opt = df["T_opt"].to_numpy(dtype=np.float64)

    print(f"Fitting isotonic regression on {len(df)} samples")
    print(
        f"  T_calc range: [{t_calc.min():.1f}, {t_calc.max():.1f}]  "
        f"mean={t_calc.mean():.1f}"
    )
    print(
        f"  T_opt  range: [{t_opt.min():.1f}, {t_opt.max():.1f}]  "
        f"mean={t_opt.mean():.1f}"
    )

    xs, ys = fit_isotonic(t_calc, t_opt, args.t_min, args.t_max)

    # Diagnostics: residual after calibration on the training data.
    pred_opt = np.interp(t_calc, xs, ys)
    pearson_before = float(np.corrcoef(t_calc, t_opt)[0, 1])
    pearson_after = float(np.corrcoef(pred_opt, t_opt)[0, 1])
    mae_before = float(np.mean(np.abs(t_calc - t_opt)))
    mae_after = float(np.mean(np.abs(pred_opt - t_opt)))

    print("\nCalibration diagnostics (on the same data, not held-out)")
    print(f"  Pearson(T_calc, T_opt)             = {pearson_before:+.4f}")
    print(f"  Pearson(calibrated T_calc, T_opt)  = {pearson_after:+.4f}")
    print(f"  mean |T_calc - T_opt|              = {mae_before:.3f}")
    print(f"  mean |calibrated T_calc - T_opt|   = {mae_after:.3f}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(
        {
            "xs": torch.from_numpy(xs).float(),
            "ys": torch.from_numpy(ys).float(),
            "metadata": {
                "n_samples": int(len(df)),
                "pearson_before": pearson_before,
                "pearson_after": pearson_after,
                "mae_before": mae_before,
                "mae_after": mae_after,
                "t_min": int(args.t_min),
                "t_max": int(args.t_max),
                "source_csv": os.path.abspath(args.per_image_csv),
            },
        },
        args.out_path,
    )
    print(f"\nWrote calibration → {args.out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit isotonic T_calc → T_opt calibration for "
        "DynamicTimestepModuleV2."
    )
    parser.add_argument("--per_image_csv", type=str, required=True)
    parser.add_argument(
        "--out_path",
        type=str,
        default="checkpoints/T_calibration.pt",
    )
    parser.add_argument("--t_min", type=int, default=800)
    parser.add_argument("--t_max", type=int, default=999)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
