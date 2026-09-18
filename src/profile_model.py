"""
StableCodec Complexity Profiler
===============================
Standalone script to measure parameter count, FLOPs, and inference time
for each sub-module: VAE Encoder, VAE Decoder, UNet, Latent Codec, Aux Encoder,
Scheduler Step, and the full forward pass.

Usage:
    # Use stage0.yaml (latent_channels=256)
    python profile_model.py --config ../configs/stage0.yaml

    # Use stage1.yaml (latent_channels=4) for comparison
    python profile_model.py --config ../configs/stage1.yaml

    # Custom input resolution
    python profile_model.py --config ../configs/stage0.yaml --resolution 512

    # Side-by-side comparison of two configs
    python profile_model.py --config ../configs/stage1.yaml ../configs/stage0.yaml
"""

import argparse
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# 1. Parameter counting (zero-dependency)
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> dict:
    """Return total and trainable parameter counts for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# 2. FLOPs estimation via torch hooks (zero-dependency)
# ---------------------------------------------------------------------------

def estimate_flops(module: nn.Module, *args, **kwargs) -> int:
    """Estimate FLOPs by hooking Conv2d and Linear layers.

    Uses a local accumulator (no global state) and try/finally to guarantee
    hooks are removed even if the forward pass raises.
    """
    count = [0]  # mutable container avoids global state

    def _conv_hook(mod, inp, out):
        if out.ndim != 4:
            return  # skip non-spatial convolutions (e.g. 1D wrapped)
        batch = out.shape[0]
        out_channels, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
        kernel_h, kernel_w = mod.kernel_size
        in_channels_per_group = mod.in_channels // mod.groups
        macs = batch * out_channels * out_h * out_w * in_channels_per_group * kernel_h * kernel_w
        if mod.bias is not None:
            macs += batch * out_channels * out_h * out_w
        count[0] += 2 * macs

    def _linear_hook(mod, inp, out):
        batch_dims = inp[0].shape[:-1].numel()
        macs = batch_dims * mod.in_features * mod.out_features
        if mod.bias is not None:
            macs += batch_dims * mod.out_features
        count[0] += 2 * macs

    hooks = []
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
    try:
        with torch.no_grad():
            module(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    return count[0]


# ---------------------------------------------------------------------------
# 3. GPU-accurate timing
# ---------------------------------------------------------------------------

def measure_inference_time(
    fn,
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> dict:
    """Measure GPU inference time with proper warmup and CUDA synchronization.

    Args:
        fn: callable that performs the forward pass (no return value needed).
        warmup_runs: number of warmup iterations.
        timed_runs: number of timed iterations.

    Returns:
        dict with avg_ms, min_ms, max_ms.
    """
    # Warmup
    for _ in range(warmup_runs):
        fn()
    torch.cuda.synchronize()

    # Timed runs using CUDA events
    times = []
    for _ in range(timed_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


# ---------------------------------------------------------------------------
# 4. Formatting helpers
# ---------------------------------------------------------------------------

def format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def format_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MFLOPs"
    return f"{n} FLOPs"


def print_report(title: str, results: OrderedDict):
    """Pretty-print the profiling report."""
    sep = "=" * 88
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    header = f"{'Module':<20} {'Total Params':>14} {'Trainable':>14} {'FLOPs':>18} {'Avg Time':>10} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * 88)
    for name, r in results.items():
        params_str = format_params(r["params"]["total"])
        train_str = format_params(r["params"]["trainable"])
        flops_str = format_flops(r["flops"]) if r["flops"] is not None else "N/A"
        avg_str = f"{r['time']['avg_ms']:.2f}ms" if r["time"] is not None else "N/A"
        min_str = f"{r['time']['min_ms']:.2f}" if r["time"] is not None else ""
        max_str = f"{r['time']['max_ms']:.2f}" if r["time"] is not None else ""
        print(f"{name:<20} {params_str:>14} {train_str:>14} {flops_str:>18} {avg_str:>10} {min_str:>8} {max_str:>8}")
    print(sep)


# ---------------------------------------------------------------------------
# 5. Build model from config (mirrors train.py logic)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and merge base + stage config."""
    import os
    with open(config_path, "r") as f:
        stage_config = yaml.safe_load(f)

    # Try to load base.yaml from same directory
    config_dir = os.path.dirname(config_path)
    base_path = os.path.join(config_dir, "base.yaml")
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        # Stage config overrides base
        base_config.update(stage_config)
        # Also merge nested 'model' dict
        if "model" in stage_config and "model" in base_config:
            base_model = base_config.get("model", {})
            stage_model = stage_config.get("model", {})
            merged_model = {**base_model, **stage_model}
            base_config["model"] = merged_model
        return base_config

    return stage_config


def build_model(config_path: str):
    """Build StableCodec from a config file."""
    config = load_config(config_path)
    model_config = config["model"]

    from StableCodec_ori import StableCodec
    model = StableCodec(
        sd_path=model_config["sd_path"],
        lmbda=config.get("lmbda", model_config.get("lambda_ref", 0.5)),
        config=model_config,
    )
    model.set_eval()
    model.cuda()
    return model, model_config


# ---------------------------------------------------------------------------
# 6. Profile a single model
# ---------------------------------------------------------------------------

@torch.no_grad()
def profile_single(
    model,
    model_config: dict,
    resolution: int = 256,
    warmup: int = 10,
    runs: int = 50,
) -> OrderedDict:
    """Profile all sub-modules and the full forward pass."""

    results = OrderedDict()
    device = "cuda"
    latent_channels = model_config.get("latent_channels", 4)

    # Dummy inputs
    dummy_img = torch.randn(1, 3, resolution, resolution, device=device)
    pos_prompt = ["a photo"]

    # Prepare caption encoding (already stored in model)
    pos_caption_enc = model.pos_caption_enc.to(device)

    # ---- Run full forward once to get intermediate shapes ----
    with torch.no_grad():
        latent2 = model.aux_codec((dummy_img + 1) / 2).detach()
    lq_latent = model.vae.encode(dummy_img).latent_dist.mode() * model.vae.config.scaling_factor
    lq_latent_hat, _, res1 = model.codec(lq_latent, latent2, resolution, resolution)

    # ==========================================================
    # (a) VAE Encoder
    # ==========================================================
    params_enc = count_parameters(model.vae.encoder)
    flops_enc = estimate_flops(model.vae.encoder, dummy_img)
    time_enc = measure_inference_time(
        lambda: model.vae.encode(dummy_img),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    results["VAE Encoder"] = {"params": params_enc, "flops": flops_enc, "time": time_enc}

    # ==========================================================
    # (b) Aux Encoder (ELIC g_a)
    # ==========================================================
    params_aux = count_parameters(model.aux_codec)
    flops_aux = estimate_flops(model.aux_codec, (dummy_img + 1) / 2)
    time_aux = measure_inference_time(
        lambda: model.aux_codec((dummy_img + 1) / 2),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    results["Aux Encoder"] = {"params": params_aux, "flops": flops_aux, "time": time_aux}

    # ==========================================================
    # (c) Latent Codec
    # ==========================================================
    params_codec = count_parameters(model.codec)
    flops_codec = estimate_flops(model.codec, lq_latent, latent2, resolution, resolution)
    time_codec = measure_inference_time(
        lambda: model.codec(lq_latent, latent2, resolution, resolution),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    results["Latent Codec"] = {"params": params_codec, "flops": flops_codec, "time": time_codec}

    # ==========================================================
    # (d) UNet
    # ==========================================================
    params_unet = count_parameters(model.unet)
    flops_unet = estimate_flops(
        model.unet, lq_latent_hat, model.timesteps,
        encoder_hidden_states=pos_caption_enc.expand(1, -1, -1),
    )
    time_unet = measure_inference_time(
        lambda: model.unet(
            lq_latent_hat, model.timesteps,
            encoder_hidden_states=pos_caption_enc.expand(1, -1, -1),
        ),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    results["UNet"] = {"params": params_unet, "flops": flops_unet, "time": time_unet}

    # ==========================================================
    # (e) Scheduler Step (element-wise, no learnable params)
    # ==========================================================
    model_pred = model.unet(
        lq_latent_hat, model.timesteps,
        encoder_hidden_states=pos_caption_enc.expand(1, -1, -1),
    ).sample
    sched_input = lq_latent_hat[:, :latent_channels]

    time_sched = measure_inference_time(
        lambda: model.sched.step(
            model_pred, model.timesteps, sched_input, return_dict=True,
        ),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    # Count elements to show the scale
    sched_elements = sched_input.numel()
    results["Sched Step"] = {
        "params": {"total": 0, "trainable": 0},
        "flops": sched_elements * 10,  # ~10 element-wise ops in DDPM step
        "time": time_sched,
    }

    # ==========================================================
    # (f) VAE Decoder
    # ==========================================================
    x_denoised = model.sched.step(
        model_pred, model.timesteps, sched_input, return_dict=True,
    ).prev_sample + res1

    params_dec = count_parameters(model.vae.decoder)
    dec_input = x_denoised / model.vae.config.scaling_factor
    flops_dec = estimate_flops(model.vae.decoder, dec_input)
    time_dec = measure_inference_time(
        lambda: model.vae.decode(dec_input),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    results["VAE Decoder"] = {"params": params_dec, "flops": flops_dec, "time": time_dec}

    # ==========================================================
    # (g) Full Forward
    # ==========================================================
    params_full = count_parameters(model)
    time_full = measure_inference_time(
        lambda: model(dummy_img, pos_prompt, resolution, resolution),
        warmup_runs=warmup,
        timed_runs=runs,
    )
    # Sum of sub-module FLOPs as approximation
    flops_full = sum(
        r["flops"] for r in results.values() if r["flops"] is not None
    )
    results["Full Forward"] = {"params": params_full, "flops": flops_full, "time": time_full}

    return results


# ---------------------------------------------------------------------------
# 7. Comparison mode
# ---------------------------------------------------------------------------

def print_comparison(title_a, results_a, title_b, results_b):
    """Print side-by-side comparison of two profiling runs."""
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  COMPARISON: [{title_a}] vs [{title_b}]")
    print(sep)
    header = (
        f"{'Module':<18} "
        f"{'Params(A)':>12} {'Params(B)':>12} {'Ratio':>8} "
        f"{'FLOPs(A)':>14} {'FLOPs(B)':>14} {'Ratio':>8} "
        f"{'Time(A)':>9} {'Time(B)':>9} {'Ratio':>8}"
    )
    print(header)
    print("-" * 100)

    all_keys = list(results_a.keys()) + [k for k in results_b if k not in results_a]
    for name in all_keys:
        ra = results_a.get(name)
        rb = results_b.get(name)
        if ra is None or rb is None:
            continue

        p_a = ra["params"]["total"]
        p_b = rb["params"]["total"]
        p_ratio = f"{p_b / p_a:.2f}x" if p_a > 0 else "N/A"

        f_a = ra["flops"] if ra["flops"] else 0
        f_b = rb["flops"] if rb["flops"] else 0
        f_ratio = f"{f_b / f_a:.2f}x" if f_a > 0 else "N/A"

        t_a = ra["time"]["avg_ms"] if ra["time"] else 0
        t_b = rb["time"]["avg_ms"] if rb["time"] else 0
        t_ratio = f"{t_b / t_a:.2f}x" if t_a > 0 else "N/A"

        print(
            f"{name:<18} "
            f"{format_params(p_a):>12} {format_params(p_b):>12} {p_ratio:>8} "
            f"{format_flops(f_a):>14} {format_flops(f_b):>14} {f_ratio:>8} "
            f"{t_a:>8.2f}ms {t_b:>8.2f}ms {t_ratio:>8}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# 8. GPU Memory summary
# ---------------------------------------------------------------------------

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[GPU Memory] Allocated: {allocated:.2f}GB | "
              f"Reserved: {reserved:.2f}GB | "
              f"Peak: {max_allocated:.2f}GB")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StableCodec Complexity Profiler")
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="Path to config YAML(s). Pass two configs for side-by-side comparison.",
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Input image resolution (default: 256).",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup runs (default: 10).",
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of timed runs (default: 50).",
    )
    args = parser.parse_args()

    if len(args.config) == 1:
        # ---------- Single config mode ----------
        print(f"[Profiler] Loading model from: {args.config[0]}")
        print(f"[Profiler] Resolution: {args.resolution}x{args.resolution}")
        print(f"[Profiler] Warmup: {args.warmup}, Runs: {args.runs}")

        model, model_config = build_model(args.config[0])
        lc = model_config.get("latent_channels", 4)
        title = f"{args.config[0]} (latent_channels={lc})"

        results = profile_single(
            model, model_config,
            resolution=args.resolution,
            warmup=args.warmup,
            runs=args.runs,
        )
        print_report(title, results)
        print_gpu_memory()

    elif len(args.config) == 2:
        # ---------- Comparison mode ----------
        print(f"[Profiler] Comparison mode:")
        print(f"  Config A: {args.config[0]}")
        print(f"  Config B: {args.config[1]}")
        print(f"  Resolution: {args.resolution}x{args.resolution}")

        # Profile A
        print(f"\n{'='*40} Profiling Config A {'='*40}")
        model_a, config_a = build_model(args.config[0])
        lc_a = config_a.get("latent_channels", 4)
        title_a = f"{args.config[0]} (ch={lc_a})"
        results_a = profile_single(
            model_a, config_a,
            resolution=args.resolution,
            warmup=args.warmup,
            runs=args.runs,
        )
        print_report(title_a, results_a)

        # Free memory
        del model_a
        torch.cuda.empty_cache()

        # Profile B
        print(f"\n{'='*40} Profiling Config B {'='*40}")
        model_b, config_b = build_model(args.config[1])
        lc_b = config_b.get("latent_channels", 4)
        title_b = f"{args.config[1]} (ch={lc_b})"
        results_b = profile_single(
            model_b, config_b,
            resolution=args.resolution,
            warmup=args.warmup,
            runs=args.runs,
        )
        print_report(title_b, results_b)

        # Side-by-side comparison
        print_comparison(title_a, results_a, title_b, results_b)
        print_gpu_memory()

    else:
        print("Error: pass 1 or 2 config files.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
