"""Profile FLOPs and parameter counts for StableCodec with latent_channels=4 vs 256.

Usage:
    cd /home/liuchaolei/Image-Coder/Diffusion/sc-mysc
    CUDA_VISIBLE_DEVICES=0 python src/experiment_profile_flops.py --latent_channels 4
    CUDA_VISIBLE_DEVICES=0 python src/experiment_profile_flops.py --latent_channels 256
"""

import argparse
import sys
import torch
import yaml
from fvcore.nn import FlopCountAnalysis


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def fmt(n):
    if n >= 1e12:
        return f"{n / 1e12:.2f} T"
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return str(int(n))


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def safe_flops(module, inputs, name=""):
    try:
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        fa = FlopCountAnalysis(module, inputs)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total()
    except Exception as e:
        print(f"  [Warning] FLOPs count failed for {name}: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_channels", type=int, required=True, choices=[4, 256])
    parser.add_argument("--input_size", type=int, default=512)
    args = parser.parse_args()

    H = W = args.input_size
    C = args.latent_channels
    B = 1

    config = load_yaml("./configs/stage2.yaml")
    model_cfg = config["model"]
    model_cfg["latent_channels"] = C
    model_cfg["codec_path"] = None

    if C == 4 and "sd-turbo_256" in model_cfg["sd_path"]:
        model_cfg["sd_path"] = model_cfg["sd_path"].replace("sd-turbo_256", "sd-turbo")
    elif C == 256 and "sd-turbo_256" not in model_cfg["sd_path"]:
        model_cfg["sd_path"] = model_cfg["sd_path"].replace("sd-turbo", "sd-turbo_256")

    print(f"\n{'=' * 70}")
    print(f"  Profiling StableCodec  latent_channels={C}  input={H}x{W}")
    print(f"  sd_path: {model_cfg['sd_path']}")
    print(f"{'=' * 70}\n")

    from StableCodec_ori import StableCodec

    net = StableCodec(
        sd_path=model_cfg["sd_path"],
        lmbda=config.get("lmbda", 2.0),
        config=model_cfg,
        stage=1,
    )
    net.cuda().eval()

    vae_h, vae_w = H // 8, W // 8
    elic_h, elic_w = H // 16, W // 16
    y_h, y_w = elic_h // 4, elic_w // 4
    z_h, z_w = y_h // 4, y_w // 4

    print("Intermediate shapes (for 512x512 input):")
    print(f"  VAE latent : (1, {C}, {vae_h}, {vae_w})")
    print(f"  ELIC aux   : (1, 320, {elic_h}, {elic_w})")
    print(f"  y (codec)  : (1, 320, {y_h}, {y_w})")
    print(f"  z (hyper)  : (1, 160, {z_h}, {z_w})")
    print()

    results = []

    with torch.no_grad():
        dummy_img = torch.randn(B, 3, H, W, device="cuda")

        # --- VAE Encoder ---
        vae_enc = net.vae.encoder
        vae_enc_p = count_params(vae_enc)
        vae_enc_f = safe_flops(vae_enc, dummy_img, "VAE Encoder")

        qconv_p = count_params(net.vae.quant_conv)
        enc_out = vae_enc(dummy_img)
        qconv_f = safe_flops(net.vae.quant_conv, enc_out, "quant_conv")

        results.append(("VAE Encoder+quant_conv",
                         vae_enc_p[0] + qconv_p[0],
                         vae_enc_p[1] + qconv_p[1],
                         (vae_enc_f if vae_enc_f >= 0 else 0) + (qconv_f if qconv_f >= 0 else 0)))

        # --- ELIC g_a (frozen) ---
        dummy_01 = torch.randn(B, 3, H, W, device="cuda")
        elic_p = count_params(net.aux_codec)
        elic_f = safe_flops(net.aux_codec, dummy_01, "ELIC g_a")
        results.append(("ELIC g_a (frozen)", elic_p[0], elic_p[1], elic_f))

        # --- LatentCodec.g_a ---
        dummy_lat = torch.randn(B, C, vae_h, vae_w, device="cuda")
        dummy_aux = torch.randn(B, 320, elic_h, elic_w, device="cuda")
        ga_p = count_params(net.codec.g_a)
        ga_f = safe_flops(net.codec.g_a, (dummy_lat, dummy_aux), "Codec g_a")
        results.append(("Codec g_a (AnalysisTransform)", ga_p[0], ga_p[1], ga_f))

        # --- LatentCodec.h_a ---
        dummy_y = torch.randn(B, 320, y_h, y_w, device="cuda")
        ha_p = count_params(net.codec.h_a)
        ha_f = safe_flops(net.codec.h_a, dummy_y, "Codec h_a")
        results.append(("Codec h_a (HyperAnalysis)", ha_p[0], ha_p[1], ha_f))

        # --- LatentCodec.h_s ---
        dummy_z = torch.randn(B, 160, z_h, z_w, device="cuda")
        hs_p = count_params(net.codec.h_s)
        hs_f = safe_flops(net.codec.h_s, dummy_z, "Codec h_s")
        results.append(("Codec h_s (HyperSynthesis)", hs_p[0], hs_p[1], hs_f))

        # --- Context model (4 passes: adapter_in -> g_c -> adapter_out + LRP) ---
        ctx_params = 0
        ctx_flops = 0
        dummy_base = torch.randn(B, 320, y_h, y_w, device="cuda")
        for i in range(4):
            ai_p = count_params(net.codec.adapter_in[i])[0]
            ao_p = count_params(net.codec.adapter_out[i])[0]
            lrp_p = count_params(net.codec.LRP[i])[0]
            ctx_params += ai_p + ao_p + lrp_p

            ai_f = safe_flops(net.codec.adapter_in[i], dummy_base, f"adapter_in[{i}]")
            ai_out = net.codec.adapter_in[i](dummy_base)
            gc_f = safe_flops(net.codec.g_c, ai_out, f"g_c pass {i}")
            gc_out = net.codec.g_c(ai_out)
            ao_f = safe_flops(net.codec.adapter_out[i], gc_out, f"adapter_out[{i}]")
            dummy_yhat_base = torch.randn(B, 640, y_h, y_w, device="cuda")
            lrp_f = safe_flops(net.codec.LRP[i], dummy_yhat_base, f"LRP[{i}]")
            for f in [ai_f, gc_f, ao_f, lrp_f]:
                if f >= 0:
                    ctx_flops += f

        gc_p = count_params(net.codec.g_c)[0]
        ctx_params += gc_p
        results.append(("Codec Context (4-pass checker)", ctx_params, ctx_params, ctx_flops))

        # --- LatentCodec.g_s ---
        gs_p = count_params(net.codec.g_s)
        gs_f = safe_flops(net.codec.g_s, dummy_y, "Codec g_s")
        results.append(("Codec g_s (SynthesisTransform)", gs_p[0], gs_p[1], gs_f))

        # --- LatentCodec.aux ---
        aux_p = count_params(net.codec.aux)
        aux_f = safe_flops(net.codec.aux, dummy_y, "Codec aux (AuxDecoder)")
        results.append(("Codec aux (AuxDecoder)", aux_p[0], aux_p[1], aux_f))

        # --- Entropy models ---
        eb_p = count_params(net.codec.entropy_bottleneck)
        gc_p2 = count_params(net.codec.gaussian_conditional)
        results.append(("Codec entropy models", eb_p[0] + gc_p2[0], eb_p[1] + gc_p2[1], 0))

        # --- UNet ---
        dummy_unet_in = torch.randn(B, 320, vae_h, vae_w, device="cuda")
        dummy_ts = torch.tensor([999], device="cuda").long()
        dummy_enc = net.pos_caption_enc.clone()
        unet_p = count_params(net.unet)
        unet_f = safe_flops(
            net.unet,
            (dummy_unet_in, dummy_ts, dummy_enc),
            "UNet",
        )
        results.append(("UNet (+ LoRA)", unet_p[0], unet_p[1], unet_f))

        # --- VAE Decoder ---
        dummy_dec_in = torch.randn(B, C, vae_h, vae_w, device="cuda")
        pqconv_p = count_params(net.vae.post_quant_conv)
        pqconv_f = safe_flops(net.vae.post_quant_conv, dummy_dec_in, "post_quant_conv")
        pqconv_out = net.vae.post_quant_conv(dummy_dec_in)

        vae_dec = net.vae.decoder
        vae_dec_p = count_params(vae_dec)
        vae_dec_f = safe_flops(vae_dec, pqconv_out, "VAE Decoder")

        results.append(("VAE Decoder+post_quant_conv",
                         vae_dec_p[0] + pqconv_p[0],
                         vae_dec_p[1] + pqconv_p[1],
                         (vae_dec_f if vae_dec_f >= 0 else 0) + (pqconv_f if pqconv_f >= 0 else 0)))

    # --- Print results ---
    print(f"\n{'=' * 70}")
    print(f"{'Component':<35} {'Params':>12} {'Trainable':>12} {'FLOPs':>14}")
    print(f"{'-' * 70}")
    total_p, total_t, total_f = 0, 0, 0
    for name, p, t, f in results:
        f_str = fmt(f) if f >= 0 else "N/A"
        print(f"{name:<35} {fmt(p):>12} {fmt(t):>12} {f_str:>14}")
        total_p += p
        total_t += t
        if f >= 0:
            total_f += f
    print(f"{'-' * 70}")
    print(f"{'TOTAL':<35} {fmt(total_p):>12} {fmt(total_t):>12} {fmt(total_f):>14}")
    print(f"{'=' * 70}\n")

    # Also run torch.profiler for a full forward pass breakdown
    print("Running torch.profiler for full forward pass...")
    with torch.no_grad():
        dummy_full = torch.randn(B, 3, H, W, device="cuda")
        pos_prompt = [1]
        # Warmup
        _ = net(dummy_full, pos_prompt, H, W)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            _ = net(dummy_full, pos_prompt, H, W)
            torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="flops", row_limit=20, top_level_events_only=False
        )
    )

    total_profiler_flops = sum(
        evt.flops for evt in prof.key_averages() if evt.flops > 0
    )
    print(f"\ntorch.profiler total FLOPs: {fmt(total_profiler_flops)}")
    print()


if __name__ == "__main__":
    main()
