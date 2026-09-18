"""
StableCodec_variable2_step_CFT.py — CFT fusion variant: replaces scale/map gate
with CodeFormer-style Controllable Feature Transformation.

CFT changes vs. A0 (StableCodec_variable2_step.py):
  • Import: uses latent_codec_variable2_step_CFT (5-tuple forward, 4-tuple decompress).
  • ScaleFeatureAdapter: scales_all → log(1+s) → 1×1 Conv(320→64) → ×4 upsample
    → 3×3 Conv(64→64) → scale_feat [B,64,H/8,W/8].
  • CFTFusionModule: concat(x0_pred, res1, scale_feat) → 2-layer Conv
    + FiLM(f(λ)) → (α, β). Output layer zero-initialized.
    sample excluded from predictor to avoid compression degradation leakage.
  • Forward fusion (controlled by rho):
      z_direct = x0_pred + res1
      z_cft = x0_pred + w * (α * x0_pred + β)
      z_hat = (1-ρ) * z_direct + ρ * z_cft
  • w: fixed scalar (constructor arg), adjustable at inference.
  • rho: passed as forward arg (training loop controls schedule).
  • save_model: additionally saves state_dict_cft.
  • Checkpoint loading: gracefully handles missing state_dict_cft (warm-start from A0).
  • set_train: CFT module parameters are trainable.

Original A0 docstring follows.
------------------------------------------------------------------------------

Changes from StableCodec_variable2.py:
  • Import: uses latent_codec_variable2_step (adds DynamicTimestepModule).
  • __init__: passes sched.alphas_cumprod to LatentCodec for DynamicTimestepModule.
  • forward:  codec returns 4 values (x_hat, rate_out, res, T_star);
              uses T_star per-image for UNet timestep instead of fixed self.timesteps;
              returns (output_image, RateLossOutput, T_star).
  • decompress: codec.decompress returns (x_hat, res, T_star); uses T_star for UNet.
  • _batched_ddpm_step: hand-rolled DDPM one-step supporting batched timesteps
              (DDPMScheduler.step() does not support batched t).

Injection Point 3 — ϵSD UNet LoRA adaptive scaling (unchanged):
    lora_scale = 1 + tanh(unet_lora_proj(film_embed.mean(0)))  ∈ (0, 2)
    model_pred_scaled = model_pred * lora_scale

    Rationale: at high λ (extreme compression) the quantised lT is degraded;
    the denoiser should contribute more. At low λ (high bitrate) lT is close to
    the original; lighter denoising preserves fidelity. The tanh keeps the scale
    bounded and differentiable. This avoids PEFT internals while achieving the
    same effect as LoRA weight scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched_cuda, my_lora_fwd
from my_utils.vaehook import VAEHook
from latent_codec_variable2_step_CFT import LatentCodec, LAMBDA_MIN, LAMBDA_MAX
import sys
sys.path.append("..")
from ELIC.model.elic_official import ELIC


class ScaleFeatureAdapter(nn.Module):
    """Maps entropy model scales to a compact feature at VAE latent resolution.

    Pipeline: log(1+s) → 1×1 Conv(320→64) → bilinear ×4 → 3×3 Conv(64→64).
    Output: scale_feat [B, 64, H/8, W/8] used as condition input to CFTFusionModule.
    """

    def __init__(self, in_channels: int = 320, out_channels: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, scales_all: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        s = torch.log1p(scales_all)
        s = self.proj(s)
        s = F.interpolate(s, size=target_hw, mode='bilinear', align_corners=False)
        s = self.refine(s)
        return s


class CFTFusionModule(nn.Module):
    """CodeFormer-style Controllable Feature Transformation for latent fusion.

    Predicts affine parameters (α, β) from condition features to modulate the
    UNet prior latent x0_pred. Output layer is zero-initialized so that at
    training start α≈0, β≈0 and CFT has no effect.

    Condition input: concat(x0_pred, res1, scale_feat) — sample excluded to
    avoid compression degradation leakage into α,β.

    Fusion formula:
        z_cft = z_prior + w * (α * z_prior + β)
        z_hat = (1-ρ) * z_direct + ρ * z_cft
    """

    def __init__(self, latent_ch: int = 256, scale_ch: int = 64, film_dim: int = 512):
        super().__init__()
        in_ch = latent_ch * 2 + scale_ch
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.film_to_affine = nn.Linear(film_dim, 512)
        self.out = nn.Conv2d(256, latent_ch * 2, 3, padding=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_prior, z_aux, scale_feat, film_embed, w, rho):
        cond = torch.cat([z_prior, z_aux, scale_feat], dim=1)
        h = self.body(cond)

        gamma, beta_film = self.film_to_affine(film_embed).chunk(2, dim=1)
        gamma = gamma.view(-1, 256, 1, 1)
        beta_film = beta_film.view(-1, 256, 1, 1)
        h = h * (1.0 + gamma) + beta_film

        alpha, beta = self.out(h).chunk(2, dim=1)
        z_cft = z_prior + w * (alpha * z_prior + beta)
        z_direct = z_prior + z_aux
        z_hat = (1.0 - rho) * z_direct + rho * z_cft
        return z_hat, alpha, beta


class StableCodec(torch.nn.Module):
    def __init__(self, sd_path=None, config=None, logger=None, stage=None):
        """
        Args:
            sd_path: path to SD-Turbo model
            config:  dict that must contain (in addition to original keys):
                       lambda_min  (float, default 0.5)
                       lambda_max  (float, default 32.0)
                     'lmbda' key is no longer used for LatentCodec init but may
                     still appear in config for backward compatibility.
        """
        super().__init__()
        self._structure_logger = logger
        self.stage = stage

        lambda_min = float(config.get('lambda_min', LAMBDA_MIN))
        lambda_max = float(config.get('lambda_max', LAMBDA_MAX))

        self.latent_tiled_size    = config['latent_tiled_size']
        self.latent_tiled_overlap = config['latent_tiled_overlap']

        print("[SD-Turbo]: Building SD-Turbo ......")
        self.tokenizer    = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched        = make_1step_sched_cuda(sd_path)
        self.guidance_scale = 1.07

        vae, self.vae_loading_info = AutoencoderKL.from_pretrained(
            sd_path, subfolder="vae",
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True)
        unet, self.unet_loading_info = UNet2DConditionModel.from_pretrained(
            sd_path, subfolder="unet",
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True)

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae

        timesteps = config['timesteps']
        self.timesteps = torch.tensor([timesteps], device="cuda").long()

        self.text_encoder.requires_grad_(False)
        self._init_tiled_vae(
            encoder_tile_size=config['vae_encoder_tiled_size'],
            decoder_tile_size=config['vae_decoder_tiled_size'])
        print("[SD-Turbo]: Done!")

        print("[LoRA]: Initializing LoRA ......")
        target_modules_vae  = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = ["to_k","to_q","to_v","to_out.0","conv","conv1","conv2",
                                "conv_shortcut","conv_out","proj_in","proj_out",
                                "ff.net.2","ff.net.0.proj"]
        lora_rank_vae  = config['lora_rank_vae']
        lora_rank_unet = config['lora_rank_unet']

        vae_lora_config  = LoraConfig(r=lora_rank_vae,  init_lora_weights="gaussian", target_modules=target_modules_vae)
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        self.vae .add_adapter(vae_lora_config,  adapter_name="vae_skip")
        self.unet.add_adapter(unet_lora_config)

        self.vae_lora_layers = []
        for name, module in self.vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)
        print("[LoRA]: Done!")

        print("[Training Setup]: Unfreezing mismatched layers ......")
        print("[Training Setup]: Done!")

        print("[Latent Codec]: Initializing Latent Codec ......")
        # Variable-rate codec with dynamic timestep: pass alphas_cumprod from scheduler
        self.codec = LatentCodec(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            alphas_cumprod=self.sched.alphas_cumprod,
        )
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.unet.conv_in = temp_layer
        print("[Latent Codec]: Done!")

        # ----------------------------------------------------------------
        # Injection Point 3 — ϵSD UNet LoRA adaptive scaling
        #
        # unet_lora_proj maps the shared f(λ) embedding to a scalar offset
        # delta_s.  The final LoRA scale is clamped to (0, 2) via tanh:
        #     lora_scale = 1 + tanh(delta_s)
        # model_pred is multiplied by lora_scale before the scheduler step.
        # ----------------------------------------------------------------
        from latent_codec_variable2_step_CFT import FILM_DIM
        self.unet_lora_proj = nn.Sequential(
            nn.Linear(FILM_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        # Only the LAST Linear is zero-initialised so that delta_s = 0 and
        # lora_scale = 1.0 at training start (identity, no effect on UNet).
        # The first Linear keeps Kaiming default init: this avoids the
        # "dead ReLU" trap (all-zero W2 would null out gradients on Linear-1,
        # leaving delta_s as a λ-independent constant equal to b2).
        nn.init.zeros_(self.unet_lora_proj[-1].weight)
        nn.init.zeros_(self.unet_lora_proj[-1].bias)

        # CFT: Scale feature adapter and CFT fusion module
        self.scale_adapter = ScaleFeatureAdapter(in_channels=320, out_channels=64)
        self.cft_module = CFTFusionModule(latent_ch=256, scale_ch=64, film_dim=FILM_DIM)
        self.w = float(config.get('cft_w', 1.0))

        print("[Prompt]: Setting Prompt ......")
        self.set_prompt(config['pos_prompt'])
        del self.tokenizer, self.text_encoder
        print("[Prompt]: Done!")

        # ----------------------------------------------------------------
        # Load pretrained weights if provided
        # ----------------------------------------------------------------
        if config.get('codec_path') is not None:
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Loading Pretrained Weights ......")
            sd = torch.load(config['codec_path'], map_location="cpu")

            # Codec
            _sd_codec = self.codec.state_dict()
            codec_ckpt = sd.get("state_dict_codec", {})
            for k in codec_ckpt:
                if k in _sd_codec:
                    if codec_ckpt[k].shape == _sd_codec[k].shape:
                        _sd_codec[k] = codec_ckpt[k]
                    else:
                        if "quantiles" in k:
                            raise RuntimeError(
                                f"FATAL: Entropy model param mismatch in '{k}'!\n"
                                f"Checkpoint: {codec_ckpt[k].shape}, Model: {_sd_codec[k].shape}")
                        print(f"[Warning] Skipping '{k}' due to shape mismatch.")
            self.codec.load_state_dict(_sd_codec)
            del _sd_codec, codec_ckpt

            # VAE
            _sd_vae, ckpt_vae = self.vae.state_dict(), sd.get("state_dict_vae", {})
            for k in ckpt_vae:
                if k in _sd_vae:
                    if ckpt_vae[k].shape == _sd_vae[k].shape:
                        _sd_vae[k] = ckpt_vae[k]
                    else:
                        print(f"[Init VAE] Skipping '{k}' due to shape mismatch")
            self.vae.load_state_dict(_sd_vae)
            del _sd_vae, ckpt_vae

            # UNet
            _sd_unet, ckpt_unet = self.unet.state_dict(), sd.get("state_dict_unet", {})
            for k in ckpt_unet:
                if k in _sd_unet:
                    if ckpt_unet[k].shape == _sd_unet[k].shape:
                        _sd_unet[k] = ckpt_unet[k]
                    else:
                        print(f"[Init UNet] Skipping '{k}' due to shape mismatch")
            self.unet.load_state_dict(_sd_unet)
            del _sd_unet, ckpt_unet

            # unet_lora_proj (may not exist in old checkpoints)
            if "state_dict_lora_proj" in sd:
                self.unet_lora_proj.load_state_dict(sd["state_dict_lora_proj"])
                print("  -> Loaded unet_lora_proj weights.")

            # CFT: scale_adapter + cft_module (may not exist when warm-starting from A0)
            if "state_dict_cft" in sd:
                cft_sd = sd["state_dict_cft"]
                scale_adapter_sd = {k.replace("scale_adapter.", ""): v for k, v in cft_sd.items() if k.startswith("scale_adapter.")}
                cft_module_sd = {k.replace("cft_module.", ""): v for k, v in cft_sd.items() if k.startswith("cft_module.")}
                if scale_adapter_sd:
                    self.scale_adapter.load_state_dict(scale_adapter_sd)
                if cft_module_sd:
                    self.cft_module.load_state_dict(cft_module_sd)
                print("  -> Loaded CFT module weights.")
            else:
                print("  -> No CFT weights in checkpoint; using default init (zero-init out).")

            print("[LoRA & Latent Codec & Auxiliary Decoder]: Done!")

        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(config['elic_path'])
        model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        print("[Auxiliary Encoder]: Done!")
    # def unfreeze_mismatched_layers(self, model, loading_info, model_name="model"):
    #     for name, param in model.named_parameters():
    #         if name in loading_info.get("mismatched_keys", []):
    #             param.requires_grad = True
    #             print(f"  -> Unfrozen {model_name}: {name}")

    def unfreeze_mismatched_layers(self, model, loading_info, model_name="model"):
        # mismatched_keys is a list of (key, ckpt_shape, model_shape) tuples
        raw_mismatched = loading_info.get("mismatched_keys", [])
        mismatched_key_set = set()
        for item in raw_mismatched:
            if isinstance(item, (list, tuple)):
                mismatched_key_set.add(item[0])
            else:
                mismatched_key_set.add(item)

        # Also match LoRA-wrapped keys: "x.weight" -> "x.base_layer.weight"
        expanded = set()
        for k in mismatched_key_set:
            expanded.add(k)
            parts = k.rsplit(".", 1)
            if len(parts) == 2:
                expanded.add(f"{parts[0]}.base_layer.{parts[1]}")
        mismatched_key_set = expanded

        unfrozen_count = 0
        for name, param in model.named_parameters():
            if name in mismatched_key_set:
                param.requires_grad = True
                unfrozen_count += 1
                print(f"  -> Unfrozen {model_name}: {name}")

        if unfrozen_count == 0 and len(mismatched_key_set) > 0:
            print(f"  -> [{model_name}] WARNING: mismatched keys found but none matched named_parameters().")
            print(f"     Keys: {sorted(mismatched_key_set)}")
        elif unfrozen_count == 0:
            print(f"  -> No mismatched layers found for {model_name}.")

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def set_prompt(self, pos_prompt):
        caption_tokens = self.tokenizer(
            pos_prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        self.pos_caption_enc = self.text_encoder(caption_tokens)[0]

    # ------------------------------------------------------------------
    # Train / eval helpers
    # ------------------------------------------------------------------

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.unet.requires_grad_(False)
        self.vae .requires_grad_(False)
        self.codec.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet .requires_grad_(False)
        self.vae  .requires_grad_(False)
        self.codec.requires_grad_(True)

        # UNet: LoRA layers + conv_in
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        # VAE: LoRA layers
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        # Variable-rate modules
        for p in self.codec.film_embed.parameters():
            p.requires_grad = True
        for p in self.unet_lora_proj.parameters():
            p.requires_grad = True

        # CFT modules
        for p in self.scale_adapter.parameters():
            p.requires_grad = True
        for p in self.cft_module.parameters():
            p.requires_grad = True

        self.unfreeze_mismatched_layers(self.vae, self.vae_loading_info, "VAE")
        self.unfreeze_mismatched_layers(self.unet, self.unet_loading_info, "UNet")

    # ------------------------------------------------------------------
    # Batched DDPM one-step  (replaces sched.step for per-image T*)
    # ------------------------------------------------------------------

    def _batched_ddpm_step(
        self,
        model_pred: torch.Tensor,   # [B, C, H, W]  noise prediction
        timesteps:  torch.Tensor,    # [B]  per-image timestep (long)
        sample:     torch.Tensor,    # [B, C, H, W]  noisy latent
    ) -> torch.Tensor:
        """DDPM one-step denoising: t → 0, supporting per-image (batched) timesteps.

        SD-Turbo uses set_timesteps(1), meaning the scheduler jumps directly
        from t to 0 in a single step (alpha_prod_t_prev = 1.0).

        With alpha_prod_t_prev = 1.0, the formula simplifies to:
            x0_pred     = (sample - sqrt(1-ᾱ_t) * model_pred) / sqrt(ᾱ_t)
            prev_sample = x0_pred    (since sqrt(1.0)=1, sqrt(0)=0)

        This is a direct prediction of the clean latent x₀.
        """
        ac = self.sched.alphas_cumprod                        # [1000]
        # Gather per-image ᾱ_t
        t = timesteps.clamp(0, len(ac) - 1)                  # [B]
        alpha_prod_t = ac[t].view(-1, 1, 1, 1)               # [B, 1, 1, 1]

        sqrt_alpha_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_prod_t).sqrt()

        # Predict x0: one-step direct jump to t=0
        x0_pred = (sample - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t

        return x0_pred

    # ------------------------------------------------------------------
    # Forward  (training / validation)
    # ------------------------------------------------------------------

    def forward(self, x, pos_prompt, ori_h, ori_w, lmbda=None, rho=0.0, w=None):
        """
        Args:
            x:          [B, 3, H, W]  input image in [-1, 1]
            pos_prompt: list[int]  (1 per image, used only for caption lookup)
            ori_h, ori_w: original image height/width
            lmbda:      Tensor [B] or None.  If None the training loop should
                        sample and pass it; if called without it a default is used.
            rho:        float in [0,1]. Controls transition from direct fusion (0)
                        to CFT fusion (1). Training loop schedules this.
            w:          float or None. CFT injection strength. If None uses self.w.

        Returns:
            output_image  [B, 3, H, W]  in [-1, 1]
            RateLossOutput
            T_star        [B]            dynamic timestep per image
        """
        B = x.shape[0]
        device = x.device

        # Resolve lmbda
        if lmbda is None:
            lmbda = torch.full((B,), self.codec.lambda_min,
                               dtype=torch.float32, device=device)
        else:
            lmbda = lmbda.to(device).float()
            if lmbda.dim() == 0:
                lmbda = lmbda.expand(B)

        # ---- Encoder ----
        with torch.no_grad():
            latent2 = self.aux_codec((x + 1) / 2).detach()
            pos_caption_enc = torch.cat(
                [self.pos_caption_enc for _ in range(B)], dim=0
            ).to(device)
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        # ---- Latent Codec (Injection Points 1, 2, 4) ----
        lq_latent_hat, RateLossOutput, res1, T_star, scales_all = self.codec(
            lq_latent, latent2, ori_h, ori_w, lmbda)

        # ---- Injection Point 3: UNet LoRA adaptive scaling ----
        # Compute shared embedding f(λ) once and derive a per-batch scalar scale.
        # film_embed: [B, FILM_DIM];  take mean over batch → [1, FILM_DIM]
        film_embed  = self.codec.film_embed(lmbda)          # [B, FILM_DIM]
        delta_s     = self.unet_lora_proj(film_embed)                 # [B, 1]
        lora_scale  = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)   # [B, 1, 1, 1]

        # ---- One-Step Denoiser with per-image dynamic timestep T* ----
        # T_star is [B] float; UNet supports batched timesteps
        t_star_long = T_star.long().to(device)
        model_pred = self.unet(
            lq_latent_hat, t_star_long,
            encoder_hidden_states=pos_caption_enc
        ).sample
        # Scale model prediction (Injection Point 3)
        model_pred = model_pred * lora_scale

        # CFT fusion: x0_pred as prior, res1/scale_feat as conditions
        sample = lq_latent_hat[:, :256]
        x0_pred = self._batched_ddpm_step(model_pred, t_star_long, sample)
        scale_feat = self.scale_adapter(scales_all, target_hw=sample.shape[-2:])
        w_val = w if w is not None else self.w
        x_denoised, _, _ = self.cft_module(
            z_prior=x0_pred, z_aux=res1,
            scale_feat=scale_feat, film_embed=film_embed,
            w=w_val, rho=rho,
        )

        # ---- Decoder ----
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        return output_image, RateLossOutput, T_star

    # ------------------------------------------------------------------
    # Compress  (inference encoding)
    # ------------------------------------------------------------------

    def compress(self, x, lmbda=None):
        """
        Args:
            lmbda: Tensor [B] or scalar float.  Stored as float16 in output dict
                   (2 bytes per image — from I2C bitstream design).
        """
        B = x.shape[0]
        device = x.device

        if lmbda is None:
            lmbda = torch.full((B,), self.codec.lambda_min,
                               dtype=torch.float32, device=device)
        else:
            lmbda = torch.tensor(lmbda, dtype=torch.float32, device=device)
            if lmbda.dim() == 0:
                lmbda = lmbda.expand(B)

        latent2   = self.aux_codec((x + 1) / 2).detach()
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        output_dict = self.codec.compress(lq_latent, latent2, lmbda)
        return output_dict

    # ------------------------------------------------------------------
    # Decompress  (inference decoding)
    # ------------------------------------------------------------------

    def decompress(self, strings, shape, pos_prompt, lmbda=None, w=None):
        """
        lmbda is read from strings["lmbda_val"] if not explicitly provided.
        w: CFT injection strength. If None uses self.w. At inference rho=1.
        """
        device = next(self.parameters()).device

        # Resolve lmbda
        if lmbda is None and isinstance(strings, dict) and "lmbda_val" in strings:
            lmbda = torch.tensor(strings["lmbda_val"], dtype=torch.float32, device=device)

        lq_latent_hat, res, T_star, scales_all = self.codec.decompress(strings, shape, lmbda)
        lq_latent_hat = lq_latent_hat.to(device)
        res           = res.to(device)
        T_star        = T_star.to(device)

        B = lq_latent_hat.size(0)
        pos_caption_enc = torch.cat(
            [self.pos_caption_enc for _ in range(len(pos_prompt))], dim=0
        ).to(device)

        # ---- Injection Point 3 at inference ----
        if lmbda is not None:
            lmbda_t    = lmbda.to(device).float()
            if lmbda_t.dim() == 0:
                lmbda_t = lmbda_t.expand(B)
            film_embed = self.codec.film_embed(lmbda_t)
            delta_s    = self.unet_lora_proj(film_embed)           # [B, 1]
            lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)
        else:
            lora_scale = 1.0
            film_embed = None

        # Use dynamic T* for denoising (per-image)
        t_star_long = T_star.long().to(device)

        # ---- Tiled one-step denoising ----
        _, _, h, w = lq_latent_hat.size()
        tile_size, tile_overlap = self.latent_tiled_size, self.latent_tiled_overlap
        if h * w <= tile_size * tile_size:
            model_pred = self.unet(
                lq_latent_hat, t_star_long,
                encoder_hidden_states=pos_caption_enc
            ).sample * lora_scale
        else:
            print(f"[Tiled Latent]: input latent is {h}x{w}, tiling ...")
            tile_size  = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(device)

            def grid_steps(dim):
                steps, cur = 0, 0
                while cur < dim:
                    cur = max(steps * tile_size - tile_overlap * steps, 0) + tile_size
                    steps += 1
                return steps

            grid_rows = grid_steps(w)
            grid_cols = grid_steps(h)

            input_list, noise_preds = [], []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    ox = (w - tile_size) if row == grid_rows - 1 else max(row * tile_size - tile_overlap * row, 0)
                    oy = (h - tile_size) if col == grid_cols - 1 else max(col * tile_size - tile_overlap * col, 0)
                    tile = lq_latent_hat[:, :, oy:oy+tile_size, ox:ox+tile_size]
                    input_list.append(tile)
                    if len(input_list) == 1 or col == grid_cols - 1:
                        pred = self.unet(
                            torch.cat(input_list, 0), t_star_long,
                            encoder_hidden_states=pos_caption_enc
                        ).sample * lora_scale
                        input_list = []
                    noise_preds.append(pred)

            noise_pred   = torch.zeros_like(lq_latent_hat[:, :4])
            contributors = torch.zeros_like(lq_latent_hat[:, :4])
            for row in range(grid_rows):
                for col in range(grid_cols):
                    ox = (w - tile_size) if row == grid_rows - 1 else max(row * tile_size - tile_overlap * row, 0)
                    oy = (h - tile_size) if col == grid_cols - 1 else max(col * tile_size - tile_overlap * col, 0)
                    noise_pred  [:, :, oy:oy+tile_size, ox:ox+tile_size] += noise_preds[row*grid_cols+col] * tile_weights
                    contributors[:, :, oy:oy+tile_size, ox:ox+tile_size] += tile_weights
            model_pred = noise_pred / contributors

        # CFT fusion at decompress (rho=1 at inference)
        sample = lq_latent_hat[:, :256]
        x0_pred = self._batched_ddpm_step(model_pred, t_star_long, sample)
        scale_feat = self.scale_adapter(scales_all, target_hw=sample.shape[-2:])
        w_val = w if w is not None else self.w
        if film_embed is not None:
            x_denoised, _, _ = self.cft_module(
                z_prior=x0_pred, z_aux=res,
                scale_feat=scale_feat, film_embed=film_embed,
                w=w_val, rho=1.0,
            )
        else:
            x_denoised = x0_pred + res
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return output_image

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae"]  = {k: v for k, v in self.vae.state_dict().items()  if "lora" in k}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        # Save variable-rate modules
        sd["state_dict_lora_proj"] = {k: v for k, v in self.unet_lora_proj.state_dict().items()}
        # CFT: Save scale_adapter and cft_module together
        cft_sd = {}
        for k, v in self.scale_adapter.state_dict().items():
            cft_sd[f"scale_adapter.{k}"] = v
        for k, v in self.cft_module.state_dict().items():
            cft_sd[f"cft_module.{k}"] = v
        sd["state_dict_cft"] = cft_sd
        torch.save(sd, outf)

    # ------------------------------------------------------------------
    # Tiling helpers  (unchanged)
    # ------------------------------------------------------------------

    def _set_latent_tile(self, latent_tiled_size=96, latent_tiled_overlap=32):
        self.latent_tiled_size    = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256,
                        fast_decoder=False, fast_encoder=False,
                        color_fix=False, vae_to_gpu=True):
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)
        self.vae.encoder.forward = VAEHook(
            self.vae.encoder, encoder_tile_size,
            is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            self.vae.decoder, decoder_tile_size,
            is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder,
            color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        from numpy import pi, exp, sqrt
        import numpy as np
        var = 0.01
        mx  = (tile_width  - 1) / 2
        my  = tile_height / 2
        x_p = [exp(-(x-mx)**2 / (tile_width **2) / (2*var)) / sqrt(2*pi*var) for x in range(tile_width)]
        y_p = [exp(-(y-my)**2 / (tile_height**2) / (2*var)) / sqrt(2*pi*var) for y in range(tile_height)]
        weights = np.outer(y_p, x_p)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))