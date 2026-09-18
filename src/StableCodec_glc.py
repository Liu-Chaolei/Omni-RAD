"""
StableCodec_glc.py — StableCodec with GLC-style variable-rate (global/local learned scalers)
and SNR-based dynamic timestep T*.

Changes from StableCodec_variable2_step.py:
  • Variable-rate: GLC global/local scalers replace λ-FiLM in codec transforms.
  • Input: discrete quality_index ∈ [0, NUM_QUALITY-1] instead of continuous λ.
  • QualityLambdaEmbed: maps quality_index → fixed λ → Fourier → MLP → R^512
    (used only for UNet LoRA adaptive scaling, Injection Point 3).
  • Codec: latent_codec_glc.LatentCodec (plain g_a/g_s/aux, no FiLM).
  • DynamicTimestepModule: unchanged (SNR-based T* from entropy model scales).

Injection Point 3 — UNet LoRA adaptive scaling (retained):
    embed = quality_embed(quality_index)
    lora_scale = 1 + tanh(unet_lora_proj(embed))  ∈ (0, 2)
    model_pred_scaled = model_pred * lora_scale
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched_cuda, my_lora_fwd
from my_utils.vaehook import VAEHook
from latent_codec_glc import LatentCodec, QualityLambdaEmbed, FILM_DIM, NUM_QUALITY
import sys
sys.path.append("..")
from ELIC.model.elic_official import ELIC


class StableCodec(torch.nn.Module):
    def __init__(self, sd_path=None, config=None, logger=None, stage=None):
        """
        Args:
            sd_path: path to SD-Turbo model
            config:  dict with model/training configuration
        """
        super().__init__()
        self._structure_logger = logger
        self.stage = stage

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
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                               "conv_shortcut", "conv_out", "proj_in", "proj_out",
                               "ff.net.2", "ff.net.0.proj"]
        lora_rank_vae  = config['lora_rank_vae']
        lora_rank_unet = config['lora_rank_unet']

        vae_lora_config  = LoraConfig(r=lora_rank_vae,  init_lora_weights="gaussian", target_modules=target_modules_vae)
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
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
        # GLC variable-rate codec with dynamic timestep
        self.codec = LatentCodec(alphas_cumprod=self.sched.alphas_cumprod)
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.unet.conv_in = temp_layer
        print("[Latent Codec]: Done!")

        # ----------------------------------------------------------------
        # QualityLambdaEmbed: quality_index → λ_table → Fourier → MLP → R^512
        # Used only for Injection Point 3 (UNet LoRA adaptive scaling).
        # ----------------------------------------------------------------
        self.quality_embed = QualityLambdaEmbed()

        # ----------------------------------------------------------------
        # Injection Point 3 — UNet LoRA adaptive scaling
        # unet_lora_proj: R^FILM_DIM → scalar delta_s
        # lora_scale = 1 + tanh(delta_s)  ∈ (0, 2)
        # ----------------------------------------------------------------
        self.unet_lora_proj = nn.Sequential(
            nn.Linear(FILM_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        # Zero-init last layer so lora_scale = 1.0 at training start
        nn.init.zeros_(self.unet_lora_proj[-1].weight)
        nn.init.zeros_(self.unet_lora_proj[-1].bias)

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

            # quality_embed (may not exist in old checkpoints)
            if "state_dict_quality_embed" in sd:
                self.quality_embed.load_state_dict(sd["state_dict_quality_embed"])
                print("  -> Loaded quality_embed weights.")

            print("[LoRA & Latent Codec & Auxiliary Decoder]: Done!")

        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(config['elic_path'])
        model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        print("[Auxiliary Encoder]: Done!")

    def unfreeze_mismatched_layers(self, model, loading_info, model_name="model"):
        raw_mismatched = loading_info.get("mismatched_keys", [])
        mismatched_key_set = set()
        for item in raw_mismatched:
            if isinstance(item, (list, tuple)):
                mismatched_key_set.add(item[0])
            else:
                mismatched_key_set.add(item)

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
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
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

        # GLC variable-rate modules
        self.codec.global_q_enc.requires_grad = True
        self.codec.global_q_dec.requires_grad = True
        for p in self.quality_embed.parameters():
            p.requires_grad = True
        for p in self.unet_lora_proj.parameters():
            p.requires_grad = True

        self.unfreeze_mismatched_layers(self.vae, self.vae_loading_info, "VAE")
        self.unfreeze_mismatched_layers(self.unet, self.unet_loading_info, "UNet")

    # ------------------------------------------------------------------
    # Batched DDPM one-step (replaces sched.step for per-image T*)
    # ------------------------------------------------------------------

    def _batched_ddpm_step(
        self,
        model_pred: torch.Tensor,   # [B, C, H, W]  noise prediction
        timesteps:  torch.Tensor,   # [B]  per-image timestep (long)
        sample:     torch.Tensor,   # [B, C, H, W]  noisy latent
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
        t = timesteps.clamp(0, len(ac) - 1)                  # [B]
        alpha_prod_t = ac[t].view(-1, 1, 1, 1)               # [B, 1, 1, 1]

        sqrt_alpha_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_prod_t).sqrt()

        # Predict x0: one-step direct jump to t=0
        x0_pred = (sample - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t

        return x0_pred

    # ------------------------------------------------------------------
    # Forward (training / validation)
    # ------------------------------------------------------------------

    def forward(self, x, pos_prompt, ori_h, ori_w, quality_index=None):
        """
        Args:
            x:             [B, 3, H, W]  input image in [-1, 1]
            pos_prompt:    list[int]  (1 per image, used only for caption lookup)
            ori_h, ori_w:  original image height/width
            quality_index: Tensor [B] int64, values in [0, NUM_QUALITY-1].
                           If None, defaults to 0 (highest quality).

        Returns:
            output_image   [B, 3, H, W]  in [-1, 1]
            RateLossOutput
            T_star         [B]  dynamic timestep per image
        """
        B = x.shape[0]
        device = x.device

        # Resolve quality_index
        if quality_index is None:
            quality_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            quality_index = quality_index.to(device).long()

        # ---- Encoder ----
        with torch.no_grad():
            latent2 = self.aux_codec((x + 1) / 2).detach()
            pos_caption_enc = torch.cat(
                [self.pos_caption_enc for _ in range(B)], dim=0
            ).to(device)
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        # ---- Latent Codec (GLC global/local scalers) ----
        lq_latent_hat, RateLossOutput, res1, T_star = self.codec(
            lq_latent, latent2, ori_h, ori_w, quality_index)

        # ---- Injection Point 3: UNet LoRA adaptive scaling ----
        embed      = self.quality_embed(quality_index)                    # [B, FILM_DIM]
        delta_s    = self.unet_lora_proj(embed)                           # [B, 1]
        lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)        # [B, 1, 1, 1]

        # ---- One-Step Denoiser with per-image dynamic timestep T* ----
        t_star_long = T_star.long().to(device)
        model_pred = self.unet(
            lq_latent_hat, t_star_long,
            encoder_hidden_states=pos_caption_enc
        ).sample
        # Scale model prediction (Injection Point 3)
        model_pred = model_pred * lora_scale

        x_denoised = (
            self._batched_ddpm_step(model_pred, t_star_long,
                                    lq_latent_hat[:, :256])
            + res1
        )

        # ---- Decoder ----
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        return output_image, RateLossOutput, T_star

    # ------------------------------------------------------------------
    # Compress (inference encoding)
    # ------------------------------------------------------------------

    def compress(self, x, quality_index=None):
        """
        Args:
            x:             [B, 3, H, W] input image in [-1, 1]
            quality_index: Tensor [B] or int. Stored as uint8 in output dict.
        """
        B = x.shape[0]
        device = x.device

        if quality_index is None:
            quality_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            quality_index = torch.tensor(quality_index, dtype=torch.long, device=device)
            if quality_index.dim() == 0:
                quality_index = quality_index.expand(B)

        latent2   = self.aux_codec((x + 1) / 2).detach()
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        output_dict = self.codec.compress(lq_latent, latent2, quality_index)
        return output_dict

    # ------------------------------------------------------------------
    # Decompress (inference decoding)
    # ------------------------------------------------------------------

    def decompress(self, strings, shape, pos_prompt, quality_index=None):
        """
        quality_index is read from strings["quality_index"] if not explicitly provided.
        """
        device = next(self.parameters()).device

        # Resolve quality_index
        if quality_index is None and isinstance(strings, dict) and "quality_index" in strings:
            quality_index = torch.tensor(strings["quality_index"], dtype=torch.long, device=device)

        lq_latent_hat, res, T_star = self.codec.decompress(strings, shape, quality_index)
        lq_latent_hat = lq_latent_hat.to(device)
        res           = res.to(device)
        T_star        = T_star.to(device)

        B = lq_latent_hat.size(0)
        pos_caption_enc = torch.cat(
            [self.pos_caption_enc for _ in range(len(pos_prompt))], dim=0
        ).to(device)

        # ---- Injection Point 3 at inference ----
        if quality_index is not None:
            qi = quality_index.to(device).long()
            if qi.dim() == 0:
                qi = qi.expand(B)
            embed      = self.quality_embed(qi)                           # [B, FILM_DIM]
            delta_s    = self.unet_lora_proj(embed)                       # [B, 1]
            lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)
        else:
            lora_scale = 1.0

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
                    noise_pred[:, :, oy:oy+tile_size, ox:ox+tile_size] += noise_preds[row*grid_cols+col] * tile_weights
                    contributors[:, :, oy:oy+tile_size, ox:ox+tile_size] += tile_weights
            model_pred = noise_pred / contributors

        x_denoised = (
            self._batched_ddpm_step(model_pred, t_star_long,
                                    lq_latent_hat[:, :4])
            + res
        )
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return output_image

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae"]  = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        # Save variable-rate modules
        sd["state_dict_lora_proj"] = {k: v for k, v in self.unet_lora_proj.state_dict().items()}
        sd["state_dict_quality_embed"] = {k: v for k, v in self.quality_embed.state_dict().items()}
        torch.save(sd, outf)

    # ------------------------------------------------------------------
    # Tiling helpers (unchanged)
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
        x_p = [exp(-(x-mx)**2 / (tile_width**2) / (2*var)) / sqrt(2*pi*var) for x in range(tile_width)]
        y_p = [exp(-(y-my)**2 / (tile_height**2) / (2*var)) / sqrt(2*pi*var) for y in range(tile_height)]
        weights = np.outer(y_p, x_p)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))

