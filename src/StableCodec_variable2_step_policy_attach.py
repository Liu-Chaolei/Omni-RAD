"""
StableCodec_variable2_step_policy_attach.py

Attach-only PolicyNet wrapper for a StableCodec checkpoint trained with
trainv1_variable3_debug_step.py. The StableCodec body is loaded and kept frozen;
only codec.timestep_policy is trained by train_policynet_attach.py.
"""

import copy

import torch

from StableCodec_variable2_step_policy import StableCodec as _PolicyStableCodec


class StableCodec(_PolicyStableCodec):
    """StableCodec with optional PolicyNet delta_T correction.

    This class intentionally reuses the PolicyNet-capable implementation because
    its codec state dict is backward-compatible with debug-step checkpoints: all
    old keys load, while the new timestep_policy keys remain newly initialized
    or are loaded from a separate policy checkpoint.
    """

    def __init__(self, sd_path=None, config=None, logger=None, stage=None):
        config = config or {}
        config = dict(config)
        policy_config = dict(config.get("policy_net", {}))
        policy_config.setdefault("t_min", 800.0)
        policy_config.setdefault("t_max", 999.0)
        config["policy_net"] = policy_config
        self.use_policy_delta = bool(config.get("use_policy_delta", False))
        super().__init__(sd_path=sd_path, config=config, logger=logger, stage=stage)
        self._match_debug_step_snr_range(config)

        policy_path = config.get("policy_path")
        if policy_path:
            self.load_policy(policy_path)

    def _match_debug_step_snr_range(self, config):
        """Make the SNR baseline match DynamicTimestepModule from debug_step.

        The PolicyNet branch defaults SNRTimestepFeature to [870, 999], while
        trainv1_variable3_debug_step.py uses DynamicTimestepModule [800, 999].
        For attach mode, use the debug range unless the config explicitly
        provides attach_snr_t_min / attach_snr_t_max.
        """
        if not hasattr(self.codec, "snr_feature") or self.codec.snr_feature is None:
            return
        self.codec.snr_feature.t_min = int(config.get("attach_snr_t_min", 800))
        self.codec.snr_feature.t_max = int(config.get("attach_snr_t_max", 999))

    def load_policy(self, path):
        """Load a standalone PolicyNet checkpoint."""
        sd = torch.load(path, map_location="cpu")
        if "state_dict_policy_net" in sd:
            policy_sd = sd["state_dict_policy_net"]
        elif "state_dict" in sd:
            policy_sd = sd["state_dict"]
        elif "state_dict_codec" in sd:
            policy_sd = {
                k[len("timestep_policy."):]: v
                for k, v in sd["state_dict_codec"].items()
                if k.startswith("timestep_policy.")
            }
        else:
            policy_sd = sd
        missing, unexpected = self.codec.timestep_policy.load_state_dict(policy_sd, strict=False)
        if missing:
            print(f"[Policy Attach] Missing PolicyNet keys: {missing}")
        if unexpected:
            print(f"[Policy Attach] Unexpected PolicyNet keys: {unexpected}")
        print(f"[Policy Attach] Loaded PolicyNet from: {path}")

    def save_policy(self, path, extra=None):
        """Save only the attached PolicyNet."""
        payload = {
            "state_dict_policy_net": self.codec.timestep_policy.state_dict(),
            "use_policy_delta": self.use_policy_delta,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def set_policy_train_only(self):
        """Freeze the StableCodec body and train only PolicyNet."""
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.unet_lora_proj.requires_grad_(False)
        self.codec.timestep_policy.train()
        for p in self.codec.timestep_policy.parameters():
            p.requires_grad = True

    def forward(self, x, pos_prompt, ori_h, ori_w, lmbda=None, use_policy_delta=None):
        B = x.shape[0]
        device = x.device
        if use_policy_delta is None:
            use_policy_delta = self.use_policy_delta

        if lmbda is None:
            lmbda = torch.full((B,), self.codec.lambda_min, dtype=torch.float32, device=device)
        else:
            lmbda = lmbda.to(device).float()
            if lmbda.dim() == 0:
                lmbda = lmbda.expand(B)

        with torch.no_grad():
            latent2 = self.aux_codec((x + 1) / 2).detach()
            pos_caption_enc = torch.cat([self.pos_caption_enc for _ in range(B)], dim=0).to(device)

        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor
        lq_latent_hat, rate_out, res1, T_pred, policy_info = self.codec(
            lq_latent, latent2, ori_h, ori_w, lmbda)

        T_snr = policy_info["T_snr"].to(device)
        T_used = T_pred if use_policy_delta else T_snr
        policy_info = dict(policy_info)
        policy_info["T_pred"] = T_pred
        policy_info["T_used"] = T_used
        policy_info["delta_T"] = T_pred - T_snr
        policy_info["use_policy_delta"] = bool(use_policy_delta)

        film_embed = self.codec.film_embed(lmbda)
        delta_s = self.unet_lora_proj(film_embed)
        lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)

        model_pred = self.unet(
            lq_latent_hat, T_used.long().to(device),
            encoder_hidden_states=pos_caption_enc
        ).sample
        model_pred = model_pred * lora_scale

        step_t = T_used if use_policy_delta else T_used.long().to(device)
        x_denoised = (
            self._batched_ddpm_step(model_pred, step_t, lq_latent_hat[:, :256])
            + res1
        )
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return output_image, rate_out, T_used, policy_info

    def compress(self, x, lmbda=None):
        """Encode bitstream and store SNR timestep under t_snr_val.

        The inherited codec compress path stores the SNR fallback as t_pred_val.
        Rename it so decompression can decide whether to use pure SNR or to run
        PolicyNet from decoder-side features.
        """
        out = super().compress(x, lmbda=lmbda)
        if isinstance(out, dict) and "t_pred_val" in out:
            out["t_snr_val"] = out.pop("t_pred_val")
        return out

    def decompress(self, strings, shape, pos_prompt, lmbda=None, use_policy_delta=None):
        if use_policy_delta is None:
            use_policy_delta = self.use_policy_delta

        adjusted = strings
        if isinstance(strings, dict):
            adjusted = copy.copy(strings)
            if use_policy_delta:
                adjusted.pop("t_pred_val", None)
            elif "t_snr_val" in adjusted:
                adjusted["t_pred_val"] = adjusted["t_snr_val"]
        return super().decompress(adjusted, shape, pos_prompt, lmbda=lmbda)
