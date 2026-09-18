import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched, my_lora_fwd
from my_utils.vaehook import VAEHook
from latent_codec_ori import LatentCodec
import sys
sys.path.append("..")
from ELIC.model.elic_official import ELIC
from cft_fuse_v3 import CFTFuseBlock, LambdaAdaptiveW


class StableCodec(torch.nn.Module):
    # SD-VAE decoder up_block output channels [block0..block3].
    # Adjust if using a non-standard VAE (e.g. SDXL).
    _UP_BLOCK_CH: list = [512, 512, 256, 128]

    def __init__(self, sd_path=None, lmbda=0.5, config=None, logger=None, stage=None):
        super().__init__()
        self._structure_logger = logger
        self.stage = stage
        self.lmbda = lmbda  # 保存 λ，供 LambdaAdaptiveW 使用

        self.latent_tiled_size = config['latent_tiled_size']
        self.latent_tiled_overlap = config['latent_tiled_overlap']

        print("[SD-Turbo]: Building SD-Turbo ......")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
        self.sched = make_1step_sched(sd_path)
        self.guidance_scale = 1.07

        vae, self.vae_loading_info = AutoencoderKL.from_pretrained(
            sd_path,
            subfolder="vae",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        unet, self.unet_loading_info = UNet2DConditionModel.from_pretrained(
            sd_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )

        self.unet, self.vae = unet, vae
        # self.timesteps = torch.tensor([999], device="cuda").long()
        # timesteps = 1 * math.log(lmbda) + 900
        timesteps = config['timesteps']
        self.timesteps = torch.tensor([timesteps]).long()

        self.text_encoder.requires_grad_(False)

        self._init_tiled_vae(encoder_tile_size=config['vae_encoder_tiled_size'], decoder_tile_size=config['vae_decoder_tiled_size'])
        print("[SD-Turbo]: Done!")

        print("[LoRA]: Initializing LoRA ......")
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        lora_rank_vae = config['lora_rank_vae']
        lora_rank_unet = config['lora_rank_unet']

        vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
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

        print("[Latent Codec]: Initializing Latent Codec ......")
        self.codec = LatentCodec(lmbda)
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unet.conv_in = temp_layer
        print("[Latent Codec]: Done!")

        print("[Prompt]: Setting Prompt ......")
        self.set_prompt(config['pos_prompt'])
        del self.tokenizer, self.text_encoder
        print("[Prompt]: Done!")

        res_ch = config['latent_channels']
        inject_blocks = getattr(config, "cft_inject_blocks", [0, 1, 2])
        self.cft_blocks = nn.ModuleDict({
            f"up_{i}": CFTFuseBlock(res_ch, self._UP_BLOCK_CH[i])
            for i in inject_blocks
        })
        self.cft_inject_blocks = inject_blocks

        # ---- fusion3 新增：λ 自适应权重 MLP ----
        self.lambda_adaptive_w = LambdaAdaptiveW(hidden_dim=64)
        self._current_w = None   # hook 读取的当前融合权重
        self._w_override = None  # stage 2/3 强制覆盖值
        self._res_feat = None

        if config['codec_path'] is not None:
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Loading Pretrained Weights ......")
            sd = torch.load(config['codec_path'], map_location="cpu")

            _sd_codec = self.codec.state_dict()
            codec_ckpt = sd["state_dict_codec"]
            for k in codec_ckpt:
                if k in _sd_codec:
                    if codec_ckpt[k].shape == _sd_codec[k].shape:
                        _sd_codec[k] = codec_ckpt[k]
                    else:
                        if "quantiles" in k:
                            raise RuntimeError(
                                f"FATAL: Entropy model parameter mismatch detected in '{k}'!\n"
                                f"Checkpoint shape: {codec_ckpt[k].shape}\n"
                                f"Current model shape: {_sd_codec[k].shape}\n"
                                "Mismatch in quantiles implies incompatible entropy bottleneck configuration. "
                                "Training cannot proceed as bitrate estimation will be wrong."
                            )
                        print(f"[Warning] Skipping layer '{k}' due to shape mismatch. "
                              f"Checkpoint: {codec_ckpt[k].shape}, Model: {_sd_codec[k].shape}")
            self.codec.load_state_dict(_sd_codec)
            del _sd_codec, codec_ckpt

            _sd_vae = self.vae.state_dict()
            ckpt_vae = sd["state_dict_vae"]
            for k in ckpt_vae:
                if k in _sd_vae:
                    if ckpt_vae[k].shape == _sd_vae[k].shape:
                        _sd_vae[k] = ckpt_vae[k]
                    else:
                        print(f"[Init VAE] Skipping '{k}' due to shape mismatch")
            self.vae.load_state_dict(_sd_vae)
            del _sd_vae, ckpt_vae

            _sd_unet = self.unet.state_dict()
            ckpt_unet = sd["state_dict_unet"]
            for k in ckpt_unet:
                if k in _sd_unet:
                    if ckpt_unet[k].shape == _sd_unet[k].shape:
                        _sd_unet[k] = ckpt_unet[k]
                    else:
                        print(f"[Init UNet] Skipping '{k}' due to shape mismatch")
            self.unet.load_state_dict(_sd_unet)
            del _sd_unet, ckpt_unet

            if "state_dict_cft" in sd:
                self.cft_blocks.load_state_dict(sd["state_dict_cft"], strict=False)
            if "state_dict_lambda_w" in sd:
                self.lambda_adaptive_w.load_state_dict(sd["state_dict_lambda_w"])
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Done!")

        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(config['elic_path'], map_location="cpu")
        model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        print("[Auxiliary Encoder]: Done!")

        for i in inject_blocks:
            self.vae.decoder.up_blocks[i].register_forward_hook(
                self._make_cft_hook(i)
            )

    def unfreeze_mismatched_layers(self, model, loading_info, model_name="model", logger=None):
        log = logger.info if logger is not None else print

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
                log(f"  -> Unfrozen {model_name}: {name}")

        if unfrozen_count == 0 and len(mismatched_key_set) > 0:
            log(f"  -> [{model_name}] WARNING: mismatched keys found but none matched named_parameters().")
            log(f"     Keys: {sorted(mismatched_key_set)}")
        elif unfrozen_count == 0:
            log(f"  -> No mismatched layers found for {model_name}.")

        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        if missing:
            log(f"  -> [{model_name}] missing_keys ({len(missing)}): {sorted(missing)}")
        else:
            log(f"  -> [{model_name}] No missing keys.")
        if unexpected:
            log(f"  -> [{model_name}] unexpected_keys ({len(unexpected)}): {sorted(unexpected)}")
        else:
            log(f"  -> [{model_name}] No unexpected keys.")

    # ------------------------------------------------------------------
    # CFT hook & decode (fusion3: λ-adaptive w + spatial attention)
    # ------------------------------------------------------------------

    def _make_cft_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if self._res_feat is not None:
                output = self.cft_blocks[f"up_{block_idx}"](
                    self._res_feat, output, self._current_w
                )
            return output
        return hook_fn

    def _decode_with_cft(self, z: torch.Tensor, res_feat: torch.Tensor, w_override=None):
        if w_override is not None:
            self._current_w = w_override
        else:
            lmbda_t = torch.tensor([[self.lmbda]], device=z.device, dtype=z.dtype)
            self._current_w = self.lambda_adaptive_w(lmbda_t)  # [1, 1]
        self._res_feat = res_feat
        try:
            out = self.vae.decode(z).sample
        finally:
            self._res_feat = None
            self._current_w = None
        return out

    def set_prompt(self, pos_prompt):
        caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        self.pos_caption_enc = self.text_encoder(caption_tokens)[0]

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.cft_blocks.eval()
        self.lambda_adaptive_w.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.cft_blocks.requires_grad_(False)
        self.lambda_adaptive_w.requires_grad_(False)

    # ------------------------------------------------------------------
    # Stage-aware training setup
    # ------------------------------------------------------------------

    def _freeze_all_trainable(self):
        """Freeze every trainable component. aux_codec is always frozen."""
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.cft_blocks.requires_grad_(False)
        self.lambda_adaptive_w.requires_grad_(False)
        self.aux_codec.requires_grad_(False)

    def set_train(self, stage: int = 1):
        """
        Configure trainable parameters for the requested stage.

        Stage 1   MLP(λ)        : full end-to-end training.
        Stage 2   w_override=0  : x_diff path only (codec.g_s / UNet / VAE-dec front half).
        Stage 3   w_override=1  : res1 path only (codec.aux / CFT blocks, spatial attn trains).
        Stage 4   MLP(λ)        : full end-to-end fine-tuning (same as stage 1).
        """
        dispatch = {
            1: self._set_train_stage1,
            2: self._set_train_stage2,
            3: self._set_train_stage3,
            4: self._set_train_stage4,
        }
        if stage not in dispatch:
            raise ValueError(f"Unknown stage {stage}. Must be 1, 2, 3, or 4.")
        dispatch[stage]()

    def _set_train_stage1(self):
        """
        Stage 1: MLP(λ) predicts w, full end-to-end training.

        Trainable: codec, unet LoRA+conv_in, vae LoRA, cft_blocks, lambda_adaptive_w.
        """
        self._w_override = None
        self._freeze_all_trainable()

        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.cft_blocks.train()
        self.lambda_adaptive_w.train()

        self.codec.requires_grad_(True)

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        self.unfreeze_mismatched_layers(self.vae, self.vae_loading_info, "VAE", logger=self._structure_logger)
        self.unfreeze_mismatched_layers(self.unet, self.unet_loading_info, "UNet", logger=self._structure_logger)

        self.cft_blocks.requires_grad_(True)
        self.lambda_adaptive_w.requires_grad_(True)

    def _set_train_stage2(self):
        """
        Stage 2: w_override=0, x_diff path only.

        Trainable: codec.g_s, unet LoRA+conv_in, vae decoder front half.
        CFT blocks and lambda_adaptive_w frozen (w=0 makes CFT injection zero).
        """
        self._w_override = 0.0
        self._freeze_all_trainable()

        self.unet.train()
        self.vae.train()
        self.codec.eval()
        self.codec.g_s.train()
        self.cft_blocks.eval()
        self.lambda_adaptive_w.eval()

        self.codec.g_s.requires_grad_(True)

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        dec = self.vae.decoder
        if hasattr(dec, "conv_in"):
            dec.conv_in.requires_grad_(True)
        if hasattr(dec, "mid_block"):
            dec.mid_block.requires_grad_(True)
        n_up = len(dec.up_blocks)
        for i in range(math.ceil(n_up / 2)):
            dec.up_blocks[i].requires_grad_(True)

        for n, p in self.codec.named_parameters():
            if n.endswith(".quantiles"):
                p.requires_grad = True

    def _set_train_stage3(self):
        """
        Stage 3: w_override=1, res1 path only.

        Trainable: codec.aux, cft_blocks (including spatial_attn).
        lambda_adaptive_w frozen; combined_w = 1.0 * attn_map = attn_map,
        so spatial attention is trained in isolation.
        """
        self._w_override = 1.0
        self._freeze_all_trainable()

        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.codec.aux.train()
        self.cft_blocks.train()
        self.lambda_adaptive_w.eval()

        self.codec.aux.requires_grad_(True)
        self.cft_blocks.requires_grad_(True)

        for n, p in self.codec.named_parameters():
            if n.endswith(".quantiles"):
                p.requires_grad = True

    def _set_train_stage4(self):
        """
        Stage 4: MLP(λ), full end-to-end fine-tuning.
        Identical parameter set to stage 1.
        """
        self._set_train_stage1()

    # ------------------------------------------------------------------
    # Forward / Compress / Decompress
    # ------------------------------------------------------------------

    def forward(self, x, pos_prompt, ori_h, ori_w):

        device = x.device
        self.timesteps = self.timesteps.to(device)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device)

        # Encoder
        with torch.no_grad():
            latent2 = self.aux_codec((x + 1) / 2).detach()
            pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
            pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(x.device)
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        # Latent Codec
        lq_latent_hat, RateLossOutput, res1 = self.codec(lq_latent, latent2, ori_h, ori_w)

        # One-Step Denoiser
        model_pred = self.unet(lq_latent_hat, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        x_diff = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :256], return_dict=True).prev_sample

        # Decoder with CFT fusion (λ-adaptive w + spatial attention)
        output_image = self._decode_with_cft(
            x_diff / self.vae.config.scaling_factor,
            res_feat=res1,
            w_override=self._w_override,
        ).clamp(-1, 1)

        return output_image, RateLossOutput

    def compress(self, x):

        # Encoder
        latent2 = self.aux_codec((x + 1) / 2).detach()
        lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

        # Latent Codec - Entropy Encoding
        output_dict = self.codec.compress(lq_latent, latent2)

        return output_dict

    def decompress(self, strings, shape, pos_prompt):

        # Latent Codec - Entropy Decoding
        lq_latent_hat, res = self.codec.decompress(strings, shape)

        device = lq_latent_hat.device
        self.timesteps = self.timesteps.to(device)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device)

        pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
        pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(lq_latent_hat.device)

        # One-Step Denoiser with tile function
        _, _, h, w = lq_latent_hat.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = self.unet(lq_latent_hat, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
        else:
            print(f"[Tiled Latent]: the input latent is {h}x{w}, need to tiled")
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(lq_latent_hat.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent_hat.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent_hat.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent_hat[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
                        input_list = []
                    noise_preds.append(model_pred)

            noise_pred = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            contributors = torch.zeros(lq_latent_hat[:, :4].shape, device=lq_latent_hat.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :4], return_dict=True).prev_sample + res

        # Decoder with CFT fusion (λ-adaptive w + spatial attention)
        output_image = self._decode_with_cft(
            x_denoised / self.vae.config.scaling_factor,
            res_feat=res,
        ).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        sd["state_dict_cft"] = dict(self.cft_blocks.state_dict())
        sd["state_dict_lambda_w"] = dict(self.lambda_adaptive_w.state_dict())
        torch.save(sd, outf)

    def _set_latent_tile(self, latent_tiled_size=96, latent_tiled_overlap=32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap

    def _init_tiled_vae(self,
            encoder_tile_size=256,
            decoder_tile_size=256,
            fast_decoder=False,
            fast_encoder=False,
            color_fix=False,
            vae_to_gpu=True):
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))
