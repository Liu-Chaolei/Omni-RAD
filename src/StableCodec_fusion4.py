import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from model import make_1step_sched_cuda, my_lora_fwd
from my_utils.vaehook import VAEHook
from latent_codec_ori import LatentCodec
import sys
sys.path.append("..")
from ELIC.model.elic_official import ELIC
from DCAE import DCAE
from collections import OrderedDict

'''
两条支路的方案，还需要去冗余
'''

class ContentAdaptiveFusion(nn.Module):
    def __init__(self, y_channels: int = 320, rate_embed_dim: int = 32):
        super().__init__()

        self.rate_embed = nn.Sequential(
            nn.Linear(1, rate_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(rate_embed_dim, rate_embed_dim),
        )

        mid_ch = y_channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(y_channels + rate_embed_dim, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),   # alpha_map ∈ (0, 1)
        )

        nn.init.zeros_(self.net[2].bias)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=1e-4)

    def forward(
        self,
        x_diff: torch.Tensor,
        res: torch.Tensor,
        lq_latent_hat: torch.Tensor,
        lambda_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, h, w = lq_latent_hat.shape

        lam = torch.full((B, 1), lambda_rate,
                         dtype=lq_latent_hat.dtype, device=lq_latent_hat.device)
        r_emb = self.rate_embed(lam)                              # (B, rate_embed_dim)
        r_map = r_emb[:, :, None, None].expand(-1, -1, h, w)     # (B, rate_embed_dim, h, w)

        feat = torch.cat([lq_latent_hat, r_map], dim=1)          # (B, C+rate_embed_dim, h, w)
        alpha_map = self.net(feat)                                 # (B, 1, h, w)

        x_fused = 2 * (alpha_map * x_diff + (1.0 - alpha_map) * res)
        return x_fused, alpha_map


class StableCodec(torch.nn.Module):
    def __init__(self, sd_path=None, lmbda=0.5, config=None, logger=None, stage=None):
        super().__init__()
        self._structure_logger = logger
        self.stage = stage

        self.latent_tiled_size = config['latent_tiled_size']
        self.latent_tiled_overlap = config['latent_tiled_overlap']

        print("[SD-Turbo]: Building SD-Turbo ......")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched_cuda(sd_path)
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

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae

        # self.timesteps = torch.tensor([999], device="cuda").long()
        # timesteps = 1 * math.log(lmbda) + 900
        timesteps = config['timesteps']
        self.timesteps = torch.tensor([timesteps], device="cuda").long()

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

        print("[Training Setup]: Unfreezing mismatched layers ......")
        
        print("[Training Setup]: Done!")

        print("[Latent Codec]: Initializing Latent Codec ......")
        self.lambda_rate = lmbda          # 存为实例属性，forward/decompress 中使用
        self.codec = LatentCodec(lmbda)
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
        self.unet.conv_in = temp_layer
        print("[Latent Codec]: Done!")

        print("[Fusion]: Initializing ContentAdaptiveFusion ......")
        self.latent_channels = config['latent_channels']
        self.fusion = ContentAdaptiveFusion().cuda()
        print("[Fusion]: Done!")

        print("[Prompt]: Setting Prompt ......")
        self.set_prompt(config['pos_prompt'])
        del self.tokenizer, self.text_encoder
        print("[Prompt]: Done!")

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

            if "state_dict_fusion" in sd:
                self.fusion.load_state_dict(sd["state_dict_fusion"])
            print("[LoRA & Latent Codec & Auxiliary Decoder]: Done!")

        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(config['elic_path'])
        model.load_state_dict(checkpoint)
        self.aux_codec = model.g_a
        self.aux_codec.eval()
        self.aux_codec.requires_grad_(False)
        del model, checkpoint
        print("[Auxiliary Encoder]: Done!")

        print("[Auxiliary2 Codec]: Loading Pretrained Weights ......")
        model = DCAE(patch_size=config['patch_size'][0])
        checkpoint = torch.load(config['dcae_path'], map_location='cpu')
        state_dict = checkpoint["state_dict"]
        has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            state_dict = OrderedDict(
                (k[7:], v) for k, v in state_dict.items()
            )
        model.load_state_dict(state_dict)
        self.aux2_codec = model
        self.aux2_codec.eval()
        self.aux2_codec.requires_grad_(True)
        del model, checkpoint, state_dict
        print("[Auxiliary2 Codec]: Done!")

    # def unfreeze_mismatched_layers(self, model, loading_info, model_name="model"):
    #     mismatched_keys = loading_info.get("mismatched_keys", [])
        
    #     unfrozen_count = 0
    #     for name, param in model.named_parameters():
    #         if name in mismatched_keys:
    #             param.requires_grad = True
    #             unfrozen_count += 1
    #             print(f"  -> Unfrozen {model_name}: {name}")
        
    #     if unfrozen_count == 0:
    #         print(f"  -> No mismatched layers found for {model_name} (or keys didn't match).")
    def unfreeze_mismatched_layers(self, model, loading_info, model_name="model", logger=None):
        log = logger.info if logger is not None else print

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
                log(f"  -> Unfrozen {model_name}: {name}")

        if unfrozen_count == 0 and len(mismatched_key_set) > 0:
            log(f"  -> [{model_name}] WARNING: mismatched keys found but none matched named_parameters().")
            log(f"     Keys: {sorted(mismatched_key_set)}")
        elif unfrozen_count == 0:
            log(f"  -> No mismatched layers found for {model_name}.")

        # Log missing / unexpected keys so silent random-init bugs are visible.
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

    def set_prompt(self, pos_prompt):
        caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        self.pos_caption_enc = self.text_encoder(caption_tokens)[0]

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.codec.eval()
        self.fusion.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.fusion.requires_grad_(False)

    def set_train(self, stage=1):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.aux2_codec.train()
        self.fusion.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.aux2_codec.requires_grad_(False)
        self.fusion.requires_grad_(False)
        if stage == 1 or stage == 3:
            self.codec.requires_grad_(True)

            for n, _p in self.unet.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True
            self.unet.conv_in.requires_grad_(True)

            for n, _p in self.vae.named_parameters():
                if "lora" in n:
                    _p.requires_grad = True

            self.unfreeze_mismatched_layers(self.vae, self.vae_loading_info, "VAE", logger=self._structure_logger)
            self.unfreeze_mismatched_layers(self.unet, self.unet_loading_info, "UNet", logger=self._structure_logger)
        if stage == 2 or stage == 3:
            self.aux2_codec.requires_grad_(True)
        if stage == 3:
            self.fusion.requires_grad_(True)

    def forward(self, x, pos_prompt, ori_h, ori_w, stage):

        if stage == 1 or stage == 3:
            # Encoder
            with torch.no_grad():
                latent2 = self.aux_codec((x + 1) / 2).detach()
                pos_caption_enc = [self.pos_caption_enc for i in range(len(pos_prompt))]
                pos_caption_enc = torch.cat(pos_caption_enc, dim=0).to(x.device)
            lq_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor

            # Latent Codec
            lq_latent_hat, RateLossOutput, res1 = self.codec(lq_latent, latent2, ori_h, ori_w)

        # diffusion codec
        if stage == 1 or stage == 3:
            # One-Step Denoiser
            model_pred = self.unet(lq_latent_hat, self.timesteps, encoder_hidden_states=pos_caption_enc).sample
            x_diff = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :self.latent_channels], return_dict=True).prev_sample + res1

        # aux2_codec
        res2 = None
        if stage == 2 or stage == 3:
            res2 = self.aux2_codec(x)

        if stage == 1:
            x_denoised = x_diff
        elif stage == 2:
            x_denoised = res2
        elif stage == 3:
            # Content-Adaptive Fusion
            # alpha_map → 1: 扩散分支（低码率/高失真区域）
            # alpha_map → 0: 残差分支（高码率/低失真区域）
            x_denoised, _ = self.fusion(x_diff, res2, lq_latent_hat, self.lambda_rate)

        # Decoder
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

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

        x_diff = self.sched.step(model_pred, self.timesteps, lq_latent_hat[:, :self.latent_channels], return_dict=True).prev_sample

        x_denoised, _ = self.fusion(x_diff, res, lq_latent_hat, self.lambda_rate)

        # Decoder
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image
    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        sd["state_dict_fusion"] = {k: v for k, v in self.fusion.state_dict().items()}
        torch.save(sd, outf)
    
    def _set_latent_tile(self, latent_tiled_size = 96, latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap
    
    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
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
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))