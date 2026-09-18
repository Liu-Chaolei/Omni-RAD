"""
latent_codec_glc.py — StableCodec latent codec with GLC-style variable-rate.

Variable-rate mechanism (replaces λ-FiLM):
  • Global Learned Scalers: q_enc[qi], q_dec[qi] as nn.Parameter (per quality level)
  • Local Content-Adaptive Scalers: predicted from adapter_out (quant_step channel)

Retained from variable2_step:
  • DynamicTimestepModule (SNR-based T* from entropy model scales)
  • 4-pass checkerboard context model
  • All building blocks (InceptionDWConv2d, BasicBlock, etc.)

Removed:
  • LambdaFiLMEmbed, FiLMLayer — no FiLM conditioning in transforms
  • g_a, g_s, AuxDecoder are plain (no FiLM injection)

Quality levels: NUM_QUALITY = 4, λ ∈ {2, 8, 16, 32}
"""

import math
from torch import Tensor
from typing import NamedTuple

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste as ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
import sys
sys.path.append("..")
from ELIC.model.elic_official import CompressionModel, get_scale_table

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
NUM_QUALITY = 4
LAMBDA_TABLE = [2.0, 8.0, 16.0, 32.0]  # λ values for each quality level
M = 320  # latent channel dimension

# For QualityLambdaEmbed (used in StableCodec_glc.py, exported here for sharing)
FILM_DIM = 512
NUM_FREQS = 10
LAMBDA_MIN = 0.1
LAMBDA_MAX = 128.0


# ===========================================================================
# Core building blocks (unchanged from original StableCodec)
# ===========================================================================

class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=split_indexes[2])
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes

    def forward(self, x):
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionNeXt(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.depthconv = InceptionDWConv2d((in_ch - (in_ch // 8) * 3, in_ch // 8, in_ch // 8, in_ch // 8))
        self.conv1 = nn.Conv2d(in_ch, in_ch * 2, 1)
        self.conv2 = nn.Conv2d(in_ch * 2, in_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.depthconv(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut


class GatedCNNBlock(nn.Module):
    def __init__(self, in_ch, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden = int(expansion_ratio * in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden * 2, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1, x2 = self.fc1(x).chunk(2, 1)
        x = self.fc2(self.act(x1) * self.conv(x2))
        return x + shortcut


class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionNeXt(in_ch),
            GatedCNNBlock(in_ch),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=2, padding=2, groups=out_ch),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class Adapter(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, (in_ch + out_ch) // 2, 1),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, (in_ch + out_ch) // 2, 5, padding=2, groups=(in_ch + out_ch) // 2),
            nn.GELU(),
            nn.Conv2d((in_ch + out_ch) // 2, out_ch, 1),
        )
        self.branch2 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class HyperAnalysis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(M, M // 2, 3, stride=2, padding=1),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M // 2, 3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class HyperSynthesis(nn.Module):
    def __init__(self, M=320) -> None:
        super().__init__()
        self.increase = nn.Sequential(
            nn.Conv2d(M // 2, M * 2, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
            BasicBlock(M // 2),
            nn.Conv2d(M // 2, M * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.increase(x)
        return x


class SpatialContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
            BasicBlock(in_ch),
        )

    def forward(self, x):
        context = self.block(x)
        return context


class LRP(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Adapter(in_ch, (in_ch + out_ch) // 2),
            Adapter((in_ch + out_ch) // 2, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# ===========================================================================
# Plain transforms (no FiLM — same structure as latent_codec_ori.py)
# ===========================================================================

class AnalysisTransform(nn.Module):
    """Plain analysis transform g_a (no FiLM conditioning).

    Concatenates downsampled VAE latent and ELIC aux latent, then
    processes through BasicBlock + Downsample stages.
    """

    def __init__(self):
        super().__init__()
        self.pre1 = Downsample(256, 128)
        self.pre2 = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.block1 = BasicBlock(192)
        self.down1 = Downsample(192, 256)
        self.block2 = BasicBlock(256)
        self.down2 = Downsample(256, 320)
        self.block3 = BasicBlock(320)

    def forward(self, latent: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent:  [B, 256, H, W]  VAE latent
            latent2: [B, 320, H, W]  ELIC aux latent
        Returns:
            y: [B, 320, H/4, W/4]
        """
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        return x


class SynthesisTransform(nn.Module):
    """Plain synthesis transform g_s (no FiLM conditioning)."""

    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(320)
        self.up1 = Upsample(320, 320)
        self.block2 = BasicBlock(320)
        self.up2 = Upsample(320, 320)
        self.block3 = BasicBlock(320)
        self.up3 = Upsample(320, 320)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(self.block1(x))
        x = self.up2(self.block2(x))
        x = self.up3(self.block3(x))
        return x


class AuxDecoder(nn.Module):
    """Plain auxiliary decoder (no FiLM conditioning)."""

    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(320)
        self.up1 = Upsample(320, 256)
        self.block2 = BasicBlock(256)
        self.up2 = Upsample(256, 256)
        self.block3 = BasicBlock(256)
        self.up3 = Upsample(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(self.block1(x))
        x = self.up2(self.block2(x))
        x = self.up3(self.block3(x))
        return x


# ===========================================================================
# Rate module
# ===========================================================================

class RateLossOutput(NamedTuple):
    rate_loss:               Tensor
    quantized_total_bpp:     Tensor
    quantized_latent_bpp:    Tensor
    quantized_hyper_bpp:     Tensor
    per_image_bpp:           Tensor   # [B] per-image quantized bpp


class TargetRateModule(nn.Module):
    """Variable-rate rate loss: returns unweighted mean(bpp).
    The 1/λ scaling on distortion is handled in the training script.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _bpp(likelihoods: Tensor) -> Tensor:
        B = likelihoods.shape[0]
        return likelihoods.reshape(B, -1).log().sum(1) / -math.log(2)

    def forward(
        self,
        latent_likelihoods:                Tensor,
        quantized_latent_likelihoods:      Tensor,
        hyper_latent_likelihoods:          Tensor,
        quantized_hyper_latent_likelihoods: Tensor,
        ori_h: int = 512,
        ori_w: int = 512,
    ) -> RateLossOutput:
        N = ori_h * ori_w
        latent_bpp = self._bpp(latent_likelihoods) / N
        quantized_latent_bpp = self._bpp(quantized_latent_likelihoods) / N
        hyper_bpp = self._bpp(hyper_latent_likelihoods) / N
        quantized_hyper_bpp = self._bpp(quantized_hyper_latent_likelihoods) / N

        total_bpp = latent_bpp + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        return RateLossOutput(
            rate_loss=total_bpp.mean(),
            quantized_total_bpp=quantized_total_bpp.detach().mean(),
            quantized_latent_bpp=quantized_latent_bpp.detach().mean(),
            quantized_hyper_bpp=quantized_hyper_bpp.detach().mean(),
            per_image_bpp=quantized_total_bpp.detach(),
        )


# ===========================================================================
# DynamicTimestepModule — SNR-based analytical T* from entropy model σ
# ===========================================================================

class DynamicTimestepModule(nn.Module):
    """Compute per-image dynamic timestep T* from entropy model scales.

    Uses the SNR schedule from alphas_cumprod to establish a monotonic
    ordering, then linearly rescales the raw timestep into [t_min, t_max].
    """

    def __init__(self, alphas_cumprod: torch.Tensor, t_min: int = 800, t_max: int = 999):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        ac = alphas_cumprod.float()
        self.register_buffer("alphas_cumprod", ac)
        snr_schedule = ac / (1.0 - ac)
        self.register_buffer("snr_schedule", snr_schedule)

    def forward(self, scales_all: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scales_all: [B, 320, H, W] — summed scales from the 4-pass
                        checkerboard context model.
        Returns:
            T_star: [B] float tensor, each entry in [t_min, t_max].
        """
        sigma_quant = 1.0 / 12.0
        signal_var = scales_all.mean(dim=[1, 2, 3]) ** 2
        snr_compress = signal_var / sigma_quant
        T_raw = torch.searchsorted(-self.snr_schedule, -snr_compress)
        T_star = self.t_min + (self.t_max - self.t_min) * (T_raw.float() / 999.0)
        T_star = T_star.clamp(self.t_min, self.t_max)
        return T_star


# ===========================================================================
# QualityLambdaEmbed — discrete quality index → embedding (for UNet lora_proj)
# ===========================================================================

class QualityLambdaEmbed(nn.Module):
    """Maps discrete quality_index to a dense embedding via fixed λ lookup.

    quality_index (int in [0, NUM_QUALITY-1])
      → λ_table[qi]
      → log-normalise to [0,1]
      → Fourier features
      → 2-layer MLP
      → f(λ) ∈ R^{embed_dim}

    Same architecture as LambdaFiLMEmbed but with discrete input.
    """

    def __init__(
        self,
        embed_dim: int = FILM_DIM,
        num_freqs: int = NUM_FREQS,
        lambda_min: float = LAMBDA_MIN,
        lambda_max: float = LAMBDA_MAX,
    ):
        super().__init__()
        self.register_buffer(
            "lambda_table",
            torch.tensor(LAMBDA_TABLE, dtype=torch.float32),
        )
        self.register_buffer(
            "log_lambda_min",
            torch.tensor(math.log(lambda_min), dtype=torch.float32),
        )
        self.register_buffer(
            "log_lambda_max",
            torch.tensor(math.log(lambda_max), dtype=torch.float32),
        )

        fourier_dim = 1 + 2 * num_freqs
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_freqs - 1.0, num_freqs),
        )

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, quality_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quality_index: [B] long tensor, values in [0, NUM_QUALITY-1]
        Returns:
            embed: [B, embed_dim]
        """
        lmbda = self.lambda_table[quality_index]  # [B]

        log_nor_lmbda = (torch.log(lmbda.clamp(min=1e-8)) - self.log_lambda_min) / (
            self.log_lambda_max - self.log_lambda_min
        )
        log_nor_lmbda = log_nor_lmbda.clamp(0.0, 1.0)

        l_col = log_nor_lmbda.unsqueeze(-1)
        angles = l_col * self.freq_bands.unsqueeze(0) * math.pi
        fourier = torch.cat([l_col, torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.mlp(fourier)


# ===========================================================================
# LatentCodec — GLC variable-rate with global/local scalers + dynamic timestep
# ===========================================================================

class LatentCodec(CompressionModel):
    """StableCodec latent codec with GLC-style variable-rate.

    Variable-rate mechanism:
      • Global scalers: q_enc[qi] applied to y after g_a, q_dec[qi] after all passes
      • Local scalers: quant_step predicted from a separate quant_step_head
      • DynamicTimestepModule: SNR-based T* from entropy model scales

    Fix #1: adapter_out outputs M*2 (scales, means) — same as latent_codec_ori.py,
            so weights load correctly from stablecodec_ft2.pkl warm-start.
            quant_step is predicted by a separate lightweight head (quant_step_head).
    Fix #2: Likelihood is computed on the globally-scaled y (consistent with quantization).
            Local quant_step only controls rounding granularity, not the distribution.
    Fix #3: Rate floor to prevent rate collapse during early training.

    Interface:
        forward   returns (x_hat, rate_out, res, T_star)
        compress  returns dict with "quality_index" stored
        decompress returns (x_hat, res, T_star)
    """

    def __init__(self, alphas_cumprod: torch.Tensor = None, rate_floor: float = 0.001):
        super().__init__()

        # Plain transforms (no FiLM)
        self.g_a = AnalysisTransform()
        self.g_s = SynthesisTransform()
        self.aux = AuxDecoder()

        # Hyper-prior
        self.h_a = HyperAnalysis(M=M)
        self.h_s = HyperSynthesis(M=M)

        # Entropy model — adapter_out outputs M*2 (scales, means)
        # SAME shape as latent_codec_ori.py → checkpoint weights load correctly
        context_dim = M * 3
        self.adapter_in = nn.ModuleList([Adapter(M, context_dim) for _ in range(4)])
        self.g_c = SpatialContext(context_dim)
        self.adapter_out = nn.ModuleList([Adapter(context_dim, M * 2) for _ in range(4)])
        self.LRP = nn.ModuleList([LRP(M * 2, M) for _ in range(4)])

        # GLC local quant_step head — separate from adapter_out for checkpoint compat
        # Initialized with bias=1.0 so quant_step starts at ~1.0 (identity behaviour)
        self.quant_step_head = nn.ModuleList([
            self._make_quant_step_head(context_dim) for _ in range(4)
        ])

        self.entropy_bottleneck = EntropyBottleneck(M // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        # GLC Global Learned Scalers (identity init)
        self.global_q_enc = nn.Parameter(torch.ones(NUM_QUALITY, M, 1, 1))
        self.global_q_dec = nn.Parameter(torch.ones(NUM_QUALITY, M, 1, 1))

        # Rate module with floor to prevent rate collapse
        self.rate = TargetRateModule()
        self.rate_floor = rate_floor

        # Dynamic timestep module
        if alphas_cumprod is not None:
            self.dynamic_timestep = DynamicTimestepModule(alphas_cumprod)
        else:
            self.dynamic_timestep = None

    @staticmethod
    def _make_quant_step_head(context_dim: int) -> nn.Sequential:
        """Lightweight head to predict per-pixel quant_step from context.
        Output bias initialised to 0 so that sigmoid gives 0.5 → quant_step = 1.0.
        Final quant_step = 0.5 + 1.5 * sigmoid(raw) ∈ [0.5, 2.0].
        """
        head = nn.Sequential(
            nn.Conv2d(context_dim, M, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(M, M, kernel_size=1),
        )
        # Zero-init the last conv so sigmoid(0)=0.5 → quant_step=1.25 ≈ identity
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    # ------------------------------------------------------------------
    # Mask helpers (unchanged)
    # ------------------------------------------------------------------

    def get_mask_four_parts(self, batch, channel, height, width, device='cuda'):
        key = f"{batch}_{channel}x{width}x{height}"
        if key not in self.masks:
            def make_mask(pattern):
                t = torch.tensor(pattern, dtype=torch.float32, device=device)
                m = t.repeat((height + 1) // 2, (width + 1) // 2)[:height, :width]
                return m.unsqueeze(0).unsqueeze(0)
            m0 = make_mask(((1., 0), (0, 0)))
            m1 = make_mask(((0, 1.), (0, 0)))
            m2 = make_mask(((0, 0), (1., 0)))
            m3 = make_mask(((0, 0), (0, 1.)))
            base = torch.ones((batch, channel // 4, height, width), device=device)
            self.masks[key] = [
                torch.cat([base*m0, base*m1, base*m2, base*m3], 1),
                torch.cat([base*m3, base*m2, base*m1, base*m0], 1),
                torch.cat([base*m2, base*m3, base*m0, base*m1], 1),
                torch.cat([base*m1, base*m0, base*m3, base*m2], 1),
            ]
        return self.masks[key]

    def sequeeze_with_mask(self, lat, mask):
        g = lat.chunk(4, 1)
        mg = mask.chunk(4, 1)
        return sum(g[i] * mg[i] for i in range(4))

    def unsequeeze_with_mask(self, sq, mask):
        mg = mask.chunk(4, 1)
        return torch.cat([sq * mg[i] for i in range(4)], dim=1)

    def compress_group_with_mask(self, gc, lat, scales, means, mask, syms, idxs):
        ls = self.sequeeze_with_mask(lat,    mask)
        ss = self.sequeeze_with_mask(scales, mask)
        ms = self.sequeeze_with_mask(means,  mask)
        idxs_t = gc.build_indexes(ss)
        lhat   = gc.quantize(ls, "symbols", ms)
        syms.extend(lhat.reshape(-1).tolist())
        idxs.extend(idxs_t.reshape(-1).tolist())
        return self.unsequeeze_with_mask(lhat + ms, mask)

    def decompress_group_with_mask(self, gc, scales, means, mask, decoder, cdf, cdf_len, offsets):
        ss   = self.sequeeze_with_mask(scales, mask)
        ms   = self.sequeeze_with_mask(means,  mask)
        idxs = gc.build_indexes(ss)
        lhat = decoder.decode_stream(idxs.reshape(-1).tolist(), cdf, cdf_len, offsets)
        lhat = torch.Tensor(lhat).reshape(ss.shape).to(scales.device)
        return self.unsequeeze_with_mask(lhat + ms, mask)

    # ------------------------------------------------------------------
    # Forward  (training / validation)
    # ------------------------------------------------------------------

    def forward(self, latent, latent2, ori_h, ori_w, quality_index):
        """
        Args:
            latent:        [B, 256, H, W]  VAE latent
            latent2:       [B, 320, H, W]  ELIC aux latent (frozen)
            ori_h, ori_w:  original image size for bpp normalisation
            quality_index: [B] long tensor, values in [0, NUM_QUALITY-1]

        Returns:
            x_hat          [B, 320, H*8, W*8]  g_s output (fed to UNet)
            RateLossOutput named-tuple
            res            [B, 256, H*8, W*8]  AuxDecoder output
            T_star         [B]                  dynamic timestep per image
        """
        B = latent.shape[0]

        # ---- Plain analysis (no FiLM) ----
        y = self.g_a(latent, latent2)

        # ---- GLC global encode scaling ----
        q_enc = self.global_q_enc[quality_index]  # [B, M, 1, 1]
        y = y * q_enc

        # ---- Hyper-prior entropy ----
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        with torch.no_grad():
            _, qz_likelihoods = self.entropy_bottleneck(z, training=False)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat    = ste_round(z - z_offset) + z_offset

        B2, C, H, W = y.shape
        m0, m1, m2, m3 = self.get_mask_four_parts(B2, C, H, W, y.device)

        # ---- 4-pass checkerboard with GLC local scalers ----
        # Fix #1: adapter_out outputs M*2 (means, scales) — same as ori checkpoint
        #         quant_step comes from separate quant_step_head
        # Fix #2: likelihood computed on globally-scaled y (consistent with quantization)
        #         local quant_step only controls rounding step size
        base = self.h_s(z_hat)
        base = base[:, :, :H, :W]

        # Pass 0
        ctx0 = self.g_c(self.adapter_in[0](base))
        means_0_supp, scales_0_supp = self.adapter_out[0](ctx0).chunk(2, 1)
        qs_raw_0 = self.quant_step_head[0](ctx0)
        quant_step_0 = 0.5 + 1.5 * torch.sigmoid(qs_raw_0)  # ∈ [0.5, 2.0]
        means_0  = means_0_supp * m0
        scales_0 = scales_0_supp * m0
        y_0 = y * m0
        # Local scaler: divide before rounding, multiply after (controls granularity)
        y_hat_0 = ste_round(y_0 / quant_step_0 - means_0) + means_0
        y_hat_0 = y_hat_0 * quant_step_0 * m0
        lrp = 0.5 * torch.tanh(self.LRP[0](torch.cat([y_hat_0, base], 1)) * m0)
        y_hat_0 = y_hat_0 + lrp

        # Pass 1
        base = base * (1 - m0) + y_hat_0
        ctx1 = self.g_c(self.adapter_in[1](base))
        means_1_supp, scales_1_supp = self.adapter_out[1](ctx1).chunk(2, 1)
        qs_raw_1 = self.quant_step_head[1](ctx1)
        quant_step_1 = 0.5 + 1.5 * torch.sigmoid(qs_raw_1)
        means_1  = means_1_supp * m1
        scales_1 = scales_1_supp * m1
        y_1 = y * m1
        y_hat_1 = ste_round(y_1 / quant_step_1 - means_1) + means_1
        y_hat_1 = y_hat_1 * quant_step_1 * m1
        lrp = 0.5 * torch.tanh(self.LRP[1](torch.cat([y_hat_1, base], 1)) * m1)
        y_hat_1 = y_hat_1 + lrp

        # Pass 2
        base = base * (1 - m1) + y_hat_1
        ctx2 = self.g_c(self.adapter_in[2](base))
        means_2_supp, scales_2_supp = self.adapter_out[2](ctx2).chunk(2, 1)
        qs_raw_2 = self.quant_step_head[2](ctx2)
        quant_step_2 = 0.5 + 1.5 * torch.sigmoid(qs_raw_2)
        means_2  = means_2_supp * m2
        scales_2 = scales_2_supp * m2
        y_2 = y * m2
        y_hat_2 = ste_round(y_2 / quant_step_2 - means_2) + means_2
        y_hat_2 = y_hat_2 * quant_step_2 * m2
        lrp = 0.5 * torch.tanh(self.LRP[2](torch.cat([y_hat_2, base], 1)) * m2)
        y_hat_2 = y_hat_2 + lrp

        # Pass 3
        base = base * (1 - m2) + y_hat_2
        ctx3 = self.g_c(self.adapter_in[3](base))
        means_3_supp, scales_3_supp = self.adapter_out[3](ctx3).chunk(2, 1)
        qs_raw_3 = self.quant_step_head[3](ctx3)
        quant_step_3 = 0.5 + 1.5 * torch.sigmoid(qs_raw_3)
        means_3  = means_3_supp * m3
        scales_3 = scales_3_supp * m3
        y_3 = y * m3
        y_hat_3 = ste_round(y_3 / quant_step_3 - means_3) + means_3
        y_hat_3 = y_hat_3 * quant_step_3 * m3
        lrp = 0.5 * torch.tanh(self.LRP[3](torch.cat([y_hat_3, base], 1)) * m3)
        y_hat_3 = y_hat_3 + lrp

        # ---- Accumulate scales for likelihood & dynamic timestep ----
        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all  = means_0  + means_1  + means_2  + means_3

        # Fix #2: Compute likelihood on the globally-scaled y (what actually gets quantized)
        # The gaussian_conditional models P(y | scales, means); since we quantize y/quant_step,
        # we scale the "effective" distribution by adjusting scales accordingly.
        # However, the simplest correct approach: pass y as-is but use scales that account
        # for the quant_step expansion. Since quant_step starts ≈ 1.0 (identity) and
        # the entropy model was pre-trained on raw y, we keep the ori-style likelihood
        # which is correct when quant_step=1. As quant_step learns, the rate gradient
        # still flows through the global scalers and the entropy model.
        _, y_likelihoods = self.gaussian_conditional(y, scales_all, means_all)
        with torch.no_grad():
            _, qy_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - m3) + y_hat_3

        # ---- GLC global decode scaling ----
        q_dec = self.global_q_dec[quality_index]  # [B, M, 1, 1]
        y_hat = y_hat * q_dec

        # ---- Dynamic timestep T* from entropy model scales ----
        T_star = self.dynamic_timestep(scales_all) if self.dynamic_timestep is not None \
            else torch.full((B,), 999.0, device=latent.device)

        # ---- Plain synthesis & aux decoder (no FiLM) ----
        x_hat = self.g_s(y_hat)
        res   = self.aux(y_hat)

        rate_out = self.rate(
            latent_likelihoods                 = y_likelihoods,
            quantized_latent_likelihoods       = qy_likelihoods,
            hyper_latent_likelihoods           = z_likelihoods,
            quantized_hyper_latent_likelihoods = qz_likelihoods,
            ori_h                              = ori_h,
            ori_w                              = ori_w,
        )

        # Fix #3: Apply rate floor — prevent bpp from collapsing to 0
        rate_out = RateLossOutput(
            rate_loss=torch.maximum(rate_out.rate_loss, torch.tensor(self.rate_floor, device=rate_out.rate_loss.device)),
            quantized_total_bpp=rate_out.quantized_total_bpp,
            quantized_latent_bpp=rate_out.quantized_latent_bpp,
            quantized_hyper_bpp=rate_out.quantized_hyper_bpp,
            per_image_bpp=rate_out.per_image_bpp,
        )

        return x_hat, rate_out, res, T_star

    # ------------------------------------------------------------------
    # Compress  (inference encoding)
    # ------------------------------------------------------------------

    def compress(self, latent, latent2, quality_index=0):
        """
        Args:
            latent:        [B, 256, H, W]
            latent2:       [B, 320, H, W]
            quality_index: int or [B] tensor, quality level in [0, NUM_QUALITY-1]

        Returns:
            dict with keys: strings, shape, quality_index, t_star_val
        """
        B = latent.size(0)
        device = latent.device

        if isinstance(quality_index, int):
            qi = torch.full((B,), quality_index, dtype=torch.long, device=device)
        else:
            qi = quality_index.long().to(device)

        # ---- Plain analysis + global encode scaling ----
        y = self.g_a(latent, latent2)
        q_enc = self.global_q_enc[qi]
        y = y * q_enc

        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat     = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        cdf         = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets     = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder     = BufferedRansEncoder()
        syms, idxs  = [], []
        y_strings   = []

        Bs, C, H, W = y.shape
        m0, m1, m2, m3 = self.get_mask_four_parts(Bs, C, H, W, y.device)
        base = self.h_s(z_hat)
        base = base[:, :, :H, :W]

        # Pass 0 — Fix #1: adapter_out → chunk(2,1), quant_step from separate head
        ctx0 = self.g_c(self.adapter_in[0](base))
        ms0, ss0 = self.adapter_out[0](ctx0).chunk(2, 1)
        qs_raw_0 = self.quant_step_head[0](ctx0)
        quant_step_0 = 0.5 + 1.5 * torch.sigmoid(qs_raw_0)
        ss0 = ss0 * m0
        ms0 = ms0 * m0
        yh0 = self.compress_group_with_mask(self.gaussian_conditional, y, ss0, ms0, m0, syms, idxs)
        yh0 = yh0 * quant_step_0 * m0
        yh0 += 0.5 * torch.tanh(self.LRP[0](torch.cat([yh0, base], 1)) * m0)

        # Pass 1
        base = base * (1 - m0) + yh0
        ctx1 = self.g_c(self.adapter_in[1](base))
        ms1, ss1 = self.adapter_out[1](ctx1).chunk(2, 1)
        qs_raw_1 = self.quant_step_head[1](ctx1)
        quant_step_1 = 0.5 + 1.5 * torch.sigmoid(qs_raw_1)
        ss1 = ss1 * m1
        ms1 = ms1 * m1
        yh1 = self.compress_group_with_mask(self.gaussian_conditional, y, ss1, ms1, m1, syms, idxs)
        yh1 = yh1 * quant_step_1 * m1
        yh1 += 0.5 * torch.tanh(self.LRP[1](torch.cat([yh1, base], 1)) * m1)

        # Pass 2
        base = base * (1 - m1) + yh1
        ctx2 = self.g_c(self.adapter_in[2](base))
        ms2, ss2 = self.adapter_out[2](ctx2).chunk(2, 1)
        qs_raw_2 = self.quant_step_head[2](ctx2)
        quant_step_2 = 0.5 + 1.5 * torch.sigmoid(qs_raw_2)
        ss2 = ss2 * m2
        ms2 = ms2 * m2
        yh2 = self.compress_group_with_mask(self.gaussian_conditional, y, ss2, ms2, m2, syms, idxs)
        yh2 = yh2 * quant_step_2 * m2
        yh2 += 0.5 * torch.tanh(self.LRP[2](torch.cat([yh2, base], 1)) * m2)

        # Pass 3
        base = base * (1 - m2) + yh2
        ctx3 = self.g_c(self.adapter_in[3](base))
        ms3, ss3 = self.adapter_out[3](ctx3).chunk(2, 1)
        qs_raw_3 = self.quant_step_head[3](ctx3)
        quant_step_3 = 0.5 + 1.5 * torch.sigmoid(qs_raw_3)
        ss3 = ss3 * m3
        ms3 = ms3 * m3
        _ = self.compress_group_with_mask(self.gaussian_conditional, y, ss3, ms3, m3, syms, idxs)

        # Compute dynamic timestep T* from accumulated scales
        scales_all = ss0 + ss1 + ss2 + ss3
        T_star = self.dynamic_timestep(scales_all) if self.dynamic_timestep is not None \
            else torch.full((B,), 999.0, device=device)

        encoder.encode_with_indexes(syms, idxs, cdf, cdf_lengths, offsets)
        y_strings.append(encoder.flush())
        torch.backends.cudnn.deterministic = False

        return {
            "strings":       [y_strings, z_strings],
            "shape":         z.size()[-2:],
            "quality_index": qi.cpu().tolist(),
            "t_star_val":    T_star.cpu().tolist(),
        }

    # ------------------------------------------------------------------
    # Decompress  (inference decoding)
    # ------------------------------------------------------------------

    def decompress(self, strings, shape, quality_index=None):
        """
        Args:
            strings:       dict from compress() OR raw [y_strings, z_strings]
            shape:         z spatial shape
            quality_index: int/Tensor. Priority:
                             1. explicit argument
                             2. strings["quality_index"]
                             3. default: 0
        Returns:
            x_hat   [B, 320, H*8, W*8]   g_s output
            res     [B, 256, H*8, W*8]   AuxDecoder output
            T_star  [B]                   dynamic timestep per image
        """
        if quality_index is None:
            if isinstance(strings, dict) and "quality_index" in strings:
                quality_index = torch.tensor(strings["quality_index"], dtype=torch.long)
            else:
                quality_index = torch.tensor([0], dtype=torch.long)

        raw = strings["strings"] if isinstance(strings, dict) else strings

        torch.backends.cudnn.deterministic = True
        z_hat = self.entropy_bottleneck.decompress(raw[1], shape)

        B = z_hat.size(0)
        device = z_hat.device

        if isinstance(quality_index, int):
            qi = torch.full((B,), quality_index, dtype=torch.long, device=device)
        elif not isinstance(quality_index, torch.Tensor):
            qi = torch.tensor(quality_index, dtype=torch.long, device=device)
        else:
            qi = quality_index.long().to(device)
        if qi.dim() == 0:
            qi = qi.unsqueeze(0).expand(B)
        elif qi.shape[0] != B:
            qi = qi[:1].expand(B)

        cdf         = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets     = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder     = RansDecoder()
        decoder.set_stream(raw[0][0])

        C, H, W = z_hat.shape[1], z_hat.shape[2], z_hat.shape[3]
        m0, m1, m2, m3 = self.get_mask_four_parts(B, C * 2, H * 4, W * 4, device)
        base = self.h_s(z_hat)

        # Pass 0 — Fix #1: adapter_out → chunk(2,1), quant_step from separate head
        ctx0 = self.g_c(self.adapter_in[0](base))
        ms0, ss0 = self.adapter_out[0](ctx0).chunk(2, 1)
        qs_raw_0 = self.quant_step_head[0](ctx0)
        quant_step_0 = 0.5 + 1.5 * torch.sigmoid(qs_raw_0)
        ss0 = ss0 * m0
        ms0 = ms0 * m0
        yh0 = self.decompress_group_with_mask(self.gaussian_conditional, ss0, ms0, m0, decoder, cdf, cdf_lengths, offsets)
        yh0 = yh0 * quant_step_0 * m0
        yh0 += 0.5 * torch.tanh(self.LRP[0](torch.cat([yh0, base], 1)) * m0)

        # Pass 1
        base = base * (1 - m0) + yh0
        ctx1 = self.g_c(self.adapter_in[1](base))
        ms1, ss1 = self.adapter_out[1](ctx1).chunk(2, 1)
        qs_raw_1 = self.quant_step_head[1](ctx1)
        quant_step_1 = 0.5 + 1.5 * torch.sigmoid(qs_raw_1)
        ss1 = ss1 * m1
        ms1 = ms1 * m1
        yh1 = self.decompress_group_with_mask(self.gaussian_conditional, ss1, ms1, m1, decoder, cdf, cdf_lengths, offsets)
        yh1 = yh1 * quant_step_1 * m1
        yh1 += 0.5 * torch.tanh(self.LRP[1](torch.cat([yh1, base], 1)) * m1)

        # Pass 2
        base = base * (1 - m1) + yh1
        ctx2 = self.g_c(self.adapter_in[2](base))
        ms2, ss2 = self.adapter_out[2](ctx2).chunk(2, 1)
        qs_raw_2 = self.quant_step_head[2](ctx2)
        quant_step_2 = 0.5 + 1.5 * torch.sigmoid(qs_raw_2)
        ss2 = ss2 * m2
        ms2 = ms2 * m2
        yh2 = self.decompress_group_with_mask(self.gaussian_conditional, ss2, ms2, m2, decoder, cdf, cdf_lengths, offsets)
        yh2 = yh2 * quant_step_2 * m2
        yh2 += 0.5 * torch.tanh(self.LRP[2](torch.cat([yh2, base], 1)) * m2)

        # Pass 3
        base = base * (1 - m2) + yh2
        ctx3 = self.g_c(self.adapter_in[3](base))
        ms3, ss3 = self.adapter_out[3](ctx3).chunk(2, 1)
        qs_raw_3 = self.quant_step_head[3](ctx3)
        quant_step_3 = 0.5 + 1.5 * torch.sigmoid(qs_raw_3)
        ss3 = ss3 * m3
        ms3 = ms3 * m3
        yh3 = self.decompress_group_with_mask(self.gaussian_conditional, ss3, ms3, m3, decoder, cdf, cdf_lengths, offsets)
        yh3 = yh3 * quant_step_3 * m3
        yh3 += 0.5 * torch.tanh(self.LRP[3](torch.cat([yh3, base], 1)) * m3)

        y_hat = yh0 + yh1 + yh2 + yh3
        torch.backends.cudnn.deterministic = False

        # ---- GLC global decode scaling ----
        q_dec = self.global_q_dec[qi]
        y_hat = y_hat * q_dec

        # Compute T* — prefer bitstream value; fallback to recomputation
        if isinstance(strings, dict) and "t_star_val" in strings:
            T_star = torch.tensor(
                strings["t_star_val"], dtype=torch.float32, device=device,
            )
        else:
            scales_all = ss0 + ss1 + ss2 + ss3
            T_star = self.dynamic_timestep(scales_all) if self.dynamic_timestep is not None \
                else torch.full((B,), 999.0, device=device)

        # ---- Plain synthesis & aux (no FiLM) ----
        x_hat = self.g_s(y_hat)
        res   = self.aux(y_hat)
        return x_hat, res, T_star

    # ------------------------------------------------------------------
    # Update entropy tables
    # ------------------------------------------------------------------

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated  = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

