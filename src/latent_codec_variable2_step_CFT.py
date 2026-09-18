"""
latent_codec_variable2_step_CFT.py  —  CFT variant of latent codec.

CFT changes vs. A0 (latent_codec_variable2_step.py):
  • forward()    returns 5-tuple: (x_hat, rate_out, res, T_star, scales_all)
  • decompress() returns 4-tuple: (x_hat, res, T_star, scales_all)
  • compress()   bitstream payload UNCHANGED (scales_all is reconstructible
                 from the entropy decode at decompress time, so no extra cost)

The exposed `scales_all` (B, 320, H/32, W/32) is consumed by the model-level
ScaleFeatureAdapter + CFTFusionModule in StableCodec_variable2_step_CFT.py.

Original A0 docstring follows.

------------------------------------------------------------------------------
latent_codec_variable.py  —  StableCodec latent codec with variable-rate λ-FiLM.

Variable-rate injection points:
  1. AnalysisTransform  (ga) : FiLM after every BasicBlock          [Injection 1]
  2. SynthesisTransform (gs) : FiLM after every BasicBlock          [Injection 2]
  3. ϵSD UNet (LoRA scale)   : handled in StableCodec_variable.py   [Injection 3]
  4. AuxDecoder        (DAux): FiLM after every BasicBlock          [Injection 4]

λ embedding:  scalar λ → log-normalise → Fourier features → 2-layer MLP → f(λ) ∈ R^512
              (FourierCond, MRIC / Agustsson et al. CVPR 2023, ported to PyTorch)

FiLM modulation:  h' = h ⊙ γ_i(f(λ)) + β_i(f(λ))
              Initialised as identity (γ→1, β→0), so training starts from the
              unmodified baseline.

              Rationale vs. baselines:
                • MRIC:  purely additive  h' = h + W_i·f(λ)  (no γ scaling).
                • I2C:   h' = (h ⊙ β) ⊕ γ  but requires strict invertibility.
                • FiLM:  affine modulation without invertibility constraint —
                  natural fit for StableCodec's non-invertible transforms.

Training strategy:
  • lmbda sampled log-uniformly per image per batch  → TargetRateModule receives
    a per-image [B] tensor, enabling continuous RD-curve coverage.
  • Two-stage IBP: Stage I uses small λ_base; Stage II uses larger λ_target values.
  • lmbda stored as float16 (2 bytes) in the compressed bitstream — from I2C's
    design, giving 2^16 = 65 536 effective variable-rate points.
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
# Global hyper-parameters for FiLM conditioning
# ---------------------------------------------------------------------------
FILM_DIM   = 512    # shared embedding dimension
NUM_FREQS  = 10     # Fourier frequencies  →  fourier_dim = 1 + 2*10 = 21
LAMBDA_MIN = 0.1    # conservative lower bound — wider than any expected lambda_min
LAMBDA_MAX = 128.0  # conservative upper bound — wider than any expected lambda_max
# NOTE: These constants serve only as fallback defaults when lambda_min/lambda_max are
# not supplied explicitly.  Always set model.lambda_min / model.lambda_max in the YAML
# config so the actual training range is reflected exactly in LambdaFiLMEmbed.
# The bounds should be at least as wide as [lambda_min_train, lambda_max_train];
# making them slightly wider avoids clamp saturation at the exact boundary values.


# ===========================================================================
# λ-FiLM conditioning modules
# ===========================================================================

class LambdaFiLMEmbed(nn.Module):
    """Converts scalar/batch λ to a dense embedding  f(λ) ∈ R^{embed_dim}.

    Implements FourierCond from MRIC (Agustsson et al., CVPR 2023) in PyTorch:
        λ  →  log-normalise to [0,1]  →  Fourier features  →  2-layer MLP

    The embedding is SHARED across all injection points; each injection site
    learns its own γ/β linear projection (see FiLMLayer).
    """

    def __init__(
        self,
        embed_dim:  int   = FILM_DIM,
        num_freqs:  int   = NUM_FREQS,
        lambda_min: float = LAMBDA_MIN,
        lambda_max: float = LAMBDA_MAX,
    ):
        super().__init__()
        # Register as buffers so they are saved in state_dict and survive resume.
        # If a checkpoint was saved with different lambda bounds, loading will
        # restore the original bounds — preventing silent range mismatch on resume.
        self.register_buffer(
            "log_lambda_min",
            torch.tensor(math.log(lambda_min), dtype=torch.float32),
        )
        self.register_buffer(
            "log_lambda_max",
            torch.tensor(math.log(lambda_max), dtype=torch.float32),
        )

        fourier_dim = 1 + 2 * num_freqs          # raw l + sin/cos pairs
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_freqs - 1.0, num_freqs),
        )

        # 2-layer MLP matching MRIC's BetaMlp (Dense + ReLU, twice)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, lmbda: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lmbda: Tensor [B] or scalar float — raw λ values.
        Returns:
            f_lmbda: Tensor [B, embed_dim]
        """
        if not isinstance(lmbda, torch.Tensor):
            lmbda = torch.tensor(lmbda, dtype=torch.float32)
        lmbda = lmbda.float()
        if lmbda.dim() == 0:
            lmbda = lmbda.unsqueeze(0)                     # [1]

        # Log-normalise λ to [0, 1]
        # log_lambda_min/max are registered buffers (0-dim tensors) that move
        # with the model device automatically.
        log_nor_lmbda = (torch.log(lmbda.clamp(min=1e-8)) - self.log_lambda_min) / (
            self.log_lambda_max - self.log_lambda_min
        )
        # No clamping needed when lmbda is within [lambda_min, lambda_max].
        # A soft warn-only clamp with a small epsilon avoids hard saturation
        # if lmbda accidentally falls slightly outside the configured range.
        log_nor_lmbda = log_nor_lmbda.clamp(0.0, 1.0)                              # [B]

        # Fourier embedding: [B, fourier_dim]
        l_col  = log_nor_lmbda.unsqueeze(-1)                           # [B, 1]
        freqs  = self.freq_bands.to(lmbda.device)          # [num_freqs]
        angles = l_col * freqs.unsqueeze(0) * math.pi      # [B, num_freqs]
        fourier = torch.cat(
            [l_col, torch.sin(angles), torch.cos(angles)], dim=-1
        )                                                  # [B, fourier_dim]

        return self.mlp(fourier)                           # [B, embed_dim]


class FiLMLayer(nn.Module):
    """Per-injection-site Feature-wise Linear Modulation.

    Applies  h' = h ⊙ γ(f(λ)) + β(f(λ))  where γ and β are produced by
    independent linear projections of the shared embedding f(λ).

    Weights initialised for identity at start of training:
        γ_proj: weight=0, bias=1  → γ=1 always at init
        β_proj: weight=0, bias=0  → β=0 always at init
    """

    def __init__(self, embed_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(embed_dim, feat_dim)
        self.beta_proj  = nn.Linear(embed_dim, feat_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_ (self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, film_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:          [B, C, H, W]
            film_embed: [B, embed_dim]
        Returns:
            [B, C, H, W]  modulated feature
        """
        B = h.size(0)
        gamma = self.gamma_proj(film_embed).view(B, -1, 1, 1)  # [B, C, 1, 1]
        beta  = self.beta_proj (film_embed).view(B, -1, 1, 1)
        return h * gamma + beta


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
# FiLM-enabled transforms  (Injection Points 1, 2, 4)
# ===========================================================================

class AnalysisTransform(nn.Module):
    """Analysis transform  ga  with λ-FiLM conditioning  [Injection Point 1].

    ESD and EAux outputs are concat-ed BEFORE entering ga (original design).
    No cross-encoder weighting is introduced — EAux remains frozen and its
    contribution is preserved intact.  λ is injected purely through FiLM on
    the analysis stack.

    FiLM sites (channel dims):
        BasicBlock(192) → FiLM(192)
        Downsample(192→256)
        BasicBlock(256) → FiLM(256)
        Downsample(256→320)
        BasicBlock(320) → FiLM(320)
    """

    def __init__(self, embed_dim: int = FILM_DIM):
        super().__init__()
        # Pre-processing adapters (same as original AnalysisTransform32)
        self.pre1 = Downsample(256, 128)
        self.pre2 = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        # Analysis stack — unrolled to interleave FiLM layers
        self.block1 = BasicBlock(192)
        self.film1 = FiLMLayer(embed_dim, 192)
        self.down1  = Downsample(192, 256)
        self.block2 = BasicBlock(256)
        self.film2 = FiLMLayer(embed_dim, 256)
        self.down2  = Downsample(256, 320)
        self.block3 = BasicBlock(320)
        self.film3 = FiLMLayer(embed_dim, 320)

    def forward(
        self,
        latent:     torch.Tensor,   # [B, 256, H,   W  ]  ESD VAE latent
        latent2:    torch.Tensor,   # [B, 320, H,   W  ]  EAux semantic latent
        film_embed: torch.Tensor,   # [B, embed_dim]
    ) -> torch.Tensor:
        x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1)  # [B,192,…]
        x = self.film1(self.block1(x), film_embed)
        x = self.down1(x)
        x = self.film2(self.block2(x), film_embed)
        x = self.down2(x)
        x = self.film3(self.block3(x), film_embed)
        return x


class SynthesisTransform(nn.Module):
    """Synthesis transform  gs  with λ-FiLM conditioning  [Injection Point 2].

    gs produces the noisy latent lT fed to ϵSD.  Conditioning on λ allows gs
    to generate coarser lT at high compression (large λ) so the Unet denoises
    more aggressively, and finer lT at low compression (small λ).

    FiLM sites (all channel dim 320):
        BasicBlock(320) → FiLM(320)
        Upsample(320→320)
        BasicBlock(320) → FiLM(320)
        Upsample(320→320)
        BasicBlock(320) → FiLM(320)
        Upsample(320→320)
    """

    def __init__(self, embed_dim: int = FILM_DIM):
        super().__init__()
        self.block1 = BasicBlock(320)
        self.film1 = FiLMLayer(embed_dim, 320)
        self.up1 = Upsample(320, 320)
        self.block2 = BasicBlock(320)
        self.film2 = FiLMLayer(embed_dim, 320)
        self.up2 = Upsample(320, 320)
        self.block3 = BasicBlock(320)
        self.film3 = FiLMLayer(embed_dim, 320)
        self.up3 = Upsample(320, 320)

    def forward(
        self,
        x:          torch.Tensor,   # [B, 320, H, W]  quantised latent y_hat
        film_embed: torch.Tensor,   # [B, embed_dim]
    ) -> torch.Tensor:
        x = self.up1(self.film1(self.block1(x), film_embed))
        x = self.up2(self.film2(self.block2(x), film_embed))
        x = self.up3(self.film3(self.block3(x), film_embed))
        return x


class AuxDecoder(nn.Module):
    """Auxiliary decoder  DAux  with λ-FiLM conditioning  [Injection Point 4].

    DAux performs structure apportionment: it decodes basic structure directly
    from ŷ, bypassing the Unet, so that gs can focus on high-frequency detail.
    Conditioning DAux on λ tells it how much structural information is available
    (more at low compression, less at high compression).

    FiLM sites:
        BasicBlock(320) → FiLM(320)
        Upsample(320→256)
        BasicBlock(256) → FiLM(256)
        Upsample(256→256)
        BasicBlock(256) → FiLM(256)
        Upsample(256→256)
    """

    def __init__(self, embed_dim: int = FILM_DIM):
        super().__init__()
        self.block1 = BasicBlock(320)
        self.film1 = FiLMLayer(embed_dim, 320)
        self.up1 = Upsample(320, 256)
        self.block2 = BasicBlock(256)
        self.film2 = FiLMLayer(embed_dim, 256)
        self.up2 = Upsample(256, 256)
        self.block3 = BasicBlock(256)
        self.film3 = FiLMLayer(embed_dim, 256)
        self.up3 = Upsample(256, 256)

    def forward(
        self,
        x:          torch.Tensor,   # [B, 320, H, W]
        film_embed: torch.Tensor,   # [B, embed_dim]
    ) -> torch.Tensor:
        x = self.up1(self.film1(self.block1(x), film_embed))
        x = self.up2(self.film2(self.block2(x), film_embed))
        x = self.up3(self.film3(self.block3(x), film_embed))
        return x


# ===========================================================================
# Rate module — variable λ
# ===========================================================================

class RateLossOutput(NamedTuple):
    rate_loss:               Tensor
    quantized_total_bpp:     Tensor
    quantized_latent_bpp:    Tensor
    quantized_hyper_bpp:     Tensor
    per_image_bpp:           Tensor   # [B] per-image quantized bpp (for diagnostics)


class TargetRateModule(nn.Module):
    """Variable-rate rate-distortion loss module.

    Loss formulation:  L = D/λ + bpp
      • High λ → D/λ small → bpp dominates → model compresses harder → lower bpp
      • Low  λ → D/λ large → distortion dominates → model allocates more bits → higher bpp

    rate_loss returns unweighted mean(bpp); the 1/λ scaling on distortion is
    handled in the training script.
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
        lmbda:                             Tensor,  # [B] or scalar (kept for API compat)
        ori_h: int = 512,
        ori_w:  int = 512,
    ) -> RateLossOutput:
        N = ori_h * ori_w
        latent_bpp   = self._bpp(latent_likelihoods)          / N
        quantized_latent_bpp  = self._bpp(quantized_latent_likelihoods) / N
        hyper_bpp   = self._bpp(hyper_latent_likelihoods)    / N
        quantized_hyper_bpp  = self._bpp(quantized_hyper_latent_likelihoods) / N

        total_bpp  = latent_bpp  + hyper_bpp
        quantized_total_bpp = quantized_latent_bpp + quantized_hyper_bpp

        return RateLossOutput(
            rate_loss            = total_bpp.mean(),                      # unweighted bpp
            quantized_total_bpp  = quantized_total_bpp.detach().mean(),
            quantized_latent_bpp = quantized_latent_bpp .detach().mean(),
            quantized_hyper_bpp  = quantized_hyper_bpp .detach().mean(),
            per_image_bpp        = quantized_total_bpp.detach(),          # [B]
        )


# ===========================================================================
# LatentCodec — variable-rate with λ-FiLM
# ===========================================================================

# ===========================================================================
# DynamicTimestepModule — SNR-based analytical T* from entropy model σ
# ===========================================================================

class DynamicTimestepModule(nn.Module):
    """Compute per-image dynamic timestep T* from entropy model scales.

    Uses the SNR schedule from ``alphas_cumprod`` to establish a monotonic
    ordering, then linearly rescales the raw timestep into [t_min, t_max]
    so that T* stays near the UNet's training point (T=999).

    Args:
        alphas_cumprod: 1-D tensor of length T (typically 1000) from DDPMScheduler.
        t_min: lower bound of the output T* range (default 800).
        t_max: upper bound of the output T* range (default 999).
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
        signal_var  = scales_all.mean(dim=[1, 2, 3]) ** 2  # [B]
        snr_compress = signal_var / sigma_quant             # [B]
        T_raw = torch.searchsorted(-self.snr_schedule, -snr_compress)
        T_star = self.t_min + (self.t_max - self.t_min) * (T_raw.float() / 999.0)
        T_star = T_star.clamp(self.t_min, self.t_max)
        return T_star


# ===========================================================================
# LatentCodec — variable-rate with λ-FiLM + dynamic timestep
# ===========================================================================

class LatentCodec(CompressionModel):
    """StableCodec latent codec with variable-rate λ-FiLM conditioning
    and SNR-based dynamic timestep T*.

    Compared to the variable-rate version (latent_codec_variable2.py):
      • film_embed (LambdaFiLMEmbed)  — unchanged: shared λ→f(λ) embedding
      • g_a (AnalysisTransform)       — unchanged: FiLM at 3 layers
      • g_s (SynthesisTransform)      — unchanged: FiLM at 3 layers
      • aux (AuxDecoder)              — unchanged: FiLM at 3 layers
      • rate (TargetRateModule)       — unchanged: unweighted mean(bpp)
      • dynamic_timestep              — NEW: DynamicTimestepModule, σ → T*

    Interface change vs. latent_codec_variable2:
        forward   returns (x_hat, rate_out, res, T_star)   ← +T_star
        compress  returns dict with added key "t_star_val"
        decompress returns (x_hat, res, T_star)            ← +T_star
    """

    def __init__(
        self,
        lambda_min: float = LAMBDA_MIN,
        lambda_max: float = LAMBDA_MAX,
        alphas_cumprod: torch.Tensor = None,
    ):
        super().__init__()

        M = 320
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Shared λ-FiLM embedding (used by Injection Points 1, 2, 4)
        self.film_embed = LambdaFiLMEmbed(
            embed_dim=FILM_DIM, num_freqs=NUM_FREQS,
            lambda_min=lambda_min, lambda_max=lambda_max,
        )

        # FiLM-conditioned transforms
        self.g_a = AnalysisTransform (embed_dim=FILM_DIM)   # Injection 1
        self.g_s = SynthesisTransform(embed_dim=FILM_DIM)   # Injection 2
        self.aux = AuxDecoder        (embed_dim=FILM_DIM)   # Injection 4

        # Hyper-prior
        self.h_a = HyperAnalysis (M=M)
        self.h_s = HyperSynthesis(M=M)

        # Entropy model
        context_dim = M * 3
        self.adapter_in  = nn.ModuleList([Adapter(M,           context_dim) for _ in range(4)])
        self.g_c         = SpatialContext(context_dim)
        self.adapter_out = nn.ModuleList([Adapter(context_dim, M * 2)       for _ in range(4)])
        self.LRP         = nn.ModuleList([LRP(M * 2, M)                     for _ in range(4)])

        self.entropy_bottleneck   = EntropyBottleneck(M // 2)
        self.gaussian_conditional = GaussianConditional(None)
        self.masks = {}

        # Variable-rate module
        self.rate = TargetRateModule()

        # Dynamic timestep module (SNR-based T*)
        if alphas_cumprod is not None:
            self.dynamic_timestep = DynamicTimestepModule(alphas_cumprod)
        else:
            self.dynamic_timestep = None

    # ------------------------------------------------------------------
    # Internal helpers: mask generation, squeeze/unsqueeze (unchanged)
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
        idxs_t  = gc.build_indexes(ss)
        lhat    = gc.quantize(ls, "symbols", ms)
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
    # Helper: resolve lmbda to [B] Tensor on the correct device
    # ------------------------------------------------------------------

    def _resolve_lmbda(self, lmbda, B: int, device) -> torch.Tensor:
        if lmbda is None:
            lmbda = torch.full((B,), self.lambda_min, dtype=torch.float32, device=device)
        elif not isinstance(lmbda, torch.Tensor):
            lmbda = torch.tensor(lmbda, dtype=torch.float32, device=device)
        lmbda = lmbda.float().to(device)
        if lmbda.dim() == 0:
            lmbda = lmbda.expand(B)
        elif lmbda.shape[0] != B:
            lmbda = lmbda[:1].expand(B)
        return lmbda

    # ------------------------------------------------------------------
    # Forward  (training / validation)
    # ------------------------------------------------------------------

    def forward(self, latent, latent2, ori_h, ori_w, lmbda):
        """
        Args:
            latent:  [B, 256, H, W]  ESD VAE latent
            latent2: [B, 320, H, W]  EAux semantic latent (frozen)
            ori_h, ori_w: original image size for bpp normalisation
            lmbda:   Tensor [B] — per-image λ values sampled in training loop

        Returns:
            x_hat          [B, 320, H*8, W*8]  gs output  (lT for Unet)
            RateLossOutput named-tuple
            res            [B, 256, H*8, W*8]  DAux output
            T_star         [B]                  dynamic timestep per image
        """
        B = latent.shape[0]
        lmbda      = self._resolve_lmbda(lmbda, B, latent.device)
        film_embed = self.film_embed(lmbda)                    # [B, FILM_DIM]

        # ---- Injection Point 1: ga ----
        y = self.g_a(latent, latent2, film_embed)

        # Hyper-prior entropy
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        with torch.no_grad():
            _, qz_likelihoods = self.entropy_bottleneck(z, training=False)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat    = ste_round(z - z_offset) + z_offset

        B2, C, H, W = y.shape
        m0, m1, m2, m3 = self.get_mask_four_parts(B2, C, H, W, y.device)

        # Quadtree autoregressive entropy coding
        base = self.h_s(z_hat)
        base = base[:, :, :H, :W]

        def _step(mask, prev_base):
            ms_, ss_ = self.adapter_out[0 if mask is m0 else
                        1 if mask is m1 else 2 if mask is m2 else 3](
                self.g_c(self.adapter_in[0 if mask is m0 else
                         1 if mask is m1 else 2 if mask is m2 else 3](prev_base))
            ).chunk(2, 1)
            return ms_, ss_

        means_0, scales_0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0  = means_0  * m0
        scales_0 = scales_0 * m0
        y_hat_0  = ste_round(y * m0 - means_0) + means_0
        lrp      = 0.5 * torch.tanh(self.LRP[0](torch.cat([y_hat_0, base], 1)) * m0)
        y_hat_0  = y_hat_0 + lrp

        base = base * (1 - m0) + y_hat_0
        means_1, scales_1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1  = means_1  * m1
        scales_1 = scales_1 * m1
        y_hat_1  = ste_round(y * m1 - means_1) + means_1
        lrp      = 0.5 * torch.tanh(self.LRP[1](torch.cat([y_hat_1, base], 1)) * m1)
        y_hat_1  = y_hat_1 + lrp

        base = base * (1 - m1) + y_hat_1
        means_2, scales_2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2  = means_2  * m2
        scales_2 = scales_2 * m2
        y_hat_2  = ste_round(y * m2 - means_2) + means_2
        lrp      = 0.5 * torch.tanh(self.LRP[2](torch.cat([y_hat_2, base], 1)) * m2)
        y_hat_2  = y_hat_2 + lrp

        base = base * (1 - m2) + y_hat_2
        means_3, scales_3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3  = means_3  * m3
        scales_3 = scales_3 * m3
        y_hat_3  = ste_round(y * m3 - means_3) + means_3
        lrp      = 0.5 * torch.tanh(self.LRP[3](torch.cat([y_hat_3, base], 1)) * m3)
        y_hat_3  = y_hat_3 + lrp

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all  = means_0  + means_1  + means_2  + means_3
        _, y_likelihoods  = self.gaussian_conditional(y, scales_all, means_all)
        with torch.no_grad():
            _, qy_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - m3) + y_hat_3

        # ---- Dynamic timestep T* from entropy model scales ----
        T_star = self.dynamic_timestep(scales_all)          # [B]

        # ---- Injection Point 2: gs ----
        x_hat = self.g_s(y_hat, film_embed)

        # ---- Injection Point 4: DAux ----
        res = self.aux(y_hat, film_embed)

        rate_out = self.rate(
            latent_likelihoods                 = y_likelihoods,
            quantized_latent_likelihoods       = qy_likelihoods,
            hyper_latent_likelihoods           = z_likelihoods,
            quantized_hyper_latent_likelihoods = qz_likelihoods,
            lmbda                              = lmbda,
            ori_h                              = ori_h,
            ori_w                              = ori_w,
        )
        return x_hat, rate_out, res, T_star, scales_all

    # ------------------------------------------------------------------
    # Compress  (inference encoding)
    # ------------------------------------------------------------------

    def compress(self, latent, latent2, lmbda=None):
        """
        Returns a dict with keys: strings, shape, lmbda_val, t_star_val.
        lmbda_val stores λ as float16 (2 bytes per image) — I2C bitstream design,
        giving 2^16 = 65 536 effective variable-rate points.
        t_star_val stores the dynamic timestep T* per image.
        """
        B          = latent.size(0)
        lmbda      = self._resolve_lmbda(lmbda, B, latent.device)
        film_embed = self.film_embed(lmbda)

        # ---- Injection Point 1: ga ----
        y = self.g_a(latent, latent2, film_embed)
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

        ms0, ss0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        yh0 = self.compress_group_with_mask(self.gaussian_conditional, y, ss0, ms0, m0, syms, idxs)
        yh0 += 0.5 * torch.tanh(self.LRP[0](torch.cat([yh0, base], 1)) * m0)

        base = base * (1 - m0) + yh0
        ms1, ss1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        yh1 = self.compress_group_with_mask(self.gaussian_conditional, y, ss1, ms1, m1, syms, idxs)
        yh1 += 0.5 * torch.tanh(self.LRP[1](torch.cat([yh1, base], 1)) * m1)

        base = base * (1 - m1) + yh1
        ms2, ss2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        yh2 = self.compress_group_with_mask(self.gaussian_conditional, y, ss2, ms2, m2, syms, idxs)
        yh2 += 0.5 * torch.tanh(self.LRP[2](torch.cat([yh2, base], 1)) * m2)

        base = base * (1 - m2) + yh2
        ms3, ss3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        _ = self.compress_group_with_mask(self.gaussian_conditional, y, ss3, ms3, m3, syms, idxs)

        # Compute dynamic timestep T* from accumulated scales
        scales_all = ss0 * m0 + ss1 * m1 + ss2 * m2 + ss3 * m3
        T_star = self.dynamic_timestep(scales_all)          # [B]

        encoder.encode_with_indexes(syms, idxs, cdf, cdf_lengths, offsets)
        y_strings.append(encoder.flush())
        torch.backends.cudnn.deterministic = False

        return {
            "strings":    [y_strings, z_strings],
            "shape":      z.size()[-2:],
            # Store lmbda as float16 list: 2 bytes per image in bitstream
            "lmbda_val":  lmbda.cpu().half().tolist(),
            # Store T* per image for UNet denoising
            "t_star_val": T_star.cpu().tolist(),
        }

    # ------------------------------------------------------------------
    # Decompress  (inference decoding)
    # ------------------------------------------------------------------

    def decompress(self, strings, shape, lmbda=None):
        """
        Args:
            strings: dict from compress() OR raw [y_strings, z_strings] list
            shape:   z spatial shape
            lmbda:   Tensor/scalar.  Resolved priority:
                       1. explicit argument
                       2. strings["lmbda_val"]  (if strings is dict)
                       3. default: lambda_min (highest quality)
        Returns:
            x_hat   [B, 320, H*8, W*8]   gs output (lT)
            res     [B, 256, H*8, W*8]   DAux output
            T_star  [B]                   dynamic timestep per image
        """
        if lmbda is None:
            if isinstance(strings, dict) and "lmbda_val" in strings:
                lmbda = torch.tensor(strings["lmbda_val"], dtype=torch.float32)
            else:
                lmbda = torch.tensor([self.lambda_min], dtype=torch.float32)

        raw = strings["strings"] if isinstance(strings, dict) else strings

        torch.backends.cudnn.deterministic = True
        z_hat = self.entropy_bottleneck.decompress(raw[1], shape)

        B = z_hat.size(0)
        lmbda      = self._resolve_lmbda(lmbda, B, z_hat.device)
        film_embed = self.film_embed(lmbda)                    # [B, FILM_DIM]

        cdf         = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets     = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder     = RansDecoder()
        decoder.set_stream(raw[0][0])

        C, H, W = z_hat.shape[1], z_hat.shape[2], z_hat.shape[3]
        m0, m1, m2, m3 = self.get_mask_four_parts(B, C * 2, H * 4, W * 4, z_hat.device)
        base = self.h_s(z_hat)

        ms0, ss0 = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        yh0 = self.decompress_group_with_mask(self.gaussian_conditional, ss0, ms0, m0, decoder, cdf, cdf_lengths, offsets)
        yh0 += 0.5 * torch.tanh(self.LRP[0](torch.cat([yh0, base], 1)) * m0)

        base = base * (1 - m0) + yh0
        ms1, ss1 = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        yh1 = self.decompress_group_with_mask(self.gaussian_conditional, ss1, ms1, m1, decoder, cdf, cdf_lengths, offsets)
        yh1 += 0.5 * torch.tanh(self.LRP[1](torch.cat([yh1, base], 1)) * m1)

        base = base * (1 - m1) + yh1
        ms2, ss2 = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        yh2 = self.decompress_group_with_mask(self.gaussian_conditional, ss2, ms2, m2, decoder, cdf, cdf_lengths, offsets)
        yh2 += 0.5 * torch.tanh(self.LRP[2](torch.cat([yh2, base], 1)) * m2)

        base = base * (1 - m2) + yh2
        ms3, ss3 = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        yh3 = self.decompress_group_with_mask(self.gaussian_conditional, ss3, ms3, m3, decoder, cdf, cdf_lengths, offsets)
        yh3 += 0.5 * torch.tanh(self.LRP[3](torch.cat([yh3, base], 1)) * m3)

        y_hat = yh0 + yh1 + yh2 + yh3
        torch.backends.cudnn.deterministic = False

        # CFT: scales_all must be returned for the model-level CFT module, so compute
        # it unconditionally (regardless of whether T* comes from the bitstream).
        scales_all = ss0 * m0 + ss1 * m1 + ss2 * m2 + ss3 * m3

        # Compute T* — prefer bitstream value; fallback to recomputation from scales
        if isinstance(strings, dict) and "t_star_val" in strings:
            T_star = torch.tensor(
                strings["t_star_val"], dtype=torch.float32, device=z_hat.device,
            )
        else:
            T_star = self.dynamic_timestep(scales_all)

        # ---- Injection Points 2 & 4 ----
        x_hat = self.g_s(y_hat, film_embed)
        res   = self.aux(y_hat, film_embed)
        return x_hat, res, T_star, scales_all

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated  = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated