"""Rate conditioning and entropy-scale timestep selection (no external weights)."""
import math
import torch
import torch.nn as nn

FILM_DIM = 512
NUM_FREQS = 10
LAMBDA_MIN = 0.1
LAMBDA_MAX = 128.0


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

