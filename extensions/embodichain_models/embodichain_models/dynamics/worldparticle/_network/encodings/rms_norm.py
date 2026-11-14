import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More efficient than LayerNorm, used in modern transformers like LLaMA.

    This implementation handles mixed precision computation by keeping normalization
    computations in fp32 for numerical stability.

    Reference: Zhang and Sennrich 2019 - "Root Mean Square Layer Normalization"
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Keep weight in fp32 for stability in mixed precision computation
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original dtype for output
        orig_dtype = x.dtype

        # Convert to fp32 for stable computation
        x_fp32 = x.float()

        # RMS normalization: x / RMS(x) * weight
        # RMS(x) = sqrt(mean(x^2) + eps)
        # Compute in fp32 for numerical stability
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x_fp32 / rms

        # Apply weight (which is in fp32) and convert back to original dtype
        return (self.weight * x_normed).to(orig_dtype)
