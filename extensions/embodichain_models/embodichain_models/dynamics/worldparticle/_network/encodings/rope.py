# Modified from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from math import log
from typing import Optional

import torch
from torch.nn import Module
from torch.amp import autocast
from torch import nn, einsum, Tensor

from einops import rearrange, repeat


# rotary embedding helper functions
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rotate_half_hf(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (
        rotate_half(t_middle) * freqs.sin() * scale
    )

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


def freqs_to_cos_sin(freqs, scale=1.0, start_index=0, head_dim=None):
    """
    Convert frequencies to cos and sin for rotary embeddings.

    Args:
        freqs (torch.Tensor): The frequency tensor of shape (..., n_freqs).
        scale (float): The scaling factor for the frequencies.
        start_index (int): The starting index of the frequencies.
        head_dim (int): The dimension of the head.
    """
    if head_dim is not None:
        # pad the freqs to match the head_dim
        freqs = freqs[..., : freqs.shape[-1] // 2]
        left_pad = start_index
        right_pad = head_dim // 2 - (left_pad + freqs.shape[-1])
        freqs = torch.cat(
            (
                torch.zeros((*freqs.shape[:-1], left_pad), device=freqs.device),
                freqs,
                torch.zeros((*freqs.shape[:-1], right_pad), device=freqs.device),
            ),
            dim=-1,
        )
        freqs = torch.cat([freqs, freqs], dim=-1)

    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    return cos, sin


@autocast("cuda", enabled=False)
def apply_rotary_emb_cossin(q, k, cos, sin):
    """
    q size: (bsz, n_q_head, seq_len, head_dim)
    k size: (bsz, n_kv_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == q.shape[-1]
    ), f"feature dimension {q.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    q = (q * cos) + (rotate_half_hf(q) * sin)
    k = (k * cos) + (rotate_half_hf(k) * sin)

    return q.type(dtype), k.type(dtype)


@autocast("cuda", enabled=False)
def apply_rotary_emb_one_cossin(one_tensor, cos, sin):
    """
    one_tensor size: (bsz, n_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = one_tensor.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == one_tensor.shape[-1]
    ), f"feature dimension {one_tensor.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    one_tensor = (one_tensor * cos) + (rotate_half_hf(one_tensor) * sin)

    return one_tensor.type(dtype)


class TriangleRotaryEmbedding(Module):

    def __init__(
        self,
        dim,
        hf_format=True,
        double_max_freq=False,
    ):
        """
        TriangleRotaryEmbedding is a class that implements the rotary embedding for the triangle.

        Args:
            dim (int): The dimension of the rotary embedding.
            hf_format (bool): Whether to use the huggingface RoPE format.
            double_max_freq (bool): Whether to double the frequency range.
        """
        super().__init__()

        self.hf_format = hf_format

        # log spaced frequencies
        max_freq = log(dim // 2 - 1, 2) if not double_max_freq else log(dim - 1, 2)
        freqs = 2 ** torch.linspace(0, max_freq, dim // 2)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # dummy for device
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_triangle_freqs(self, pos: Tensor):
        # generate all frequencies for all triangles
        freqs = self.forward(pos)
        freqs = rearrange(
            freqs, "batch n_tris n_verts d -> batch 1 n_tris (n_verts d)"
        )  # 1 for head dim

        if self.hf_format:
            freqs = torch.cat([freqs, freqs], dim=-1)
        else:
            freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        return freqs

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        freqs = self.freqs
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)

        return freqs


class ParticleRotaryEmbeddingObsolete(Module):

    def __init__(
        self,
        dim,
        hf_format=True,
        double_max_freq=False,
    ):
        """
        ParticleRotaryEmbedding is a class that implements the rotary embedding for the particle.

        Args:
            dim (int): The dimension of the rotary embedding.
            hf_format (bool): Whether to use the huggingface RoPE format.
            double_max_freq (bool): Whether to double the frequency range.
        """
        super().__init__()

        self.hf_format = hf_format

        # log spaced frequencies
        max_freq = log(dim // 2 - 1, 2) if not double_max_freq else log(dim - 1, 2)
        freqs = 2 ** torch.linspace(0, max_freq, dim // 2)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # dummy for device
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_particle_freqs(self, pos: Tensor):
        # generate all frequencies for all particles
        freqs = self.forward(pos)
        freqs = rearrange(
            freqs, "batch n_particles d -> batch 1 n_particles d"
        )  # 1 for head dim

        if self.hf_format:
            freqs = torch.cat([freqs, freqs], dim=-1)
        else:
            freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        return freqs

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        """
        Forward pass for particle positions.

        Args:
            t: Tensor of shape (..., 3) containing 3D particle positions (x, y, z)
               or (..., 1) for 1D positions
        Returns:
            Tensor of shape (..., dim//2) containing frequency encodings
        """
        freqs = self.freqs

        # Check input dimensionality
        input_dim = t.shape[-1]

        if input_dim == 3:  # 3D positions (x, y, z)
            # For 3D positions, we split frequencies across the 3 spatial dimensions
            # Each dimension gets dim//6 frequencies (since dim//2 total, divided by 3)
            assert (
                len(freqs) % 3 == 0
            ), f"Number of frequencies {len(freqs)} must be divisible by 3 for 3D positions"

            freqs_per_dim = len(freqs) // 3

            # Apply frequencies to each spatial dimension separately
            # x coordinate modulates first third of frequencies
            x_freqs = einsum(
                "..., f -> ... f", t[..., 0].type(freqs.dtype), freqs[:freqs_per_dim]
            )
            # y coordinate modulates second third of frequencies
            y_freqs = einsum(
                "..., f -> ... f",
                t[..., 1].type(freqs.dtype),
                freqs[freqs_per_dim : 2 * freqs_per_dim],
            )
            # z coordinate modulates third third of frequencies
            z_freqs = einsum(
                "..., f -> ... f",
                t[..., 2].type(freqs.dtype),
                freqs[2 * freqs_per_dim :],
            )

            # Concatenate frequencies from all dimensions
            freqs = torch.cat([x_freqs, y_freqs, z_freqs], dim=-1)

        elif input_dim == 1:  # 1D positions (scalar per particle)
            # For 1D positions, use all frequencies
            freqs = einsum("..., f -> ... f", t[..., 0].type(freqs.dtype), freqs)

        else:
            raise ValueError(
                f"ParticleRotaryEmbedding expects input with last dimension 1 or 3, got {input_dim}"
            )

        return freqs


class ParticleRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for PhysFormer's 3D point clouds.
    Handles both state tokens and vertex positions.
    """

    def __init__(
        self,
        dim: int,
        max_freq: float = 10.0,
        learnable_freqs: bool = True,
        include_time: bool = False,
    ):
        """
        Args:
            dim: Dimension of the embeddings (must be divisible by 2, or 6/8 for 3D)
            max_freq: Maximum frequency for sinusoidal embeddings
            use_3d: Whether to use 3D positional encoding (x, y, z)
            learnable_freqs: Whether to make frequencies learnable
            include_time: Whether to include time dimension (for dynamics)
        """
        super().__init__()

        self.dim = dim
        self.include_time = include_time

        # Determine how many dimensions per coordinate
        if include_time:
            # 4D: (x, y, z, t)
            self.n_coords = 4
            assert (
                dim % (self.n_coords * 2) == 0
            ), f"dim {dim} must be divisible by {self.n_coords * 2}"
            self.dim_per_coord = dim // (self.n_coords * 2)
        else:
            # 3D: (x, y, z)
            self.n_coords = 3
            assert (
                dim % (self.n_coords * 2) == 0
            ), f"dim {dim} must be divisible by {self.n_coords * 2}"
            self.dim_per_coord = dim // (self.n_coords * 2)

        # Create frequency bands
        freqs = torch.logspace(0, math.log10(max_freq), self.dim_per_coord)

        if learnable_freqs:
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("freqs", freqs)

        # For state tokens, we might want learnable position embeddings
        self.state_pos_embed = None

    def forward_3d(self, positions: torch.Tensor) -> torch.Tensor:

        # Validate input
        n_coords = 4 if self.include_time else 3
        assert positions.shape[-1] == n_coords

        # Shape: positions (..., n_coords)
        # Shape: freqs (dim_per_coord,)

        # Compute angles for all coordinates at once
        # Reshape positions: (..., n_coords, 1)
        # Broadcast with freqs: (..., n_coords, dim_per_coord)
        angles = positions.unsqueeze(-1) * self.freqs  # (..., n_coords, dim_per_coord)

        # Duplicate each angle for sin/cos pairs
        # Shape: (..., n_coords, 2*dim_per_coord)
        emb = torch.cat([angles, angles], dim=-1)

        # Reshape to concatenate coordinate embeddings
        # From: (..., n_coords, 2*dim_per_coord)
        # To: (..., n_coords * 2*dim_per_coord)
        rope_emb = emb.flatten(-2, -1)
        rope_emb = rope_emb.unsqueeze(1)

        return rope_emb

    def forward(
        self,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flexible forward function that handles different input types.

        Args:
            positions: 3D/4D positions for vertices
            batch_size: Batch size for generating state token positions
            seq_len: Sequence length for state tokens

        Returns:
            RoPE embeddings
        """
        if positions is not None:
            return self.forward_3d(positions)
        else:
            raise ValueError("Must provide positions (3D or 4D) for RoPE")
