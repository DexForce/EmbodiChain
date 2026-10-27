# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
RoPE (Rotary Position Embedding) for PhysFormer.
Adapted from RenderFormer's implementation for 3D point clouds and physics simulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List


def rotate_half(x):
    """Rotate interleaved channel pairs: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...).

    Split into pairs, unbind, stack, merge (the einops rearrangement, without its per-call pattern parsing).
    """
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# Per-step memo of rotary angles and of their cos/sin. Within one model step every attention call that sees
# the same position tensor recomputes the same angles (16 calls per step in the decoder and predictor); the
# memo computes each once. Entries are keyed by the input's storage pointer, shape, strides, dtype and version
# counter, and keep a reference to it, so a hit can only return values computed from identical memory (an
# in-place change bumps the version; the pinned storage cannot be reused). Only inputs that do not require
# gradients are memoized, so autograd is unaffected and results are bit-identical. The model clears the memo
# at the start of every step (rope_memo_reset), which also keeps CUDA-graph capture self-contained.
_ROPE_MEMO = {}
ROPE_MEMO_ENABLED = True


def rope_memo_reset():
    _ROPE_MEMO.clear()


_ROPE_MEMO_MAX = 512


def _memo_put(key, value):
    if (
        len(_ROPE_MEMO) >= _ROPE_MEMO_MAX
    ):  # paths that never reset the memo cannot grow it without bound
        _ROPE_MEMO.clear()
    _ROPE_MEMO[key] = value


def _memo_key(tag, x, extra=()):
    return (
        tag,
        x.device,
        x.data_ptr(),
        tuple(x.shape),
        x.stride(),
        x.dtype,
        x._version,
    ) + tuple(extra)


def _memo_usable(*tensors):
    # Under torch.compile the graph de-duplicates identical subexpressions itself; the memo (which reads storage
    # pointers) is skipped there.
    if torch.compiler.is_compiling():
        return False
    return ROPE_MEMO_ENABLED and not any(t.requires_grad for t in tensors)


def _cos_sin(freqs):
    if _memo_usable(freqs):
        key = _memo_key("cs", freqs)
        hit = _ROPE_MEMO.get(key)
        if hit is not None:
            return hit[1], hit[2]
        c, s_ = freqs.cos(), freqs.sin()
        _memo_put(key, (freqs, c, s_))
        return c, s_
    return freqs.cos(), freqs.sin()


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, scale=1.0):
    """
    Apply rotary embeddings to tensor t using frequency tensor freqs.

    Args:
        freqs: Frequency tensor
        t: Input tensor to apply RoPE to
        scale: Scaling factor
    """
    dtype = t.dtype
    rot_dim = freqs.shape[-1]

    assert (
        rot_dim <= t.shape[-1]
    ), f"Feature dimension {t.shape[-1]} is not sufficient for rotation dimension {rot_dim}"

    # Apply to the first rot_dim dimensions
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    # Apply rotary embeddings (x * 1.0 == x exactly, so the unit scale is skipped)
    cos, sin = _cos_sin(freqs)
    if scale == 1.0:
        t_rot = (t_rot * cos) + (rotate_half(t_rot) * sin)
    else:
        t_rot = (t_rot * cos * scale) + (rotate_half(t_rot) * sin * scale)

    return torch.cat([t_rot, t_pass], dim=-1).type(dtype)


class PhysFormerRoPE(nn.Module):
    """
    Rotary Position Embedding for PhysFormer's 3D point clouds.
    Handles both state tokens and vertex positions.
    """

    # Pair sin/cos channels the way rotate_half pairs them (interleaved), which
    # keeps the relative-position property; the half-split layout does not. A
    # model is evaluated with the layout it was trained with.
    INTERLEAVED_PAIRING = False
    _PAIRING_LOGGED = False

    def _paired_angles(self, angles: torch.Tensor) -> torch.Tensor:
        cls = type(self)
        if cls.INTERLEAVED_PAIRING:
            if not cls._PAIRING_LOGGED:
                cls._PAIRING_LOGGED = True
                print(
                    f"[RoPE] interleaved rotary pairing ACTIVE ({angles.shape[-1]} angles -> {2 * angles.shape[-1]} dims)",
                    flush=True,
                )
            return torch.repeat_interleave(angles, 2, dim=-1)
        return torch.cat([angles, angles], dim=-1)

    def __init__(
        self,
        dim: int,
        max_freq: float = 10.0,
        use_3d: bool = True,
        learnable_freqs: bool = False,
        include_time: bool = False,
        use_set_aggregation: bool = False,
        dim_per_coord_3d: int = None,  # Explicit dim per coordinate for single-point (3D) mode
        dim_per_coord_multi: int = None,  # Explicit dim per coordinate for multi-point mode
    ):
        """
        Args:
            dim: Dimension of the embeddings (rope_dim, e.g., 96)
            max_freq: Maximum frequency for sinusoidal embeddings
            use_3d: Whether to use 3D positional encoding (x, y, z)
            learnable_freqs: Whether to make frequencies learnable
            include_time: Whether to include time dimension (for dynamics)
            use_set_aggregation: Whether to use set aggregation for multi-point encoding
                                If True: encode each point independently then aggregate
                                If False: use dim_per_coord_multi to encode all coords without truncation
            dim_per_coord_3d: Explicit dimension per coordinate for single-point 3D mode.
                             If None, auto-compute: dim // (3 * 2)
                             Example: 32 for 3 coords × 32 = 96
            dim_per_coord_multi: Explicit dimension per coordinate for multi-point mode.
                                If None, will raise error when forward_multi_point is called
                                Example: 4 for 24 coords × 4 = 96
        """
        super().__init__()

        self.dim = dim
        self.use_3d = use_3d
        self.include_time = include_time
        self.use_set_aggregation = use_set_aggregation
        self.max_freq = max_freq  # Store for dynamic freq generation

        # Determine how many dimensions per coordinate (for forward_3d)
        if use_3d:
            if include_time:
                # 4D: (x, y, z, t)
                self.n_coords = 4
                if dim_per_coord_3d is not None:
                    # Use explicit configuration
                    self.dim_per_coord_3d = dim_per_coord_3d
                    # Verify correctness
                    expected = self.n_coords * self.dim_per_coord_3d
                    if expected != dim:
                        raise ValueError(
                            f"dim_per_coord_3d mismatch: {self.n_coords} coords × {dim_per_coord_3d} = {expected}, "
                            f"but rope_dim={dim}"
                        )
                else:
                    # Auto-compute
                    assert (
                        dim % (self.n_coords * 2) == 0
                    ), f"dim {dim} must be divisible by {self.n_coords * 2}"
                    self.dim_per_coord_3d = dim // (self.n_coords * 2)
            else:
                # 3D: (x, y, z)
                self.n_coords = 3
                if dim_per_coord_3d is not None:
                    # Use explicit configuration
                    self.dim_per_coord_3d = dim_per_coord_3d
                    # Verify correctness
                    expected = self.n_coords * self.dim_per_coord_3d
                    if expected != dim:
                        raise ValueError(
                            f"dim_per_coord_3d mismatch: {self.n_coords} coords × {dim_per_coord_3d} = {expected}, "
                            f"but rope_dim={dim}"
                        )
                else:
                    # Auto-compute
                    assert (
                        dim % (self.n_coords * 2) == 0
                    ), f"dim {dim} must be divisible by {self.n_coords * 2}"
                    self.dim_per_coord_3d = dim // (self.n_coords * 2)
        else:
            raise ValueError(
                "PhysFormerRoPE now supports only 3D (or 4D with time) positions"
            )

        # For multi-point mode: explicit dim_per_coord (optional)
        self.dim_per_coord_multi = dim_per_coord_multi  # Can be None

        # Create frequency bands for 3D mode
        # dim_per_coord_3d is the total dimension per coordinate (including sin+cos)
        # so freq_len = dim_per_coord_3d // 2
        freq_len_3d = self.dim_per_coord_3d // 2
        if self.dim_per_coord_3d % 2 != 0:
            raise ValueError(
                f"dim_per_coord_3d ({self.dim_per_coord_3d}) must be even for sin/cos pairs"
            )

        freqs = torch.logspace(0, math.log10(max_freq), freq_len_3d)

        if learnable_freqs:
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer("freqs", freqs)

        # Optional learnable position embedding for state tokens (unused by default)
        self.state_pos_embed = None

    def _freqs_signature(self):
        """Value signature of self.freqs, so layers with identical frequencies share memo entries.
        Recomputed (one small device-to-host copy) only when the buffer's storage or version changes.
        """
        f = self.freqs
        state = (f.data_ptr(), f._version, f.dtype, f.device)
        cached = getattr(self, "_freqs_sig_cache", None)
        if cached is None or cached[0] != state:
            sig = (
                f.detach().cpu().numpy().tobytes(),
                str(f.dtype),
                tuple(f.shape),
            )  # exact values
            self._freqs_sig_cache = (state, sig)
        return self._freqs_sig_cache[1]

    def _memo_extra(self):
        return (
            self._freqs_signature(),
            type(self).INTERLEAVED_PAIRING,
            self.n_coords,
            self.include_time,
            self.dim_per_coord_3d,
        )

    def forward_3d(self, positions: torch.Tensor) -> torch.Tensor:
        if _memo_usable(positions, self.freqs):
            key = _memo_key("3d", positions, self._memo_extra())
            hit = _ROPE_MEMO.get(key)
            if hit is None:
                hit = (positions, self._forward_3d(positions))
                _memo_put(key, hit)
            return hit[1]
        return self._forward_3d(positions)

    def _forward_3d(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute RoPE for 3D positions (single point).

        Args:
            positions: 3D positions of shape (..., 3) or (..., 4) if include_time

        Returns:
            RoPE embeddings of shape (..., dim)
        """
        device = positions.device
        dtype = positions.dtype

        # Split positions into coordinates
        if self.include_time:
            assert positions.shape[-1] == 4, "Expected 4D positions (x, y, z, t)"
            coords = positions  # (..., 4)
        else:
            assert positions.shape[-1] == 3, "Expected 3D positions (x, y, z)"
            coords = positions  # (..., 3)

        # Compute frequencies for each coordinate
        embeddings = []
        for i in range(self.n_coords):
            coord = coords[..., i : i + 1]  # (..., 1)
            # Outer product with frequencies (use 3D freqs)
            angles = coord * self.freqs  # (..., dim_per_coord_3d)
            # Create sin/cos pairs
            emb = self._paired_angles(angles)  # (..., 2*dim_per_coord_3d)
            embeddings.append(emb)

        # Concatenate all coordinate embeddings
        rope_emb = torch.cat(embeddings, dim=-1)  # (..., dim)

        return rope_emb

    def forward_multi_point(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute RoPE for multiple 3D points per token (RenderFormer-style).

        RenderFormer encodes 3 triangle vertices (9D); here Nq anchor points
        (Nq*3 D) represent each object.

        Two modes:
        1. Set aggregation (use_set_aggregation=True): encode each point
           independently with the full 3D encoding, then mean-pool over points.
           No truncation; the result is invariant to point order.
           NOTE: forward_3d returns rotary *angles*, which are linear in the
           position, so the mean over anchors equals the angles of the anchor
           centroid: ARoPE({x_k}) == RoPE(mean_k x_k). The descriptor carries
           the anchor mean position only; shape/extent information reaches
           attention through the anchor features (AVP / encoder), not through
           these phases.
        2. Concatenation (use_set_aggregation=False): encode every coordinate with
           dim_per_coord_multi dims and concatenate, spreading rope_dim across
           all Nq*3 coordinates.

        Args:
            positions: Multi-point positions of shape (..., Nq*3), formatted as
                concat([p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...])

        Returns:
            RoPE embeddings of shape (..., dim)

        Example:
            8 anchors: input (B, Nobj, 24) -> output (B, Nobj, dim)
        """
        device = positions.device
        dtype = positions.dtype

        if self.use_set_aggregation:
            # ===== Set aggregation mode =====
            # Encode each anchor independently, then aggregate

            # Reshape: (..., Nq*3) -> (..., Nq, 3)
            orig_shape = positions.shape[:-1]  # Save original batch shape
            n_coords_total = positions.shape[-1]

            assert (
                n_coords_total % 3 == 0
            ), f"Multi-point positions must be divisible by 3, got {n_coords_total}"
            Nq = n_coords_total // 3

            # Reshape to separate anchors: (..., Nq, 3)
            positions_reshaped = positions.view(*orig_shape, Nq, 3)

            # Flatten batch dimensions: (..., Nq, 3) -> (total_points, 3)
            positions_flat = positions_reshaped.reshape(-1, 3)

            # Encode each anchor independently
            rope_per_anchor = self.forward_3d(positions_flat)  # (total_points, dim)

            # Reshape back: (total_points, dim) -> (..., Nq, dim)
            rope_per_anchor = rope_per_anchor.view(*orig_shape, Nq, -1)

            # Aggregate via mean pooling: (..., Nq, dim) -> (..., dim)
            rope_emb = rope_per_anchor.mean(dim=-2)

        else:
            # ===== Concatenation mode: explicit dim_per_coord, no truncation =====
            # Use explicit dim_per_coord to encode all coords without truncation

            n_coords_total = positions.shape[-1]

            # Use explicit dim_per_coord from config (if provided)
            if self.dim_per_coord_multi is not None:
                dim_per_coord = self.dim_per_coord_multi

                # Verify configuration correctness
                expected_dim = n_coords_total * dim_per_coord
                if expected_dim != self.dim:
                    raise ValueError(
                        f"Dimension mismatch! rope_dim={self.dim} but "
                        f"n_coords_total({n_coords_total}) × dim_per_coord({dim_per_coord}) = {expected_dim}. "
                        f"Please adjust dim_per_coord in config to: {self.dim // n_coords_total}"
                    )
            else:
                # Fallback: auto-compute to avoid truncation
                if self.dim % n_coords_total != 0:
                    raise ValueError(
                        f"rope_dim ({self.dim}) must be divisible by n_coords_total ({n_coords_total}) "
                        f"when dim_per_coord is not explicitly provided. "
                        f"Please set dim_per_coord={self.dim // n_coords_total} in config."
                    )
                dim_per_coord = self.dim // n_coords_total

            # Frequency vector length (each coord -> sin+cos pairs)
            freq_len = dim_per_coord // 2
            if dim_per_coord % 2 != 0:
                raise ValueError(
                    f"dim_per_coord ({dim_per_coord}) must be even (for sin/cos pairs). "
                    f"Got odd value."
                )

            # Generate frequencies for this specific dim_per_coord
            freqs_multi = torch.logspace(
                0, math.log10(self.max_freq), freq_len, device=device, dtype=dtype
            )

            # Encode each coordinate
            embeddings = []
            for i in range(n_coords_total):
                coord = positions[..., i : i + 1]  # (..., 1)
                # Outer product with frequencies
                angles = coord * freqs_multi  # (..., freq_len)
                # Create sin/cos pairs
                emb = self._paired_angles(angles)  # (..., dim_per_coord)
                embeddings.append(emb)

            # Concatenate all coordinate embeddings
            rope_emb = torch.cat(
                embeddings, dim=-1
            )  # (..., n_coords_total * dim_per_coord)

            # Verify: should exactly match self.dim
            assert rope_emb.shape[-1] == self.dim, (
                f"RoPE encoding dimension mismatch: expected {self.dim}, got {rope_emb.shape[-1]}. "
                f"n_coords={n_coords_total}, dim_per_coord={dim_per_coord}"
            )

        return rope_emb

    def forward_sa_rope(self, positions: torch.Tensor) -> torch.Tensor:
        if _memo_usable(positions, self.freqs):
            key = _memo_key("sa", positions, self._memo_extra())
            hit = _ROPE_MEMO.get(key)
            if hit is None:
                hit = (positions, self._forward_sa_rope(positions))
                _memo_put(key, hit)
            return hit[1]
        return self._forward_sa_rope(positions)

    def _forward_sa_rope(self, positions: torch.Tensor) -> torch.Tensor:
        """
        SA-RoPE: Symmetric Anchor RoPE (Deep Sets style).

        Encode each anchor independently via 3D RoPE (high-resolution:
        dim_per_coord_3d dims per coordinate), then mean-pool over anchors.
        Produces a per-object positional descriptor that is **strictly
        permutation-invariant** to anchor ordering within each object.

        Mathematical form: the descriptor is the mean of the per-anchor rotary
        angles a(q_ik) = ω ⊗ q_ik (a linear map of the anchor position),
            ā_i = (1/Nq) Σ_k a(q_ik) = a( (1/Nq) Σ_k q_ik ),
        and attention applies the rotation R(ā_i) to the query/key. Because
        the phases are linear, this equals single-point 3D RoPE evaluated at the
        anchor centroid; the mean is the Deep Sets aggregation that guarantees
        invariance under any anchor permutation π ∈ S_{Nq}. It does not encode
        the anchors' spread (a set scaled about its mean gives the same ā_i).

        Args:
            positions: Multi-point positions (..., Nq*3) where Nq is
                       the number of anchor points per object.
                       Format: [q1_x, q1_y, q1_z, q2_x, q2_y, q2_z, ...]

        Returns:
            RoPE embeddings (..., dim) — permutation-invariant over anchors
        """
        orig_shape = positions.shape[:-1]  # (...,)
        n_coords = positions.shape[-1]
        assert (
            n_coords % 3 == 0
        ), f"SA-RoPE: positions last dim must be multiple of 3, got {n_coords}"
        Nq = n_coords // 3

        # Reshape: (..., Nq*3) -> (..., Nq, 3)
        pos_per_anchor = positions.view(*orig_shape, Nq, 3)

        # Encode each anchor independently via 3D RoPE
        # Flatten for batch processing: (..., Nq, 3) -> (total_anchors, 3)
        pos_flat = pos_per_anchor.reshape(-1, 3)
        rope_per_anchor = self.forward_3d(pos_flat)  # (total_anchors, rope_dim)

        # Reshape back: (total_anchors, rope_dim) -> (..., Nq, rope_dim)
        rope_per_anchor = rope_per_anchor.view(*orig_shape, Nq, -1)

        # Mean pool over anchors: (..., Nq, rope_dim) -> (..., rope_dim)
        # This is the Deep Sets symmetric aggregation — strictly permutation-invariant
        rope_emb = rope_per_anchor.mean(dim=-2)

        return rope_emb

    def forward(
        self,
        positions: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
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


class PhysFormerAttentionWithRoPE(nn.Module):
    """
    Multi-head attention with RoPE for PhysFormer.
    Handles both state-body and body-body interactions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        rope_dim: Optional[int] = None,
        use_3d_rope: bool = True,
        include_time: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # RoPE
        self.rope_dim = rope_dim or self.head_dim
        self.rope = PhysFormerRoPE(
            dim=self.rope_dim, use_3d=use_3d_rope, include_time=include_time
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE.

        Args:
            query: Query tensor (B, N_q, D)
            key: Key tensor (B, N_k, D)
            value: Value tensor (B, N_k, D)
            query_pos: Query positions for RoPE
            key_pos: Key positions for RoPE
            attn_mask: Attention mask

        Returns:
            Attended features (B, N_q, D)
        """
        B, N_q, D = query.shape
        _, N_k, _ = key.shape

        # Reshape for multi-head attention
        query = query.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if positions are provided
        if query_pos is not None:
            query_rope = self.rope(positions=query_pos)
            query_rope = query_rope.view(B, N_q, self.num_heads, -1).transpose(1, 2)
            query = apply_rotary_emb(query_rope, query)

        if key_pos is not None:
            key_rope = self.rope(positions=key_pos)
            key_rope = key_rope.view(B, N_k, self.num_heads, -1).transpose(1, 2)
            key = apply_rotary_emb(key_rope, key)

        # Reshape back for standard attention
        query = query.transpose(1, 2).reshape(B, N_q, D)
        key = key.transpose(1, 2).reshape(B, N_k, D)
        value = value.transpose(1, 2).reshape(B, N_k, D)

        # Apply attention
        out, _ = self.attention(query, key, value, attn_mask=attn_mask)

        return out


class PhysFormerCrossAttentionBlock(nn.Module):
    """
    Cross-attention block with RoPE for state-body interactions.
    """

    def __init__(
        self,
        state_dim: int = 512,
        body_dim: int = 256,
        num_heads: int = 8,
        rope_dim: Optional[int] = None,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project to common dimension
        self.common_dim = max(state_dim, body_dim)
        self.state_proj = (
            nn.Linear(state_dim, self.common_dim)
            if state_dim != self.common_dim
            else nn.Identity()
        )
        self.body_proj = (
            nn.Linear(body_dim, self.common_dim)
            if body_dim != self.common_dim
            else nn.Identity()
        )

        # State -> Body attention (state queries body features)
        self.state_to_body_attn = PhysFormerAttentionWithRoPE(
            self.common_dim,
            num_heads,
            rope_dim=rope_dim,
            use_3d_rope=False,  # State uses 1D encoding
            dropout=dropout,
        )

        # Body -> State attention (body queries state features)
        self.body_to_state_attn = PhysFormerAttentionWithRoPE(
            self.common_dim,
            num_heads,
            rope_dim=rope_dim,
            use_3d_rope=True,  # Body uses 3D encoding
            dropout=dropout,
        )

        # Feedforward networks
        self.state_ffn = nn.Sequential(
            nn.Linear(self.common_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, self.common_dim),
            nn.Dropout(dropout),
        )

        self.body_ffn = nn.Sequential(
            nn.Linear(self.common_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, self.common_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.state_norm1 = nn.LayerNorm(self.common_dim)
        self.state_norm2 = nn.LayerNorm(self.common_dim)
        self.body_norm1 = nn.LayerNorm(self.common_dim)
        self.body_norm2 = nn.LayerNorm(self.common_dim)

    def forward(
        self,
        state_features: torch.Tensor,  # (B, N_state, D_state)
        body_features: torch.Tensor,  # (B, N_body, D_body)
        body_positions: torch.Tensor,  # (B, N_body, 3)
        state_indices: Optional[torch.Tensor] = None,  # (B, N_state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention with RoPE.
        """
        # Project to common dimension
        state_feat = self.state_proj(state_features)
        body_feat = self.body_proj(body_features)

        # State -> Body: State queries body features
        state_feat2 = self.state_to_body_attn(
            query=state_feat,
            key=body_feat,
            value=body_feat,
            query_pos=state_indices,  # 1D indices for state
            key_pos=body_positions,  # 3D positions for bodies
        )
        state_feat = state_feat + state_feat2
        state_feat = self.state_norm1(state_feat)

        # Body -> State: Body queries state features
        body_feat2 = self.body_to_state_attn(
            query=body_feat,
            key=state_feat,
            value=state_feat,
            query_pos=body_positions,  # 3D positions for bodies
            key_pos=state_indices,  # 1D indices for state
        )
        body_feat = body_feat + body_feat2
        body_feat = self.body_norm1(body_feat)

        # Feedforward
        state_feat = state_feat + self.state_ffn(state_feat)
        state_feat = self.state_norm2(state_feat)

        body_feat = body_feat + self.body_ffn(body_feat)
        body_feat = self.body_norm2(body_feat)

        return state_feat, body_feat


class PhysFormerDecoderWithRoPE(nn.Module):
    """
    Complete decoder with RoPE for PhysFormer.
    Combines ideas from CUT3R and RenderFormer.
    """

    def __init__(
        self,
        state_dim: int = 512,
        body_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        rope_dim: Optional[int] = None,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        include_time: bool = True,  # For dynamics
        return_all_layers: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        self.include_time = include_time

        # Create layers
        self.layers = nn.ModuleList(
            [
                PhysFormerCrossAttentionBlock(
                    state_dim=state_dim if i == 0 else max(state_dim, body_dim),
                    body_dim=body_dim if i == 0 else max(state_dim, body_dim),
                    num_heads=num_heads,
                    rope_dim=rope_dim,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        state_features: torch.Tensor,
        body_features: torch.Tensor,
        body_positions: torch.Tensor,
        time_step: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all layers with RoPE.

        Args:
            state_features: State token features (B, N_state, D_state)
            body_features: Body vertex features (B, N_body*8, D_body)
            body_positions: Body vertex 3D positions (B, N_body*8, 3)
            time_step: Optional time for dynamics (scalar or tensor)

        Returns:
            All layer outputs if return_all_layers, else final outputs
        """
        B, N_state, _ = state_features.shape

        # Generate state indices
        state_indices = torch.arange(N_state, device=state_features.device)
        state_indices = state_indices.unsqueeze(0).expand(B, -1)

        # Add time dimension to positions if needed
        if self.include_time and time_step is not None:
            if isinstance(time_step, (int, float)):
                time_tensor = torch.full(
                    (B, body_positions.shape[1], 1),
                    time_step,
                    device=body_positions.device,
                    dtype=body_positions.dtype,
                )
            else:
                time_tensor = time_step.view(B, 1, 1).expand(
                    B, body_positions.shape[1], 1
                )

            body_positions = torch.cat(
                [body_positions, time_tensor], dim=-1
            )  # (B, N, 4)

        # Store outputs
        all_states = []
        all_bodies = []

        # Process through layers
        for layer in self.layers:
            state_features, body_features = layer(
                state_features, body_features, body_positions, state_indices
            )

            all_states.append(state_features)
            all_bodies.append(body_features)

        if self.return_all_layers:
            return all_states, all_bodies
        else:
            return state_features, body_features
