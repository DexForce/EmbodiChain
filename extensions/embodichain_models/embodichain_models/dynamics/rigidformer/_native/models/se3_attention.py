# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Pairwise relative SE(3) attention bias.

Baseline for the positional-encoding ablation: replaces ARoPE's anchor-based
rotary position encoding with pairwise relative SE(3) features between
object pairs.

Design notes:
- Translation uses the current object center c_i (not the Kabsch t_i)
- Relative translations are normalized by the RMS scale s_i
- 7D pairwise feature: [t_ij, omega_ij, log(s_j / s_i)]
- Last MLP layer is zero-initialized, so attention starts unbiased
- Padding pairs receive a finite -1e4 bias (avoids NaN from -inf)
- Outputs are re-masked by obj_mask after attention

Used only when pe_type='se3'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_object_pose_features(
    curr_pos: torch.Tensor,  # (B, Nobj, Nv, 3) - current frame positions
    ref_pose: torch.Tensor,  # (B, Nobj, Nv, 3) - reference pose (frame 0)
    vtx_mask: torch.Tensor,  # (B, Nobj, Nv) - valid vertex mask
    obj_mask: torch.Tensor,  # (B, Nobj) - valid object mask
    eps: float = 1e-6,
) -> tuple:
    """
    Extract per-object rigid pose features using center-based Kabsch.

    For each valid object i:
      1. Compute centers: c_ref = mean(X_ref[valid]), c_i = mean(X_t[valid])
      2. Kabsch on centered points: R_i = Kabsch(X_ref - c_ref, X_t - c_i)
      3. Scale: s_i = RMS radius of reference point cloud

    Returns:
        rotmats: (B, Nobj, 3, 3) - rotation matrices (identity for padding)
        centers: (B, Nobj, 3)    - current frame centers (zero for padding)
        scales:  (B, Nobj, 1)    - RMS radius of ref point cloud (1.0 for padding)
    """
    B, Nobj, Nv, _ = curr_pos.shape
    device = curr_pos.device
    dtype = curr_pos.dtype

    rotmats = torch.eye(3, device=device, dtype=dtype)[None, None].repeat(B, Nobj, 1, 1)
    centers = torch.zeros(B, Nobj, 3, device=device, dtype=dtype)
    scales = torch.ones(B, Nobj, 1, device=device, dtype=dtype)

    for b in range(B):
        for i in range(Nobj):
            if not obj_mask[b, i]:
                continue

            valid = vtx_mask[b, i].bool()
            n_valid = valid.sum()
            if n_valid == 0:
                continue

            Xr = ref_pose[b, i][valid]  # (Ni, 3) - reference
            Xt = curr_pos[b, i][valid]  # (Ni, 3) - current

            # Compute centers
            c_ref = Xr.mean(dim=0)  # (3,)
            c_cur = Xt.mean(dim=0)  # (3,)

            # The translation feature is the current center, not the Kabsch t_i
            centers[b, i] = c_cur

            if n_valid < 3:
                # Not enough points for Kabsch, keep identity R
                continue

            # Center the point clouds
            Xr0 = Xr - c_ref  # (Ni, 3)
            Xt0 = Xt - c_cur  # (Ni, 3)

            # Kabsch alignment on centered points yields the rotation only
            try:
                Ri, _ = roma.rigid_points_registration(Xr0, Xt0)
                rotmats[b, i] = Ri
            except Exception:
                # Degenerate case (e.g., highly symmetric object), keep identity
                pass

            # RMS radius of reference point cloud (scale normalization)
            si = torch.sqrt(torch.clamp(Xr0.pow(2).sum(dim=-1).mean(), min=eps))
            scales[b, i, 0] = si

    return rotmats, centers, scales


class SE3PairwiseBias(nn.Module):
    """
    Compute pairwise relative SE(3) attention bias between object pairs.

    7D pairwise feature:
      t_ij = R_i^T (c_j - c_i) / s_i  — scale-normalized relative translation (3D)
      omega_ij = log_SO3(R_i^T R_j)   — relative rotation as axis-angle (3D)
      log(s_j / s_i)                  — log scale ratio (1D)

    An MLP maps the 7D feature to a per-head bias; its last layer is zero-initialized.
    """

    def __init__(
        self, num_heads: int, hidden_dim: int = 64, use_scale_ratio: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_scale_ratio = use_scale_ratio
        d_in = 7 if use_scale_ratio else 6

        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

        # Zero-init last layer: start as vanilla attention (no geometric bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self._logged = False

    def forward(
        self,
        rotmats: torch.Tensor,  # (B, Nobj, 3, 3)
        centers: torch.Tensor,  # (B, Nobj, 3)
        scales: torch.Tensor,  # (B, Nobj, 1)
        obj_mask: torch.Tensor,  # (B, Nobj)
    ) -> torch.Tensor:
        """
        Compute pairwise SE(3) attention bias.

        Returns:
            bias: (B, num_heads, Nobj, Nobj) - additive attention bias
        """
        B, Nobj = obj_mask.shape

        # R_i^T: (B, Nobj, 3, 3)
        Ri_T = rotmats.transpose(-1, -2)

        # Relative translation: t_ij = R_i^T (c_j - c_i) / s_i
        ci = centers.unsqueeze(2)  # (B, Nobj, 1, 3)
        cj = centers.unsqueeze(1)  # (B, 1, Nobj, 3)
        delta_c = cj - ci  # (B, Nobj, Nobj, 3)

        Ri_T_expanded = Ri_T.unsqueeze(2)  # (B, Nobj, 1, 3, 3)
        t_ij = torch.matmul(Ri_T_expanded, delta_c.unsqueeze(-1)).squeeze(
            -1
        )  # (B, Nobj, Nobj, 3)

        # Scale normalization: divide by s_i
        t_ij = t_ij / scales.clamp_min(1e-6).unsqueeze(2)  # (B, Nobj, 1, 1) broadcast

        # Relative rotation: R_ij = R_i^T @ R_j, then log map to axis-angle
        Rj = rotmats.unsqueeze(1)  # (B, 1, Nobj, 3, 3)
        R_ij = torch.matmul(Ri_T_expanded, Rj)  # (B, Nobj, Nobj, 3, 3)
        omega_flat = roma.rotmat_to_rotvec(R_ij.reshape(-1, 3, 3))
        omega_ij = omega_flat.reshape(B, Nobj, Nobj, 3)

        # Build feature vector
        feats = [t_ij, omega_ij]

        if self.use_scale_ratio:
            # log(s_j / s_i) = log(s_j) - log(s_i)
            log_s = scales.clamp_min(1e-6).log().squeeze(-1)  # (B, Nobj)
            log_ratio = log_s.unsqueeze(2) - log_s.unsqueeze(1)  # (B, Nobj, Nobj)
            feats.append(log_ratio.unsqueeze(-1))

        phi_ij = torch.cat(feats, dim=-1)  # (B, Nobj, Nobj, 7)

        # MLP to per-head bias
        bias = self.mlp(phi_ij)  # (B, Nobj, Nobj, H)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, Nobj, Nobj)

        # Mask padding objects: use -1e4 instead of -inf (avoids NaN issues)
        pair_mask = obj_mask[:, :, None] & obj_mask[:, None, :]  # (B, Nobj, Nobj)
        neg = torch.full_like(bias, -1e4)
        bias = torch.where(pair_mask.unsqueeze(1), bias, neg)

        if not self._logged:
            n_valid = obj_mask.sum(dim=1).float().mean().item()
            valid_bias = bias[pair_mask.unsqueeze(1).expand_as(bias)]
            logger.info(
                f"[SE3 Bias] Computed pairwise SE(3) bias: "
                f"shape={tuple(bias.shape)}, "
                f"avg_valid_objects={n_valid:.1f}, "
                f"phi_ij dim={phi_ij.shape[-1]}, "
                f"bias range=[{valid_bias.min().item():.6f}, {valid_bias.max().item():.6f}] "
                f"(zero-init: expected near 0)"
            )
            self._logged = True

        return bias


class SelfAttentionWithSE3Bias(nn.Module):
    """
    Self-attention with pairwise SE(3) attention bias (no RoPE).

    Drop-in replacement for SelfAttentionWithRoPE when pe_type='se3'.
    Uses PyTorch SDPA instead of Flash Attention (FA2 doesn't support custom attn_bias).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        qk_norm: bool = True,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm
        self.elementwise_gate = elementwise_gate

        # Projections
        if self.elementwise_gate:
            self.q_proj = nn.Linear(d_model, d_model * 2, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout

        # QK-normalization
        if self.qk_norm:
            from embodichain_models.dynamics.rigidformer._native.models.decoder import (
                RMSNorm,
            )

            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._logged = False

    def forward(
        self,
        query: torch.Tensor,  # (B, N, D)
        key_value: torch.Tensor,  # (B, N, D) - same as query for self-attn
        se3_bias: torch.Tensor,  # (B, H, N, N) - pairwise SE(3) bias
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, N) object mask
    ) -> torch.Tensor:
        """
        Self-attention with SE(3) pairwise bias (no RoPE).
        """
        B, N, _ = query.shape

        # Q projection
        if self.elementwise_gate:
            q_gate = self.q_proj(query)
            q_states, gate_score = torch.split(q_gate, self.d_model, dim=-1)
            q = q_states.view(B, N, self.num_heads, self.head_dim)
            gate_score = gate_score.view(B, N, self.num_heads, self.head_dim)
        else:
            q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim)
            gate_score = None

        k = self.k_proj(key_value).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, N, self.num_heads, self.head_dim)

        # QK-normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Transpose to (B, H, N, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # The SE(3) bias serves directly as the additive attention bias; padding
        # pairs already carry -1e4, so no separate padding mask is needed.
        attn_bias = se3_bias.to(dtype=q.dtype)

        # PyTorch SDPA
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, scale=self.scale
        )  # (B, H, N, D)

        # Safety: replace any NaN from degenerate softmax
        out = torch.nan_to_num(out, nan=0.0)

        # Transpose back to (B, N, H, D)
        out = out.transpose(1, 2)

        if self.elementwise_gate:
            out = out * torch.sigmoid(gate_score)

        # Merge heads
        out = out.reshape(B, N, self.d_model)
        out = self.out_proj(out)

        # Zero out padding query outputs after out_proj so its bias is zeroed as well
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                query_mask = attn_mask[:, 0, 0, :]  # (B, N)
            else:
                query_mask = attn_mask  # (B, N)
            out = out * query_mask.unsqueeze(-1).to(out.dtype)  # (B, N, 1)

        if not self._logged:
            valid_bias = se3_bias[se3_bias > -1e3]
            if valid_bias.numel() > 0:
                logger.info(
                    f"[SE3 SelfAttn] Forward: "
                    f"q={tuple(q.shape)}, se3_bias={tuple(se3_bias.shape)}, "
                    f"valid bias range=[{valid_bias.min().item():.6f}, {valid_bias.max().item():.6f}]"
                )
            self._logged = True

        return out
