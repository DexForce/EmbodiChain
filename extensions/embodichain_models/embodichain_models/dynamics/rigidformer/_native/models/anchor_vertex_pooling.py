# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Anchor-Vertex Pooling (AVP) — collision-aware anchor feature enrichment.

Each anchor aggregates per-vertex features from its spatial neighborhood using
exponential distance weighting exp(-||a - v|| / sigma). This gives anchors
collision-aware representations without requiring vertex-level attention
(O(Na*Nv) instead of O(Nv^2)).

Properties:
- Translation consistent: weights depend only on the relative distance ||anchor - vertex||
- Permutation equivariant: operates per-object independently
- Learnable sigma: the model learns the neighborhood radius
"""

from __future__ import annotations

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AnchorVertexPooling(nn.Module):
    """
    Distance-based attention pooling from vertices to anchors.

    For each anchor, computes exponential distance weights exp(-||a - v|| / sigma)
    over all vertices, then pools per-vertex features with the normalized weights.

    Args:
        vertex_dim: Dimension of per-vertex features from the encoder backbone (e.g. 1024)
        output_dim: Output dimension per anchor (e.g. 256)
        init_sigma: Initial value of the learnable kernel radius sigma
    """

    def __init__(
        self, vertex_dim: int = 1024, output_dim: int = 256, init_sigma: float = 0.1
    ):
        super().__init__()
        self.vertex_dim = vertex_dim
        self.output_dim = output_dim

        # Learnable neighborhood radius (log-space for numerical stability)
        # sigma = exp(log_sigma), so sigma is always positive
        self.log_sigma = nn.Parameter(torch.tensor(float(init_sigma)).log())

        # Projection: vertex_dim -> output_dim
        self.proj = nn.Sequential(
            nn.Linear(vertex_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Zero-init last layer: the module starts out contributing near-zero features
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

        self._logged = False

    def forward(
        self,
        vertex_features: torch.Tensor,  # (B_flat, Nv, vertex_dim)
        vertex_positions: torch.Tensor,  # (B_flat, Nv, 3)
        anchor_positions: torch.Tensor,  # (B_flat, Na, 3)
        vtx_mask: torch.Tensor,  # (B_flat, Nv) bool
    ) -> torch.Tensor:
        """
        Args:
            vertex_features:  (B_flat, Nv, vertex_dim) — per-vertex backbone features;
                              the caller must transpose from the encoder's (B_flat, vertex_dim, Nv)
            vertex_positions: (B_flat, Nv, 3) — current frame vertex positions
            anchor_positions: (B_flat, Na, 3) — current frame anchor positions
            vtx_mask:         (B_flat, Nv) — True for valid vertices, False for padding

        Returns:
            anchor_features:  (B_flat, Na, output_dim)
        """
        B_flat, Nv, D = vertex_features.shape
        Na = anchor_positions.shape[1]

        # Compute pairwise spatial distance: (B_flat, Na, Nv)
        # anchor_positions: (B_flat, Na, 1, 3) - vertex_positions: (B_flat, 1, Nv, 3)
        diff = anchor_positions.unsqueeze(2) - vertex_positions.unsqueeze(
            1
        )  # (B_flat, Na, Nv, 3)
        dist = diff.norm(dim=-1)  # (B_flat, Na, Nv)

        # Exponential distance weights
        sigma = self.log_sigma.exp()  # always positive
        attn = torch.exp(-dist / (sigma + 1e-8))  # (B_flat, Na, Nv)

        # Mask padding vertices (set weight to 0)
        attn = attn * vtx_mask.unsqueeze(1).float()  # (B_flat, Na, Nv)

        # Normalize (sum over vertices)
        attn_sum = attn.sum(dim=-1, keepdim=True)  # (B_flat, Na, 1)
        attn = attn / (attn_sum + 1e-8)  # (B_flat, Na, Nv)

        # Weighted pooling of vertex features
        pooled = torch.bmm(attn, vertex_features)  # (B_flat, Na, vertex_dim)

        # Project to output dimension
        output = self.proj(pooled)  # (B_flat, Na, output_dim)

        # One-time diagnostic log
        if not self._logged:
            n_valid_vtx = vtx_mask.sum().item()
            n_total_vtx = vtx_mask.numel()
            n_fully_masked = (
                (vtx_mask.sum(dim=-1) == 0).sum().item()
            )  # objects with all vertices masked
            attn_max_on_masked = (
                (attn * (~vtx_mask).unsqueeze(1).float()).max().item()
            )  # expected 0
            output_has_nan = torch.isnan(output).any().item()
            logger.info(
                f"[AVP] Anchor-Vertex Pooling: "
                f"vertex_feat=({B_flat},{Nv},{D}), anchors={Na}, "
                f"sigma={sigma.item():.4f}, output=({B_flat},{Na},{self.output_dim})"
            )
            logger.info(
                f"[AVP] Masking check: valid_vtx={n_valid_vtx}/{n_total_vtx} "
                f"({100*n_valid_vtx/max(n_total_vtx,1):.1f}%), "
                f"fully_masked_objs={n_fully_masked}/{B_flat}, "
                f"attn_on_padding={attn_max_on_masked:.2e} (expected 0), "
                f"has_nan={output_has_nan}"
            )
            self._logged = True

        return output
