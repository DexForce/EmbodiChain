# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Geometry-based positional encoders for the ARoPE ablation study.

These encoders produce 12D geometry descriptors that replace the ARoPE
anchor-based 12D vector (4 anchors x 3D) in the RoPE encoding path. The
downstream RoPE encoding (12D -> 96D) is reused unchanged.

Baselines:
  - OBBEncoder: OBB center + 3 half-axis vectors (12D)
  - PCAFrameEncoder: centroid + sqrt(eigenvalue)-scaled eigenvectors (12D)

Only the 12D positional descriptor changes; the encoder, decoder, loss,
Kabsch alignment, and training schedule stay identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class OBBEncoder(nn.Module):
    """
    Oriented Bounding Box (OBB) geometry encoder.

    For each object's point cloud, computes:
        g_i^obb = [center(3D), half_axis_1(3D), half_axis_2(3D), half_axis_3(3D)] ∈ R^12

    The half-axis vectors are the PCA eigenvectors scaled by the half-extent of the
    projection along each axis (i.e., half the range of points projected onto each
    eigenvector). This captures the bounding box shape and orientation.

    Handles sign ambiguity by:
      1. Enforcing a right-handed frame (det > 0)
      2. Temporal sign continuity with a reference frame (if provided)
    """

    def __init__(self):
        super().__init__()
        self._prev_axes = None  # Cache for temporal sign continuity

    @torch.no_grad()
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        points: torch.Tensor,
        vtx_mask: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, Nobj, Nv, 3) - current frame positions
            vtx_mask: (B, Nobj, Nv) - valid vertex mask (bool)
            obj_mask: (B, Nobj) - valid object mask (bool), optional

        Returns:
            feat: (B, Nobj, 12) - OBB geometry descriptor
        """
        B, Nobj, Nv, _ = points.shape
        device = points.device
        dtype = points.dtype

        feat = torch.zeros(B, Nobj, 12, device=device, dtype=torch.float32)

        # torch.linalg.eigh does not support bfloat16/float16, and AMP autocast
        # re-casts matmuls even under custom_fwd, so autocast is fully disabled
        # for the geometry computation.
        with torch.amp.autocast("cuda", enabled=False):
            points_f32 = points.float()

            # Process each (batch, object) pair
            for b in range(B):
                for i in range(Nobj):
                    if obj_mask is not None and not obj_mask[b, i]:
                        continue  # padding object: zero feature

                    mask_i = vtx_mask[b, i]  # (Nv,)
                    pts = points_f32[b, i][mask_i]  # (N_valid, 3) in FP32

                    if pts.shape[0] < 3:
                        continue  # degenerate: zero feature

                    # Centroid
                    center = pts.mean(dim=0)  # (3,)

                    # PCA of the covariance matrix
                    pts_c = pts - center
                    cov = (pts_c.T @ pts_c) / max(pts.shape[0] - 1, 1)
                    cov = cov.float()  # eigh requires FP32
                    evals, evecs = torch.linalg.eigh(cov)  # ascending order

                    # Sort descending
                    idx = torch.argsort(evals, descending=True)
                    evals = evals[idx]
                    evecs = evecs[:, idx]

                    # Right-handed frame
                    if torch.det(evecs) < 0:
                        evecs[:, -1] = -evecs[:, -1]

                    # Temporal sign continuity
                    if (
                        self._prev_axes is not None
                        and b < self._prev_axes.shape[0]
                        and i < self._prev_axes.shape[1]
                    ):
                        ref = self._prev_axes[b, i]  # (3, 3)
                        for k in range(3):
                            if torch.dot(evecs[:, k], ref[:, k]) < 0:
                                evecs[:, k] = -evecs[:, k]

                    # Half-extent along each axis (half of projection range)
                    proj = pts_c @ evecs  # (N_valid, 3)
                    half_extents = (
                        proj.max(dim=0).values - proj.min(dim=0).values
                    ) / 2  # (3,)

                    # Scale axes by half-extents
                    axis1 = half_extents[0] * evecs[:, 0]
                    axis2 = half_extents[1] * evecs[:, 1]
                    axis3 = half_extents[2] * evecs[:, 2]

                    feat[b, i] = torch.cat([center, axis1, axis2, axis3], dim=0)

        # Cache axes for temporal continuity (detached from the graph)
        # by extracting axis columns from feat (B, Nobj, 12)
        with torch.no_grad():
            axes_cache = torch.zeros(B, Nobj, 3, 3, device=device, dtype=torch.float32)
            for b in range(B):
                for i in range(Nobj):
                    f = feat[b, i]
                    # Normalize cached axes (direction only)
                    for k in range(3):
                        v = f[3 + k * 3 : 6 + k * 3]
                        norm = v.norm()
                        if norm > 1e-8:
                            axes_cache[b, i, :, k] = v / norm
            self._prev_axes = axes_cache

        return feat


class PCAFrameEncoder(nn.Module):
    """
    PCA (covariance) Frame geometry encoder.

    For each object's point cloud, computes:
        g_i^pca = [center(3D), sqrt(l_1)*u_1(3D), sqrt(l_2)*u_2(3D), sqrt(l_3)*u_3(3D)] ∈ R^12

    This encodes second-order shape moments (covariance eigenframe), distinct from
    OBB which uses half-extents of the bounding box projection.

    Same sign ambiguity handling as OBBEncoder.
    """

    def __init__(self):
        super().__init__()
        self._prev_axes = None

    @torch.no_grad()
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        points: torch.Tensor,
        vtx_mask: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, Nobj, Nv, 3) - current frame positions
            vtx_mask: (B, Nobj, Nv) - valid vertex mask (bool)
            obj_mask: (B, Nobj) - valid object mask (bool), optional

        Returns:
            feat: (B, Nobj, 12) - PCA frame geometry descriptor
        """
        B, Nobj, Nv, _ = points.shape
        device = points.device
        dtype = points.dtype
        eps = 1e-6

        feat = torch.zeros(B, Nobj, 12, device=device, dtype=torch.float32)

        # torch.linalg.eigh does not support bfloat16/float16, and AMP autocast
        # re-casts matmuls even under custom_fwd, so autocast is fully disabled
        # for the geometry computation.
        with torch.amp.autocast("cuda", enabled=False):
            points_f32 = points.float()

            for b in range(B):
                for i in range(Nobj):
                    if obj_mask is not None and not obj_mask[b, i]:
                        continue

                    mask_i = vtx_mask[b, i]
                    pts = points_f32[b, i][mask_i]  # (N_valid, 3) in FP32

                    if pts.shape[0] < 3:
                        continue

                    center = pts.mean(dim=0)
                    pts_c = pts - center
                    cov = (pts_c.T @ pts_c) / max(pts.shape[0] - 1, 1)

                    cov = cov.float()  # eigh requires FP32
                    evals, evecs = torch.linalg.eigh(cov)
                    idx = torch.argsort(evals, descending=True)
                    evals = evals[idx]
                    evecs = evecs[:, idx]

                    # Right-handed frame
                    if torch.det(evecs) < 0:
                        evecs[:, -1] = -evecs[:, -1]

                    # Temporal sign continuity
                    if (
                        self._prev_axes is not None
                        and b < self._prev_axes.shape[0]
                        and i < self._prev_axes.shape[1]
                    ):
                        ref = self._prev_axes[b, i]
                        for k in range(3):
                            if torch.dot(evecs[:, k], ref[:, k]) < 0:
                                evecs[:, k] = -evecs[:, k]

                    # Scale by sqrt(eigenvalue) — standard deviation along each axis
                    scales = torch.sqrt(torch.clamp(evals, min=eps))

                    axis1 = scales[0] * evecs[:, 0]
                    axis2 = scales[1] * evecs[:, 1]
                    axis3 = scales[2] * evecs[:, 2]

                    feat[b, i] = torch.cat([center, axis1, axis2, axis3], dim=0)

        # Cache axes for temporal continuity
        with torch.no_grad():
            axes_cache = torch.zeros(B, Nobj, 3, 3, device=device, dtype=torch.float32)
            for b in range(B):
                for i in range(Nobj):
                    f = feat[b, i]
                    for k in range(3):
                        v = f[3 + k * 3 : 6 + k * 3]
                        norm = v.norm()
                        if norm > 1e-8:
                            axes_cache[b, i, :, k] = v / norm
            self._prev_axes = axes_cache

        return feat
