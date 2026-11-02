# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Differentiable intersection handling.

Resolves object-ground and object-object penetrations by translating whole
objects, with a straight-through estimator (STE) keeping the correction
differentiable. The correction formulation follows UNIC (arXiv:2603.25580,
Sec. 3.3), Eq. 8:

    v_corrected = v + detach(target_position - v)
    Forward: moves v to target_position
    Backward: gradients pass through to v unchanged

Whereas UNIC drags individual garment vertices, rigid objects are translated
as a whole:
    - Ground plane: shift the object upward when its lowest vertex falls
      below ground_z + buffer
    - Object-object: push objects apart when their bounding spheres overlap
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def resolve_ground_penetration(
    positions: torch.Tensor,
    obj_mask: torch.Tensor,
    vtx_mask: torch.Tensor,
    ground_z: float = 0.0,
    buffer: float = 0.005,
) -> torch.Tensor:
    """
    Resolve object-ground penetrations by translating objects upward.

    For each valid object, find the LOWEST valid vertex. If min_z < ground_z + buffer,
    shift ALL vertices of that object upward by the deficit.
    This guarantees no valid vertex penetrates the ground, while preserving rigidity.

    Uses STE trick: correction is detached so gradients flow through original positions.

    Args:
        positions: (B, Nobj, Nv, 3) predicted vertex positions
        obj_mask: (B, Nobj) boolean mask for valid objects
        vtx_mask: (B, Nobj, Nv) boolean mask for valid vertices
        ground_z: ground plane height (z-up coordinate system)
        buffer: relaxation buffer above ground plane

    Returns:
        corrected_positions: (B, Nobj, Nv, 3) with ground penetrations resolved
    """
    B, Nobj, Nv, _ = positions.shape
    target_z = ground_z + buffer

    # Find per-object minimum z among valid vertices
    # Set invalid vertices to +inf so they don't affect min
    z_vals = positions[..., 2]  # (B, Nobj, Nv)
    z_vals_masked = z_vals.masked_fill(
        ~vtx_mask, float("inf")
    )  # invalid vertices -> +inf
    min_z, _ = z_vals_masked.min(dim=2)  # (B, Nobj) lowest valid vertex z per object

    # Find objects that penetrate the ground
    penetration_depth = target_z - min_z  # positive means below ground
    # Padding objects have min_z=+inf, giving depth=-inf; clamp to >= 0 before
    # the expand below so 0*(-inf) can never produce NaN
    penetration_depth = penetration_depth.clamp(min=0.0)
    is_penetrating = (penetration_depth > 0) & obj_mask  # (B, Nobj)

    # No early exit on "nothing penetrates": the correction below is identically
    # zero in that case, and the shortcut would cost a host sync per call.

    # Compute correction: shift entire object upward by penetration_depth
    correction = torch.zeros_like(positions)  # (B, Nobj, Nv, 3)
    # Only z-component needs correction
    correction[..., 2] = penetration_depth.unsqueeze(-1).expand(
        -1, -1, Nv
    )  # (B, Nobj, Nv)

    # Zero out correction for non-penetrating objects
    correction = correction * is_penetrating.float().unsqueeze(-1).unsqueeze(
        -1
    )  # (B, Nobj, Nv, 3)

    # STE trick (UNIC Eq. 8): forward moves position, backward passes gradient through
    corrected = positions + correction.detach()

    return corrected


def resolve_object_penetration(
    positions: torch.Tensor,
    obj_mask: torch.Tensor,
    vtx_mask: torch.Tensor,
    buffer: float = 0.005,
) -> torch.Tensor:
    """
    Resolve object-object penetrations using bounding sphere overlap detection.

    For each pair of valid objects, compute COM and bounding sphere radius.
    If spheres overlap, push both objects apart along the COM-COM direction.

    Uses STE trick for differentiability.

    Args:
        positions: (B, Nobj, Nv, 3) predicted vertex positions
        obj_mask: (B, Nobj) boolean mask for valid objects
        vtx_mask: (B, Nobj, Nv) boolean mask for valid vertices
        buffer: relaxation buffer between objects

    Returns:
        corrected_positions: (B, Nobj, Nv, 3) with object penetrations resolved
    """
    B, Nobj, Nv, _ = positions.shape

    if Nobj < 2:
        return positions

    # Compute per-object COM and bounding sphere radius
    vtx_weight = vtx_mask.float().unsqueeze(-1)  # (B, Nobj, Nv, 1)
    num_valid = vtx_weight.sum(dim=2).clamp(min=1)  # (B, Nobj, 1)
    com = (positions * vtx_weight).sum(dim=2) / num_valid  # (B, Nobj, 3)

    # Bounding sphere radius: max distance from COM to any valid vertex
    diff_to_com = positions - com.unsqueeze(2)  # (B, Nobj, Nv, 3)
    dist_to_com = diff_to_com.norm(dim=-1)  # (B, Nobj, Nv)
    # Mask invalid vertices with 0 distance
    dist_to_com = dist_to_com * vtx_mask.float()  # (B, Nobj, Nv)
    radius = dist_to_com.max(dim=2).values  # (B, Nobj)

    # Accumulate corrections per object
    correction = torch.zeros(B, Nobj, 3, device=positions.device, dtype=positions.dtype)

    # Check all pairs (i, j) where i < j
    for i in range(Nobj):
        for j in range(i + 1, Nobj):
            # Both objects must be valid
            pair_valid = obj_mask[:, i] & obj_mask[:, j]  # (B,)
            if not pair_valid.any():
                continue

            # COM distance
            com_diff = com[:, j] - com[:, i]  # (B, 3)
            com_dist = com_diff.norm(dim=-1).clamp(min=1e-8)  # (B,)

            # Minimum allowed distance
            min_dist = radius[:, i] + radius[:, j] + buffer  # (B,)

            # Overlap amount
            overlap = min_dist - com_dist  # (B,) positive means penetrating
            is_overlapping = (overlap > 0) & pair_valid  # (B,)

            if not is_overlapping.any():
                continue

            # Push direction: from i to j (normalized)
            push_dir = com_diff / com_dist.unsqueeze(-1)  # (B, 3)

            # Each object gets pushed by half the overlap
            push_amount = (overlap / 2).clamp(min=0) * is_overlapping.float()  # (B,)

            correction[:, i] -= (
                push_amount.unsqueeze(-1) * push_dir
            )  # push i away from j
            correction[:, j] += (
                push_amount.unsqueeze(-1) * push_dir
            )  # push j away from i

    if (correction.abs() < 1e-10).all():
        return positions

    # Expand correction to per-vertex: (B, Nobj, 3) -> (B, Nobj, Nv, 3)
    correction_expanded = correction.unsqueeze(2).expand(-1, -1, Nv, -1)

    # STE trick
    corrected = positions + correction_expanded.detach()

    return corrected


def handle_intersections(
    positions: torch.Tensor,
    obj_mask: torch.Tensor,
    vtx_mask: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Main entry point: apply all intersection resolution steps.

    Args:
        positions: (B, Nobj, Nv, 3) predicted vertex positions
        obj_mask: (B, Nobj) boolean mask
        vtx_mask: (B, Nobj, Nv) boolean mask for valid vertices
        config: intersection_handling config dict

    Returns:
        corrected positions: (B, Nobj, Nv, 3)
    """
    if not config.get("enable", False):
        return positions

    ground_cfg = config.get("ground", {})
    object_cfg = config.get("object", {})

    # Step 1: Resolve ground penetrations
    if ground_cfg.get("enable", True):
        positions = resolve_ground_penetration(
            positions,
            obj_mask,
            vtx_mask,
            ground_z=ground_cfg.get("z", 0.0),
            buffer=ground_cfg.get("buffer", 0.005),
        )

    # Step 2: Resolve object-object penetrations
    if object_cfg.get("enable", False):
        positions = resolve_object_penetration(
            positions,
            obj_mask,
            vtx_mask,
            buffer=object_cfg.get("buffer", 0.005),
        )

    return positions
