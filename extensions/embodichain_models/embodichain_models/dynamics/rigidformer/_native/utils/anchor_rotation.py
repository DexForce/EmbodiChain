# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""Rotation part of an anchor-position error, in units of the object's size."""

from __future__ import annotations
import torch


def centered_anchor_error(
    pred: torch.Tensor, gt: torch.Tensor, min_radius: float = 1e-3
) -> torch.Tensor:
    """pred, gt: (..., Nq, 3) anchor positions of rigid objects. Returns (..., Nq, 3): the error without its mean over the
    anchors (the translation part), divided by the RMS distance of the ground-truth anchors from their centroid.

    For a rotation error of angle theta and no other error, the RMS norm of the result is 2 sin(theta / 2) times the
    RMS component of the centred anchors perpendicular to the rotation axis over their RMS radius: about theta, and the
    same for a small and a large object."""
    gt_c = gt - gt.mean(dim=-2, keepdim=True)
    pred_c = pred - pred.mean(dim=-2, keepdim=True)
    radius = (
        gt_c.pow(2)
        .sum(dim=-1)
        .mean(dim=-1, keepdim=True)
        .sqrt()
        .unsqueeze(-1)
        .clamp_min(min_radius)
    )
    return (pred_c - gt_c) / radius
