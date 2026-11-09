# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tensor interpolation of sampled gripper surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from embodichain.utils.math import pose_inv

__all__ = [
    "interpolate_hand_samples",
    "full_object_opening_mask",
    "GraspApproach",
    "adaptive_grasp_approaches",
    "opposite_grasp_end",
]


def interpolate_hand_samples(
    body: torch.Tensor,
    pads: torch.Tensor,
    gaps: torch.Tensor,
    widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate closing samples at candidate pad gaps, clamping at endpoints.

    Args:
        body: Body samples of shape ``(G, V, 3)``.
        pads: Two pad surfaces of shape ``(G, 2, P, 3)``.
        gaps: Strictly descending sample gaps of shape ``(G,)``, with G >= 2.
        widths: Candidate pad gaps of shape ``(K,)``; owns output dtype/device.

    Returns:
        Interpolated body and pad samples, with leading candidate dimension K.
        Inputs are not modified. Numeric validity and monotonicity belong to the
        geometry-construction boundary, avoiding repeated device synchronization.
    """
    if body.ndim != 3 or pads.ndim != 4 or gaps.ndim != 1 or widths.ndim != 1:
        raise ValueError("Invalid sampled hand tensor dimensions.")
    if len(gaps) < 2 or body.shape[0] != len(gaps) or pads.shape[0] != len(gaps):
        raise ValueError("Hand surfaces must share at least two gap samples.")
    if body.shape[-1] != 3 or pads.shape[1] != 2 or pads.shape[-1] != 3:
        raise ValueError("Hand geometry requires XYZ samples and two pads.")
    gaps = gaps.to(widths)
    high = torch.searchsorted(-gaps.contiguous(), -widths).clamp(1, len(gaps) - 1)
    low = high - 1
    alpha = ((gaps[low] - widths) / (gaps[low] - gaps[high])).clamp(0, 1)
    body, pads = body.to(widths), pads.to(widths)
    return (
        torch.lerp(body[low], body[high], alpha[:, None, None]),
        torch.lerp(pads[low], pads[high], alpha[:, None, None, None]),
    )


def full_object_opening_mask(
    vertices: torch.Tensor,
    poses: torch.Tensor,
    object_pose: torch.Tensor,
    opening: float,
) -> torch.Tensor:
    """Require the full local mesh to fit between open pads for each proposal.

    Args:
        vertices: Object-local points, shape (V, 3).
        poses: Candidate world grasp poses, shape (K, 4, 4).
        object_pose: Current object-to-world pose, broadcastable over candidates.
        opening: Calibrated maximum pad opening, in metres.

    Returns:
        Candidate validity mask. The source two-millimetre pad reserve and
        ordered 64-pose chunks remain unchanged; no physical grip is inferred.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Opening clearance requires non-empty mesh vertices (N, 3).")
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or object_pose.shape not in ((4, 4), (1, 4, 4))
    ):
        raise ValueError(
            "Opening clearance needs candidate poses (K,4,4) and one object pose."
        )
    if not math.isfinite(opening) or opening <= 0:
        raise ValueError("Opening clearance needs a positive finite opening.")
    relative = pose_inv(object_pose) @ poses
    accepted = torch.isfinite(relative).all(-1).all(-1)
    # Leave clearance at both pads beyond the millimetre-scale contact envelope.
    half_gap = opening * 0.5 - 0.002
    for start in range(0, len(poses), 64):
        chunk = relative[start : start + 64]
        axes = chunk[:, :3, 0]
        centers = (chunk[:, :3, 3] * axes).sum(-1)
        projection = vertices.to(poses) @ axes.T - centers[None]
        accepted[start : start + len(chunk)] &= torch.isfinite(projection).all(0) & (
            projection.abs().amax(0) <= half_gap
        )
    return accepted


@dataclass(frozen=True, slots=True)
class GraspApproach:
    """Read-only numeric direction and optional object-axis region for one trial."""

    direction: torch.Tensor
    axis: torch.Tensor | None
    positive: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.direction.ndim != 2
            or self.direction.shape[1] != 3
            or not torch.isfinite(self.direction).all()
        ):
            raise ValueError("Grasp approach directions must be finite (B,3).")
        if self.axis is not None and (
            self.axis.shape != self.direction.shape
            or not torch.isfinite(self.axis).all()
        ):
            raise ValueError("Grasp approach axes must be finite (B,3).")
        if (
            self.positive.shape != self.direction.shape[:1]
            or self.positive.dtype != torch.bool
        ):
            raise ValueError("Grasp approach region signs must be boolean (B,).")
        for name in ("direction", "axis", "positive"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.detach().clone())


def adaptive_grasp_approaches(
    axis: torch.Tensor,
    pose: torch.Tensor,
    tcp: torch.Tensor,
    *,
    ends: bool,
    positive: torch.Tensor | None = None,
    purpose: str = "pour",
) -> tuple[GraspApproach, ...]:
    """Order declared geometric approach alternatives without robot-name inference.

    Args:
        axis: Observed unit object axis, shape (B, 3).
        pose: Observed object poses, shape (B, 4, 4).
        tcp: Observed TCP poses with the same batch.
        ends: Whether opposite object-axis end regions are allowed.
        positive: Optional exact per-row end-region selection.
        purpose: Declared manipulation intent selecting source order.

    Returns:
        Ordered vertical/diagonal approaches and optional axis regions. The
        source direction, region and tie order are unchanged.
    """
    if purpose not in {"ordinary", "pour", "handover_source", "handover_receive"}:
        raise ValueError(f"Unknown grasp purpose: {purpose!r}.")
    down = torch.zeros_like(axis)
    down[:, 2] = -1
    toward = pose[:, :3, 3] - tcp[:, :3, 3]
    toward[:, 2] = 0
    length = torch.linalg.vector_norm(toward, dim=-1, keepdim=True)
    diagonal = toward / length.clamp_min(1e-6) * math.sin(math.pi / 3)
    diagonal[:, 2] = -0.5
    diagonal = torch.where(length > 1e-6, diagonal, down)
    vertical = axis[:, 2:3].abs() >= math.sqrt(0.5)
    directions = (
        (down, diagonal)
        if purpose in {"ordinary", "pour", "handover_source"}
        else (
            torch.where(vertical, diagonal, down),
            torch.where(vertical, down, diagonal),
        )
    )
    nearest = ((tcp[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1) >= 0
    parts = (
        (positive,)
        if positive is not None
        else ((nearest, ~nearest) if ends else (nearest,))
    )
    if purpose == "handover_source":
        # Preserve the source-side end before trying a crossed handover layout.
        return tuple(
            GraspApproach(direction, axis if ends else None, part)
            for part in parts
            for direction in directions
        )
    return tuple(
        GraspApproach(direction, axis if ends else None, part)
        for direction in directions
        for part in parts
    )


def opposite_grasp_end(
    axis: torch.Tensor,
    pose: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
) -> torch.Tensor:
    """Choose the receiver end opposite the source's projected grasp.

    Args:
        axis: Observed unit object axes ``(B,3)``.
        pose: Object poses ``(B,4,4)``.
        source: Current source TCP poses in the same world frame.
        destination: Current receiver TCP poses in the same world frame.

    Returns:
        Positive-end mask. Source projections within 1e-5 of the center retain
        the receiver-nearest end, including the original exact-zero tie.
    """
    projection = ((source[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1)
    near_destination = ((destination[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1) >= 0
    return torch.where(projection.abs() > 1e-5, projection < 0, near_destination)
