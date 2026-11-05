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

"""Pure source-only gravity-top proposals, never poured or physical evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypedDict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

__all__ = [
    "source_pour_staging_frame",
    "PourFrameSet",
    "JawPourFrameSet",
    "PourGraspProjection",
    "gravity_top_frame",
    "pour_staging_pose",
    "pour_frames",
    "jaw_tilt_axis",
    "jaw_pour_frames",
]


class PourFrameSet(TypedDict):
    """Owned B=1 numeric tilt arc and ordered candidate-projection targets.

    Arrays are independent of caller input buffers. The five targets retain
    staging, peak, restore, return transport and return placement order; they
    establish no IK, collision, physical retention or liquid-transfer evidence.
    """

    object_arc: np.ndarray
    object_targets: np.ndarray
    target_roles: tuple[str, ...]
    pivot_object: np.ndarray
    pivot_world: np.ndarray
    up_object: np.ndarray
    tilt_axis_object: np.ndarray
    effective_angle_rad: float
    peak_index: int
    receiver_bearing_m: float
    receiver_bearing_tie: bool
    tilt_direction_policy: str
    physical_evidence: bool


class JawPourFrameSet(TypedDict):
    """Owned tensor arc and ordered-jaw diagnostics for the legacy convention."""

    object_poses: torch.Tensor
    tilt_axis_object: torch.Tensor
    tilt_direction_object: torch.Tensor


_UP = np.array([0.0, 0.0, 1.0])
_BEARING_TIE_M = 1e-5
_AXIS_EPS = 1e-9


def _pose(value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("Pour geometry requires a finite real SE3 pose.")
    pose = np.array(value, dtype=np.float64, copy=True)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5, rtol=0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=2e-5, rtol=0)
    ):
        raise ValueError("Pour geometry requires a finite proper SE3 pose.")
    return pose


def _vertices(value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("Pour geometry requires finite real native vertices.")
    vertices = np.array(value, dtype=np.float64, copy=True)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Pour geometry requires nonempty finite native vertices.")
    return vertices


def _gravity(value: np.ndarray) -> None:
    gravity = np.asarray(value)
    if (
        gravity.shape != (3,)
        or np.iscomplexobj(gravity)
        or not np.isfinite(gravity).all()
        or gravity[0] != 0
        or gravity[1] != 0
        or gravity[2] >= 0
    ):
        raise ValueError(
            "Pour geometry requires finite nonzero negative-Z World gravity."
        )


def gravity_top_frame(
    vertices: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the object-local gravity-up axis and source top-plane pivot.

    Args:
        vertices: Nonempty validated native vertices, shape ``(V, 3)``. The
            input dtype is preserved for the source bounds/center reductions.
        rotation: Validated proper ``(3, 3)`` rotation from object to world.

    Returns:
        Owned local up-axis and top-plane pivot arrays. This geometric point
        is neither a detected spout nor a fluid-transfer observation. Callers
        own pose/mesh validation; this kernel preserves the original arithmetic.
    """
    up = rotation.T @ _UP
    up /= np.linalg.norm(up)
    center = (vertices.min(0) + vertices.max(0)) / 2
    pivot = center + up * ((vertices @ up).max() - center @ up)
    return up, pivot


def pour_staging_pose(
    source_vertices: np.ndarray,
    receiver_vertices: np.ndarray,
    observed_source_pose: np.ndarray,
    observed_receiver_pose: np.ndarray,
    *,
    clearance_m: float,
    gravity_world: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Preserve observed entry R in a conservative source-only staging proposal.

    Args:
        source_vertices: Already-scaled source vertices in its runtime-local basis.
        receiver_vertices: Already-scaled receiver vertices in its runtime-local basis.
        observed_source_pose: Current unbatched source arena SE3, not initial-R.
        observed_receiver_pose: Current unbatched receiver arena SE3.
        clearance_m: Explicit nonnegative source-envelope clearance in meters.
        gravity_world: Actual World gravity; this MVP requires finite negative-Z.

    Returns:
        A fresh float64 4x4 pose retaining every observed rotation value. The
        source sweep sphere certifies neither the hand, a receiver opening,
        a collision-free path nor native bearing. No rotation is reprojected.
    """
    _gravity(gravity_world)
    source, receiver = _pose(observed_source_pose), _pose(observed_receiver_pose)
    if (
        isinstance(clearance_m, (bool, np.bool_))
        or not isinstance(clearance_m, Real)
        or not math.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("clearance_m must be finite and nonnegative.")
    vertices, receiving = _vertices(source_vertices), _vertices(receiver_vertices)
    _, pivot = gravity_top_frame(vertices, source[:3, :3])
    # Bound float32 matrix stretch without normalizing the observed R.
    stretch = np.sqrt(np.abs(source[:3, :3].T @ source[:3, :3]).sum(axis=1).max())
    radius = stretch * np.linalg.norm(vertices - pivot, axis=-1).max()
    center = (
        receiver[:3, :3] @ ((receiving.min(0) + receiving.max(0)) / 2) + receiver[:3, 3]
    )
    world = receiving @ receiver[:3, :3].T + receiver[:3, 3]
    anchor = center.copy()
    anchor[2] = world[:, 2].max() + radius + clearance_m
    result = source.copy()
    result[:3, 3] = anchor - source[:3, :3] @ pivot
    return result


def pour_frames(
    source_vertices: np.ndarray,
    receiver_vertices: np.ndarray,
    observed_staging_pose: np.ndarray,
    observed_receiver_pose: np.ndarray,
    jaw_line_object: np.ndarray,
    declared_angle_rad: float,
    return_pose: np.ndarray,
    *,
    sample_count: int,
    gravity_world: np.ndarray | tuple[float, float, float],
) -> PourFrameSet:
    """Build one fixed-pivot tilt/restore arc using an ordered object-local jaw.

    Args:
        source_vertices: Already-scaled source vertices in runtime-local meters.
        receiver_vertices: Already-scaled receiver vertices in runtime-local meters.
        observed_staging_pose: Current unbatched source arena pose at Pour entry.
        observed_receiver_pose: Current unbatched receiver arena pose.
        jaw_line_object: Ordered pad-line vector in the source object's frame.
        declared_angle_rad: Nonzero tilt magnitude; no explicit sign is owned.
        return_pose: Explicit measured-entry return SE3; never inferred here.
        sample_count: Number of samples including entry, peak and restored end.
        gravity_world: Actual World gravity; this MVP requires finite negative-Z.

    Returns:
        ``object_arc`` (N,4,4), ``object_targets`` (5,4,4), ordered ``target_roles``
        and axis/pivot/bearing diagnostics. Targets are staging, peak, restore,
        return transport and return Place. ``physical_evidence`` is always False.
        The caller owns candidate G/TCP composition and semantic/physical gates.
    """
    _gravity(gravity_world)
    stage, receiver, returned = (
        _pose(observed_staging_pose),
        _pose(observed_receiver_pose),
        _pose(return_pose),
    )
    vertices, receiving = _vertices(source_vertices), _vertices(receiver_vertices)
    if np.iscomplexobj(jaw_line_object):
        raise ValueError("A finite real object-local jaw line is required.")
    jaw = np.array(jaw_line_object, dtype=np.float64, copy=True)
    norm = np.linalg.norm(jaw)
    if (
        jaw.shape != (3,)
        or not np.isfinite(jaw).all()
        or not math.isfinite(norm)
        or norm <= _AXIS_EPS
    ):
        raise ValueError("A finite nonzero object-local ordered jaw line is required.")
    if (
        isinstance(declared_angle_rad, (bool, np.bool_))
        or not isinstance(declared_angle_rad, Real)
        or not math.isfinite(declared_angle_rad)
        or declared_angle_rad == 0
    ):
        raise ValueError("A finite nonzero declared angle is required.")
    if type(sample_count) is not int or sample_count < 3:
        raise ValueError("sample_count must be an integer >=3.")
    up, pivot = gravity_top_frame(vertices, stage[:3, :3])
    jaw /= norm
    axis = jaw - (jaw @ up) * up
    magnitude = np.linalg.norm(axis)
    if magnitude <= _AXIS_EPS:
        raise ValueError(
            "Jaw axis is parallel to gravity; projected tilt axis is degenerate."
        )
    axis /= magnitude
    anchor = stage[:3, :3] @ pivot + stage[:3, 3]
    center = (
        receiver[:3, :3] @ ((receiving.min(0) + receiving.max(0)) / 2) + receiver[:3, 3]
    )
    visible = stage[:3, :3] @ np.cross(axis, up)
    visible /= np.linalg.norm(visible)
    bearing = float((center - anchor) @ visible)
    tie = abs(bearing) <= _BEARING_TIE_M
    # The ordered jaw owns the axis, not a receiver-facing spout.
    # Millimetric tracking error must not reverse the qualified tilt arc.
    theta, peak = abs(float(declared_angle_rad)), sample_count // 2
    angles = np.concatenate(
        (
            np.linspace(0, theta, peak + 1),
            np.linspace(theta, 0, sample_count - peak)[1:],
        )
    )
    rotations = stage[:3, :3] @ Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
    arc = np.tile(np.eye(4), (sample_count, 1, 1))
    arc[:, :3, :3], arc[:, :3, 3] = rotations, anchor - rotations @ pivot
    arc[0], arc[-1] = stage, stage
    return {
        "object_arc": arc,
        "object_targets": np.stack((stage, arc[peak], stage, returned, returned)),
        "target_roles": (
            "staging",
            "peak",
            "restore",
            "return_transport",
            "return_place",
        ),
        "pivot_object": pivot,
        "pivot_world": anchor,
        "up_object": up,
        "tilt_axis_object": axis,
        "effective_angle_rad": theta,
        "peak_index": peak,
        "receiver_bearing_m": bearing,
        "receiver_bearing_tie": tie,
        "tilt_direction_policy": "ordered_jaw_axis_plus_angle",
        "physical_evidence": False,
    }


def jaw_tilt_axis(
    jaw_line: torch.Tensor, upright: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project an ordered jaw line into the plane normal to the vessel upright.

    Args:
        jaw_line: Batched object-local jaw vectors, shape (..., 3).
        upright: Object-local upright vectors with the same shape as jaw_line.

    Returns:
        Projected unit axes, visible tilt directions and finite nondegenerate
        validity masks. The source norm thresholds and signed jaw order remain
        unchanged; invalid rows do not establish a usable tilt axis.
    """

    def unit(value: torch.Tensor) -> torch.Tensor:
        return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(
            1e-6
        )

    up = unit(upright)
    raw = jaw_line - (jaw_line * up).sum(-1, keepdim=True) * up
    valid = torch.isfinite(raw).all(-1) & (torch.linalg.vector_norm(raw, dim=-1) > 1e-5)
    axis = unit(raw)
    direction = unit(torch.linalg.cross(axis, up))
    return axis, direction, valid


def jaw_pour_frames(
    object_pose: torch.Tensor,
    jaw_line_object: torch.Tensor,
    upright: torch.Tensor,
    mesh_vertices: torch.Tensor | None,
    receiver_pose: torch.Tensor | None,
    *,
    angle_rad: float,
    sample_count: int,
) -> JawPourFrameSet:
    """Build the distinct legacy jaw-oriented, mouth-pivot tilt/return samples.

    Args:
        object_pose: Batched current object transforms, shape (B, 4, 4).
        jaw_line_object: Ordered pad-line vectors in object coordinates (B, 3).
        upright: One object-local upright vector (3,).
        mesh_vertices: Optional local vertices for the gravity-top mouth pivot.
        receiver_pose: Optional batched receiver transforms selecting tilt sign.
        angle_rad: Signed angle from the resolved action options.
        sample_count: Resolved motion sample count, at least three.

    Returns:
        Object poses and local axis/direction diagnostics. The first pose follows
        the source's nonzero-angle convention; callers discard it when prepending
        observed joint state. Thus count=3 omits the peak in the executed path.
        The direction diagnostic precedes the optional receiver-sign flip, as in
        the source contract. No physical pouring or collision is inferred.
    """
    upright = upright / torch.linalg.vector_norm(
        upright, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    axis, direction, valid = jaw_tilt_axis(
        jaw_line_object, upright.expand_as(jaw_line_object)
    )
    if not valid.all():
        raise ValueError("Pour found a jaw line parallel to the vessel upright axis.")
    if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
        raise ValueError("Pour requires one stable jaw-line axis across environments.")
    if receiver_pose is not None:
        receiver_direction = receiver_pose[:, :3, 3] - object_pose[:, :3, 3]
        upright_world = object_pose[:, :3, :3] @ upright.expand_as(axis)[..., None]
        axis_world = object_pose[:, :3, :3] @ axis[..., None]
        receiver_direction = receiver_direction - (
            receiver_direction * upright_world.squeeze(-1)
        ).sum(-1, keepdim=True) * upright_world.squeeze(-1)
        receiver_direction = receiver_direction - (
            receiver_direction * axis_world.squeeze(-1)
        ).sum(-1, keepdim=True) * axis_world.squeeze(-1)
        visible_direction = torch.linalg.cross(
            axis_world.squeeze(-1), upright_world.squeeze(-1)
        )
        signed_visible = visible_direction * float(angle_rad)
        if torch.linalg.vector_norm(receiver_direction, dim=-1).min() <= 1e-5:
            raise ValueError("Pour receiver direction is degenerate.")
        flip = (signed_visible * receiver_direction).sum(-1, keepdim=True) < 0
        axis = torch.where(flip, -axis, axis)
    if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
        raise ValueError(
            "Pour receiver direction selects inconsistent tilt signs across environments."
        )
    if sample_count < 3:
        raise ValueError("Jaw Pour needs at least three trajectory samples.")
    first = sample_count // 2
    angles = torch.cat(
        (
            torch.linspace(0.0, 1.0, first + 1, device=object_pose.device)[1:],
            torch.linspace(
                1.0, 0.0, sample_count - first + 1, device=object_pose.device
            )[1:],
        )
    ) * float(angle_rad)
    from embodichain.utils.math import axis_angle_to_rotation_matrix

    rotations = axis_angle_to_rotation_matrix(axis[:, None] * angles[None, :, None])
    object_poses = object_pose[:, None].repeat(1, sample_count, 1, 1)
    pivot = torch.zeros(3, device=object_pose.device, dtype=object_pose.dtype)
    if isinstance(mesh_vertices, torch.Tensor) and mesh_vertices.numel():
        vertices = mesh_vertices.to(device=object_pose.device, dtype=object_pose.dtype)
        center = (vertices.amin(0) + vertices.amax(0)) * 0.5
        up0 = upright.reshape(-1, 3)[0]
        top = (vertices @ up0).amax()
        pivot = center + up0 * (top - (center * up0).sum())
    anchor = (
        object_pose[:, None, :3, 3]
        + torch.einsum("bij,j->bi", object_pose[:, :3, :3], pivot)[:, None]
    )
    object_poses[:, :, :3, :3] = object_pose[:, None, :3, :3] @ rotations
    object_poses[:, :, :3, 3] = anchor - torch.einsum(
        "bnij,j->bni", object_poses[:, :, :3, :3], pivot
    )
    return {
        "object_poses": object_poses,
        "tilt_axis_object": axis,
        "tilt_direction_object": direction,
    }


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class PourGraspProjection:
    """Owned numeric inputs for candidate-dependent Pour look-ahead before IK.

    Args:
        source_vertices: Already-scaled source-local vertices, shape (V, 3).
        receiver_vertices: Already-scaled receiver-local vertices, shape (R, 3).
        staging: Resolved world staging transform, shape (4, 4).
        receiver_pose: Same-observation receiver world transform.
        return_pose: Resolved object target shared by return transport/placement.
        pad_line_tcp: Ordered jaw vector in TCP coordinates, shape (3,).
        gravity_world: Actual finite negative-Z gravity vector.
        angle_rad: Angle resolved from the future Pour invocation's options.
        sample_count: Sample count resolved from that invocation's motion policy.

    No robot, program, provider or future-state callback is retained. The angle
    and sample count are already-resolved query operands, not another policy
    configuration. Invalid candidates keep the five staging fallback targets.
    """

    source_vertices: np.ndarray
    receiver_vertices: np.ndarray
    staging: np.ndarray
    receiver_pose: np.ndarray
    return_pose: np.ndarray
    pad_line_tcp: torch.Tensor
    gravity_world: tuple[float, float, float]
    angle_rad: float
    sample_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_vertices",
            "receiver_vertices",
            "staging",
            "receiver_pose",
            "return_pose",
        ):
            value = (_vertices if name.endswith("vertices") else _pose)(
                getattr(self, name)
            )
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        line = self.pad_line_tcp
        if (
            not isinstance(line, torch.Tensor)
            or line.shape != (3,)
            or not line.is_floating_point()
            or not torch.isfinite(line).all()
        ):
            raise ValueError(
                "Pour projection requires one finite ordered TCP jaw vector."
            )
        object.__setattr__(self, "pad_line_tcp", line.detach().cpu().clone())
        gravity = tuple(self.gravity_world)
        _gravity(gravity)
        object.__setattr__(self, "gravity_world", gravity)
        if (
            isinstance(self.angle_rad, bool)
            or not isinstance(self.angle_rad, Real)
            or not math.isfinite(self.angle_rad)
            or self.angle_rad == 0
        ):
            raise ValueError("Pour projection angle must be finite and nonzero.")
        if type(self.sample_count) is not int or self.sample_count < 3:
            raise ValueError("Pour projection sample_count must be an integer >=3.")

    def valid_mask(self, object_to_eef: torch.Tensor) -> torch.Tensor:
        """Identify finite jaw axes with nonzero gravity-plane projection.

        Args:
            object_to_eef: Candidate transforms, shape (..., 4, 4).

        Returns:
            Per-candidate geometry mask on the input device; no IK is performed.
        """
        if (
            not isinstance(object_to_eef, torch.Tensor)
            or object_to_eef.shape[-2:] != (4, 4)
            or not object_to_eef.is_floating_point()
        ):
            raise ValueError("Pour candidates require floating (..., 4, 4) transforms.")
        matrices = object_to_eef.detach().cpu().double().numpy()
        jaw = matrices[..., :3, :3] @ self.pad_line_tcp.double().numpy()
        norm = np.linalg.norm(jaw, axis=-1, keepdims=True)
        up = self.staging[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        up /= np.linalg.norm(up)
        unit = jaw / np.maximum(norm, 1e-9)
        projection = unit - (unit * up).sum(-1, keepdims=True) * up
        valid = (
            np.isfinite(projection).all(-1)
            & (norm[..., 0] > 1e-9)
            & (np.linalg.norm(projection, axis=-1) > 1e-9)
        )
        return torch.as_tensor(valid, device=object_to_eef.device)

    def targets(self, object_to_eef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the five ordered object targets for every exact grasp variant.

        Args:
            object_to_eef: Candidate transforms, shape (..., 4, 4).

        Returns:
            TCP targets shaped (5, ..., 4, 4) and the candidate geometry mask.
            Invalid candidates retain staging projections in all five slots.
        """
        valid = self.valid_mask(object_to_eef)
        matrices = object_to_eef.detach().cpu().numpy()
        targets = np.broadcast_to(self.staging, (5, *matrices.shape)).copy()
        line = self.pad_line_tcp.double().numpy()
        for index in np.ndindex(matrices.shape[:-2]):
            if bool(valid[index]):
                frames = pour_frames(
                    self.source_vertices,
                    self.receiver_vertices,
                    self.staging,
                    self.receiver_pose,
                    matrices[index][:3, :3] @ line,
                    self.angle_rad,
                    self.return_pose,
                    sample_count=self.sample_count,
                    gravity_world=self.gravity_world,
                )
                targets[(slice(None), *index)] = frames["object_targets"]
        objects = torch.as_tensor(
            targets, device=object_to_eef.device, dtype=object_to_eef.dtype
        )
        return objects @ object_to_eef[None], valid


def source_pour_staging_frame(
    vertices: np.ndarray,
    upright_axis: np.ndarray,
    initial_rotation: np.ndarray,
    reference_top: float,
    finger_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the source static bounding-sphere Pour staging construction.

    Args:
        vertices: Imported body-local source vertices in their original order.
        upright_axis: Source longest-axis direction in that same local basis.
        initial_rotation: Authored source world rotation.
        reference_top: Receiver upper offset from its origin along world Z.
        finger_radius: Explicit nominal hand reserve, not measured collision data.

    Returns:
        Receiver-relative translation pose, world orientation and local pivot.
        The original 15 mm reserve and axis-alignment order remain unchanged.
    """
    vertices = np.asarray(vertices)
    upright = np.asarray(upright_axis)
    rotation = np.asarray(initial_rotation)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Pour staging requires nonempty (N,3) local vertices.")
    if upright.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Pour staging requires an axis and a 3x3 rotation.")
    if not all(np.isfinite(value).all() for value in (vertices, upright, rotation)):
        raise ValueError("Pour staging geometry must be finite.")
    if (
        not math.isfinite(reference_top)
        or not math.isfinite(finger_radius)
        or finger_radius < 0
    ):
        raise ValueError(
            "Pour staging offsets must be finite with nonnegative reserve."
        )
    pivot = (vertices.min(0) + vertices.max(0)) * 0.5
    radius = float(np.linalg.norm(vertices - pivot, axis=1).max())
    world_up = rotation @ upright
    vertical = np.array([0.0, 0.0, 1.0])
    cross = np.cross(world_up, vertical)
    sine = np.linalg.norm(cross)
    cosine = np.clip(world_up @ vertical, -1.0, 1.0)
    if sine > 1e-8:
        rotation = (
            Rotation.from_rotvec(cross / sine * math.atan2(sine, cosine)).as_matrix()
            @ rotation
        )
    elif cosine < 0:
        transverse = rotation[:, int(np.argmin(np.abs(upright)))]
        rotation = Rotation.from_rotvec(transverse * math.pi).as_matrix() @ rotation
    offset = np.array([0.0, 0.0, reference_top + radius + finger_radius + 0.015])
    relative = np.eye(4)
    relative[:3, 3] = rotation.T @ offset - pivot
    return relative, rotation, pivot
