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

"""Grasp-pose and constrained qpos operators requiring downstream validation."""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from .contracts import (
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
    _pose,
    _tensor,
)

__all__ = [
    "rotate_grasp_about_object_axis",
    "joint_residual",
    "retime",
    "validate_motion_limits",
]


def rotate_grasp_about_object_axis(
    object_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    *,
    axis: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Rotate a reference grasp about an axis through a fixed object's origin.

    The axis is expressed in the object's local frame. Both the grasp position
    and orientation rotate about that axis; the object itself is unchanged.
    For example, a cube permits quarter-turn variants of a centered top grasp.
    The caller owns object symmetry and grasp validity: arbitrary rotations need
    not preserve contacts. Every candidate requires fresh IK/path planning and
    physical grasp validation. This function does not modify a qpos template.

    Args:
        object_pose: Object transform in the scene frame, shape ``(4, 4)``.
        grasp_pose: Reference TCP transform in that same frame, shape ``(4, 4)``.
        axis: Finite nonzero object-local rotation axis, shape ``(3,)``.
        angles: Finite rotation angles in radians, shape ``(C,)``, with C > 0.

    Returns:
        Owned candidate TCP transforms with shape ``(C, 4, 4)``, using the
        object pose's device and dtype, in the original scene frame.

    Raises:
        ValueError: If transforms, axis, or angles are malformed or nonfinite.
    """
    source_object = _pose(object_pose, "object_pose")
    source_grasp = _pose(grasp_pose, "grasp_pose")
    direction = _tensor(axis, "axis", 1).to(source_object.device, torch.float64)
    rotations = _tensor(angles, "angles", 1).to(source_object.device, torch.float64)
    if direction.shape != (3,) or rotations.numel() == 0:
        raise ValueError("axis must have shape (3,) and angles must be nonempty")
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("axis must have a finite nonzero norm")
    # Calculate in float64 to avoid overflow in low-precision axis norms and
    # accumulated transforms. Wrapping also bounds the Rodrigues input.
    rotations = torch.remainder(rotations + math.pi, 2 * math.pi) - math.pi
    transforms = torch.eye(4, device=source_object.device, dtype=torch.float64).repeat(
        rotations.numel(), 1, 1
    )
    transforms[:, :3, :3] = axis_angle_to_rotation_matrix(
        rotations[:, None] * (direction / norm)
    )
    anchor = source_object.to(dtype=torch.float64)
    reference = source_grasp.to(device=anchor.device, dtype=anchor.dtype)
    return (anchor @ transforms @ pose_inv(anchor) @ reference).to(source_object)


def _allowed_phases(template: TrajectoryTemplate, operator: str) -> list:
    if operator not in template.allowed_operators:
        raise ValueError(f"Template does not allow {operator}.")
    phases = [
        phase
        for phase in template.phases
        if phase.kind == "free" and operator in phase.allowed_operators
    ]
    if not phases:
        raise ValueError(f"No explicitly annotated free phase allows {operator}.")
    if not template.controlled_joint_indices:
        raise ValueError("Augmentation requires explicit controlled_joint_indices.")
    return phases


def joint_residual(
    template: TrajectoryTemplate,
    *,
    joint_limits: torch.Tensor,
    normalized_scale: float,
    generator: torch.Generator,
) -> TrajectoryTemplate:
    """Add one smooth, endpoint-preserving residual per allowed free phase.

    This operator applies only to explicitly permitted joint-space motion.
    Cartesian contact constraints require a planning adapter and cannot be
    inferred from qpos. It changes only the declared controlled joints.

    Args:
        template: Annotated full-joint reference.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        normalized_scale: Maximum offset as a fraction of each joint range.
        generator: Explicit local generator; global RNG is never consumed.

    Returns:
        A new template requiring path and physical validation.

    Raises:
        ValueError: If permissions, limits or resulting positions are invalid.
    """
    phases = _allowed_phases(template, "joint_residual")
    if not math.isfinite(normalized_scale) or not 0 <= normalized_scale <= 1:
        raise ValueError("normalized_scale must be finite and within [0, 1].")
    q = template.positions.clone()
    limits = joint_limits.to(device=q.device, dtype=q.dtype)
    if (
        limits.shape != (q.shape[1], 2)
        or not torch.isfinite(limits).all()
        or not (limits[:, 1] > limits[:, 0]).all()
    ):
        raise ValueError("joint_limits must be finite increasing intervals (D,2).")
    indices = list(template.controlled_joint_indices)
    for phase in phases:
        length = phase.stop_index - phase.start_index
        if length < 3:
            raise ValueError("A spatially variable phase needs an interior sample.")
        progress = torch.linspace(0, 1, length, device=q.device, dtype=q.dtype)
        # This basis has zero value and derivative at both phase endpoints.
        envelope = 16 * progress.square() * (1 - progress).square()
        values = torch.rand(
            len(indices), generator=generator, device=generator.device, dtype=q.dtype
        ).to(q.device)
        offsets = (
            (2 * values - 1)
            * normalized_scale
            * (limits[indices, 1] - limits[indices, 0])
        )
        q[phase.start_index : phase.stop_index, indices] += envelope[:, None] * offsets
    if ((q < limits[:, 0]) | (q > limits[:, 1])).any():
        raise ValueError(
            "Sampled residual violates joint limits; reject this proposal."
        )
    return replace(template, positions=q)


def retime(
    template: TrajectoryTemplate,
    *,
    duration_scale: float,
    control_dt: float,
    max_samples: int = 1_000_000,
) -> TrajectoryTemplate:
    """Retime permitted free phases and resample onto the host control clock.

    Contact and wait durations, phase endpoints, and inter-phase intervals are
    preserved. Phase boundaries must already align with the host clock. Scaled
    phase durations round up to whole control steps; sample counts therefore
    change. The function does not claim dynamic or collision validity.

    Args:
        template: Reference containing explicitly permitted ``retime`` phases.
        duration_scale: Positive factor; greater than one slows free motion.
        control_dt: Authoritative host control period in seconds.
        max_samples: Allocation bound for the resulting trajectory.

    Returns:
        A new uniformly timed template with remapped phase indices.

    Raises:
        ValueError: If permissions, timing or allocation constraints fail.
    """
    phases = _allowed_phases(template, "retime")
    if not all(math.isfinite(x) and x > 0 for x in (duration_scale, control_dt)):
        raise ValueError(
            "Duration scale and control period must be positive and finite."
        )
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be an integer of at least two.")
    q = template.positions
    intervals = template.dt.to(dtype=torch.float64).clone()
    original_times = intervals.cumsum(0)
    anchors = {0, q.shape[0] - 1}
    variable_interior = set()
    for phase in phases:
        variable_interior.update(range(phase.start_index + 1, phase.stop_index - 1))
    # Every contact, wait and unlabelled sample is a locked event. Refuse a
    # coarser clock that would silently remove an intermediate tool command.
    anchors.update(i for i in range(q.shape[0]) if i not in variable_interior)
    for phase in template.phases:
        anchors.update((phase.start_index, phase.stop_index - 1))
    for index in anchors:
        steps = float(original_times[index]) / control_dt
        if not math.isclose(steps, round(steps), abs_tol=1e-5, rel_tol=1e-6):
            raise ValueError("Phase anchors must align with the host control clock.")
    uncontrolled = [
        i for i in range(q.shape[1]) if i not in template.controlled_joint_indices
    ]
    for phase in phases:
        start, stop = phase.start_index, phase.stop_index
        if stop - start < 2:
            raise ValueError("A retimed phase needs at least two samples.")
        if uncontrolled and not torch.equal(
            q[start:stop, uncontrolled],
            q[start : start + 1, uncontrolled].expand(stop - start, -1),
        ):
            raise ValueError(
                "Retiming may not change an uncontrolled joint's schedule."
            )
        duration = float(intervals[start + 1 : stop].sum())
        steps = max(1, math.ceil(duration * duration_scale / control_dt - 1e-6))
        intervals[start + 1 : stop] *= steps * control_dt / duration
    times = intervals.cumsum(0)
    count = round(float(times[-1]) / control_dt) + 1
    if count < 2 or count > max_samples:
        raise ValueError("Retimed trajectory exceeds the declared sample budget.")
    grid = torch.arange(count, dtype=times.dtype, device=times.device) * control_dt
    upper = torch.searchsorted(times, grid).clamp(1, q.shape[0] - 1)
    lower = upper - 1
    weight = ((grid - times[lower]) / (times[upper] - times[lower])).to(q.dtype)
    result = torch.lerp(q[lower], q[upper], weight[:, None])
    # Preserve the exact declared anchors, including closed-contact endpoints.
    for index in anchors:
        result[round(float(times[index]) / control_dt)] = q[index]
    remapped = tuple(
        replace(
            phase,
            start_index=round(float(times[phase.start_index]) / control_dt),
            stop_index=round(float(times[phase.stop_index - 1]) / control_dt) + 1,
        )
        for phase in template.phases
    )
    dt = torch.full((count,), control_dt, dtype=template.dt.dtype, device=q.device)
    dt[0] = 0
    return replace(template, positions=result, dt=dt, phases=remapped)


def validate_motion_limits(
    template: TrajectoryTemplate,
    *,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
) -> ValidationResult:
    """Check sampled finite-difference motion limits without asserting task success.

    Args:
        template: Timed full-joint path to check.
        velocity_limits: Positive per-joint speed bounds, shape ``(D,)``.
        acceleration_limits: Positive per-joint acceleration bounds, shape ``(D,)``.

    Returns:
        A single ``motion_limits`` check with maximum normalized ratios.
    """
    # Accumulate derivatives outside the source's storage dtype. Half-precision
    # ratios can overflow even for ordinary joint motion and finite profiles.
    q = template.positions.to(dtype=torch.float64)
    dt = template.dt.to(dtype=torch.float64)
    limits = [
        x.to(device=q.device, dtype=q.dtype)
        for x in (velocity_limits, acceleration_limits)
    ]
    if any(
        x.shape != (q.shape[1],) or not torch.isfinite(x).all() or not (x > 0).all()
        for x in limits
    ):
        raise ValueError("Dynamic limits must be positive finite per-joint vectors.")
    velocity = (q[1:] - q[:-1]) / dt[1:, None]
    acceleration = (velocity[1:] - velocity[:-1]) / ((dt[2:] + dt[1:-1])[:, None] / 2)
    speed_ratio = float((velocity.abs() / limits[0]).max()) if velocity.numel() else 0.0
    acceleration_ratio = (
        float((acceleration.abs() / limits[1]).max()) if acceleration.numel() else 0.0
    )
    passed = (
        math.isfinite(speed_ratio)
        and math.isfinite(acceleration_ratio)
        and max(speed_ratio, acceleration_ratio) <= 1 + 1e-6
    )
    metrics = {
        name: value
        for name, value in (
            ("speed_ratio", speed_ratio),
            ("acceleration_ratio", acceleration_ratio),
        )
        if math.isfinite(value)
    }
    return ValidationResult(
        (
            ValidationCheck(
                "motion_limits",
                "passed" if passed else "failed",
                detail="" if len(metrics) == 2 else "Non-finite sampled derivatives.",
                metrics=metrics,
            ),
        )
    )
