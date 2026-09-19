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
    "perturb_approach_direction",
    "joint_residual",
    "via_points",
    "nullspace_residual",
    "retime",
    "validate_motion_limits",
]

TIMING_PROFILES = ("uniform", "ease_in", "ease_out")
"""Monotone within-phase time warps selectable by :func:`retime`."""


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


def perturb_approach_direction(
    contact_pose: torch.Tensor,
    *,
    approach_axis: torch.Tensor,
    standoff_distance: float,
    polar_angles: torch.Tensor,
    azimuth_angles: torch.Tensor,
    align_tool: bool = True,
) -> torch.Tensor:
    """Place standoff poses on a cone around a nominal approach direction.

    The contact transform itself is never modified, so a fixed waypoint stays
    exact; only the pose the motion departs from changes. ``approach_axis`` is
    expressed in the contact frame and points from the standoff toward the
    contact, which is the direction the tool travels while closing the gap.
    Each candidate tilts that axis by its polar angle about a perpendicular
    selected by its azimuth angle.

    The caller owns reachability and clearance: a tilted standoff need not be
    collision free, within the workspace, or compatible with the object's
    contact geometry. Every candidate requires fresh IK and path planning.

    Args:
        contact_pose: Fixed contact transform in the scene frame, shape ``(4, 4)``.
        approach_axis: Finite nonzero contact-local travel direction, shape ``(3,)``.
        standoff_distance: Positive distance from the standoff to the contact.
        polar_angles: Finite tilt angles in radians from the nominal axis, shape ``(C,)``.
        azimuth_angles: Finite angles in radians selecting the tilt plane, shape ``(C,)``.
        align_tool: Rotate each standoff so its own local ``approach_axis`` points
            at the contact, making the approach a screw motion. ``False`` keeps the
            contact orientation and varies only the standoff position.

    Returns:
        Owned standoff transforms with shape ``(C, 4, 4)``, using the contact
        pose's device and dtype, in the original scene frame.

    Raises:
        ValueError: If the transform, axis, distance, or angles are malformed.
    """
    source = _pose(contact_pose, "contact_pose")
    direction = _tensor(approach_axis, "approach_axis", 1).to(
        source.device, torch.float64
    )
    polar = _tensor(polar_angles, "polar_angles", 1).to(source.device, torch.float64)
    azimuth = _tensor(azimuth_angles, "azimuth_angles", 1).to(
        source.device, torch.float64
    )
    if direction.shape != (3,) or polar.numel() == 0:
        raise ValueError(
            "approach_axis must have shape (3,) and angles must be nonempty"
        )
    if azimuth.shape != polar.shape:
        raise ValueError("polar_angles and azimuth_angles must have the same length")
    if type(align_tool) is not bool:
        raise ValueError("align_tool must be a boolean")
    if not math.isfinite(standoff_distance) or standoff_distance <= 0:
        raise ValueError("standoff_distance must be finite and positive")
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("approach_axis must have a finite nonzero norm")
    axis = direction / norm
    # Pick the basis vector least aligned with the approach axis so the
    # perpendicular reference stays well conditioned for every input axis.
    basis = torch.eye(3, device=source.device, dtype=torch.float64)
    reference = basis[int(torch.argmin(axis.abs()))]
    perpendicular = reference - (reference @ axis) * axis
    perpendicular = perpendicular / torch.linalg.vector_norm(perpendicular)
    # Wrapping bounds the Rodrigues inputs the same way grasp rotation does.
    polar = torch.remainder(polar + math.pi, 2 * math.pi) - math.pi
    azimuth = torch.remainder(azimuth + math.pi, 2 * math.pi) - math.pi
    spin = axis_angle_to_rotation_matrix(azimuth[:, None] * axis)
    tilt_axis = torch.einsum("cij,j->ci", spin, perpendicular)
    tilt = axis_angle_to_rotation_matrix(polar[:, None] * tilt_axis)
    travel = torch.einsum("cij,j->ci", tilt, axis)
    local = torch.eye(4, device=source.device, dtype=torch.float64).repeat(
        polar.numel(), 1, 1
    )
    local[:, :3, 3] = -float(standoff_distance) * travel
    if align_tool:
        local[:, :3, :3] = tilt
    return (source.to(dtype=torch.float64) @ local).to(source)


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


def _resolve_joint_limits(
    positions: torch.Tensor, joint_limits: torch.Tensor
) -> torch.Tensor:
    limits = joint_limits.to(device=positions.device, dtype=positions.dtype)
    if (
        limits.shape != (positions.shape[1], 2)
        or not torch.isfinite(limits).all()
        or not (limits[:, 1] > limits[:, 0]).all()
    ):
        raise ValueError("joint_limits must be finite increasing intervals (D,2).")
    return limits


def _reject_limit_violations(positions: torch.Tensor, limits: torch.Tensor) -> None:
    if ((positions < limits[:, 0]) | (positions > limits[:, 1])).any():
        raise ValueError(
            "Sampled residual violates joint limits; reject this proposal."
        )


def _normalized_scale(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("normalized_scale must be finite and within [0, 1].")
    return float(value)


def _endpoint_envelope(length: int, reference: torch.Tensor) -> torch.Tensor:
    progress = torch.linspace(
        0, 1, length, device=reference.device, dtype=reference.dtype
    )
    # This basis has zero value and derivative at both phase endpoints.
    return 16 * progress.square() * (1 - progress).square()


def _clamped_hermite(knot_values: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Interpolate interior knots with zero value and slope at both ends.

    Knots sit at uniformly spaced interior progress values. Cubic Hermite
    segments with clamped end tangents reproduce every knot exactly while
    leaving the enclosing phase endpoints and their derivatives untouched.

    Args:
        knot_values: Interior knot values with shape ``(K, J)``.
        progress: Normalized evaluation points in ``[0, 1]`` with shape ``(L,)``.

    Returns:
        Interpolated values with shape ``(L, J)``.
    """
    count = knot_values.shape[0]
    pad = knot_values.new_zeros(1, knot_values.shape[1])
    values = torch.cat((pad, knot_values, pad))
    step = 1.0 / (count + 1)
    slopes = torch.zeros_like(values)
    slopes[1:-1] = (values[2:] - values[:-2]) / (2 * step)
    index = (progress / step).floor().clamp(0, count).to(torch.long)
    local = (progress / step - index)[:, None]
    square, cube = local * local, local * local * local
    return (
        (2 * cube - 3 * square + 1) * values[index]
        + (cube - 2 * square + local) * step * slopes[index]
        + (-2 * cube + 3 * square) * values[index + 1]
        + (cube - square) * step * slopes[index + 1]
    )


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
    scale = _normalized_scale(normalized_scale)
    q = template.positions.clone()
    limits = _resolve_joint_limits(q, joint_limits)
    indices = list(template.controlled_joint_indices)
    for phase in phases:
        length = phase.stop_index - phase.start_index
        if length < 3:
            raise ValueError("A spatially variable phase needs an interior sample.")
        envelope = _endpoint_envelope(length, q)
        values = torch.rand(
            len(indices), generator=generator, device=generator.device, dtype=q.dtype
        ).to(q.device)
        offsets = (2 * values - 1) * scale * (limits[indices, 1] - limits[indices, 0])
        q[phase.start_index : phase.stop_index, indices] += envelope[:, None] * offsets
    _reject_limit_violations(q, limits)
    return replace(template, positions=q)


def via_points(
    template: TrajectoryTemplate,
    *,
    joint_limits: torch.Tensor,
    via_count: int,
    normalized_scale: float,
    generator: torch.Generator,
) -> TrajectoryTemplate:
    """Route each allowed free phase through sampled interior joint knots.

    Unlike :func:`joint_residual`, which adds one fixed-shape bump per phase,
    this operator samples ``via_count`` independent interior knots and
    interpolates them with clamped cubic Hermite segments. Two or more knots
    admit sign changes along a phase, so the resulting paths differ in shape
    rather than only in amplitude. Phase endpoints and their derivatives are
    unchanged, which keeps every annotated waypoint exact.

    Only the declared controlled joints move. Cartesian contact constraints
    cannot be inferred from qpos, and the result still requires path and
    physical validation.

    Args:
        template: Annotated full-joint reference.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        via_count: Number of interior knots per phase, at least one.
        normalized_scale: Maximum knot offset as a fraction of each joint range.
        generator: Explicit local generator; global RNG is never consumed.

    Returns:
        A new template requiring path and physical validation.

    Raises:
        ValueError: If permissions, knot counts, limits or positions are invalid.
    """
    phases = _allowed_phases(template, "via_points")
    if type(via_count) is not int or via_count < 1:
        raise ValueError("via_count must be an integer of at least one.")
    scale = _normalized_scale(normalized_scale)
    q = template.positions.clone()
    limits = _resolve_joint_limits(q, joint_limits)
    indices = list(template.controlled_joint_indices)
    spans = limits[indices, 1] - limits[indices, 0]
    for phase in phases:
        length = phase.stop_index - phase.start_index
        if length < via_count + 2:
            raise ValueError("A via-point phase needs one interior sample per knot.")
        values = torch.rand(
            (via_count, len(indices)),
            generator=generator,
            device=generator.device,
            dtype=q.dtype,
        ).to(q.device)
        knots = (2 * values - 1) * scale * spans
        progress = torch.linspace(0, 1, length, device=q.device, dtype=q.dtype)
        q[phase.start_index : phase.stop_index, indices] += _clamped_hermite(
            knots, progress
        )
    _reject_limit_violations(q, limits)
    return replace(template, positions=q)


def nullspace_residual(
    template: TrajectoryTemplate,
    *,
    task_jacobians: torch.Tensor,
    joint_limits: torch.Tensor,
    normalized_scale: float,
    generator: torch.Generator,
    rank_tolerance: float = 1e-6,
) -> TrajectoryTemplate:
    """Move through the redundancy of a declared task while holding its rows.

    The caller supplies one task Jacobian per sample for the declared
    controlled joints, already reduced to the rows the task actually
    constrains. A six-row Jacobian on a six-joint arm generically leaves no
    redundancy; dropping the rows a gripper is symmetric about, such as spin
    about its own approach axis, is what creates alternative arm postures for
    the same tool pose.

    The residual is projected onto each sample's Jacobian null space, so the
    constrained task rows are preserved to **first order** only. Phase
    endpoints are exact because the envelope vanishes there, which keeps
    annotated waypoints unchanged regardless of linearization error. Interior
    samples require forward-kinematics verification by the host; this operator
    certifies neither pose accuracy nor collision freedom.

    Args:
        template: Annotated full-joint reference.
        task_jacobians: Per-sample task Jacobians with shape ``(N, R, C)``,
            where ``N`` matches the template samples and ``C`` the declared
            controlled joints, in ``controlled_joint_indices`` order.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        normalized_scale: Maximum offset as a fraction of each joint range,
            measured before null-space projection shrinks it.
        generator: Explicit local generator; global RNG is never consumed.
        rank_tolerance: Relative singular-value cutoff for the pseudoinverse.

    Returns:
        A new template requiring pose, path and physical validation.

    Raises:
        ValueError: If permissions, shapes, limits or positions are invalid, or
            if the declared task rows leave no usable redundancy.
    """
    phases = _allowed_phases(template, "nullspace_residual")
    scale = _normalized_scale(normalized_scale)
    if not math.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be finite and positive.")
    q = template.positions.clone()
    limits = _resolve_joint_limits(q, joint_limits)
    indices = list(template.controlled_joint_indices)
    jacobians = _tensor(task_jacobians, "task_jacobians", 3).to(
        device=q.device, dtype=torch.float64
    )
    if jacobians.shape[0] != q.shape[0] or jacobians.shape[2] != len(indices):
        raise ValueError(
            "task_jacobians must have shape (N, R, C) matching the template samples "
            "and controlled joints."
        )
    if jacobians.shape[1] < 1:
        raise ValueError("task_jacobians must constrain at least one task row.")
    # Project in float64: the projector is a difference of nearly equal terms
    # and loses the small redundant directions in lower precision.
    identity = torch.eye(len(indices), device=q.device, dtype=torch.float64)
    projector = identity - torch.linalg.pinv(jacobians, rtol=rank_tolerance).matmul(
        jacobians
    )
    spans = (limits[indices, 1] - limits[indices, 0]).to(torch.float64)
    largest = 0.0
    for phase in phases:
        length = phase.stop_index - phase.start_index
        if length < 3:
            raise ValueError("A spatially variable phase needs an interior sample.")
        values = torch.rand(
            len(indices),
            generator=generator,
            device=generator.device,
            dtype=torch.float64,
        ).to(q.device)
        offsets = (2 * values - 1) * scale * spans
        window = projector[phase.start_index : phase.stop_index].matmul(offsets)
        largest = max(largest, float(window.abs().max()))
        envelope = _endpoint_envelope(length, window)
        q[phase.start_index : phase.stop_index, indices] += (
            envelope[:, None] * window
        ).to(q.dtype)
    if largest <= 0:
        raise ValueError(
            "The declared task rows leave no joint redundancy; drop the rows the "
            "task does not constrain or use a redundant control part."
        )
    _reject_limit_violations(q, limits)
    return replace(template, positions=q)


_TIME_WARP_STRENGTH = 0.5
"""Bounded quadratic warp strength; arrival intervals stay within [0.5, 1.5]x."""


def _warp_phase_progress(fraction: torch.Tensor, profile: str) -> torch.Tensor:
    bulge = _TIME_WARP_STRENGTH * fraction * (1 - fraction)
    return fraction + bulge if profile == "ease_in" else fraction - bulge


def retime(
    template: TrajectoryTemplate,
    *,
    duration_scale: float,
    control_dt: float,
    max_samples: int = 1_000_000,
    profile: str = "uniform",
) -> TrajectoryTemplate:
    """Retime permitted free phases and resample onto the host control clock.

    Contact and wait durations, phase endpoints, and inter-phase intervals are
    preserved. Phase boundaries must already align with the host clock. Scaled
    phase durations round up to whole control steps; sample counts therefore
    change. The function does not claim dynamic or collision validity.

    ``profile`` redistributes time *within* a phase without changing its total
    duration or its geometric path, so a single path yields several distinct
    velocity profiles. The warp is bounded, keeping every arrival interval
    within a factor of three of the uniform spacing.

    Args:
        template: Reference containing explicitly permitted ``retime`` phases.
        duration_scale: Positive factor; greater than one slows free motion.
        control_dt: Authoritative host control period in seconds.
        max_samples: Allocation bound for the resulting trajectory.
        profile: ``uniform`` keeps proportional spacing, ``ease_in`` starts
            slower and finishes faster, and ``ease_out`` does the reverse.

    Returns:
        A new uniformly timed template with remapped phase indices.

    Raises:
        ValueError: If permissions, timing or allocation constraints fail.
    """
    phases = _allowed_phases(template, "retime")
    if profile not in TIMING_PROFILES:
        raise ValueError(f"profile must be one of {list(TIMING_PROFILES)}.")
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
        target = steps * control_dt
        if profile == "uniform":
            intervals[start + 1 : stop] *= target / duration
        else:
            fraction = intervals[start + 1 : stop].cumsum(0) / duration
            # The final sample anchors the phase end exactly, so a warp can
            # never move a phase boundary off the host control clock.
            fraction[-1] = 1
            warped = _warp_phase_progress(fraction, profile) * target
            intervals[start + 1 : stop] = torch.diff(
                warped, prepend=warped.new_zeros(1)
            )
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
