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

"""Batched trapezoidal and jerk-limited joint trajectory planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from embodichain.utils import configclass

from .base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    validate_plan_options,
)
from .utils import MoveType, PlanResult, PlanState, TrajectorySampleMethod

__all__ = [
    "TrapezoidalPlanOptions",
    "TrapezoidalPlanner",
    "TrapezoidalPlannerCfg",
]


@configclass
class TrapezoidalPlannerCfg(BasePlannerCfg):
    """Configure the batched joint-space time-profile planner."""

    planner_type: str = "trapezoidal"


@configclass
class TrapezoidalPlanOptions(PlanOptions):
    """Configure trapezoidal or Double-S trajectory generation.

    Args:
        profile: Scalar time profile used for every linear joint-path segment.
        constraints: Positive scalar or per-joint ``velocity``, ``acceleration``,
            and ``jerk`` limits. Jerk is required by the ``double_s`` profile.
        sample_method: Fixed output quantity or approximately fixed time step.
        sample_interval: Output count for ``QUANTITY`` or seconds for ``TIME``.
        minimum_duration: Optional lower bound on each environment trajectory's
            duration. Slower trajectories retain the same path and limits.
        stop_at_waypoints: Whether every supplied waypoint is a rest point. When
            false, redundant points on straight, same-direction runs are removed.
        collinearity_tolerance: Cosine tolerance used by waypoint compression.
        backend: Profile-construction and sampling backend. ``auto`` selects
            Warp for CUDA float32 trajectories and Torch otherwise.
    """

    profile: Literal["trapezoidal", "double_s"] = "trapezoidal"
    constraints: dict = {  # noqa: RUF012
        "velocity": 0.2,
        "acceleration": 0.5,
        "jerk": 2.0,
    }
    sample_method: TrajectorySampleMethod = TrajectorySampleMethod.QUANTITY
    sample_interval: float | int = 100
    minimum_duration: float | None = None
    stop_at_waypoints: bool = True
    collinearity_tolerance: float = 1e-5
    backend: Literal["auto", "torch", "warp"] = "auto"

    def __post_init__(self) -> None:
        if self.profile not in {"trapezoidal", "double_s"}:
            raise ValueError("profile must be 'trapezoidal' or 'double_s'.")
        required = {"velocity", "acceleration"}
        if self.profile == "double_s":
            required.add("jerk")
        missing = sorted(required.difference(self.constraints))
        if missing:
            raise ValueError(f"constraints is missing required keys: {missing}.")
        if self.sample_method is TrajectorySampleMethod.QUANTITY:
            if (
                isinstance(self.sample_interval, bool)
                or not isinstance(self.sample_interval, int)
                or self.sample_interval < 2
            ):
                raise ValueError(
                    "QUANTITY sample_interval must be an integer of at least 2."
                )
        elif self.sample_method is TrajectorySampleMethod.TIME:
            if (
                isinstance(self.sample_interval, bool)
                or not isinstance(self.sample_interval, (int, float))
                or not math.isfinite(float(self.sample_interval))
                or float(self.sample_interval) <= 0.0
            ):
                raise ValueError(
                    "TIME sample_interval must be finite and greater than zero."
                )
        else:
            raise ValueError(f"Unsupported sample method: {self.sample_method!r}.")
        if self.minimum_duration is not None and (
            isinstance(self.minimum_duration, bool)
            or not isinstance(self.minimum_duration, (int, float))
            or not math.isfinite(float(self.minimum_duration))
            or self.minimum_duration <= 0.0
        ):
            raise ValueError(
                "minimum_duration must be finite and greater than zero when set."
            )
        if not isinstance(self.stop_at_waypoints, bool):
            raise TypeError("stop_at_waypoints must be a bool.")
        if (
            isinstance(self.collinearity_tolerance, bool)
            or not isinstance(self.collinearity_tolerance, (int, float))
            or not math.isfinite(float(self.collinearity_tolerance))
            or not 0.0 <= self.collinearity_tolerance < 1.0
        ):
            raise ValueError(
                "collinearity_tolerance must be finite and in the range [0, 1)."
            )
        if self.backend not in {"auto", "torch", "warp"}:
            raise ValueError("backend must be 'auto', 'torch', or 'warp'.")


@dataclass(slots=True)
class _ProfileBatch:
    """Internal fixed-shape phase representation."""

    durations: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    jerks: torch.Tensor


def _compress_collinear_waypoints(
    waypoints: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Remove duplicate and straight-run interior points in a batched path.

    Rows may retain different numbers of points. Shorter rows are padded by
    repeating their final point, which the profile builder treats as zero-time
    segments.
    """
    if waypoints.shape[1] <= 2:
        return waypoints
    edges = waypoints[:, 1:] - waypoints[:, :-1]
    previous = edges[:, :-1]
    following = edges[:, 1:]
    previous_norm = torch.linalg.vector_norm(previous, dim=-1)
    following_norm = torch.linalg.vector_norm(following, dim=-1)
    epsilon = max(1e-8, float(tolerance) * 1e-3)
    active = (previous_norm > epsilon) & (following_norm > epsilon)
    cosine = (previous * following).sum(dim=-1) / (
        previous_norm * following_norm
    ).clamp_min(epsilon)
    straight = active & (cosine >= 1.0 - tolerance)
    duplicate_neighbor = (previous_norm <= epsilon) | (following_norm <= epsilon)
    keep = torch.ones(waypoints.shape[:2], dtype=torch.bool, device=waypoints.device)
    keep[:, 1:-1] = ~(straight | duplicate_neighbor)
    retained_count = keep.sum(dim=1)
    output_count = max(2, int(retained_count.max().item()))
    output = waypoints[:, -1:].expand(-1, output_count, -1).clone()
    output_index = keep.cumsum(dim=1) - 1
    batch_index = torch.arange(waypoints.shape[0], device=waypoints.device)[:, None]
    batch_index = batch_index.expand_as(keep)
    output[batch_index[keep], output_index[keep]] = waypoints[keep]
    return output


def _limit_tensor(
    value: float | list | torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Return a positive per-joint limit tensor broadcastable to ``(B, S, D)``."""
    limit = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if limit.ndim == 0:
        limit = limit.expand(reference.shape[-1])
    if limit.shape != (reference.shape[-1],):
        raise ValueError(
            f"Joint limits must be scalar or shape ({reference.shape[-1]},), "
            f"got {tuple(limit.shape)}."
        )
    if not bool(torch.isfinite(limit).all().item()) or bool((limit <= 0).any().item()):
        raise ValueError("Joint limits must contain finite positive values.")
    return limit


def _scalar_path_limits(
    delta: torch.Tensor,
    constraints: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project per-joint derivative limits onto linear scalar path segments."""
    absolute_delta = delta.abs()
    active = absolute_delta > 1e-8

    def project(key: str) -> torch.Tensor:
        joint_limit = _limit_tensor(constraints[key], delta)
        ratios = joint_limit / absolute_delta.clamp_min(1e-8)
        ratios.masked_fill_(~active, torch.inf)
        return ratios.amin(dim=-1)

    velocity = project("velocity")
    acceleration = project("acceleration")
    jerk = (
        project("jerk")
        if "jerk" in constraints
        else torch.full_like(velocity, torch.inf)
    )
    stationary = ~active.any(dim=-1)
    zeros = torch.zeros_like(velocity)
    return (
        torch.where(stationary, zeros, velocity),
        torch.where(stationary, zeros, acceleration),
        torch.where(stationary, zeros, jerk),
    )


def _integrate_phases(durations: torch.Tensor, jerks: torch.Tensor) -> _ProfileBatch:
    """Integrate piecewise-constant jerk phases from rest."""
    positions = torch.zeros_like(durations)
    velocities = torch.zeros_like(durations)
    accelerations = torch.zeros_like(durations)
    position = torch.zeros_like(durations[..., 0])
    velocity = torch.zeros_like(position)
    acceleration = torch.zeros_like(position)
    for phase in range(durations.shape[-1]):
        positions[..., phase] = position
        velocities[..., phase] = velocity
        accelerations[..., phase] = acceleration
        duration = durations[..., phase]
        jerk = jerks[..., phase]
        position = (
            position
            + velocity * duration
            + 0.5 * acceleration * duration.square()
            + jerk * duration.pow(3) / 6.0
        )
        velocity = velocity + acceleration * duration + 0.5 * jerk * duration.square()
        acceleration = acceleration + jerk * duration
    return _ProfileBatch(durations, positions, velocities, accelerations, jerks)


def _apply_minimum_duration(
    profile: _ProfileBatch,
    minimum_duration: float | None,
) -> _ProfileBatch:
    """Uniformly slow complete batch rows to a requested minimum duration."""
    if minimum_duration is None:
        return profile
    total_duration = profile.durations.sum(dim=(-2, -1))
    epsilon = torch.finfo(total_duration.dtype).eps
    stationary = total_duration <= epsilon
    requested_scale = minimum_duration / total_duration.clamp_min(epsilon)
    scale = torch.maximum(torch.ones_like(total_duration), requested_scale)
    scale = torch.where(stationary, torch.ones_like(scale), scale)
    phase_scale = scale[:, None, None]
    slowed = _ProfileBatch(
        durations=profile.durations * phase_scale,
        positions=profile.positions,
        velocities=profile.velocities / phase_scale,
        accelerations=profile.accelerations / phase_scale.square(),
        jerks=profile.jerks / phase_scale.pow(3),
    )
    if bool(stationary.any().item()):
        slowed.durations[stationary] = 0.0
        slowed.durations[stationary, 0, slowed.durations.shape[-1] // 2] = (
            minimum_duration
        )
        slowed.positions[stationary] = 0.0
        slowed.velocities[stationary] = 0.0
        slowed.accelerations[stationary] = 0.0
        slowed.jerks[stationary] = 0.0
    return slowed


def _scale_profile_time(profile: _ProfileBatch, scale: torch.Tensor) -> _ProfileBatch:
    """Apply per-segment uniform time scaling without changing path position."""
    phase_scale = scale[..., None]
    return _ProfileBatch(
        durations=profile.durations * phase_scale,
        positions=profile.positions,
        velocities=profile.velocities / phase_scale,
        accelerations=profile.accelerations / phase_scale.square(),
        jerks=profile.jerks / phase_scale.pow(3),
    )


def _build_trapezoidal_profile(
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
) -> _ProfileBatch:
    """Build rest-to-rest unit-distance triangular or trapezoidal profiles."""
    stationary = velocity_limit <= 0.0
    acceleration_time = velocity_limit / acceleration_limit.clamp_min(1e-12)
    acceleration_distance_twice = (
        velocity_limit.square() / acceleration_limit.clamp_min(1e-12)
    )
    reaches_velocity = acceleration_distance_twice < 1.0
    peak_velocity = torch.where(
        reaches_velocity,
        velocity_limit,
        torch.sqrt(acceleration_limit),
    )
    acceleration_time = peak_velocity / acceleration_limit.clamp_min(1e-12)
    cruise_time = torch.where(
        reaches_velocity,
        (1.0 - peak_velocity.square() / acceleration_limit) / peak_velocity,
        torch.zeros_like(peak_velocity),
    )
    acceleration_time = torch.where(
        stationary, torch.zeros_like(acceleration_time), acceleration_time
    )
    cruise_time = torch.where(stationary, torch.zeros_like(cruise_time), cruise_time)
    durations = torch.stack([acceleration_time, cruise_time, acceleration_time], dim=-1)
    # A trapezoidal profile has piecewise-constant acceleration. Represent it as
    # zero-jerk phases and seed their accelerations explicitly below.
    profile = _integrate_phases(durations, torch.zeros_like(durations))
    profile.accelerations[..., 0] = acceleration_limit
    profile.accelerations[..., 1] = 0.0
    profile.accelerations[..., 2] = -acceleration_limit
    profile.velocities[..., 0] = 0.0
    profile.velocities[..., 1] = peak_velocity
    profile.velocities[..., 2] = peak_velocity
    profile.positions[..., 0] = 0.0
    profile.positions[..., 1] = 0.5 * peak_velocity * acceleration_time
    profile.positions[..., 2] = profile.positions[..., 1] + peak_velocity * cruise_time
    return profile


def _build_double_s_profile(
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    jerk_limit: torch.Tensor,
) -> _ProfileBatch:
    """Build HolisticMotion's rest-to-rest seven-phase Double-S profile.

    This follows ``TrajectoryDoubleS::_ComputeDoubleSProfile`` for zero path
    boundary velocities. In particular, a move without a constant-velocity
    phase repeatedly lowers the candidate acceleration by ``0.9`` until both
    acceleration halves can contain their two jerk ramps. That deliberately
    differs from the closed-form triangular-jerk fallback commonly used by
    simplified Double-S implementations.
    """
    stationary = velocity_limit <= 0.0
    epsilon = 1e-12
    safe_velocity = velocity_limit.clamp_min(epsilon)
    safe_acceleration = acceleration_limit.clamp_min(epsilon)
    safe_jerk = jerk_limit.clamp_min(epsilon)

    reaches_acceleration = safe_velocity * safe_jerk >= safe_acceleration.square()
    tj = torch.where(
        reaches_acceleration,
        safe_acceleration / safe_jerk,
        torch.sqrt(safe_velocity / safe_jerk),
    )
    ta = torch.where(
        reaches_acceleration,
        tj + safe_velocity / safe_acceleration,
        2.0 * tj,
    )
    cruise_time = 1.0 / safe_velocity - ta
    no_cruise = (~stationary) & (cruise_time <= 0.0)

    # Match HolisticMotion's intentionally discrete feasibility search rather
    # than substituting an analytic triangular-jerk solution.
    candidate_acceleration = safe_acceleration.clone()
    for _ in range(1001):
        if not bool(no_cruise.any().item()):
            break
        candidate_tj = candidate_acceleration / safe_jerk
        delta = torch.sqrt(
            candidate_acceleration.pow(4) / safe_jerk.square()
            + 4.0 * candidate_acceleration
        )
        candidate_ta = (candidate_acceleration.square() / safe_jerk + delta) / (
            2.0 * candidate_acceleration
        )
        accepted = no_cruise & (candidate_ta >= 2.0 * candidate_tj)
        tj = torch.where(accepted, candidate_tj, tj)
        ta = torch.where(accepted, candidate_ta, ta)
        no_cruise = no_cruise & ~accepted
        candidate_acceleration = torch.where(
            no_cruise, candidate_acceleration * 0.9, candidate_acceleration
        )
    if bool(no_cruise.any().item()):
        raise RuntimeError(
            "HolisticMotion Double-S acceleration search did not converge."
        )
    cruise_time = torch.clamp_min(cruise_time, 0.0)
    tj = torch.where(stationary, torch.zeros_like(tj), tj)
    ta = torch.where(stationary, torch.zeros_like(ta), ta)
    cruise_time = torch.where(stationary, torch.zeros_like(cruise_time), cruise_time)
    constant_acceleration_time = torch.clamp_min(ta - 2.0 * tj, 0.0)
    durations = torch.stack(
        [
            tj,
            constant_acceleration_time,
            tj,
            cruise_time,
            tj,
            constant_acceleration_time,
            tj,
        ],
        dim=-1,
    )
    signs = torch.tensor(
        [1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        dtype=jerk_limit.dtype,
        device=jerk_limit.device,
    )
    jerks = jerk_limit[..., None] * signs
    jerks = torch.where(stationary[..., None], torch.zeros_like(jerks), jerks)
    return _integrate_phases(durations, jerks)


def _use_warp_backend(
    reference: torch.Tensor,
    backend: Literal["auto", "torch", "warp"],
) -> bool:
    """Return whether a profile stage should execute through Warp."""
    return backend == "warp" or (
        backend == "auto" and reference.is_cuda and reference.dtype == torch.float32
    )


def _build_scalar_profile(
    *,
    profile_name: Literal["trapezoidal", "double_s"],
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    jerk_limit: torch.Tensor,
    backend: Literal["auto", "torch", "warp"],
) -> _ProfileBatch:
    """Build scalar phases with the selected backend and shared post-processing."""
    use_warp = _use_warp_backend(velocity_limit, backend)
    if use_warp and velocity_limit.dtype != torch.float32:
        if backend == "warp":
            raise ValueError("The Warp trajectory backend requires float32 input.")
        use_warp = False
    if use_warp:
        try:
            from embodichain.utils.warp.kinematics.trapezoidal_warp import (
                build_profile_warp,
            )
        except ImportError:
            if backend == "warp":
                raise ImportError(
                    "The Warp trajectory backend requires the 'warp' package."
                ) from None
            use_warp = False
        else:
            tensors = build_profile_warp(
                profile=profile_name,
                velocity_limits=velocity_limit,
                acceleration_limits=acceleration_limit,
                jerk_limits=jerk_limit,
            )
            result = _ProfileBatch(*tensors)
    if not use_warp:
        result = (
            _build_double_s_profile(velocity_limit, acceleration_limit, jerk_limit)
            if profile_name == "double_s"
            else _build_trapezoidal_profile(velocity_limit, acceleration_limit)
        )

    if profile_name == "double_s":
        # HolisticMotion's EnforceJointLimits leaves a 1% margin whenever a
        # sampled derivative reaches a limit. Linear Double-S segments always
        # reach their projected jerk limit, so the resulting scale is 1.01.
        margin = torch.where(
            velocity_limit > 0.0,
            torch.full_like(velocity_limit, 1.01),
            torch.ones_like(velocity_limit),
        )
        result = _scale_profile_time(result, margin)
    return result


def _compose_profile_samples_torch(
    *,
    times: torch.Tensor,
    cumulative_duration: torch.Tensor,
    profile: _ProfileBatch,
    segment_starts: torch.Tensor,
    segment_deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate scalar profiles and compose joint states with Torch."""
    segment = torch.searchsorted(
        cumulative_duration.contiguous(), times.contiguous(), right=True
    )
    segment = segment.clamp_max(segment_deltas.shape[1] - 1)
    previous_duration = torch.cat(
        [torch.zeros_like(cumulative_duration[:, :1]), cumulative_duration[:, :-1]],
        dim=-1,
    )
    local_time = times - torch.gather(previous_duration, 1, segment)
    batch_ids = torch.arange(times.shape[0], device=times.device)[:, None].expand_as(
        segment
    )
    selected_durations = profile.durations[batch_ids, segment]
    cumulative_phase_time = selected_durations.cumsum(dim=-1)
    phase = torch.searchsorted(
        cumulative_phase_time.contiguous(),
        local_time.unsqueeze(-1).contiguous(),
        right=True,
    ).squeeze(-1)
    phase = phase.clamp_max(profile.durations.shape[-1] - 1)
    phase_start_time = torch.cat(
        [
            torch.zeros_like(cumulative_phase_time[..., :1]),
            cumulative_phase_time[..., :-1],
        ],
        dim=-1,
    )
    gather = phase[..., None]
    tau = (
        local_time - torch.gather(phase_start_time, -1, gather).squeeze(-1)
    ).clamp_min(0.0)
    p0 = profile.positions[batch_ids, segment, phase]
    v0 = profile.velocities[batch_ids, segment, phase]
    a0 = profile.accelerations[batch_ids, segment, phase]
    jerk = profile.jerks[batch_ids, segment, phase]
    path_position = p0 + v0 * tau + 0.5 * a0 * tau.square() + jerk * tau.pow(3) / 6.0
    path_velocity = v0 + a0 * tau + 0.5 * jerk * tau.square()
    path_acceleration = a0 + jerk * tau
    selected_delta = segment_deltas[batch_ids, segment]
    selected_start = segment_starts[batch_ids, segment]
    return (
        selected_start + selected_delta * path_position[..., None],
        selected_delta * path_velocity[..., None],
        selected_delta * path_acceleration[..., None],
    )


def _compose_profile_samples(
    *,
    times: torch.Tensor,
    cumulative_duration: torch.Tensor,
    profile: _ProfileBatch,
    segment_starts: torch.Tensor,
    segment_deltas: torch.Tensor,
    backend: Literal["auto", "torch", "warp"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch sample composition to Torch or the optional Warp kernel."""
    use_warp = _use_warp_backend(times, backend)
    if use_warp:
        if times.dtype != torch.float32:
            if backend == "warp":
                raise ValueError("The Warp trajectory backend requires float32 input.")
        else:
            try:
                from embodichain.utils.warp.kinematics.trapezoidal_warp import (
                    compose_profile_samples_warp,
                )
            except ImportError:
                if backend == "warp":
                    raise ImportError(
                        "The Warp trajectory backend requires the 'warp' package."
                    ) from None
            else:
                return compose_profile_samples_warp(
                    times=times,
                    cumulative_segment_time=cumulative_duration,
                    phase_durations=profile.durations,
                    phase_positions=profile.positions,
                    phase_velocities=profile.velocities,
                    phase_accelerations=profile.accelerations,
                    phase_jerks=profile.jerks,
                    segment_starts=segment_starts,
                    segment_deltas=segment_deltas,
                )
    return _compose_profile_samples_torch(
        times=times,
        cumulative_duration=cumulative_duration,
        profile=profile,
        segment_starts=segment_starts,
        segment_deltas=segment_deltas,
    )


def _make_sample_times(
    duration: torch.Tensor, options: TrapezoidalPlanOptions
) -> torch.Tensor:
    """Create batched sample times, padding shorter rows at their endpoint."""
    if options.sample_method is TrajectorySampleMethod.QUANTITY:
        count = int(options.sample_interval)
        alpha = torch.linspace(
            0.0, 1.0, count, dtype=duration.dtype, device=duration.device
        )
        return duration[:, None] * alpha[None]

    counts = torch.maximum(
        torch.ceil(duration / float(options.sample_interval)).to(torch.long) + 1,
        torch.full_like(duration, 2, dtype=torch.long),
    )
    count = int(counts.max().item())
    sample_ids = torch.arange(count, device=duration.device)[None]
    valid = sample_ids < counts[:, None]
    row_last = torch.maximum(counts - 1, torch.ones_like(counts))
    row_alpha = sample_ids / row_last[:, None]
    return torch.where(valid, duration[:, None] * row_alpha, duration[:, None])


def _plan_linear_profiles(
    waypoints: torch.Tensor,
    options: TrapezoidalPlanOptions,
) -> PlanResult:
    """Plan batched piecewise-linear joint paths without simulation state."""
    if waypoints.ndim != 3 or waypoints.shape[1] < 2 or waypoints.shape[2] < 1:
        raise ValueError("waypoints must have shape (B, K, DOF) with K >= 2.")
    if not waypoints.is_floating_point() or not bool(
        torch.isfinite(waypoints).all().item()
    ):
        raise ValueError("waypoints must be a finite floating-point tensor.")
    if not options.stop_at_waypoints:
        waypoints = _compress_collinear_waypoints(
            waypoints, options.collinearity_tolerance
        )
    delta = waypoints[:, 1:] - waypoints[:, :-1]
    stationary_path = ~(delta.abs() > 1e-8).any(dim=(-2, -1))
    batch_size, _, dof = waypoints.shape
    if bool(stationary_path.all().item()):
        hold_duration = options.minimum_duration or 0.0
        duration = waypoints.new_full((batch_size,), hold_duration)
        times = _make_sample_times(duration, options)
        positions = waypoints[:, :1].expand(-1, times.shape[1], dof).clone()
        velocities = torch.zeros_like(positions)
        accelerations = torch.zeros_like(positions)
        return PlanResult(
            success=torch.ones(batch_size, dtype=torch.bool, device=waypoints.device),
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1])),
        )

    velocity_limit, acceleration_limit, jerk_limit = _scalar_path_limits(
        delta, options.constraints
    )
    profile = _build_scalar_profile(
        profile_name=options.profile,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        jerk_limit=jerk_limit,
        backend=options.backend,
    )
    profile = _apply_minimum_duration(profile, options.minimum_duration)
    segment_duration = profile.durations.sum(dim=-1)
    cumulative_duration = segment_duration.cumsum(dim=-1)
    duration = cumulative_duration[:, -1]
    times = _make_sample_times(duration, options)
    positions, velocities, accelerations = _compose_profile_samples(
        times=times,
        cumulative_duration=cumulative_duration,
        profile=profile,
        segment_starts=waypoints[:, :-1],
        segment_deltas=delta,
        backend=options.backend,
    )
    positions[:, 0] = waypoints[:, 0]
    positions[:, -1] = waypoints[:, -1]
    velocities[:, -1] = 0.0
    accelerations[:, -1] = 0.0
    dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    success = torch.ones(batch_size, dtype=torch.bool, device=waypoints.device)
    if bool(stationary_path.any().item()):
        positions[stationary_path] = waypoints[stationary_path, :1]
        velocities[stationary_path] = 0.0
        accelerations[stationary_path] = 0.0
        if options.minimum_duration is None:
            dt[stationary_path] = 0.0
    return PlanResult(
        success=success,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        dt=dt,
    )


class TrapezoidalPlanner(BasePlanner):
    """Plan batched linear joint paths with trapezoidal or Double-S timing."""

    supported_move_types = frozenset({MoveType.JOINT_MOVE})

    def default_plan_options(self) -> TrapezoidalPlanOptions:
        """Return backend-default planning options."""
        return TrapezoidalPlanOptions()

    @validate_plan_options(options_cls=TrapezoidalPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: TrapezoidalPlanOptions = TrapezoidalPlanOptions(),  # noqa: B008
    ) -> PlanResult:
        """Generate one batched joint trajectory through all target states.

        Args:
            target_states: Joint waypoints with ``qpos`` shape ``(B, DOF)``.
            options: Time-profile, derivative limits, and sampling configuration.

        Returns:
            Batched positions, velocities, accelerations, and explicit timing.

        Raises:
            ValueError: If fewer than two valid joint waypoints are supplied.
        """
        if len(target_states) < 2 or any(state.qpos is None for state in target_states):
            raise ValueError("TrapezoidalPlanner requires at least two qpos waypoints.")
        waypoints = torch.stack([state.qpos for state in target_states], dim=1).to(
            self.device
        )
        return _plan_linear_profiles(waypoints, options)
