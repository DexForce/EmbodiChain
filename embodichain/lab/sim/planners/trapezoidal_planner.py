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
from typing import Literal

import torch

from embodichain.utils import configclass
from .bezier import (
    compose_quintic_blend_jerk,
    compose_quintic_blend_state,
    quintic_blend_segments,
)

from .base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    validate_plan_options,
)
from ._blend_constraints import bound_blended_derivatives, bound_path_derivatives
from ._scalar_time_law import ScalarTimeLaw
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
        blend_tolerance: Non-negative corner deviation used for quintic blends.
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
    blend_tolerance: float = 0.0
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
        if (
            isinstance(self.blend_tolerance, bool)
            or not isinstance(self.blend_tolerance, (int, float))
            or not math.isfinite(float(self.blend_tolerance))
            or self.blend_tolerance < 0.0
        ):
            raise ValueError("blend_tolerance must be finite and non-negative.")
        if self.backend not in {"auto", "torch", "warp"}:
            raise ValueError("backend must be 'auto', 'torch', or 'warp'.")


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
    epsilon = max(1e-8, float(tolerance) * 1e-3)
    deduplicate = torch.ones(
        waypoints.shape[:2], dtype=torch.bool, device=waypoints.device
    )
    deduplicate[:, 1:] = torch.linalg.vector_norm(edges, dim=-1) > epsilon
    deduplicated_count = deduplicate.sum(dim=1)
    deduplicated_size = max(2, int(deduplicated_count.max().item()))
    deduplicated = waypoints[:, -1:].expand(-1, deduplicated_size, -1).clone()
    deduplicated_index = deduplicate.cumsum(dim=1) - 1
    batch_index = torch.arange(waypoints.shape[0], device=waypoints.device)[:, None]
    batch_index = batch_index.expand_as(deduplicate)
    deduplicated[batch_index[deduplicate], deduplicated_index[deduplicate]] = waypoints[
        deduplicate
    ]

    if deduplicated_size <= 2:
        return deduplicated
    edges = deduplicated[:, 1:] - deduplicated[:, :-1]
    previous = edges[:, :-1]
    following = edges[:, 1:]
    previous_norm = torch.linalg.vector_norm(previous, dim=-1)
    following_norm = torch.linalg.vector_norm(following, dim=-1)
    active = (previous_norm > epsilon) & (following_norm > epsilon)
    cosine = (previous * following).sum(dim=-1) / (
        previous_norm * following_norm
    ).clamp_min(epsilon)
    straight = active & (cosine >= 1.0 - tolerance)
    point_ids = torch.arange(deduplicated_size, device=waypoints.device)[None]
    last_point = (deduplicated_count - 1)[:, None]
    keep = (point_ids == 0) | (point_ids == last_point)
    real_interior = (point_ids[:, 1:-1] > 0) & (point_ids[:, 1:-1] < last_point)
    keep[:, 1:-1] |= real_interior & ~straight
    retained_count = keep.sum(dim=1)
    output_count = max(2, int(retained_count.max().item()))
    output = deduplicated[:, -1:].expand(-1, output_count, -1).clone()
    output_index = keep.cumsum(dim=1) - 1
    batch_index = torch.arange(waypoints.shape[0], device=waypoints.device)[:, None]
    batch_index = batch_index.expand_as(keep)
    output[batch_index[keep], output_index[keep]] = deduplicated[keep]
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


def _sample_times_from_intervals(
    segment_duration: torch.Tensor,
    intervals: torch.Tensor,
    output_count: int,
) -> torch.Tensor:
    """Compose sorted sample times whose segment endpoints are exact."""
    cumulative_intervals = intervals.cumsum(dim=1)
    cumulative_duration = segment_duration.cumsum(dim=1)
    sample_ids = torch.arange(output_count, device=segment_duration.device)[None]
    last_sample = cumulative_intervals[:, -1:]
    clamped_ids = sample_ids.expand(segment_duration.shape[0], -1).clamp_max(
        last_sample
    )
    segment = torch.searchsorted(
        cumulative_intervals.contiguous(), clamped_ids.contiguous(), right=False
    ).clamp_max(segment_duration.shape[1] - 1)
    previous_intervals = torch.cat(
        (torch.zeros_like(cumulative_intervals[:, :1]), cumulative_intervals[:, :-1]),
        dim=1,
    )
    previous_duration = torch.cat(
        (torch.zeros_like(cumulative_duration[:, :1]), cumulative_duration[:, :-1]),
        dim=1,
    )
    segment_intervals = intervals.gather(1, segment).clamp_min(1)
    local_interval = clamped_ids - previous_intervals.gather(1, segment)
    times = previous_duration.gather(1, segment) + segment_duration.gather(
        1, segment
    ) * (local_interval.to(segment_duration.dtype) / segment_intervals)
    times[:, 0] = 0.0
    return torch.where(last_sample > 0, times, torch.zeros_like(times))


def _make_sample_times(
    segment_duration: torch.Tensor, options: TrapezoidalPlanOptions
) -> torch.Tensor:
    """Create samples per segment so every retained boundary is represented."""
    active = segment_duration > 1e-12
    if options.sample_method is TrajectorySampleMethod.QUANTITY:
        count = int(options.sample_interval)
        required = active.sum(dim=1) + 1
        if count < int(required.max().item()):
            raise ValueError(
                "Quantity sampling requires at least one sample per retained waypoint; "
                f"received {count}, requires at least {int(required.max().item())}."
            )
        base = active.to(torch.long)
        extra = count - 1 - base.sum(dim=1)
        total_duration = segment_duration.sum(dim=1, keepdim=True).clamp_min(1e-12)
        raw_extra = segment_duration / total_duration * extra[:, None]
        allocated_extra = torch.floor(raw_extra).to(torch.long)
        remainder_count = extra - allocated_extra.sum(dim=1)
        fractional = torch.where(
            active,
            raw_extra - allocated_extra,
            torch.full_like(raw_extra, -1.0),
        )
        order = fractional.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=order.device)[None].expand_as(order),
        )
        allocated_extra += (rank < remainder_count[:, None]).to(torch.long)
        intervals = base + allocated_extra
        return _sample_times_from_intervals(segment_duration, intervals, count)

    intervals = torch.where(
        active,
        torch.ceil(segment_duration / float(options.sample_interval)).to(torch.long),
        torch.zeros_like(segment_duration, dtype=torch.long),
    )
    counts = torch.maximum(
        intervals.sum(dim=1) + 1,
        torch.full(
            (segment_duration.shape[0],),
            2,
            dtype=torch.long,
            device=segment_duration.device,
        ),
    )
    return _sample_times_from_intervals(
        segment_duration, intervals, int(counts.max().item())
    )


def _plan_blended_profiles(
    waypoints: torch.Tensor, options: TrapezoidalPlanOptions
) -> PlanResult:
    """Plan batched quintic-blended paths with a scalar time profile."""
    velocity_joint = _limit_tensor(options.constraints["velocity"], waypoints)
    acceleration_joint = _limit_tensor(options.constraints["acceleration"], waypoints)
    jerk_joint = _limit_tensor(options.constraints.get("jerk", 1.0), waypoints)
    lengths = []
    path_bounds = []
    projected = []
    for row in waypoints:
        segments, segment_lengths = quintic_blend_segments(row, options.blend_tolerance)
        bounds = bound_path_derivatives(segments, segment_lengths)
        tangent, curvature, _ = bounds.amax(dim=(0, 2)).to(row.dtype)
        epsilon = torch.finfo(row.dtype).eps
        tangent = tangent.clamp_min(epsilon)
        path_velocity = (velocity_joint / tangent).amin()
        curvature_velocity = (
            (acceleration_joint / curvature.clamp_min(epsilon)).sqrt().amin()
        )
        path_velocity = torch.minimum(path_velocity, curvature_velocity)
        path_acceleration = (acceleration_joint / tangent).amin()
        path_jerk = (jerk_joint / tangent).amin()
        lengths.append(segment_lengths)
        path_bounds.append(bounds)
        projected.append(
            torch.stack((path_velocity, path_acceleration, path_jerk))
            / segment_lengths.sum()
        )
    path_length = torch.stack([value.sum() for value in lengths])
    limits = torch.stack(projected)
    profile = ScalarTimeLaw.build(
        profile_name=options.profile,
        velocity_limit=limits[:, 0:1],
        acceleration_limit=limits[:, 1:2],
        jerk_limit=limits[:, 2:3],
        backend=options.backend,
    )
    profile = profile.with_minimum_duration(options.minimum_duration)
    upper_bounds = torch.stack(
        [
            bound_blended_derivatives(
                lengths[index], path_bounds[index], profile, index
            )
            for index in range(waypoints.shape[0])
        ]
    )
    scale = torch.maximum(
        (upper_bounds[:, 0] / velocity_joint).amax(dim=1),
        (upper_bounds[:, 1] / acceleration_joint).amax(dim=1).sqrt(),
    )
    if options.profile == "double_s":
        scale = torch.maximum(
            scale, (upper_bounds[:, 2] / jerk_joint).amax(dim=1).pow(1.0 / 3.0)
        )
    scale = scale.clamp_min(1.0)
    profile = profile.scaled(scale[:, None])
    orders = waypoints.new_tensor([1.0, 2.0, 3.0])
    upper_bounds = upper_bounds / scale[:, None, None].pow(orders[None, :, None])
    segment_duration = profile.durations.sum(dim=-1)
    times = _make_sample_times(segment_duration, options)
    scalar = profile.evaluate(times)
    states = [
        compose_quintic_blend_state(
            waypoints[index],
            options.blend_tolerance,
            path_length[index] * scalar.position[index],
            path_length[index] * scalar.velocity[index],
            path_length[index] * scalar.acceleration[index],
        )
        for index in range(waypoints.shape[0])
    ]
    positions = torch.stack([state[0] for state in states])
    velocities = torch.stack([state[1] for state in states])
    accelerations = torch.stack([state[2] for state in states])
    positions[:, 0], positions[:, -1] = waypoints[:, 0], waypoints[:, -1]
    velocities[:, 0] = velocities[:, -1] = 0.0
    accelerations[:, 0] = accelerations[:, -1] = 0.0
    dt = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    if options.profile == "double_s":
        joint_jerks = torch.stack(
            [
                compose_quintic_blend_jerk(
                    waypoints[index],
                    options.blend_tolerance,
                    path_length[index] * scalar.position[index],
                    path_length[index] * scalar.velocity[index],
                    path_length[index] * scalar.acceleration[index],
                    path_length[index] * scalar.jerk[index],
                )
                for index in range(waypoints.shape[0])
            ]
        )
        peak_jerk_joint = joint_jerks.abs().amax(dim=1)
        peak_jerk = peak_jerk_joint.amax(dim=1)
    else:
        peak_jerk_joint = torch.zeros_like(velocities[:, 0])
        peak_jerk = torch.zeros(
            waypoints.shape[0], dtype=waypoints.dtype, device=waypoints.device
        )
    report = {
        "peak_velocity": velocities.abs().amax(dim=(1, 2)),
        "peak_acceleration": accelerations.abs().amax(dim=(1, 2)),
        "peak_jerk": peak_jerk,
        "velocity_limit": velocity_joint.amax().expand(waypoints.shape[0]),
        "acceleration_limit": acceleration_joint.amax().expand(waypoints.shape[0]),
        "jerk_limit": jerk_joint.amax().expand(waypoints.shape[0]),
    }
    peak_velocity_joint = velocities.abs().amax(dim=1)
    peak_acceleration_joint = accelerations.abs().amax(dim=1)
    report.update(
        {
            "peak_velocity_per_joint": peak_velocity_joint,
            "peak_acceleration_per_joint": peak_acceleration_joint,
            "peak_jerk_per_joint": peak_jerk_joint,
            "velocity_utilization": peak_velocity_joint / velocity_joint,
            "acceleration_utilization": peak_acceleration_joint / acceleration_joint,
            "jerk_utilization": peak_jerk_joint / jerk_joint,
            "velocity_upper_bound_per_joint": upper_bounds[:, 0],
            "acceleration_upper_bound_per_joint": upper_bounds[:, 1],
            # Trapezoidal acceleration jumps are not jerk constrained.
            "jerk_upper_bound_per_joint": (
                upper_bounds[:, 2]
                if options.profile == "double_s"
                else torch.full_like(upper_bounds[:, 2], float("inf"))
            ),
            "within_limits": (
                (upper_bounds[:, 0] <= velocity_joint * (1.0 + 1e-6)).all(dim=1)
                & (upper_bounds[:, 1] <= acceleration_joint * (1.0 + 1e-6)).all(dim=1)
                & (
                    (upper_bounds[:, 2] <= jerk_joint * (1.0 + 1e-6)).all(dim=1)
                    if options.profile == "double_s"
                    else torch.ones_like(scale, dtype=torch.bool)
                )
            ),
        }
    )
    return PlanResult(
        success=torch.ones(
            waypoints.shape[0], dtype=torch.bool, device=waypoints.device
        ),
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        dt=dt,
        constraint_report=report,
    )


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
        times = _make_sample_times(duration[:, None], options)
        positions = waypoints[:, :1].expand(-1, times.shape[1], dof).clone()
        velocities = torch.zeros_like(positions)
        accelerations = torch.zeros_like(positions)
        zero = waypoints.new_zeros((batch_size,))
        zero_joint = waypoints.new_zeros((batch_size, dof))
        velocity_joint = _limit_tensor(options.constraints["velocity"], waypoints)
        acceleration_joint = _limit_tensor(
            options.constraints["acceleration"], waypoints
        )
        jerk_joint = _limit_tensor(options.constraints.get("jerk", 1.0), waypoints)
        report = {
            "peak_velocity": zero,
            "peak_acceleration": zero.clone(),
            "peak_jerk": zero.clone(),
            "velocity_limit": velocity_joint.amax().expand(batch_size),
            "acceleration_limit": acceleration_joint.amax().expand(batch_size),
            "jerk_limit": jerk_joint.amax().expand(batch_size),
            "peak_velocity_per_joint": zero_joint,
            "peak_acceleration_per_joint": zero_joint.clone(),
            "peak_jerk_per_joint": zero_joint.clone(),
            "velocity_utilization": zero_joint.clone(),
            "acceleration_utilization": zero_joint.clone(),
            "jerk_utilization": zero_joint.clone(),
            "velocity_upper_bound_per_joint": zero_joint.clone(),
            "acceleration_upper_bound_per_joint": zero_joint.clone(),
            "jerk_upper_bound_per_joint": zero_joint.clone(),
            "within_limits": torch.ones(
                batch_size, dtype=torch.bool, device=waypoints.device
            ),
        }
        return PlanResult(
            success=torch.ones(batch_size, dtype=torch.bool, device=waypoints.device),
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dt=torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1])),
            constraint_report=report,
        )

    if options.blend_tolerance > 0.0 and bool(stationary_path.any().item()):
        rows = [
            _plan_linear_profiles(waypoints[index : index + 1], options)
            for index in range(batch_size)
        ]
        output_count = max(row.positions.shape[1] for row in rows)

        def pad(value: torch.Tensor, *, repeat_final: bool) -> torch.Tensor:
            missing = output_count - value.shape[1]
            if missing == 0:
                return value
            tail = (
                value[:, -1:].expand(-1, missing, *value.shape[2:])
                if repeat_final
                else value.new_zeros((1, missing, *value.shape[2:]))
            )
            return torch.cat((value, tail), dim=1)

        report_keys = set.intersection(
            *(set(row.constraint_report or {}) for row in rows)
        )
        report = {
            key: torch.cat([row.constraint_report[key] for row in rows], dim=0)
            for key in report_keys
        }
        return PlanResult(
            success=torch.cat([row.success for row in rows]),
            positions=torch.cat(
                [pad(row.positions, repeat_final=True) for row in rows]
            ),
            velocities=torch.cat(
                [pad(row.velocities, repeat_final=False) for row in rows]
            ),
            accelerations=torch.cat(
                [pad(row.accelerations, repeat_final=False) for row in rows]
            ),
            dt=torch.cat([pad(row.dt, repeat_final=False) for row in rows]),
            constraint_report=report,
        )

    if options.blend_tolerance > 0.0:
        return _plan_blended_profiles(waypoints, options)

    velocity_limit, acceleration_limit, jerk_limit = _scalar_path_limits(
        delta, options.constraints
    )
    profile = ScalarTimeLaw.build(
        profile_name=options.profile,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        jerk_limit=jerk_limit,
        backend=options.backend,
    )
    profile = profile.with_minimum_duration(options.minimum_duration)
    segment_duration = profile.durations.sum(dim=-1)
    times = _make_sample_times(segment_duration, options)
    positions, velocities, accelerations = profile.compose(
        times=times,
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
    # Keep a backend-independent diagnostic report for callers that need to
    # validate the realized trajectory against projected limits.  Values are
    # per environment and are computed from the returned samples.
    report = {
        "peak_velocity": velocities.abs().amax(dim=(1, 2)),
        "peak_acceleration": accelerations.abs().amax(dim=(1, 2)),
        "velocity_limit": velocity_limit.amax(dim=1),
        "acceleration_limit": acceleration_limit.amax(dim=1),
    }
    # Use analytic phase jerk. Finite differences across trapezoidal
    # acceleration jumps otherwise produce sample-rate-dependent spikes.
    peak_jerk = (
        (profile.jerks[..., None] * delta[:, :, None, :]).abs().amax(dim=(1, 2, 3))
    )
    report["peak_jerk"] = peak_jerk
    if jerk_limit is not None:
        report["jerk_limit"] = jerk_limit.amax(dim=1)
    joint_velocity_limit = _limit_tensor(options.constraints["velocity"], waypoints)
    joint_acceleration_limit = _limit_tensor(
        options.constraints["acceleration"], waypoints
    )
    joint_jerk_limit = _limit_tensor(options.constraints.get("jerk", 1.0), waypoints)
    peak_velocity_joint = velocities.abs().amax(dim=1)
    peak_acceleration_joint = accelerations.abs().amax(dim=1)
    peak_jerk_joint = (
        (profile.jerks[..., None] * delta[:, :, None, :]).abs().amax(dim=(1, 2))
    )
    report.update(
        {
            "peak_velocity_per_joint": peak_velocity_joint,
            "peak_acceleration_per_joint": peak_acceleration_joint,
            "peak_jerk_per_joint": peak_jerk_joint,
            "velocity_utilization": peak_velocity_joint / joint_velocity_limit,
            "acceleration_utilization": peak_acceleration_joint
            / joint_acceleration_limit,
            "jerk_utilization": peak_jerk_joint / joint_jerk_limit,
            "within_limits": (
                (peak_velocity_joint <= joint_velocity_limit * (1.0 + 1e-9)).all(dim=1)
                & (
                    peak_acceleration_joint <= joint_acceleration_limit * (1.0 + 1e-9)
                ).all(dim=1)
                & (peak_jerk_joint <= joint_jerk_limit * (1.0 + 1e-9)).all(dim=1)
            ),
        }
    )
    return PlanResult(
        success=success,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        dt=dt,
        constraint_report=report,
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
