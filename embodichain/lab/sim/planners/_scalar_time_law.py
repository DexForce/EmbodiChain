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

"""Shared batched trapezoidal and jerk-limited scalar time laws."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, NamedTuple

import torch

__all__ = ["ScalarState", "ScalarTimeLaw"]


class ScalarState(NamedTuple):
    """Scalar position and its first three time derivatives, each shaped (B, N)."""

    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor


def validate_profile_name(profile: str) -> None:
    """Reject unsupported time laws before backend dispatch."""
    if profile not in ("trapezoidal", "double_s"):
        raise ValueError("profile must be 'trapezoidal' or 'double_s'.")


def validate_minimum_duration(minimum_duration: float | None) -> None:
    """Accept no lower bound or a finite non-negative duration in seconds."""
    if minimum_duration is not None and (
        isinstance(minimum_duration, bool)
        or not isinstance(minimum_duration, (int, float))
        or not math.isfinite(minimum_duration)
        or minimum_duration < 0.0
    ):
        raise ValueError("minimum_duration must be finite and non-negative when set.")


@dataclass(slots=True)
class ScalarTimeLaw:
    """Shared batched rest-to-rest timing with phase tensors shaped (B, S, P).

    Each of the S segments traverses a unit interval. Joint and Cartesian
    planners own geometry; this type owns phase construction and evaluation.
    """

    durations: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    jerks: torch.Tensor

    @classmethod
    def build(
        cls,
        *,
        profile_name: Literal["trapezoidal", "double_s"],
        velocity_limit: torch.Tensor,
        acceleration_limit: torch.Tensor,
        jerk_limit: torch.Tensor,
        backend: Literal["auto", "torch", "warp"] = "torch",
    ) -> ScalarTimeLaw:
        """Build unit-distance time laws from projected limits shaped (B, S)."""
        validate_profile_name(profile_name)
        return _build_scalar_profile(
            profile_name=profile_name,
            velocity_limit=velocity_limit,
            acceleration_limit=acceleration_limit,
            jerk_limit=jerk_limit,
            backend=backend,
        )

    def with_minimum_duration(self, duration: float | None) -> ScalarTimeLaw:
        """Return a uniformly slowed law with a lower bound on each row's duration."""
        validate_minimum_duration(duration)
        return _apply_minimum_duration(self, duration)

    def scaled(self, scale: torch.Tensor) -> ScalarTimeLaw:
        """Scale time by factors shaped (B, S), preserving path positions."""
        return _scale_profile_time(self, scale)

    def evaluate(self, times: torch.Tensor) -> ScalarState:
        """Evaluate all derivatives at (B, N) times using one phase lookup."""
        _, state = _evaluate_scalar_samples_torch(
            times, self.durations.sum(dim=-1).cumsum(dim=-1), self
        )
        return state

    def compose(
        self,
        times: torch.Tensor,
        segment_starts: torch.Tensor,
        segment_deltas: torch.Tensor,
        *,
        backend: Literal["auto", "torch", "warp"] = "torch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose scalar timing with affine path segments shaped (B, S, D)."""
        return _compose_profile_samples(
            times=times,
            cumulative_duration=self.durations.sum(dim=-1).cumsum(dim=-1),
            profile=self,
            segment_starts=segment_starts,
            segment_deltas=segment_deltas,
            backend=backend,
        )


def _integrate_phases(durations: torch.Tensor, jerks: torch.Tensor) -> ScalarTimeLaw:
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
    return ScalarTimeLaw(durations, positions, velocities, accelerations, jerks)


def _apply_minimum_duration(
    profile: ScalarTimeLaw,
    minimum_duration: float | None,
) -> ScalarTimeLaw:
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
    slowed = ScalarTimeLaw(
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


def _scale_profile_time(profile: ScalarTimeLaw, scale: torch.Tensor) -> ScalarTimeLaw:
    """Apply per-segment uniform time scaling without changing path position."""
    phase_scale = scale[..., None]
    return ScalarTimeLaw(
        durations=profile.durations * phase_scale,
        positions=profile.positions,
        velocities=profile.velocities / phase_scale,
        accelerations=profile.accelerations / phase_scale.square(),
        jerks=profile.jerks / phase_scale.pow(3),
    )


def _build_trapezoidal_profile(
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
) -> ScalarTimeLaw:
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
) -> ScalarTimeLaw:
    """Build the rest-to-rest seven-phase Double-S profile.

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

    # Use the intentionally discrete feasibility search required by the
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
        raise RuntimeError("Double-S acceleration search did not converge.")
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
) -> ScalarTimeLaw:
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
            result = ScalarTimeLaw(*tensors)
    if not use_warp:
        result = (
            _build_double_s_profile(velocity_limit, acceleration_limit, jerk_limit)
            if profile_name == "double_s"
            else _build_trapezoidal_profile(velocity_limit, acceleration_limit)
        )

    if profile_name in {"double_s", "trapezoidal"}:
        # HolisticMotion applies a 1% duration margin after limit projection
        # so sampled derivatives remain strictly inside their constraints.
        margin = torch.where(
            velocity_limit > 0.0,
            torch.full_like(velocity_limit, 1.01),
            torch.ones_like(velocity_limit),
        )
        result = _scale_profile_time(result, margin)
    return result


def _evaluate_scalar_samples_torch(
    times: torch.Tensor,
    cumulative_duration: torch.Tensor,
    profile: ScalarTimeLaw,
) -> tuple[torch.Tensor, ScalarState]:
    """Look up the active segment and phase once for all scalar derivatives."""
    segment = torch.searchsorted(
        cumulative_duration.contiguous(), times.contiguous(), right=True
    )
    segment = segment.clamp_max(cumulative_duration.shape[1] - 1)
    previous_duration = torch.cat(
        [torch.zeros_like(cumulative_duration[:, :1]), cumulative_duration[:, :-1]],
        dim=-1,
    )
    local_time = times - torch.gather(previous_duration, 1, segment)
    batch_ids = torch.arange(times.shape[0], device=times.device)[:, None].expand_as(
        segment
    )
    selected_durations = profile.durations[batch_ids, segment]
    # HolisticMotion clamps evaluation outside the trajectory domain to the
    # segment endpoint.  Clamp before phase lookup so extrapolation cannot
    # produce overshoot for callers supplying out-of-range sample times.
    local_time = local_time.clamp_min(0.0).clamp_max(selected_durations.sum(dim=-1))
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
    return segment, ScalarState(path_position, path_velocity, path_acceleration, jerk)


def _compose_profile_samples_torch(
    *,
    times: torch.Tensor,
    cumulative_duration: torch.Tensor,
    profile: ScalarTimeLaw,
    segment_starts: torch.Tensor,
    segment_deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate scalar profiles and compose joint states with Torch."""
    segment, state = _evaluate_scalar_samples_torch(times, cumulative_duration, profile)
    batch_ids = torch.arange(times.shape[0], device=times.device)[:, None]
    selected_delta = segment_deltas[batch_ids, segment]
    selected_start = segment_starts[batch_ids, segment]
    return (
        selected_start + selected_delta * state.position[..., None],
        selected_delta * state.velocity[..., None],
        selected_delta * state.acceleration[..., None],
    )


def _compose_profile_samples(
    *,
    times: torch.Tensor,
    cumulative_duration: torch.Tensor,
    profile: ScalarTimeLaw,
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
