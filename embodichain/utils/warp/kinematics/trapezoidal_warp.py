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

"""Warp profile construction and sampling for the trapezoidal planner."""

from __future__ import annotations

import torch
import warp as wp

__all__ = ["build_profile_warp", "compose_profile_samples_warp"]


@wp.kernel(enable_backward=False)
def _build_trapezoidal_profile_kernel(
    velocity_limits: wp.array(dtype=wp.float32),
    acceleration_limits: wp.array(dtype=wp.float32),
    durations: wp.array(dtype=wp.float32),
    positions: wp.array(dtype=wp.float32),
    velocities: wp.array(dtype=wp.float32),
    accelerations: wp.array(dtype=wp.float32),
    jerks: wp.array(dtype=wp.float32),
) -> None:
    profile = wp.tid()
    velocity_limit = velocity_limits[profile]
    requested_acceleration = acceleration_limits[profile]
    acceleration_limit = wp.max(requested_acceleration, 1.0e-12)
    stationary = velocity_limit <= 0.0
    peak_velocity = velocity_limit
    reaches_velocity = velocity_limit * velocity_limit / acceleration_limit < 1.0
    if not reaches_velocity:
        peak_velocity = wp.sqrt(acceleration_limit)
    acceleration_time = peak_velocity / acceleration_limit
    cruise_time = float(0.0)
    if reaches_velocity and not stationary:
        cruise_time = (
            1.0 - peak_velocity * peak_velocity / acceleration_limit
        ) / peak_velocity
    if stationary:
        acceleration_time = 0.0
        cruise_time = 0.0

    offset = profile * 3
    durations[offset] = acceleration_time
    durations[offset + 1] = cruise_time
    durations[offset + 2] = acceleration_time
    positions[offset] = 0.0
    positions[offset + 1] = 0.5 * peak_velocity * acceleration_time
    positions[offset + 2] = positions[offset + 1] + peak_velocity * cruise_time
    velocities[offset] = 0.0
    velocities[offset + 1] = peak_velocity
    velocities[offset + 2] = peak_velocity
    accelerations[offset] = requested_acceleration
    accelerations[offset + 1] = 0.0
    accelerations[offset + 2] = -requested_acceleration
    jerks[offset] = 0.0
    jerks[offset + 1] = 0.0
    jerks[offset + 2] = 0.0


@wp.kernel(enable_backward=False)
def _build_double_s_profile_kernel(
    velocity_limits: wp.array(dtype=wp.float32),
    acceleration_limits: wp.array(dtype=wp.float32),
    jerk_limits: wp.array(dtype=wp.float32),
    durations: wp.array(dtype=wp.float32),
    positions: wp.array(dtype=wp.float32),
    velocities: wp.array(dtype=wp.float32),
    accelerations: wp.array(dtype=wp.float32),
    jerks: wp.array(dtype=wp.float32),
) -> None:
    profile = wp.tid()
    velocity_limit = velocity_limits[profile]
    acceleration_limit = wp.max(acceleration_limits[profile], 1.0e-12)
    jerk_limit = wp.max(jerk_limits[profile], 1.0e-12)
    stationary = velocity_limit <= 0.0
    safe_velocity = wp.max(velocity_limit, 1.0e-12)

    tj = float(acceleration_limit / jerk_limit)
    ta = float(tj + safe_velocity / acceleration_limit)
    if safe_velocity * jerk_limit < acceleration_limit * acceleration_limit:
        tj = wp.sqrt(safe_velocity / jerk_limit)
        ta = 2.0 * tj
    cruise_time = float(1.0 / safe_velocity - ta)
    if not stationary and cruise_time <= 0.0:
        candidate_acceleration = float(acceleration_limit)
        accepted = bool(False)
        reduction = int(0)
        while not accepted and reduction <= 1000:
            candidate_tj = candidate_acceleration / jerk_limit
            acceleration_squared = candidate_acceleration * candidate_acceleration
            delta = wp.sqrt(
                acceleration_squared * acceleration_squared / (jerk_limit * jerk_limit)
                + 4.0 * candidate_acceleration
            )
            candidate_ta = (acceleration_squared / jerk_limit + delta) / (
                2.0 * candidate_acceleration
            )
            if candidate_ta >= 2.0 * candidate_tj:
                tj = candidate_tj
                ta = candidate_ta
                accepted = True
            else:
                candidate_acceleration *= 0.9
            reduction += 1
        cruise_time = 0.0
    if stationary:
        tj = 0.0
        ta = 0.0
        cruise_time = 0.0

    constant_acceleration_time = wp.max(ta - 2.0 * tj, 0.0)
    offset = profile * 7
    for phase in range(7):
        duration = tj
        jerk = jerk_limit
        if phase == 1 or phase == 5:
            duration = constant_acceleration_time
            jerk = 0.0
        elif phase == 3:
            duration = cruise_time
            jerk = 0.0
        elif phase == 2 or phase == 4:
            jerk = -jerk_limit
        if stationary:
            jerk = 0.0
        durations[offset + phase] = duration
        jerks[offset + phase] = jerk

    position = float(0.0)
    velocity = float(0.0)
    acceleration = float(0.0)
    for phase in range(7):
        index = offset + phase
        positions[index] = position
        velocities[index] = velocity
        accelerations[index] = acceleration
        duration = durations[index]
        jerk = jerks[index]
        position += (
            velocity * duration
            + 0.5 * acceleration * duration * duration
            + jerk * duration * duration * duration / 6.0
        )
        velocity += acceleration * duration + 0.5 * jerk * duration * duration
        acceleration += jerk * duration


@wp.kernel(enable_backward=False)
def _evaluate_profile_samples_kernel(
    times: wp.array(dtype=wp.float32),
    cumulative_segment_time: wp.array(dtype=wp.float32),
    phase_durations: wp.array(dtype=wp.float32),
    phase_positions: wp.array(dtype=wp.float32),
    phase_velocities: wp.array(dtype=wp.float32),
    phase_accelerations: wp.array(dtype=wp.float32),
    phase_jerks: wp.array(dtype=wp.float32),
    sample_count: int,
    segment_count: int,
    phase_count: int,
    sample_segments: wp.array(dtype=wp.int32),
    path_positions: wp.array(dtype=wp.float32),
    path_velocities: wp.array(dtype=wp.float32),
    path_accelerations: wp.array(dtype=wp.float32),
) -> None:
    batch, sample = wp.tid()
    sample_offset = batch * sample_count + sample
    time = times[sample_offset]
    low = int(0)  # noqa: RUF046, UP018 - mutable Warp binary-search bound.
    high = int(segment_count)
    while low < high:
        middle = (low + high) // 2
        end_time = cumulative_segment_time[batch * segment_count + middle]
        if time >= end_time:
            low = middle + 1
        else:
            high = middle
    segment = low
    if segment >= segment_count:
        segment = segment_count - 1

    previous_segment_time = float(0.0)  # noqa: UP018 - mutable Warp value.
    if segment > 0:
        previous_segment_time = cumulative_segment_time[
            batch * segment_count + segment - 1
        ]
    local_time = time - previous_segment_time
    profile_offset = (batch * segment_count + segment) * phase_count
    phase = int(0)  # noqa: RUF046, UP018 - Warp requires a mutable typed value.
    phase_start_time = float(0.0)  # noqa: UP018 - mutable Warp value.
    cumulative_phase_time = float(0.0)  # noqa: UP018 - mutable Warp value.
    for candidate in range(phase_count):
        duration = phase_durations[profile_offset + candidate]
        cumulative_phase_time += duration
        if local_time >= cumulative_phase_time:
            phase = candidate + 1
            phase_start_time = cumulative_phase_time
    if phase >= phase_count:
        phase = phase_count - 1
        phase_start_time = (
            cumulative_phase_time - phase_durations[profile_offset + phase]
        )

    profile_index = profile_offset + phase
    tau = wp.max(local_time - phase_start_time, 0.0)
    p0 = phase_positions[profile_index]
    v0 = phase_velocities[profile_index]
    a0 = phase_accelerations[profile_index]
    jerk = phase_jerks[profile_index]
    path_position = p0 + v0 * tau + 0.5 * a0 * tau * tau + jerk * tau * tau * tau / 6.0
    path_velocity = v0 + a0 * tau + 0.5 * jerk * tau * tau
    path_acceleration = a0 + jerk * tau

    sample_segments[sample_offset] = segment
    path_positions[sample_offset] = path_position
    path_velocities[sample_offset] = path_velocity
    path_accelerations[sample_offset] = path_acceleration


@wp.kernel(enable_backward=False)
def _compose_joint_samples_kernel(
    sample_segments: wp.array(dtype=wp.int32),
    path_positions: wp.array(dtype=wp.float32),
    path_velocities: wp.array(dtype=wp.float32),
    path_accelerations: wp.array(dtype=wp.float32),
    segment_starts: wp.array(dtype=wp.float32),
    segment_deltas: wp.array(dtype=wp.float32),
    sample_count: int,
    segment_count: int,
    dof: int,
    positions: wp.array(dtype=wp.float32),
    velocities: wp.array(dtype=wp.float32),
    accelerations: wp.array(dtype=wp.float32),
) -> None:
    batch, sample, joint = wp.tid()
    sample_offset = batch * sample_count + sample
    segment = sample_segments[sample_offset]
    joint_offset = (batch * segment_count + segment) * dof + joint
    output_offset = sample_offset * dof + joint
    delta = segment_deltas[joint_offset]
    positions[output_offset] = (
        segment_starts[joint_offset] + delta * path_positions[sample_offset]
    )
    velocities[output_offset] = delta * path_velocities[sample_offset]
    accelerations[output_offset] = delta * path_accelerations[sample_offset]


def build_profile_warp(
    *,
    profile: str,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
    jerk_limits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct all scalar segment profiles in one Warp launch.

    Args:
        profile: ``"trapezoidal"`` or ``"double_s"``.
        velocity_limits: Projected path velocity limits shaped ``(B, S)``.
        acceleration_limits: Projected acceleration limits shaped ``(B, S)``.
        jerk_limits: Projected jerk limits shaped ``(B, S)``.

    Returns:
        Phase durations, positions, velocities, accelerations, and jerks, each
        shaped ``(B, S, P)`` where ``P`` is three or seven.

    Raises:
        ValueError: If the profile, shapes, or dtypes are unsupported.
    """
    if profile not in {"trapezoidal", "double_s"}:
        raise ValueError(f"Unsupported Warp trajectory profile: {profile!r}.")
    limits = (velocity_limits, acceleration_limits, jerk_limits)
    if any(limit.dtype != torch.float32 for limit in limits):
        raise ValueError("The Warp trajectory backend requires float32 tensors.")
    if any(limit.device != velocity_limits.device for limit in limits[1:]):
        raise ValueError("Warp trajectory limit tensors must share one device.")
    if any(limit.shape != velocity_limits.shape for limit in limits[1:]):
        raise ValueError("Warp trajectory limit tensors must have matching shapes.")
    if velocity_limits.ndim != 2:
        raise ValueError("Warp trajectory limits must have shape (B, S).")

    wp.init()
    phase_count = 3 if profile == "trapezoidal" else 7
    output_shape = (*velocity_limits.shape, phase_count)
    outputs = [
        torch.empty(output_shape, dtype=torch.float32, device=velocity_limits.device)
        for _ in range(5)
    ]
    warp_limits = [wp.from_torch(limit.contiguous().flatten()) for limit in limits]
    warp_outputs = [wp.from_torch(output.flatten()) for output in outputs]
    kernel = (
        _build_trapezoidal_profile_kernel
        if profile == "trapezoidal"
        else _build_double_s_profile_kernel
    )
    kernel_inputs = warp_limits[:2] if profile == "trapezoidal" else warp_limits
    wp.launch(
        kernel=kernel,
        dim=velocity_limits.numel(),
        inputs=kernel_inputs,
        outputs=warp_outputs,
        device=str(velocity_limits.device),
    )
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


def compose_profile_samples_warp(
    *,
    times: torch.Tensor,
    cumulative_segment_time: torch.Tensor,
    phase_durations: torch.Tensor,
    phase_positions: torch.Tensor,
    phase_velocities: torch.Tensor,
    phase_accelerations: torch.Tensor,
    phase_jerks: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate scalar profiles once per sample, then compose all joints."""
    wp.init()
    tensors = (
        times,
        cumulative_segment_time,
        phase_durations,
        phase_positions,
        phase_velocities,
        phase_accelerations,
        phase_jerks,
        segment_starts,
        segment_deltas,
    )
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("The Warp trajectory backend requires float32 tensors.")
    batch_size, sample_count = times.shape
    segment_count = segment_deltas.shape[1]
    dof = segment_deltas.shape[2]
    phase_count = phase_durations.shape[2]
    sample_segments = torch.empty(
        (batch_size, sample_count), dtype=torch.int32, device=times.device
    )
    path_positions = torch.empty_like(times)
    path_velocities = torch.empty_like(times)
    path_accelerations = torch.empty_like(times)
    positions = torch.empty(
        (batch_size, sample_count, dof), dtype=torch.float32, device=times.device
    )
    velocities = torch.empty_like(positions)
    accelerations = torch.empty_like(positions)
    inputs = [wp.from_torch(tensor.contiguous().flatten()) for tensor in tensors]
    scalar_outputs = [
        wp.from_torch(sample_segments.flatten()),
        wp.from_torch(path_positions.flatten()),
        wp.from_torch(path_velocities.flatten()),
        wp.from_torch(path_accelerations.flatten()),
    ]
    joint_outputs = [
        wp.from_torch(positions.flatten()),
        wp.from_torch(velocities.flatten()),
        wp.from_torch(accelerations.flatten()),
    ]
    wp.launch(
        kernel=_evaluate_profile_samples_kernel,
        dim=(batch_size, sample_count),
        inputs=[
            *inputs[:7],
            sample_count,
            segment_count,
            phase_count,
        ],
        outputs=scalar_outputs,
        device=str(times.device),
    )
    wp.launch(
        kernel=_compose_joint_samples_kernel,
        dim=(batch_size, sample_count, dof),
        inputs=[
            *scalar_outputs,
            *inputs[7:],
            sample_count,
            segment_count,
            dof,
        ],
        outputs=joint_outputs,
        device=str(times.device),
    )
    return positions, velocities, accelerations
