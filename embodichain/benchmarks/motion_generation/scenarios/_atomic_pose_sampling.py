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

"""Pure deterministic pose sampling and typed randomization context."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields

import torch

from ..config import AtomicPoseRandomizationCfg

__all__: list[str] = []


def float_vector(
    value: object,
    *,
    name: str,
    length: int,
    default: Sequence[float] | None = None,
) -> list[float]:
    """Resolve a finite numeric vector from a YAML-compatible value."""
    resolved = default if value is None else value
    if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of {length} numbers.")
    result = [float(item) for item in resolved]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


def seeded_jitter(
    amplitude: Sequence[float],
    *,
    seed: int,
    stream: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
    seed_stride: int = 1,
    sampling_pattern: str = "uniform",
) -> torch.Tensor:
    """Sample deterministic jitter, optionally one row per environment.

    The legacy ``batch_size=1`` result remains a one-dimensional vector.  For
    a pose-batch benchmark the returned tensor has shape ``(B, D)``. Sampling
    is stable across devices and independent of planner execution order.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if type(seed_stride) is not int or seed_stride < 1:
        raise ValueError("seed_stride must be a positive integer.")
    if sampling_pattern not in {"uniform", "stratified", "workspace_shell"}:
        raise ValueError(
            "sampling_pattern must be 'uniform', 'stratified', or " "'workspace_shell'."
        )
    if (
        sampling_pattern == "workspace_shell"
        and len(amplitude) == 3
        and ((float(amplitude[0]) > 0.0) != (float(amplitude[1]) > 0.0))
    ):
        raise ValueError(
            "workspace_shell requires both X and Y amplitudes when either is active."
        )
    base_seed = (int(seed) * 1_000_003 + int(stream) * 97_409) % (2**63 - 1)
    # Keep the historical uniform sequence byte-for-byte stable.  Structured
    # patterns use deterministic CPU streams so device and planner order do
    # not change the generated population.
    if sampling_pattern == "uniform":
        rows: list[torch.Tensor] = []
        for row in range(batch_size):
            generator = torch.Generator(device="cpu")
            generator.manual_seed((base_seed + row * seed_stride) % (2**63 - 1))
            rows.append(
                torch.rand(len(amplitude), generator=generator, dtype=torch.float32)
            )
        unit = torch.stack(rows, dim=0)
        normalized = 2.0 * unit - 1.0
    elif (
        sampling_pattern == "stratified"
        or len(amplitude) != 3
        or not any(float(value) > 0.0 for value in amplitude[:2])
    ):
        # Each active dimension visits every stratum exactly once.  Inclusive
        # endpoints deliberately exercise both configured workspace bounds;
        # independent seeded permutations avoid coupling dimensions along one
        # diagonal while retaining exact reproducibility.
        levels = (
            torch.zeros(1, dtype=torch.float32)
            if batch_size == 1
            else torch.linspace(-1.0, 1.0, batch_size, dtype=torch.float32)
        )
        columns: list[torch.Tensor] = []
        for dimension in range(len(amplitude)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                (base_seed + (dimension + 1) * 1_000_033 * seed_stride) % (2**63 - 1)
            )
            columns.append(levels[torch.randperm(batch_size, generator=generator)])
        normalized = torch.stack(columns, dim=1)
    else:
        radii, _ = workspace_shell_profile(batch_size)
        angle_generator = torch.Generator(device="cpu")
        angle_generator.manual_seed((base_seed + 1_000_033 * seed_stride) % (2**63 - 1))
        angle_order = torch.randperm(batch_size, generator=angle_generator)
        phase = (
            2.0
            * math.pi
            * torch.rand(1, generator=angle_generator, dtype=torch.float32)[0]
        )
        angles = 2.0 * math.pi * angle_order.to(torch.float32) / batch_size + phase
        z_levels = (
            torch.zeros(1, dtype=torch.float32)
            if batch_size == 1
            else torch.linspace(-1.0, 1.0, batch_size, dtype=torch.float32)
        )
        z_generator = torch.Generator(device="cpu")
        z_generator.manual_seed((base_seed + 2_000_066 * seed_stride) % (2**63 - 1))
        normalized = torch.stack(
            (
                radii * torch.cos(angles),
                radii * torch.sin(angles),
                z_levels[torch.randperm(batch_size, generator=z_generator)],
            ),
            dim=1,
        )
    values = normalized * torch.tensor(amplitude, dtype=torch.float32)
    values = values.to(dtype=dtype, device=device)
    return values[0] if batch_size == 1 else values


def workspace_shell_profile(batch_size: int) -> tuple[torch.Tensor, list[str]]:
    """Return ordered normalized radii and near/mid/far workspace labels."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    radii = torch.linspace(
        1.0 / batch_size,
        1.0,
        batch_size,
        dtype=torch.float32,
    )
    labels = [
        "near" if radius <= 1.0 / 3.0 else "mid" if radius <= 2.0 / 3.0 else "far"
        for radius in radii.tolist()
    ]
    return radii, labels


def seeded_yaw_rotation(
    *,
    batch_size: int,
    amplitude: object,
    name: str,
    seed: int,
    stream: int,
    seed_stride: int,
    dtype: torch.dtype,
    device: torch.device,
    sampling_pattern: str = "uniform",
) -> torch.Tensor:
    """Return deterministic world-Z rotation matrices for a pose batch."""
    yaw_amplitude = float(amplitude)
    if not math.isfinite(yaw_amplitude) or yaw_amplitude < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    yaw = seeded_jitter(
        (yaw_amplitude,),
        seed=seed,
        stream=stream,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        seed_stride=seed_stride,
        sampling_pattern=sampling_pattern,
    )
    if yaw.dim() == 1:
        yaw = yaw[:1]
    else:
        yaw = yaw[:, 0]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    yaw_rotation = torch.eye(3, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    yaw_rotation[:, 0, 0] = cosine
    yaw_rotation[:, 0, 1] = -sine
    yaw_rotation[:, 1, 0] = sine
    yaw_rotation[:, 1, 1] = cosine
    return yaw_rotation


def seeded_roll_pitch_rotation(
    *,
    batch_size: int,
    amplitude: object,
    name: str,
    seed: int,
    stream: int,
    seed_stride: int,
    dtype: torch.dtype,
    device: torch.device,
    sampling_pattern: str = "uniform",
) -> torch.Tensor:
    """Return deterministic ``Ry(pitch) @ Rx(roll)`` rotations."""
    roll_pitch_amplitude = float_vector(
        amplitude,
        name=name,
        length=2,
        default=(0.0, 0.0),
    )
    if any(value < 0.0 for value in roll_pitch_amplitude):
        raise ValueError(f"{name} values must be non-negative.")
    angles = seeded_jitter(
        roll_pitch_amplitude,
        seed=seed,
        stream=stream,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        seed_stride=seed_stride,
        sampling_pattern=sampling_pattern,
    )
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    roll = angles[:, 0]
    pitch = angles[:, 1]
    roll_rotation = torch.eye(3, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    roll_rotation[:, 1, 1] = torch.cos(roll)
    roll_rotation[:, 1, 2] = -torch.sin(roll)
    roll_rotation[:, 2, 1] = torch.sin(roll)
    roll_rotation[:, 2, 2] = torch.cos(roll)
    pitch_rotation = torch.eye(3, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    pitch_rotation[:, 0, 0] = torch.cos(pitch)
    pitch_rotation[:, 0, 2] = torch.sin(pitch)
    pitch_rotation[:, 2, 0] = -torch.sin(pitch)
    pitch_rotation[:, 2, 2] = torch.cos(pitch)
    return pitch_rotation @ roll_rotation


def seeded_spatial_rotation(
    *,
    batch_size: int,
    roll_pitch_amplitude: object,
    yaw_amplitude: object,
    roll_pitch_name: str,
    yaw_name: str,
    seed: int,
    roll_pitch_stream: int,
    yaw_stream: int,
    seed_stride: int,
    dtype: torch.dtype,
    device: torch.device,
    sampling_pattern: str = "uniform",
) -> torch.Tensor:
    """Return deterministic world-frame ``Rz @ Ry @ Rx`` rotations."""
    roll_pitch = seeded_roll_pitch_rotation(
        batch_size=batch_size,
        amplitude=roll_pitch_amplitude,
        name=roll_pitch_name,
        seed=seed,
        stream=roll_pitch_stream,
        seed_stride=seed_stride,
        dtype=dtype,
        device=device,
        sampling_pattern=sampling_pattern,
    )
    yaw = seeded_yaw_rotation(
        batch_size=batch_size,
        amplitude=yaw_amplitude,
        name=yaw_name,
        seed=seed,
        stream=yaw_stream,
        seed_stride=seed_stride,
        dtype=dtype,
        device=device,
        sampling_pattern=sampling_pattern,
    )
    return yaw @ roll_pitch


def apply_seeded_spatial_rotation(
    pose: torch.Tensor,
    *,
    roll_pitch_amplitude: object,
    yaw_amplitude: object,
    roll_pitch_name: str,
    yaw_name: str,
    seed: int,
    roll_pitch_stream: int,
    yaw_stream: int,
    seed_stride: int,
    sampling_pattern: str = "uniform",
) -> None:
    """Pre-multiply each pose by a deterministic world-frame rotation."""
    rotation = seeded_spatial_rotation(
        batch_size=pose.shape[0],
        roll_pitch_amplitude=roll_pitch_amplitude,
        yaw_amplitude=yaw_amplitude,
        roll_pitch_name=roll_pitch_name,
        yaw_name=yaw_name,
        seed=seed,
        roll_pitch_stream=roll_pitch_stream,
        yaw_stream=yaw_stream,
        seed_stride=seed_stride,
        dtype=pose.dtype,
        device=pose.device,
        sampling_pattern=sampling_pattern,
    )
    pose[:, :3, :3] = rotation @ pose[:, :3, :3]


_DEFAULT_POSE_SETTINGS = AtomicPoseRandomizationCfg()


def _pose_settings(config: Mapping[str, object]) -> AtomicPoseRandomizationCfg:
    """Read the typed context, with decoding only for legacy private mappings."""
    if "_pose_batch" in config:
        settings = config["_pose_batch"]
        if not isinstance(settings, AtomicPoseRandomizationCfg):
            raise TypeError("_pose_batch must be an AtomicPoseRandomizationCfg.")
        return settings
    if not any(str(key).startswith("_pose_batch_") for key in config):
        return _DEFAULT_POSE_SETTINGS
    aliases = {
        "pose_batch_size": "size",
        "target_translation_jitter_m": "translation_jitter_m",
    }
    values = {}
    for field in fields(AtomicPoseRandomizationCfg):
        key = "_pose_batch_" + aliases.get(field.name, field.name)
        if key in config:
            value = config[key]
            values[field.name] = (
                tuple(value)
                if isinstance(getattr(_DEFAULT_POSE_SETTINGS, field.name), tuple)
                else value
            )
    return AtomicPoseRandomizationCfg(**values)
