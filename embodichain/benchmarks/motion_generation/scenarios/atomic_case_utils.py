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

"""Deterministic case identity, pose randomization, and goal construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    HeldObjectState,
    ObjectSemantics,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from ..models import BenchmarkCase
from ._atomic_pose_sampling import (
    _pose_settings,
    apply_seeded_spatial_rotation,
    float_vector,
    seeded_jitter,
    seeded_roll_pitch_rotation,
    seeded_spatial_rotation,
    seeded_yaw_rotation,
    workspace_shell_profile,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from .atomic_skill import AtomicSkillScenario

__all__ = [
    "float_vector",
    "case_name",
    "atomic_case_id",
    "difficulty",
    "slide_direction",
    "seeded_jitter",
    "workspace_shell_profile",
    "seeded_yaw_rotation",
    "seeded_roll_pitch_rotation",
    "seeded_spatial_rotation",
    "apply_seeded_spatial_rotation",
    "task_frame_rotation",
    "apply_task_frame_randomization",
    "case_generation_seed",
    "canonical_case_qpos",
    "snapshot_case_qpos",
    "randomized_vector",
    "randomized_vector_batch",
    "apply_target_rotation_randomization",
    "randomization_parameters",
    "case_pose",
    "held_object_state",
    "nearest_half_turn_pose",
    "twist_arc_poses",
]


TOP_DOWN_ROTATION = (
    (-0.0539, -0.9985, -0.0022),
    (-0.9977, 0.0540, -0.0401),
    (0.0401, 0.0000, -0.9992),
)


TASK_DIFFICULTIES = {"simple", "medium", "hard"}


TASK_FRAME_TRANSLATION_STREAM = 90_001


TASK_FRAME_YAW_STREAM = 90_002


TASK_FRAME_ROLL_PITCH_STREAM = 90_003


CASE_QPOS_RESOLUTION_RAD = 1.0e-4


def case_name(config: Mapping[str, object]) -> str:
    """Return and validate a stable case name."""
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Every atomic skill case must define a non-empty name.")
    return name


def atomic_case_id(
    track_id: str,
    skill_id: str,
    name: str,
    *,
    batch_size: int,
    seed: int,
) -> str:
    """Return an Atomic Skill identity that distinguishes simulator batches."""
    return f"{track_id}:{skill_id}:{name}:b{batch_size}:s{seed}"


def difficulty(config: Mapping[str, object]) -> str:
    """Resolve the explicit, frozen Atomic Skill difficulty label."""
    difficulty = str(config.get("task_difficulty", "simple"))
    if difficulty not in TASK_DIFFICULTIES:
        raise ValueError(f"task_difficulty must be one of {sorted(TASK_DIFFICULTIES)}.")
    return difficulty


def slide_direction(value: object) -> Literal["pull", "push"]:
    """Validate one Slide direction without coercing arbitrary values."""
    if value == "pull":
        return "pull"
    if value == "push":
        return "push"
    raise ValueError("direction must be either 'pull' or 'push'.")


def task_frame_rotation(
    config: Mapping[str, object],
    *,
    seed: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Resolve the shared deterministic 3D task-frame rotation."""
    return seeded_spatial_rotation(
        batch_size=batch_size,
        roll_pitch_amplitude=_pose_settings(config).task_frame_roll_pitch_jitter_rad,
        yaw_amplitude=_pose_settings(config).task_frame_yaw_jitter_rad,
        roll_pitch_name="task_frame_roll_pitch_jitter_rad",
        yaw_name="task_frame_yaw_jitter_rad",
        seed=seed,
        roll_pitch_stream=TASK_FRAME_ROLL_PITCH_STREAM,
        yaw_stream=TASK_FRAME_YAW_STREAM,
        seed_stride=int(_pose_settings(config).seed_stride),
        sampling_pattern=str(_pose_settings(config).sampling_pattern),
        dtype=dtype,
        device=device,
    )


def apply_task_frame_randomization(
    pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    seed: int,
) -> None:
    """Apply one shared 3D task-frame transform to every scene entity."""
    stride = int(_pose_settings(config).seed_stride)
    translation_amplitude = float_vector(
        _pose_settings(config).task_frame_translation_jitter_m,
        name="task_frame_translation_jitter_m",
        length=3,
        default=(0.0, 0.0, 0.0),
    )
    if any(value < 0.0 for value in translation_amplitude):
        raise ValueError("task_frame_translation_jitter_m values must be non-negative.")
    rotation = task_frame_rotation(
        config,
        seed=seed,
        batch_size=pose.shape[0],
        dtype=pose.dtype,
        device=pose.device,
    )
    translation = seeded_jitter(
        translation_amplitude,
        seed=seed,
        stream=TASK_FRAME_TRANSLATION_STREAM,
        dtype=pose.dtype,
        device=pose.device,
        batch_size=pose.shape[0],
        seed_stride=stride,
        sampling_pattern=str(_pose_settings(config).sampling_pattern),
    )
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)
    pose[:, :3, 3] = torch.bmm(rotation, pose[:, :3, 3, None]).squeeze(-1)
    pose[:, :3, 3] += translation
    pose[:, :3, :3] = rotation @ pose[:, :3, :3]


def case_generation_seed(seed: int, *, skill_index: int, case_index: int) -> int:
    """Derive one stable RNG seed for all stochastic IK work in a case."""
    return (
        int(seed) * 1_000_003 + int(skill_index) * 97_409 + int(case_index) * 13_007
    ) % (2**63 - 1)


def canonical_case_qpos(qpos: torch.Tensor) -> torch.Tensor:
    """Remove insignificant device-level IK noise from frozen case states."""
    return torch.round(qpos / CASE_QPOS_RESOLUTION_RAD) * CASE_QPOS_RESOLUTION_RAD


def snapshot_case_qpos(
    robot: "Robot", control_part: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture one canonical full state and its controlled joints."""
    full_qpos = canonical_case_qpos(robot.get_qpos())
    joint_ids = robot.get_joint_ids(name=control_part)
    return full_qpos[:, joint_ids].clone(), full_qpos


def randomized_vector(
    base: Sequence[float],
    config: Mapping[str, object],
    *,
    jitter_name: str,
    seed: int,
    stream: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return one configured vector plus optional deterministic uniform jitter."""
    base_values = [float(value) for value in base]
    amplitude = float_vector(
        config.get(jitter_name),
        name=jitter_name,
        length=len(base_values),
        default=(0.0,) * len(base_values),
    )
    if any(value < 0.0 for value in amplitude):
        raise ValueError(f"{jitter_name} values must be non-negative.")
    return torch.tensor(base_values, dtype=dtype, device=device) + seeded_jitter(
        amplitude,
        seed=seed,
        stream=stream,
        dtype=dtype,
        device=device,
    )


def randomized_vector_batch(
    base: Sequence[float],
    config: Mapping[str, object],
    *,
    jitter_name: str,
    seed: int,
    stream: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return one independently jittered vector for every environment row.

    Skill-local jitter keys take precedence.  A track-level pose-batch
    translation range (injected as ``_pose_batch_translation_jitter_m`` during
    case generation) is used as a fallback, which keeps suite YAML concise.
    """
    base_values = [float(value) for value in base]
    configured = config.get(jitter_name)
    if configured is None and len(base_values) == 3:
        # Track-level ranges describe *pose* perturbations only.  Do not leak
        # an object-position range into grasp/held offsets, whose semantics
        # are local robot/object geometry and should remain deterministic
        # unless explicitly configured by the skill.
        if jitter_name in {"target_offset_jitter_m", "target_object_offset_jitter_m"}:
            configured = _pose_settings(config).target_translation_jitter_m
        elif jitter_name == "object_position_jitter_m":
            configured = _pose_settings(config).object_translation_jitter_m
    amplitude = float_vector(
        configured,
        name=jitter_name,
        length=len(base_values),
        default=(0.0,) * len(base_values),
    )
    if any(value < 0.0 for value in amplitude):
        raise ValueError(f"{jitter_name} values must be non-negative.")
    stride = int(_pose_settings(config).seed_stride)
    jitter = seeded_jitter(
        amplitude,
        seed=seed,
        stream=stream,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        seed_stride=stride,
        sampling_pattern=str(_pose_settings(config).sampling_pattern),
    )
    if jitter.dim() == 1:
        jitter = jitter.unsqueeze(0)
    base_tensor = torch.tensor(base_values, dtype=dtype, device=device).expand(
        batch_size, -1
    )
    randomized = base_tensor + jitter
    if jitter_name in {"target_offset_jitter_m", "target_object_offset_jitter_m"}:
        frame_rotation = task_frame_rotation(
            config,
            seed=seed,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        randomized = torch.bmm(frame_rotation, randomized[:, :, None]).squeeze(-1)
    return randomized


def apply_target_rotation_randomization(
    pose: torch.Tensor,
    config: Mapping[str, object],
    *,
    seed: int,
    stream: int,
) -> None:
    """Apply configured goal-relative roll, pitch, and yaw to a target pose."""
    amplitude = config.get("target_yaw_jitter_rad")
    if amplitude is None:
        amplitude = _pose_settings(config).target_yaw_jitter_rad
    roll_pitch_amplitude = config.get("target_roll_pitch_jitter_rad")
    if roll_pitch_amplitude is None:
        roll_pitch_amplitude = _pose_settings(config).target_roll_pitch_jitter_rad
    apply_seeded_spatial_rotation(
        pose,
        roll_pitch_amplitude=roll_pitch_amplitude,
        yaw_amplitude=amplitude,
        roll_pitch_name="target_roll_pitch_jitter_rad",
        yaw_name="target_yaw_jitter_rad",
        seed=seed,
        roll_pitch_stream=stream + 1,
        yaw_stream=stream,
        seed_stride=int(_pose_settings(config).seed_stride),
        sampling_pattern=str(_pose_settings(config).sampling_pattern),
    )


def randomization_parameters(
    config: Mapping[str, object], *, seed: int
) -> dict[str, object]:
    """Serialize the deterministic randomization contract into a case manifest."""
    ranges = {
        str(key): value
        for key, value in config.items()
        if (
            not str(key).startswith("_pose_batch_")
            and (str(key).endswith("_jitter_m") or str(key).endswith("_jitter_rad"))
        )
    }
    pose_batch_enabled = bool(_pose_settings(config).enabled)
    parameters: dict[str, object] = {
        "enabled": bool(ranges) or pose_batch_enabled,
        "seed": int(seed),
        "distribution": (
            "deterministic_stratified"
            if _pose_settings(config).sampling_pattern == "stratified"
            else (
                "deterministic_workspace_shell"
                if _pose_settings(config).sampling_pattern == "workspace_shell"
                else "independent_uniform"
            )
        ),
        "ranges": ranges,
    }
    if pose_batch_enabled:
        from dataclasses import asdict

        settings = _pose_settings(config)
        aliases = {
            "target_translation_jitter_m": "translation_jitter_m",
            "pose_batch_size": "size",
        }
        parameters["pose_batch"] = True
        parameters["batch_semantics"] = "one_pose_per_environment_row"
        for name, value in asdict(settings).items():
            if name == "enabled":
                continue
            parameters["pose_batch_" + aliases.get(name, name)] = (
                list(value) if isinstance(value, tuple) else value
            )
        if settings.sampling_pattern == "workspace_shell":
            radii, labels = workspace_shell_profile(settings.pose_batch_size)
            parameters.update(
                {
                    "spatial_bucket_axis": "normalized_xy_offset_radius",
                    "spatial_bucket_radii": radii.tolist(),
                    "spatial_buckets": labels,
                }
            )
    return parameters


def case_pose(case: BenchmarkCase, name: str, *, device: torch.device) -> torch.Tensor:
    """Restore one frozen pose tensor from JSON-compatible case parameters."""
    return torch.tensor(case.case_parameters[name], dtype=torch.float32, device=device)


def held_object_state(
    scenario: "AtomicSkillScenario", case: BenchmarkCase
) -> HeldObjectState:
    """Reconstruct the frozen held-object relation for an isolated case."""
    handle = scenario.object_handle(case.object_id)
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        properties={"benchmark_object_id": handle.object_id},
        label=handle.object_id,
        entity_id=handle.entity.uid,
    )
    return HeldObjectState(
        semantics=semantics,
        object_to_eef=case_pose(case, "object_to_eef", device=scenario.robot.device),
        grasp_xpos=case_pose(case, "grasp_pose", device=scenario.robot.device),
    )


def nearest_half_turn_pose(
    target_pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    """Match the half-turn grasp symmetry used by the Twist primitive."""
    symmetric = target_pose.clone()
    symmetric[:, :3, 0] = -symmetric[:, :3, 0]
    symmetric[:, :3, 1] = -symmetric[:, :3, 1]
    reference_rotation = reference_pose[:, :3, :3].transpose(-1, -2)
    target_trace = torch.diagonal(
        torch.bmm(reference_rotation, target_pose[:, :3, :3]),
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    symmetric_trace = torch.diagonal(
        torch.bmm(reference_rotation, symmetric[:, :3, :3]),
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    return torch.where(
        (target_trace > symmetric_trace)[:, None, None], target_pose, symmetric
    )


def twist_arc_poses(
    target_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    twist_axis: torch.Tensor,
    axis_origin: Sequence[float],
    twist_angle: float,
    waypoint_count: int,
) -> torch.Tensor:
    """Build the same target-local circular keyframes as the Twist primitive."""
    axis = torch.nn.functional.normalize(
        twist_axis.to(device=target_pose.device, dtype=torch.float32), dim=0
    )
    angles = torch.linspace(
        twist_angle / waypoint_count,
        twist_angle,
        waypoint_count,
        dtype=torch.float32,
        device=target_pose.device,
    )
    rotations = torch.eye(4, dtype=torch.float32, device=target_pose.device).repeat(
        waypoint_count, 1, 1
    )
    rotations[:, :3, :3] = axis_angle_to_rotation_matrix(angles[:, None] * axis)
    link_to_eef = torch.bmm(pose_inv(target_pose), grasp_pose)
    origin = torch.tensor(axis_origin, dtype=torch.float32, device=target_pose.device)
    to_origin = torch.eye(4, dtype=torch.float32, device=target_pose.device)
    from_origin = torch.eye(4, dtype=torch.float32, device=target_pose.device)
    to_origin[:3, 3] = origin
    from_origin[:3, 3] = -origin
    local_rotations = torch.matmul(
        torch.matmul(to_origin[None], rotations), from_origin[None]
    )
    return torch.matmul(
        torch.matmul(target_pose[:, None], local_rotations[None]),
        link_to_eef[:, None],
    )
