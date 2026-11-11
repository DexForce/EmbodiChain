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

from embodichain.lab.sim.atomic_skills import (
    Affordance,
    HeldObjectState,
    ObjectSemantics,
)
from embodichain.compute.geometry import rotate_rigid_grasp

from ..models import BenchmarkCase
from ._atomic_pose_sampling import (
    _pose_settings,
    task_frame_rotation,
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
    "case_generation_seed",
    "canonical_case_qpos",
    "snapshot_case_qpos",
    "randomized_vector_batch",
    "apply_target_rotation_randomization",
    "randomization_parameters",
    "case_pose",
    "held_object_state",
    "twist_arc_poses",
]


# Nearest proper rotation to the original four-decimal fixture. Retain full
# precision: rounding its entries makes independent rigid-pose IK validation fail.
TOP_DOWN_ROTATION = (
    (-0.053930418806384106, -0.9985423102658203, -0.002182781795413258),
    (-0.9977379548674098, 0.05397457111992954, -0.04007142484885853),
    (0.04013082785547817, 0.00001677552021006612, -0.9991944337185912),
)


TASK_DIFFICULTIES = {"simple", "medium", "hard"}


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
    translation range from the typed context is used as a fallback, which keeps suite YAML concise.
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


def twist_arc_poses(
    target_pose: torch.Tensor,
    grasp_pose: torch.Tensor,
    twist_axis: torch.Tensor,
    axis_origin: Sequence[float],
    twist_angle: float,
    waypoint_count: int,
) -> torch.Tensor:
    """Build the same target-local circular keyframes as the Twist primitive."""
    angles = torch.linspace(
        twist_angle / waypoint_count,
        twist_angle,
        waypoint_count,
        dtype=target_pose.dtype,
        device=target_pose.device,
    )
    return rotate_rigid_grasp(target_pose, grasp_pose, twist_axis, axis_origin, angles)[
        1
    ]
