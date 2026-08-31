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

"""Physics-backed Atomic Task track shared by planner adapters."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    Affordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
    EndEffectorPoseGoal,
    GraspGoal,
    HeldObjectPoseGoal,
    HeldObjectState,
    JointPositionGoal,
    MotionPolicy,
    MoveHeldObjectOptions,
    MoveJointsOptions,
    ObjectSemantics,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    PressAffordance,
    PressGoal,
    PressOptions,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
from embodichain.lab.sim.planners.utils import PlanResult

from ..config import SuiteCfg, TrackCfg
from ..metrics.trajectory import compute_case_outcomes
from ..models import BenchmarkCase, CaseOutcome
from ..registry import register_scenario_provider
from ..video import VideoRecordCfg, build_video_path, record_with_window
from .atomic_objects import AtomicObjectHandle, create_atomic_object
from .base import ScenarioEvaluation, ScenarioProvider

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot

    from ..planners.base import PlannerAdapter

__all__ = [
    "AtomicSkillCaseProvider",
    "AtomicTaskScenario",
    "atomic_skill_provider_names",
    "create_atomic_skill_provider",
    "register_atomic_skill_provider",
]

_TOP_DOWN_ROTATION = (
    (-0.0539, -0.9985, -0.0022),
    (-0.9977, 0.0540, -0.0401),
    (0.0401, 0.0000, -0.9992),
)
_TASK_DIFFICULTIES = {"simple", "medium", "hard"}
_GRIPPER_SKILLS = {"pick_up", "move_held_object", "place", "press"}
_CASE_QPOS_RESOLUTION_RAD = 1.0e-4


@dataclass(frozen=True)
class _ExecutionObservation:
    """Common-execution measurements used by skill-specific task rules."""

    execution_success: torch.Tensor
    final_tcp_pose: torch.Tensor
    joint_tracking_rmse_rad: torch.Tensor
    execution_time_ms: float
    task_completion_time_s: float
    object_lift_delta_m: torch.Tensor | None = None
    final_arm_qpos: torch.Tensor | None = None
    final_object_pose: torch.Tensor | None = None


@dataclass(frozen=True)
class _PhysicsReplaySettings:
    """Shared physics-step counts for timed evaluation and video replay."""

    steps_per_waypoint: int
    hold_steps: int
    hold_sim_steps: int
    joint_tracking_tolerance_rad: float


class AtomicSkillCaseProvider(ABC):
    """Generate and ground one Atomic Action without planner-specific logic."""

    skill_id: str
    requires_gripper = False

    @abstractmethod
    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        """Generate one frozen case and independent IK validity evidence."""

    @abstractmethod
    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        """Ground the case into one planner-independent action invocation."""

    def object_id(self, case: BenchmarkCase) -> str | None:
        """Return the manipulated object identifier when one exists."""
        return case.object_id

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Return the first lift waypoint that should release object dynamics."""
        return None

    def initial_task_state(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> TaskState | None:
        """Return an optional symbolic precondition for isolated action testing."""
        del scenario, case
        return None

    @abstractmethod
    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        """Return per-environment task success and its stable failure code."""


AtomicSkillProviderType = type[AtomicSkillCaseProvider]
_ATOMIC_SKILL_PROVIDERS: dict[str, AtomicSkillProviderType] = {}


def register_atomic_skill_provider(
    skill_id: str, provider_type: AtomicSkillProviderType
) -> None:
    """Register one Atomic Action case provider."""
    if not skill_id:
        raise ValueError("Atomic skill id must not be empty.")
    previous = _ATOMIC_SKILL_PROVIDERS.get(skill_id)
    if previous is not None and previous is not provider_type:
        raise ValueError(f"Atomic skill provider {skill_id!r} is already registered.")
    _ATOMIC_SKILL_PROVIDERS[skill_id] = provider_type


def atomic_skill_provider_names() -> tuple[str, ...]:
    """Return registered Atomic Action case providers."""
    return tuple(sorted(_ATOMIC_SKILL_PROVIDERS))


def create_atomic_skill_provider(skill_id: str) -> AtomicSkillCaseProvider:
    """Construct one registered Atomic Action case provider."""
    try:
        provider_type = _ATOMIC_SKILL_PROVIDERS[skill_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown atomic skill {skill_id!r}; registered skills: "
            f"{atomic_skill_provider_names()}."
        ) from exc
    return provider_type()


def _float_vector(
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


def _case_name(config: Mapping[str, object]) -> str:
    """Return and validate a stable case name."""
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Every atomic skill case must define a non-empty name.")
    return name


def _difficulty(config: Mapping[str, object]) -> str:
    """Resolve the explicit, frozen Atomic Task difficulty label."""
    difficulty = str(config.get("task_difficulty", "simple"))
    if difficulty not in _TASK_DIFFICULTIES:
        raise ValueError(
            f"task_difficulty must be one of {sorted(_TASK_DIFFICULTIES)}."
        )
    return difficulty


def _seeded_jitter(
    amplitude: Sequence[float],
    *,
    seed: int,
    stream: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sample deterministic independent uniform jitter in ``[-amplitude, +amplitude]``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) * 1_000_003 + int(stream) * 97_409) % (2**63 - 1))
    unit = torch.rand(len(amplitude), generator=generator, dtype=torch.float32)
    values = (2.0 * unit - 1.0) * torch.tensor(amplitude, dtype=torch.float32)
    return values.to(dtype=dtype, device=device)


def _case_generation_seed(seed: int, *, skill_index: int, case_index: int) -> int:
    """Derive one stable RNG seed for all stochastic IK work in a case."""
    return (
        int(seed) * 1_000_003 + int(skill_index) * 97_409 + int(case_index) * 13_007
    ) % (2**63 - 1)


def _canonical_case_qpos(qpos: torch.Tensor) -> torch.Tensor:
    """Remove insignificant device-level IK noise from frozen case states."""
    return torch.round(qpos / _CASE_QPOS_RESOLUTION_RAD) * _CASE_QPOS_RESOLUTION_RAD


def _randomized_vector(
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
    amplitude = _float_vector(
        config.get(jitter_name),
        name=jitter_name,
        length=len(base_values),
        default=(0.0,) * len(base_values),
    )
    if any(value < 0.0 for value in amplitude):
        raise ValueError(f"{jitter_name} values must be non-negative.")
    return torch.tensor(base_values, dtype=dtype, device=device) + _seeded_jitter(
        amplitude,
        seed=seed,
        stream=stream,
        dtype=dtype,
        device=device,
    )


def _randomization_parameters(
    config: Mapping[str, object], *, seed: int
) -> dict[str, object]:
    """Serialize the deterministic randomization contract into a case manifest."""
    ranges = {
        str(key): value
        for key, value in config.items()
        if str(key).endswith("_jitter_m") or str(key).endswith("_jitter_rad")
    }
    return {
        "enabled": bool(ranges),
        "seed": int(seed),
        "distribution": "independent_uniform",
        "ranges": ranges,
    }


def _motion_valid_mask(
    motion_outcomes: tuple[CaseOutcome, ...], *, device: torch.device
) -> torch.Tensor:
    """Return the common motion-valid result as one device-local mask."""
    return torch.tensor(
        [item.motion_valid for item in motion_outcomes],
        dtype=torch.bool,
        device=device,
    )


def _case_pose(case: BenchmarkCase, name: str, *, device: torch.device) -> torch.Tensor:
    """Restore one frozen pose tensor from JSON-compatible case parameters."""
    return torch.tensor(case.case_parameters[name], dtype=torch.float32, device=device)


def _held_object_state(
    scenario: "AtomicTaskScenario", case: BenchmarkCase
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
        object_to_eef=_case_pose(case, "object_to_eef", device=scenario.robot.device),
        grasp_xpos=_case_pose(case, "grasp_pose", device=scenario.robot.device),
    )


class _MoveEndEffectorCases(AtomicSkillCaseProvider):
    """Deterministic robot-relative MoveEndEffector cases."""

    skill_id = "move_end_effector"

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        scenario.restore_base_robot()
        scenario.randomize_robot_start(config, seed=seed, stream=11)
        raw_offsets = config.get("target_offsets_m")
        if not isinstance(raw_offsets, Sequence) or isinstance(
            raw_offsets, (str, bytes)
        ):
            raise TypeError("target_offsets_m must be a non-empty list of xyz vectors.")
        base_offsets = [
            _float_vector(value, name="target_offsets_m", length=3)
            for value in raw_offsets
        ]
        if not base_offsets:
            raise ValueError("target_offsets_m must not be empty.")

        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )
        offsets_tensor = torch.stack(
            [
                _randomized_vector(
                    offset,
                    config,
                    jitter_name="target_offset_jitter_m",
                    seed=seed,
                    stream=101 + index,
                    dtype=start_pose.dtype,
                    device=start_pose.device,
                )
                for index, offset in enumerate(base_offsets)
            ]
        )
        offsets = offsets_tensor.detach().cpu().tolist()
        targets = start_pose[:, None].repeat(1, len(offsets), 1, 1)
        targets[:, :, :3, 3] += offsets_tensor[None]
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=len(offsets),
            path_shape="robot_relative_waypoints",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "target_offsets_m": offsets,
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        return scenario.require_engine().make_invocation(
            self.skill_id,
            EndEffectorPoseGoal(case.target_waypoints),
            control_parts={"primary": {"motion": scenario.control_part}},
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
        )

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del compiled
        target = case.target_waypoints[:, -1]
        translation = torch.linalg.vector_norm(
            observation.final_tcp_pose[:, :3, 3] - target[:, :3, 3], dim=-1
        )
        relative = (
            target[:, :3, :3].transpose(-1, -2) @ observation.final_tcp_pose[:, :3, :3]
        )
        trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
        rotation = torch.arccos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
        motion_valid = torch.tensor(
            [item.motion_valid for item in motion_outcomes],
            dtype=torch.bool,
            device=translation.device,
        )
        success = (
            observation.execution_success
            & motion_valid
            & (translation <= scenario.suite.protocol.position_threshold_m)
            & (rotation <= scenario.suite.protocol.rotation_threshold_rad)
        )
        return success, "task_goal_miss"


class _PickUpCases(AtomicSkillCaseProvider):
    """Explicit-grasp PickUp cases that isolate motion-planner performance."""

    skill_id = "pick_up"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("PickUp cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()

        object_pose = scenario.randomize_object_pose(
            handle, config, seed=seed, stream=201
        )
        arm_start = scenario.robot.get_qpos(name=scenario.control_part)
        pre_pick_pose = scenario.robot.compute_fk(
            arm_start, name=scenario.control_part, to_matrix=True
        ).clone()
        pre_pick_pose[:, :2, 3] = object_pose[:, :2, 3]
        pre_pick_pose[:, 2, 3] = float(config.get("pre_pick_height_m", 0.36))
        success, pre_pick_qpos = scenario.robot.compute_ik(
            pose=pre_pick_pose,
            joint_seed=arm_start,
            name=scenario.control_part,
        )
        if not bool(torch.as_tensor(success).all().item()):
            raise RuntimeError(
                f"Independent IK rejected PickUp case {_case_name(config)!r}."
            )
        pre_pick_qpos = _canonical_case_qpos(pre_pick_qpos)
        scenario.set_robot_start(pre_pick_qpos, open_gripper=True)

        approach = torch.tensor(
            _float_vector(
                config.get("approach_direction"),
                name="approach_direction",
                length=3,
                default=(0.0, 0.0, -1.0),
            ),
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        approach_norm = torch.linalg.vector_norm(approach)
        if float(approach_norm.item()) <= 1.0e-6:
            raise ValueError("approach_direction must be non-zero.")
        approach = approach / approach_norm
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        lift_height = float(config.get("lift_height_m", 0.16))

        grasp_source = str(config.get("grasp_source", "fixed"))
        if grasp_source == "antipodal":
            grasp_pose = scenario.resolve_antipodal_grasp(
                handle,
                object_pose,
                approach,
                seed=seed,
                start_qpos=scenario.robot.get_qpos(name=scenario.control_part),
                pre_grasp_distance=pre_grasp_distance,
                lift_height=lift_height,
                n_sample=int(config.get("grasp_sample_count", 10_000)),
                max_candidates=int(config.get("grasp_max_candidates", 128)),
                alignment_max_angle_deg=float(
                    config.get("grasp_alignment_max_angle_deg", 10.0)
                ),
            )
        elif grasp_source == "fixed":
            grasp_pose = object_pose.clone()
            rotation_value = config.get("grasp_rotation", _TOP_DOWN_ROTATION)
            if not isinstance(rotation_value, Sequence) or len(rotation_value) != 3:
                raise ValueError("grasp_rotation must be a 3x3 matrix.")
            rotation = torch.tensor(
                rotation_value, dtype=grasp_pose.dtype, device=grasp_pose.device
            )
            if rotation.shape != (3, 3):
                raise ValueError("grasp_rotation must be a 3x3 matrix.")
            grasp_pose[:, :3, :3] = rotation
        else:
            raise ValueError("grasp_source must be 'fixed' or 'antipodal'.")
        grasp_offset = _randomized_vector(
            _float_vector(
                config.get("grasp_offset_m"),
                name="grasp_offset_m",
                length=3,
                default=(0.0, 0.0, 0.0),
            ),
            config,
            jitter_name="grasp_offset_jitter_m",
            seed=seed,
            stream=202,
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        grasp_pose[:, :3, 3] += grasp_offset
        pre_grasp = grasp_pose.clone()
        pre_grasp[:, :3, 3] -= approach * pre_grasp_distance
        lift = grasp_pose.clone()
        lift[:, 2, 3] += lift_height
        targets = torch.stack([pre_grasp, grasp_pose, lift], dim=1)
        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=3,
            path_shape="approach_grasp_lift",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                "sample_count": int(config.get("sample_count", 120)),
                "grasp_source": grasp_source,
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "approach_direction": approach.detach().cpu().tolist(),
                "pre_grasp_distance_m": pre_grasp_distance,
                "lift_height_m": lift_height,
                "minimum_object_lift_m": float(
                    config.get("minimum_object_lift_m", 0.04)
                ),
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "object_config": dict(handle.config),
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        handle = scenario.object_handle(case.object_id)
        if scenario.end_effector_part is None:
            raise RuntimeError(
                "PickUp requires a configured end-effector control part."
            )
        semantics = ObjectSemantics(
            affordance=Affordance(),
            geometry={},
            properties={"benchmark_object_id": handle.object_id},
            label=handle.object_id,
            entity_id=handle.entity.uid,
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            GraspGoal(
                semantics=semantics,
                grasp_xpos=case.target_waypoints[:, 1],
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=PickUpOptions(
                approach_direction=torch.tensor(
                    case.case_parameters["approach_direction"],
                    dtype=torch.float32,
                    device=scenario.robot.device,
                ),
                pre_grasp_distance=float(case.case_parameters["pre_grasp_distance_m"]),
                lift_height=float(case.case_parameters["lift_height_m"]),
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
            ),
        )

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Return the lift boundary emitted by PickUp."""
        return compiled.segment(0, "lift").start

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        held_created = (
            compiled.projected_context.get_held_object(scenario.control_part)
            is not None
        )
        lift = observation.object_lift_delta_m
        if lift is None:
            lift = torch.full(
                (case.batch_size,),
                -torch.inf,
                device=observation.execution_success.device,
            )
        motion_valid = torch.tensor(
            [item.motion_valid for item in motion_outcomes],
            dtype=torch.bool,
            device=lift.device,
        )
        success = (
            observation.execution_success
            & motion_valid
            & held_created
            & (lift >= float(case.case_parameters["minimum_object_lift_m"]))
        )
        return success, "object_not_grasped"


class _MoveJointsCases(AtomicSkillCaseProvider):
    """Deterministic relative joint-space waypoint cases."""

    skill_id = "move_joints"

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        scenario.restore_base_robot()
        scenario.randomize_robot_start(config, seed=seed, stream=21)
        raw_offsets = config.get("target_offsets_rad")
        if not isinstance(raw_offsets, Sequence) or isinstance(
            raw_offsets, (str, bytes)
        ):
            raise TypeError("target_offsets_rad must be a non-empty list.")
        base_offsets = [
            _float_vector(value, name="target_offsets_rad", length=7)
            for value in raw_offsets
        ]
        if not base_offsets:
            raise ValueError("target_offsets_rad must not be empty.")
        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        offset_tensor = torch.stack(
            [
                _randomized_vector(
                    offset,
                    config,
                    jitter_name="target_offset_jitter_rad",
                    seed=seed,
                    stream=211 + index,
                    dtype=start_qpos.dtype,
                    device=start_qpos.device,
                )
                for index, offset in enumerate(base_offsets)
            ]
        )
        offsets = offset_tensor.detach().cpu().tolist()
        targets = start_qpos[:, None] + torch.cumsum(offset_tensor, dim=0)[None]
        limits = scenario.robot.get_qpos_limits(name=scenario.control_part)[0]
        margin = float(config.get("joint_limit_margin_rad", 0.05))
        if bool(
            (
                (targets < limits[:, 0][None, None] + margin)
                | (targets > limits[:, 1][None, None] - margin)
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                f"Joint limits rejected MoveJoints case {_case_name(config)!r}."
            )
        target_poses = torch.stack(
            [
                scenario.robot.compute_fk(
                    targets[:, index], name=scenario.control_part, to_matrix=True
                )
                for index in range(targets.shape[1])
            ],
            dim=1,
        )
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=len(offsets),
            path_shape="relative_joint_waypoints",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=target_poses,
            reference_qpos=targets,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "target_offsets_rad": offsets,
                "target_qpos": targets.detach().cpu().tolist(),
                "joint_threshold_rad": float(config.get("joint_threshold_rad", 0.02)),
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        target = torch.tensor(
            case.case_parameters["target_qpos"],
            dtype=torch.float32,
            device=scenario.robot.device,
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            JointPositionGoal(target),
            control_parts={"primary": {"motion": scenario.control_part}},
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=MoveJointsOptions(),
        )

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, compiled
        if observation.final_arm_qpos is None:
            return torch.zeros_like(observation.execution_success), "joint_goal_miss"
        target = torch.tensor(
            case.case_parameters["target_qpos"],
            dtype=observation.final_arm_qpos.dtype,
            device=observation.final_arm_qpos.device,
        )[:, -1]
        error = torch.amax(torch.abs(observation.final_arm_qpos - target), dim=-1)
        success = (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=error.device)
            & (error <= float(case.case_parameters["joint_threshold_rad"]))
        )
        return success, "joint_goal_miss"


class _HeldObjectCases(AtomicSkillCaseProvider):
    """Shared isolated held-object precondition for transport and placement."""

    requires_gripper = True

    def _prepare_start(
        self,
        scenario: "AtomicTaskScenario",
        config: Mapping[str, object],
        *,
        seed: int,
    ) -> tuple[
        AtomicObjectHandle,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{self.skill_id} cases require an object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()
        table_object_pose = scenario.randomize_object_pose(
            handle, config, seed=seed, stream=301
        )
        object_pose = table_object_pose.clone()
        held_offset = _randomized_vector(
            _float_vector(
                config.get("held_object_offset_m"),
                name="held_object_offset_m",
                length=3,
                default=(0.0, 0.0, 0.18),
            ),
            config,
            jitter_name="held_object_offset_jitter_m",
            seed=seed,
            stream=302,
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        object_pose[:, :3, 3] += held_offset
        grasp_pose = object_pose.clone()
        grasp_pose[:, :3, :3] = torch.tensor(
            config.get("grasp_rotation", _TOP_DOWN_ROTATION),
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        grasp_pose[:, :3, 3] += _randomized_vector(
            _float_vector(
                config.get("grasp_offset_m"),
                name="grasp_offset_m",
                length=3,
                default=(0.0, 0.0, 0.0),
            ),
            config,
            jitter_name="grasp_offset_jitter_m",
            seed=seed,
            stream=303,
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        arm_seed = scenario.robot.get_qpos(name=scenario.control_part)
        success, arm_start = scenario.robot.compute_ik(
            pose=grasp_pose,
            joint_seed=arm_seed,
            name=scenario.control_part,
        )
        if not bool(torch.as_tensor(success).all().item()):
            raise RuntimeError(
                f"Independent IK rejected held-object start {_case_name(config)!r}."
            )
        arm_start = _canonical_case_qpos(arm_start)
        scenario.set_robot_start(arm_start, open_gripper=False, grasp_gripper=True)
        handle.entity.set_local_pose(object_pose)
        handle.entity.clear_dynamics()
        if scenario.simulation is not None:
            scenario.simulation.update(step=2)
            handle.entity.set_local_pose(object_pose)
            handle.entity.clear_dynamics()
            scenario.set_robot_start(
                arm_start,
                open_gripper=False,
                grasp_gripper=True,
            )
        object_to_eef = torch.bmm(torch.linalg.inv(object_pose), grasp_pose)
        return handle, object_pose, grasp_pose, object_to_eef, table_object_pose

    def initial_task_state(
        self, scenario: "AtomicTaskScenario", case: BenchmarkCase
    ) -> TaskState:
        return TaskState(
            batch_size=case.batch_size,
            device=scenario.robot.device,
            held_objects={scenario.control_part: _held_object_state(scenario, case)},
        )

    @staticmethod
    def _held_parameters(
        handle: AtomicObjectHandle,
        object_pose: torch.Tensor,
        grasp_pose: torch.Tensor,
        object_to_eef: torch.Tensor,
    ) -> dict[str, object]:
        return {
            "object_initial_pose": object_pose.detach().cpu().tolist(),
            "grasp_pose": grasp_pose.detach().cpu().tolist(),
            "object_to_eef": object_to_eef.detach().cpu().tolist(),
            "object_config": dict(handle.config),
        }


class _MoveHeldObjectCases(_HeldObjectCases):
    """Move an already held rigid object to one frozen target pose."""

    skill_id = "move_held_object"

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        (
            handle,
            object_pose,
            grasp_pose,
            object_to_eef,
            _table_object_pose,
        ) = self._prepare_start(scenario, config, seed=seed)
        target_object = object_pose.clone()
        target_object[:, :3, 3] += _randomized_vector(
            _float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=(0.08, 0.08, 0.04),
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=311,
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        target_eef = torch.bmm(target_object, object_to_eef)
        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        references = scenario.solve_reference_qpos(start_qpos, target_eef[:, None])
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=1,
            path_shape="held_object_transport",
            start_state_bin="object_held",
            start_qpos=start_qpos,
            target_waypoints=target_eef[:, None],
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                **self._held_parameters(handle, object_pose, grasp_pose, object_to_eef),
                "sample_count": int(config.get("sample_count", 80)),
                "target_object_pose": target_object.detach().cpu().tolist(),
                "object_position_threshold_m": float(
                    config.get("object_position_threshold_m", 0.04)
                ),
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        return scenario.require_engine().make_invocation(
            self.skill_id,
            HeldObjectPoseGoal(
                _case_pose(case, "target_object_pose", device=scenario.robot.device)
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=MoveHeldObjectOptions(pick_rotate_upright=0.0),
        )

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        final_object = observation.final_object_pose
        if final_object is None:
            return torch.zeros_like(observation.execution_success), "object_goal_miss"
        target = _case_pose(case, "target_object_pose", device=final_object.device)
        error = torch.linalg.vector_norm(
            final_object[:, :3, 3] - target[:, :3, 3], dim=-1
        )
        held = compiled.projected_context.get_held_object(scenario.control_part)
        success = (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=error.device)
            & (held is not None)
            & (error <= float(case.case_parameters["object_position_threshold_m"]))
        )
        return success, "object_goal_miss"


class _PlaceCases(_HeldObjectCases):
    """Place an already held rigid object and release it under physics."""

    skill_id = "place"

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        (
            handle,
            object_pose,
            grasp_pose,
            object_to_eef,
            table_object_pose,
        ) = self._prepare_start(scenario, config, seed=seed)
        target_object = table_object_pose.clone()
        target_object[:, :3, 3] += _randomized_vector(
            _float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=(0.10, 0.10, 0.0),
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=321,
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        release = torch.bmm(target_object, object_to_eef)
        lift_height = float(config.get("retract_height_m", 0.10))
        approach = release.clone()
        approach[:, 2, 3] += lift_height
        retract = approach.clone()
        targets = torch.stack([approach, release, retract], dim=1)
        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=3,
            path_shape="approach_release_retract",
            start_state_bin="object_held",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                **self._held_parameters(handle, object_pose, grasp_pose, object_to_eef),
                "sample_count": int(config.get("sample_count", 120)),
                "release_pose": release.detach().cpu().tolist(),
                "target_object_pose": target_object.detach().cpu().tolist(),
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "retract_height_m": lift_height,
                "object_position_threshold_m": float(
                    config.get("object_position_threshold_m", 0.05)
                ),
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        return scenario.require_engine().make_invocation(
            self.skill_id,
            PlaceGoal(_case_pose(case, "release_pose", device=scenario.robot.device)),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=PlaceOptions(
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                lift_height=float(case.case_parameters["retract_height_m"]),
            ),
        )

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        final_object = observation.final_object_pose
        if final_object is None:
            return torch.zeros_like(observation.execution_success), "object_not_placed"
        target = _case_pose(case, "target_object_pose", device=final_object.device)
        error = torch.linalg.vector_norm(
            final_object[:, :3, 3] - target[:, :3, 3], dim=-1
        )
        released = (
            compiled.projected_context.get_held_object(scenario.control_part) is None
        )
        success = (
            observation.execution_success
            & _motion_valid_mask(motion_outcomes, device=error.device)
            & released
            & (error <= float(case.case_parameters["object_position_threshold_m"]))
        )
        return success, "object_not_placed"


class _PressCases(AtomicSkillCaseProvider):
    """Close the gripper, reach a contact pose, and retract."""

    skill_id = "press"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicTaskScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        scenario.restore_base_robot()
        scenario.randomize_robot_start(config, seed=seed, stream=31)
        start_qpos = scenario.robot.get_qpos(name=scenario.control_part).clone()
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )
        target_pose = start_pose.clone()
        target_pose[:, :3, 3] += _randomized_vector(
            _float_vector(
                config.get("target_offset_m"),
                name="target_offset_m",
                length=3,
                default=(0.0, 0.0, -0.08),
            ),
            config,
            jitter_name="target_offset_jitter_m",
            seed=seed,
            stream=331,
            dtype=start_pose.dtype,
            device=start_pose.device,
        )
        press_axis = _float_vector(
            config.get("press_axis"),
            name="press_axis",
            length=3,
            default=(0.0, 0.0, 1.0),
        )
        press_position = _float_vector(
            config.get("press_position"),
            name="press_position",
            length=3,
            default=(0.0, 0.0, 0.0),
        )
        options = PressOptions(
            hand_interp_steps=int(config.get("hand_interp_steps", 8)),
            approach_distance=float(config.get("approach_distance_m", 0.1)),
            press_distance=float(config.get("press_distance_m", 0.05)),
            press_position=tuple(press_position),
        )
        affordance = PressAffordance(
            press_axis=torch.tensor(
                press_axis,
                dtype=target_pose.dtype,
                device=target_pose.device,
            ),
            press_position=tuple(press_position),
        )
        contact_pose = affordance.get_press_pose(
            target_pose,
            press_position=options.press_position,
        )
        approach_pose = contact_pose.clone()
        approach_pose[:, :3, 3] -= contact_pose[:, :3, 2] * options.approach_distance
        pressed_pose = contact_pose.clone()
        pressed_pose[:, :3, 3] += contact_pose[:, :3, 2] * options.press_distance
        targets = torch.stack(
            [approach_pose, contact_pose, pressed_pose, approach_pose],
            dim=1,
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = _case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=f"{track.id}:{self.skill_id}:{name}:s{seed}",
            seed=seed,
            batch_size=batch_size,
            num_waypoints=4,
            path_shape="approach_contact_press_retract",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            task_difficulty=_difficulty(config),
            primary_success="task_success",
            full_start_qpos=_canonical_case_qpos(scenario.robot.get_qpos().clone()),
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "press_target_pose": target_pose.detach().cpu().tolist(),
                "press_axis": press_axis,
                "press_position": press_position,
                "hand_interp_steps": options.hand_interp_steps,
                "approach_distance_m": options.approach_distance,
                "press_distance_m": options.press_distance,
                "randomization": _randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        press_axis = torch.tensor(
            case.case_parameters["press_axis"],
            dtype=torch.float32,
            device=scenario.robot.device,
        )
        press_position = tuple(case.case_parameters["press_position"])
        semantics = ObjectSemantics(
            affordance=PressAffordance(
                press_axis=press_axis,
                press_position=press_position,
            ),
            geometry={},
            entity_id=f"benchmark.press_target.{case.case_id}",
            label="benchmark_press_target",
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            PressGoal(
                semantics,
                _case_pose(
                    case,
                    "press_target_pose",
                    device=scenario.robot.device,
                ),
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                sample_count=int(case.case_parameters["sample_count"]),
            ),
            skill_options=PressOptions(
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                approach_distance=float(case.case_parameters["approach_distance_m"]),
                press_distance=float(case.case_parameters["press_distance_m"]),
                press_position=press_position,
            ),
        )

    def task_result(
        self,
        scenario: "AtomicTaskScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: _ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, case, compiled
        success = observation.execution_success & _motion_valid_mask(
            motion_outcomes, device=observation.execution_success.device
        )
        return success, "press_pose_miss"


class AtomicTaskScenario(ScenarioProvider):
    """Run fixed Atomic Actions through an adapter-owned MotionGenerator."""

    required_capabilities = frozenset(
        {"eef_waypoint", "joint_waypoint", "atomic_action"}
    )

    def __init__(self) -> None:
        self.simulation: SimulationManager | None = None
        self.robot: Robot | None = None
        self.suite: SuiteCfg | None = None
        self.track: TrackCfg | None = None
        self.control_part = "arm"
        self.end_effector_part: str | None = None
        self._objects: dict[str, AtomicObjectHandle] = {}
        self._case_providers: dict[str, AtomicSkillCaseProvider] = {}
        self._base_full_qpos: torch.Tensor | None = None
        self._gripper_open: torch.Tensor | None = None
        self._gripper_grasp: torch.Tensor | None = None
        self._engine: AtomicActionEngine | None = None

    def batch_sizes(self, suite: SuiteCfg, track: TrackCfg) -> list[int]:
        """Return the explicitly configured physical-execution batch sizes."""
        del suite
        values = [int(value) for value in track.config.get("batch_sizes", [1])]
        if values != [1]:
            raise ValueError(
                "The initial Atomic Task implementation supports batch_sizes: [1] only."
            )
        return values

    def configure_runtime(
        self,
        simulation: SimulationManager,
        robot: Robot,
        suite: SuiteCfg,
        track: TrackCfg,
        control_part: str,
    ) -> None:
        """Create the declarative object pool and cache robot command states."""
        self.simulation = simulation
        self.robot = robot
        self.suite = suite
        self.track = track
        self.control_part = control_part
        self._base_full_qpos = robot.get_qpos().clone()

        object_values = track.config.get("objects", [])
        if not isinstance(object_values, list):
            raise TypeError("atomic-task objects must be a list of mappings.")
        for value in object_values:
            if not isinstance(value, Mapping):
                raise TypeError("Every atomic-task object must be a mapping.")
            handle = create_atomic_object(simulation, value)
            if handle.object_id in self._objects:
                raise ValueError(f"Duplicate atomic object id {handle.object_id!r}.")
            self._objects[handle.object_id] = handle
        for index, handle in enumerate(self._objects.values()):
            handle.park(index)
        simulation.update(step=1)

        if any(
            isinstance(value, Mapping) and value.get("id") in _GRIPPER_SKILLS
            for value in track.config.get("skills", [])
        ):
            gripper_value = track.config.get("gripper")
            if not isinstance(gripper_value, Mapping):
                raise ValueError(
                    "Atomic gripper-action tracks must define a gripper mapping."
                )
            end_effector_part = gripper_value.get("control_part")
            if not isinstance(end_effector_part, str) or not end_effector_part:
                raise ValueError("gripper.control_part must be a non-empty string.")
            self.end_effector_part = end_effector_part
            limits = robot.get_qpos_limits(name=end_effector_part)[0].to(
                device=robot.device, dtype=torch.float32
            )
            dofs = limits.shape[0]
            open_qpos = _float_vector(
                gripper_value.get("open_qpos"),
                name="gripper.open_qpos",
                length=dofs,
                default=limits[:, 0].detach().cpu().tolist(),
            )
            grasp_qpos = _float_vector(
                gripper_value.get("grasp_qpos"),
                name="gripper.grasp_qpos",
                length=dofs,
            )
            self._gripper_open = torch.tensor(
                open_qpos, dtype=limits.dtype, device=limits.device
            )
            self._gripper_grasp = torch.tensor(
                grasp_qpos, dtype=limits.dtype, device=limits.device
            )
            if bool(
                (
                    (self._gripper_open < limits[:, 0])
                    | (self._gripper_open > limits[:, 1])
                    | (self._gripper_grasp < limits[:, 0])
                    | (self._gripper_grasp > limits[:, 1])
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "gripper open/grasp qpos must lie within joint limits."
                )

    def generate_cases(
        self,
        suite: SuiteCfg,
        track: TrackCfg,
        robot: Robot,
        control_part: str,
        batch_size: int,
    ) -> list[BenchmarkCase]:
        """Generate the algorithm-independent Atomic Task case manifest."""
        if robot is not self.robot or control_part != self.control_part:
            raise RuntimeError(
                "AtomicTaskScenario must be configured before generation."
            )
        skill_values = track.config.get("skills", [])
        if not isinstance(skill_values, list) or not skill_values:
            raise ValueError("atomic-task skills must be a non-empty list.")
        seeds = [int(value) for value in track.config.get("seeds", [11])]
        if not seeds:
            raise ValueError("atomic-task seeds must not be empty.")

        cases: list[BenchmarkCase] = []
        for skill_index, skill_value in enumerate(skill_values):
            if not isinstance(skill_value, Mapping):
                raise TypeError("Every atomic-task skill entry must be a mapping.")
            skill_id = skill_value.get("id")
            if not isinstance(skill_id, str) or not skill_id:
                raise ValueError("Every atomic-task skill entry must define an id.")
            raw_cases = skill_value.get("cases", [])
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError(f"Atomic skill {skill_id!r} needs at least one case.")
            defaults = {
                key: value
                for key, value in skill_value.items()
                if key not in {"id", "cases"}
            }
            for case_index, raw_case in enumerate(raw_cases):
                if not isinstance(raw_case, Mapping):
                    raise TypeError("Every atomic skill case must be a mapping.")
                config = {**defaults, **dict(raw_case)}
                for seed in seeds:
                    provider = create_atomic_skill_provider(skill_id)
                    rng_devices = (
                        [robot.device]
                        if torch.device(robot.device).type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(
                            _case_generation_seed(
                                seed,
                                skill_index=skill_index,
                                case_index=case_index,
                            )
                        )
                        case = provider.generate_case(
                            self,
                            suite,
                            track,
                            config,
                            seed=seed,
                            batch_size=batch_size,
                        )
                    if case.case_id in self._case_providers:
                        raise ValueError(
                            f"Duplicate Atomic Task case {case.case_id!r}."
                        )
                    self._case_providers[case.case_id] = provider
                    cases.append(case)
        self.restore_base_robot()
        for index, handle in enumerate(self._objects.values()):
            handle.park(index)
        return cases

    def prepare_planner(
        self, adapter: PlannerAdapter, first_case: BenchmarkCase
    ) -> None:
        """Bind one AtomicActionEngine to the adapter-owned MotionGenerator."""
        del first_case
        control_profiles = None
        if any(provider.requires_gripper for provider in self._case_providers.values()):
            if (
                self.end_effector_part is None
                or self._gripper_open is None
                or self._gripper_grasp is None
            ):
                raise RuntimeError("Configured gripper command states are unavailable.")
            control_profiles = {
                self.end_effector_part: ControlPartCommandProfile.joint_positions(
                    open=self._gripper_open,
                    grasp=self._gripper_grasp,
                )
            }
        self._engine = AtomicActionEngine(
            motion_generator=adapter.require_motion_generator(),
            control_profiles=control_profiles,
        )

    def close_planner(self, adapter: PlannerAdapter) -> None:
        """Drop the engine before its adapter closes the shared generator."""
        del adapter
        self._engine = None

    def require_engine(self) -> AtomicActionEngine:
        """Return the prepared Atomic Action engine."""
        if self._engine is None:
            raise RuntimeError("Atomic Task planner resources were not prepared.")
        return self._engine

    def reset_case(
        self,
        simulation: SimulationManager,
        robot: Robot,
        case: BenchmarkCase,
        control_part: str,
    ) -> None:
        """Restore full robot and object state before every planner call."""
        del control_part
        if case.full_start_qpos is None:
            raise ValueError("Atomic Task cases require full_start_qpos.")
        for target in (False, True):
            robot.set_qpos(case.full_start_qpos, target=target)
        robot.clear_dynamics()
        active_id = case.object_id
        for index, handle in enumerate(self._objects.values()):
            if handle.object_id == active_id:
                initial_pose = case.case_parameters.get("object_initial_pose")
                if initial_pose is None:
                    handle.reset()
                else:
                    handle.entity.set_local_pose(
                        torch.tensor(
                            initial_pose,
                            dtype=torch.float32,
                            device=robot.device,
                        )
                    )
                    handle.entity.clear_dynamics()
            else:
                handle.park(index)
        simulation.update(step=2)

    def plan_case(self, adapter: PlannerAdapter, case: BenchmarkCase) -> object:
        """Compile one Atomic Action with an explicitly pinned motion backend."""
        engine = self.require_engine()
        if self.simulation is None:
            raise RuntimeError("Atomic Task simulation is not configured.")
        provider = self._case_providers[case.case_id]
        invocation = provider.build_invocation(self, case, adapter)
        policy = invocation.motion_policy
        if policy.strategy != "motion_gen":
            raise RuntimeError(
                "Atomic Task invocations must use the motion_gen strategy."
            )
        if engine.motion_generator is not adapter.require_motion_generator():
            raise RuntimeError(
                "Atomic Task engine must own the selected adapter's MotionGenerator."
            )
        task = provider.initial_task_state(self, case)
        context = engine.initial_context(
            task=task,
            control_dt=float(self.simulation.sim_config.physics_dt),
        )
        return engine.compile((invocation,), context=context)

    def plan_contract_error(self, result: object) -> str | None:
        """Accept compiled Atomic Action trajectories instead of raw plans."""
        if isinstance(result, CompiledTrajectory):
            return None
        return f"Expected CompiledTrajectory, got {type(result).__name__}."

    def failure_outcomes(
        self, case: BenchmarkCase, failure_code: str
    ) -> tuple[CaseOutcome, ...]:
        """Mark execution/task stages false after runner-level failures."""
        return tuple(
            replace(
                outcome,
                execution_success=False,
                task_success=False,
                replan_count=0,
            )
            for outcome in super().failure_outcomes(case, failure_code)
        )

    def evaluate_case(
        self,
        result: object,
        case: BenchmarkCase,
        robot: Robot,
        control_part: str,
        suite: SuiteCfg,
        *,
        planning_time_ms: float,
    ) -> ScenarioEvaluation:
        """Validate motion, replay physics, and evaluate physical task success."""
        if not isinstance(result, CompiledTrajectory):
            raise TypeError(self.plan_contract_error(result))
        arm_joint_ids = list(robot.get_joint_ids(name=control_part))
        arm_positions = result.trajectory.positions[:, :, arm_joint_ids]
        motion_outcomes = compute_case_outcomes(
            PlanResult(
                success=result.plan_success,
                positions=arm_positions,
                dt=result.trajectory.dt,
            ),
            case,
            robot,
            control_part,
            validation_samples=suite.protocol.validation_samples,
            position_threshold_m=suite.protocol.position_threshold_m,
            rotation_threshold_rad=suite.protocol.rotation_threshold_rad,
            joint_limit_tolerance_rad=suite.protocol.joint_limit_tolerance_rad,
        )
        provider = self._case_providers[case.case_id]
        observation = self._execute(result, case, provider)
        if observation is None:
            execution_success = torch.zeros(
                case.batch_size, dtype=torch.bool, device=robot.device
            )
            task_success = execution_success.clone()
            execution_time_ms = None
            tracking = torch.full((case.batch_size,), torch.nan, device=robot.device)
            object_lift = None
            task_failure_code = "task_goal_miss"
        else:
            execution_success = observation.execution_success
            task_success, task_failure_code = provider.task_result(
                self, case, result, observation, motion_outcomes
            )
            execution_time_ms = observation.execution_time_ms
            tracking = observation.joint_tracking_rmse_rad
            object_lift = observation.object_lift_delta_m

        executed_translation_mm: torch.Tensor | None = None
        executed_rotation_deg: torch.Tensor | None = None
        if observation is not None:
            final_target = case.target_waypoints[:, -1]
            executed_translation_mm = (
                torch.linalg.vector_norm(
                    observation.final_tcp_pose[:, :3, 3] - final_target[:, :3, 3],
                    dim=-1,
                )
                * 1000.0
            )
            executed_relative = (
                final_target[:, :3, :3].transpose(-1, -2)
                @ observation.final_tcp_pose[:, :3, :3]
            )
            executed_trace = torch.diagonal(executed_relative, dim1=-2, dim2=-1).sum(
                dim=-1
            )
            executed_rotation_deg = (
                torch.arccos(torch.clamp((executed_trace - 1.0) * 0.5, -1.0, 1.0))
                * 180.0
                / math.pi
            )

        durations = result.trajectory.duration.detach().to("cpu")
        outcomes: list[CaseOutcome] = []
        for index, outcome in enumerate(motion_outcomes):
            executed = bool(execution_success[index].item())
            task_done = bool(task_success[index].item())
            if outcome.failure_code is not None:
                failure_code = outcome.failure_code
            elif not executed:
                failure_code = "controller_tracking_failure"
            elif not task_done:
                failure_code = task_failure_code
            else:
                failure_code = None
            tracking_value = float(tracking[index].item())
            outcomes.append(
                replace(
                    outcome,
                    execution_success=executed,
                    task_success=task_done,
                    task_completion_time_s=(
                        observation.task_completion_time_s
                        if observation is not None and task_done
                        else None
                    ),
                    joint_tracking_rmse_rad=(
                        tracking_value if math.isfinite(tracking_value) else None
                    ),
                    object_lift_delta_m=(
                        None
                        if object_lift is None
                        else float(object_lift[index].item())
                    ),
                    replan_count=0,
                    failure_code=failure_code,
                    executed_final_translation_err_mm=(
                        None
                        if executed_translation_mm is None
                        else float(executed_translation_mm[index].item())
                    ),
                    executed_final_rotation_err_deg=(
                        None
                        if executed_rotation_deg is None
                        else float(executed_rotation_deg[index].item())
                    ),
                )
            )
        trajectory_duration = (
            float(durations.mean().item()) if durations.numel() else None
        )
        return ScenarioEvaluation(
            outcomes=tuple(outcomes),
            execution_time_ms=execution_time_ms,
            end_to_end_time_ms=(
                planning_time_ms + execution_time_ms
                if execution_time_ms is not None
                else None
            ),
            trajectory_duration_s=trajectory_duration,
            trajectory_waypoints=result.trajectory.waypoint_count,
            metadata={
                "timing_scope": "atomic_action_compile",
                "motion_policy_strategy": "motion_gen",
                "physics_validation": "common_joint_target_replay",
                "constraint_information": (
                    "empty external world; manipulated target excluded from "
                    "cuRobo collision obstacles"
                ),
            },
        )

    def record_replay(
        self,
        result: object,
        case: BenchmarkCase,
        evaluation: ScenarioEvaluation | None,
        *,
        output_dir: Path,
        algorithm_id: str,
        video: VideoRecordCfg,
    ) -> Path | None:
        """Reset the case and record a second untimed physics replay."""
        del evaluation
        if self.simulation is None or self.robot is None:
            return None
        self.reset_case(self.simulation, self.robot, case, self.control_part)
        compiled = result if isinstance(result, CompiledTrajectory) else None
        replayable = compiled is not None and self._is_recordable(compiled)
        provider = self._case_providers.get(case.case_id)
        video_path = build_video_path(
            output_dir, algorithm_id, case.skill_id, case.case_id
        )

        def _replay() -> None:
            if replayable and compiled is not None:
                self._replay_physics(compiled, case, provider, collect_metrics=False)
            else:
                self._hold_static()

        return record_with_window(self.simulation, video, video_path, _replay)

    def _execute(
        self,
        compiled: CompiledTrajectory,
        case: BenchmarkCase,
        provider: AtomicSkillCaseProvider,
    ) -> _ExecutionObservation | None:
        """Replay a successful full-robot trajectory under common physics."""
        if self.simulation is None or self.robot is None or self.track is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        if not self._is_replayable(compiled):
            return None
        object_handle = (
            None if case.object_id is None else self.object_handle(case.object_id)
        )
        initial_object_position = (
            None
            if object_handle is None
            else object_handle.entity.get_local_pose(to_matrix=True)[:, :3, 3].clone()
        )
        settings = self._physics_settings()
        self._synchronize()
        started = time.perf_counter()
        tracking_parts = self._replay_physics(
            compiled, case, provider, collect_metrics=True
        )
        self._synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if tracking_parts is None:
            raise RuntimeError("Timed Atomic Task replay did not return metrics.")
        squared_tracking_error, tracking_value_count = tracking_parts
        observed = self.robot.get_qpos()
        tracking = torch.sqrt(squared_tracking_error / max(tracking_value_count, 1))
        trajectory = compiled.trajectory.positions
        simulated_execution_time = (
            trajectory.shape[1] * settings.steps_per_waypoint
            + settings.hold_steps * settings.hold_sim_steps
        ) * float(self.simulation.sim_config.physics_dt)
        final_tcp = self.robot.compute_fk(
            self.robot.get_qpos(name=self.control_part),
            name=self.control_part,
            to_matrix=True,
        )
        object_lift = None
        final_object_pose = None
        if object_handle is not None and initial_object_position is not None:
            final_object_pose = object_handle.entity.get_local_pose(
                to_matrix=True
            ).clone()
            final_object_position = final_object_pose[:, :3, 3]
            object_lift = final_object_position[:, 2] - initial_object_position[:, 2]
        return _ExecutionObservation(
            execution_success=torch.isfinite(observed).all(dim=1)
            & (tracking <= settings.joint_tracking_tolerance_rad),
            final_tcp_pose=final_tcp,
            joint_tracking_rmse_rad=tracking,
            execution_time_ms=elapsed_ms,
            task_completion_time_s=simulated_execution_time,
            object_lift_delta_m=object_lift,
            final_arm_qpos=self.robot.get_qpos(name=self.control_part).clone(),
            final_object_pose=final_object_pose,
        )

    def _physics_settings(self) -> _PhysicsReplaySettings:
        """Resolve and validate the common physics-replay step counts."""
        if self.track is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        physics = dict(self.track.config.get("physics", {}))
        settings = _PhysicsReplaySettings(
            steps_per_waypoint=int(physics.get("steps_per_waypoint", 4)),
            hold_steps=int(physics.get("hold_steps", 80)),
            hold_sim_steps=int(physics.get("hold_sim_steps", 2)),
            joint_tracking_tolerance_rad=float(
                physics.get("joint_tracking_tolerance_rad", 0.05)
            ),
        )
        if (
            settings.steps_per_waypoint < 1
            or settings.hold_steps < 0
            or settings.hold_sim_steps < 1
        ):
            raise ValueError("Atomic Task physics step counts are invalid.")
        return settings

    @staticmethod
    def _is_replayable(compiled: CompiledTrajectory) -> bool:
        """Return whether a compiled trajectory can be physically replayed."""
        trajectory = compiled.trajectory.positions
        return (
            trajectory.shape[1] > 0
            and bool(compiled.plan_success.all().item())
            and bool(torch.isfinite(trajectory).all().item())
        )

    @staticmethod
    def _is_recordable(compiled: CompiledTrajectory) -> bool:
        """Allow finite failed planner rollouts in diagnostic videos."""
        trajectory = compiled.trajectory.positions
        return trajectory.shape[1] > 0 and bool(torch.isfinite(trajectory).all().item())

    def _replay_physics(
        self,
        compiled: CompiledTrajectory,
        case: BenchmarkCase,
        provider: AtomicSkillCaseProvider | None,
        *,
        collect_metrics: bool,
    ) -> tuple[torch.Tensor, int] | None:
        """Replay one compiled trajectory with the evaluation physics contract."""
        if self.simulation is None or self.robot is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        trajectory = compiled.trajectory.positions
        object_handle = (
            None if case.object_id is None else self.object_handle(case.object_id)
        )
        settings = self._physics_settings()
        lift_start = None if provider is None else provider.lift_segment_start(compiled)
        dynamics_cleared = False
        squared_tracking_error: torch.Tensor | None = None
        tracking_value_count = 0
        if collect_metrics:
            squared_tracking_error = torch.zeros(
                case.batch_size, dtype=trajectory.dtype, device=trajectory.device
            )
        for waypoint_index in range(trajectory.shape[1]):
            positions = trajectory[:, waypoint_index]
            self.robot.set_qpos(positions, target=True)
            self.simulation.update(step=settings.steps_per_waypoint)
            if squared_tracking_error is not None:
                observed = self.robot.get_qpos()
                squared_tracking_error += ((observed - positions) ** 2).sum(dim=1)
                tracking_value_count += observed.shape[1]
            if (
                object_handle is not None
                and lift_start is not None
                and not dynamics_cleared
                and waypoint_index + 1 >= lift_start
            ):
                object_handle.entity.clear_dynamics()
                dynamics_cleared = True
        final_command = trajectory[:, -1]
        for _ in range(settings.hold_steps):
            self.robot.set_qpos(final_command, target=True)
            self.simulation.update(step=settings.hold_sim_steps)
            if squared_tracking_error is not None:
                observed = self.robot.get_qpos()
                squared_tracking_error += ((observed - final_command) ** 2).sum(dim=1)
                tracking_value_count += observed.shape[1]
        if collect_metrics:
            if squared_tracking_error is None:
                raise RuntimeError("Timed replay lost its tracking accumulator.")
            return squared_tracking_error, tracking_value_count
        return None

    def _hold_static(self) -> None:
        """Hold the current scene so a failed-case debug video has frames."""
        if self.simulation is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        settings = self._physics_settings()
        for _ in range(max(settings.hold_steps, 1)):
            self.simulation.update(step=settings.hold_sim_steps)

    def solve_reference_qpos(
        self, start_qpos: torch.Tensor, target_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """Build independent sequential-IK validity evidence for a case."""
        if self.robot is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        seed = start_qpos
        references: list[torch.Tensor] = []
        for index in range(target_waypoints.shape[1]):
            success, seed = self.robot.compute_ik(
                pose=target_waypoints[:, index],
                joint_seed=seed,
                name=self.control_part,
            )
            if not bool(torch.as_tensor(success).all().item()):
                raise RuntimeError(
                    f"Independent IK rejected atomic target waypoint {index}."
                )
            seed = _canonical_case_qpos(seed)
            references.append(seed.clone())
        return torch.stack(references, dim=1)

    def resolve_antipodal_grasp(
        self,
        handle: AtomicObjectHandle,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        *,
        seed: int,
        start_qpos: torch.Tensor,
        pre_grasp_distance: float,
        lift_height: float,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float,
    ) -> torch.Tensor:
        """Freeze one geometry-aware, independently reachable PGI grasp pose.

        Grasp sampling and sequential IK screening happen once while the case
        manifest is built, before any planner adapter is evaluated.  Every
        planner therefore receives the same explicit grasp pose and planning
        latency excludes grasp generation.
        """
        if self.robot is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        if n_sample < 1 or max_candidates < 1:
            raise ValueError(
                "Antipodal grasp sample/candidate counts must be positive."
            )
        if not 0.0 < alignment_max_angle_deg <= 90.0:
            raise ValueError("grasp_alignment_max_angle_deg must be in (0, 90].")
        from scripts.tutorials.atomic_action.tutorial_utils import (
            create_antipodal_semantics,
            create_parallel_jaw_grasp_pose_generator,
        )

        fork_devices = (
            []
            if self.robot.device.type != "cuda"
            else [self.robot.device.index or torch.cuda.current_device()]
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            semantics = create_antipodal_semantics(
                handle.entity,
                label=handle.object_id,
            )
            affordance = semantics.affordance
            generator = create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample,
                force_refresh=False,
            )
            candidates, costs = generator.get_valid_grasp_poses(
                mesh_vertices=affordance.mesh_vertices,
                mesh_triangles=affordance.mesh_triangles,
                obj_poses=object_pose,
                approach_direction=approach_direction,
            )[0]
        if candidates.shape[0] == 0:
            raise RuntimeError(
                f"No antipodal grasp candidates were found for {handle.object_id!r}."
            )
        finite_indices = torch.nonzero(torch.isfinite(costs), as_tuple=False).flatten()
        if finite_indices.numel() == 0:
            raise RuntimeError(
                f"No valid antipodal grasp candidates were found for {handle.object_id!r}."
            )
        ranked = finite_indices[torch.argsort(costs[finite_indices])]
        ranked = ranked[:max_candidates].detach().to("cpu").tolist()
        minimum_alignment = math.cos(math.radians(alignment_max_angle_deg))
        for candidate_index in ranked:
            candidate = candidates[candidate_index].to(
                device=self.robot.device, dtype=torch.float32
            )
            if (
                float(torch.dot(candidate[:3, 2], approach_direction).item())
                < minimum_alignment
            ):
                continue
            mirrored = candidate.clone()
            mirrored[:3, 0] = -mirrored[:3, 0]
            mirrored[:3, 1] = -mirrored[:3, 1]
            for variant in (candidate, mirrored):
                grasp = variant.unsqueeze(0)
                pre_grasp = grasp.clone()
                pre_grasp[:, :3, 3] -= approach_direction * pre_grasp_distance
                lift = grasp.clone()
                lift[:, 2, 3] += lift_height
                seed = start_qpos
                feasible = True
                for pose in (pre_grasp, grasp, lift):
                    success, seed = self.robot.compute_ik(
                        pose=pose,
                        joint_seed=seed,
                        name=self.control_part,
                    )
                    if not bool(torch.as_tensor(success).all().item()):
                        feasible = False
                        break
                    seed = _canonical_case_qpos(seed)
                if feasible:
                    return grasp
        raise RuntimeError(
            f"No independently reachable antipodal grasp remained for "
            f"{handle.object_id!r} after screening {len(ranked)} candidates."
        )

    def restore_base_robot(self) -> None:
        """Restore the robot state captured before scenario case generation."""
        if self.robot is None or self._base_full_qpos is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        for target in (False, True):
            self.robot.set_qpos(self._base_full_qpos, target=target)
        self.robot.clear_dynamics()

    def randomize_robot_start(
        self,
        config: Mapping[str, object],
        *,
        seed: int,
        stream: int,
    ) -> torch.Tensor:
        """Apply a deterministic, bounded perturbation to the arm start state."""
        if self.robot is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        start = self.robot.get_qpos(name=self.control_part).clone()
        amplitude = _float_vector(
            config.get("start_qpos_jitter_rad"),
            name="start_qpos_jitter_rad",
            length=start.shape[-1],
            default=(0.0,) * start.shape[-1],
        )
        if any(value < 0.0 for value in amplitude):
            raise ValueError("start_qpos_jitter_rad values must be non-negative.")
        randomized = start + _seeded_jitter(
            amplitude,
            seed=seed,
            stream=stream,
            dtype=start.dtype,
            device=start.device,
        )
        limits = self.robot.get_qpos_limits(name=self.control_part)[0]
        margin = float(config.get("start_joint_limit_margin_rad", 0.05))
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("start_joint_limit_margin_rad must be non-negative.")
        lower = limits[:, 0] + margin
        upper = limits[:, 1] - margin
        if bool(((randomized < lower) | (randomized > upper)).any().item()):
            raise RuntimeError(
                "Randomized arm start violates the configured joint-limit margin."
            )
        self.set_robot_start(
            randomized,
            open_gripper=False,
            grasp_gripper=False,
        )
        return randomized

    def randomize_object_pose(
        self,
        handle: AtomicObjectHandle,
        config: Mapping[str, object],
        *,
        seed: int,
        stream: int,
    ) -> torch.Tensor:
        """Install one deterministic position/yaw perturbation for an object."""
        pose = handle.initial_pose.clone()
        position_amplitude = _float_vector(
            config.get("object_position_jitter_m"),
            name="object_position_jitter_m",
            length=3,
            default=(0.0, 0.0, 0.0),
        )
        if any(value < 0.0 for value in position_amplitude):
            raise ValueError("object_position_jitter_m values must be non-negative.")
        pose[:, :3, 3] += _seeded_jitter(
            position_amplitude,
            seed=seed,
            stream=stream,
            dtype=pose.dtype,
            device=pose.device,
        )

        yaw_amplitude = float(config.get("object_yaw_jitter_rad", 0.0))
        if not math.isfinite(yaw_amplitude) or yaw_amplitude < 0.0:
            raise ValueError("object_yaw_jitter_rad must be finite and non-negative.")
        yaw = _seeded_jitter(
            (yaw_amplitude,),
            seed=seed,
            stream=stream + 1,
            dtype=pose.dtype,
            device=pose.device,
        )[0]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        yaw_rotation = torch.eye(3, dtype=pose.dtype, device=pose.device)
        yaw_rotation[0, 0] = cosine
        yaw_rotation[0, 1] = -sine
        yaw_rotation[1, 0] = sine
        yaw_rotation[1, 1] = cosine
        pose[:, :3, :3] = yaw_rotation.unsqueeze(0) @ pose[:, :3, :3]

        handle.entity.set_local_pose(pose)
        handle.entity.clear_dynamics()
        if self.simulation is not None:
            self.simulation.update(step=2)
            handle.entity.set_local_pose(pose)
            handle.entity.clear_dynamics()
        return pose

    def set_robot_start(
        self,
        manipulator_qpos: torch.Tensor,
        *,
        open_gripper: bool,
        grasp_gripper: bool = False,
    ) -> None:
        """Install a manipulator start and optional gripper command."""
        if self.robot is None:
            raise RuntimeError("Atomic Task runtime is not configured.")
        if open_gripper and grasp_gripper:
            raise ValueError("The gripper cannot start both open and grasping.")
        for target in (False, True):
            self.robot.set_qpos(manipulator_qpos, name=self.control_part, target=target)
            if open_gripper or grasp_gripper:
                gripper_qpos = (
                    self._gripper_open if open_gripper else self._gripper_grasp
                )
                if self.end_effector_part is None or gripper_qpos is None:
                    raise RuntimeError("Requested gripper state is unavailable.")
                command = gripper_qpos.unsqueeze(0).expand(
                    manipulator_qpos.shape[0], -1
                )
                self.robot.set_qpos(command, name=self.end_effector_part, target=target)
        self.robot.clear_dynamics()

    def activate_object(self, object_id: str) -> AtomicObjectHandle:
        """Reset one case object and park every other configured object."""
        handle = self.object_handle(object_id)
        for index, candidate in enumerate(self._objects.values()):
            if candidate is handle:
                candidate.reset()
            else:
                candidate.park(index)
        if self.simulation is not None:
            self.simulation.update(step=2)
        return handle

    def object_handle(self, object_id: str | None) -> AtomicObjectHandle:
        """Resolve a configured object identifier with an actionable error."""
        if object_id is None:
            raise ValueError("This Atomic Task case has no object id.")
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown atomic object {object_id!r}; configured objects: "
                f"{sorted(self._objects)}."
            ) from exc

    def close_runtime(self) -> None:
        """Release Python references before SimulationManager teardown."""
        self._engine = None
        self._case_providers.clear()
        self._objects.clear()
        self._base_full_qpos = None
        self.end_effector_part = None
        self._gripper_open = None
        self._gripper_grasp = None
        self.simulation = None
        self.robot = None
        self.suite = None
        self.track = None

    @staticmethod
    def _synchronize() -> None:
        """Synchronize CUDA around physical wall-time measurement."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()


register_atomic_skill_provider("move_end_effector", _MoveEndEffectorCases)
register_atomic_skill_provider("move_held_object", _MoveHeldObjectCases)
register_atomic_skill_provider("move_joints", _MoveJointsCases)
register_atomic_skill_provider("pick_up", _PickUpCases)
register_atomic_skill_provider("place", _PlaceCases)
register_atomic_skill_provider("press", _PressCases)
register_scenario_provider("atomic_task", AtomicTaskScenario)
