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

"""Representative end-effector and joint-motion integration cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    EndEffectorPoseGoal,
    JointPositionGoal,
    MotionPolicy,
    MoveJointsOptions,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory

from ..atomic_skill import AtomicSkillCaseProvider, register_atomic_skill_provider
from ..contracts import ExecutionObservation
from ..config import SuiteCfg, TrackCfg
from ..models import BenchmarkCase, CaseOutcome
from .atomic_evaluation import (
    MoveEndEffectorEvaluator,
    MoveJointsEvaluator,
    executed_final_pose_success,
    motion_valid_mask,
)
from .atomic_case_utils import (
    atomic_case_id,
    case_name,
    difficulty,
    float_vector,
    randomization_parameters,
    randomized_vector_batch,
    snapshot_case_qpos,
)

if TYPE_CHECKING:
    from .atomic_skill import AtomicSkillScenario

__all__ = ["MoveEndEffectorCases", "MoveJointsCases"]


class MoveEndEffectorCases(AtomicSkillCaseProvider):
    """Deterministic robot-relative MoveEndEffector cases."""

    skill_id = "move_end_effector"

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
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
            float_vector(value, name="target_offsets_m", length=3)
            for value in raw_offsets
        ]
        if not base_offsets:
            raise ValueError("target_offsets_m must not be empty.")

        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )
        if bool(config.get("_pose_batch_enabled", False)):
            # One batch row represents one perturbed target pose.  Reuse the
            # same translation at every waypoint so the requested path shape
            # is preserved while each environment explores a different pose.
            shared_jitter = randomized_vector_batch(
                (0.0, 0.0, 0.0),
                config,
                jitter_name="target_offset_jitter_m",
                seed=seed,
                stream=101,
                batch_size=batch_size,
                dtype=start_pose.dtype,
                device=start_pose.device,
            )
            base_tensor = torch.tensor(
                base_offsets,
                dtype=start_pose.dtype,
                device=start_pose.device,
            )
            offsets_tensor = base_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            offsets_tensor = offsets_tensor + shared_jitter[:, None, :]
        else:
            offsets_tensor = torch.stack(
                [
                    randomized_vector_batch(
                        offset,
                        config,
                        jitter_name="target_offset_jitter_m",
                        seed=seed,
                        stream=101 + index,
                        batch_size=batch_size,
                        dtype=start_pose.dtype,
                        device=start_pose.device,
                    )
                    for index, offset in enumerate(base_offsets)
                ],
                dim=1,
            )
        offsets = offsets_tensor.detach().cpu().tolist()
        # Preserve the historical manifest shape for the ordinary B=1
        # benchmark (`[waypoint][xyz]`), while retaining one row per pose
        # sample for the batched variant (`[batch][waypoint][xyz]`).
        if batch_size == 1:
            offsets = offsets[0]
        targets = start_pose[:, None].repeat(1, len(base_offsets), 1, 1)
        targets[:, :, :3, 3] += offsets_tensor
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=atomic_case_id(
                track.id,
                self.skill_id,
                name,
                batch_size=batch_size,
                seed=seed,
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=len(base_offsets),
            path_shape="robot_relative_waypoints",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            task_difficulty=difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "target_offsets_m": offsets,
                "randomization": randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
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
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del compiled
        return (
            executed_final_pose_success(
                scenario,
                case,
                observation,
                motion_outcomes,
            ),
            "task_goal_miss",
        )

    def physical_evaluator(self) -> MoveEndEffectorEvaluator:
        """Evaluate the terminal measured TCP position under the v1 standard."""
        return MoveEndEffectorEvaluator()


class MoveJointsCases(AtomicSkillCaseProvider):
    """Deterministic relative joint-space waypoint cases."""

    skill_id = "move_joints"

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
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
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        base_offsets = [
            float_vector(
                value,
                name="target_offsets_rad",
                length=start_qpos.shape[-1],
            )
            for value in raw_offsets
        ]
        if not base_offsets:
            raise ValueError("target_offsets_rad must not be empty.")
        offset_tensor = torch.stack(
            [
                randomized_vector_batch(
                    offset,
                    config,
                    jitter_name="target_offset_jitter_rad",
                    seed=seed,
                    stream=211 + index,
                    batch_size=batch_size,
                    dtype=start_qpos.dtype,
                    device=start_qpos.device,
                )
                for index, offset in enumerate(base_offsets)
            ],
            dim=1,
        )
        offsets = offset_tensor.detach().cpu().tolist()
        # Keep B=1 case manifests backward-compatible; B>1 exposes the
        # per-environment perturbations explicitly.
        if batch_size == 1:
            offsets = offsets[0]
        targets = start_qpos[:, None] + torch.cumsum(offset_tensor, dim=1)
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
                f"Joint limits rejected MoveJoints case {case_name(config)!r}."
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
        name = case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=atomic_case_id(
                track.id,
                self.skill_id,
                name,
                batch_size=batch_size,
                seed=seed,
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=len(base_offsets),
            path_shape="relative_joint_waypoints",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=target_poses,
            reference_qpos=targets,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            task_difficulty=difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "target_offsets_rad": offsets,
                "target_qpos": targets.detach().cpu().tolist(),
                "joint_threshold_rad": float(config.get("joint_threshold_rad", 0.02)),
                "motion_validity": "ordered_joint_waypoints",
                "randomization": randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
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
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: ExecutionObservation,
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
            & motion_valid_mask(motion_outcomes, device=error.device)
            & (error <= float(case.case_parameters["joint_threshold_rad"]))
        )
        return success, "joint_goal_miss"

    def physical_evaluator(self) -> MoveJointsEvaluator:
        """Evaluate the terminal measured joint state under the v1 standard."""
        return MoveJointsEvaluator()


register_atomic_skill_provider("move_end_effector", MoveEndEffectorCases)


register_atomic_skill_provider("move_joints", MoveJointsCases)
