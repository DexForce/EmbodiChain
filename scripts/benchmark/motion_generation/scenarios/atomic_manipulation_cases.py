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

"""Pickup, held-object motion, and placement benchmark cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    Affordance,
    GraspGoal,
    HeldObjectPoseGoal,
    MotionPolicy,
    MoveHeldObjectOptions,
    ObjectSemantics,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory

from ..atomic_skill import AtomicSkillCaseProvider, register_atomic_skill_provider
from ..contracts import ExecutionObservation, ReplayObservationWindow
from ..config import SuiteCfg, TrackCfg
from ..models import BenchmarkCase, CaseOutcome
from .atomic_objects import AtomicObjectHandle
from .atomic_evaluation import (
    MoveHeldObjectEvaluator,
    PickUpEvaluator,
    PlaceEvaluator,
    object_lift_success,
    object_pose_success,
)
from .atomic_case_utils import (
    TOP_DOWN_ROTATION,
    apply_target_rotation_randomization,
    atomic_case_id,
    canonical_case_qpos,
    case_name,
    case_pose,
    difficulty,
    float_vector,
    held_object_state,
    randomization_parameters,
    randomized_vector_batch,
    snapshot_case_qpos,
)

if TYPE_CHECKING:
    from .atomic_skill import AtomicSkillScenario

__all__ = ["PickUpCases", "HeldObjectCases", "MoveHeldObjectCases", "PlaceCases"]


class PickUpCases(AtomicSkillCaseProvider):
    """Explicit-grasp PickUp cases that isolate motion-planner performance."""

    skill_id = "pick_up"
    requires_gripper = True

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
        object_id = config.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("PickUp cases must reference a non-empty object id.")
        handle = scenario.activate_object(object_id)
        scenario.restore_base_robot()

        raw_settle_steps = config.get(
            "pre_action_settle_steps", handle.config.get("settle_steps", 2)
        )
        if type(raw_settle_steps) is not int or raw_settle_steps < 0:
            raise ValueError("pre_action_settle_steps must be a non-negative integer.")
        # Freeze the pose after gravity has settled the object.  Planning from
        # the configured spawn pose while the dynamic cube is still falling
        # makes the grasp target stale before the first controller command.
        object_pose = scenario.randomize_object_pose(
            handle,
            config,
            seed=seed,
            stream=201,
            settle_steps=raw_settle_steps,
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
                f"Independent IK rejected PickUp case {case_name(config)!r}."
            )
        pre_pick_qpos = canonical_case_qpos(pre_pick_qpos)
        scenario.set_robot_start(pre_pick_qpos, open_gripper=True)

        approach = torch.tensor(
            float_vector(
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
            rotation_value = config.get("grasp_rotation", TOP_DOWN_ROTATION)
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
        grasp_offset = randomized_vector_batch(
            float_vector(
                config.get("grasp_offset_m"),
                name="grasp_offset_m",
                length=3,
                default=(0.0, 0.0, 0.0),
            ),
            config,
            jitter_name="grasp_offset_jitter_m",
            seed=seed,
            stream=202,
            batch_size=batch_size,
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        grasp_pose[:, :3, 3] += grasp_offset
        pre_grasp = grasp_pose.clone()
        pre_grasp[:, :3, 3] -= approach * pre_grasp_distance
        lift = grasp_pose.clone()
        lift[:, 2, 3] += lift_height
        targets = torch.stack([pre_grasp, grasp_pose, lift], dim=1)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
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
            num_waypoints=3,
            path_shape="approach_grasp_lift",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=object_id,
            task_difficulty=difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
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
                "pre_action_settle_steps": raw_settle_steps,
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "object_initial_pose": object_pose.detach().cpu().tolist(),
                "object_config": dict(handle.config),
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
                object_pose=(
                    None
                    if "object_initial_pose" not in case.case_parameters
                    else case_pose(
                        case,
                        "object_initial_pose",
                        device=scenario.robot.device,
                    )
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

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect grasp, lift, and terminal retention evidence."""
        del case
        close = compiled.segment(0, "close")
        lift = compiled.segment(0, "lift")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "pickup_grasp",
                close.start,
                close.stop,
                frozenset({"gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "pickup_lift",
                lift.start,
                lift.stop,
                frozenset({"object_pose", "tcp_pose"}),
            ),
            ReplayObservationWindow(
                "pickup_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose", "tcp_pose", "gripper_qpos"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> PickUpEvaluator:
        """Return the v1 PickUp evaluator."""
        return PickUpEvaluator()

    def task_result(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        compiled: CompiledTrajectory,
        observation: ExecutionObservation,
        motion_outcomes: tuple[CaseOutcome, ...],
    ) -> tuple[torch.Tensor, str]:
        del scenario, compiled
        return (
            object_lift_success(case, observation, motion_outcomes),
            "object_not_lifted",
        )


class HeldObjectCases(AtomicSkillCaseProvider):
    """Shared isolated held-object precondition for transport and placement."""

    requires_gripper = True

    def lift_segment_start(self, compiled: CompiledTrajectory) -> int | None:
        """Return the prepended PickUp lift boundary for composite cases.

        ``move_held_object`` and ``place`` are compiled as
        ``PickUp -> transport/place`` sequences.  Once the first PickUp lift
        segment begins, the benchmark replay should clear residual rigid-body
        dynamics just as it does for an isolated PickUp; otherwise gravity and
        contact impulses can make the cube drift out of the gripper before the
        second action starts.
        """
        return compiled.segment(0, "lift").start

    def _prepare_start(
        self,
        scenario: "AtomicSkillScenario",
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
        # Match the tutorials: obtain the grasp target only after gravity has
        # settled the object.  Reading the configured spawn pose here would
        # leave the case grounded at one height while the grasp was planned
        # around a different, manually offset height.
        settle_steps = int(config.get("pre_action_settle_steps", 50))
        if settle_steps < 0:
            raise ValueError("pre_action_settle_steps must be non-negative.")
        if scenario.simulation is not None and settle_steps:
            scenario.simulation.update(step=settle_steps)
            table_object_pose = handle.entity.get_local_pose(to_matrix=True).clone()
        approach_direction = torch.tensor(
            config.get("approach_direction", [0.0, 0.0, -1.0]),
            dtype=table_object_pose.dtype,
            device=table_object_pose.device,
        )
        direction_norm = torch.linalg.vector_norm(approach_direction)
        if float(direction_norm.item()) <= 1.0e-6:
            raise ValueError("approach_direction must be non-zero.")
        approach_direction = approach_direction / direction_norm
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        lift_height = float(config.get("lift_height_m", 0.16))
        grasp_source = str(config.get("grasp_source", "antipodal"))
        if grasp_source == "antipodal":
            object_pose = table_object_pose.clone()
            grasp_pose = scenario.resolve_antipodal_grasp(
                handle,
                object_pose,
                approach_direction,
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
            # These cases now prepend a real PickUp action at replay time.
            # Keep the object at its settled table pose; ``held_object_offset``
            # belonged to the old pre-held initialization and must not lift
            # either the physical object or the pickup TCP above the table.
            object_pose = table_object_pose.clone()
            grasp_pose = object_pose.clone()
            grasp_pose[:, :3, :3] = torch.tensor(
                config.get("grasp_rotation", TOP_DOWN_ROTATION),
                dtype=grasp_pose.dtype,
                device=grasp_pose.device,
            )
            grasp_pose[:, :3, 3] += randomized_vector_batch(
                float_vector(
                    config.get("grasp_offset_m"),
                    name="grasp_offset_m",
                    length=3,
                    default=(0.0, 0.0, 0.0),
                ),
                config,
                jitter_name="grasp_offset_jitter_m",
                seed=seed,
                stream=303,
                batch_size=table_object_pose.shape[0],
                dtype=grasp_pose.dtype,
                device=grasp_pose.device,
            )
        else:
            raise ValueError("grasp_source must be 'antipodal' or 'fixed'.")
        arm_seed = scenario.robot.get_qpos(name=scenario.control_part)
        success, arm_start = scenario.robot.compute_ik(
            pose=grasp_pose,
            joint_seed=arm_seed,
            name=scenario.control_part,
        )
        if not bool(torch.as_tensor(success).all().item()):
            raise RuntimeError(
                f"Independent IK rejected held-object start {case_name(config)!r}."
            )
        arm_start = canonical_case_qpos(arm_start)
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
        self, scenario: "AtomicSkillScenario", case: BenchmarkCase
    ) -> TaskState:
        return TaskState(
            batch_size=case.batch_size,
            device=scenario.robot.device,
            held_objects={scenario.control_part: held_object_state(scenario, case)},
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


class MoveHeldObjectCases(HeldObjectCases):
    """Move an already held rigid object to one frozen target pose."""

    skill_id = "move_held_object"

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
        (
            handle,
            object_pose,
            grasp_pose,
            object_to_eef,
            table_object_pose,
        ) = self._prepare_start(scenario, config, seed=seed)
        lift_height = float(config.get("lift_height_m", 0.16))
        # The preceding PickUp lifts the object from the settled table pose;
        # derive the transport target from that same pose rather than from the
        # legacy pre-held offset.
        target_object = table_object_pose.clone()
        target_object[:, :3, 3] += randomized_vector_batch(
            float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=(0.08, 0.08, 0.04),
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=311,
            batch_size=object_pose.shape[0],
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        apply_target_rotation_randomization(
            target_object,
            config,
            seed=seed,
            stream=312,
        )
        # The object is now picked from the table and lifted before the
        # transport action.  Express the transport target relative to that
        # lifted state instead of the old teleported held-object pose.
        target_object[:, 2, 3] += lift_height
        target_eef = torch.bmm(target_object, object_to_eef)
        approach_direction = torch.tensor(
            config.get("approach_direction", [0.0, 0.0, -1.0]),
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        pre_grasp = grasp_pose.clone()
        pre_grasp[:, :3, 3] -= approach_direction * pre_grasp_distance
        ik_success, pre_pick_qpos = scenario.robot.compute_ik(
            pose=pre_grasp,
            joint_seed=scenario.robot.get_qpos(name=scenario.control_part),
            name=scenario.control_part,
        )
        if not bool(torch.as_tensor(ik_success).all().item()):
            raise RuntimeError(
                "Independent IK rejected MoveHeldObject pre-pick pose "
                f"{case_name(config)!r}."
            )
        pre_pick_qpos = canonical_case_qpos(pre_pick_qpos)
        scenario.set_robot_start(pre_pick_qpos, open_gripper=True)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        references = scenario.solve_reference_qpos(start_qpos, target_eef[:, None])
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
            num_waypoints=1,
            path_shape="pick_up_then_held_object_transport",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=target_eef[:, None],
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                **self._held_parameters(handle, object_pose, grasp_pose, object_to_eef),
                "object_initial_pose": table_object_pose.detach().cpu().tolist(),
                "sample_count": int(config.get("sample_count", 80)),
                "pick_sample_count": int(config.get("pick_sample_count", 120)),
                "approach_direction": approach_direction.detach().cpu().tolist(),
                "pre_grasp_distance_m": pre_grasp_distance,
                "lift_height_m": lift_height,
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "pre_action_settle_steps": int(
                    config.get("pre_action_settle_steps", 50)
                ),
                "target_object_pose": target_object.detach().cpu().tolist(),
                "object_position_threshold_m": float(
                    config.get("object_position_threshold_m", 0.04)
                ),
                "object_tilt_threshold_rad": float(
                    config.get(
                        "object_tilt_threshold_rad",
                        suite.protocol.rotation_threshold_rad,
                    )
                ),
                "object_rotation_threshold_rad": float(
                    config.get(
                        "object_rotation_threshold_rad",
                        suite.protocol.rotation_threshold_rad,
                    )
                ),
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
            HeldObjectPoseGoal(
                case_pose(case, "target_object_pose", device=scenario.robot.device)
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
            skill_options=MoveHeldObjectOptions(),
        )

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect transport plus settled pose and grasp state."""
        del case
        transport = compiled.segment(1, "transport")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "held_object_transport",
                transport.start,
                transport.stop,
                frozenset({"object_pose"}),
            ),
            ReplayObservationWindow(
                "held_object_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose", "gripper_qpos"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> MoveHeldObjectEvaluator:
        """Return the v1 MoveHeldObject evaluator."""
        return MoveHeldObjectEvaluator()

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
            object_pose_success(scenario, case, observation, motion_outcomes),
            "object_goal_miss",
        )


class PlaceCases(HeldObjectCases):
    """Place an already held rigid object and release it under physics."""

    skill_id = "place"

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
        (
            handle,
            object_pose,
            grasp_pose,
            object_to_eef,
            table_object_pose,
        ) = self._prepare_start(scenario, config, seed=seed)
        target_object = table_object_pose.clone()
        target_object[:, :3, 3] += randomized_vector_batch(
            float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=(0.10, 0.10, 0.0),
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=321,
            batch_size=object_pose.shape[0],
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        apply_target_rotation_randomization(
            target_object,
            config,
            seed=seed,
            stream=322,
        )
        release = torch.bmm(target_object, object_to_eef)
        lift_height = float(config.get("retract_height_m", 0.10))
        approach = release.clone()
        approach[:, 2, 3] += lift_height
        retract = approach.clone()
        targets = torch.stack([approach, release, retract], dim=1)
        approach_direction = torch.tensor(
            config.get("approach_direction", [0.0, 0.0, -1.0]),
            dtype=grasp_pose.dtype,
            device=grasp_pose.device,
        )
        approach_direction = approach_direction / torch.linalg.vector_norm(
            approach_direction
        )
        pre_grasp_distance = float(config.get("pre_grasp_distance_m", 0.15))
        pre_grasp = grasp_pose.clone()
        pre_grasp[:, :3, 3] -= approach_direction * pre_grasp_distance
        ik_success, pre_pick_qpos = scenario.robot.compute_ik(
            pose=pre_grasp,
            joint_seed=scenario.robot.get_qpos(name=scenario.control_part),
            name=scenario.control_part,
        )
        if not bool(torch.as_tensor(ik_success).all().item()):
            raise RuntimeError(
                f"Independent IK rejected Place pre-pick pose {case_name(config)!r}."
            )
        pre_pick_qpos = canonical_case_qpos(pre_pick_qpos)
        scenario.set_robot_start(pre_pick_qpos, open_gripper=True)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
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
            num_waypoints=3,
            path_shape="approach_release_retract",
            start_state_bin="pre_pick",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=handle.object_id,
            task_difficulty=difficulty(config),
            primary_success="task_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                **self._held_parameters(handle, object_pose, grasp_pose, object_to_eef),
                # The replay starts with the cube on the table.  PickUp is
                # compiled as the first action of this Place case, so the
                # object is never teleported into the closed gripper.
                "object_initial_pose": table_object_pose.detach().cpu().tolist(),
                "sample_count": int(config.get("sample_count", 120)),
                "pick_sample_count": int(config.get("pick_sample_count", 120)),
                "release_pose": release.detach().cpu().tolist(),
                "target_object_pose": target_object.detach().cpu().tolist(),
                "approach_direction": list(approach_direction.detach().cpu().tolist()),
                "pre_grasp_distance_m": pre_grasp_distance,
                "lift_height_m": float(config.get("lift_height_m", 0.16)),
                "pre_action_settle_steps": int(
                    config.get("pre_action_settle_steps", 50)
                ),
                "hand_interp_steps": int(config.get("hand_interp_steps", 12)),
                "retract_height_m": lift_height,
                "object_position_threshold_m": float(
                    config.get("object_position_threshold_m", 0.05)
                ),
                "object_tilt_threshold_rad": float(
                    config.get(
                        "object_tilt_threshold_rad",
                        suite.protocol.rotation_threshold_rad,
                    )
                ),
                "object_rotation_threshold_rad": float(
                    config.get(
                        "object_rotation_threshold_rad",
                        suite.protocol.rotation_threshold_rad,
                    )
                ),
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
            PlaceGoal(case_pose(case, "release_pose", device=scenario.robot.device)),
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

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect release and terminal stability evidence."""
        del case
        release = compiled.segment(1, "release")
        waypoint_count = compiled.trajectory.waypoint_count
        return (
            ReplayObservationWindow(
                "place_release",
                release.start,
                release.stop,
                frozenset({"gripper_qpos"}),
            ),
            ReplayObservationWindow(
                "place_settle",
                waypoint_count,
                waypoint_count,
                frozenset({"object_pose"}),
                include_settle=True,
            ),
        )

    def physical_evaluator(self) -> PlaceEvaluator:
        """Return the v1 Place evaluator."""
        return PlaceEvaluator()

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
            object_pose_success(scenario, case, observation, motion_outcomes),
            "object_not_placed",
        )


register_atomic_skill_provider("move_held_object", MoveHeldObjectCases)


register_atomic_skill_provider("pick_up", PickUpCases)


register_atomic_skill_provider("place", PlaceCases)
