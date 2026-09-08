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

"""CoordinatedPlacement atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils import logger

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_hand_qpos,
    translate_pose_world,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    plan_named_arm_trajectory,
    require_shared_task_state_key,
    repeat_qpos,
    resolve_batched_pose,
    resolve_object_target,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPlacementGoal:
    """Object-centric target for dual-arm coordinated placement."""

    placing_object_target_pose: PoseGoalValue
    """Target pose for the object released by the placing arm."""

    support_object_target_pose: PoseGoalValue
    """Target pose for the object held by the support arm."""

    placing_height_offset: float | None = None
    """World-Z offset above the placing object target pose."""

    support_height_offset: float | None = None
    """World-Z offset above the support object target pose."""

    release: bool | None = None
    """Whether the placing hand releases. ``None`` uses invocation options."""

    def __post_init__(self) -> None:
        validate_pose_goal(
            self.placing_object_target_pose,
            "placing_object_target_pose",
            allow_waypoints=False,
        )
        validate_pose_goal(
            self.support_object_target_pose,
            "support_object_target_pose",
            allow_waypoints=False,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPlacementOptions(ActionOptions):
    """Per-invocation coordinated placement behavior."""

    release: bool = True
    """Whether to open the placing hand at the aligned placement pose."""

    placing_height_offset: float = 0.0
    """Default World-Z offset above the placing object target pose."""

    support_height_offset: float = 0.0
    """Default World-Z offset above the support object target pose."""

    lift_height: float = 0.08
    """World-Z lift distance for the placing arm after release."""

    hand_interp_steps: int = 10
    """Number of waypoints for the placing-hand release interpolation."""

    hold_steps: int = 4
    """Number of waypoints to hold alignment before releasing."""

    retreat_steps: int = 16
    """Number of waypoints used for the placing-arm lift retreat."""

    def __post_init__(self) -> None:
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        for name in ("hand_interp_steps", "hold_steps", "retreat_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass(frozen=True, slots=True, eq=False)
class _CoordinatedPlacementResources:
    """Invocation-bound control parts and compatible hand commands."""

    placing_task_state_key: str
    support_task_state_key: str
    placing_arm: JointPositionTarget
    support_arm: JointPositionTarget
    placing_hand: JointPositionTarget
    support_hand: JointPositionTarget
    placing_hand_open_qpos: torch.Tensor
    placing_hand_close_qpos: torch.Tensor
    support_hand_close_qpos: torch.Tensor


class CoordinatedPlacement(
    AtomicAction[CoordinatedPlacementGoal, CoordinatedPlacementOptions]
):
    """Coordinate two held objects: support object below, placing object above."""

    skill_id: ClassVar[str] = "coordinated_placement"
    GoalType: ClassVar[type] = CoordinatedPlacementGoal
    OptionsType: ClassVar[type] = CoordinatedPlacementOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "placing",
                motion_capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
            make_manipulation_slot(
                "support",
                motion_capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
        constraints=(DisjointResourceSlots(("placing", "support")),),
    )
    _repeat_qpos = staticmethod(repeat_qpos)

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[
            CoordinatedPlacementGoal, CoordinatedPlacementOptions
        ],
    ) -> _CoordinatedPlacementResources:
        """Resolve placing/support roles from robot control parts."""
        binding = request.binding
        placing_motion = binding.endpoint("placing", "motion")
        support_motion = binding.endpoint("support", "motion")
        placing_grasp = binding.endpoint("placing", "grasp")
        support_grasp = binding.endpoint("support", "grasp")
        placing_arm = placing_motion.require_target(JointPositionTarget)
        support_arm = support_motion.require_target(JointPositionTarget)
        placing_hand = placing_grasp.require_target(JointPositionTarget)
        support_hand = support_grasp.require_target(JointPositionTarget)
        placing_task_state_key = require_shared_task_state_key(
            placing_motion,
            placing_grasp,
            participant="CoordinatedPlacement placing participant",
        )
        support_task_state_key = require_shared_task_state_key(
            support_motion,
            support_grasp,
            participant="CoordinatedPlacement support participant",
        )
        if placing_task_state_key == support_task_state_key:
            raise ValueError(
                "CoordinatedPlacement placing and support participants must "
                "use different task_state_key values."
            )
        if placing_arm.control_part == support_arm.control_part:
            raise ValueError(
                "CoordinatedPlacement placing and support roles must use "
                "different manipulator control parts."
            )
        if placing_hand.control_part == support_hand.control_part:
            raise ValueError(
                "CoordinatedPlacement placing and support roles must use "
                "different end-effector control parts."
            )
        return _CoordinatedPlacementResources(
            placing_task_state_key=placing_task_state_key,
            support_task_state_key=support_task_state_key,
            placing_arm=placing_arm,
            support_arm=support_arm,
            placing_hand=placing_hand,
            support_hand=support_hand,
            placing_hand_open_qpos=placing_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            placing_hand_close_qpos=placing_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            support_hand_close_qpos=support_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _plan(
        self,
        request: ResolvedActionRequest[
            CoordinatedPlacementGoal, CoordinatedPlacementOptions
        ],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan coordinated placement without committing attachment changes."""
        target = request.goal
        options = request.skill_options
        resources = self._resolve_resources(request)
        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by the cuRobo backend."
            )
        state = context
        (
            placing_xpos,
            support_xpos,
            release,
            placing_held_object,
            support_held_object,
        ) = self._resolve_target(target, state, resources, options)
        eligible = context.task.exclusive_held_object_mask(
            resources.placing_task_state_key
        ) & context.task.exclusive_held_object_mask(resources.support_task_state_key)
        if not eligible.any():
            logger.log_warning(
                "CoordinatedPlacement requires two exclusively held objects."
            )
            return self.failed_plan(
                request,
                context,
                message="Placing and support objects must be held exclusively.",
            )
        placing_start_qpos, support_start_qpos = self._resolve_start_qpos(
            state, resources
        )
        segments = self._compute_segment_lengths(
            release, request.motion_policy.sample_count, options
        )

        placing_lift_xpos = translate_pose_world(
            placing_xpos,
            torch.tensor(
                [0.0, 0.0, options.lift_height],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        success_mask = eligible.clone()
        segment_success, placing_approach_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.placing_arm.control_part,
            placing_start_qpos,
            torch.stack([placing_lift_xpos, placing_xpos], dim=1),
            segments["approach"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Placing-approach success",
        )
        if not success_mask.any():
            logger.log_warning("CoordinatedPlacement failed to plan placing approach.")
            return self.failed_plan(
                request, context, message="Placing approach failed."
            )

        segment_success, support_approach_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.support_arm.control_part,
            support_start_qpos,
            support_xpos.unsqueeze(1),
            segments["approach"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Support-approach success",
        )
        if not success_mask.any():
            logger.log_warning("CoordinatedPlacement failed to plan support approach.")
            return self.failed_plan(
                request, context, message="Support approach failed."
            )

        placing_place_qpos = placing_approach_traj[:, -1]
        support_place_qpos = support_approach_traj[:, -1]
        approach_trajectory = self._assemble_segment(
            state.last_qpos,
            placing_approach_traj,
            support_approach_traj,
            self._repeat_qpos(resources.placing_hand_close_qpos, segments["approach"]),
            self._repeat_qpos(resources.support_hand_close_qpos, segments["approach"]),
            resources=resources,
        )

        hold_trajectory = self._empty_segment()
        if segments["hold"] > 0:
            hold_trajectory = self._assemble_segment(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["hold"]),
                self._repeat_qpos(support_place_qpos, segments["hold"]),
                self._repeat_qpos(resources.placing_hand_close_qpos, segments["hold"]),
                self._repeat_qpos(resources.support_hand_close_qpos, segments["hold"]),
                resources=resources,
            )

        release_trajectory = self._empty_segment()
        if release:
            release_trajectory = self._assemble_segment(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["release"]),
                self._repeat_qpos(support_place_qpos, segments["release"]),
                interpolate_hand_qpos(
                    resources.placing_hand_close_qpos,
                    resources.placing_hand_open_qpos,
                    n_waypoints=segments["release"],
                ),
                self._repeat_qpos(
                    resources.support_hand_close_qpos, segments["release"]
                ),
                resources=resources,
            )

        segment_success, placing_retreat_traj = plan_named_arm_trajectory(
            self.motion_generator,
            resources.placing_arm.control_part,
            placing_place_qpos,
            placing_lift_xpos.unsqueeze(1),
            segments["retreat"],
            request.motion_policy,
            context.control_dt,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            num_envs=self.num_envs,
            device=self.device,
            name="Placing-retreat success",
        )
        if not success_mask.any():
            logger.log_warning("CoordinatedPlacement failed to plan placing retreat.")
            return self.failed_plan(request, context, message="Placing retreat failed.")

        placing_hand_retreat_qpos = (
            resources.placing_hand_open_qpos
            if release
            else resources.placing_hand_close_qpos
        )
        retreat_trajectory = self._assemble_segment(
            state.last_qpos,
            placing_retreat_traj,
            self._repeat_qpos(support_place_qpos, segments["retreat"]),
            self._repeat_qpos(placing_hand_retreat_qpos, segments["retreat"]),
            self._repeat_qpos(resources.support_hand_close_qpos, segments["retreat"]),
            resources=resources,
        )

        full = torch.cat(
            [
                approach_trajectory,
                hold_trajectory,
                release_trajectory,
                retreat_trajectory,
            ],
            dim=1,
        )
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={
                    resources.placing_task_state_key: (
                        None if release else placing_held_object
                    ),
                    resources.support_task_state_key: support_held_object,
                },
            ),
            segment_lengths={
                "approach": approach_trajectory.shape[1],
                "hold": hold_trajectory.shape[1],
                "release": release_trajectory.shape[1],
                "retreat": retreat_trajectory.shape[1],
            },
        )

    def _resolve_object_pose(
        self,
        pose: PoseGoalValue,
        height_offset: float,
        name: str,
        context: PlanningContext,
    ) -> torch.Tensor:
        object_pose = resolve_object_target(
            resolve_pose_goal(pose, context, name=name),
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )
        return translate_pose_world(
            object_pose,
            torch.tensor(
                [0.0, 0.0, height_offset],
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def _resolve_object_to_eef(
        self,
        held_state: HeldObjectState,
        name: str,
    ) -> torch.Tensor:
        return self._resolve_held_matrix(
            held_state.object_to_eef,
            f"{name}.object_to_eef",
        )

    def _resolve_held_matrix(self, matrix: torch.Tensor, name: str) -> torch.Tensor:
        return resolve_batched_pose(
            matrix,
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )

    def _resolve_held_state(
        self,
        held_state: HeldObjectState,
        name: str,
        object_to_eef: torch.Tensor,
    ) -> HeldObjectState:
        return HeldObjectState(
            semantics=held_state.semantics,
            object_to_eef=object_to_eef,
            grasp_xpos=self._resolve_held_matrix(
                held_state.grasp_xpos,
                f"{name}.grasp_xpos",
            ),
        )

    def _resolve_target(
        self,
        target: CoordinatedPlacementGoal,
        state: PlanningContext,
        resources: _CoordinatedPlacementResources,
        options: CoordinatedPlacementOptions,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        bool,
        HeldObjectState,
        HeldObjectState,
    ]:
        placing_task_state_key = resources.placing_task_state_key
        support_task_state_key = resources.support_task_state_key
        placing_held_object = state.get_held_object(placing_task_state_key)
        if placing_held_object is None:
            raise ValueError(
                "CoordinatedPlacement requires an object held by placing "
                f"task-state resource {placing_task_state_key!r}."
            )
        support_held_object = state.get_held_object(support_task_state_key)
        if support_held_object is None:
            raise ValueError(
                "CoordinatedPlacement requires an object held by support "
                f"task-state resource {support_task_state_key!r}."
            )
        placing_height_offset = (
            options.placing_height_offset
            if target.placing_height_offset is None
            else target.placing_height_offset
        )
        support_height_offset = (
            options.support_height_offset
            if target.support_height_offset is None
            else target.support_height_offset
        )
        placing_object_pose = self._resolve_object_pose(
            target.placing_object_target_pose,
            placing_height_offset,
            "placing_object_target_pose",
            state,
        )
        support_object_pose = self._resolve_object_pose(
            target.support_object_target_pose,
            support_height_offset,
            "support_object_target_pose",
            state,
        )
        placing_object_to_eef = self._resolve_object_to_eef(
            placing_held_object,
            "placing_held_object",
        )
        support_object_to_eef = self._resolve_object_to_eef(
            support_held_object,
            "support_held_object",
        )
        placing_xpos = torch.bmm(placing_object_pose, placing_object_to_eef)
        support_xpos = torch.bmm(support_object_pose, support_object_to_eef)
        release = options.release if target.release is None else target.release
        return (
            placing_xpos,
            support_xpos,
            release,
            self._resolve_held_state(
                placing_held_object,
                "placing_held_object",
                placing_object_to_eef,
            ),
            self._resolve_held_state(
                support_held_object,
                "support_held_object",
                support_object_to_eef,
            ),
        )

    def _resolve_start_qpos(
        self,
        state: PlanningContext,
        resources: _CoordinatedPlacementResources,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.last_qpos.shape != (self.num_envs, self.robot_dof):
            raise ValueError(
                "PlanningContext.last_qpos must have shape "
                f"({self.num_envs}, {self.robot_dof}), "
                f"but got {state.last_qpos.shape}"
            )
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        return (
            start_qpos[:, list(resources.placing_arm.joint_ids)],
            start_qpos[:, list(resources.support_arm.joint_ids)],
        )

    def _compute_segment_lengths(
        self,
        release: bool,
        sample_count: int,
        options: CoordinatedPlacementOptions,
    ) -> dict[str, int]:
        """Split the invocation sample budget across placement segments."""
        n_release = max(2, options.hand_interp_steps) if release else 0
        n_hold = max(0, options.hold_steps)
        n_retreat = max(2, options.retreat_steps)
        n_approach = sample_count - n_hold - n_release - n_retreat
        if n_approach < 2:
            raise ValueError(
                "Not enough waypoints for coordinated placement. Increase "
                "sample_count or decrease hold/release/retreat steps."
            )
        return {
            "approach": n_approach,
            "hold": n_hold,
            "release": n_release,
            "retreat": n_retreat,
        }

    def _empty_segment(self) -> torch.Tensor:
        return torch.empty(
            (self.num_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )

    def _assemble_segment(
        self,
        base_full_qpos: torch.Tensor,
        placing_arm_traj: torch.Tensor,
        support_arm_traj: torch.Tensor,
        placing_hand_traj: torch.Tensor,
        support_hand_traj: torch.Tensor,
        *,
        resources: _CoordinatedPlacementResources,
    ) -> torch.Tensor:
        return assemble_full_robot_trajectory(
            base_full_qpos,
            (
                (resources.placing_arm.joint_ids, placing_arm_traj),
                (resources.support_arm.joint_ids, support_arm_traj),
                (resources.placing_hand.joint_ids, placing_hand_traj),
                (resources.support_hand.joint_ids, support_hand_traj),
            ),
        )


__all__ = [
    "CoordinatedPlacement",
    "CoordinatedPlacementGoal",
    "CoordinatedPlacementOptions",
]
