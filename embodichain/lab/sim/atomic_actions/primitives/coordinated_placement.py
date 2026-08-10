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

from ._helpers import resolve_object_target
from ..bindings import ResolvedControlPart
from ..control import GRASP_COMMAND, OPEN_COMMAND, JointPositionCommand
from ..core import AtomicAction
from ..effects import StateDelta
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan, normalize_success_mask
from ..policies import MotionPolicy
from ..requirements import (
    ActionBindingRoute,
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    GRASP_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from ..state import HeldObjectState, PlanningContext
from ..trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPlacementGoal:
    """Object-centric target for dual-arm coordinated placement."""

    goal_kind: ClassVar[str] = "coordinated_placement"

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

    placing_arm: ResolvedControlPart
    support_arm: ResolvedControlPart
    placing_hand: ResolvedControlPart
    support_hand: ResolvedControlPart
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
    manipulator_roles: ClassVar[tuple[str, ...]] = ("placing", "support")
    end_effector_roles: ClassVar[tuple[str, ...]] = ("placing", "support")
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="placing",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                        route=ActionBindingRoute("manipulator", "placing"),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={
                            OPEN_COMMAND: JointPositionCommand,
                            GRASP_COMMAND: JointPositionCommand,
                        },
                        route=ActionBindingRoute("end_effector", "placing"),
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
            SkillResourceSlot(
                slot_id="support",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                        route=ActionBindingRoute("manipulator", "support"),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={GRASP_COMMAND: JointPositionCommand},
                        route=ActionBindingRoute("end_effector", "support"),
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
        ),
        constraints=(DisjointResourceSlots(("placing", "support")),),
    )

    def __init__(
        self,
        default_options: CoordinatedPlacementOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve engine-wide resources from the owning engine."""
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[
            CoordinatedPlacementGoal, CoordinatedPlacementOptions
        ],
    ) -> _CoordinatedPlacementResources:
        """Resolve placing/support roles from robot control parts."""
        binding = request.binding
        placing_arm = binding.manipulator("placing")
        support_arm = binding.manipulator("support")
        placing_hand = binding.end_effector("placing")
        support_hand = binding.end_effector("support")
        if placing_arm.name == support_arm.name:
            raise ValueError(
                "CoordinatedPlacement placing and support roles must use "
                "different manipulator control parts."
            )
        if placing_hand.name == support_hand.name:
            raise ValueError(
                "CoordinatedPlacement placing and support roles must use "
                "different end-effector control parts."
            )
        return _CoordinatedPlacementResources(
            placing_arm=placing_arm,
            support_arm=support_arm,
            placing_hand=placing_hand,
            support_hand=support_hand,
            placing_hand_open_qpos=placing_hand.joint_positions(
                OPEN_COMMAND,
                n_envs=self.n_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            placing_hand_close_qpos=placing_hand.joint_positions(
                GRASP_COMMAND,
                n_envs=self.n_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            support_hand_close_qpos=support_hand.joint_positions(
                GRASP_COMMAND,
                n_envs=self.n_envs,
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
        target = self.require_goal(request)
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

        success_mask = torch.ones(
            self.n_envs,
            dtype=torch.bool,
            device=self.device,
        )
        segment_success, placing_approach_traj = self._plan_named_arm_trajectory(
            resources.placing_arm.name,
            placing_start_qpos,
            torch.stack([placing_lift_xpos, placing_xpos], dim=1),
            segments["approach"],
            request.motion_policy,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            n_envs=self.n_envs,
            device=self.device,
            name="Placing-approach success",
        )
        if not success_mask.any():
            logger.log_warning("CoordinatedPlacement failed to plan placing approach.")
            return self.failed_plan(
                request, context, message="Placing approach failed."
            )

        segment_success, support_approach_traj = self._plan_named_arm_trajectory(
            resources.support_arm.name,
            support_start_qpos,
            support_xpos.unsqueeze(1),
            segments["approach"],
            request.motion_policy,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            n_envs=self.n_envs,
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

        segment_success, placing_retreat_traj = self._plan_named_arm_trajectory(
            resources.placing_arm.name,
            placing_place_qpos,
            placing_lift_xpos.unsqueeze(1),
            segments["retreat"],
            request.motion_policy,
        )
        success_mask &= normalize_success_mask(
            segment_success,
            n_envs=self.n_envs,
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
        involved_control_parts = {
            resources.placing_arm.name,
            resources.support_arm.name,
        }
        coordinated_removals = {
            key: None
            for key in state.coordinated_held_objects
            if not involved_control_parts.isdisjoint(key)
        }
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=full,
            expected_effects=StateDelta(
                held_object_updates={
                    resources.placing_arm.name: (
                        None if release else placing_held_object
                    ),
                    resources.support_arm.name: support_held_object,
                },
                coordinated_held_object_updates=coordinated_removals,
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
            n_envs=self.n_envs,
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
        matrix = matrix.to(device=self.device, dtype=torch.float32)
        if matrix.shape == (4, 4):
            matrix = matrix.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if matrix.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"{name} must have shape (4, 4) or ({self.n_envs}, 4, 4), "
                f"but got {matrix.shape}",
                ValueError,
            )
        return matrix

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
        placing_control_part = resources.placing_arm.name
        support_control_part = resources.support_arm.name
        placing_held_object = state.get_held_object(placing_control_part)
        if placing_held_object is None:
            logger.log_error(
                "CoordinatedPlacement requires an object held by placing control "
                f"part {placing_control_part!r}.",
                ValueError,
            )
        support_held_object = state.get_held_object(support_control_part)
        if support_held_object is None:
            logger.log_error(
                "CoordinatedPlacement requires an object held by support control "
                f"part {support_control_part!r}.",
                ValueError,
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
        if state.last_qpos.shape != (self.n_envs, self.robot_dof):
            logger.log_error(
                "PlanningContext.last_qpos must have shape "
                f"({self.n_envs}, {self.robot_dof}), "
                f"but got {state.last_qpos.shape}",
                ValueError,
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
            logger.log_error(
                "Not enough waypoints for coordinated placement. Increase "
                "sample_count or decrease hold/release/retreat steps.",
                ValueError,
            )
        return {
            "approach": n_approach,
            "hold": n_hold,
            "release": n_release,
            "retreat": n_retreat,
        }

    def _plan_named_arm_trajectory(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        motion_policy: MotionPolicy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.motion_generator.generate(
            build_pose_plan_states(target_poses),
            options=motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
                sample_count=n_waypoints,
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        return result.success, result.positions

    @staticmethod
    def _repeat_qpos(qpos: torch.Tensor, n_waypoints: int) -> torch.Tensor:
        return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    def _empty_segment(self) -> torch.Tensor:
        return torch.empty(
            (self.n_envs, 0, self.robot_dof),
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
        n_waypoints = placing_arm_traj.shape[1]
        full = base_full_qpos.to(device=self.device, dtype=torch.float32)
        full = full.unsqueeze(1).repeat(1, n_waypoints, 1).clone()
        full[:, :, list(resources.placing_arm.joint_ids)] = placing_arm_traj
        full[:, :, list(resources.support_arm.joint_ids)] = support_arm_traj
        full[:, :, list(resources.placing_hand.joint_ids)] = placing_hand_traj
        full[:, :, list(resources.support_hand.joint_ids)] = support_hand_traj
        return full


__all__ = [
    "CoordinatedPlacement",
    "CoordinatedPlacementGoal",
    "CoordinatedPlacementOptions",
]
