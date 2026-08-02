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

"""Place atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..affordance import AssembleAffordance
from ..core import (
    ActionCfg,
    AtomicAction,
)
from ..effects import StateDelta
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionInvocation
from ..plans import ActionPlan
from ..state import PlanningContext

TcpSymmetry = Literal["none", "z_roll_180"]


@dataclass(frozen=True, slots=True, eq=False)
class PlaceGoal:
    """End-effector release-pose target used by :class:`Place`."""

    goal_kind: ClassVar[str] = "place_pose"

    xpos: PoseGoalValue
    """Target end-effector release pose.

    Accepts ``(4, 4)``, ``(n_envs, 4, 4)``, or
    ``(n_envs, n_waypoint, 4, 4)``.
    """

    tcp_symmetry: TcpSymmetry = "none"
    """Optional TCP-frame symmetry allowed by the placement semantics.

    ``"none"`` preserves the pose exactly. ``"z_roll_180"`` lets placement
    choose between the pose and its TCP z-roll 180 equivalent, which flips TCP
    x/y while preserving TCP z and translation.
    """

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=True)
        if self.tcp_symmetry not in ("none", "z_roll_180"):
            raise ValueError(
                "tcp_symmetry must be one of 'none' or 'z_roll_180', "
                f"but got {self.tcp_symmetry!r}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class AssembleGoal:
    """Place a held assemble object onto a base object at a relative pose.

    The base object pose is read at planning time from
    :attr:`AssembleAffordance.base_object_entity`, and the assemble object's
    target pose is ``base_pose @ assemble_to_base_pose``. The held-object
    transform (``object_to_eef``) is read from :class:`PlanningContext`
    for the place control part, which a prior :class:`PickUp` populates.
    """

    goal_kind: ClassVar[str] = "assemble"

    affordance: AssembleAffordance
    """Assembly affordance anchoring the assemble object to the base object."""


@configclass
class PlaceCfg(ActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    control_part: str = "arm"
    """Manipulator resource used by this configured action instance."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper open interpolation phase."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    lift_height: float = 0.1
    """Height (m) to retract the end-effector after opening the gripper."""

    max_approach_retract_z: float | None = None
    """Optional maximum world-frame TCP z for approach and retract poses (m)."""

    cartesian_waypoint_count: int = 1
    """Number of fixed-orientation Cartesian keyframes per translation segment."""


class Place(AtomicAction[PlaceGoal | AssembleGoal]):
    """Lower the held object to a place pose, open the gripper, retract.

    The :class:`PlaceGoal` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    down phase visits every waypoint in order; approaching from above the
    first waypoint, descending through each waypoint, then opening the gripper
    at the final waypoint and retracting to above the last waypoint. Starting
    joint positions are inherited from :class:`PlanningContext`.

    An :class:`AssembleGoal` replaces the explicit EEF pose with an assembly
    affordance: the place pose is derived from the base object's current pose
    and ``assemble_to_base_pose``, converted to an EEF pose through the held
    object's ``object_to_eef`` (read from :class:`PlanningContext`).
    """

    skill_id: ClassVar[str] = "place"
    GoalType: ClassVar[type | tuple[type, ...]] = (
        PlaceGoal,
        AssembleGoal,
    )
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        cfg: PlaceCfg | None = None,
    ) -> None:
        super().__init__(cfg or PlaceCfg())
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceCfg", ValueError)
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PlaceCfg", ValueError
            )
        if self.cfg.cartesian_waypoint_count < 1:
            logger.log_error("cartesian_waypoint_count must be at least 1.", ValueError)

    def _on_bind(self) -> None:
        """Resolve robot-dependent resources from the owning engine."""
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof
        assert self.cfg.hand_open_qpos is not None
        assert self.cfg.hand_close_qpos is not None
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def plan(
        self,
        invocation: ActionInvocation[PlaceGoal | AssembleGoal],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan approach, release, and retract without committing detachment."""
        target = self.require_goal(invocation)
        if invocation.binding.manipulator() != self.cfg.control_part:
            raise ValueError("Place manipulator binding does not match its config.")
        if invocation.binding.end_effector() != self.cfg.hand_control_part:
            raise ValueError("Place end-effector binding does not match its config.")
        state = context
        place_xpos = self._resolve_place_xpos(target, state)
        if place_xpos.dim() == 3:
            place_xpos = place_xpos.unsqueeze(1)

        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        if isinstance(target, PlaceGoal) and target.tcp_symmetry == "z_roll_180":
            place_xpos = self._select_tcp_symmetric_place_variant(
                place_xpos, start_arm_qpos
            )
        n_down, n_open, n_back = self.builder.split_three_phase(
            invocation.motion_policy.sample_count,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="back",
        )

        approach_xpos = self._lifted_pose(place_xpos[:, 0])
        retract_xpos = self._lifted_pose(place_xpos[:, -1])

        start_xpos = self.robot.compute_fk(
            qpos=start_arm_qpos,
            name=self.cfg.control_part,
            to_matrix=True,
        )
        down_xpos = torch.cat([approach_xpos.unsqueeze(1), place_xpos], dim=1)
        down_xpos = self._translation_keyframes(start_xpos, down_xpos)

        target_states_list = [
            [
                PlanState(xpos=down_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(down_xpos.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        down_success, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=invocation.motion_policy,
        )
        reach_arm_qpos = down_arm[:, -1, :]

        back_xpos = self._translation_keyframes(
            place_xpos[:, -1], retract_xpos.unsqueeze(1)
        )
        target_states_list = [
            [
                PlanState(xpos=back_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(back_xpos.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        back_success, back_arm = self.builder.plan_arm_traj(
            target_states_list,
            reach_arm_qpos,
            n_back,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=invocation.motion_policy,
        )
        success = down_success & back_success

        hand_open_path = self.builder.interpolate_hand_qpos(
            self.hand_close_qpos, self.hand_open_qpos, n_waypoints=n_open
        )

        # Allocate from the actually-returned phase lengths so collision-aware
        # planners (which preserve their own sample count) are accommodated.
        n_down_actual = down_arm.shape[1]
        n_back_actual = back_arm.shape[1]
        full = torch.empty(
            (self.n_envs, n_down_actual + n_open + n_back_actual, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down_actual, self.arm_joint_ids] = down_arm
        full[:, :n_down_actual, self.hand_joint_ids] = self.hand_close_qpos
        full[:, n_down_actual : n_down_actual + n_open, self.arm_joint_ids] = (
            reach_arm_qpos.unsqueeze(1)
        )
        full[:, n_down_actual : n_down_actual + n_open, self.hand_joint_ids] = (
            hand_open_path
        )
        full[:, n_down_actual + n_open :, self.arm_joint_ids] = back_arm
        full[:, n_down_actual + n_open :, self.hand_joint_ids] = self.hand_open_qpos

        coordinated_updates = {
            key: None
            for key in state.coordinated_held_objects
            if self.cfg.control_part in key
        }
        return self.build_plan(
            invocation,
            context,
            success=success,
            trajectory=full,
            expected_effects=StateDelta(
                held_object_updates={self.cfg.control_part: None},
                coordinated_held_object_updates=coordinated_updates,
            ),
            phase_name="place",
        )

    def _resolve_place_xpos(
        self, target: PlaceGoal | AssembleGoal, state: PlanningContext
    ) -> torch.Tensor:
        """Resolve the place EEF poses from a typed target.

        Args:
            target: Either an explicit EEF pose target or an assembly target.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(n_envs, 4, 4)`` or
            ``(n_envs, n_waypoint, 4, 4)``.
        """
        if isinstance(target, PlaceGoal):
            return self.builder.resolve_pose_target(
                resolve_pose_goal(target.xpos, state, name="xpos"),
                n_envs=self.n_envs,
            )
        return self._resolve_assemble_place_xpos(target, state)

    def _resolve_assemble_place_xpos(
        self, target: AssembleGoal, state: PlanningContext
    ) -> torch.Tensor:
        """Derive the place EEF pose from an assembly affordance.

        The assemble object target pose is ``base_pose @ assemble_to_base_pose``;
        the EEF pose is that target posed through the held object's
        ``object_to_eef``.

        Args:
            target: Assembly target carrying the base/assemble affordance.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(n_envs, 4, 4)``.

        Raises:
            ValueError: If no held object or no base object entity is available.
        """
        held = state.get_held_object(self.cfg.control_part)
        if held is None:
            logger.log_error(
                "Place with AssembleGoal requires an object held by control "
                f"part {self.cfg.control_part!r} (run PickUp first).",
                ValueError,
            )
        affordance = target.affordance
        if affordance.base_object_entity is None:
            logger.log_error(
                "AssembleAffordance.base_object_entity must be set to assemble "
                "onto a base object.",
                ValueError,
            )
        base_pose = affordance.base_object_entity.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        assemble_object_pose = affordance.get_assemble_object_pose(base_pose)
        object_to_eef = resolve_object_target(
            held.object_to_eef,
            n_envs=self.n_envs,
            device=self.device,
            name="object_to_eef",
        )
        return torch.bmm(assemble_object_pose, object_to_eef)

    def _lifted_pose(self, release_xpos: torch.Tensor) -> torch.Tensor:
        """Build an above-release pose while respecting the optional TCP z cap."""
        lifted_xpos = release_xpos.clone()
        lifted_z = release_xpos[:, 2, 3] + self.cfg.lift_height
        if self.cfg.max_approach_retract_z is not None:
            max_z = torch.as_tensor(
                self.cfg.max_approach_retract_z,
                dtype=release_xpos.dtype,
                device=release_xpos.device,
            )
            lifted_z = torch.maximum(
                release_xpos[:, 2, 3],
                torch.clamp_max(lifted_z, max_z),
            )
        lifted_xpos[:, 2, 3] = lifted_z
        return lifted_xpos

    def _translation_keyframes(
        self, start_xpos: torch.Tensor, target_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate translations while holding each segment's target rotation."""
        count = self.cfg.cartesian_waypoint_count
        if count == 1:
            return target_xpos

        segment_starts = torch.cat(
            [start_xpos.unsqueeze(1), target_xpos[:, :-1]], dim=1
        )
        alpha = torch.linspace(
            1.0 / count,
            1.0,
            count,
            dtype=target_xpos.dtype,
            device=self.device,
        )
        keyframes = target_xpos.unsqueeze(2).repeat(1, 1, count, 1, 1)
        start_position = segment_starts[..., :3, 3].unsqueeze(2)
        target_position = target_xpos[..., :3, 3].unsqueeze(2)
        keyframes[..., :3, 3] = start_position + alpha[None, None, :, None] * (
            target_position - start_position
        )
        return keyframes.flatten(1, 2)

    def _select_tcp_symmetric_place_variant(
        self, place_xpos: torch.Tensor, start_qpos: torch.Tensor
    ) -> torch.Tensor:
        """Choose the closest TCP z-roll variant for an opt-in place target."""
        mirrored_place_xpos = place_xpos.clone()
        mirrored_place_xpos[..., :3, 0] = -mirrored_place_xpos[..., :3, 0]
        mirrored_place_xpos[..., :3, 1] = -mirrored_place_xpos[..., :3, 1]
        place_variants = torch.stack([place_xpos, mirrored_place_xpos], dim=2)

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=self.cfg.control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        first_waypoint_quat = quat_from_matrix(place_variants[:, 0, :, :3, :3])
        start_quat = start_quat[:, None, :].expand_as(first_waypoint_quat)
        rotation_error = quat_error_magnitude(
            first_waypoint_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(self.n_envs, 2)
        best_variant_idx = rotation_error.argmin(dim=1)

        env_idx = torch.arange(self.n_envs, device=self.device)[:, None]
        waypoint_idx = torch.arange(place_xpos.shape[1], device=self.device)[None, :]
        return place_variants[
            env_idx,
            waypoint_idx,
            best_variant_idx[:, None],
        ]


__all__ = ["AssembleGoal", "Place", "PlaceCfg", "PlaceGoal"]
