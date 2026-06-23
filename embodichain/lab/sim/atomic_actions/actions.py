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

"""Concrete atomic actions built on :class:`AtomicAction` and :class:`TrajectoryBuilder`.

Six sibling actions live here: :class:`MoveEndEffector`, :class:`MoveJoints`,
:class:`PickUp`, :class:`MoveHeldObject`, :class:`Place`, and :class:`Press`.
Each inherits :class:`AtomicAction` directly and composes a
:class:`TrajectoryBuilder` for shared trajectory math. ``execute`` takes a typed
target plus a
:class:`WorldState` and returns an :class:`ActionResult` whose trajectory is
full-robot DoF shaped ``(n_envs, n_waypoints, robot.dof)``.
"""

from __future__ import annotations

import torch
from typing import ClassVar

from embodichain.lab.sim.planners import PlanState, MoveType
from embodichain.utils import configclass, logger
from embodichain.utils.math import pose_inv

from .affordance import AntipodalAffordance
from .core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    HeldObjectPoseTarget,
    JointPositionTarget,
    NamedJointPositionTarget,
    ObjectSemantics,
    EndEffectorPoseTarget,
    WorldState,
)
from .trajectory import TrajectoryBuilder

# =============================================================================
# Cfg classes (flat — no inheritance among action configs)
# =============================================================================


@configclass
class MoveEndEffectorCfg(ActionCfg):
    name: str = "move_end_effector"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""


@configclass
class MoveJointsCfg(ActionCfg):
    name: str = "move_joints"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the interpolated joint-space trajectory."""

    named_joint_positions: dict[str, torch.Tensor] | None = None
    """Optional named joint targets resolved by ``NamedJointPositionTarget``."""


@configclass
class PickUpCfg(ActionCfg):
    name: str = "pick_up"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (approach + hand + lift)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper close interpolation phase."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after closing the gripper."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """Approach direction in the object local frame."""


@configclass
class MoveHeldObjectCfg(ActionCfg):
    name: str = "move_held_object"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""


@configclass
class PlaceCfg(ActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (down + hand + back)."""

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


@configclass
class PressCfg(ActionCfg):
    name: str = "press"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (hand close + down + back)."""

    hand_interp_steps: int = 5
    """Number of waypoints for closing the gripper before pressing."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""


# =============================================================================
# Shared helpers private to this module
# =============================================================================


def _resolve_object_target(
    target: torch.Tensor, *, n_envs: int, device: torch.device
) -> torch.Tensor:
    """Broadcast an object target pose to ``(n_envs, 4, 4)`` or validate it."""
    target = target.to(device=device, dtype=torch.float32)
    if target.shape == (4, 4):
        target = target.unsqueeze(0).repeat(n_envs, 1, 1)
    if target.shape != (n_envs, 4, 4):
        logger.log_error(
            f"object_target_pose must be (4, 4) or ({n_envs}, 4, 4), but got {target.shape}",
            ValueError,
        )
    return target


def _arm_qpos_from_state(
    state: WorldState, arm_joint_ids, robot_dof: int
) -> torch.Tensor:
    """Extract the arm slice of the full-DoF last_qpos carried in WorldState."""
    return state.last_qpos[:, arm_joint_ids]


# =============================================================================
# MoveEndEffector
# =============================================================================


class MoveEndEffector(AtomicAction):
    """Plan a free-space end-effector move to a target pose.

    The :class:`EndEffectorPoseTarget` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    action plans a single trajectory that visits every waypoint in order,
    starting from the inherited ``WorldState.last_qpos`` — IK is solved for each
    waypoint with the previous waypoint's solution as the seed.
    """

    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveEndEffectorCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveEndEffectorCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        move_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        target_states_list = self._build_target_states(move_xpos)
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return self._fail(state)
        full = self._embed(arm_traj, state.last_qpos)
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(), held_object=state.held_object
            ),
        )

    def _build_target_states(self, move_xpos: torch.Tensor) -> list[list[PlanState]]:
        """Build per-env PlanState lists from a single- or multi-waypoint target.

        ``move_xpos`` is the resolved target: 3D ``(n_envs, 4, 4)`` for a single
        waypoint or 4D ``(n_envs, n_waypoint, 4, 4)`` for a trajectory.
        """
        if move_xpos.dim() == 3:
            move_xpos = move_xpos.unsqueeze(1)
        n_waypoint = move_xpos.shape[1]
        return [
            [
                PlanState(xpos=move_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(n_waypoint)
            ]
            for i in range(self.n_envs)
        ]

    def _embed(
        self, arm_traj: torch.Tensor, last_full_qpos: torch.Tensor
    ) -> torch.Tensor:
        n_wp = arm_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_wp, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_full_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        return full

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


# =============================================================================
# MoveJoints
# =============================================================================


class MoveJoints(AtomicAction):
    """Plan a joint-space move for the configured control part.

    The :class:`JointPositionTarget` may carry either a single waypoint
    ``(n_envs, control_dof)`` or a multi-waypoint trajectory
    ``(n_envs, n_waypoint, control_dof)``. In the multi-waypoint case the
    action plans a single trajectory that visits every waypoint in order,
    starting from the inherited ``WorldState.last_qpos``.
    """

    TargetType: ClassVar[tuple[type, ...]] = (
        JointPositionTarget,
        NamedJointPositionTarget,
    )

    def __init__(
        self,
        motion_generator,
        cfg: MoveJointsCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveJointsCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.joint_dof = len(self.joint_ids)
        self.robot_dof = self.robot.dof
        self.named_joint_positions = self.cfg.named_joint_positions or {}

    def execute(
        self,
        target: JointPositionTarget | NamedJointPositionTarget,
        state: WorldState,
    ) -> ActionResult:
        target_qpos = self.builder.resolve_joint_target(
            self._resolve_target_qpos(target),
            n_envs=self.n_envs,
            joint_dof=self.joint_dof,
            control_part=self.cfg.control_part,
        )
        start_qpos = self.builder.resolve_start_qpos(
            state.last_qpos[:, self.joint_ids],
            n_envs=self.n_envs,
            arm_dof=self.joint_dof,
            control_part=self.cfg.control_part,
        )
        joint_traj = self.builder.plan_joint_traj(
            start_qpos, target_qpos, self.cfg.sample_interval
        )
        full = self._embed(joint_traj, state.last_qpos)
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(), held_object=state.held_object
            ),
        )

    def _resolve_target_qpos(
        self, target: JointPositionTarget | NamedJointPositionTarget
    ) -> torch.Tensor:
        if isinstance(target, JointPositionTarget):
            return target.qpos
        if target.name not in self.named_joint_positions:
            logger.log_error(
                f"Unknown named joint-position target '{target.name}' for "
                f"MoveJoints. Available targets: {sorted(self.named_joint_positions)}",
                KeyError,
            )
        return self.named_joint_positions[target.name]

    def _embed(
        self, joint_traj: torch.Tensor, last_full_qpos: torch.Tensor
    ) -> torch.Tensor:
        n_wp = joint_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_wp, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_full_qpos.unsqueeze(1)
        full[:, :, self.joint_ids] = joint_traj
        return full


# =============================================================================
# PickUp
# =============================================================================


class PickUp(AtomicAction):
    """Approach a grasp pose, close the gripper, lift."""

    TargetType: ClassVar[type] = GraspTarget

    def __init__(
        self,
        motion_generator,
        cfg: PickUpCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PickUpCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error(
                "hand_open_qpos must be specified in PickUpCfg", ValueError
            )
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PickUpCfg", ValueError
            )
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        self.approach_direction = self.cfg.approach_direction.to(self.device)

    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult:
        sem = target.semantics
        if not isinstance(sem.affordance, AntipodalAffordance):
            logger.log_error(
                "PickUp requires an AntipodalAffordance on the target semantics.",
                ValueError,
            )
        if sem.entity is None:
            logger.log_error(
                "PickUp requires an entity on the target semantics.", ValueError
            )

        is_success, grasp_xpos = self._resolve_grasp_pose(sem)
        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self._fail(state)

        # Pre-grasp by offsetting backwards along grasp z.
        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
        )

        start_arm_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        n_approach, n_close, n_lift = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        # Phase 1: approach
        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        ok, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUp failed to plan the approach trajectory.")
            return self._fail(state)

        # Phase 3: lift (planned from grasp qpos)
        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUp failed to plan the lift trajectory.")
            return self._fail(state)

        # Phase 2: hand close (arm held at grasp qpos)
        hand_close_path = self.builder.interpolate_hand_qpos(
            self.hand_open_qpos, self.hand_close_qpos, n_waypoints=n_close
        )

        full = torch.empty(
            (self.n_envs, n_approach + n_close + n_lift, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_approach, self.arm_joint_ids] = approach_arm
        full[:, :n_approach, self.hand_joint_ids] = self.hand_open_qpos
        full[:, n_approach : n_approach + n_close, self.arm_joint_ids] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full[:, n_approach : n_approach + n_close, self.hand_joint_ids] = (
            hand_close_path
        )
        full[:, n_approach + n_close :, self.arm_joint_ids] = lift_arm
        full[:, n_approach + n_close :, self.hand_joint_ids] = self.hand_close_qpos

        obj_poses = sem.entity.get_local_pose(to_matrix=True)
        object_to_eef = torch.bmm(pose_inv(obj_poses), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(last_qpos=full[:, -1, :].clone(), held_object=held),
        )

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )
        n_envs = obj_poses.shape[0]
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        grasp_xpos_padding = torch.zeros(
            (n_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (n_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(n_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            grasp_xpos_padding[i, :n_pose] = grasp_poses_result[i][0]
            grasp_cost_padding[i, :n_pose] = grasp_poses_result[i][1]
            grasp_xpos_padding[i, n_pose:] = grasp_poses_result[i][0][0]
            grasp_cost_padding[i, n_pose:] = grasp_poses_result[i][1][0]
        init_qpos_repeat = init_qpos[:, None, :].repeat(1, n_max_pose, 1)
        ik_success, _ = self.robot.compute_batch_ik(
            pose=grasp_xpos_padding,
            name=self.cfg.control_part,
            joint_seed=init_qpos_repeat,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos


# =============================================================================
# MoveHeldObject
# =============================================================================


class MoveHeldObject(AtomicAction):
    """Move the held object to a target object pose; keep the gripper closed."""

    TargetType: ClassVar[type] = HeldObjectPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveHeldObjectCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveHeldObjectCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in MoveHeldObjectCfg", ValueError
            )
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def execute(self, target: HeldObjectPoseTarget, state: WorldState) -> ActionResult:
        if state.held_object is None:
            logger.log_error(
                "MoveHeldObject requires WorldState.held_object — run PickUp first.",
                ValueError,
            )
        object_target_pose = _resolve_object_target(
            target.object_target_pose, n_envs=self.n_envs, device=self.device
        )
        object_to_eef = state.held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

        start_arm_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        target_states_list = [
            [PlanState(xpos=move_eef_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("MoveHeldObject failed to plan trajectory.")
            return self._fail(state)

        full = torch.empty(
            (self.n_envs, arm_traj.shape[1], self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        full[:, :, self.hand_joint_ids] = self.hand_close_qpos

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
            ),
        )

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


# =============================================================================
# Place
# =============================================================================


class Place(AtomicAction):
    """Lower the held object to a place pose, open the gripper, retract.

    The :class:`EndEffectorPoseTarget` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    down phase visits every waypoint in order — approaching from above the
    first waypoint, descending through each waypoint, then opening the gripper
    at the final waypoint and retracting to above the last waypoint. Starting
    joint positions are inherited from ``WorldState.last_qpos``.
    """

    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: PlaceCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PlaceCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceCfg", ValueError)
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PlaceCfg", ValueError
            )
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        place_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        # Normalize a single-waypoint (n_envs, 4, 4) target to (n_envs, 1, 4, 4)
        # so the multi-waypoint descent path below is uniform.
        if place_xpos.dim() == 3:
            place_xpos = place_xpos.unsqueeze(1)
        n_waypoint = place_xpos.shape[1]

        start_arm_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        n_down, n_open, n_back = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="back",
        )

        lift_offset = torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height
        # Approach from above the first waypoint; retract to above the last.
        # For a single waypoint these coincide, matching the legacy behavior.
        approach_xpos = self.builder.apply_local_offset(place_xpos[:, 0], lift_offset)
        retract_xpos = self.builder.apply_local_offset(place_xpos[:, -1], lift_offset)

        # Phase 1: down (approach → every place waypoint in order)
        target_states_list = [
            [PlanState(xpos=approach_xpos[i], move_type=MoveType.EEF_MOVE)]
            + [
                PlanState(xpos=place_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(n_waypoint)
            ]
            for i in range(self.n_envs)
        ]
        ok, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return self._fail(state)
        reach_arm_qpos = down_arm[:, -1, :]

        # Phase 3: back (retract to above the last waypoint)
        target_states_list = [
            [PlanState(xpos=retract_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, back_arm = self.builder.plan_arm_traj(
            target_states_list,
            reach_arm_qpos,
            n_back,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return self._fail(state)

        # Phase 2: hand open (arm held at reach qpos)
        hand_open_path = self.builder.interpolate_hand_qpos(
            self.hand_close_qpos, self.hand_open_qpos, n_waypoints=n_open
        )

        full = torch.empty(
            (self.n_envs, n_down + n_open + n_back, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down, self.arm_joint_ids] = down_arm
        full[:, :n_down, self.hand_joint_ids] = self.hand_close_qpos
        full[:, n_down : n_down + n_open, self.arm_joint_ids] = (
            reach_arm_qpos.unsqueeze(1)
        )
        full[:, n_down : n_down + n_open, self.hand_joint_ids] = hand_open_path
        full[:, n_down + n_open :, self.arm_joint_ids] = back_arm
        full[:, n_down + n_open :, self.hand_joint_ids] = self.hand_open_qpos

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(last_qpos=full[:, -1, :].clone(), held_object=None),
        )

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


# =============================================================================
# Press
# =============================================================================


class Press(AtomicAction):
    """Close the gripper, press down to a target pose, then return."""

    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: PressCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PressCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PressCfg", ValueError
            )
        self.hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.hand_dof,
        )

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        press_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_arm_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        start_hand_qpos = state.last_qpos[:, self.hand_joint_ids]

        n_close, n_down, n_back = self._compute_phase_waypoints()

        # Phase 1: close the hand while holding the current EEF pose.
        hand_close_path = self.builder.interpolate_hand_qpos(
            start_hand_qpos,
            self.hand_close_qpos,
            n_waypoints=n_close,
        )

        # Phase 2: press down to the target pose.
        target_states_list = [
            [PlanState(xpos=press_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("Press failed to plan the down trajectory.")
            return self._fail(state)

        # Phase 3: return to the arm pose from before pressing.
        press_arm_qpos = down_arm[:, -1, :]
        back_arm = self.builder.plan_joint_traj(press_arm_qpos, start_arm_qpos, n_back)

        full = torch.empty(
            (self.n_envs, n_close + n_down + n_back, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_close, self.arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, :n_close, self.hand_joint_ids] = hand_close_path
        full[:, n_close : n_close + n_down, self.arm_joint_ids] = down_arm
        full[:, n_close : n_close + n_down, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )
        full[:, n_close + n_down :, self.arm_joint_ids] = back_arm
        full[:, n_close + n_down :, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
            ),
        )

    def _compute_phase_waypoints(self) -> tuple[int, int, int]:
        n_close = self.cfg.hand_interp_steps
        if n_close < 1:
            logger.log_error(
                "hand_interp_steps must be at least 1 for PressCfg.", ValueError
            )

        motion_waypoints = self.cfg.sample_interval - n_close
        n_down = motion_waypoints // 2
        n_back = motion_waypoints - n_down
        if n_down < 2 or n_back < 2:
            logger.log_error(
                "Not enough waypoints for press trajectory. Increase "
                "sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return n_close, n_down, n_back

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


__all__ = [
    "MoveEndEffector",
    "MoveEndEffectorCfg",
    "MoveJoints",
    "MoveJointsCfg",
    "MoveHeldObject",
    "MoveHeldObjectCfg",
    "PickUp",
    "PickUpCfg",
    "Place",
    "PlaceCfg",
    "Press",
    "PressCfg",
]
