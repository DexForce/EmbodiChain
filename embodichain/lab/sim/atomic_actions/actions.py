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
:class:`PickUp`, :class:`MoveHeldObject`, :class:`Place`, and
:class:`CoordinatedPlacement`. Each inherits
:class:`AtomicAction` directly and composes a :class:`TrajectoryBuilder` for
shared trajectory math. ``execute`` takes a typed target plus a
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
    CoordinatedPlacementTarget,
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
class CoordinatedPlacementCfg(ActionCfg):
    name: str = "coordinated_placement"
    """Name of the action, used for identification and logging."""

    control_part: str = "dual_arm"
    """Robot control part containing both placing and support arms."""

    placing_arm_control_part: str = "left_arm"
    """Arm that places and releases its held object."""

    support_arm_control_part: str = "right_arm"
    """Arm that moves the support object and keeps holding it."""

    placing_hand_control_part: str = "left_hand"
    """Hand attached to the placing arm."""

    support_hand_control_part: str = "right_hand"
    """Hand attached to the support arm."""

    placing_hand_open_qpos: torch.Tensor | None = None
    """Placing-hand qpos for the open state, shape ``[hand_dof,]``."""

    placing_hand_close_qpos: torch.Tensor | None = None
    """Placing-hand qpos for the closed state, shape ``[hand_dof,]``."""

    support_hand_close_qpos: torch.Tensor | None = None
    """Support-hand qpos for the closed state, shape ``[hand_dof,]``."""

    release: bool = True
    """Whether to open the placing hand at the aligned placement pose."""

    placing_height_offset: float = 0.0
    """Default World-Z offset above the placing object target pose."""

    support_height_offset: float = 0.0
    """Default World-Z offset above the support object target pose."""

    lift_height: float = 0.08
    """World-Z lift distance for the placing arm after release."""

    sample_interval: int = 100
    """Number of waypoints for the full coordinated placement trajectory."""

    hand_interp_steps: int = 10
    """Number of waypoints for the placing-hand release interpolation."""

    hold_steps: int = 4
    """Number of waypoints to hold alignment before releasing."""

    retreat_steps: int = 16
    """Number of waypoints used for the placing-arm lift retreat."""


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
    """Plan a free-space end-effector move to a target pose."""

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
        target_states_list = [
            [PlanState(xpos=move_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
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
    """Plan a joint-space move for the configured control part."""

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
    """Lower the held object to a place pose, open the gripper, retract."""

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

        lift_xpos = self.builder.apply_local_offset(
            place_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )

        # Phase 1: down (lift → place)
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=place_xpos[i], move_type=MoveType.EEF_MOVE),
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

        # Phase 3: back (retract to lift)
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
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
# CoordinatedPlacement
# =============================================================================


class CoordinatedPlacement(AtomicAction):
    """Coordinate two held objects: support object below, placing object above."""

    TargetType: ClassVar[type] = CoordinatedPlacementTarget

    def __init__(
        self,
        motion_generator,
        cfg: CoordinatedPlacementCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or CoordinatedPlacementCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

        self.dual_arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.placing_arm_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.placing_arm_control_part
        )
        self.support_arm_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.support_arm_control_part
        )
        self.placing_hand_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.placing_hand_control_part
        )
        self.support_hand_joint_ids = self.robot.get_joint_ids(
            name=self.cfg.support_hand_control_part
        )
        self.joint_ids = (
            self.dual_arm_joint_ids
            + self.placing_hand_joint_ids
            + self.support_hand_joint_ids
        )
        self.placing_arm_dof = len(self.placing_arm_joint_ids)
        self.support_arm_dof = len(self.support_arm_joint_ids)
        self.placing_hand_dof = len(self.placing_hand_joint_ids)
        self.support_hand_dof = len(self.support_hand_joint_ids)

        self._validate_hand_qpos_cfg()
        self.placing_hand_open_qpos = self.builder.expand_hand_qpos(
            self.cfg.placing_hand_open_qpos,
            n_envs=self.n_envs,
            hand_dof=self.placing_hand_dof,
        )
        self.placing_hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.placing_hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.placing_hand_dof,
        )
        self.support_hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.support_hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.support_hand_dof,
        )

    def execute(
        self, target: CoordinatedPlacementTarget, state: WorldState
    ) -> ActionResult:
        placing_xpos, support_xpos, release, support_held_object = self._resolve_target(
            target
        )
        placing_start_qpos, support_start_qpos = self._resolve_start_qpos(state)
        segments = self._compute_segment_lengths(release)

        placing_lift_xpos = self.builder.apply_local_offset(
            placing_xpos,
            torch.tensor(
                [0.0, 0.0, self.cfg.lift_height],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        ok, placing_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.placing_arm_control_part,
            placing_start_qpos,
            torch.stack([placing_lift_xpos, placing_xpos], dim=1),
            segments["approach"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan placing approach.")
            return self._fail(state)

        ok, support_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.support_arm_control_part,
            support_start_qpos,
            support_xpos.unsqueeze(1),
            segments["approach"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan support approach.")
            return self._fail(state)

        placing_place_qpos = placing_approach_traj[:, -1]
        support_place_qpos = support_approach_traj[:, -1]
        approach_trajectory = self._assemble_phase(
            state.last_qpos,
            placing_approach_traj,
            support_approach_traj,
            self._repeat_qpos(self.placing_hand_close_qpos, segments["approach"]),
            self._repeat_qpos(self.support_hand_close_qpos, segments["approach"]),
        )

        hold_trajectory = self._empty_phase(state)
        if segments["hold"] > 0:
            hold_trajectory = self._assemble_phase(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["hold"]),
                self._repeat_qpos(support_place_qpos, segments["hold"]),
                self._repeat_qpos(self.placing_hand_close_qpos, segments["hold"]),
                self._repeat_qpos(self.support_hand_close_qpos, segments["hold"]),
            )

        release_trajectory = self._empty_phase(state)
        if release:
            release_trajectory = self._assemble_phase(
                state.last_qpos,
                self._repeat_qpos(placing_place_qpos, segments["release"]),
                self._repeat_qpos(support_place_qpos, segments["release"]),
                self.builder.interpolate_hand_qpos(
                    self.placing_hand_close_qpos,
                    self.placing_hand_open_qpos,
                    n_waypoints=segments["release"],
                ),
                self._repeat_qpos(self.support_hand_close_qpos, segments["release"]),
            )

        ok, placing_retreat_traj = self._plan_named_arm_trajectory(
            self.cfg.placing_arm_control_part,
            placing_place_qpos,
            placing_lift_xpos.unsqueeze(1),
            segments["retreat"],
        )
        if not ok:
            logger.log_warning("CoordinatedPlacement failed to plan placing retreat.")
            return self._fail(state)

        placing_hand_retreat_qpos = (
            self.placing_hand_open_qpos if release else self.placing_hand_close_qpos
        )
        retreat_trajectory = self._assemble_phase(
            state.last_qpos,
            placing_retreat_traj,
            self._repeat_qpos(support_place_qpos, segments["retreat"]),
            self._repeat_qpos(placing_hand_retreat_qpos, segments["retreat"]),
            self._repeat_qpos(self.support_hand_close_qpos, segments["retreat"]),
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
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=support_held_object,
            ),
        )

    def get_segment_lengths(self, release: bool | None = None) -> dict[str, int]:
        """Return waypoint counts for the coordinated placement phase sequence."""
        release = self.cfg.release if release is None else release
        return self._compute_segment_lengths(release)

    def _validate_hand_qpos_cfg(self) -> None:
        """Ensure all hand state tensors are provided."""
        required_names = (
            "placing_hand_open_qpos",
            "placing_hand_close_qpos",
            "support_hand_close_qpos",
        )
        for name in required_names:
            if getattr(self.cfg, name) is None:
                logger.log_error(
                    f"{name} must be specified in CoordinatedPlacementCfg",
                    ValueError,
                )

    def _resolve_object_pose(
        self,
        pose: torch.Tensor,
        height_offset: float,
        name: str,
    ) -> torch.Tensor:
        """Resolve an object target pose into a batched pose with height offset."""
        object_pose = _resolve_object_target(
            pose,
            n_envs=self.n_envs,
            device=self.device,
        )
        return self.builder.apply_local_offset(
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
        """Resolve a held-object transform into batched shape."""
        object_to_eef = held_state.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if object_to_eef.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"{name}.object_to_eef must have shape (4, 4) or "
                f"({self.n_envs}, 4, 4), but got {object_to_eef.shape}",
                ValueError,
            )
        return object_to_eef

    def _resolve_target(
        self,
        target: CoordinatedPlacementTarget,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, HeldObjectState]:
        """Resolve object-centric target into placing and support TCP poses."""
        placing_height_offset = (
            self.cfg.placing_height_offset
            if target.placing_height_offset is None
            else target.placing_height_offset
        )
        support_height_offset = (
            self.cfg.support_height_offset
            if target.support_height_offset is None
            else target.support_height_offset
        )
        placing_object_pose = self._resolve_object_pose(
            target.placing_object_target_pose,
            placing_height_offset,
            "placing_object_target_pose",
        )
        support_object_pose = self._resolve_object_pose(
            target.support_object_target_pose,
            support_height_offset,
            "support_object_target_pose",
        )
        placing_xpos = torch.bmm(
            placing_object_pose,
            self._resolve_object_to_eef(
                target.placing_held_object, "placing_held_object"
            ),
        )
        support_xpos = torch.bmm(
            support_object_pose,
            self._resolve_object_to_eef(
                target.support_held_object, "support_held_object"
            ),
        )
        release = self.cfg.release if target.release is None else target.release
        return placing_xpos, support_xpos, release, target.support_held_object

    def _resolve_start_qpos(
        self, state: WorldState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract per-arm start qpos from full-robot WorldState qpos."""
        if state.last_qpos.shape != (self.n_envs, self.robot_dof):
            logger.log_error(
                f"WorldState.last_qpos must have shape ({self.n_envs}, {self.robot_dof}), "
                f"but got {state.last_qpos.shape}",
                ValueError,
            )
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        return (
            start_qpos[:, self.placing_arm_joint_ids],
            start_qpos[:, self.support_arm_joint_ids],
        )

    def _compute_segment_lengths(self, release: bool) -> dict[str, int]:
        """Compute waypoint counts for coordinated placement phases."""
        n_release = max(2, self.cfg.hand_interp_steps) if release else 0
        n_hold = max(0, self.cfg.hold_steps)
        n_retreat = max(2, self.cfg.retreat_steps)
        n_approach = self.cfg.sample_interval - n_hold - n_release - n_retreat
        if n_approach < 2:
            logger.log_error(
                "Not enough waypoints for coordinated placement. Increase "
                "sample_interval or decrease hold/release/retreat steps.",
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
    ) -> tuple[bool, torch.Tensor]:
        """Plan a batched arm trajectory for a named control part."""
        target_states_list = [
            [
                PlanState(xpos=target_poses[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(target_poses.shape[1])
            ]
            for i in range(self.n_envs)
        ]
        return self.builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            n_waypoints,
            control_part=control_part,
            arm_dof=start_qpos.shape[-1],
        )

    @staticmethod
    def _repeat_qpos(qpos: torch.Tensor, n_waypoints: int) -> torch.Tensor:
        """Repeat batched qpos across waypoints."""
        return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    def _empty_phase(self, state: WorldState) -> torch.Tensor:
        """Return an empty full-DoF phase for optional segments."""
        return torch.empty(
            (self.n_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )

    def _assemble_phase(
        self,
        base_full_qpos: torch.Tensor,
        placing_arm_traj: torch.Tensor,
        support_arm_traj: torch.Tensor,
        placing_hand_traj: torch.Tensor,
        support_hand_traj: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble arm and hand trajectories into full-robot DoF order."""
        n_waypoints = placing_arm_traj.shape[1]
        full = base_full_qpos.to(device=self.device, dtype=torch.float32)
        full = full.unsqueeze(1).repeat(1, n_waypoints, 1).clone()
        full[:, :, self.placing_arm_joint_ids] = placing_arm_traj
        full[:, :, self.support_arm_joint_ids] = support_arm_traj
        full[:, :, self.placing_hand_joint_ids] = placing_hand_traj
        full[:, :, self.support_hand_joint_ids] = support_hand_traj
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


__all__ = [
    "CoordinatedPlacement",
    "CoordinatedPlacementCfg",
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
]
