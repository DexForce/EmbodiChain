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
from embodichain.utils.math import (
    matrix_from_quat,
    pose_inv,
    quat_from_matrix,
    axis_angle_to_rotation_matrix,
)
from .affordance import AntipodalAffordance
from .core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedHeldObjectState,
    CoordinatedPickmentTarget,
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

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction to align with world up after grasping. By dafault we will use (0, 0, 1)."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp y-axis to apply to the grasp pose"""


@configclass
class MoveHeldObjectCfg(ActionCfg):
    name: str = "move_held_object"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""

    pick_rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp y-axis to apply to the grasp pose"""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction to align with world up after grasping. By dafault we will use (0, 0, 1)."""

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
class CoordinatedPickmentCfg(ActionCfg):
    name: str = "coordinated_pickment"
    """Name of the action, used for identification and logging."""

    control_part: str = "dual_arm"
    """Combined control part containing left and right arm joints."""

    left_arm_control_part: str = "left_arm"
    """Left arm control part used to grasp one end of the object."""

    right_arm_control_part: str = "right_arm"
    """Right arm control part used to grasp the other end of the object."""

    left_hand_control_part: str = "left_hand"
    """Hand attached to the left arm."""

    right_hand_control_part: str = "right_hand"
    """Hand attached to the right arm."""

    left_hand_open_qpos: torch.Tensor | None = None
    """Left hand qpos for the open state."""

    left_hand_close_qpos: torch.Tensor | None = None
    """Left hand qpos for the closed state."""

    right_hand_open_qpos: torch.Tensor | None = None
    """Right hand qpos for the open state."""

    right_hand_close_qpos: torch.Tensor | None = None
    """Right hand qpos for the closed state."""

    object_motion_keyframes: int = 6
    """Number of object-pose keyframes solved by IK before joint-space interpolation."""

    pre_grasp_distance: float = 0.10
    """World distance to retreat from each grasp pose along negative TCP z."""

    lift_height: float = 0.08
    """World-Z lift distance before moving to the object target pose."""

    sample_interval: int = 120
    """Number of waypoints for the full coordinated pickment trajectory."""

    hand_interp_steps: int = 10
    """Number of waypoints used for the simultaneous hand close phase."""

    hold_steps: int = 4
    """Number of waypoints to hold the final object target pose."""


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


class _DualArmHelpers:
    """Shared trajectory helpers for dual-arm coordinated actions."""

    def _init_dual_arm_parts(
        self,
        *,
        first_arm_control_part: str,
        second_arm_control_part: str,
        first_hand_control_part: str,
        second_hand_control_part: str,
    ) -> None:
        self.builder = TrajectoryBuilder(self.motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof
        self.dual_arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.first_arm_joint_ids = self.robot.get_joint_ids(name=first_arm_control_part)
        self.second_arm_joint_ids = self.robot.get_joint_ids(
            name=second_arm_control_part
        )
        self.first_hand_joint_ids = self.robot.get_joint_ids(
            name=first_hand_control_part
        )
        self.second_hand_joint_ids = self.robot.get_joint_ids(
            name=second_hand_control_part
        )
        self.first_arm_dof = len(self.first_arm_joint_ids)
        self.second_arm_dof = len(self.second_arm_joint_ids)
        self.dual_arm_dof = len(self.dual_arm_joint_ids)
        self.first_hand_dof = len(self.first_hand_joint_ids)
        self.second_hand_dof = len(self.second_hand_joint_ids)
        self._dual_id_to_col = {
            joint_id: col for col, joint_id in enumerate(self.dual_arm_joint_ids)
        }
        self._first_arm_cols = self._lookup_joint_columns(
            self.first_arm_joint_ids,
            self._dual_id_to_col,
            first_arm_control_part,
        )
        self._second_arm_cols = self._lookup_joint_columns(
            self.second_arm_joint_ids,
            self._dual_id_to_col,
            second_arm_control_part,
        )

    @staticmethod
    def _lookup_joint_columns(
        joint_ids: list[int],
        joint_id_to_col: dict[int, int],
        control_part: str,
    ) -> list[int]:
        """Map global joint ids into local trajectory columns."""
        missing = [
            joint_id for joint_id in joint_ids if joint_id not in joint_id_to_col
        ]
        if missing:
            logger.log_error(
                f"Joints {missing} from '{control_part}' are not included in "
                "the configured dual-arm control part.",
                ValueError,
            )
        return [joint_id_to_col[joint_id] for joint_id in joint_ids]

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

    def _expand_qpos(self, qpos: torch.Tensor, dof: int, name: str) -> torch.Tensor:
        """Resolve qpos to batched shape ``(n_envs, dof)``."""
        qpos = qpos.to(device=self.device, dtype=torch.float32)
        if qpos.shape == (dof,):
            return qpos.unsqueeze(0).repeat(self.n_envs, 1)
        if qpos.shape == (self.n_envs, dof):
            return qpos
        logger.log_error(
            f"{name} must have shape ({dof},) or "
            f"({self.n_envs}, {dof}), but got {qpos.shape}",
            ValueError,
        )
        raise AssertionError("unreachable")

    def _resolve_pose(self, pose: torch.Tensor, name: str) -> torch.Tensor:
        """Resolve a pose tensor into batched shape ``(n_envs, 4, 4)``."""
        pose = pose.to(device=self.device, dtype=torch.float32)
        if pose.shape == (4, 4):
            pose = pose.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if pose.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"{name} must have shape (4, 4) or "
                f"({self.n_envs}, 4, 4), but got {pose.shape}",
                ValueError,
            )
        return pose

    def _resolve_dual_arm_start(
        self,
        state: WorldState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve full-robot state into the two arm qpos tensors."""
        dual_start = state.last_qpos[:, self.dual_arm_joint_ids].to(
            device=self.device, dtype=torch.float32
        )
        return (
            dual_start[:, self._first_arm_cols],
            dual_start[:, self._second_arm_cols],
        )

    def _plan_named_arm_trajectory(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
    ) -> tuple[bool, torch.Tensor]:
        """Plan a batched arm trajectory for a named control part."""
        n_state = target_poses.shape[1]
        arm_dof = start_qpos.shape[-1]
        trajectory = torch.zeros(
            (self.n_envs, n_state, arm_dof),
            dtype=torch.float32,
            device=self.device,
        )
        qpos_seed = start_qpos
        for i in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=target_poses[:, i],
                name=control_part,
                joint_seed=qpos_seed,
            )
            if not self.builder.all_envs_success(is_success):
                logger.log_warning(
                    f"Failed to compute IK for {control_part} target state {i}."
                )
                return False, trajectory
            trajectory[:, i] = qpos
            qpos_seed = qpos

        trajectory = torch.cat([start_qpos.unsqueeze(1), trajectory], dim=1)
        return True, (
            self.builder.plan_joint_traj(
                trajectory[:, 0],
                trajectory[:, -1],
                n_waypoints,
            )
            if n_state == 1
            else self._interpolate_keyframe_qpos(trajectory, n_waypoints)
        )

    def _compose_dual_arm_trajectory(
        self,
        first_arm_traj: torch.Tensor,
        second_arm_traj: torch.Tensor,
    ) -> torch.Tensor:
        """Compose first and second arm trajectories in dual-arm joint order."""
        n_waypoints = first_arm_traj.shape[1]
        dual_arm_traj = torch.zeros(
            (self.n_envs, n_waypoints, self.dual_arm_dof),
            dtype=torch.float32,
            device=self.device,
        )
        dual_arm_traj[:, :, self._first_arm_cols] = first_arm_traj
        dual_arm_traj[:, :, self._second_arm_cols] = second_arm_traj
        return dual_arm_traj

    def _assemble_phase(
        self,
        state: WorldState,
        first_arm_traj: torch.Tensor,
        second_arm_traj: torch.Tensor,
        first_hand_traj: torch.Tensor,
        second_hand_traj: torch.Tensor,
    ) -> torch.Tensor:
        """Embed dual-arm and hand trajectories into full robot DoF order."""
        n_waypoints = first_arm_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_waypoints, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.to(self.device).unsqueeze(1)
        full[:, :, self.dual_arm_joint_ids] = self._compose_dual_arm_trajectory(
            first_arm_traj, second_arm_traj
        )
        full[:, :, self.first_hand_joint_ids] = first_hand_traj
        full[:, :, self.second_hand_joint_ids] = second_hand_traj
        return full

    @staticmethod
    def _repeat_qpos(qpos: torch.Tensor, n_waypoints: int) -> torch.Tensor:
        """Repeat batched qpos across waypoints."""
        return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    def _interpolate_qpos(
        self,
        start_qpos: torch.Tensor,
        end_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate batched qpos between two states."""
        weights = torch.linspace(
            0.0,
            1.0,
            steps=n_waypoints,
            device=self.device,
            dtype=start_qpos.dtype,
        )
        return torch.lerp(
            start_qpos.unsqueeze(1),
            end_qpos.unsqueeze(1),
            weights[None, :, None],
        )

    def _interpolate_keyframe_qpos(
        self, keyframe_qpos: torch.Tensor, n_waypoints: int
    ) -> torch.Tensor:
        """Interpolate a sequence of qpos keyframes into ``n_waypoints`` samples."""
        n_keyframes = keyframe_qpos.shape[1]
        keyframe_indices = (
            torch.linspace(
                0,
                n_waypoints - 1,
                steps=n_keyframes,
                device=self.device,
            )
            .round()
            .to(dtype=torch.long)
        )
        return self._interpolate_qpos_keyframes(
            keyframe_qpos, keyframe_indices, n_waypoints
        )

    def _interpolate_qpos_keyframes(
        self,
        keyframe_qpos: torch.Tensor,
        keyframe_indices: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate qpos keyframes using shared waypoint indices."""
        trajectory = torch.zeros(
            (self.n_envs, n_waypoints, keyframe_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        for segment_idx in range(len(keyframe_indices) - 1):
            start_idx = int(keyframe_indices[segment_idx].item())
            end_idx = int(keyframe_indices[segment_idx + 1].item())
            n_segment = end_idx - start_idx + 1
            weights = torch.linspace(
                0.0,
                1.0,
                steps=n_segment,
                dtype=keyframe_qpos.dtype,
                device=self.device,
            )
            segment = torch.lerp(
                keyframe_qpos[:, segment_idx : segment_idx + 1],
                keyframe_qpos[:, segment_idx + 1 : segment_idx + 2],
                weights[None, :, None],
            )
            trajectory[:, start_idx : end_idx + 1] = segment
        return trajectory

    def _interpolate_object_pose(
        self,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        n_waypoints: int,
        *,
        include_orientation: bool,
    ) -> torch.Tensor:
        """Interpolate object translation and optionally orientation."""
        weights = torch.linspace(
            0.0,
            1.0,
            steps=n_waypoints,
            device=self.device,
            dtype=start_pose.dtype,
        )
        poses = start_pose.unsqueeze(1).repeat(1, n_waypoints, 1, 1)
        poses[:, :, :3, 3] = torch.lerp(
            start_pose[:, None, :3, 3],
            end_pose[:, None, :3, 3],
            weights[None, :, None],
        )
        if not include_orientation:
            return poses

        start_quat = quat_from_matrix(start_pose[:, :3, :3])
        end_quat = quat_from_matrix(end_pose[:, :3, :3])
        quat_dot = torch.sum(start_quat * end_quat, dim=-1, keepdim=True)
        end_quat = torch.where(quat_dot < 0.0, -end_quat, end_quat)
        quat = torch.lerp(
            start_quat.unsqueeze(1),
            end_quat.unsqueeze(1),
            weights[None, :, None],
        )
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
        poses[:, :, :3, :3] = matrix_from_quat(quat.reshape(-1, 4)).reshape(
            self.n_envs, n_waypoints, 3, 3
        )
        return poses


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
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
                coordinated_held_object=state.coordinated_held_object,
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
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
                coordinated_held_object=state.coordinated_held_object,
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

        # apply grasp yr rotation offset if specified
        if self.cfg.rotate_upright is not None:
            if self.cfg.obj_upright_direction is None:
                upright_direction = torch.tensor([0, 0, 1], device=self.device)
            else:
                upright_direction = self.cfg.obj_upright_direction.to(self.device)
            obj_pose = sem.entity.get_local_pose(to_matrix=True)
            obj_upright = (upright_direction * obj_pose[:, :3, :3]).sum(axis=2)
            grasp_ry = grasp_xpos[:, :3, 1]
            dot_result = (grasp_ry * obj_upright).sum(axis=1)
            # revert flag is -1 if the dot product is negative, 1 if positive
            revert_flag = torch.where(dot_result < 0, 1.0, -1.0)
            grasp_rx = grasp_xpos[:, :3, 0]
            rota_axis_angle = self.cfg.rotate_upright * revert_flag * grasp_rx
            rota_offset = axis_angle_to_rotation_matrix(rota_axis_angle)
            upright_grasp_rota = torch.bmm(rota_offset, grasp_xpos[:, :3, :3])
            grasp_xpos[:, :3, :3] = upright_grasp_rota

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
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=held,
                coordinated_held_object=state.coordinated_held_object,
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
        start_arm_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        if self.cfg.pick_rotate_upright is not None:
            held_eef_xpos = self.robot.compute_fk(
                qpos=start_arm_qpos, name=self.cfg.control_part, to_matrix=True
            )
            held_obj_xpos = state.held_object.semantics.entity.get_local_pose(
                to_matrix=True
            )
            if self.cfg.obj_upright_direction is None:
                upright_direction = torch.tensor([0, 0, 1], device=self.device)
            else:
                upright_direction = self.cfg.obj_upright_direction.to(self.device)
            obj_upright = (upright_direction * held_obj_xpos[:, :3, :3]).sum(axis=2)

            grasp_ry = held_eef_xpos[:, :3, 1]
            dot_result = (grasp_ry * obj_upright).sum(axis=1)
            # revert flag is -1 if the dot product is negative, 1 if positive
            revert_flag = torch.where(dot_result < 0, 1.0, -1.0)
            grasp_rx = held_eef_xpos[:, :3, 0]
            # rotate util upright
            rota_axis_angle = -0.5 * torch.pi * revert_flag * grasp_rx
            gripper_rotate_offset = axis_angle_to_rotation_matrix(rota_axis_angle)
            # modified target xpos rotation
            object_target_pose[:, :3, :3] = torch.bmm(
                gripper_rotate_offset, held_obj_xpos[:, :3, :3]
            )
        object_to_eef = state.held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

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
                coordinated_held_object=state.coordinated_held_object,
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
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=None,
                coordinated_held_object=state.coordinated_held_object,
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
# CoordinatedPickment
# =============================================================================


class CoordinatedPickment(AtomicAction):
    """Pick and move a single object pinched by two hands."""

    TargetType: ClassVar[type] = CoordinatedPickmentTarget

    _assemble_phase = _DualArmHelpers._assemble_phase
    _compose_dual_arm_trajectory = _DualArmHelpers._compose_dual_arm_trajectory
    _expand_qpos = _DualArmHelpers._expand_qpos
    _fail = _DualArmHelpers._fail
    _init_dual_arm_parts = _DualArmHelpers._init_dual_arm_parts
    _interpolate_keyframe_qpos = _DualArmHelpers._interpolate_keyframe_qpos
    _interpolate_object_pose = _DualArmHelpers._interpolate_object_pose
    _interpolate_qpos = _DualArmHelpers._interpolate_qpos
    _interpolate_qpos_keyframes = _DualArmHelpers._interpolate_qpos_keyframes
    _lookup_joint_columns = staticmethod(_DualArmHelpers._lookup_joint_columns)
    _plan_named_arm_trajectory = _DualArmHelpers._plan_named_arm_trajectory
    _repeat_qpos = staticmethod(_DualArmHelpers._repeat_qpos)
    _resolve_dual_arm_start = _DualArmHelpers._resolve_dual_arm_start
    _resolve_pose = _DualArmHelpers._resolve_pose

    def __init__(
        self,
        motion_generator,
        cfg: CoordinatedPickmentCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or CoordinatedPickmentCfg())
        self._init_dual_arm_parts(
            first_arm_control_part=self.cfg.left_arm_control_part,
            second_arm_control_part=self.cfg.right_arm_control_part,
            first_hand_control_part=self.cfg.left_hand_control_part,
            second_hand_control_part=self.cfg.right_hand_control_part,
        )
        self.left_arm_joint_ids = self.first_arm_joint_ids
        self.right_arm_joint_ids = self.second_arm_joint_ids
        self.left_hand_joint_ids = self.first_hand_joint_ids
        self.right_hand_joint_ids = self.second_hand_joint_ids
        self.left_arm_dof = self.first_arm_dof
        self.right_arm_dof = self.second_arm_dof
        self.left_hand_dof = self.first_hand_dof
        self.right_hand_dof = self.second_hand_dof

        self._validate_hand_qpos_cfg()
        self.left_hand_open_qpos = self._expand_qpos(
            self.cfg.left_hand_open_qpos, self.left_hand_dof, "left_hand_open_qpos"
        )
        self.left_hand_close_qpos = self._expand_qpos(
            self.cfg.left_hand_close_qpos, self.left_hand_dof, "left_hand_close_qpos"
        )
        self.right_hand_open_qpos = self._expand_qpos(
            self.cfg.right_hand_open_qpos, self.right_hand_dof, "right_hand_open_qpos"
        )
        self.right_hand_close_qpos = self._expand_qpos(
            self.cfg.right_hand_close_qpos,
            self.right_hand_dof,
            "right_hand_close_qpos",
        )

    def _validate_hand_qpos_cfg(self) -> None:
        """Ensure all hand state tensors are provided."""
        for name in (
            "left_hand_open_qpos",
            "left_hand_close_qpos",
            "right_hand_open_qpos",
            "right_hand_close_qpos",
        ):
            if getattr(self.cfg, name) is None:
                logger.log_error(f"{name} must be specified in CoordinatedPickmentCfg")

    def _resolve_object_initial_pose(
        self, target: CoordinatedPickmentTarget
    ) -> torch.Tensor:
        """Resolve the current pose of the object being grasped."""
        if target.object_initial_pose is not None:
            return self._resolve_pose(target.object_initial_pose, "object_initial_pose")
        if target.object_semantics.entity is None:
            logger.log_error(
                "CoordinatedPickmentTarget requires object_initial_pose when "
                "object_semantics.entity is not provided.",
                ValueError,
            )
        return self._resolve_pose(
            target.object_semantics.entity.get_local_pose(to_matrix=True),
            "object_initial_pose",
        )

    def _resolve_target(
        self,
        target: CoordinatedPickmentTarget,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        CoordinatedHeldObjectState,
    ]:
        """Resolve an object-centric pickment target into left/right TCP poses."""
        object_initial_pose = self._resolve_object_initial_pose(target)
        object_target_pose = self._resolve_pose(
            target.object_target_pose, "object_target_pose"
        )
        left_object_to_eef = self._resolve_pose(
            target.left_object_to_eef, "left_object_to_eef"
        )
        right_object_to_eef = self._resolve_pose(
            target.right_object_to_eef, "right_object_to_eef"
        )

        left_grasp_xpos = torch.bmm(object_initial_pose, left_object_to_eef)
        right_grasp_xpos = torch.bmm(object_initial_pose, right_object_to_eef)
        left_target_xpos = torch.bmm(object_target_pose, left_object_to_eef)
        right_target_xpos = torch.bmm(object_target_pose, right_object_to_eef)
        held_state = CoordinatedHeldObjectState(
            semantics=target.object_semantics,
            left_object_to_eef=left_object_to_eef,
            right_object_to_eef=right_object_to_eef,
            left_grasp_xpos=left_grasp_xpos,
            right_grasp_xpos=right_grasp_xpos,
        )
        return (
            object_initial_pose,
            object_target_pose,
            left_grasp_xpos,
            right_grasp_xpos,
            left_target_xpos,
            right_target_xpos,
            held_state,
        )

    def _compute_segment_lengths(self) -> dict[str, int]:
        """Compute waypoint counts for coordinated pickment phases."""
        n_close = max(2, self.cfg.hand_interp_steps)
        n_hold = max(0, self.cfg.hold_steps)
        n_motion = self.cfg.sample_interval - n_close - n_hold
        n_approach = n_motion // 3
        n_lift = n_motion // 3
        n_move = n_motion - n_approach - n_lift
        if min(n_approach, n_lift, n_move) < 2:
            logger.log_error(
                "Not enough waypoints for coordinated pickment. Please increase "
                "sample_interval or decrease hand_interp_steps/hold_steps.",
                ValueError,
            )
        return {
            "approach": n_approach,
            "close": n_close,
            "lift": n_lift,
            "move": n_move,
            "hold": n_hold,
        }

    def get_segment_lengths(self) -> dict[str, int]:
        """Return waypoint counts for the coordinated pickment phase sequence."""
        return self._compute_segment_lengths()

    def _compute_pre_grasp_xpos(self, grasp_xpos: torch.Tensor) -> torch.Tensor:
        """Compute pre-grasp poses by backing away along each TCP z axis."""
        grasp_z = grasp_xpos[:, :3, 2]
        return self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
        )

    def _select_motion_keyframe_indices(self, n_waypoints: int) -> torch.Tensor:
        """Select sparse object motion keyframes for IK, including endpoints."""
        n_keyframes = min(max(2, int(self.cfg.object_motion_keyframes)), n_waypoints)
        return (
            torch.linspace(
                0,
                n_waypoints - 1,
                steps=n_keyframes,
                device=self.device,
            )
            .round()
            .to(dtype=torch.long)
        )

    def _plan_synchronized_object_motion(
        self,
        left_start_qpos: torch.Tensor,
        right_start_qpos: torch.Tensor,
        object_pose_traj: torch.Tensor,
        left_object_to_eef: torch.Tensor,
        right_object_to_eef: torch.Tensor,
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        """Plan both arms from the same sparse object-pose trajectory."""
        n_waypoints = object_pose_traj.shape[1]
        keyframe_indices = self._select_motion_keyframe_indices(n_waypoints)
        left_traj = torch.zeros(
            (self.n_envs, len(keyframe_indices), left_start_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        right_traj = torch.zeros(
            (self.n_envs, len(keyframe_indices), right_start_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        left_qpos_seed = left_start_qpos
        right_qpos_seed = right_start_qpos
        for keyframe_col, waypoint_idx in enumerate(keyframe_indices.tolist()):
            left_xpos = torch.bmm(object_pose_traj[:, waypoint_idx], left_object_to_eef)
            right_xpos = torch.bmm(
                object_pose_traj[:, waypoint_idx], right_object_to_eef
            )
            left_success, left_qpos = self.robot.compute_ik(
                pose=left_xpos,
                name=self.cfg.left_arm_control_part,
                joint_seed=left_qpos_seed,
            )
            right_success, right_qpos = self.robot.compute_ik(
                pose=right_xpos,
                name=self.cfg.right_arm_control_part,
                joint_seed=right_qpos_seed,
            )
            if not self.builder.all_envs_success(left_success):
                logger.log_warning(
                    f"Failed to compute IK for {self.cfg.left_arm_control_part} "
                    f"object waypoint {waypoint_idx}."
                )
                return False, left_traj, right_traj
            if not self.builder.all_envs_success(right_success):
                logger.log_warning(
                    f"Failed to compute IK for {self.cfg.right_arm_control_part} "
                    f"object waypoint {waypoint_idx}."
                )
                return False, left_traj, right_traj
            left_traj[:, keyframe_col] = left_qpos
            right_traj[:, keyframe_col] = right_qpos
            left_qpos_seed = left_qpos
            right_qpos_seed = right_qpos

        return (
            True,
            self._interpolate_qpos_keyframes(left_traj, keyframe_indices, n_waypoints),
            self._interpolate_qpos_keyframes(right_traj, keyframe_indices, n_waypoints),
        )

    def execute(
        self, target: CoordinatedPickmentTarget, state: WorldState
    ) -> ActionResult:
        (
            object_initial_pose,
            object_target_pose,
            left_grasp_xpos,
            right_grasp_xpos,
            left_target_xpos,
            right_target_xpos,
            held_state,
        ) = self._resolve_target(target)
        left_start_qpos, right_start_qpos = self._resolve_dual_arm_start(state)
        segments = self._compute_segment_lengths()

        left_pre_grasp_xpos = self._compute_pre_grasp_xpos(left_grasp_xpos)
        right_pre_grasp_xpos = self._compute_pre_grasp_xpos(right_grasp_xpos)
        left_approach_targets = torch.stack(
            [left_pre_grasp_xpos, left_grasp_xpos], dim=1
        )
        right_approach_targets = torch.stack(
            [right_pre_grasp_xpos, right_grasp_xpos], dim=1
        )
        ok, left_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.left_arm_control_part,
            left_start_qpos,
            left_approach_targets,
            segments["approach"],
        )
        if not ok:
            return self._fail(state)
        ok, right_approach_traj = self._plan_named_arm_trajectory(
            self.cfg.right_arm_control_part,
            right_start_qpos,
            right_approach_targets,
            segments["approach"],
        )
        if not ok:
            return self._fail(state)

        left_grasp_qpos = left_approach_traj[:, -1]
        right_grasp_qpos = right_approach_traj[:, -1]
        approach_trajectory = self._assemble_phase(
            state,
            left_approach_traj,
            right_approach_traj,
            self._repeat_qpos(self.left_hand_open_qpos, segments["approach"]),
            self._repeat_qpos(self.right_hand_open_qpos, segments["approach"]),
        )

        close_trajectory = self._assemble_phase(
            state,
            self._repeat_qpos(left_grasp_qpos, segments["close"]),
            self._repeat_qpos(right_grasp_qpos, segments["close"]),
            self._interpolate_qpos(
                self.left_hand_open_qpos,
                self.left_hand_close_qpos,
                segments["close"],
            ),
            self._interpolate_qpos(
                self.right_hand_open_qpos,
                self.right_hand_close_qpos,
                segments["close"],
            ),
        )

        lift_object_pose = self.builder.apply_local_offset(
            object_initial_pose,
            torch.tensor([0.0, 0.0, self.cfg.lift_height], device=self.device),
        )
        lift_object_traj = self._interpolate_object_pose(
            object_initial_pose,
            lift_object_pose,
            segments["lift"],
            include_orientation=False,
        )
        ok, left_lift_traj, right_lift_traj = self._plan_synchronized_object_motion(
            left_grasp_qpos,
            right_grasp_qpos,
            lift_object_traj,
            held_state.left_object_to_eef,
            held_state.right_object_to_eef,
        )
        if not ok:
            return self._fail(state)

        left_lift_qpos = left_lift_traj[:, -1]
        right_lift_qpos = right_lift_traj[:, -1]
        lift_trajectory = self._assemble_phase(
            state,
            left_lift_traj,
            right_lift_traj,
            self._repeat_qpos(self.left_hand_close_qpos, segments["lift"]),
            self._repeat_qpos(self.right_hand_close_qpos, segments["lift"]),
        )

        move_object_traj = self._interpolate_object_pose(
            lift_object_pose,
            object_target_pose,
            segments["move"],
            include_orientation=True,
        )
        ok, left_move_traj, right_move_traj = self._plan_synchronized_object_motion(
            left_lift_qpos,
            right_lift_qpos,
            move_object_traj,
            held_state.left_object_to_eef,
            held_state.right_object_to_eef,
        )
        if not ok:
            return self._fail(state)

        left_target_qpos = left_move_traj[:, -1]
        right_target_qpos = right_move_traj[:, -1]
        move_trajectory = self._assemble_phase(
            state,
            left_move_traj,
            right_move_traj,
            self._repeat_qpos(self.left_hand_close_qpos, segments["move"]),
            self._repeat_qpos(self.right_hand_close_qpos, segments["move"]),
        )

        hold_trajectory = torch.empty(
            (self.n_envs, 0, self.robot_dof), dtype=torch.float32, device=self.device
        )
        if segments["hold"] > 0:
            hold_trajectory = self._assemble_phase(
                state,
                self._repeat_qpos(left_target_qpos, segments["hold"]),
                self._repeat_qpos(right_target_qpos, segments["hold"]),
                self._repeat_qpos(self.left_hand_close_qpos, segments["hold"]),
                self._repeat_qpos(self.right_hand_close_qpos, segments["hold"]),
            )

        full = torch.cat(
            [
                approach_trajectory,
                close_trajectory,
                lift_trajectory,
                move_trajectory,
                hold_trajectory,
            ],
            dim=1,
        )
        coordinated_held_object = CoordinatedHeldObjectState(
            semantics=held_state.semantics,
            left_object_to_eef=held_state.left_object_to_eef,
            right_object_to_eef=held_state.right_object_to_eef,
            left_grasp_xpos=left_target_xpos,
            right_grasp_xpos=right_target_xpos,
        )
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=None,
                coordinated_held_object=coordinated_held_object,
            ),
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
                coordinated_held_object=state.coordinated_held_object,
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
    "CoordinatedPickment",
    "CoordinatedPickmentCfg",
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
