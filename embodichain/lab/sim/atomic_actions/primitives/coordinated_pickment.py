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

"""CoordinatedPickment atomic action implementation."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.utils import configclass, logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedHeldObjectState,
    CoordinatedPickmentTarget,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


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
        return qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    def _interpolate_qpos(
        self,
        start_qpos: torch.Tensor,
        end_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
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
        for name in (
            "left_hand_open_qpos",
            "left_hand_close_qpos",
            "right_hand_open_qpos",
            "right_hand_close_qpos",
        ):
            if getattr(self.cfg, name) is None:
                logger.log_error(
                    f"{name} must be specified in CoordinatedPickmentCfg",
                    ValueError,
                )

    def _resolve_object_initial_pose(
        self, target: CoordinatedPickmentTarget
    ) -> torch.Tensor:
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
        return self._compute_segment_lengths()

    def _compute_pre_grasp_xpos(self, grasp_xpos: torch.Tensor) -> torch.Tensor:
        grasp_z = grasp_xpos[:, :3, 2]
        return self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
        )

    def _select_motion_keyframe_indices(self, n_waypoints: int) -> torch.Tensor:
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


__all__ = ["CoordinatedPickment", "CoordinatedPickmentCfg"]
