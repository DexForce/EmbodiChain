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

from embodichain.lab.sim.planners import MoveType, PlanState
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedHeldObjectState,
    CoordinatedPickmentTarget,
    GraspTarget,
    ObjectSemantics,
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

    TargetType: ClassVar[type] = GraspTarget

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
        if (
            self.cfg.motion_source == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            logger.log_error(
                "Coordinated dual-arm planning is not supported by the cuRobo "
                "backend. Use a single-arm action or a dedicated multi-arm "
                "planner.",
                ValueError,
            )
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

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        left_start_qpos: torch.Tensor,
        right_start_qpos: torch.Tensor,
    ):
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        left_solver = self.robot._solvers.get("left_arm", None)
        right_solver = self.robot._solvers.get("right_arm", None)
        if left_solver is None or right_solver is None:
            logger.log_error(
                "CoordinatedPickment requires both left_arm and right_arm solvers "
                "to be configured in the robot.",
                ValueError,
            )
        left_root_pose = self.robot.get_link_pose(
            link_name=left_solver.root_link_name, to_matrix=True
        )
        right_root_pose = self.robot.get_link_pose(
            link_name=right_solver.root_link_name, to_matrix=True
        )
        left_to_right_direc = right_root_pose[:, :3, 3] - left_root_pose[:, :3, 3]
        left_to_right_direc = left_to_right_direc / torch.linalg.norm(
            left_to_right_direc, dim=-1, keepdim=True
        )
        grasp_poses_result = semantics.affordance.get_dual_arm_valid_grasp_poses(
            obj_poses=obj_poses,
            left_to_right_arm_direction=left_to_right_direc,
            approach_direction=torch.tensor([0.0, 0.0, -1.0], device=self.device),
        )

        n_envs = obj_poses.shape[0]
        n_left_max_pose = 0
        n_right_max_pose = 0
        for i in range(n_envs):
            left_result = grasp_poses_result[i]["left"]
            right_result = grasp_poses_result[i]["right"]
            if left_result is None or right_result is None:
                logger.log_warning(
                    f"Failed to find valid dual-arm grasp poses for {i}-th enviroment."
                )
                continue
            n_left_max_pose = max(n_left_max_pose, left_result["grasp_poses"].shape[0])
            n_right_max_pose = max(
                n_right_max_pose, right_result["grasp_poses"].shape[0]
            )
        if n_left_max_pose == 0 or n_right_max_pose == 0:
            logger.log_error(
                "Failed to find valid dual-arm grasp poses for any environment.",
                ValueError,
            )

        left_grasp_xpos_padding = torch.zeros(
            (n_envs, n_left_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        right_grasp_xpos_padding = torch.zeros(
            (n_envs, n_right_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        left_grasp_costs_padding = torch.full(
            (n_envs, n_left_max_pose),
            fill_value=float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        right_grasp_costs_padding = torch.full(
            (n_envs, n_right_max_pose),
            fill_value=float("inf"),
            dtype=torch.float32,
            device=self.device,
        )

        for i in range(n_envs):
            left_result = grasp_poses_result[i]["left"]
            right_result = grasp_poses_result[i]["right"]
            if left_result is not None:
                n_left_pose = left_result["grasp_poses"].shape[0]
                left_grasp_xpos_padding[i, :n_left_pose] = left_result["grasp_poses"]
                left_grasp_costs_padding[i, :n_left_pose] = left_result["total_cost"]
                left_grasp_xpos_padding[i, n_left_pose:] = left_grasp_xpos_padding[
                    i, :1
                ]
                left_grasp_costs_padding[i, n_left_pose:] = left_grasp_costs_padding[
                    i, :1
                ]
            else:
                left_grasp_xpos_padding[i, :] = torch.eye(4, device=self.device)
                left_grasp_costs_padding[i, :] = float("inf")
            if right_result is not None:
                n_right_pose = right_result["grasp_poses"].shape[0]
                right_grasp_xpos_padding[i, :n_right_pose] = right_result["grasp_poses"]
                right_grasp_costs_padding[i, :n_right_pose] = right_result["total_cost"]
                right_grasp_xpos_padding[i, n_right_pose:] = right_grasp_xpos_padding[
                    i, :1
                ]
                right_grasp_costs_padding[i, n_right_pose:] = right_grasp_costs_padding[
                    i, :1
                ]
            else:
                right_grasp_xpos_padding[i, :] = torch.eye(4, device=self.device)
                right_grasp_costs_padding[i, :] = float("inf")

        # TODO: masked ik valid grasp poses
        # TODO: find nearest rotation symmetric pose
        left_best_idx = torch.argmin(left_grasp_costs_padding, dim=1)
        right_best_idx = torch.argmin(right_grasp_costs_padding, dim=1)
        left_grasp_xpos = left_grasp_xpos_padding[torch.arange(n_envs), left_best_idx]
        right_grasp_xpos = right_grasp_xpos_padding[
            torch.arange(n_envs), right_best_idx
        ]
        return left_grasp_xpos, right_grasp_xpos

    def _get_full_pickup_trajectory(
        self,
        grasp_xpos: torch.Tensor,
        start_arm_qpos: torch.Tensor,
        approach_direction: torch.Tensor,
        hand_open_qpos: torch.Tensor,
        hand_close_qpos: torch.Tensor,
        control_part: str,
    ):
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -approach_direction * self.cfg.pre_grasp_distance
        )
        arm_dof = start_arm_qpos.shape[-1]
        hand_dof = hand_open_qpos.shape[-1]
        n_approach, n_close, n_lift = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        approach_success, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=control_part,
            arm_dof=arm_dof,
            cfg=self.cfg,
        )

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        lift_success, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=control_part,
            arm_dof=arm_dof,
            cfg=self.cfg,
        )
        is_success = approach_success & lift_success

        hand_close_path = self.builder.interpolate_hand_qpos(
            hand_open_qpos, hand_close_qpos, n_waypoints=n_close
        )
        n_approach_actual = approach_arm.shape[1]
        n_lift_actual = lift_arm.shape[1]

        full_arm_traj = torch.empty(
            (self.n_envs, n_approach_actual + n_close + n_lift_actual, arm_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full_hand_traj = torch.empty(
            (self.n_envs, n_approach_actual + n_close + n_lift_actual, hand_dof),
            dtype=torch.float32,
            device=self.device,
        )
        # approach
        full_arm_traj[:, :n_approach_actual, :] = approach_arm
        full_hand_traj[:, :n_approach_actual, :] = hand_open_qpos
        # close
        full_arm_traj[:, n_approach_actual : n_approach_actual + n_close, :] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full_hand_traj[:, n_approach_actual : n_approach_actual + n_close, :] = (
            hand_close_path
        )
        # lift
        full_arm_traj[:, n_approach_actual + n_close :, :] = lift_arm
        full_hand_traj[:, n_approach_actual + n_close :, :] = hand_close_qpos
        return is_success, full_arm_traj, full_hand_traj

    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult:
        left_start_qpos, right_start_qpos = self._resolve_dual_arm_start(state)
        left_grasp_xpos, right_grasp_xpos = self._resolve_grasp_pose(
            target.semantics, left_start_qpos, right_start_qpos
        )
        is_left_success, left_arm_traj, left_hand_traj = (
            self._get_full_pickup_trajectory(
                grasp_xpos=left_grasp_xpos,
                start_arm_qpos=left_start_qpos,
                approach_direction=torch.tensor([0, 0, -1], device=self.device),
                hand_open_qpos=self.left_hand_open_qpos,
                hand_close_qpos=self.left_hand_close_qpos,
                control_part=self.cfg.left_arm_control_part,
            )
        )
        is_right_success, right_arm_traj, right_hand_traj = (
            self._get_full_pickup_trajectory(
                grasp_xpos=right_grasp_xpos,
                start_arm_qpos=right_start_qpos,
                approach_direction=torch.tensor([0, 0, -1], device=self.device),
                hand_open_qpos=self.right_hand_open_qpos,
                hand_close_qpos=self.right_hand_close_qpos,
                control_part=self.cfg.right_arm_control_part,
            )
        )
        is_success = is_left_success & is_right_success
        last_qpos = state.last_qpos.to(self.device)
        full_dof = last_qpos.shape[-1]
        n_left_waypoints = left_arm_traj.shape[1]
        n_right_waypoints = right_arm_traj.shape[1]
        n_waypoints = max(n_left_waypoints, n_right_waypoints)
        full_trajectory = torch.empty(
            (self.n_envs, n_waypoints, full_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full_trajectory[:, :, :] = last_qpos.unsqueeze(1)

        # pading trajectory end to match the max length
        full_trajectory[:, :n_left_waypoints, self.left_arm_joint_ids] = left_arm_traj
        full_trajectory[:, :n_left_waypoints, self.left_hand_joint_ids] = left_hand_traj
        full_trajectory[:, :n_right_waypoints, self.right_arm_joint_ids] = (
            right_arm_traj
        )
        full_trajectory[:, :n_right_waypoints, self.right_hand_joint_ids] = (
            right_hand_traj
        )
        return ActionResult(
            success=is_success,
            trajectory=full_trajectory,
            next_state=WorldState(
                last_qpos=full_trajectory[:, -1, :].clone(),
                held_object=None,
                coordinated_held_object=state.coordinated_held_object,
            ),
        )


__all__ = ["CoordinatedPickment", "CoordinatedPickmentCfg"]
