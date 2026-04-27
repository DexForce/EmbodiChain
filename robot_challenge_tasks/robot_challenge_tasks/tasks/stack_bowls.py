# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    MoveActionCfg,
    PickUpActionCfg,
    PlaceActionCfg,
)
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator, ToppraPlannerCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.utils import logger
from scipy.spatial.transform import Rotation as R

__all__ = ["StackBowlsEnv"]


@register_env("StackBowls-v1")
class StackBowlsEnv(EmbodiedEnv):

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        interp_times = self.cfg.extensions.get("interp_times", [15] * 16)
        self.interp_times = torch.tensor(
            interp_times,
            dtype=torch.int32,
            device=self.device,
        )

        # Define some constants for the demo trajectory generation.
        downward_r = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        down_rota_np = (
            R.from_euler("xyz", [20, -30, 0], degrees=True).as_matrix() @ downward_r
        )

        self.right_down_rota = torch.tensor(down_rota_np, device=self.device)

        left_down_rota_np = (
            R.from_euler("xyz", [-20, -30, 0], degrees=True).as_matrix() @ downward_r
        )
        self.left_down_rota = torch.tensor(left_down_rota_np, device=self.device)

        self.right_arm_ids = self.robot.get_joint_ids("right_arm")
        self.right_eef_ids = self.robot.get_joint_ids("right_eef")
        self.left_arm_ids = self.robot.get_joint_ids("left_arm")
        self.left_eef_ids = self.robot.get_joint_ids("left_eef")

        self.right_pick_offset = torch.tensor([0.00, -0.06, 0.01], device=self.device)
        self.left_pick_offset = torch.tensor([0.00, 0.06, 0.01], device=self.device)
        self.bowl_mid_pre_pick_offset = torch.tensor(
            [0.0, 0.0, 0.12], device=self.device
        )
        self.bowl_min_pre_pick_offset = torch.tensor(
            [0.0, 0.0, 0.16], device=self.device
        )

        self.right_bowl_mid_place_offset = torch.tensor(
            [0.00, -0.05, 0.12], device=self.device
        )
        self.left_bowl_mid_place_offset = torch.tensor(
            [0.00, 0.05, 0.12], device=self.device
        )
        self.right_bowl_min_place_offset = torch.tensor(
            [0.00, -0.045, 0.16], device=self.device
        )
        self.left_bowl_min_place_offset = torch.tensor(
            [0.00, 0.045, 0.16], device=self.device
        )

        init_qpos = self.robot.get_qpos()
        init_right_arm_qpos = init_qpos[:, self.right_arm_ids]
        self.init_right_arm_xpos = self.robot.compute_fk(
            qpos=init_right_arm_qpos, name="right_arm", to_matrix=True
        )
        init_left_arm_qpos = init_qpos[:, self.left_arm_ids]
        self.init_left_arm_xpos = self.robot.compute_fk(
            qpos=init_left_arm_qpos, name="left_arm", to_matrix=True
        )

        self.atomic_action_engines = {}
        self._init_atomic_demo_generators()

    def _init_atomic_demo_generators(self) -> None:
        """Initialize atomic-action planners used by the optional demo branch."""
        if not getattr(self, "use_atomic_demo_actions", False):
            return

        extensions = getattr(self.cfg, "extensions", {}) or {}
        self.atomic_pick_sample_interval = int(
            extensions.get("atomic_pick_sample_interval", 36)
        )
        self.atomic_place_sample_interval = int(
            extensions.get("atomic_place_sample_interval", 36)
        )
        self.atomic_move_sample_interval = int(
            extensions.get("atomic_move_sample_interval", 24)
        )
        self.atomic_hand_interp_steps = int(
            extensions.get("atomic_hand_interp_steps", 5)
        )
        self.atomic_pre_grasp_distance = float(
            extensions.get("atomic_pre_grasp_distance", 0.02)
        )
        self.atomic_lift_height = float(extensions.get("atomic_lift_height", 0.10))

        self.left_eef_open_qpos = self.robot.get_qpos_limits(name="left_eef")[
            0, :, 1
        ].clone()
        self.left_eef_close_qpos = self.robot.get_qpos_limits(name="left_eef")[
            0, :, 0
        ].clone()
        self.right_eef_open_qpos = self.robot.get_qpos_limits(name="right_eef")[
            0, :, 1
        ].clone()
        self.right_eef_close_qpos = self.robot.get_qpos_limits(name="right_eef")[
            0, :, 0
        ].clone()

        motion_cfg = MotionGenCfg(
            planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid)
        )
        self.atomic_motion_generator = MotionGenerator(cfg=motion_cfg)
        self.atomic_action_engines = {
            "left_arm": self._build_atomic_action_engine("left_arm"),
            "right_arm": self._build_atomic_action_engine("right_arm"),
        }

    def _build_atomic_action_engine(self, arm_type: str) -> AtomicActionEngine:
        """Create a per-arm atomic action engine with arm-specific configs."""
        hand_type = "left_eef" if arm_type == "left_arm" else "right_eef"
        hand_open_qpos = (
            self.left_eef_open_qpos
            if arm_type == "left_arm"
            else self.right_eef_open_qpos
        )
        hand_close_qpos = (
            self.left_eef_close_qpos
            if arm_type == "left_arm"
            else self.right_eef_close_qpos
        )
        pickup_cfg = PickUpActionCfg(
            control_part=arm_type,
            hand_control_part=hand_type,
            hand_open_qpos=hand_open_qpos.clone(),
            hand_close_qpos=hand_close_qpos.clone(),
            approach_direction=torch.tensor(
                [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
            ),
            pre_grasp_distance=self.atomic_pre_grasp_distance,
            lift_height=self.atomic_lift_height,
            sample_interval=self.atomic_pick_sample_interval,
            hand_interp_steps=self.atomic_hand_interp_steps,
        )
        place_cfg = PlaceActionCfg(
            control_part=arm_type,
            hand_control_part=hand_type,
            hand_open_qpos=hand_open_qpos.clone(),
            hand_close_qpos=hand_close_qpos.clone(),
            lift_height=self.atomic_lift_height,
            sample_interval=self.atomic_place_sample_interval,
            hand_interp_steps=self.atomic_hand_interp_steps,
        )
        move_cfg = MoveActionCfg(
            control_part=arm_type,
            sample_interval=self.atomic_move_sample_interval,
        )
        return AtomicActionEngine(
            robot=self.robot,
            motion_generator=self.atomic_motion_generator,
            device=self.device,
            actions_cfg_dict={
                "pick_up": pickup_cfg,
                "place": place_cfg,
                "move": move_cfg,
            },
        )

    def _get_home_pose(self, arm_type: str) -> torch.Tensor:
        """Return the initial end-effector pose for the selected arm."""
        if arm_type == "left_arm":
            return self.init_left_arm_xpos.clone()
        return self.init_right_arm_xpos.clone()

    def _get_arm_joint_ids(self, arm_type: str) -> List[int]:
        """Return arm joint ids for the selected arm."""
        if arm_type == "left_arm":
            return self.left_arm_ids
        return self.right_arm_ids

    def _get_hand_joint_ids(self, arm_type: str) -> List[int]:
        """Return gripper joint ids for the selected arm."""
        if arm_type == "left_arm":
            return self.left_eef_ids
        return self.right_eef_ids

    def _get_open_hand_qpos(self, arm_type: str) -> torch.Tensor:
        """Return the open gripper joint positions for the selected arm."""
        if arm_type == "left_arm":
            return self.left_eef_open_qpos
        return self.right_eef_open_qpos

    def _get_close_hand_qpos(self, arm_type: str) -> torch.Tensor:
        """Return the closed gripper joint positions for the selected arm."""
        if arm_type == "left_arm":
            return self.left_eef_close_qpos
        return self.right_eef_close_qpos

    def _offset_pose_z(self, pose: torch.Tensor, offset_z: float) -> torch.Tensor:
        """Offset a pose along the world z-axis."""
        offset_pose = pose.clone()
        offset_pose[:, 2, 3] = offset_pose[:, 2, 3] + offset_z
        return offset_pose

    def _is_current_pose_close(
        self,
        current_qpos: torch.Tensor,
        arm_type: str,
        target_pose: torch.Tensor,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ) -> bool:
        """Check whether the arm is already close enough to a target pose."""
        arm_joint_ids = self._get_arm_joint_ids(arm_type)
        current_pose = self.robot.compute_fk(
            qpos=current_qpos[:, arm_joint_ids], name=arm_type, to_matrix=True
        )
        pos_diff = torch.norm(current_pose[:, :3, 3] - target_pose[:, :3, 3], dim=-1)
        rot_diff = torch.norm(
            current_pose[:, :3, :3] - target_pose[:, :3, :3], dim=(-2, -1)
        )
        return bool(((pos_diff < pos_tol) & (rot_diff < rot_tol)).all().item())

    def _expand_atomic_trajectory(
        self,
        trajectory: torch.Tensor,
        joint_ids: List[int],
        current_qpos: torch.Tensor,
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        """Expand a local atomic trajectory into full environment actions."""
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)

        action_list = []
        full_qpos = current_qpos.clone()
        for waypoint_idx in range(trajectory.shape[1]):
            next_qpos = full_qpos.clone()
            next_qpos[:, joint_ids] = trajectory[:, waypoint_idx, :]
            action_list.append(next_qpos[:, self.active_joint_ids].clone())
            full_qpos = next_qpos
        return action_list, full_qpos

    def _append_atomic_action(
        self,
        action_list: List[torch.Tensor],
        current_qpos: torch.Tensor,
        action_name: str,
        arm_type: str,
        target_pose: torch.Tensor,
        log_name: str,
    ) -> Optional[torch.Tensor]:
        """Execute one atomic action and append the resulting environment actions."""
        engine = self.atomic_action_engines[arm_type]
        arm_joint_ids = self._get_arm_joint_ids(arm_type)
        if action_name == "move" and self._is_current_pose_close(
            current_qpos=current_qpos,
            arm_type=arm_type,
            target_pose=target_pose,
        ):
            return current_qpos

        start_qpos = current_qpos[:, arm_joint_ids]
        try:
            is_success, trajectory, joint_ids = engine.execute(
                action_name=action_name,
                target=target_pose,
                start_qpos=start_qpos,
                control_part=arm_type,
            )
        except Exception as exc:
            logger.log_warning(
                f"Atomic action `{log_name}` raised {type(exc).__name__}: {exc}"
            )
            return None
        success_flag = (
            bool(is_success.all().item())
            if isinstance(is_success, torch.Tensor)
            else bool(is_success)
        )
        if not success_flag:
            logger.log_warning(
                f"Atomic action `{log_name}` failed for {arm_type}. Returning None."
            )
            return None

        segment_actions, next_qpos = self._expand_atomic_trajectory(
            trajectory=trajectory,
            joint_ids=joint_ids,
            current_qpos=current_qpos,
        )
        action_list.extend(segment_actions)
        return next_qpos

    def _append_gripper_segment(
        self,
        action_list: List[torch.Tensor],
        current_qpos: torch.Tensor,
        arm_type: str,
        target_hand_qpos: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Interpolate gripper joints while keeping the arm fixed."""
        hand_joint_ids = self._get_hand_joint_ids(arm_type)
        start_hand_qpos = current_qpos[:, hand_joint_ids]
        full_qpos = current_qpos.clone()

        for weight in torch.linspace(0.0, 1.0, steps=num_steps, device=self.device):
            next_qpos = full_qpos.clone()
            interp_qpos = torch.lerp(
                start_hand_qpos,
                target_hand_qpos.unsqueeze(0).expand_as(start_hand_qpos),
                weight,
            )
            next_qpos[:, hand_joint_ids] = interp_qpos
            action_list.append(next_qpos[:, self.active_joint_ids].clone())
            full_qpos = next_qpos
        return full_qpos

    def _append_place_with_fallback(
        self,
        action_list: List[torch.Tensor],
        current_qpos: torch.Tensor,
        arm_type: str,
        target_pose: torch.Tensor,
        log_name: str,
    ) -> Optional[torch.Tensor]:
        """Run place action and fallback to a move-based open-place-lift sequence."""
        next_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="place",
            arm_type=arm_type,
            target_pose=target_pose,
            log_name=log_name,
        )
        if next_qpos is not None:
            return next_qpos

        logger.log_warning(
            f"Falling back to move-based place sequence for `{log_name}` on {arm_type}."
        )
        lift_pose = self._offset_pose_z(target_pose, self.atomic_lift_height)

        next_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="move",
            arm_type=arm_type,
            target_pose=lift_pose,
            log_name=f"{log_name}_fallback_lift_pose",
        )
        if next_qpos is None:
            return None
        next_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=next_qpos,
            action_name="move",
            arm_type=arm_type,
            target_pose=target_pose,
            log_name=f"{log_name}_fallback_place_pose",
        )
        if next_qpos is None:
            return None

        next_qpos = self._append_gripper_segment(
            action_list=action_list,
            current_qpos=next_qpos,
            arm_type=arm_type,
            target_hand_qpos=self._get_open_hand_qpos(arm_type),
            num_steps=max(2, self.atomic_hand_interp_steps),
        )
        return self._append_atomic_action(
            action_list=action_list,
            current_qpos=next_qpos,
            action_name="move",
            arm_type=arm_type,
            target_pose=lift_pose,
            log_name=f"{log_name}_fallback_retreat",
        )

    def _create_atomic_demo_action_list(self) -> Optional[torch.Tensor]:
        """Create a first-version atomic-action demo for the bowl stacking task."""
        if self.num_envs > 1:
            logger.log_error(
                "Atomic demo trajectory generation is only supported for a single environment. "
                f"Got num_envs={self.num_envs}."
            )

        if not self.atomic_action_engines:
            self._init_atomic_demo_generators()

        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)

        arm_to_use_mid = self.choice_arm_by_distance(bowl_mid_pose[:, :3, 3])
        mid_pick_offset = (
            self.left_pick_offset
            if arm_to_use_mid == "left_arm"
            else self.right_pick_offset
        )
        mid_down_rota = (
            self.left_down_rota
            if arm_to_use_mid == "left_arm"
            else self.right_down_rota
        )
        bowl_mid_pick_pose = bowl_mid_pose.clone()
        bowl_mid_pick_pose[:, :3, :3] = mid_down_rota
        bowl_mid_pick_pose[:, :3, 3] = bowl_mid_pick_pose[:, :3, 3] + mid_pick_offset
        bowl_mid_pre_pick_pose = bowl_mid_pick_pose.clone()
        bowl_mid_pre_pick_pose[:, :3, 3] = (
            bowl_mid_pre_pick_pose[:, :3, 3] + self.bowl_mid_pre_pick_offset
        )
        bowl_mid_place_offset = (
            self.left_bowl_mid_place_offset
            if arm_to_use_mid == "left_arm"
            else self.right_bowl_mid_place_offset
        )
        bowl_mid_place_pose = bowl_max_pose.clone()
        bowl_mid_place_pose[:, :3, :3] = mid_down_rota
        bowl_mid_place_pose[:, :3, 3] = (
            bowl_mid_place_pose[:, :3, 3] + bowl_mid_place_offset
        )

        arm_to_use_min = self.choice_arm_by_distance(bowl_min_pose[:, :3, 3])
        min_pick_offset = (
            self.left_pick_offset
            if arm_to_use_min == "left_arm"
            else self.right_pick_offset
        )
        min_down_rota = (
            self.left_down_rota
            if arm_to_use_min == "left_arm"
            else self.right_down_rota
        )
        bowl_min_pick_pose = bowl_min_pose.clone()
        bowl_min_pick_pose[:, :3, :3] = min_down_rota
        bowl_min_pick_pose[:, :3, 3] = bowl_min_pick_pose[:, :3, 3] + min_pick_offset
        bowl_min_pre_pick_pose = bowl_min_pick_pose.clone()
        bowl_min_pre_pick_pose[:, :3, 3] = (
            bowl_min_pre_pick_pose[:, :3, 3] + self.bowl_min_pre_pick_offset
        )
        bowl_min_place_offset = (
            self.left_bowl_min_place_offset
            if arm_to_use_min == "left_arm"
            else self.right_bowl_min_place_offset
        )
        bowl_min_place_pose = bowl_max_pose.clone()
        bowl_min_place_pose[:, :3, :3] = min_down_rota
        bowl_min_place_pose[:, :3, 3] = (
            bowl_min_place_pose[:, :3, 3] + bowl_min_place_offset
        )

        same_arm = arm_to_use_mid == arm_to_use_min
        transition_pose = bowl_min_pre_pick_pose.clone()
        if same_arm:
            transition_pose[:, :3, 3] = (
                bowl_min_pre_pick_pose[:, :3, 3] + bowl_min_place_pose[:, :3, 3]
            ) / 2.0
        else:
            transition_pose = self._get_home_pose(arm_to_use_mid)

        current_qpos = self.robot.get_qpos().clone()
        action_list = []

        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="move",
            arm_type=arm_to_use_mid,
            target_pose=bowl_mid_pre_pick_pose,
            log_name="move_to_bowl_mid_pre_pick",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="pick_up",
            arm_type=arm_to_use_mid,
            target_pose=bowl_mid_pick_pose,
            log_name="pick_up_bowl_mid",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_place_with_fallback(
            action_list=action_list,
            current_qpos=current_qpos,
            arm_type=arm_to_use_mid,
            target_pose=bowl_mid_place_pose,
            log_name="place_bowl_mid",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="move",
            arm_type=arm_to_use_mid,
            target_pose=transition_pose,
            log_name="transition_after_bowl_mid",
        )
        if current_qpos is None:
            return None

        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="move",
            arm_type=arm_to_use_min,
            target_pose=bowl_min_pre_pick_pose,
            log_name="move_to_bowl_min_pre_pick",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="pick_up",
            arm_type=arm_to_use_min,
            target_pose=bowl_min_pick_pose,
            log_name="pick_up_bowl_min",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_place_with_fallback(
            action_list=action_list,
            current_qpos=current_qpos,
            arm_type=arm_to_use_min,
            target_pose=bowl_min_place_pose,
            log_name="place_bowl_min",
        )
        if current_qpos is None:
            return None
        current_qpos = self._append_atomic_action(
            action_list=action_list,
            current_qpos=current_qpos,
            action_name="move",
            arm_type=arm_to_use_min,
            target_pose=self._get_home_pose(arm_to_use_min),
            log_name="return_home_after_bowl_min",
        )
        if current_qpos is None:
            return None

        if len(action_list) == 0:
            return None

        return torch.stack(action_list, dim=0)

    def choice_arm_by_distance(self, target_pos) -> str:
        """Choose the arm that is closer to the target position."""
        left_arm_xpos = self.init_left_arm_xpos[:, :3, 3]
        right_arm_xpos = self.init_right_arm_xpos[:, :3, 3]
        left_dist = torch.norm(left_arm_xpos - target_pos, dim=-1)
        right_dist = torch.norm(right_arm_xpos - target_pos, dim=-1)
        if left_dist < right_dist:
            return "left_arm"
        else:
            return "right_arm"

    def create_demo_action_list(
        self, *args, **kwargs
    ) -> Optional[torch.Tensor | List[torch.Tensor]]:
        """Create a scripted stacking demo for stacking the top bowl into the bottom bowl.

        This produces a sequence of joint-space actions for the right arm and right gripper
        which (1) grasps the top bowl at its rim, (2) lifts it up, and
        (3) places it on top of the bottom bowl.
        """
        if getattr(self, "use_atomic_demo_actions", False):
            return self._create_atomic_demo_action_list()

        if self.num_envs > 1:
            logger.log_error(
                f"Demo trajectory generation is only supported for single environment. Got num_envs={self.num_envs}."
            )

        eef_open = self.robot.get_qpos_limits(name="left_eef")[:, :, 1]
        eef_close = self.robot.get_qpos_limits(name="left_eef")[:, :, 0]

        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)

        arm_to_use_mid = self.choice_arm_by_distance(bowl_mid_pose[:, :3, 3])
        pick_offset = (
            self.left_pick_offset
            if arm_to_use_mid == "left_arm"
            else self.right_pick_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_mid == "left_arm"
            else self.right_down_rota
        )
        bowl_mid_pick_pose = bowl_mid_pose.clone()
        bowl_mid_pick_pose[:, :3, :3] = down_rota
        bowl_mid_pick_pose[:, :3, 3] = bowl_mid_pick_pose[:, :3, 3] + pick_offset

        bowl_mid_pre_pick_pose = bowl_mid_pick_pose.clone()
        bowl_mid_pre_pick_pose[:, :3, 3] = (
            bowl_mid_pre_pick_pose[:, :3, 3] + self.bowl_mid_pre_pick_offset
        )
        arm_to_use_min = self.choice_arm_by_distance(bowl_min_pose[:, :3, 3])
        pick_offset = (
            self.left_pick_offset
            if arm_to_use_min == "left_arm"
            else self.right_pick_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_min == "left_arm"
            else self.right_down_rota
        )

        bowl_min_pick_pose = bowl_min_pose.clone()
        bowl_min_pick_pose[:, :3, :3] = down_rota
        bowl_min_pick_pose[:, :3, 3] = bowl_min_pick_pose[:, :3, 3] + pick_offset

        # Check whether arm is changed.
        return_pose = None
        if arm_to_use_mid != arm_to_use_min:
            return_pose = (
                self.init_left_arm_xpos
                if arm_to_use_mid == "left_arm"
                else self.init_right_arm_xpos
            )

        bowl_min_pre_pick_pose = bowl_min_pick_pose.clone()
        bowl_min_pre_pick_pose[:, :3, 3] = (
            bowl_min_pre_pick_pose[:, :3, 3] + self.bowl_min_pre_pick_offset
        )

        bowl_mid_place_offset = (
            self.left_bowl_mid_place_offset
            if arm_to_use_mid == "left_arm"
            else self.right_bowl_mid_place_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_mid == "left_arm"
            else self.right_down_rota
        )
        bowl_mid_place_pose = bowl_max_pose.clone()
        bowl_mid_place_pose[:, :3, :3] = down_rota
        bowl_mid_place_pose[:, :3, 3] = (
            bowl_mid_place_pose[:, :3, 3] + bowl_mid_place_offset
        )

        bowl_min_place_offset = (
            self.left_bowl_min_place_offset
            if arm_to_use_min == "left_arm"
            else self.right_bowl_min_place_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_min == "left_arm"
            else self.right_down_rota
        )
        bowl_min_place_pose = bowl_max_pose.clone()
        bowl_min_place_pose[:, :3, :3] = down_rota
        bowl_min_place_pose[:, :3, 3] = (
            bowl_min_place_pose[:, :3, 3] + bowl_min_place_offset
        )

        bowl_mid_mid_pose = (bowl_mid_pre_pick_pose + bowl_mid_place_pose) / 2.0
        bowl_min_mid_pose = (bowl_min_pre_pick_pose + bowl_min_place_pose) / 2.0

        mid_critical_pose = (
            bowl_min_mid_pose if arm_to_use_mid == arm_to_use_min else return_pose
        )

        start_pose = (
            self.init_left_arm_xpos
            if arm_to_use_mid == "left_arm"
            else self.init_right_arm_xpos
        )
        final_return_pose = (
            self.init_left_arm_xpos
            if arm_to_use_min == "left_arm"
            else self.init_right_arm_xpos
        )

        xpos_list = [
            start_pose,
            bowl_mid_pre_pick_pose,
            bowl_mid_pick_pose,
            bowl_mid_pick_pose,
            bowl_mid_pre_pick_pose,
            bowl_mid_mid_pose,
            bowl_mid_place_pose,
            bowl_mid_place_pose,
            mid_critical_pose,
            bowl_min_pre_pick_pose,
            bowl_min_pick_pose,
            bowl_min_pick_pose,
            bowl_min_pre_pick_pose,
            bowl_min_mid_pose,
            bowl_min_place_pose,
            bowl_min_place_pose,
            final_return_pose,
        ]

        eef_list = [
            eef_close,
            eef_open,
            eef_open,
            eef_close,
            eef_close,
            eef_close,
            eef_close,
            eef_open,
            eef_open,
            eef_open,
            eef_open,
            eef_close,
            eef_close,
            eef_close,
            eef_close,
            eef_open,
            eef_close,
        ]

        arm_type_list = [
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
        ]
        if return_pose is not None:
            arm_type_list[8] = (
                "right_arm" if arm_type_list[0] == "right_arm" else "left_arm"
            )

        # # [n_env, n_waypoints, dof]
        pack_traj = self._pack_trajectory(xpos_list, eef_list, arm_type_list)
        if pack_traj is None:
            return None

        pack_traj = pack_traj.permute(
            1, 0, 2
        )  # switch to (n_waypoints, n_env, dof) -> (n_env, n_waypoints, dof)
        interp_traj = interpolate_with_nums(
            pack_traj, interp_nums=self.interp_times, device=self.device
        )
        interp_traj = interp_traj.permute(
            1, 0, 2
        )  # switch back to (n_env, n_waypoints, dof) -> (n_waypoints, n_env, dof)

        return interp_traj[:, :, self.active_joint_ids]

    def _pack_trajectory(self, xpos_list, eef_list, arm_type_list):
        assert len(xpos_list) == len(
            eef_list
        ), "xpos_list and eef_list must have the same length."

        init_qpos = self.robot.get_qpos()
        n_waypoints = len(xpos_list)
        dof = init_qpos.shape[-1]
        n_env = init_qpos.shape[0]
        traj = torch.zeros(
            size=(n_waypoints, n_env, dof), dtype=torch.float32, device=self.device
        )
        traj[:] = init_qpos
        for i, (xpos, eef, arm_type) in enumerate(
            zip(xpos_list, eef_list, arm_type_list)
        ):
            arm_ids = (
                self.left_arm_ids if arm_type == "left_arm" else self.right_arm_ids
            )
            eef_ids = (
                self.left_eef_ids if arm_type == "left_arm" else self.right_eef_ids
            )

            is_success, qpos = self.robot.compute_ik(pose=xpos, name=arm_type)
            if not is_success:
                logger.log_warning(f"IK failed for {i}-th waypoint. Returning None.")
                return None

            traj[i, :, arm_ids] = qpos
            traj[i, :, eef_ids] = eef
        return traj

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Check if bowl_mid and bowl_min are successfully stacked on bowl_max.

        This checks the x/y position and rotation error of both bowl_mid and bowl_min relative to bowl_max.
        """
        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")

        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)

        # Compute x/y position difference relative to bowl_max
        mid_xy_diff = torch.norm(
            bowl_mid_pose[:, :2, 3] - bowl_max_pose[:, :2, 3], dim=-1
        )
        min_xy_diff = torch.norm(
            bowl_min_pose[:, :2, 3] - bowl_max_pose[:, :2, 3], dim=-1
        )
        xy_threshold = getattr(self.cfg.extensions, "success_xy_tol", 0.03)

        mid_success = mid_xy_diff < xy_threshold
        min_success = min_xy_diff < xy_threshold
        success = mid_success & min_success
        return success
