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

import torch
import numpy as np

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.planners.motion_generator import MotionGenerator
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.utils import logger

__all__ = ["BlocksRankingRGBEnv"]


@register_env("BlocksRankingRGB-v1", max_episode_steps=600)
class BlocksRankingRGBEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

    def create_demo_action_list(self, *args, **kwargs):
        """
        Create a demonstration action list for ranking three blocks in RGB order from left to right.

        Now the expert trajectory follows the following strategy:
        - Do not move the green block block_2 (as the middle reference)
        - Right hand (right_arm + right_eef) takes the red block block_1 and places it on the left of the green block
        - Left hand (left_arm + left_eef) takes the blue block block_3 and places it on the right of the green block

        Returns:
            list: A list of demo actions (torch.Tensor) to be executed by env.step().
        """
        try:
            block1 = self.sim.get_rigid_object("block_1")  # Red
            block2 = self.sim.get_rigid_object("block_2")  # Green
            block3 = self.sim.get_rigid_object("block_3")  # Blue
        except Exception as e:
            logger.log_warning(f"Blocks not found: {e}, returning empty action list.")
            return []

        # Get block poses and positions
        b1_pose = block1.get_local_pose(to_matrix=True)
        b2_pose = block2.get_local_pose(to_matrix=True)
        b3_pose = block3.get_local_pose(to_matrix=True)
        b1_pos = b1_pose[:, :3, 3]
        b2_pos = b2_pose[:, :3, 3]
        b3_pos = b3_pose[:, :3, 3]

        # Construct the target line centered on the green block (Red < Green < Blue)
        base_x = b2_pos[:, 0]
        base_y = b2_pos[:, 1]
        base_z = b2_pos[:, 2]

        tgt_green = b2_pos  # not moved, kept for reference
        tgt_red = torch.stack([base_x - 0.12, base_y, base_z], dim=1)
        tgt_blue = torch.stack([base_x + 0.10, base_y, base_z], dim=1)

        # Right arm / right hand
        right_arm_ids = self.robot.get_joint_ids(name="right_arm")
        right_eef_ids = self.robot.get_joint_ids(name="right_eef")
        # Left arm / left hand
        left_arm_ids = self.robot.get_joint_ids(name="left_arm")
        left_eef_ids = self.robot.get_joint_ids(name="left_eef")

        init_qpos = self.robot.get_qpos()
        init_right_arm_qpos = init_qpos[:, right_arm_ids]
        init_right_arm_xpos = self.robot.compute_fk(
            qpos=init_right_arm_qpos, name="right_arm", to_matrix=True
        )
        init_left_arm_qpos = init_qpos[:, left_arm_ids]
        init_left_arm_xpos = self.robot.compute_fk(
            qpos=init_left_arm_qpos, name="left_arm", to_matrix=True
        )

        motion_gen_right = MotionGenerator(
            robot=self.robot,
            uid="right_arm",
            planner_type="toppra",
            default_velocity=0.2,
            default_acceleration=0.5,
        )
        motion_gen_left = MotionGenerator(
            robot=self.robot,
            uid="left_arm",
            planner_type="toppra",
            default_velocity=0.2,
            default_acceleration=0.5,
        )

        gripper_open = torch.tensor(
            [0.05, 0.05], dtype=torch.float32, device=self.device
        )
        gripper_close = torch.tensor(
            [0.0, 0.0], dtype=torch.float32, device=self.device
        )

        action_list = []

        def _ik_to_qpos(
            target_xpos: torch.Tensor, seed: torch.Tensor, arm_name: str, name: str
        ):
            is_success, qpos = self.robot.compute_ik(
                pose=target_xpos, joint_seed=seed, name=arm_name
            )
            success_flag = (
                is_success.all() if isinstance(is_success, torch.Tensor) else is_success
            )
            if not success_flag:
                logger.log_warning(f"IK failed for {name}, using previous qpos.")
                qpos = seed
            return qpos

        def _append_hold(
            qpos: torch.Tensor,
            num_steps: int,
            gripper_state: torch.Tensor,
            arm_ids,
            eef_ids,
        ):
            for _ in range(num_steps):
                action = init_qpos.clone()
                action[:, arm_ids] = qpos
                action[:, eef_ids] = gripper_state.unsqueeze(0).expand(
                    self.num_envs, -1
                )
                action_list.append(action)

        def _append_move_for_arm(
            qpos_start: torch.Tensor,
            qpos_end: torch.Tensor,
            num_steps: int,
            gripper_state: torch.Tensor,
            arm_ids,
            eef_ids,
            motion_gen: MotionGenerator,
        ):
            qpos_list = [
                qpos_start[0].detach().cpu().numpy(),
                qpos_end[0].detach().cpu().numpy(),
            ]
            out_qpos_list, _ = motion_gen.create_discrete_trajectory(
                qpos_list=qpos_list,
                is_linear=False,
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_num=num_steps,
                is_use_current_qpos=False,
            )
            for qpos_item in out_qpos_list:
                qpos = torch.as_tensor(
                    qpos_item, dtype=torch.float32, device=self.device
                )
                qpos = qpos.flatten()
                if qpos.shape[0] != len(arm_ids):
                    logger.log_warning(
                        f"Qpos shape mismatch: got {qpos.shape[0]}, expected {len(arm_ids)}"
                    )
                    continue
                qpos = qpos.unsqueeze(0).expand(self.num_envs, -1)
                action = init_qpos.clone()
                action[:, arm_ids] = qpos
                action[:, eef_ids] = gripper_state.unsqueeze(0).expand(
                    self.num_envs, -1
                )
                action_list.append(action)

        def _pick_and_place(
            block_pos: torch.Tensor,
            place_pos: torch.Tensor,
            seed_qpos: torch.Tensor,
            init_arm_xpos: torch.Tensor,
            arm_ids,
            eef_ids,
            arm_name: str,
            motion_gen: MotionGenerator,
            tag: str,
        ):
            pick = init_arm_xpos.clone()
            pick[:, :3, 3] = block_pos + torch.tensor(
                [0.02, 0.0, -0.025], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            lift = pick.clone()
            lift[:, 2, 3] += 0.15

            place = init_arm_xpos.clone()
            place[:, :3, 3] = place_pos + torch.tensor(
                [0.025, 0.0, 0.02], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # inverse kinematics for three key poses
            q_pick = _ik_to_qpos(pick, seed_qpos, arm_name, f"{tag}_pick")
            q_lift = _ik_to_qpos(lift, q_pick, arm_name, f"{tag}_lift")
            q_place = _ik_to_qpos(place, q_lift, arm_name, f"{tag}_place")

            # execution segments: seed -> pick(open gripper) -> lift -> place(close gripper)
            _append_move_for_arm(
                seed_qpos, q_pick, 20, gripper_open, arm_ids, eef_ids, motion_gen
            )
            _append_hold(q_pick, 5, gripper_close, arm_ids, eef_ids)  # close gripper
            _append_move_for_arm(
                q_pick, q_lift, 20, gripper_close, arm_ids, eef_ids, motion_gen
            )
            _append_move_for_arm(
                q_lift, q_place, 30, gripper_close, arm_ids, eef_ids, motion_gen
            )
            _append_hold(q_place, 5, gripper_open, arm_ids, eef_ids)  # open gripper

            return q_place

        # 1. Right hand handles the red block
        current_seed_right = init_right_arm_qpos
        current_seed_right = _pick_and_place(
            b1_pos,
            tgt_red,
            current_seed_right,
            init_right_arm_xpos,
            right_arm_ids,
            right_eef_ids,
            "right_arm",
            motion_gen_right,
            "red",
        )

        init_qpos[:, right_arm_ids] = current_seed_right

        # 2. Left hand handles the blue block
        current_seed_left = init_left_arm_qpos
        current_seed_left = _pick_and_place(
            b3_pos,
            tgt_blue,
            current_seed_left,
            init_left_arm_xpos,
            left_arm_ids,
            left_eef_ids,
            "left_arm",
            motion_gen_left,
            "blue",
        )

        logger.log_info(f"Generated {len(action_list)} demo actions for RGB ranking")
        self.action_length = len(action_list)
        return action_list

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Three blocks are arranged in RGB order from front to back:
           - Red block (block_1) x < Green block (block_2) x < Blue block (block_3) x
        2. All blocks are close together (within tolerance)

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            block1 = self.sim.get_rigid_object("block_1")  # Red
            block2 = self.sim.get_rigid_object("block_2")  # Green
            block3 = self.sim.get_rigid_object("block_3")  # Blue
        except Exception as e:
            logger.log_warning(f"Blocks not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get block poses
        block1_pose = block1.get_local_pose(to_matrix=True)
        block2_pose = block2.get_local_pose(to_matrix=True)
        block3_pose = block3.get_local_pose(to_matrix=True)

        # Extract positions (x, y, z)
        block1_pos = block1_pose[:, :3, 3]  # (num_envs, 3)
        block2_pos = block2_pose[:, :3, 3]
        block3_pos = block3_pose[:, :3, 3]

        # Tolerance for checking if blocks are close together
        eps = torch.tensor([0.13, 0.03], dtype=torch.float32, device=self.device)

        # Check if blocks are close together in x-y plane
        # block1 and block2 should be close
        block1_block2_diff = torch.abs(block1_pos[:, :2] - block2_pos[:, :2])
        blocks_close_12 = torch.all(block1_block2_diff < eps.unsqueeze(0), dim=1)

        # block2 and block3 should be close
        block2_block3_diff = torch.abs(block2_pos[:, :2] - block3_pos[:, :2])
        blocks_close_23 = torch.all(block2_block3_diff < eps.unsqueeze(0), dim=1)

        # Check RGB order: block1 (red) x < block2 (green) x < block3 (blue) x
        rgb_order = (block1_pos[:, 0] < block2_pos[:, 0]) & (
            block2_pos[:, 0] < block3_pos[:, 0]
        )

        # Task succeeds if blocks are close together and in RGB order
        success = blocks_close_12 & blocks_close_23 & rgb_order

        return success
