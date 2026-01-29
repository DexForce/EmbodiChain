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

"""Base environment for reinforcement learning tasks."""

import torch
from typing import Dict, Any, Sequence, Optional, Tuple

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.utils.math import matrix_from_quat, matrix_from_euler


__all__ = ["RLEnv"]


class RLEnv(EmbodiedEnv):
    """Base class for reinforcement learning tasks.

    Provides common utilities for RL tasks:
    - Goal pose management and visualization
    - Flexible action preprocessing (scaling, IK, normalization)
    - Standardized info dictionary structure
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

        # Set default values for common RL parameters
        if not hasattr(self, "action_scale"):
            self.action_scale = 1.0
        if not hasattr(self, "episode_length"):
            self.episode_length = 1000

    @property
    def goal_pose(self) -> Optional[torch.Tensor]:
        """Get current goal poses (4x4 matrices) for all environments."""
        return self._goal_pose

    def draw_goal_marker(self, env_ids: Sequence[int] | None = None):
        """Draw axis markers at goal positions for visualization.

        Args:
            env_ids: Environment IDs to draw markers for (None = all envs)
        """
        if self.goal_pose is None:
            return

        if env_ids is None:
            env_ids = range(self.num_envs)

        for arena_idx in env_ids:
            marker_name = f"goal_marker_{arena_idx}"
            self.sim.remove_marker(marker_name)
            goal_pose = self.goal_pose[arena_idx].detach().cpu().numpy()
            marker_cfg = MarkerCfg(
                name=marker_name,
                marker_type="axis",
                axis_xpos=[goal_pose],
                axis_size=0.003,
                axis_len=0.02,
                arena_index=arena_idx,
            )
            self.sim.draw_marker(cfg=marker_cfg)

    def _preprocess_action(self, action: EnvAction) -> EnvAction:
        """Preprocess action for RL tasks with flexible transformation.

        Supports multiple action formats:
        1. Dict input (keys specify action type):
           - {"delta_qpos": tensor}: Delta joint positions (scaled and added to current)
           - {"qpos": tensor}: Absolute joint positions (scaled)
           - {"qpos_normalized": tensor}: Normalized qpos in [-1, 1]
           - {"eef_pose": tensor}: End-effector pose (6D or 7D) converted via IK
           - {"qvel": tensor}: Joint velocities (scaled)
           - {"qf": tensor}: Joint forces/torques (scaled)

        2. Tensor input: Interpreted based on self.action_type attribute
           (default: "qpos")

        Args:
            action: Raw action from policy (tensor or dict)

        Returns:
            Dict action ready for robot control
        """
        # Convert tensor input to dict based on action_type
        if not isinstance(action, dict):
            action_type = getattr(self, "action_type", "delta_qpos")
            action = {action_type: action}

        # Step 1: Scale all action values by action_scale
        scaled_action = {}
        for key, value in action.items():
            if isinstance(value, torch.Tensor):
                scaled_action[key] = value * self.action_scale
            else:
                scaled_action[key] = value

        # Step 2: Process based on dict keys
        result = {}

        if "qpos" in scaled_action:
            result["qpos"] = scaled_action["qpos"]
        elif "delta_qpos" in scaled_action:
            result["qpos"] = self._process_delta_qpos(scaled_action["delta_qpos"])
        elif "qpos_normalized" in scaled_action:
            result["qpos"] = self._denormalize_action(scaled_action["qpos_normalized"])
        elif "eef_pose" in scaled_action:
            result["qpos"] = self._process_eef_pose(scaled_action["eef_pose"])

        # Velocity and force controls
        if "qvel" in scaled_action:
            result["qvel"] = scaled_action["qvel"]

        if "qf" in scaled_action:
            result["qf"] = scaled_action["qf"]

        return result

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to actual range.

        Args:
            action: Normalized action in [-1, 1]

        Returns:
            Denormalized action
        """
        qpos_limits = self.robot.body_data.qpos_limits[0]
        low = qpos_limits[:, 0]
        high = qpos_limits[:, 1]

        # Map [-1, 1] to [low, high]
        return low + (action + 1.0) * 0.5 * (high - low)

    def _process_delta_qpos(self, action: torch.Tensor) -> torch.Tensor:
        """Process delta joint position action.

        Args:
            action: Delta joint positions

        Returns:
            Absolute joint positions
        """
        current_qpos = self.robot.get_qpos()
        return current_qpos + action

    def _process_eef_pose(self, action: torch.Tensor) -> torch.Tensor:
        """Process end-effector pose action via inverse kinematics.

        TODO: Currently only supports single-arm robots (6-axis or 7-axis).
        For multi-arm or complex robots, please use qpos/delta_qpos actions instead.

        Args:
            action: End-effector pose (position + orientation)
                   Shape: (num_envs, 6) for pos+euler or (num_envs, 7) for pos+quat

        Returns:
            Joint positions from IK
        """
        # Get current joint positions as IK seed
        current_qpos = self.robot.get_qpos()

        # Convert action to target pose matrix (4x4)
        batch_size = action.shape[0]
        target_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        if action.shape[-1] == 6:
            # pos (3) + euler angles (3)
            target_pose[:, :3, 3] = action[:, :3]
            target_pose[:, :3, :3] = matrix_from_euler(action[:, 3:6])
        elif action.shape[-1] == 7:
            # pos (3) + quaternion (4)
            target_pose[:, :3, 3] = action[:, :3]
            target_pose[:, :3, :3] = matrix_from_quat(action[:, 3:7])
        else:
            raise ValueError(
                f"EEF pose action must be 6D or 7D, got {action.shape[-1]}D"
            )

        # Solve IK for each environment
        ik_solutions = []
        for env_idx in range(self.num_envs):
            qpos_ik = self.robot.compute_ik(
                pose=target_pose[env_idx],
                joint_seed=current_qpos[env_idx],
            )
            ik_solutions.append(qpos_ik)

        # Stack IK solutions
        result_qpos = torch.stack(ik_solutions, dim=0)

        return result_qpos

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute task-specific state: success, fail, and metrics.

        Override this method in subclass to define task-specific logic.

        Returns:
            Tuple of (success, fail, metrics):
                - success: Boolean tensor of shape (num_envs,)
                - fail: Boolean tensor of shape (num_envs,)
                - metrics: Dict of metric tensors
        """
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics = {}
        return success, fail, metrics

    def get_info(self, **kwargs) -> Dict[str, Any]:
        """Get environment info dictionary.

        Calls compute_task_state() to get task-specific success/fail/metrics.
        Subclasses should override compute_task_state() instead of this method.

        Returns:
            Info dictionary with success, fail, elapsed_steps, goal_pose, metrics
        """
        success, fail, metrics = self.compute_task_state(**kwargs)

        info = {
            "success": success,
            "fail": fail,
            "elapsed_steps": self._elapsed_steps,
        }
        if hasattr(self, "_goal_pose") and self._goal_pose is not None:
            info["goal_pose"] = self._goal_pose
        info["metrics"] = metrics
        return info

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        """Check if episode should be truncated (timeout).

        Args:
            obs: Current observation
            info: Info dictionary

        Returns:
            Boolean tensor of shape (num_envs,)
        """
        return self._elapsed_steps >= self.episode_length

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Evaluate the environment state.

        Returns:
            Evaluation dictionary with success and metrics
        """
        info = self.get_info(**kwargs)
        eval_dict = {
            "success": info["success"][0].item(),
        }
        if "metrics" in info:
            for key, value in info["metrics"].items():
                eval_dict[key] = value
        return eval_dict
