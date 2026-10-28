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

from __future__ import annotations

import torch
from typing import Dict, Any, Tuple

from embodichain.lab.gym.utils.registration import register_env
from collections.abc import Iterable, Sequence

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.types import EnvObs

__all__ = ["PushCubeEnv", "VisualPushCubeEnv"]


@register_env("PushCubeRL", max_episode_steps=50, override=True, supports_rl=True)
class PushCubeEnv(EmbodiedEnv):
    """Push cube task for reinforcement learning.

    The task involves pushing a cube to a target goal position using a robotic arm.
    The reward consists of reaching reward, placing reward, action penalty, and success bonus.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.get_local_pose(to_matrix=True)[:, :3, 3]

        # Check if goal_pose is defined (set by randomize_target_pose event)
        goal_pose = getattr(self, "goal_pose", None)
        if goal_pose is not None:
            goal_pos = goal_pose[:, :3, 3]
            xy_distance = torch.norm(cube_pos[:, :2] - goal_pos[:, :2], dim=1)
            is_success = xy_distance < self.success_threshold
        else:
            xy_distance = torch.zeros(self.num_envs, device=self.device)
            is_success = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics = {"distance_to_goal": xy_distance}

        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        cube = self.sim.get_rigid_object("cube")
        cube_pos = cube.get_local_pose(to_matrix=True)[:, :3, 3]
        is_fallen = cube_pos[:, 2] < -0.1
        return is_fallen


@register_env("PushCube-Franka-Visual-v1", max_episode_steps=300)
class VisualPushCubeEnv(EmbodiedEnv):
    """Push a cube into a visible target using the stock Franka embodiment.

    The expert consumes simulator geometry; policy observation and action
    contracts remain the camera, joint state and ActionManager commands.
    """

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        ids = torch.as_tensor(
            range(self.num_envs) if env_ids is None else env_ids,
            dtype=torch.long,
            device=self.device,
        )
        cube = self.sim.get_rigid_object("cube")
        pose = cube.get_local_pose().clone()
        pose[ids, :3] = torch.tensor([-0.52, -0.08, 0.02], device=self.device)
        pose[ids, :2] += (torch.rand(len(ids), 2, device=self.device) - 0.5) * 0.2
        pose[ids, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        cube.clear_dynamics(env_ids=ids)
        cube.set_local_pose(pose[ids], env_ids=ids)
        if not hasattr(self, "goal_position"):
            self.goal_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_position[ids] = pose[ids, :3]
        self.goal_position[ids, 0] += 0.2
        marker = self.sim.get_rigid_object("goal_marker")
        marker_pose = marker.get_local_pose().clone()
        marker_pose[ids, :2] = self.goal_position[ids, :2]
        marker.set_local_pose(marker_pose[ids], env_ids=ids)

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Measure goal distance and table retention from the simulated cube."""
        pose = self.sim.get_rigid_object("cube").get_local_pose()
        goal = getattr(self, "goal_position", torch.zeros_like(pose[:, :3]))
        distance = torch.linalg.vector_norm(pose[:, :2] - goal[:, :2], dim=-1)
        on_table = (pose[:, 2] >= 0.0) & (pose[:, 2] <= 0.08)
        return (
            (distance < 0.1) & on_table,
            pose[:, 2] < -0.05,
            {"goal_distance": distance},
        )

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return measured task success for recorder and expert validation."""
        if not hasattr(self, "goal_position"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.compute_task_state()[0]

    def _expert_waypoints(
        self, targets: torch.Tensor, steps: int
    ) -> Iterable[torch.Tensor]:
        arm_ids = self.robot.get_joint_ids("arm")
        start_qpos = self.robot.get_qpos()[:, arm_ids]
        start_pose = self.robot.compute_fk(start_qpos, name="arm", to_matrix=True)
        for index in range(steps):
            fraction = float(index + 1) / steps
            pose = targets.clone()
            pose[:, :3, 3] = (
                start_pose[:, :3, 3] * (1 - fraction) + targets[:, :3, 3] * fraction
            )
            seed = self.robot.get_qpos()[:, arm_ids]
            result = self.robot.compute_ik(pose, joint_seed=seed, name="arm")
            if result is None or not bool(torch.as_tensor(result[0]).all()):
                raise RuntimeError(f"PushCube expert IK failed at waypoint {index}")
            qpos = result[1].reshape(self.num_envs, -1)
            yield torch.cat(
                (qpos, torch.full((self.num_envs, 1), -1.0, device=self.device)), dim=-1
            )

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Produce bounded approach, descend and push action segments."""
        cube = self.sim.get_rigid_object("cube").get_local_pose()
        target = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        target[:, :3, :3] = torch.diag(
            torch.tensor([-1.0, 1.0, -1.0], device=self.device)
        )
        target[:, :3, 3] = cube[:, :3]
        target[:, 0, 3] -= 0.08
        target[:, 2, 3] = 0.18
        yield DemoSegment(
            name="approach",
            target_uid="cube",
            instruction="Move behind the cube.",
            actions=self._expert_waypoints(target, 65),
        )
        target = target.clone()
        target[:, 2, 3] = 0.025
        yield DemoSegment(
            name="descend",
            target_uid="cube",
            instruction="Lower the gripper.",
            actions=self._expert_waypoints(target, 35),
        )
        target = target.clone()
        target[:, 0, 3] += 0.22
        yield DemoSegment(
            name="push",
            target_uid="cube",
            instruction="Push the cube into the target.",
            actions=self._expert_waypoints(target, 100),
            validator=self.is_task_success,
        )
