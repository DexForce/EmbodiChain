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

"""Run forward with a 17-joint Humanoid using effort control."""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils.math import quat_apply, quat_apply_inverse
from .mdp import humanoid_observation, humanoid_reward

__all__ = ["HumanoidRunEnv"]
_CONFIG = json.loads(
    files("embodichain_tasks.configs")
    .joinpath("tasks/classic_control/humanoid/task.json")
    .read_text()
)


@register_env("HumanoidRun-v1", max_episode_steps=900, override=True, supports_rl=True)
class HumanoidRunEnv(EmbodiedEnv):
    """Preserve the benchmark's 63 observations and clipped motor effort action."""

    def _init_sim_state(self, **kwargs: Any) -> None:
        super()._init_sim_state(**kwargs)
        if not math.isclose(self.physics_dt, _CONFIG["physics_dt"], abs_tol=1e-9):
            raise ValueError(
                "Humanoid physics timestep differs from the task contract."
            )
        if self.cfg.sim_steps_per_control != _CONFIG["decimation"]:
            raise ValueError(
                "Humanoid control decimation differs from the task contract."
            )
        self.add_detached_uids_for_reset([self.robot.uid])
        self.policy_joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in _CONFIG["joint_names"]],
            device=self.device,
        )
        self._lower = torch.tensor(_CONFIG["joint_lower"], device=self.device)
        self._upper = torch.tensor(_CONFIG["joint_upper"], device=self.device)
        self._gear = torch.tensor(
            [_CONFIG["action"]["joint_gears"][n] for n in _CONFIG["joint_names"]],
            device=self.device,
        )
        self._effort_limit = torch.tensor(_CONFIG["joint_effort"], device=self.device)
        self._action = torch.zeros((self.num_envs, 17), device=self.device)
        self._previous_potential = torch.full(
            (self.num_envs,), -1000.0 / self.physics_dt, device=self.device
        )
        self._progress = torch.zeros_like(self._previous_potential)
        self._values_cache = None
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(17,), dtype=np.float32
        )

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if not ids.numel():
            return
        pose = torch.zeros((len(ids), 7), device=self.device)
        pose[:, 2] = _CONFIG["spawn_height"]
        pose[:, 6] = 1.0
        self.robot.clear_dynamics(env_ids=ids)
        self.robot.set_local_pose(pose, env_ids=ids)
        qpos = (
            torch.zeros_like(self._lower)
            .clamp(self._lower, self._upper)
            .expand(len(ids), -1)
        )
        for target in (False, True):
            self.robot.set_qpos(
                qpos, joint_ids=self.policy_joint_ids, env_ids=ids, target=target
            )
            self.robot.set_qvel(
                torch.zeros_like(qpos),
                joint_ids=self.policy_joint_ids,
                env_ids=ids,
                target=target,
            )
        self.robot.set_qf(
            torch.zeros_like(qpos), joint_ids=self.policy_joint_ids, env_ids=ids
        )
        self._action[ids] = 0.0
        self._progress[ids] = 0.0
        self._previous_potential[ids] = -1000.0 / self.physics_dt
        self._values_cache = None

    def _step_action(self, action: torch.Tensor) -> torch.Tensor:
        self._action.copy_(
            action.clamp(-_CONFIG["action"]["clip"], _CONFIG["action"]["clip"])
        )
        effort = (self._action * self._gear).clamp(
            -self._effort_limit, self._effort_limit
        )
        self.robot.set_qf(effort, joint_ids=self.policy_joint_ids)
        return action

    def _values(self) -> dict[str, torch.Tensor]:
        if self._values_cache is not None:
            return self._values_cache
        data = self.robot.body_data
        pose = data.root_pose
        quat = pose[:, 3:7]
        x, y, z, w = quat.unbind(-1)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        direction = -pose[:, :3].clone()
        direction[:, 0] += 1000.0
        direction[:, 2] = 0.0
        distance = direction.norm(dim=-1)
        unit_direction = direction / distance.clamp_min(1e-9).unsqueeze(-1)
        up = torch.zeros_like(direction)
        up[:, 2] = 1.0
        heading = torch.zeros_like(direction)
        heading[:, 0] = 1.0
        qpos = self.robot.get_qpos()[:, self.policy_joint_ids]
        qvel = self.robot.get_qvel()[:, self.policy_joint_ids]
        values = {
            "height": pose[:, 2],
            "linear_velocity_local": quat_apply_inverse(quat, data.root_lin_vel),
            "angular_velocity_local": quat_apply_inverse(quat, data.root_ang_vel),
            "yaw": yaw,
            "roll": roll,
            "angle_to_target": torch.atan2(direction[:, 1], direction[:, 0]) - yaw,
            "up_projection": quat_apply(quat, up)[:, 2],
            "heading_projection": (quat_apply(quat, heading) * unit_direction).sum(-1),
            "scaled_joint_position": 2
            * (qpos - self._lower)
            / (self._upper - self._lower)
            - 1,
            "joint_velocity": qvel,
            "potential": -distance / self.physics_dt,
        }
        self._values_cache = values
        return values

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        self._values_cache = None
        potential = self._values()["potential"]
        self._progress.copy_(potential - self._previous_potential)
        self._previous_potential.copy_(potential)

    def _extend_obs(self, obs: Any, **kwargs: Any) -> Any:
        obs = super()._extend_obs(obs, **kwargs)
        v = self._values()
        reward = _CONFIG["reward"]
        policy = humanoid_observation(
            v["height"],
            v["linear_velocity_local"],
            v["angular_velocity_local"],
            v["yaw"],
            v["roll"],
            v["angle_to_target"],
            v["up_projection"],
            v["heading_projection"],
            v["scaled_joint_position"],
            v["joint_velocity"],
            self._action,
            angular_velocity_scale=reward["angular_velocity_scale"],
            joint_velocity_scale=reward["dof_velocity_scale"],
        )
        obs["policy"] = policy
        obs["critic"] = policy.clone()
        return obs

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Terminate fallen or nonfinite states; running has no success terminal."""
        v = self._values()
        finite = torch.stack(
            [torch.isfinite(t).reshape(self.num_envs, -1).all(-1) for t in v.values()]
        ).all(0)
        failed = (~finite) | (
            v["height"] < _CONFIG["termination"]["minimum_torso_height"]
        )
        return (
            torch.zeros_like(failed),
            failed,
            {"root_height": v["height"].clone(), "progress": self._progress.clone()},
        )

    def get_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        """Return progress, posture and motor-cost rewards from the benchmark."""
        v = self._values()
        r = _CONFIG["reward"]
        _, failed, _ = self.compute_task_state()
        return humanoid_reward(
            action=self._action,
            terminated=failed,
            heading_projection=v["heading_projection"],
            up_projection=v["up_projection"],
            joint_velocity=v["joint_velocity"],
            scaled_joint_position=v["scaled_joint_position"],
            progress=self._progress,
            motor_effort_ratio=torch.ones_like(self._gear),
            heading_weight=r["heading"],
            up_weight=r["up"],
            actions_cost_scale=r["actions_cost"],
            energy_cost_scale=r["energy_cost"],
            joint_velocity_scale=r["dof_velocity_scale"],
            death_cost=r["death"],
            alive_reward_scale=r["alive"],
        )
