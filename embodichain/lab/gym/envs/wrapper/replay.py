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

from typing import Any

import gymnasium as gym
import torch

from embodichain.lab.gym.utils.gym_utils import load_trajectory

__all__ = ["ReplayWrapper"]


class ReplayWrapper(gym.Wrapper):
    """Replay a recorded environment trajectory in pure-kinematic or dynamic mode.

    In ``kinematic`` mode physics is disabled and every recorded object's
    pose/qpos is written directly each step, producing observations only (no
    reward / success / action). In ``dynamic`` mode the recorded robot actions
    are fed back through :meth:`env.step` so physics re-simulates the scene;
    the full ``obs/reward/terminated/truncated/info`` tuple is returned.

    Args:
        env: The environment to wrap (constructed without ``record_trajectory``).
        trajectory: A ``.pt`` path or loaded dict from
            :meth:`EmbodiedEnv.save_trajectory`.
        mode: ``"kinematic"`` or ``"dynamic"``.
    """

    def __init__(
        self,
        env: gym.Env,
        trajectory: str | dict,
        mode: str = "dynamic",
    ):
        super().__init__(env)
        if mode not in ("kinematic", "dynamic"):
            raise ValueError(
                f"Invalid replay mode {mode!r}; use 'kinematic' or 'dynamic'."
            )
        self._mode = mode
        self._trajectory = load_trajectory(trajectory)
        self._num_steps = int(self._trajectory["meta"]["num_steps"])
        self._idx = 0
        self._expand_to_env_count()

    def _expand_to_env_count(self) -> None:
        """Broadcast a single-env trajectory to the wrapped env's env count."""
        traj_envs = int(self._trajectory["meta"]["num_envs"])
        env_envs = int(self.env.num_envs)
        if traj_envs == env_envs:
            return
        if traj_envs != 1:
            raise ValueError(
                f"Trajectory has {traj_envs} envs but wrapped env has {env_envs}; "
                "only single-env trajectories can be broadcast."
            )
        states = self._trajectory["states"]
        self._trajectory["states"] = states.expand(env_envs, *states.shape[1:]).clone()
        actions = self._trajectory["actions"]
        self._trajectory["actions"] = actions.expand(
            env_envs, *actions.shape[1:]
        ).clone()
        self._trajectory["meta"]["num_envs"] = env_envs

    def _set_all_states(self, states: Any) -> None:
        """Write one timestep's object states directly (kinematic write)."""
        env = self.env
        robot = env.robot
        non_mimic_ids = robot.get_joint_ids(remove_mimic=True)
        robot.set_local_pose(states["robot"]["root_pose"])
        robot.set_qpos(
            states["robot"]["qpos"][:, non_mimic_ids],
            joint_ids=non_mimic_ids,
            target=False,
        )
        if "articulations" in states.keys():
            for uid, art in env.sim._articulations.items():
                if uid in states["articulations"].keys():
                    art.set_local_pose(states["articulations"][uid]["root_pose"])
                    art.set_qpos(states["articulations"][uid]["qpos"], target=False)
        if "rigid_objects" in states.keys():
            for uid, obj in env.sim._rigid_objects.items():
                if uid in states["rigid_objects"].keys():
                    obj.set_local_pose(states["rigid_objects"][uid]["pose"])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Disable physics during restore so set_local_pose's internal update
        # does not integrate dynamics.
        self.env.sim.enable_physics(False)
        self._set_all_states(self._trajectory["states"][:, 0])
        if self._mode == "dynamic":
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = True
        self._idx = 0
        return self.env.get_obs(), info

    def step(self, action):
        env = self.env
        if self._idx >= self._num_steps:
            obs = env.get_obs()
            return (
                obs,
                torch.zeros(env.num_envs, device=env.device),
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                {},
            )

        if self._mode == "kinematic":
            self._set_all_states(self._trajectory["states"][:, self._idx])
            env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
            obs = env.get_obs()
            self._idx += 1
            trunc = self._idx >= self._num_steps
            return (
                obs,
                torch.zeros(env.num_envs, device=env.device),
                torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
                torch.full((env.num_envs,), trunc, dtype=torch.bool, device=env.device),
                {},
            )

        # dynamic
        action_t = self._trajectory["actions"][:, self._idx]
        obs, reward, term, trunc, info = env.step(action_t)
        self._idx += 1
        trunc = trunc | (self._idx >= self._num_steps)
        return obs, reward, term, trunc, info

    def close(self) -> None:
        try:
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = False
        finally:
            self.env.close()
