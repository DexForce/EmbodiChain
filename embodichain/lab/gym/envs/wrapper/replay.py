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

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch

from embodichain.lab.gym.utils.gym_utils import load_trajectory
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.types import EnvObs

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
        meta = self._trajectory["meta"]

        # Sanity-check that the trajectory matches the replay env's robot.
        traj_robot_dof = int(meta.get("robot_dof", self.env.robot.dof))
        traj_active_joint_ids = list(meta.get("active_joint_ids", []))
        env_robot_dof = int(self.env.robot.dof)
        env_active_joint_ids = list(self.env.active_joint_ids)
        if (
            traj_robot_dof != env_robot_dof
            or traj_active_joint_ids != env_active_joint_ids
        ):
            raise ValueError(
                f"Trajectory was recorded with robot_dof={traj_robot_dof} / "
                f"active_joint_ids={traj_active_joint_ids} but replay env has "
                f"robot_dof={env_robot_dof} / active_joint_ids={env_active_joint_ids}."
            )

        self._expand_to_env_count()

        # Per-env lengths: fall back to uniform num_steps for legacy files.
        num_envs = int(self._trajectory["meta"]["num_envs"])
        lengths = meta.get("lengths", [int(meta["num_steps"])] * num_envs)
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=self.env.device)

        # Clamp replay length to the wrapped env's horizon.
        max_steps = int(self.env.max_episode_steps)
        if bool((self._lengths > max_steps).any()):
            logger.log_warning(
                f"Trajectory lengths exceed env max_episode_steps={max_steps}; clamping."
            )
            self._lengths = self._lengths.clamp(max=max_steps)
        self._replay_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )

    def _expand_to_env_count(self) -> None:
        """Broadcast a single-env trajectory to the wrapped env's env count."""
        meta = self._trajectory["meta"]
        traj_envs = int(meta["num_envs"])
        env_envs = int(self.env.num_envs)
        if traj_envs == env_envs:
            return
        if traj_envs != 1:
            raise ValueError(
                f"Trajectory has {traj_envs} envs but wrapped env has {env_envs}; "
                "only single-env trajectories can be broadcast."
            )
        for key in ("states", "actions"):
            t = self._trajectory[key]
            self._trajectory[key] = t.expand(env_envs, *t.shape[1:]).clone()
        meta["num_envs"] = env_envs
        if "lengths" in meta:
            meta["lengths"] = meta["lengths"] * env_envs

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
            traj_art_uids = set(states["articulations"].keys())
            scene_art_uids = set(env.sim._articulations.keys())
            for uid in traj_art_uids - scene_art_uids:
                logger.log_warning(
                    f"Trajectory articulation '{uid}' is not present in the scene; skipping."
                )
            for uid in traj_art_uids & scene_art_uids:
                art = env.sim._articulations[uid]
                art.set_local_pose(states["articulations"][uid]["root_pose"])
                art.set_qpos(states["articulations"][uid]["qpos"], target=False)
            for uid in scene_art_uids - traj_art_uids:
                logger.log_warning(
                    f"Scene articulation '{uid}' is not in the trajectory; leaving initial state."
                )
        if "rigid_objects" in states.keys():
            traj_rigid_uids = set(states["rigid_objects"].keys())
            scene_rigid_uids = set(env.sim._rigid_objects.keys())
            for uid in traj_rigid_uids - scene_rigid_uids:
                logger.log_warning(
                    f"Trajectory rigid object '{uid}' is not present in the scene; skipping."
                )
            for uid in traj_rigid_uids & scene_rigid_uids:
                obj = env.sim._rigid_objects[uid]
                obj.set_local_pose(states["rigid_objects"][uid]["pose"])
            for uid in scene_rigid_uids - traj_rigid_uids:
                logger.log_warning(
                    f"Scene rigid object '{uid}' is not in the trajectory; leaving initial state."
                )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[EnvObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Disable physics during restore so set_local_pose's internal update
        # does not integrate dynamics.
        self.env.sim.enable_physics(False)
        self._set_all_states(self._trajectory["states"][:, 0])
        if self._mode == "dynamic":
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = True
        self._replay_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        return self.env.get_obs(), info

    def step(
        self, action: Any
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        env = self.env
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        st = self._replay_steps.clamp(max=self._lengths - 1)  # finished envs hold last

        if self._mode == "kinematic":
            self._set_all_states(self._trajectory["states"][idx, st])
            env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
            obs = env.get_obs()
            self._replay_steps = (self._replay_steps + 1).clamp(max=self._lengths)
            trunc = self._replay_steps >= self._lengths
            return (
                obs,
                torch.zeros(n, device=env.device),
                torch.zeros(n, dtype=torch.bool, device=env.device),
                trunc,
                {},
            )

        # dynamic: feed the recorded (pre-process) action; env.step re-preprocesses.
        action_t = self._trajectory["actions"][idx, st]
        obs, reward, term, trunc, info = env.step(action_t)
        self._replay_steps = (self._replay_steps + 1).clamp(max=self._lengths)
        trunc = trunc | (self._replay_steps >= self._lengths)
        return obs, reward, term, trunc, info

    def go_to_step(self, step: int) -> EnvObs:
        """Scrub to a specific recorded step (kinematic).

        Disables physics, writes the recorded object states at ``step``, commits
        the scene, and returns the rendered observation. Used by interactive
        control mode (forward/back/jump are all O(1) since kinematic replay sets
        states directly). ``step`` is clamped to ``[0, min(lengths)-1]``.

        Args:
            step: Target step index.

        Returns:
            The observation at the target step.
        """
        env = self.env
        max_step = int(self._lengths.min().item()) - 1
        step = max(0, min(int(step), max_step))
        env.sim.enable_physics(False)
        self._set_all_states(self._trajectory["states"][:, step])
        env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
        self._replay_steps = torch.full(
            (env.num_envs,), step, dtype=torch.long, device=env.device
        )
        return env.get_obs()

    def close(self) -> None:
        try:
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = False
        finally:
            self.env.close()
