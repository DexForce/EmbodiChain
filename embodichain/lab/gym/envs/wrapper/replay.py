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
from tensordict import TensorDict

from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.gym_utils import load_trajectory
from embodichain.lab.gym.utils.trajectory_state import restore_trajectory_state
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.types import EnvObs

__all__ = ["ReplayWrapper"]


class ReplayWrapper(gym.Wrapper):
    """Replay a recorded environment trajectory.

    In ``kinematic`` mode physics is disabled and every recorded object's
    pose/qpos is written directly each step, producing observations only (no
    reward / success / action). In ``dynamic`` mode the recorded robot actions
    are fed back through :meth:`env.step` so physics re-simulates the scene;
    the full ``obs/reward/terminated/truncated/info`` tuple is returned. Flat
    position-velocity expert actions are reconstructed as controller commands
    before they are fed back. The ``control`` mode uses the same kinematic
    behavior while exposing :meth:`go_to_step` for interactive scrubbing.

    Args:
        env: The environment to wrap (constructed without ``record_trajectory``).
        trajectory: A ``.pt`` path or loaded dict from
            :meth:`EmbodiedEnv.save_trajectory`.
        mode: ``"kinematic"``, ``"dynamic"``, or ``"control"``.
    """

    def __init__(
        self,
        env: gym.Env,
        trajectory: str | dict,
        mode: str = "dynamic",
    ):
        super().__init__(env)
        if mode not in ("kinematic", "dynamic", "control"):
            raise ValueError(
                f"Invalid replay mode {mode!r}; use 'kinematic', 'dynamic', or "
                "'control'."
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

        # Per-env lengths support async vector trajectories.
        lengths = meta["lengths"]
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
        meta["lengths"] = list(meta["lengths"]) * env_envs

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[EnvObs, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Disable physics during restore so set_local_pose's internal update
        # does not integrate dynamics.
        self.env.sim.enable_physics(False)
        restore_trajectory_state(self.env, self._trajectory["states"][:, 0])
        if self._mode == "dynamic":
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = True
        self._replay_steps = torch.zeros(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        obs = self.env.get_obs()

        # If the wrapped environment also records this replay, replace the
        # default-reset pending state with the state restored from the file.
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        seed_recording_state = getattr(self.env, "_seed_recording_state", None)
        if seed_recording_state is not None:
            seed_recording_state(obs, env_ids)
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        env = self.env
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        st = self._replay_steps.clamp(max=self._lengths - 1)  # finished envs hold last

        if self._mode in ("kinematic", "control"):
            restore_trajectory_state(self.env, self._trajectory["states"][idx, st])
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
        obs, reward, term, trunc, info = env.step(self._decode_dynamic_action(action_t))
        self._replay_steps = (self._replay_steps + 1).clamp(max=self._lengths)
        trunc = trunc | (self._replay_steps >= self._lengths)
        return obs, reward, term, trunc, info

    def _decode_dynamic_action(
        self, action: torch.Tensor
    ) -> torch.Tensor | ControllerAction:
        """Convert persisted expert actions back to controller commands.

        Position-velocity expert trajectories are stored as one flat vector so
        they can be serialized alongside position-only actions.  The environment
        control boundary expects a ``ControllerAction`` containing separate
        ``qpos`` and ``qvel`` entries, so reconstruct that envelope before
        calling ``env.step``. Policy actions and position-only expert actions
        already use the environment's native flat representation.
        """
        meta = self._trajectory["meta"]
        if (
            meta.get("action_kind", "expert") != "expert"
            or meta.get("joint_command_mode") != "position_velocity"
        ):
            return action

        qpos_slice = meta.get("qpos_slice")
        qvel_slice = meta.get("qvel_slice")
        if (
            not isinstance(qpos_slice, (list, tuple))
            or len(qpos_slice) != 2
            or not isinstance(qvel_slice, (list, tuple))
            or len(qvel_slice) != 2
        ):
            raise ValueError(
                "Position-velocity trajectory metadata must define qpos_slice "
                "and qvel_slice."
            )

        try:
            qpos_start, qpos_end = (int(value) for value in qpos_slice)
            qvel_start, qvel_end = (int(value) for value in qvel_slice)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Position-velocity trajectory slices must contain integers."
            ) from error

        width = int(action.shape[-1])
        if not (
            0 <= qpos_start < qpos_end <= width and 0 <= qvel_start < qvel_end <= width
        ):
            raise ValueError(
                "Position-velocity trajectory slices exceed the stored action "
                f"width {width}: qpos={qpos_slice}, qvel={qvel_slice}."
            )

        return ControllerAction(
            value=TensorDict(
                {
                    "qpos": action[..., qpos_start:qpos_end],
                    "qvel": action[..., qvel_start:qvel_end],
                },
                batch_size=[self.env.num_envs],
                device=action.device,
            )
        )

    def go_to_step(self, step: int) -> EnvObs:
        """Scrub to a specific recorded state (kinematic).

        State index ``t`` is the state immediately before recorded action ``t``.

        Args:
            step: Target step index.

        Returns:
            The observation at the target step.
        """
        env = self.env
        max_step = self.control_max_step
        step = max(0, min(int(step), max_step))
        env.sim.enable_physics(False)
        restore_trajectory_state(env, self._trajectory["states"][:, step])
        env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
        self._replay_steps = torch.full(
            (env.num_envs,), step, dtype=torch.long, device=env.device
        )
        return env.get_obs()

    @property
    def control_max_step(self) -> int:
        """Largest state index available to interactive control replay."""
        transition_count = int(self._lengths.min().item())
        return transition_count - 1

    def close(self) -> None:
        try:
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = False
        finally:
            self.env.close()
