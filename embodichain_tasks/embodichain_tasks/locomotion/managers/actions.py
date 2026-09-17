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

"""Action terms for native locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from embodichain.lab.gym.envs.managers import ActionTerm
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["DefaultJointPositionTerm", "DelayedDefaultJointPositionTerm"]


class DefaultJointPositionTerm(ActionTerm):
    """Map normalized policy actions to named default-offset joint targets."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        """Initialize the manager term from the environment and configuration.

        Args:
            cfg: Manager configuration containing the term parameters.
            env: Environment providing the task state and robot.
        """
        super().__init__(cfg, env)
        clip = cfg.params.get("clip")
        self._clip = None if clip is None else float(clip)
        task_config = getattr(env, "locomotion_task_config", None)
        if task_config is None:
            task_config = getattr(env, "velocity_task_config", None)
        joint_names: Sequence[str] = cfg.params.get(
            "joint_names",
            () if task_config is None else task_config.joint_names,
        )
        if not joint_names:
            raise ValueError("DefaultJointPositionTerm requires joint_names.")
        missing = [name for name in joint_names if name not in env.robot.joint_names]
        if missing:
            raise ValueError(f"Unknown locomotion joint names: {missing}")
        self._runtime_joint_ids = torch.tensor(
            [env.robot.joint_names.index(name) for name in joint_names],
            dtype=torch.long,
            device=env.device,
        )
        active_joint_ids = torch.as_tensor(
            env.active_joint_ids,
            dtype=torch.long,
            device=env.device,
        )
        if not torch.equal(active_joint_ids, self._runtime_joint_ids):
            active_joint_names = tuple(
                env.robot.joint_names[index]
                for index in active_joint_ids.detach().cpu().tolist()
            )
            raise ValueError(
                "Locomotion joint_names must match the environment's active-joint "
                f"order; configured={tuple(joint_names)}, active={active_joint_names}."
            )
        self._default_active_qpos = torch.as_tensor(
            env.robot.cfg.init_qpos,
            dtype=torch.float32,
            device=env.device,
        )[self._runtime_joint_ids].unsqueeze(0)
        if task_config is not None:
            task_default_qpos = torch.as_tensor(
                task_config.default_joint_position,
                dtype=torch.float32,
                device=env.device,
            ).unsqueeze(0)
            if task_default_qpos.shape != self._default_active_qpos.shape:
                raise ValueError(
                    "Task default joint positions must contain one value per "
                    f"active joint; received {task_default_qpos.shape[-1]} values "
                    f"for {self.action_dim} joints."
                )
            mismatch = ~torch.isclose(
                task_default_qpos,
                self._default_active_qpos,
                atol=1.0e-6,
                rtol=0.0,
            )[0]
            if bool(mismatch.any()):
                mismatched_names = [
                    joint_names[index]
                    for index in mismatch.nonzero(as_tuple=False).flatten().tolist()
                ]
                raise ValueError(
                    "Robot init_qpos does not match the task default joint "
                    f"positions for joints {mismatched_names}."
                )
        scale = torch.as_tensor(
            cfg.params.get(
                "scale",
                1.0 if task_config is None else task_config.action_scale,
            ),
            dtype=torch.float32,
            device=env.device,
        )
        if scale.ndim == 0:
            scale = scale.expand(self.action_dim)
        if scale.shape != (self.action_dim,):
            raise ValueError(
                "Locomotion action scale must be a scalar or contain one value "
                f"per joint; received shape {tuple(scale.shape)} for "
                f"{self.action_dim} joints."
            )
        self._scale = scale.unsqueeze(0)

    @property
    def input_key(self) -> str:
        """Return the robot command type produced by this term."""
        return "qpos"

    @property
    def action_dim(self) -> int:
        """Return the number of policy-ordered joints."""
        return int(self._runtime_joint_ids.numel())

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip and apply one policy action around the named default pose.

        Args:
            action: Policy actions with one row per environment and one column per controlled joint.

        Returns:
            Joint position targets in the environment active-joint order.
        """
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} locomotion actions, got "
                f"{action.shape[-1]}."
            )
        normalized = (
            action if self._clip is None else action.clamp(-self._clip, self._clip)
        )
        record_action = getattr(self._env, "record_locomotion_action", None)
        if callable(record_action):
            record_action(normalized)
        target = self._default_active_qpos + normalized * self._scale
        encoder_bias = getattr(self._env, "encoder_bias", None)
        return target if encoder_bias is None else target - encoder_bias


class DelayedDefaultJointPositionTerm(DefaultJointPositionTerm):
    """Apply per-environment physics-step delay to joint position targets."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv) -> None:
        """Initialize the manager term from the environment and configuration.

        Args:
            cfg: Manager configuration containing the term parameters.
            env: Environment providing the task state and robot.
        """
        super().__init__(cfg, env)
        self._minimum_delay = int(cfg.params.get("minimum_delay", 0))
        self._maximum_delay = int(cfg.params.get("maximum_delay", 0))
        if self._minimum_delay < 0 or self._maximum_delay < self._minimum_delay:
            raise ValueError(
                "Action delay must satisfy 0 <= minimum_delay <= maximum_delay."
            )
        history_length = self._maximum_delay + 1
        default_target = self._default_active_qpos.expand(env.num_envs, -1).clone()
        self._history = default_target.unsqueeze(0).repeat(history_length, 1, 1)
        self._pending_target = default_target.clone()
        self._output_target = default_target.clone()
        self._cursor = 0
        self._time_lag = torch.full(
            (env.num_envs,),
            self._minimum_delay,
            dtype=torch.long,
            device=env.device,
        )

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        """Store the latest target and return the currently delayed target.

        Args:
            action: Policy actions with one row per environment and one column per controlled joint.

        Returns:
            Joint position targets in the environment active-joint order.
        """
        self._pending_target.copy_(super().process_action(action))
        immediate = self._time_lag == 0
        self._output_target[immediate] = self._pending_target[immediate]
        return self._output_target

    def reset(
        self,
        env_ids: torch.Tensor,
        initial_target: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """Reset delay history and sample one lag for selected environments.

        Args:
            env_ids: Selected environment rows; None selects all rows when supported.
            initial_target: Joint targets for the selected rows used to initialize delay history.
            generator: Random generator used to sample the task noise or delays.
        """
        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        initial_target = initial_target.to(device=self._env.device, dtype=torch.float32)
        if initial_target.shape != (len(env_ids), self.action_dim):
            raise ValueError(
                "Initial delayed target must contain one value per selected "
                "environment and joint."
            )
        self._history[:, env_ids] = initial_target.unsqueeze(0)
        self._pending_target[env_ids] = initial_target
        self._output_target[env_ids] = initial_target
        self._time_lag[env_ids] = torch.randint(
            self._minimum_delay,
            self._maximum_delay + 1,
            (len(env_ids),),
            generator=generator,
            device=self._env.device,
        )

    def advance_physics_step(self) -> None:
        """Advance the target history and apply the next delayed setpoint."""
        self._cursor = (self._cursor + 1) % self._history.shape[0]
        self._history[self._cursor].copy_(self._pending_target)
        env_ids = torch.arange(
            self._env.num_envs, dtype=torch.long, device=self._env.device
        )
        history_offset = torch.clamp(self._time_lag - 1, min=0)
        delayed_indices = (self._cursor - history_offset) % self._history.shape[0]
        self._output_target.copy_(self._history[delayed_indices, env_ids])
        self._env.robot.set_qpos(
            self._output_target,
            joint_ids=self._runtime_joint_ids,
            target=True,
        )
