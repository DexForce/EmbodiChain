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

from typing import Callable

import torch
from tensordict import TensorDict

from embodichain.learning.rl.utils import (
    dict_to_tensordict,
    flatten_observation_groups,
)
from .base import BaseCollector

__all__ = ["SyncCollector"]


class SyncCollector(BaseCollector):
    """Synchronously collect rollouts from a vectorized environment."""

    def __init__(
        self,
        env,
        policy,
        device: torch.device,
        reset_every_rollout: bool = False,
    ) -> None:
        self.env = env
        self.policy = policy
        self.device = device
        self.reset_every_rollout = reset_every_rollout
        policy_module = getattr(policy, "module", policy)
        self.actor_obs_groups = getattr(policy_module, "actor_obs_groups", None)
        self.critic_obs_groups = getattr(policy_module, "critic_obs_groups", None)
        self.uses_separate_critic_obs = bool(
            getattr(policy_module, "uses_separate_critic_obs", False)
        )
        self.critic_obs_dim = getattr(policy_module, "critic_obs_dim", None)
        self.obs_dim = getattr(policy_module, "obs_dim", None)
        self.action_dim = getattr(policy_module, "action_dim", None)
        self.distribution_param_dim = getattr(
            policy_module, "distribution_param_dim", None
        )
        self._supports_shared_rollout = hasattr(self.env, "set_rollout_buffer")
        self.obs_td = self._reset_env()

    @torch.no_grad()
    def collect(
        self,
        num_steps: int,
        rollout: TensorDict | None = None,
        on_step_callback: Callable[[TensorDict, dict], None] | None = None,
    ) -> TensorDict:
        self.policy.train()
        if self.reset_every_rollout:
            self.obs_td = self._reset_env()

        if rollout is None:
            raise ValueError(
                "SyncCollector.collect() requires a preallocated rollout TensorDict."
            )
        if tuple(rollout.batch_size) != (self.env.num_envs, num_steps + 1):
            raise ValueError(
                "Preallocated rollout batch size mismatch: "
                f"expected ({self.env.num_envs}, {num_steps + 1}), got {tuple(rollout.batch_size)}."
            )
        self._validate_rollout(rollout, num_steps)
        if self._supports_shared_rollout:
            self.env.set_rollout_buffer(rollout)

        initial_obs, initial_critic_obs = self._model_observations(self.obs_td)
        rollout["obs"][:, 0] = initial_obs
        if initial_critic_obs is not None:
            rollout["critic_obs"][:, 0] = initial_critic_obs
        for step_idx in range(num_steps):
            step_fields = {"obs": rollout["obs"][:, step_idx]}
            if self.uses_separate_critic_obs:
                step_fields["critic_obs"] = rollout["critic_obs"][:, step_idx]
            step_td = TensorDict(
                step_fields,
                batch_size=[rollout.batch_size[0]],
                device=self.device,
            )
            step_td = self.policy.get_action(step_td)

            next_obs, reward, terminated, truncated, env_info = self.env.step(
                self._to_action_dict(step_td["action"])
            )
            next_obs_td = dict_to_tensordict(next_obs, self.device)
            self._write_step(
                rollout=rollout,
                step_idx=step_idx,
                step_td=step_td,
            )
            if not self._supports_shared_rollout:
                self._write_env_step(
                    rollout=rollout,
                    step_idx=step_idx,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            actor_obs, critic_obs = self._model_observations(next_obs_td)
            rollout["obs"][:, step_idx + 1] = actor_obs
            if critic_obs is not None:
                rollout["critic_obs"][:, step_idx + 1] = critic_obs
            self._update_policy_normalization(actor_obs, critic_obs)

            if on_step_callback is not None:
                on_step_callback(rollout[:, step_idx], env_info)

            self.obs_td = next_obs_td

        self._attach_final_value(rollout)
        return rollout

    def _attach_final_value(self, rollout: TensorDict) -> None:
        """Populate the bootstrap value for the final observed state."""
        final_fields = {"obs": rollout["obs"][:, -1]}
        if self.uses_separate_critic_obs:
            final_fields["critic_obs"] = rollout["critic_obs"][:, -1]
        last_next_td = TensorDict(
            final_fields,
            batch_size=[rollout.batch_size[0]],
            device=self.device,
        )
        self.policy.get_value(last_next_td)
        rollout["value"][:, -1] = last_next_td["value"]

    def _reset_env(self) -> TensorDict:
        obs, _ = self.env.reset()
        return dict_to_tensordict(obs, self.device)

    def _model_observations(
        self,
        observation: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build actor and optional privileged critic observations."""
        actor_obs = flatten_observation_groups(
            observation,
            self.actor_obs_groups,
        )
        if not self.uses_separate_critic_obs:
            return actor_obs, None
        critic_obs = flatten_observation_groups(
            observation,
            self.critic_obs_groups,
        )
        return actor_obs, critic_obs

    def _to_action_dict(self, action: torch.Tensor) -> TensorDict | torch.Tensor:
        am = getattr(self.env, "action_manager", None)
        if am is None:
            return action
        else:
            return am.convert_policy_action_to_env_action(action)

    def _update_policy_normalization(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor | None,
    ) -> None:
        """Update optional policy observation normalizers from the next state."""
        policy_module = getattr(self.policy, "module", self.policy)
        update = getattr(policy_module, "update_normalization", None)
        if update is None:
            return
        fields = {"obs": actor_obs}
        if critic_obs is not None:
            fields["critic_obs"] = critic_obs
        update(
            TensorDict(
                fields,
                batch_size=[actor_obs.shape[0]],
                device=self.device,
            )
        )

    def _write_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        step_td: TensorDict,
    ) -> None:
        """Write policy-side fields for one transition into the shared rollout TensorDict."""
        rollout["action"][:, step_idx] = step_td["action"]
        rollout["sample_log_prob"][:, step_idx] = step_td["sample_log_prob"]
        rollout["value"][:, step_idx] = step_td["value"]
        for name in ("action_mean", "action_std"):
            if name in rollout.keys():
                if name not in step_td.keys():
                    raise KeyError(
                        f"policy must return '{name}' for adaptive PPO scheduling"
                    )
                rollout[name][:, step_idx] = step_td[name]

    def _write_env_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Populate transition-side fields when the environment does not own the rollout."""
        done = terminated | truncated
        rollout["reward"][:, step_idx] = reward.to(self.device)
        rollout["done"][:, step_idx] = done.to(self.device)
        rollout["terminated"][:, step_idx] = terminated.to(self.device)
        rollout["truncated"][:, step_idx] = truncated.to(self.device)

    def _validate_rollout(self, rollout: TensorDict, num_steps: int) -> None:
        """Validate rollout layout expected by the collector."""
        expected_shapes = {
            "obs": (self.env.num_envs, num_steps + 1, self.obs_dim),
            "action": (self.env.num_envs, num_steps + 1, self.action_dim),
            "sample_log_prob": (self.env.num_envs, num_steps + 1),
            "value": (self.env.num_envs, num_steps + 1),
            "reward": (self.env.num_envs, num_steps + 1),
            "done": (self.env.num_envs, num_steps + 1),
            "terminated": (self.env.num_envs, num_steps + 1),
            "truncated": (self.env.num_envs, num_steps + 1),
        }
        if self.uses_separate_critic_obs:
            expected_shapes["critic_obs"] = (
                self.env.num_envs,
                num_steps + 1,
                self.critic_obs_dim,
            )
        if all(name in rollout.keys() for name in ("action_mean", "action_std")):
            for name in ("action_mean", "action_std"):
                expected_shapes[name] = (
                    self.env.num_envs,
                    num_steps + 1,
                    self.distribution_param_dim,
                )
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(rollout[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Preallocated rollout field '{key}' shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}."
                )
