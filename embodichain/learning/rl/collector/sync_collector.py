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

from collections.abc import Callable
from contextlib import nullcontext

import torch
from tensordict import TensorDict

from embodichain.learning.rl.utils import (
    dict_to_tensordict,
    flatten_observation_groups,
)
from embodichain.learning.rl.utils.helper import _pack_observation_groups
from .base import BaseCollector

__all__ = ["SyncCollector"]


class SyncCollector(BaseCollector):
    """Synchronously collect rollouts from a vectorized environment.

    Args:
        env: Batched environment with automatic row resets.
        policy: Policy used for action and value inference.
        device: Device owning the rollout tensors.
        reset_every_rollout: Reset all environments before each collection.
        bootstrap_timeouts: Collect terminal values for algorithms using GAE.
    """

    def __init__(
        self,
        env,
        policy,
        device: torch.device,
        reset_every_rollout: bool = False,
        *,
        bootstrap_timeouts: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.device = device
        self.reset_every_rollout = reset_every_rollout
        self.bootstrap_timeouts = bootstrap_timeouts
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
        self.training_observation_groups = getattr(
            policy_module, "training_observation_groups", {}
        )
        self.training_field_specs = getattr(policy_module, "training_field_specs", {})
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
        if self.bootstrap_timeouts and "timeout_value" in rollout.keys():
            # A caller may reuse the TensorDict without starting a new buffer rollout.
            rollout["timeout_value"].fill_(float("nan"))
        scope = getattr(self.env, "_action_history_scope", None)
        with scope() if scope is not None else nullcontext() as update_history:
            if self.reset_every_rollout:
                self.obs_td = self._reset_env()
            if update_history is None and self._supports_shared_rollout:
                self.env.set_rollout_buffer(rollout)

            self._write_observations(rollout, 0, self.obs_td)
            for step_idx in range(num_steps):
                # Copy before env.step: auto-reset and reused observation buffers may
                # replace the labels with those of the following state.
                for name, groups in self.training_observation_groups.items():
                    labels = flatten_observation_groups(self.obs_td, groups)
                    target = rollout[name][:, step_idx]
                    if labels.numel() != target.numel():
                        raise ValueError(
                            f"Training observation '{name}' shape mismatch."
                        )
                    if labels.dtype != target.dtype:
                        raise TypeError(
                            f"Training observation '{name}' dtype mismatch."
                        )
                    target.copy_(labels.reshape_as(target))
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
                    self._to_env_action(step_td["action"])
                )
                next_obs_td = dict_to_tensordict(next_obs, self.device)
                self._write_observations(rollout, step_idx + 1, next_obs_td)
                self._write_timeout_values(
                    rollout, step_idx, env_info, terminated, truncated
                )
                self._write_step(
                    rollout=rollout,
                    step_idx=step_idx,
                    step_td=step_td,
                )
                if update_history is not None or not self._supports_shared_rollout:
                    self._write_env_step(
                        rollout=rollout,
                        step_idx=step_idx,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    )
                if update_history is not None:
                    update_history(
                        rollout["action"][:, step_idx], rollout["done"][:, step_idx]
                    )
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

    def _write_timeout_values(
        self,
        rollout: TensorDict,
        step_idx: int,
        env_info: dict,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Use pre-reset terminal observations for time-limit bootstrapping."""
        if not self.bootstrap_timeouts or "timeout_value" not in rollout.keys():
            return
        terminal = env_info.get("terminal_observation")
        if terminal is None:
            return
        timeout = truncated.to(self.device).bool() & ~terminated.to(self.device).bool()
        mask = env_info.get("terminal_observation_mask")
        if mask is not None:
            timeout &= mask.to(self.device).bool()
        env_ids = env_info.get("terminal_env_ids")
        if env_ids is None:
            rows = timeout.nonzero(as_tuple=False).squeeze(-1)
            env_ids = rows
        else:
            env_ids = env_ids.to(self.device)
            rows = timeout[env_ids].nonzero(as_tuple=False).squeeze(-1)
            env_ids = env_ids[rows]
        if rows.numel() == 0:
            return
        observation = dict_to_tensordict(terminal, self.device)
        actor_obs, critic_obs = self._model_observations(observation)
        # Reconstruct the complete pre-reset batch. Batch-dependent critics
        # still need peer rows in eval mode when only one row times out.
        fields = {"obs": rollout["obs"][:, step_idx + 1].clone()}
        terminal_ids = env_info.get("terminal_env_ids")
        if terminal_ids is None:
            fields["obs"].copy_(actor_obs)
        else:
            terminal_ids = terminal_ids.to(self.device)
            fields["obs"][terminal_ids] = actor_obs.to(fields["obs"].dtype)
        if critic_obs is not None:
            fields["critic_obs"] = rollout["critic_obs"][:, step_idx + 1].clone()
            if terminal_ids is None:
                fields["critic_obs"].copy_(critic_obs)
            else:
                fields["critic_obs"][terminal_ids] = critic_obs.to(
                    fields["critic_obs"].dtype
                )
        terminal_td = TensorDict(
            fields, batch_size=[self.env.num_envs], device=self.device
        )
        # Value inference must not update running statistics or sample Dropout.
        modes = [(module, module.training) for module in self.policy.modules()]
        try:
            self.policy.eval()
            self.policy.get_value(terminal_td)
        finally:
            for module, training in modes:
                module.training = training
        rollout["timeout_value"][env_ids, step_idx] = terminal_td["value"][env_ids].to(
            rollout["timeout_value"].dtype
        )

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

    def _write_observations(
        self, rollout: TensorDict, step_idx: int, observation: TensorDict
    ) -> None:
        """Snapshot raw model inputs directly into the owned rollout slot."""
        _pack_observation_groups(
            observation, self.actor_obs_groups, out=rollout["obs"][:, step_idx]
        )
        if self.uses_separate_critic_obs:
            _pack_observation_groups(
                observation,
                self.critic_obs_groups,
                out=rollout["critic_obs"][:, step_idx],
            )

    def _to_env_action(self, action: torch.Tensor) -> torch.Tensor:
        """Move a policy action to the environment device without preprocessing it.

        Action preprocessing belongs to :meth:`env.step`.
        """
        env_device = getattr(self.env, "device", self.device)
        return action.to(env_device)

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
        for name, (shape, _) in self.training_field_specs.items():
            expected_shapes[name] = (self.env.num_envs, num_steps + 1, *shape)
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(rollout[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Preallocated rollout field '{key}' shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}."
                )
