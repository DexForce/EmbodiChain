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

from embodichain.agents.rl.utils import dict_to_tensordict, flatten_dict_observation
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
        if tuple(rollout.batch_size) != (self.env.num_envs, num_steps):
            raise ValueError(
                "Preallocated rollout batch size mismatch: "
                f"expected ({self.env.num_envs}, {num_steps}), got {tuple(rollout.batch_size)}."
            )
        if self._supports_shared_rollout:
            self.env.set_rollout_buffer(rollout)

        for step_idx in range(num_steps):
            obs_tensor = flatten_dict_observation(self.obs_td)
            step_td = TensorDict(
                {"obs": obs_tensor},
                batch_size=[obs_tensor.shape[0]],
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
                    next_obs_td=next_obs_td,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )

            if on_step_callback is not None:
                on_step_callback(rollout[:, step_idx], env_info)

            self.obs_td = next_obs_td

        self._attach_next_values(rollout)
        return rollout

    def _attach_next_values(self, rollout: TensorDict) -> None:
        """Populate `next.value` for GAE bootstrap."""
        next_values = torch.zeros_like(rollout["value"])
        next_values[:, :-1] = rollout["value"][:, 1:]

        last_next_td = TensorDict(
            {"obs": rollout["next", "obs"][:, -1]},
            batch_size=[rollout.batch_size[0]],
            device=self.device,
        )
        self.policy.get_value(last_next_td)
        next_values[:, -1] = last_next_td["value"]
        rollout["next", "value"] = next_values

    def _reset_env(self) -> TensorDict:
        obs, _ = self.env.reset()
        return dict_to_tensordict(obs, self.device)

    def _to_action_dict(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        am = getattr(self.env, "action_manager", None)
        action_type = (
            am.action_type if am else getattr(self.env, "action_type", "delta_qpos")
        )
        return {action_type: action}

    def _write_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        step_td: TensorDict,
    ) -> None:
        """Write policy-side fields for one transition into the shared rollout TensorDict."""
        rollout["obs"][:, step_idx] = step_td["obs"]
        rollout["action"][:, step_idx] = step_td["action"]
        rollout["sample_log_prob"][:, step_idx] = step_td["sample_log_prob"]
        rollout["value"][:, step_idx] = step_td["value"]

    def _write_env_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        next_obs_td: TensorDict,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Populate transition-side fields when the environment does not own the rollout."""
        done = terminated | truncated
        rollout["next", "obs"][:, step_idx] = flatten_dict_observation(next_obs_td)
        rollout["next", "reward"][:, step_idx] = reward.to(self.device)
        rollout["next", "done"][:, step_idx] = done.to(self.device)
        rollout["next", "terminated"][:, step_idx] = terminated.to(self.device)
        rollout["next", "truncated"][:, step_idx] = truncated.to(self.device)
