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
        self.obs_td = self._reset_env()

    def collect(
        self,
        num_steps: int,
        on_step_callback: Callable[[TensorDict, dict], None] | None = None,
    ) -> TensorDict:
        self.policy.train()
        if self.reset_every_rollout:
            self.obs_td = self._reset_env()

        rollout_steps: list[TensorDict] = []

        for _ in range(num_steps):
            obs_tensor = flatten_dict_observation(self.obs_td)
            step_td = TensorDict(
                {"observation": obs_tensor},
                batch_size=[obs_tensor.shape[0]],
                device=self.device,
            )
            self.policy.forward(step_td)

            next_obs, reward, terminated, truncated, env_info = self.env.step(
                self._to_action_dict(step_td["action"])
            )
            next_obs_td = dict_to_tensordict(next_obs, self.device)
            next_obs_tensor = flatten_dict_observation(next_obs_td)
            done = (terminated | truncated).bool()

            step_td["next"] = TensorDict(
                {
                    "observation": next_obs_tensor,
                    "reward": reward.float(),
                    "done": done,
                    "terminated": terminated.bool(),
                    "truncated": truncated.bool(),
                },
                batch_size=step_td.batch_size,
                device=self.device,
            )
            rollout_steps.append(step_td.clone())

            if on_step_callback is not None:
                on_step_callback(step_td, env_info)

            self.obs_td = next_obs_td

        rollout = torch.stack(rollout_steps, dim=1)
        self._attach_next_values(rollout)
        return rollout

    def _attach_next_values(self, rollout: TensorDict) -> None:
        """Populate `next.value` for GAE bootstrap."""
        next_values = torch.zeros_like(rollout["value"])
        next_values[:, :-1] = rollout["value"][:, 1:]

        last_next_td = TensorDict(
            {"observation": rollout["next", "observation"][:, -1]},
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
