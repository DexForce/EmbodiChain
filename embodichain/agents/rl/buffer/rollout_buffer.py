# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, Iterator

import torch


class RolloutBuffer:
    """On-device rollout buffer for on-policy algorithms.

    Stores (obs, actions, rewards, dones, values, logprobs) over time.
    After finalize(), exposes advantages/returns and minibatch iteration.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        T, N = num_steps, num_envs
        self.obs = torch.zeros(T, N, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(T, N, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(T, N, dtype=torch.float32, device=device)
        self.dones = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.values = torch.zeros(T, N, dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(T, N, dtype=torch.float32, device=device)

        self.step = 0
        # Container for algorithm-specific extra fields (e.g., advantages, returns)
        self._extras: dict[str, torch.Tensor] = {}

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        logprob: torch.Tensor,
    ) -> None:
        t = self.step
        self.obs[t].copy_(obs)
        self.actions[t].copy_(action)
        self.rewards[t].copy_(reward)
        self.dones[t].copy_(done)
        self.values[t].copy_(value)
        self.logprobs[t].copy_(logprob)
        self.step += 1

    def set_extras(self, extras: dict[str, torch.Tensor]) -> None:
        """Attach algorithm-specific tensors (shape [T, N, ...]) for batching.

        Examples:
            {"advantages": adv, "returns": ret}
        """
        self._extras = extras or {}

    def iterate_minibatches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        T, N = self.num_steps, self.num_envs
        total = T * N
        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            t_idx = idx // N
            n_idx = idx % N
            batch = {
                "obs": self.obs[t_idx, n_idx],
                "actions": self.actions[t_idx, n_idx],
                "rewards": self.rewards[t_idx, n_idx],
                "dones": self.dones[t_idx, n_idx],
                "values": self.values[t_idx, n_idx],
                "logprobs": self.logprobs[t_idx, n_idx],
            }
            # Slice extras if present and shape aligned to [T, N, ...]
            for name, tensor in self._extras.items():
                try:
                    batch[name] = tensor[t_idx, n_idx]
                except Exception:
                    # Skip misaligned extras silently
                    continue
            yield batch
