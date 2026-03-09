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

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict

from .policy import Policy


class ActorOnly(Policy):
    """Actor-only Gaussian policy with TensorDict I/O."""

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.device = device

        self.actor = actor
        self.actor.to(self.device)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def _extract_obs_tensor(self, tensordict: TensorDict) -> torch.Tensor:
        """Extract a flattened observation tensor from nested TensorDict leaves."""
        obs = tensordict["observation"]
        obs_list: list[torch.Tensor] = []

        def _collect(item) -> None:
            if hasattr(item, "keys"):
                for key in sorted(item.keys()):
                    _collect(item[key])
            else:
                obs_list.append(item.flatten(start_dim=1))

        _collect(obs)

        if not obs_list:
            raise ValueError("No tensors found in observation")
        if len(obs_list) == 1:
            return obs_list[0]
        return torch.cat(obs_list, dim=-1)

    @torch.no_grad()
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs_tensor = self._extract_obs_tensor(tensordict)
        mean = self.actor(obs_tensor)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = torch.zeros(
            (obs_tensor.shape[0], 1), device=self.device, dtype=obs_tensor.dtype
        )

        tensordict["action"] = action
        tensordict["sample_log_prob"] = log_prob
        tensordict["value"] = value
        tensordict["loc"] = mean
        tensordict["scale"] = std
        return tensordict

    @torch.no_grad()
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        obs_tensor = self._extract_obs_tensor(tensordict)
        tensordict["value"] = torch.zeros(
            (obs_tensor.shape[0], 1), device=self.device, dtype=obs_tensor.dtype
        )
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        obs_tensor = self._extract_obs_tensor(tensordict)
        actions = tensordict["action"]
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        mean = self.actor(obs_tensor)
        std = log_std.exp().expand(mean.shape[0], -1)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = torch.zeros(
            (obs_tensor.shape[0], 1), device=self.device, dtype=obs_tensor.dtype
        )

        tensordict["sample_log_prob"] = log_prob
        tensordict["entropy"] = entropy
        tensordict["value"] = value
        return tensordict
