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

from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tensordict import TensorDict
from .mlp import MLP
from .policy import Policy


class ActorCritic(Policy):
    """Actor-Critic with learnable log_std for Gaussian policy.

    Uses TensorDict for all data I/O following TorchRL conventions.
    This implementation:
    - Encapsulates MLP networks (actor + critic) trained by RL algorithms
    - Handles internal computation: MLP output → mean + learnable log_std → Normal distribution
    - Provides a uniform TensorDict-based interface for RL algorithms (PPO, SAC, etc.)

    This allows seamless swapping with other policy implementations (e.g., VLAPolicy)
    without modifying RL algorithm code.

    Implements:
      - forward(tensordict) -> tensordict (adds action, sample_log_prob, value)
      - get_value(tensordict) -> tensordict (adds value)
      - evaluate_actions(tensordict) -> tensordict (adds sample_log_prob, entropy, value)
    """

    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        actor: nn.Module,
        critic: nn.Module,
    ):
        super().__init__()
        # Observation handling done via TensorDict - no need for obs_space
        self.action_dim = action_dim
        self.device = device

        # Require external injection of actor and critic
        self.actor = actor
        self.critic = critic
        self.actor.to(self.device)
        self.critic.to(self.device)

        # learnable log_std per action dim
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        self.log_std_min = -5.0
        self.log_std_max = 2.0

    def _extract_obs_tensor(self, tensordict: TensorDict) -> torch.Tensor:
        """Extract observation as flat tensor from TensorDict.

        For nested TensorDict observations, flattens all leaf tensors.
        For plain tensor observations, returns as is.

        Args:
            tensordict: Input TensorDict with "observation" key

        Returns:
            Flattened observation tensor
        """
        obs = tensordict["observation"]

        # Handle nested TensorDict by collecting all leaf tensors
        obs_list = []

        def _collect(item):
            # Duck typing: if it has keys(), treat as TensorDict
            if hasattr(item, "keys"):
                for key in sorted(item.keys()):
                    _collect(item[key])
            else:
                # Leaf tensor
                obs_list.append(item.flatten(start_dim=1))

        _collect(obs)

        if len(obs_list) == 0:
            raise ValueError("No tensors found in observation")
        elif len(obs_list) == 1:
            return obs_list[0]
        else:
            return torch.cat(obs_list, dim=-1)

    @torch.no_grad()
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Forward pass: sample action and compute value (in-place modification).

        Args:
            tensordict: Must contain "observation" key
            deterministic: If True, use mean instead of sampling

        Returns:
            Same tensordict with added keys:
                - "action": Sampled or deterministic action
                - "sample_log_prob": Log probability of action
                - "value": Value estimate
                - "loc": Distribution mean
                - "scale": Distribution std
        """
        obs_tensor = self._extract_obs_tensor(tensordict)

        # Actor forward
        mean = self.actor(obs_tensor)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)

        dist = Normal(mean, std)

        # Sample action or use mean
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Critic forward - keep shape [N, 1] for consistency with reward/done
        value = self.critic(obs_tensor)

        # Add to tensordict (in-place)
        tensordict["action"] = action
        tensordict["sample_log_prob"] = log_prob
        tensordict["value"] = value
        tensordict["loc"] = mean
        tensordict["scale"] = std

        return tensordict

    @torch.no_grad()
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Get value estimate for observations (in-place modification).

        Args:
            tensordict: Must contain "observation" key

        Returns:
            Same tensordict with added key:
                - "value": Value estimate
        """
        obs_tensor = self._extract_obs_tensor(tensordict)
        value = self.critic(obs_tensor)  # Keep shape [N, 1]
        tensordict["value"] = value
        return tensordict

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate actions for policy gradient computation (in-place modification).

        Args:
            tensordict: Must contain "observation" and "action" keys

        Returns:
            Same tensordict with added keys:
                - "sample_log_prob": Log probability of actions
                - "entropy": Entropy of action distribution
                - "value": Value estimate
        """
        obs_tensor = self._extract_obs_tensor(tensordict)
        actions = tensordict["action"]

        # Actor forward
        mean = self.actor(obs_tensor)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand(mean.shape[0], -1)
        dist = Normal(mean, std)

        # Evaluate given actions - keep shape [N, 1] for consistency
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # Critic forward - keep shape [N, 1]
        value = self.critic(obs_tensor)

        # Add to tensordict (in-place)
        tensordict["sample_log_prob"] = log_prob
        tensordict["entropy"] = entropy
        tensordict["value"] = value

        return tensordict
