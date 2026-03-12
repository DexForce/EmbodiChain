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

"""Policy base class for RL algorithms.

This module defines an abstract Policy base class that all RL policies must
inherit from. A Policy encapsulates the neural networks and exposes a uniform
interface for RL algorithms (e.g., PPO, SAC) to interact with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch.nn as nn

import torch
from tensordict import TensorDict


class Policy(nn.Module, ABC):
    """Abstract base class that all RL policies must implement.

    A Policy:
    - Encapsulates neural networks that are trained by RL algorithms
    - Handles internal computations (e.g., network output → distribution)
    - Provides a uniform interface for algorithms (PPO, SAC, etc.)
    """

    device: torch.device
    """Device where the policy parameters are located."""

    def __init__(self) -> None:
        super().__init__()

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility layer for tensor-only callers.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            deterministic: If True, return the mean action; otherwise sample

        Returns:
            Tuple of (action, log_prob, value):
            - action: Sampled action tensor of shape (batch_size, action_dim)
            - log_prob: Log probability of the action, shape (batch_size,)
            - value: Value estimate, shape (batch_size,)
        """
        with torch.no_grad():
            td = TensorDict(
                {"obs": obs},
                batch_size=[obs.shape[0]],
                device=obs.device,
            )
            td = self.forward(td, deterministic=deterministic)
        return td["action"], td["sample_log_prob"], td["value"]

    @abstractmethod
    def forward(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        """Write sampled actions and value estimates into the TensorDict."""
        raise NotImplementedError

    @abstractmethod
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Write value estimate for the given observations into the TensorDict.

        Args:
            tensordict: Input TensorDict containing `obs`.

        Returns:
            TensorDict with `value` populated.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate actions and return current policy outputs.

        Args:
            tensordict: TensorDict containing `obs` and `action`.

        Returns:
            A new TensorDict containing `sample_log_prob`, `entropy`, and `value`.
        """
        raise NotImplementedError
