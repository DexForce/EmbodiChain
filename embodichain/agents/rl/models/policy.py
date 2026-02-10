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

"""Policy base class for RL algorithms.

This module defines an abstract Policy base class that all RL policies must
inherit from. A Policy encapsulates the neural networks and exposes a uniform
interface for RL algorithms (e.g., PPO, SAC) to interact with.

All data I/O now uses TensorDict for structured, extensible data flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch.nn as nn
from tensordict import TensorDict

import torch


class Policy(nn.Module, ABC):
    """Abstract base class that all RL policies must implement.

    A Policy:
    - Encapsulates neural networks that are trained by RL algorithms
    - Handles internal computations (e.g., network output → distribution)
    - Provides a uniform interface for algorithms (PPO, SAC, etc.)
    - Uses TensorDict for all inputs and outputs (no tensor fallback)
    """

    device: torch.device
    """Device where the policy parameters are located."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass that adds action to the input tensordict (in-place).

        This is the main inference method following TorchRL conventions.

        Args:
            tensordict: Input TensorDict containing at minimum:
                - "observation": Observation tensor or nested TensorDict

        Returns:
            The same TensorDict (modified in-place) with added fields:
                - "action": Sampled action tensor
                - "sample_log_prob": Log probability of the sampled action
                - "value": Value estimate (optional, for actor-critic)
                - "loc": Distribution mean (optional, for continuous actions)
                - "scale": Distribution std (optional, for continuous actions)
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, tensordict: TensorDict) -> TensorDict:
        """Get value estimate for given observations.

        Args:
            tensordict: Input TensorDict containing:
                - "observation": Observation data

        Returns:
            TensorDict with added field:
                - "value": Value estimate tensor
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        """Evaluate actions and compute log probabilities, entropy, and values.

        Used during policy updates to recompute action probabilities.

        Args:
            tensordict: Input TensorDict containing:
                - "observation": Observation data
                - "action": Actions to evaluate

        Returns:
            TensorDict with added fields:
                - "sample_log_prob": Log probability of actions
                - "entropy": Entropy of the action distribution
                - "value": Value estimate
        """
        raise NotImplementedError
