# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

"""Policy base class for RL algorithms.

This module defines an abstract Policy base class that all RL policies must
inherit from. A Policy encapsulates the neural networks and exposes a uniform
interface for RL algorithms (e.g., PPO, SAC) to interact with.
"""

from __future__ import annotations

from typing import Tuple
from abc import ABC, abstractmethod
import torch.nn as nn

import torch


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

    @abstractmethod
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            deterministic: If True, return the mean action; otherwise sample

        Returns:
            Tuple of (action, log_prob, value):
            - action: Sampled action tensor of shape (batch_size, action_dim)
            - log_prob: Log probability of the action, shape (batch_size,)
            - value: Value estimate, shape (batch_size,)
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for given observations.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)

        Returns:
            Value estimate tensor of shape (batch_size,)
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and compute log probabilities, entropy, and values.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            actions: Action tensor of shape (batch_size, action_dim)

        Returns:
            Tuple of (log_prob, entropy, value):
            - log_prob: Log probability of actions, shape (batch_size,)
            - entropy: Entropy of the action distribution, shape (batch_size,)
            - value: Value estimate, shape (batch_size,)
        """
        raise NotImplementedError
