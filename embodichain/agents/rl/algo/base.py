# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import torch


class BaseAlgorithm:
    """Base class for RL algorithms.

    Algorithms must implement buffer initialization, rollout collection, and
    policy update. Trainer depends only on this interface to remain
    algorithm-agnostic.
    """

    device: torch.device

    def initialize_buffer(
        self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int
    ) -> None:
        """Initialize internal buffer(s) required by the algorithm."""
        raise NotImplementedError

    def collect_rollout(
        self,
        env,
        policy,
        obs: torch.Tensor,
        num_steps: int,
        on_step_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Collect trajectories and return logging info (e.g., reward components)."""
        raise NotImplementedError

    def update(self) -> Dict[str, float]:
        """Update policy using collected data and return training losses."""
        raise NotImplementedError
