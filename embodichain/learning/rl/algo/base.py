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

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Dict, Generic, TypeVar

import torch

from embodichain.learning.rl.utils import (
    AlgorithmCfg,
    bind_scheduler_horizon,
    build_lr_scheduler,
    build_optimizer,
    coerce_lr_scheduler_cfg,
    coerce_optimizer_cfg,
    scheduler_needs_horizon,
)

__all__ = ["BaseAlgorithm", "RolloutKind"]

RolloutT = TypeVar("RolloutT")


class RolloutKind(str, Enum):
    """Rollout semantics required by an algorithm."""

    STANDARD = "standard"
    DIFFERENTIABLE = "differentiable"


class BaseAlgorithm(ABC, Generic[RolloutT]):
    """Base class for RL algorithms."""

    device: torch.device
    rollout_kind = RolloutKind.STANDARD
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None

    @abstractmethod
    def update(self, rollout: RolloutT) -> Dict[str, float]:
        """Update policy using collected data and return training losses."""
        raise NotImplementedError

    def _setup_optimization(
        self,
        cfg: AlgorithmCfg,
        parameters: Iterable[torch.nn.Parameter],
    ) -> None:
        cfg.optimizer = coerce_optimizer_cfg(cfg.optimizer)
        cfg.lr_scheduler = coerce_lr_scheduler_cfg(cfg.lr_scheduler)
        self._lr_scheduler_cfg = cfg.lr_scheduler
        self.optimizer = build_optimizer(parameters, cfg.optimizer)
        self.lr_scheduler = None
        if self._lr_scheduler_cfg.name is not None and not scheduler_needs_horizon(
            self._lr_scheduler_cfg
        ):
            self.lr_scheduler = build_lr_scheduler(
                self.optimizer, self._lr_scheduler_cfg
            )

    def bind_schedule(self, *, total_updates: int) -> None:
        """Bind horizon-dependent LR schedules from the training budget."""
        if total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if not scheduler_needs_horizon(self._lr_scheduler_cfg):
            return
        self._lr_scheduler_cfg = bind_scheduler_horizon(
            self._lr_scheduler_cfg, total_updates
        )
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self._lr_scheduler_cfg)

    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _step_scheduler(self) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
