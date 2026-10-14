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
from copy import deepcopy
from typing import Any, Dict, Generic, TypeVar

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
        self._configured_scheduler = deepcopy(cfg.lr_scheduler)
        self._configured_learning_rate = cfg.optimizer.learning_rate
        self._checkpoint_loaded = False
        self._restart_schedule = False
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

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer and learning-rate state for a trainer checkpoint.

        Returns:
            Optimizer state and the saved scheduler configuration/progress.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "lr_scheduler_cfg": {
                "name": self._lr_scheduler_cfg.name,
                "kwargs": dict(self._lr_scheduler_cfg.kwargs),
            },
        }

    def validate_state_dict(
        self, state: dict[str, Any], *, resume_scheduler: str = "restore"
    ) -> None:
        """Check optimization structure before changing the live algorithm.

        Args:
            state: Optimization fields returned by :meth:`state_dict`.
            resume_scheduler: Restore the saved curve or restart configuration.

        Raises:
            ValueError: If the mode, optimizer layout, or scheduler fields differ.
        """
        if resume_scheduler not in {"restore", "restart"}:
            raise ValueError("resume_scheduler must be 'restore' or 'restart'.")
        groups = state["optimizer"]["param_groups"]
        if len(groups) != len(self.optimizer.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(groups, self.optimizer.param_groups)
        ):
            raise ValueError("Incompatible checkpoint optimizer parameter groups.")
        for saved, current in zip(groups, self.optimizer.param_groups):
            for identifier, parameter in zip(saved["params"], current["params"]):
                for value in state["optimizer"]["state"].get(identifier, {}).values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape != parameter.shape
                    ):
                        raise ValueError(
                            "Incompatible checkpoint optimizer state shape."
                        )
        if state.get("lr_scheduler") is not None:
            if state.get("lr_scheduler_cfg") is None:
                raise ValueError("Checkpoint scheduler configuration is missing.")
            cfg = coerce_lr_scheduler_cfg(state["lr_scheduler_cfg"])
            probe_optimizer = torch.optim.SGD(
                [torch.nn.Parameter(torch.zeros(()))], lr=1.0
            )
            probe_scheduler = build_lr_scheduler(probe_optimizer, cfg)
            if probe_scheduler is None:
                raise ValueError("Checkpoint scheduler state has no scheduler name.")
            if "last_epoch" not in state["lr_scheduler"]:
                raise ValueError("Checkpoint scheduler progress is missing.")
            probe_scheduler.load_state_dict(state["lr_scheduler"])

    def load_state_dict(
        self, state: dict[str, Any], *, resume_scheduler: str = "restore"
    ) -> None:
        """Restore optimization state or restart its configured LR curve.

        Args:
            state: Optimization fields returned by :meth:`state_dict`.
            resume_scheduler: Restore the saved LR curve, or restart the
                configured curve while retaining optimizer momentum.
        """
        self.validate_state_dict(state, resume_scheduler=resume_scheduler)
        self.lr_scheduler = None
        if resume_scheduler == "restore":
            if state.get("lr_scheduler_cfg") is not None:
                self._lr_scheduler_cfg = coerce_lr_scheduler_cfg(
                    state["lr_scheduler_cfg"]
                )
            if state.get("lr_scheduler") is not None:
                self.lr_scheduler = build_lr_scheduler(
                    self.optimizer, self._lr_scheduler_cfg
                )
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.optimizer.load_state_dict(state["optimizer"])
        else:
            self.optimizer.load_state_dict(state["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = self._configured_learning_rate
                group["initial_lr"] = self._configured_learning_rate
            self._lr_scheduler_cfg = deepcopy(self._configured_scheduler)
            if not scheduler_needs_horizon(self._lr_scheduler_cfg):
                self.lr_scheduler = build_lr_scheduler(
                    self.optimizer, self._lr_scheduler_cfg
                )
        self._checkpoint_loaded = True
        self._restart_schedule = resume_scheduler == "restart"

    def _prepare_schedule(self, total_updates: int, remaining_updates: int) -> None:
        if remaining_updates <= 0:
            return
        self.bind_schedule(
            total_updates=remaining_updates if self._restart_schedule else total_updates
        )
        self._restart_schedule = False
        cfg = self._lr_scheduler_cfg
        if (
            self._checkpoint_loaded
            and self.lr_scheduler is not None
            and cfg.name is not None
            and cfg.name.lower() == "linear"
            and float(cfg.kwargs.get("end_factor", 0.0)) == 0.0
            and self.lr_scheduler.last_epoch + remaining_updates
            > int(cfg.kwargs["total_iters"])
        ):
            raise ValueError(
                "Restored linear scheduler ends at zero before the requested budget; "
                "use --resume-scheduler restart or shorten the training budget."
            )

    def _sync_gradients(self) -> None:
        """Average gradients when action evaluation bypasses DDP forward."""
        if not isinstance(self.policy, torch.nn.parallel.DistributedDataParallel):
            return
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        present = torch.tensor(
            [p.grad is not None for p in parameters],
            device=self.device,
            dtype=torch.int,
        )
        torch.distributed.all_reduce(present)
        gradients = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                for p in parameters
            ]
        )
        torch.distributed.all_reduce(gradients)
        gradients.div_(torch.distributed.get_world_size())
        offset = 0
        for p, used in zip(parameters, present.tolist()):
            if used:
                p.grad = gradients[offset : offset + p.numel()].view_as(p).clone()
            offset += p.numel()

    def _step_scheduler(self) -> None:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
