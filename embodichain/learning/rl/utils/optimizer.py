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

"""Shared optimizer and LR-scheduler construction for RL algorithms."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from embodichain.learning.rl.utils.config import LRSchedulerCfg, OptimizerCfg

__all__ = [
    "bind_scheduler_horizon",
    "build_lr_scheduler",
    "build_optimizer",
    "coerce_lr_scheduler_cfg",
    "coerce_optimizer_cfg",
    "get_registered_lr_scheduler_names",
    "get_registered_optimizer_names",
    "scheduler_needs_horizon",
]

_OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

_HORIZON_KEYS: dict[str, str] = {
    "linear": "total_iters",
    "cosine": "T_max",
}


def _build_linear(
    optimizer: torch.optim.Optimizer, kwargs: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    if "total_iters" not in kwargs:
        raise ValueError(
            "linear scheduler requires kwargs['total_iters'] "
            "(or bind it via BaseAlgorithm.bind_schedule)."
        )
    kwargs.setdefault("start_factor", 1.0)
    kwargs.setdefault("end_factor", 0.0)
    return torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)


def _build_cosine(
    optimizer: torch.optim.Optimizer, kwargs: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    if "T_max" not in kwargs:
        raise ValueError(
            "cosine scheduler requires kwargs['T_max'] "
            "(or bind it via BaseAlgorithm.bind_schedule)."
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)


def _build_step(
    optimizer: torch.optim.Optimizer, kwargs: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    if "step_size" not in kwargs:
        raise ValueError("step scheduler requires kwargs['step_size'].")
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)


def _build_exponential(
    optimizer: torch.optim.Optimizer, kwargs: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    if "gamma" not in kwargs:
        raise ValueError("exponential scheduler requires kwargs['gamma'].")
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)


_SCHEDULER_BUILDERS: dict[
    str,
    Callable[
        [torch.optim.Optimizer, dict[str, Any]], torch.optim.lr_scheduler.LRScheduler
    ],
] = {
    "linear": _build_linear,
    "cosine": _build_cosine,
    "step": _build_step,
    "exponential": _build_exponential,
}


def get_registered_optimizer_names() -> list[str]:
    return sorted(_OPTIMIZER_REGISTRY)


def get_registered_lr_scheduler_names() -> list[str]:
    return sorted(_SCHEDULER_BUILDERS)


def coerce_optimizer_cfg(
    value: OptimizerCfg | Mapping[str, Any] | None,
) -> OptimizerCfg:
    if value is None:
        return OptimizerCfg()
    if isinstance(value, OptimizerCfg):
        return value
    if isinstance(value, Mapping):
        return OptimizerCfg(**dict(value))
    raise TypeError(f"Expected OptimizerCfg or mapping, got {type(value)!r}.")


def coerce_lr_scheduler_cfg(
    value: LRSchedulerCfg | Mapping[str, Any] | None,
) -> LRSchedulerCfg:
    if value is None:
        return LRSchedulerCfg()
    if isinstance(value, LRSchedulerCfg):
        return value
    if isinstance(value, Mapping):
        return LRSchedulerCfg(**dict(value))
    raise TypeError(f"Expected LRSchedulerCfg or mapping, got {type(value)!r}.")


def scheduler_needs_horizon(cfg: LRSchedulerCfg | Mapping[str, Any] | None) -> bool:
    cfg = coerce_lr_scheduler_cfg(cfg)
    if cfg.name is None:
        return False
    horizon_key = _HORIZON_KEYS.get(cfg.name.lower())
    return horizon_key is not None and horizon_key not in dict(cfg.kwargs)


def bind_scheduler_horizon(
    cfg: LRSchedulerCfg | Mapping[str, Any] | None,
    total_updates: int,
) -> LRSchedulerCfg:
    """Fill ``total_iters`` / ``T_max`` from the training update budget."""
    cfg = coerce_lr_scheduler_cfg(cfg)
    if cfg.name is None:
        return cfg
    horizon_key = _HORIZON_KEYS.get(cfg.name.lower())
    if horizon_key is None:
        return cfg
    kwargs = dict(cfg.kwargs)
    kwargs.setdefault(horizon_key, total_updates)
    return LRSchedulerCfg(name=cfg.name, kwargs=kwargs)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: OptimizerCfg | Mapping[str, Any] | None = None,
) -> torch.optim.Optimizer:
    opt_cfg = coerce_optimizer_cfg(cfg)
    key = opt_cfg.name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        available = ", ".join(get_registered_optimizer_names())
        raise ValueError(
            f"Unsupported optimizer '{opt_cfg.name}'. Available: {available}"
        )
    opt_kwargs = dict(opt_cfg.kwargs)
    if "lr" in opt_kwargs:
        raise ValueError(
            "Pass learning rate via OptimizerCfg.learning_rate, not kwargs['lr']."
        )
    return _OPTIMIZER_REGISTRY[key](parameters, lr=opt_cfg.learning_rate, **opt_kwargs)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: LRSchedulerCfg | Mapping[str, Any] | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a scheduler, or ``None`` when ``name`` is unset."""
    cfg = coerce_lr_scheduler_cfg(cfg)
    if cfg.name is None:
        return None
    key = cfg.name.lower()
    builder = _SCHEDULER_BUILDERS.get(key)
    if builder is None:
        available = ", ".join(get_registered_lr_scheduler_names())
        raise ValueError(
            f"Unsupported lr_scheduler '{cfg.name}'. Available: {available}"
        )
    return builder(optimizer, dict(cfg.kwargs))
