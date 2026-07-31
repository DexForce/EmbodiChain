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

"""Algorithm registry and construction helpers (``BaseAlgorithm``, ``PPO``, ``GRPO``, ``compute_gae``, ``build_algo``)."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .apg import APG, APGCfg, segmented_discounted_return
from .base import BaseAlgorithm
from .common import compute_gae
from .grpo import GRPO, GRPOCfg
from .ppo import PPO, PPOCfg

_ALGO_REGISTRY: Dict[str, Tuple[Type[Any], Type[Any]]] = {
    "ppo": (PPOCfg, PPO),
    "grpo": (GRPOCfg, GRPO),
}


def get_registered_algo_names() -> list[str]:
    return list(_ALGO_REGISTRY.keys())


def build_algo(
    name: str,
    cfg_kwargs: Dict[str, Any],
    policy,
    device: torch.device,
    *,
    distributed: bool = False,
):
    key = name.lower()
    if key == "apg":
        raise ValueError(
            "APG uses differentiable rollouts and is not supported by the "
            "standard build_algo()/train.py path. Construct APG with "
            "DifferentiableTrainer directly, or use the experimental Newton "
            "training entry point."
        )
    if key not in _ALGO_REGISTRY:
        raise ValueError(
            f"Algorithm '{name}' not found. Available: {get_registered_algo_names()}"
        )
    CfgCls, AlgoCls = _ALGO_REGISTRY[key]
    cfg = CfgCls(device=str(device), **cfg_kwargs)
    if distributed:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "Distributed training requested (distributed=True), but "
                "torch.distributed is not initialized. Please call "
                "torch.distributed.init_process_group() before building the "
                "algorithm, or set distributed=False."
            )
        policy = DDP(
            policy, device_ids=[device.index] if device.index is not None else None
        )
    return AlgoCls(cfg, policy)


__all__ = [
    "BaseAlgorithm",
    "APGCfg",
    "APG",
    "segmented_discounted_return",
    "PPOCfg",
    "PPO",
    "GRPOCfg",
    "GRPO",
    "compute_gae",
    "get_registered_algo_names",
    "build_algo",
]
