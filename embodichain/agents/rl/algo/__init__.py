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

from typing import Dict, Tuple, Type, Any
import torch

from .base import BaseAlgorithm
from .ppo import PPOCfg, PPO

# name -> (CfgClass, AlgoClass)
_ALGO_REGISTRY: Dict[str, Tuple[Type[Any], Type[Any]]] = {
    "ppo": (PPOCfg, PPO),
}


def get_registered_algo_names() -> list[str]:
    return list(_ALGO_REGISTRY.keys())


def build_algo(name: str, cfg_kwargs: Dict[str, float], policy, device: torch.device):
    key = name.lower()
    if key not in _ALGO_REGISTRY:
        raise ValueError(
            f"Algorithm '{name}' not found. Available: {get_registered_algo_names()}"
        )
    CfgCls, AlgoCls = _ALGO_REGISTRY[key]
    cfg = CfgCls(device=str(device), **cfg_kwargs)
    return AlgoCls(cfg, policy)


__all__ = [
    "BaseAlgorithm",
    "PPOCfg",
    "PPO",
    "get_registered_algo_names",
    "build_algo",
]
