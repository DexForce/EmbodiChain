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

from typing import Dict, Type
import torch
from gymnasium import spaces

from .actor_critic import ActorCritic
from .policy import Policy
from .mlp import MLP

# In-module policy registry
_POLICY_REGISTRY: Dict[str, Type[Policy]] = {}


def register_policy(name: str, policy_cls: Type[Policy]) -> None:
    if name in _POLICY_REGISTRY:
        raise ValueError(f"Policy '{name}' is already registered")
    _POLICY_REGISTRY[name] = policy_cls


def get_registered_policy_names() -> list[str]:
    return list(_POLICY_REGISTRY.keys())


def get_policy_class(name: str) -> Type[Policy] | None:
    return _POLICY_REGISTRY.get(name)


def build_policy(
    policy_block: dict,
    obs_space: spaces.Space,
    action_space: spaces.Space,
    device: torch.device,
    actor: torch.nn.Module | None = None,
    critic: torch.nn.Module | None = None,
) -> Policy:
    """Build policy strictly from json-like block: { name: ..., cfg: {...} }"""
    name = policy_block["name"].lower()
    if name not in _POLICY_REGISTRY:
        available = ", ".join(get_registered_policy_names())
        raise ValueError(
            f"Policy '{name}' is not registered. Available policies: {available}"
        )
    policy_cls = _POLICY_REGISTRY[name]
    if name == "actor_critic":
        if actor is None or critic is None:
            raise ValueError(
                "ActorCritic policy requires external 'actor' and 'critic' modules."
            )
        return policy_cls(obs_space, action_space, device, actor=actor, critic=critic)
    else:
        return policy_cls(obs_space, action_space, device)


def build_mlp_from_cfg(module_cfg: Dict, in_dim: int, out_dim: int) -> MLP:
    """Construct an MLP module from a minimal json-like config.

    Expected schema:
      module_cfg = {
        "type": "mlp",
        "hidden_sizes": [256, 256],
        "activation": "relu",
      }
    """
    if module_cfg.get("type", "").lower() != "mlp":
        raise ValueError("Only 'mlp' type is supported for actor/critic in this setup.")

    hidden_sizes = module_cfg["network_cfg"]["hidden_sizes"]
    activation = module_cfg["network_cfg"]["activation"]
    return MLP(in_dim, out_dim, hidden_sizes, activation)


# default registrations
register_policy("actor_critic", ActorCritic)

__all__ = [
    "ActorCritic",
    "register_policy",
    "get_registered_policy_names",
    "build_policy",
    "build_mlp_from_cfg",
    "get_policy_class",
    "Policy",
    "MLP",
]
