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

import inspect
from typing import Any, Dict, Optional, Type

from gymnasium import spaces
import torch

from .actor_critic import ActorCritic
from .actor_only import ActorOnly
from .policy import Policy
from .mlp import MLP
from .vla_policy import VLAPolicy

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


def _resolve_space_dim(space_or_dim: spaces.Space | int, name: str) -> int:
    """Resolve a flattened feature dimension from an integer or simple Box space."""
    if isinstance(space_or_dim, int):
        return space_or_dim
    if isinstance(space_or_dim, spaces.Box) and len(space_or_dim.shape) > 0:
        return int(space_or_dim.shape[-1])
    raise TypeError(
        f"{name} must be an int or a flat Box space for MLP-based policies, got {type(space_or_dim)!r}."
    )


def build_policy(
    policy_block: dict,
    obs_space: spaces.Space | int,
    action_space: spaces.Space | int,
    device: torch.device,
    actor: torch.nn.Module | None = None,
    critic: torch.nn.Module | None = None,
    env: Optional[Any] = None,
) -> Policy:
    """Build a policy from config using spaces for extensibility.

    Built-in MLP policies still resolve flattened `obs_dim` / `action_dim`, while
    custom policies may accept richer `obs_space` / `action_space` inputs.
    For vla_policy, pass env to enable set_env and _load_vla initialization.
    """
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
        obs_dim = _resolve_space_dim(obs_space, "obs_space")
        action_dim = _resolve_space_dim(action_space, "action_space")
        return policy_cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            actor=actor,
            critic=critic,
        )
    elif name == "actor_only":
        if actor is None:
            raise ValueError("ActorOnly policy requires external 'actor' module.")
        obs_dim = _resolve_space_dim(obs_space, "obs_space")
        action_dim = _resolve_space_dim(action_space, "action_space")
        return policy_cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            actor=actor,
        )

    init_params = inspect.signature(policy_cls.__init__).parameters
    build_kwargs: dict[str, object] = {"device": device}
    if "obs_space" in init_params:
        build_kwargs["obs_space"] = obs_space
    elif "obs_dim" in init_params:
        build_kwargs["obs_dim"] = _resolve_space_dim(obs_space, "obs_space")

    if "action_space" in init_params:
        build_kwargs["action_space"] = action_space
    elif "action_dim" in init_params:
        build_kwargs["action_dim"] = _resolve_space_dim(action_space, "action_space")

    if "actor" in init_params and actor is not None:
        build_kwargs["actor"] = actor
    if "critic" in init_params and critic is not None:
        build_kwargs["critic"] = critic
    if "policy_cfg" in init_params:
        build_kwargs["policy_cfg"] = policy_block
    policy = policy_cls(**build_kwargs)
    if name == "vla_policy" and env is not None:
        policy.set_env(env)
        policy._load_vla()
    return policy


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
register_policy("actor_only", ActorOnly)
register_policy("vla_policy", VLAPolicy)

__all__ = [
    "ActorCritic",
    "ActorOnly",
    "VLAPolicy",
    "register_policy",
    "get_registered_policy_names",
    "build_policy",
    "build_mlp_from_cfg",
    "get_policy_class",
    "Policy",
    "MLP",
]
