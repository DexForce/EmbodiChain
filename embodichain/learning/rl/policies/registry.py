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

"""RL policy registration and construction for actor-critic and actor-only models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from typing import Type

from gymnasium import spaces
import torch

from .actor_critic import ActorCritic
from .actor_only import ActorOnly
from .policy import Policy
from .proprioceptive_actor_critic import ProprioceptiveActorCritic

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


def resolve_policy_obs_groups(
    policy_block: Mapping[str, object],
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Resolve actor and critic observation groups from policy configuration.

    Args:
        policy_block: Policy configuration containing an optional
            ``obs_groups`` mapping.

    Returns:
        Actor and critic observation group names. Both are ``None`` when the
        existing flatten-all behavior is selected.

    Raises:
        TypeError: If the observation-group configuration is malformed.
        ValueError: If the actor observation set is empty.
    """
    raw_groups = policy_block.get("obs_groups")
    if raw_groups is None:
        return None, None
    if not isinstance(raw_groups, Mapping):
        raise TypeError("policy.obs_groups must be a mapping.")

    def _normalize(name: str, value: object) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(f"policy.obs_groups.{name} must be a list of names.")
        groups = tuple(value)
        if not all(isinstance(group, str) for group in groups):
            raise TypeError(f"policy.obs_groups.{name} must contain only string names.")
        if len(groups) == 0:
            raise ValueError(f"policy.obs_groups.{name} cannot be empty.")
        return groups

    actor_groups = _normalize("actor", raw_groups.get("actor"))
    if actor_groups is None:
        raise ValueError("policy.obs_groups.actor is required when obs_groups is set.")
    critic_groups = _normalize("critic", raw_groups.get("critic"))
    return actor_groups, actor_groups if critic_groups is None else critic_groups


def build_policy(
    policy_block: dict,
    obs_space: spaces.Space | int,
    action_space: spaces.Space | int,
    device: torch.device,
    actor: torch.nn.Module | None = None,
    critic: torch.nn.Module | None = None,
    critic_obs_space: spaces.Space | int | None = None,
) -> Policy:
    """Build a policy from config using spaces for extensibility.

    Built-in MLP policies still resolve flattened `obs_dim` / `action_dim`, while
    custom policies may accept richer `obs_space` / `action_space` inputs.
    """
    name = policy_block["name"].lower()
    if name not in _POLICY_REGISTRY:
        available = ", ".join(get_registered_policy_names())
        raise ValueError(
            f"Policy '{name}' is not registered. Available policies: {available}"
        )
    policy_cls = _POLICY_REGISTRY[name]
    actor_obs_groups, critic_obs_groups = resolve_policy_obs_groups(policy_block)

    if name == "actor_critic":
        if actor is None or critic is None:
            raise ValueError(
                "ActorCritic policy requires external 'actor' and 'critic' modules."
            )
        obs_dim = _resolve_space_dim(obs_space, "obs_space")
        critic_obs_dim = (
            None
            if critic_obs_space is None
            else _resolve_space_dim(critic_obs_space, "critic_obs_space")
        )
        if actor_obs_groups != critic_obs_groups and critic_obs_dim is None:
            raise ValueError(
                "A distinct critic observation group requires critic_obs_space."
            )
        action_dim = _resolve_space_dim(action_space, "action_space")
        return policy_cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            actor=actor,
            critic=critic,
            critic_obs_dim=critic_obs_dim,
            actor_obs_groups=actor_obs_groups,
            critic_obs_groups=critic_obs_groups,
            actor_obs_normalization=bool(
                policy_block.get("actor_obs_normalization", False)
            ),
            critic_obs_normalization=bool(
                policy_block.get("critic_obs_normalization", False)
            ),
            initial_action_std=float(policy_block.get("initial_action_std", 1.0)),
            action_std_range=tuple(policy_block.get("action_std_range", (1e-6, 1e6))),
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
            actor_obs_groups=actor_obs_groups,
        )

    init_params = inspect.signature(policy_cls.__init__).parameters
    raw_cfg = policy_block.get("cfg", {})
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("policy.cfg must be a mapping.")
    build_kwargs: dict[str, object] = {"device": device}
    if "cfg" in init_params:
        build_kwargs["cfg"] = dict(raw_cfg)
    elif raw_cfg:
        unknown = set(raw_cfg) - set(init_params)
        if unknown:
            raise TypeError(f"Unsupported policy.cfg fields: {sorted(unknown)}.")
        build_kwargs.update(raw_cfg)
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
    if "critic_obs_dim" in init_params and critic_obs_space is not None:
        build_kwargs["critic_obs_dim"] = _resolve_space_dim(
            critic_obs_space, "critic_obs_space"
        )
    for key, value in (
        ("actor_obs_groups", actor_obs_groups),
        ("critic_obs_groups", critic_obs_groups),
    ):
        if key in init_params:
            build_kwargs[key] = value
    for key in (
        "actor_obs_normalization",
        "critic_obs_normalization",
        "initial_action_std",
        "action_std_range",
    ):
        if key in init_params and key in policy_block:
            build_kwargs[key] = policy_block[key]
    return policy_cls(**build_kwargs)


# default registrations
register_policy("actor_critic", ActorCritic)
register_policy("actor_only", ActorOnly)
register_policy("proprioceptive_actor_critic", ProprioceptiveActorCritic)

__all__ = [
    "register_policy",
    "get_registered_policy_names",
    "build_policy",
    "get_policy_class",
    "resolve_policy_obs_groups",
]
