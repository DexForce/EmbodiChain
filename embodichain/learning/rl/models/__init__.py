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

"""RL policy models and registration, loaded on demand."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActorCritic",
    "ActorOnly",
    "register_policy",
    "get_registered_policy_names",
    "build_policy",
    "get_policy_class",
    "resolve_policy_obs_groups",
    "Policy",
]

_EXPORT_MODULES = {
    "ActorCritic": "actor_critic",
    "ActorOnly": "actor_only",
    "Policy": "policy",
    "register_policy": "registry",
    "get_registered_policy_names": "registry",
    "build_policy": "registry",
    "get_policy_class": "registry",
    "resolve_policy_obs_groups": "registry",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORT_MODULES[name]}"), name)
    globals()[name] = value
    return value
