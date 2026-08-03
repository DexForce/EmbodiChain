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

"""Planner and scenario registries used by the generic benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PlannerSpecCfg
    from .planners.base import PlannerAdapter, PlannerContext
    from .scenarios.base import ScenarioProvider

__all__ = [
    "create_planner_adapter",
    "create_scenario_provider",
    "planner_adapter_names",
    "register_planner_adapter",
    "register_scenario_provider",
    "scenario_provider_names",
]

_PLANNER_ADAPTERS: dict[str, type["PlannerAdapter"]] = {}
_SCENARIO_PROVIDERS: dict[str, type["ScenarioProvider"]] = {}


def register_planner_adapter(name: str, adapter_cls: type["PlannerAdapter"]) -> None:
    """Register one adapter class under a stable configuration name."""
    if not name:
        raise ValueError("Planner adapter name must not be empty.")
    previous = _PLANNER_ADAPTERS.get(name)
    if previous is not None and previous is not adapter_cls:
        raise ValueError(f"Planner adapter {name!r} is already registered.")
    _PLANNER_ADAPTERS[name] = adapter_cls


def planner_adapter_names() -> tuple[str, ...]:
    """Return registered adapter names in deterministic order."""
    return tuple(sorted(_PLANNER_ADAPTERS))


def create_planner_adapter(
    spec: "PlannerSpecCfg", context: "PlannerContext"
) -> "PlannerAdapter":
    """Construct the adapter selected by a planner specification."""
    try:
        adapter_cls = _PLANNER_ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise ValueError(
            f"Unknown planner adapter {spec.adapter!r}; "
            f"registered adapters: {planner_adapter_names()}."
        ) from exc
    return adapter_cls(spec=spec, context=context)


def register_scenario_provider(
    name: str, provider_cls: type["ScenarioProvider"]
) -> None:
    """Register one scenario provider class under a stable configuration name."""
    if not name:
        raise ValueError("Scenario provider name must not be empty.")
    previous = _SCENARIO_PROVIDERS.get(name)
    if previous is not None and previous is not provider_cls:
        raise ValueError(f"Scenario provider {name!r} is already registered.")
    _SCENARIO_PROVIDERS[name] = provider_cls


def scenario_provider_names() -> tuple[str, ...]:
    """Return registered scenario provider names in deterministic order."""
    return tuple(sorted(_SCENARIO_PROVIDERS))


def create_scenario_provider(name: str) -> "ScenarioProvider":
    """Construct the scenario provider selected by a track specification."""
    try:
        provider_cls = _SCENARIO_PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scenario provider {name!r}; "
            f"registered providers: {scenario_provider_names()}."
        ) from exc
    return provider_cls()
