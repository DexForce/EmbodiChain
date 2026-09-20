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

"""Bind GenSim inside placement to the same inspected part as E6."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .articulation_binding import (
    PrismaticBinding,
    SLIDE_CALL,
    WITHDRAW_CALL,
    graph_bindings,
    discover_prismatic_parts,
    inspect_prismatic_part,
)
from .drawer_geometry import drawer_region

__all__: list[str] = []


def inside_parts(affordance: str) -> tuple[str, str, str | None]:
    parts = affordance.split("__")
    if len(parts) not in (3, 4) or parts[0] != "inside" or not all(parts):
        raise ValueError(f"Unsupported generated inside affordance {affordance!r}.")
    if len(parts) == 4 and not parts[3].startswith("part_"):
        raise ValueError("Drawer inside target requires a stable part ID.")
    return parts[1], parts[2], parts[3] if len(parts) == 4 else None


def prepare_drawer_graph(graph: dict[str, Any], scene: Any) -> dict[str, Any]:
    """Canonicalize only composite drawers; ambiguous parts never default to first."""
    configs = {str(c["uid"]): c for c in getattr(scene, "articulations", ())}
    if not configs:
        return graph
    places = [
        n
        for n in graph["nodes"]
        if n["call"].get("kind") == "place" and "inside" in n["call"]
    ]
    if not any(inside_parts(n["call"]["inside"])[0] in configs for n in places):
        return graph
    result = deepcopy(graph)
    bindings = graph_bindings(graph, scene)
    canonical = {}
    for binding in bindings.values():
        if not any(
            inside_parts(n["call"]["inside"])[0] == binding.object_id for n in places
        ):
            continue
        part = next(
            p
            for p in discover_prismatic_parts(configs[binding.object_id])
            if p.joint == binding.joint
        )
        canonical[(binding.object_id, binding.part_id)] = inspect_prismatic_part(
            configs[binding.object_id], part.part_id
        )
    for node in result["nodes"]:
        call = node["call"]
        if call.get("call_id") in {SLIDE_CALL, WITHDRAW_CALL}:
            args = call["arguments"]
            binding = canonical.get((args["object"], args.get("part", "")))
            if binding is not None:
                args["part"] = binding.part_id
        elif call.get("kind") == "place" and "inside" in call:
            container, obj, part = inside_parts(call["inside"])
            if container not in configs:
                continue
            matches = [
                b
                for b in dict.fromkeys(canonical.values())
                if b.object_id == container and (part is None or part == b.part_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Drawer placement requires one explicit E6 part; use its step_result or target part binding."
                )
            binding = matches[0]
            call["inside"] = f"inside__{container}__{obj}__{binding.part_id}"
    return result


@dataclass(frozen=True, slots=True)
class DrawerRoute:
    affordance: str
    object_id: str
    binding: PrismaticBinding
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        container, obj, part = inside_parts(self.affordance)
        if (container, obj, part) != (
            self.binding.object_id,
            self.object_id,
            self.binding.part_id,
        ):
            raise ValueError("Drawer route identity differs from the E6 part binding.")
        low, high = np.asarray(self.lower), np.asarray(self.upper)
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite([low, high]).all()
            or (low > high).any()
        ):
            raise ValueError("Drawer placement bounds must be finite and ordered.")
        object.__setattr__(self, "lower", tuple(map(float, low)))
        object.__setattr__(self, "upper", tuple(map(float, high)))

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "binding": self.binding.payload()}

    @classmethod
    def decode(cls, value: dict[str, Any]) -> DrawerRoute:
        if type(value) is not dict or set(value) != {
            "affordance",
            "object_id",
            "binding",
            "lower",
            "upper",
        }:
            raise ValueError("Drawer route requires exactly its declared fields.")
        return cls(**{**value, "binding": PrismaticBinding.decode(value["binding"])})


def drawer_routes(graph: dict[str, Any], scene: Any) -> tuple[DrawerRoute, ...]:
    configs = {str(c["uid"]): c for c in getattr(scene, "articulations", ())}
    if not configs:
        return ()
    bindings = graph_bindings(graph, scene)
    routes = {}
    for node in graph["nodes"]:
        call = node["call"]
        if call.get("kind") != "place" or "inside" not in call:
            continue
        container, obj, part = inside_parts(call["inside"])
        if container not in configs:
            continue
        if part is None or f"{container}::{part}" not in bindings:
            raise ValueError("Drawer placement has no canonical E6 part binding.")
        binding = bindings[f"{container}::{part}"]
        low, high = drawer_region(configs[container], binding)
        routes[call["inside"]] = DrawerRoute(
            call["inside"], obj, binding, tuple(low), tuple(high)
        )
    return tuple(routes.values())
