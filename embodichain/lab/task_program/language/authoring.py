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

"""Deterministic authoring shorthand expansion into the strict Task Program AST."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = ["NormalizedTaskProgram", "normalize_task_program"]


@dataclass(frozen=True, slots=True)
class NormalizedTaskProgram:
    """Owned canonical declaration and canonical-to-authored diagnostic paths."""

    declaration: dict[str, Any]
    source_map: Mapping[tuple[str | int, ...], tuple[str | int, ...]]

    def __deepcopy__(self, memo: dict[int, Any]) -> NormalizedTaskProgram:
        """Copy configuration metadata without pickling mapping proxies."""
        return NormalizedTaskProgram(deepcopy(self.declaration, memo), self.source_map)

    def original_path(self, path: tuple[str | int, ...]) -> tuple[str | int, ...]:
        """Translate the longest known canonical prefix back to its input."""
        for size in range(len(path), -1, -1):
            if path[:size] in self.source_map:
                return (*self.source_map[path[:size]], *path[size:])
        return path


def normalize_task_program(
    data: object, *, demo_skills: Mapping[str, str] | None = None
) -> NormalizedTaskProgram:
    """Expand bounded, inert shorthand without resolving live providers.

    Args:
        data: JSON/YAML-shaped program, including its trusted integration.
        demo_skills: Explicit alias-to-registered-call mapping from a deployment.

    Returns:
        Independently owned canonical data and source locations. Canonical
        programs retain their original structure and therefore their identity.
    """
    from .decoder import TaskProgramDecodeError, _clone_untrusted_value

    owned = _clone_untrusted_value(
        data, path=(), active=set(), budget=[100_000], depth=0
    )
    locations = {(): ()}

    def fail(path, message):
        raise TaskProgramDecodeError("invalid_shorthand", path, message)

    def reference(value, original, canonical):
        if type(value) is str:
            locations[canonical] = original
            locations[(*canonical, "target")] = original
            return {"kind": "target_ref", "target": value}
        return value

    def call(value, original, canonical, *, generated_kind=False):
        locations[canonical] = original
        if type(value) is not dict:
            return value
        result = dict(value)
        if generated_kind:
            locations[(*canonical, "kind")] = original
        if result.get("kind") == "demo_skill":
            if set(result) - {"kind", "skill", "roles", "resources"}:
                fail(original, "Demo Skill accepts only skill, roles and resources.")
            alias = result.get("skill")
            if type(alias) is not str or alias not in (demo_skills or {}):
                fail(
                    (*original, "skill"),
                    f"Unknown Demo Skill {alias!r}; select a registered deployment alias.",
                )
            result = {
                "kind": "registered",
                "call_id": demo_skills[alias],
                "arguments": result.get("roles", {}),
                "resources": result.get("resources", {}),
            }
            locations[(*canonical, "arguments")] = (*original, "roles")
        for name in ("at", "final_target"):
            if name in result:
                result[name] = reference(
                    result[name], (*original, name), (*canonical, name)
                )
        return result

    def policy(value, original, canonical):
        locations[canonical] = original
        if type(value) is dict and "kind" not in value and len(value) == 1:
            kind, fields = next(iter(value.items()))
            if type(fields) is not dict:
                fail(original, "A policy requires a field mapping.")
            if "kind" in fields:
                fail(original, "A shorthand policy cannot override its kind.")
            original = (*original, kind)
            locations[canonical] = original
            locations[(*canonical, "kind")] = original
            value = {"kind": kind, **fields}
        if type(value) is dict and "at" in value:
            value = dict(value)
            value["at"] = reference(value["at"], (*original, "at"), (*canonical, "at"))
        return value

    def node(value, original, canonical):
        locations[canonical] = original
        if type(value) is list:
            if len(value) == 1:
                return node(value[0], (*original, 0), canonical)
            return {
                "kind": "sequence",
                "items": [
                    node(item, (*original, i), (*canonical, "items", i))
                    for i, item in enumerate(value)
                ],
            }
        if type(value) is not dict:
            return value
        value = dict(value)
        kind = value.get("kind")
        if kind is None:
            if "segment" in value:
                if set(value) - {
                    "segment",
                    "steps",
                    "instruction",
                    "contract",
                    "post",
                    "validators",
                }:
                    fail(original, "Unknown segment fields.")
                value = {"kind": "segment", "name": value.pop("segment"), **value}
                locations[(*canonical, "name")] = (*original, "segment")
            elif len(value) == 1:
                name, fields = next(iter(value.items()))
                if type(fields) is not dict:
                    fail(original, "A node or action requires a field mapping.")
                if "kind" in fields:
                    fail(original, "A shorthand node cannot override its kind.")
                original = (*original, name)
                locations[canonical] = original
                if name in ("repeat", "parallel", "sequence"):
                    value = {"kind": name, **fields}
                else:
                    return {
                        "kind": "invoke",
                        "call": call(
                            {"kind": name, **fields},
                            original,
                            (*canonical, "call"),
                            generated_kind=True,
                        ),
                    }
            else:
                if set(value) & {
                    "body",
                    "count",
                    "items",
                    "call",
                    "branches",
                    "steps",
                    "name",
                }:
                    return value
                fail(original, "An action item must contain exactly one action name.")
            kind = value["kind"]
        if kind == "sequence" and type(value.get("items")) is list:
            value["items"] = [
                node(item, (*original, "items", i), (*canonical, "items", i))
                for i, item in enumerate(value["items"])
            ]
        elif kind == "segment":
            value["steps"] = node(
                value.get("steps"), (*original, "steps"), (*canonical, "steps")
            )
            for name in ("post", "validators"):
                if type(value.get(name)) is list:
                    value[name] = [
                        policy(item, (*original, name, i), (*canonical, name, i))
                        for i, item in enumerate(value[name])
                    ]
        elif kind == "repeat":
            value["body"] = node(
                value.get("body"), (*original, "body"), (*canonical, "body")
            )
        elif kind == "parallel" and type(value.get("branches")) is list:
            value["branches"] = [
                node(item, (*original, "branches", i), (*canonical, "branches", i))
                for i, item in enumerate(value["branches"])
            ]
        elif kind == "invoke":
            value["call"] = call(
                value.get("call"), (*original, "call"), (*canonical, "call")
            )
        return value

    if type(owned) is dict and "program" in owned:
        owned["program"] = node(owned["program"], ("program",), ("program",))
    return NormalizedTaskProgram(owned, MappingProxyType(locations))
