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

"""Bind unchanged hand-command authority without granting staging recovery."""

from __future__ import annotations

from typing import Any

from embodichain.lab.task_program.semantics.calls import Pick, RegisteredSemanticCall

from .invocation_policy import _invocation_ids
from .staging_recovery import verified_retention_sources

__all__: list[str] = []

HELD_COMMAND_REVISION = 2
_TRANSPORT = "simulation.move_held_object"


def _retains(call_id: str | None) -> bool:
    return call_id in {"pick", "gen_sim.align_held", _TRANSPORT} or (
        isinstance(call_id, str) and call_id.startswith("gen_sim.pick.")
    )


def held_command_declarations(graph: dict[str, Any]) -> dict[str, str]:
    """Select adjacent same-object/resource held commands, independent of E type."""
    result = {}
    for previous, current in zip(graph["nodes"], graph["nodes"][1:]):
        before, after = previous["call"], current["call"]
        before_id = "pick" if before["kind"] == "pick" else before.get("call_id")
        entity = before.get("object", before.get("arguments", {}).get("object"))
        resource = before.get("resources", {}).get("primary")
        if (
            after.get("call_id") == _TRANSPORT
            and _retains(before_id)
            and isinstance(entity, str)
            and entity == after.get("arguments", {}).get("object")
            and isinstance(resource, str)
            and resource == after.get("resources", {}).get("primary")
            and current.get("depends_on") == [previous["id"]]
        ):
            result[current["id"]] = previous["id"]
    return result


def _identity(call: Any) -> tuple[str | None, str | None, str | None]:
    if type(call) is Pick:
        return "pick", call.object.entity_id, call.resources.get("primary")
    if type(call) is RegisteredSemanticCall:
        return call.call_id, call.arguments.get("object"), call.resources.get("primary")
    return None, None, None


def bind_held_command_sources(
    declarations: object,
    program: Any,
    profile: Any,
    preset_id: str | None,
    rigid_ids: frozenset[str],
    *,
    revision: int = 1,
) -> dict[str, dict[str, Any]]:
    """Derive provenance from exact compiled adjacent calls, not declared IDs."""
    if type(revision) is not int or revision not in (1, 2):
        raise ValueError("Unsupported held command revision.")
    if type(declarations) is not dict or any(
        type(name) is not str or type(source) is not str
        for name, source in declarations.items()
    ):
        raise ValueError("Held command sources must map segment names to predecessors.")
    if not declarations:
        return {}
    ids = _invocation_ids(program)
    if len(set(ids.values())) != len(ids):
        raise ValueError(
            "Held command invocations must have unique canonical identities."
        )
    segments = tuple(program.iter_segments())
    positions = {segment.name: index for index, segment in enumerate(segments)}
    sources = []
    for name, predecessor in declarations.items():
        index = positions.get(name, -1)
        if index < 1 or segments[index - 1].name != predecessor:
            raise ValueError(
                "Held command source must be the exact adjacent predecessor."
            )
        before, after = segments[index - 1], segments[index]
        if any(
            segment.parallel_block is not None or segment.repeat_frames
            for segment in (before, after)
        ):
            raise ValueError(
                "Held command authority requires non-repeated serial calls."
            )
        source_id, source_entity, source_resource = _identity(before.calls[0].call)
        call_id, entity, resource = _identity(after.calls[0].call)
        if (
            call_id != _TRANSPORT
            or not _retains(source_id)
            or entity not in rigid_ids
            or entity != source_entity
            or not isinstance(resource, str)
            or resource != source_resource
        ):
            raise ValueError(
                "Held command source has incompatible call, object or resource."
            )
        sources.append((ids[name], ids[predecessor], source_id, entity, resource))
    verified = verified_retention_sources(
        profile, preset_id, tuple(sources), retention_call_id=_TRANSPORT
    )
    if revision == 2:
        for source in verified.values():
            source["command_continuity_revision"] = revision
    return verified
