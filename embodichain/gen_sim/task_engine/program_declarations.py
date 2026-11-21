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

"""Program projection from semantic nodes and explicit owner-resolved bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .semantic_graph import validate_semantic_task_graph

__all__ = ["ProgramSegmentBinding", "build_program_declaration"]


@dataclass(frozen=True, slots=True)
class ProgramSegmentBinding:
    """Transient program binding, not an additional runtime configuration store.

    The owner supplies post-policy references and validators with their already
    resolved tolerances. Optional terminal fields identify checks copied to the
    group's final cleanup segment. Deferral removes the original segment checks
    only when explicitly selected; action release/effect contracts are untouched.
    """

    node_id: str
    call: Mapping[str, Any] | None = None
    post: tuple[Mapping[str, Any], ...] | None = None
    validators: tuple[Mapping[str, Any], ...] | None = None
    terminal_post: tuple[Mapping[str, Any], ...] | None = None
    terminal_validators: tuple[Mapping[str, Any], ...] | None = None
    defer_to_terminal: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.node_id) is not str
            or not self.node_id
            or self.node_id != self.node_id.strip()
        ):
            raise ValueError("Program binding requires an exact semantic node ID.")
        if type(self.defer_to_terminal) is not bool:
            raise TypeError("Program acceptance deferral must be explicit bool.")
        if self.call is not None:
            if not isinstance(self.call, Mapping):
                raise TypeError("Projected call must be a declaration mapping.")
            object.__setattr__(self, "call", deepcopy(dict(self.call)))
        for name in ("post", "validators", "terminal_post", "terminal_validators"):
            values = getattr(self, name)
            if values is None:
                continue
            if type(values) is not tuple or any(
                not isinstance(value, Mapping) for value in values
            ):
                raise TypeError(f"{name} must be a tuple of declaration mappings.")
            object.__setattr__(
                self, name, tuple(deepcopy(dict(value)) for value in values)
            )


def build_program_declaration(
    graph: Mapping[str, Any],
    program_id: str,
    *,
    bindings: Sequence[ProgramSegmentBinding] = (),
    final_validators: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the existing program format without motion, geometry or live reads.

    Each graph node retains its execution segment identity and ordering. Call
    overrides must be exact graph projections, including the existing declared
    placement-stage selector rule. Owners choose acceptance presets/thresholds;
    no task ID, object name or analysis-cache index changes execution policies.
    The caller completes ordinary catalog preflight after integration assembly.
    """
    source = validate_semantic_task_graph(graph)
    if (
        type(program_id) is not str
        or not program_id
        or program_id != program_id.strip()
    ):
        raise ValueError("Program ID must be an exact nonempty string.")
    selected = {}
    known = {node["id"] for node in source["nodes"]}
    for item in bindings:
        if type(item) is not ProgramSegmentBinding:
            raise TypeError("Program bindings must be typed declarations.")
        if item.node_id not in known or item.node_id in selected:
            raise ValueError(
                "Program bindings must select unique existing semantic nodes."
            )
        selected[item.node_id] = item
    items = []
    for node in source["nodes"]:
        binding = selected.get(node["id"], ProgramSegmentBinding(node["id"]))
        segment = {
            "kind": "segment",
            "name": node["id"],
            "steps": {
                "kind": "invoke",
                "call": deepcopy(
                    dict(node["call"] if binding.call is None else binding.call)
                ),
            },
        }
        if binding.post is not None:
            segment["post"] = deepcopy(list(binding.post))
        if binding.validators is not None:
            segment["validators"] = deepcopy(list(binding.validators))
        items.append(segment)
    by_id = {item["name"]: item for item in items}
    for group in source["task_groups"]:
        terminal = by_id[group["node_ids"][-1]]
        owners = [
            selected[node_id]
            for node_id in group["node_ids"]
            if node_id in selected
            and (
                selected[node_id].terminal_post is not None
                or selected[node_id].terminal_validators is not None
                or selected[node_id].defer_to_terminal
            )
        ]
        if len(owners) > 1:
            raise ValueError(
                "A group terminal acceptance binding must have one explicit source occurrence."
            )
        for binding in owners:
            original = by_id[binding.node_id]
            if original is terminal:
                if binding.defer_to_terminal:
                    raise ValueError(
                        "A terminal segment cannot defer its checks to itself."
                    )
                if (
                    binding.terminal_post is not None
                    and binding.terminal_post != binding.post
                ) or (
                    binding.terminal_validators is not None
                    and binding.terminal_validators != binding.validators
                ):
                    raise ValueError(
                        "Terminal checks must be declared directly on their own segment."
                    )
                continue
            if binding.defer_to_terminal:
                if tuple(binding.terminal_post or ()) != tuple(
                    binding.post or ()
                ) or tuple(binding.terminal_validators or ()) != tuple(
                    binding.validators or ()
                ):
                    raise ValueError(
                        "Deferred acceptance must retain every original check at the terminal boundary."
                    )
                original.pop("post", None)
                original.pop("validators", None)
            if binding.terminal_post is not None:
                terminal.setdefault("post", []).extend(
                    deepcopy(list(binding.terminal_post))
                )
            if binding.terminal_validators is not None:
                terminal.setdefault("validators", []).extend(
                    deepcopy(list(binding.terminal_validators))
                )
    if final_validators:
        last = items[-1].setdefault("validators", [])
        for validator in final_validators:
            if not isinstance(validator, Mapping):
                raise TypeError("Final validators must be declaration mappings.")
            value = deepcopy(dict(validator))
            if value not in last:
                last.append(value)
    program = {
        "program_id": program_id,
        "targets": deepcopy(source["targets"]),
        "program": {"kind": "sequence", "items": items},
    }
    _verify_program_projection(program, source)
    return program


def _verify_program_projection(
    program: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    """Require the executable program to be an exact projection of the graph."""
    if type(program) is not dict:
        raise ValueError("Task Program artifact must contain an exact mapping.")
    if program.get("targets") != graph["targets"]:
        raise ValueError("Task Program targets do not match SemanticTaskGraph targets.")
    body = program.get("program")
    if type(body) is not dict or body.get("kind") != "sequence":
        raise ValueError("Generated Task Program must contain one sequence body.")
    items = body.get("items")
    if type(items) is not list or len(items) != len(graph["nodes"]):
        raise ValueError(
            "Task Program segment count does not match SemanticTaskGraph nodes."
        )
    for index, (item, node) in enumerate(zip(items, graph["nodes"], strict=True)):
        if type(item) is not dict or item.get("kind") != "segment":
            raise ValueError(f"Task Program item {index} must be one segment.")
        if item.get("name") != node["id"]:
            raise ValueError(
                f"Task Program item {index} does not match graph node ID "
                f"{node['id']!r}."
            )
        steps = item.get("steps")
        projected_call = steps.get("call") if type(steps) is dict else None
        # A runtime-only placement selector must name this graph stage before
        # it can be omitted from the immutable call projection.
        if type(projected_call) is dict:
            projected_call = deepcopy(projected_call)
            arguments = projected_call.get("arguments")
            graph_arguments = node["call"].get("arguments")
            declared_stage = (
                type(graph_arguments) is dict and "stage_id" in graph_arguments
            )
            if (
                type(arguments) is dict
                and "stage_id" in arguments
                and not declared_stage
            ):
                if (
                    projected_call.get("kind") != "registered"
                    or projected_call.get("call_id")
                    not in {
                        "simulation.place_relative",
                        "gen_sim.place_upright",
                        "gen_sim.stack_place",
                    }
                    or type(arguments["stage_id"]) is not str
                    or arguments["stage_id"] != node["task_instance_id"]
                ):
                    raise ValueError(
                        f"Task Program segment {node['id']!r} runtime stage_id "
                        "must select its exact relative-placement task instance."
                    )
                arguments = dict(arguments)
                arguments.pop("stage_id", None)
                projected_call["arguments"] = arguments
        if (
            type(steps) is not dict
            or steps.get("kind") != "invoke"
            or projected_call != node["call"]
        ):
            raise ValueError(
                f"Task Program segment {node['id']!r} is not an exact "
                "SemanticTaskGraph call projection."
            )
