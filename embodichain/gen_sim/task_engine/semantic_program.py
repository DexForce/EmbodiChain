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

"""Bind semantic program segments to declared checks without executing them."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any
import math
from embodichain.lab.sim.scene.source_geometry import source_longest_local_axis
from embodichain.lab.task_program.integrations.simulation.stability import (
    SimulationStabilityConditionCfg,
)
from .semantic_graph import validate_semantic_task_graph
from .program_declarations import ProgramSegmentBinding, build_program_declaration
from .placement_declarations import (
    resolve_relative_placement_routes,
    _route_key_for_node,
    _inside_parts,
    _PLACE_RELATIVE_CALL_ID,
    _UPRIGHT_PLACE_CALL_ID,
    _STACK_PLACE_CALL_ID,
    _COORDINATED_TRANSPORT_CALL_ID,
)
from .stability_declarations import (
    _route_key,
    _lookup_route,
    _RELATIVE_POSITION_TOLERANCE,
)

__all__ = ["build_semantic_program_declaration"]
_CLEAR_RELEASED_CALL_ID = "gen_sim.clear_released"
_PARK_CALL_ID = "simulation.park"


def _program_node(
    node: dict[str, Any],
    *,
    relative_routes: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
    axis_by_object: dict[str, list[float]] | None = None,
    task_stability: bool = False,
    task_stability_kind: str | None = None,
    articulated_reference: bool = False,
    relative_position_tolerance: float = _RELATIVE_POSITION_TOLERANCE,
) -> dict[str, Any]:
    """Materialize one task node as one canonical runtime segment."""
    call = deepcopy(node["call"])
    if call["kind"] == "registered" and call["call_id"] in {
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        arguments = call["arguments"]
        stage_key = _route_key_for_node(
            node,
            str(arguments["object"]),
            str(arguments["reference"]),
            str(arguments["relation"]),
        )
        # Keep the historical three-argument program shape when a placement
        # occurs only once.  A stage selector is needed only when the same
        # object/reference/relation tuple has multiple occurrences; this also
        # keeps old consumers and serialized fixtures byte-for-byte stable.
        repeated_route = False
        if relative_routes is not None and stage_key in relative_routes:
            repeated_route = (
                sum(
                    1
                    for key in relative_routes
                    if len(key) == 4 and key[1:] == stage_key[1:]
                )
                > 1
            )
        if repeated_route:
            arguments["stage_id"] = stage_key[0]
    segment: dict[str, Any] = {
        "kind": "segment",
        "name": str(node["id"]),
        "steps": {"kind": "invoke", "call": call},
    }
    settle_entities: list[str] = []
    settle_preset = "rigid_object"
    if call["kind"] == "place":
        inside = call.get("inside")
        if inside is not None:

            _inside_parts(str(inside))
            settle_preset = "contained_rigid_object"
        settle_entities.append(str(call["object"]))
    elif call["kind"] == "registered" and call["call_id"] in {
        _COORDINATED_TRANSPORT_CALL_ID,
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        if call["call_id"] == _COORDINATED_TRANSPORT_CALL_ID:
            settle_preset = "transported_rigid_object"
        settle_entities.append(str(call["arguments"]["object"]))
    if settle_entities:
        segment["post"] = [
            {
                "kind": "wait_stable",
                "entity": settle_entity,
                "preset": settle_preset,
            }
            for settle_entity in dict.fromkeys(settle_entities)
        ]
    if call["kind"] == "registered" and call["call_id"] in {
        _PLACE_RELATIVE_CALL_ID,
        _UPRIGHT_PLACE_CALL_ID,
        _STACK_PLACE_CALL_ID,
    }:
        arguments = call["arguments"]
        try:
            route = _lookup_route(
                relative_routes or {},
                node,
                str(arguments["object"]),
                str(arguments["reference"]),
                str(arguments["relation"]),
            )
        except KeyError as exc:
            raise ValueError(
                "Relative placement has no generated route for "
                f"{_route_key_for_node(node, str(arguments['object']), str(arguments['reference']), str(arguments['relation']))!r}."
            ) from exc
        if articulated_reference:
            # The shared validator is rigid-only. The GenSim post-policy uses
            # the same displacement/tolerance and observes either native root.
            if not task_stability:
                raise ValueError(
                    "Articulated references require task stability acceptance."
                )
        elif not (
            task_stability
            and task_stability_kind == "supported_region"
            and node.get("task_type") == "E1"
            and call["call_id"] == _PLACE_RELATIVE_CALL_ID
            and arguments.get("reference") == "table"
            and arguments.get("relation") == "on"
            and set(arguments).issubset({"object", "reference", "relation", "stage_id"})
        ):
            segment["validators"] = [
                {
                    "kind": "object_near_relative_target",
                    "object": route["object_id"],
                    "reference": route["reference_entity_id"],
                    "displacement": deepcopy(route["world_displacement"]),
                    "position_tolerance": relative_position_tolerance,
                }
            ]
    if (
        not task_stability
        and node.get("task_type") == "E2"
        and (
            call["kind"] == "place"
            or (
                call["kind"] == "registered"
                and call["call_id"] in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
            )
        )
    ):
        raise ValueError("E2 release requires a task-owned upright stability preset.")
    if node.get("task_type") == "E3" and call["kind"] == "place":
        target = call.get("at")
        if type(target) is not dict or set(target) != {"kind", "target"}:
            raise ValueError("Generated E3 Place requires one exact return target.")
        segment["validators"] = [
            {
                "kind": "object_near_target",
                "object": str(call["object"]),
                "target": str(target["target"]),
                "position_tolerance": 0.06,
            }
        ]
    if task_stability:
        arguments = call.get("arguments", {})
        entity = call.get("object", arguments.get("object"))
        segment.setdefault("post", []).append(
            {
                "kind": "wait_stable",
                "entity": entity,
                "preset": f"gen_sim.{node['id']}.stable",
            }
        )
    return segment


def build_semantic_program_declaration(
    graph: Mapping[str, Any],
    program_id: str,
    scene: Any,
    *,
    stability_conditions: Mapping[str, SimulationStabilityConditionCfg],
    operation_groups: Mapping[str, Sequence[ProgramSegmentBinding]],
    relative_position_tolerance: float = _RELATIVE_POSITION_TOLERANCE,
) -> dict[str, Any]:
    """Bind declared acceptance to exact semantic segments and cleanup boundaries.

    Args:
        graph: Validated semantic graph with independent occurrence identities.
        program_id: Exact public program identity.
        scene: Frozen scene used only for shared source geometry descriptors.
        stability_conditions: Owned standard named stability declarations.
        operation_groups: Explicit complete per-segment bindings for control
            operations. Their owners supply joint limits, phase presets and
            native link identities; this function does not infer those values.
        relative_position_tolerance: Explicit relative-target validator margin.
            The ordinary declaration default is 0.05 m; source frozen/experiment
            generation supplies 0.10 m explicitly without changing shared defaults.

    Returns:
        Existing Task Program language declaration. It retains final joint
        rechecks and release/cleanup acceptance ordering. Ordinary catalog
        preflight remains mandatory before any bundle is published.
    """
    if (
        type(relative_position_tolerance) not in (int, float)
        or not math.isfinite(relative_position_tolerance)
        or relative_position_tolerance <= 0
    ):
        raise ValueError("Relative position tolerance must be finite and positive.")
    source = validate_semantic_task_graph(graph)
    groups = {group["id"]: group for group in source["task_groups"]}
    if set(operation_groups) - set(groups):
        raise ValueError("Operation bindings must name existing semantic groups.")
    conditions = {}
    node_ids = {node["id"] for node in source["nodes"]}
    for name, condition in stability_conditions.items():
        if type(condition) is not SimulationStabilityConditionCfg:
            raise TypeError("Stability conditions must use the public typed contract.")
        if name not in {f"gen_sim.{node_id}.stable" for node_id in node_ids}:
            raise ValueError(
                "Stability condition must name an existing semantic occurrence."
            )
        conditions[name] = condition.to_condition_dict()
    routes = {
        _route_key(route): route
        for route in resolve_relative_placement_routes(source, scene, settled=True)
    }
    upright_routes = {
        _route_key(route): route
        for route in resolve_relative_placement_routes(
            source, scene, settled=True, upright_only=True
        )
    }
    axes = {
        str(item["runtime_uid"]): source_longest_local_axis(item)
        for item in scene.planner_objects
    }
    articulated = {
        str(item["runtime_uid"])
        for item in scene.planner_objects
        if item.get("role") == "articulation"
    }
    items = {}
    for node in source["nodes"]:
        call = node["call"]
        args = call.get("arguments", {})
        name = f"gen_sim.{node['id']}.stable"
        condition = conditions.get(name, {})
        item = _program_node(
            node,
            relative_routes=(
                upright_routes
                if call.get("call_id") == _UPRIGHT_PLACE_CALL_ID
                else routes
            ),
            axis_by_object=axes,
            task_stability=name in conditions,
            task_stability_kind=(
                condition.get("kind")
                if condition.get("entity") == args.get("object")
                and condition.get("reference") == args.get("reference")
                else None
            ),
            articulated_reference=args.get("reference") in articulated,
            relative_position_tolerance=relative_position_tolerance,
        )
        if condition.get("support_contract_revision") == 1:
            item["post"] = [
                policy
                for policy in item.get("post", ())
                if policy.get("preset") == name
            ]
        items[node["id"]] = item
    bindings = {
        node_id: ProgramSegmentBinding(
            node_id,
            call=item["steps"]["call"],
            post=tuple(item["post"]) if "post" in item else None,
            validators=tuple(item["validators"]) if "validators" in item else None,
        )
        for node_id, item in items.items()
    }
    for group_id, group in groups.items():
        selected = operation_groups.get(group_id)
        control_calls = [
            items[node_id]["steps"]["call"].get("call_id", "")
            for node_id in group["node_ids"]
        ]
        requires_owner = any(
            call == "gen_sim.articulation_slide"
            or call in {"gen_sim.press_prepare", "gen_sim.twist_prepare"}
            or call.startswith(("gen_sim.press_prepare.", "gen_sim.twist_prepare."))
            for call in control_calls
        )
        if selected is None and requires_owner:
            raise ValueError(
                f"Operation group {group_id!r} requires explicit owner-resolved bindings."
            )
        if selected is not None:
            selected = tuple(selected)
            if any(type(binding) is not ProgramSegmentBinding for binding in selected):
                raise TypeError("Operation groups require typed segment bindings.")
            if [binding.node_id for binding in selected] != group["node_ids"]:
                raise ValueError(
                    "Operation bindings must cover their exact ordered group occurrences."
                )
            for binding in selected:
                if (
                    binding.terminal_post is not None
                    or binding.terminal_validators is not None
                    or binding.defer_to_terminal
                ):
                    raise ValueError(
                        "Operation-owned checks must bind directly to each segment."
                    )
                if not binding.post:
                    raise ValueError(
                        "Operation segments require their explicit physical post-policy."
                    )
                bindings[binding.node_id] = binding
            continue
        terminal = group["node_ids"][-1]
        for node_id in reversed(group["node_ids"]):
            binding = bindings[node_id]
            policies = tuple(
                policy
                for policy in binding.post or ()
                if policy["preset"].startswith("gen_sim.")
            )
            if not policies:
                policies = tuple(binding.post or ())
            validators = tuple(binding.validators or ())
            if not policies and not validators:
                continue
            if node_id != terminal:
                remaining = group["node_ids"][group["node_ids"].index(node_id) + 1 :]
                cleanup = [items[name]["steps"]["call"] for name in remaining]
                call = items[node_id]["steps"]["call"]
                defer = (
                    (
                        call.get("kind") == "place"
                        or call.get("call_id")
                        in {
                            _PLACE_RELATIVE_CALL_ID,
                            _UPRIGHT_PLACE_CALL_ID,
                            _STACK_PLACE_CALL_ID,
                        }
                    )
                    and (
                        any(
                            call.get("call_id") == _CLEAR_RELEASED_CALL_ID
                            for call in cleanup
                        )
                        or conditions.get(f"gen_sim.{node_id}.stable", {}).get(
                            "support_contract_revision"
                        )
                        == 1
                    )
                    and all(
                        call.get("kind") == "registered"
                        and call.get("call_id")
                        in {_CLEAR_RELEASED_CALL_ID, _PARK_CALL_ID}
                        for call in cleanup
                    )
                )
                bindings[node_id] = replace(
                    binding,
                    terminal_post=tuple(binding.post or ()) if defer else policies,
                    terminal_validators=validators,
                    defer_to_terminal=defer,
                )
            break
    final_joints = {}
    for node in source["nodes"]:
        for validator in bindings[node["id"]].validators or ():
            if validator["kind"] == "articulation_joint_position":
                final_joints[(validator["articulation"], validator["joint"])] = (
                    validator
                )
    return build_program_declaration(
        source,
        program_id,
        bindings=tuple(bindings[node["id"]] for node in source["nodes"]),
        final_validators=tuple(final_joints.values()),
    )
