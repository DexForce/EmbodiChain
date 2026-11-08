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

"""Semantic-only plan checking, separate from grounded bundle validation."""

from __future__ import annotations
from typing import Any
import json
from .contracts import canonical_hash, validate_task_candidate
from .semantic_graph import validate_semantic_task_graph
from embodichain.lab.task_program.language.semantic_contracts import (
    semantic_planning_catalog,
    validate_semantic_call_payload,
)

__all__ = ["validate_semantic_plan", "semantic_plan_rules"]

from .ontology import TASK_CONTRACTS


class _PlanDiagnostics(ValueError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.diagnostics = {
            "schema_version": "semantic_plan_diagnostics/v1",
            "code": "symbolic_state_invalid",
            "errors": errors,
            "note": "States after an invalid node are projected only to discover independent errors; any diagnostic rejects the complete plan.",
        }
        super().__init__(json.dumps(self.diagnostics, ensure_ascii=False))


def semantic_plan_rules() -> dict[str, Any]:
    """Derive producer obligations from task affordances and call declarations.

    Returns:
        Required catalog capabilities and candidate calls for each task contract.
    """
    catalog = semantic_planning_catalog()["calls"]
    return {
        key: {
            "semantics": contract.semantics,
            "requires_capabilities": sorted(
                contract.required_affordances - {"graspable"}
            ),
            "requires_one_of": sorted(
                cid
                for cid, call in catalog.items()
                if set(call["capabilities"])
                & (contract.required_affordances - {"graspable"})
            ),
        }
        for key, contract in TASK_CONTRACTS.items()
    }


def validate_semantic_plan(
    graph: dict[str, Any],
    candidate: dict[str, Any],
    grounding: dict[str, Any],
    annotations: dict[str, Any],
    requirements: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check calls, symbolic state and source coverage without physical evidence.

    Args:
        graph: Canonical graph carrying unbound logical identities.
        candidate: Source task candidate with validated task intent.
        grounding: Visible, instruction-declared and unresolved entity evidence.
        annotations: Per-node semantic subject, reference, relation and intent.
        requirements: Quoted source clauses with implementing node IDs.
        bindings: Explicit runtime binding requests; not execution evidence.
    Returns:
        Recomputable phase-specific coverage and pending-binding summary.
    Raises:
        ValueError: Invalid call, topology, state, identity or missing semantics.
            State diagnostics may propose resource-slot edits for a repair
            request; these suggestions never change the input or establish
            validity without a complete subsequent validation.
    """
    graph = validate_semantic_task_graph(graph)
    candidate = validate_task_candidate(candidate)
    if (
        graph["task_id"] != candidate["draft"]["task_id"]
        or graph["instruction"] != candidate["draft"]["instruction"]
    ):
        raise ValueError("Plan source identity mismatch")
    for entity in grounding["objects"]:
        if entity.get("provenance") not in {
            "observed",
            "instruction_declared",
            "runtime_required",
        }:
            raise ValueError("Invalid evidence provenance")
        box = entity.get("bbox")
        if entity["provenance"] == "observed":
            import math

            if (
                type(box) is not list
                or len(box) != 4
                or any(
                    type(v) not in {int, float}
                    or not math.isfinite(v)
                    or not 0 <= v <= 1
                    for v in box
                )
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError("Invalid normalized image evidence")
        elif box is not None:
            raise ValueError("Unobserved entities cannot have boxes")
    steps = {s["id"]: s for s in candidate["draft"]["steps"]}
    nodes = {n["id"]: n for n in graph["nodes"]}
    entities = {o["id"] for o in grounding["objects"]}
    entity_by_id = {o["id"]: o for o in grounding["objects"]}

    def subject_matches(actual: str, expected: str, step: dict[str, Any]) -> bool:
        seen = set()
        while actual is not None:
            if actual == expected:
                return True
            if actual in seen:
                raise ValueError("Cyclic part hierarchy")
            seen.add(actual)
            if TASK_CONTRACTS[step["task_type"]].source_structure != "articulation":
                return False
            actual = entity_by_id[actual].get("parent")
            if actual is not None and actual not in entities:
                raise ValueError("Unknown part parent")
        return False

    if set(annotations) != set(nodes):
        raise ValueError("Every node requires exactly one semantic annotation")
    for annotation in annotations.values():
        if (
            type(annotation) is not dict
            or set(annotation) != {"object", "reference", "relation", "intent"}
            or type(annotation["intent"]) is not dict
        ):
            raise ValueError("Invalid semantic annotation: intent must be a dictionary")
    if set(g["id"] for g in graph["task_groups"]) != set(steps):
        raise ValueError("Graph must retain every source step")
    _validate_reference_identity(
        grounding["bindings"], candidate["scene_request"]["references"]
    )
    expected_refs = {
        f"{step['id']}.{role}"
        for step in steps.values()
        for role in ("object", "target")
        if step[role]["kind"] != "none"
    }
    if set(grounding["bindings"]) != expected_refs:
        raise ValueError("Logical bindings must cover every source selector exactly")
    for step in steps.values():
        for role in ("object", "target"):
            selector = step[role]
            if selector["kind"] == "step_result" and grounding["bindings"][
                f"{step['id']}.{role}"
            ] != grounding["bindings"].get(f"{selector['step_id']}.object"):
                raise ValueError("Cross-step result identity was lost")
    for ref, ids in grounding["bindings"].items():
        if not ids or set(ids) - entities:
            raise ValueError(f"Unresolved logical identity for {ref}")
    bound_kinds = {}
    for binding in bindings:
        if (
            set(binding)
            != {"id", "node_id", "kind", "parameter", "subject", "value", "source"}
            or binding["node_id"] not in nodes
            or binding["subject"] not in entities
        ):
            raise ValueError("Malformed runtime binding requirement")
        node = nodes[binding["node_id"]]
        ann = annotations[binding["node_id"]]
        allowed = set(validate_semantic_call_payload(node["call"])["runtime_bindings"])
        if ann["intent"].get("return_to_origin") is True:
            allowed.add("source_pose")
        expected_parameter = (
            "runtime.initial_pose"
            if binding["kind"] == "source_pose"
            else "runtime." + binding["kind"]
        )
        if (
            binding["subject"] != ann["object"]
            or binding["kind"] not in allowed
            or binding["value"] != ann
            or binding["source"] != node["task_instance_id"]
            or binding["parameter"] != expected_parameter
        ):
            raise ValueError(
                "Runtime binding does not match its source node and intent"
            )
        key = (binding["node_id"], binding["kind"])
        if key in bound_kinds:
            raise ValueError("Duplicate binding requirement")
        bound_kinds[key] = binding
    held_after: dict[str, dict[str, str]] = {}
    held = {}
    acquisition_nodes: dict[str, str] = {}
    last_object_node = {}
    effects = {}
    state_errors: list[dict[str, Any]] = []

    dependency_ancestors = _ancestor_sets(graph["nodes"])

    def ancestors(nid: str) -> set[str]:
        return dependency_ancestors[nid]

    for node in graph["nodes"]:
        cid = node["call"].get("call_id", node["call"]["kind"])
        contract = validate_semantic_call_payload(node["call"])
        ann = annotations[node["id"]]
        if (
            set(ann) != {"object", "reference", "relation", "intent"}
            or ann["object"] not in entities
        ):
            raise ValueError(f"Invalid annotation for {node['id']}")
        if ann["reference"] is not None and ann["reference"] not in entities:
            raise ValueError(
                f"Unknown reference entity {ann['reference']!r} on {node['id']}. An original pose/support alias is NOT an observed entity: for return_to_origin set semantic.reference to the source object ID {ann['object']!r} (or null); keep the pose/affordance alias only in call arguments and the source_pose binding."
            )
        if any(
            key in ann["intent"]
            for key in {
                "position",
                "pose",
                "qpos",
                "quaternion",
                "angle",
                "twist_angle",
                "joint_position",
                "waypoint",
                "trajectory",
                "grasp_pose",
                "target_pose",
            }
        ):
            raise ValueError(
                "Metric execution data is forbidden in semantic intent; declare a runtime binding"
            )
        if type(ann["intent"]) is not dict:
            raise ValueError("Node intent must be declarative")
        if (
            ann["reference"] == ann["object"]
            and ann["relation"] in {"inside", "on"}
            and not (
                ann["relation"] == "on"
                and ann["intent"].get("return_to_origin") is True
            )
        ):
            raise ValueError(
                "An object cannot be its own placement destination; original-pose binding must be explicit"
            )
        args = node["call"].get("arguments", node["call"])
        obj = args.get("object", args.get("handle", ann["object"]))
        part_argument = next(
            (name for name, kind in contract["required"].items() if kind == "part"),
            None,
        )
        if obj != ann["object"] and not (
            part_argument is not None
            and subject_matches(obj, ann["object"], steps[node["task_instance_id"]])
        ):
            raise ValueError(
                f"Node {node['id']} call subject {obj!r} differs from annotation {ann['object']!r}; control parts require a declared parent relationship"
            )
        if args.get("reference") is not None and args["reference"] != ann["reference"]:
            raise ValueError("Call reference differs from annotation")
        if cid == "place":
            destination = (
                "inside" if "inside" in args else "on" if "on" in args else None
            )
            target_id = args.get(destination) if destination else None
            if (
                isinstance(target_id, str)
                and target_id in entities
                and ann["reference"] is not None
                and ann["intent"].get("return_to_origin") is not True
                and target_id != ann["reference"]
                and entity_by_id[target_id].get("parent") != ann["reference"]
            ):
                raise ValueError(
                    f"Node {node['id']} placement destination {target_id!r} differs from its declared reference {ann['reference']!r}"
                )
            if destination is not None and ann["relation"] != destination:
                raise ValueError(
                    f"Node {node['id']} uses place.{destination} but semantic.relation={ann['relation']!r}. Set THIS placement node relation to {destination!r}. Pour relation above belongs only on pouring/transport nodes; restoration place.on requires relation on."
                )
        if "state" in args and args["state"] != ann["intent"].get("target_state"):
            raise ValueError("Control state differs from requested task state")
        if args.get("relation") is not None and args["relation"] != ann["relation"]:
            raise ValueError("Call relation differs from annotation")
        if obj in last_object_node and last_object_node[obj] not in ancestors(
            node["id"]
        ):
            raise ValueError(
                "Operations on the same object must have an explicit dependency"
            )
        last_object_node[obj] = node["id"]
        resources = node["call"].get("resources", {})
        actor = resources.get("primary", "auto")
        effect = contract["effect"]

        def state_error(
            code: str,
            hint: str,
            repair_operations: list[dict[str, Any]] | None = None,
        ) -> None:
            diagnostic = {
                "code": code,
                "node_id": node["id"],
                "call_id": cid,
                "subject": obj,
                "primary_resource": actor,
                "held_before": dict(held),
                "hint": hint,
            }
            if repair_operations:
                diagnostic["repair_operations"] = repair_operations
            state_errors.append(diagnostic)

        if effect in {"acquire", "acquire_align"}:
            if obj in held or actor in held.values():
                state_error(
                    "occupied_acquisition",
                    f"Remove a redundant acquisition of {obj!r} or release the occupied resource first. Pour restores orientation; never add axis_align to a held pouring source.",
                )
            held[obj] = actor
            acquisition_nodes[obj] = node["id"]
        elif effect in {"release", "held"}:
            if obj not in held:
                state_error(
                    "missing_held_object",
                    f"Insert a canonical pick of object {obj!r} BEFORE THIS node {node['id']!r}, or remove this redundant transport. A place releases the object and a later relocation requires reacquisition. Do not acquire the reference object.",
                )
            elif held[obj] != actor:
                state_error(
                    "holder_mismatch",
                    f"Use the same acquisition resource {held[obj]!r} for THIS node {node['id']!r}; omit resources consistently when it is auto. Do not change the object's identity.",
                    [
                        {
                            "operation": "remove" if held[obj] == "auto" else "set",
                            "node_id": node["id"],
                            "path": "/call/resources/primary",
                            **({"value": held[obj]} if held[obj] != "auto" else {}),
                        }
                    ],
                )
            if effect == "release":
                held.pop(obj, None)
                acquisition_nodes.pop(obj, None)
        elif effect == "transfer":
            source, dest = resources["source"], resources["destination"]
            if held.get(obj) != source or dest in held.values():
                state_error(
                    "handover_state",
                    f"Handover requires {obj!r} held by {source!r} and receiver {dest!r} free. Acquire using the source arm before THIS node.",
                    (
                        [
                            {
                                "operation": "set",
                                "node_id": acquisition_nodes[obj],
                                "path": "/call/resources/primary",
                                "value": source,
                            }
                        ]
                        if held.get(obj) == "auto" and obj in acquisition_nodes
                        else None
                    ),
                )
            held[obj] = dest
            acquisition_nodes.pop(obj, None)
        elif effect == "coordinated_release":
            if held:
                state_error(
                    "coordinated_arms_busy",
                    "Release existing held objects before this two-arm operation.",
                )
        elif effect == "control" and actor in held.values():
            state_error(
                "control_arm_busy",
                "Release the selected arm's held object before this control operation.",
            )
        for kind in contract["runtime_bindings"]:
            if (node["id"], kind) not in bound_kinds:
                raise ValueError(f"Missing runtime binding: {node['id']}.{kind}")
        effects[node["id"]] = cid
        held_after[node["id"]] = dict(held)
    if state_errors:
        raise _PlanDiagnostics(state_errors)
    referenced_nodes = set()
    for requirement in requirements:
        if (
            set(requirement) != {"id", "quote", "node_ids"}
            or not isinstance(requirement["quote"], str)
            or requirement["quote"] not in graph["instruction"]
            or not requirement["node_ids"]
            or set(requirement["node_ids"]) - set(nodes)
        ):
            raise ValueError(
                "Instruction requirements must quote the source and reference real nodes"
            )
        referenced_nodes.update(requirement["node_ids"])
    if referenced_nodes != set(nodes):
        raise ValueError("Every call must be justified by the original instruction")
    for step_id, step in steps.items():
        group = next(g for g in graph["task_groups"] if g["id"] == step_id)
        calls = [nodes[n] for n in group["node_ids"]]
        if not set(effects[n["id"]] for n in calls) & set(
            semantic_plan_rules()[step["task_type"]]["requires_one_of"]
        ):
            raise ValueError(
                f"Missing semantic producer for {step_id}:{step['task_type']}"
            )
        kinds = {effects[n["id"]] for n in calls}
        if step.get("terminal_behavior") == "place" and not kinds & {
            "place",
            "simulation.place_relative",
            "simulation.coordinated_transport",
        }:
            raise ValueError(f"Missing terminal placement/release for {step_id}")
        if (
            "target_setting"
            in TASK_CONTRACTS[step["task_type"]].applicable_intent_fields
            and step.get("twist_goal_mode") != "coarse"
            and not any(
                annotations[n["id"]]["intent"].get("target_setting")
                == step["target_setting"]
                for n in calls
            )
        ):
            raise ValueError("Requested printed knob setting was lost")
        producer_ids = set(semantic_plan_rules()[step["task_type"]]["requires_one_of"])
        for subject in grounding["bindings"].get(f"{step_id}.object", []):
            producers = [
                n
                for n in calls
                if effects[n["id"]] in producer_ids
                and subject_matches(annotations[n["id"]]["object"], subject, step)
            ]
            if len(producers) != 1:
                raise ValueError(
                    f"{step_id} requires exactly one semantic producer per declared subject, got {len(producers)} for {subject}"
                )
        expected = grounding["bindings"].get(f"{step_id}.object")
        if expected and not all(
            any(
                subject_matches(annotations[n["id"]]["object"], obj, step)
                for n in calls
            )
            for obj in expected
        ):
            actual = [annotations[n["id"]]["object"] for n in calls]
            raise ValueError(
                f"Step {step_id} expects objects {expected}, got {actual}. A control part must declare parent: instruction_object_id in grounding, or use that pending logical object identity."
            )
        target = grounding["bindings"].get(f"{step_id}.target")
        if step.get("terminal_behavior") == "hold":
            if TASK_CONTRACTS[step["task_type"]].resource_mode == "coordinated":
                raise ValueError(
                    "Coordinated hold requires a matching hold call; the current transport call releases"
                )
            if any(obj not in held_after[calls[-1]["id"]] for obj in expected or []):
                raise ValueError("Requested terminal hold is not retained")
        if target:
            placement_goal = (
                "placeable" in TASK_CONTRACTS[step["task_type"]].required_affordances
                or step.get("terminal_behavior") == "place"
            )
            for obj in expected or []:
                applicable = [
                    n
                    for n in calls
                    if subject_matches(annotations[n["id"]]["object"], obj, step)
                    and (
                        validate_semantic_call_payload(n["call"])["effect"]
                        in {"release", "coordinated_release"}
                        if placement_goal
                        else effects[n["id"]] in producer_ids
                    )
                ]
                if not any(
                    annotations[n["id"]]["reference"] in target
                    and (
                        step["relation"] == "none"
                        or annotations[n["id"]]["relation"] == step["relation"]
                    )
                    for n in applicable
                ):
                    raise ValueError(
                        f"Requested target/relation missing on the semantic producer for {step_id}.{obj}"
                    )
        for key in (
            "relation",
            "orientation_goal",
            "target_state",
            "layout",
            "axis",
            "direction",
            "terminal_behavior",
            "pour_goal_mode",
            "twist_goal_mode",
        ):
            val = step.get(key, "none")
            if val not in ("none", None, 0) and not any(
                (
                    annotations[n["id"]].get("relation")
                    if key == "relation"
                    else annotations[n["id"]]["intent"].get(key)
                )
                == val
                for n in calls
            ):
                raise ValueError(f"Missing explicit intent {step_id}.{key}={val}")
        for field, slot in (
            ("required_arm", "primary"),
            ("transfer_arm", "source"),
            ("receive_arm", "destination"),
        ):
            arm = step.get(field, "none")
            if arm not in {"none", "auto"} and not any(
                n["call"].get("resources", {}).get(slot) == arm.removesuffix("_arm")
                for n in calls
            ):
                raise ValueError(f"Missing required arm {step_id}.{field}")
        for parent in step["depends_on"]:
            parent_group = next(g for g in graph["task_groups"] if g["id"] == parent)
            if not all(
                any(p in ancestors(n["id"]) for p in parent_group["node_ids"])
                for n in calls
            ):
                raise ValueError("Source step ordering was lost")
    allowed_held = set()
    for step in steps.values():
        if step.get("terminal_behavior") == "hold":
            allowed_held.update(grounding["bindings"].get(f"{step['id']}.object", []))
    if set(held) - allowed_held:
        raise ValueError("Task ends holding objects that must be released")
    referenced = {uid for values in grounding["bindings"].values() for uid in values}
    referenced.update(
        a[key]
        for a in annotations.values()
        for key in ("object", "reference")
        if a[key] is not None
    )
    for node in graph["nodes"]:
        declaration = validate_semantic_call_payload(node["call"])
        args = node["call"].get("arguments", node["call"])
        for key, rule in {**declaration["required"], **declaration["optional"]}.items():
            value = args.get(key)
            if (
                isinstance(rule, str)
                and rule in {"entity", "part", "affordance", "target"}
                and isinstance(value, str)
                and value in entities
            ):
                referenced.add(value)
    observed = sum(
        o["provenance"] == "observed"
        for o in grounding["objects"]
        if o["id"] in referenced
    )
    total = len(referenced)
    # Every runtime binding is pending, even when the relevant object is visible.
    return {
        "status": "covered",
        "semantic_status": "valid",
        "visual_status": ("complete" if observed == total else "pending"),
        "pending_bindings": len(bindings),
        "visual_observed_entities": observed,
        "visual_total_entities": total,
        "execution_ready": False,
        "runtime_evidence_status": "not_run",
        "goals": [],
        "issues": [],
        "lineage": {},
    }


def _ancestor_sets(nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Compute closure once in validated topological order, with local ownership."""
    result: dict[str, set[str]] = {}
    for node in nodes:
        ancestors = set(node["depends_on"])
        for parent in node["depends_on"]:
            ancestors.update(result[parent])
        result[node["id"]] = ancestors
    return result


def _validate_reference_identity(
    bindings: dict[str, list[str]], references: list[dict[str, Any]]
) -> None:
    """A repeated exact scene selector denotes the same logical entity set."""
    selected: dict[tuple[str, str, int], set[str]] = {}
    for reference in references:
        key = (
            reference["reference"].strip().casefold(),
            reference["quantifier"],
            reference["count"],
        )
        ids = set(bindings[reference["reference_id"]])
        if key in selected and selected[key] != ids:
            raise ValueError(
                f"Repeated scene reference {reference['reference']!r} must retain identities {sorted(selected[key])}; do not split the same object when its role changes"
            )
        selected[key] = ids
