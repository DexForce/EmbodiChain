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

"""Normative coverage beside the unchanged v38 graph and bundle protocol."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any

from embodichain.gen_sim.task_spec import Predicate, TaskSpec, validate_task_spec

from .contracts import canonical_hash, validate_task_candidate
from .ontology import task_success_type
from .orchestration.contracts import validate_role_bindings
from .semantic_graph import validate_semantic_task_graph
from .task_spec import task_spec_from_candidate

__all__ = ["validate_graph_coverage"]


def _signature(predicate: Predicate) -> str:
    value = predicate.to_dict()
    value.pop("id", None)
    value.pop("observation", None)
    return canonical_hash(value)


def validate_graph_coverage(
    graph: Mapping[str, Any],
    task_spec: TaskSpec | Mapping[str, Any],
    candidate: Mapping[str, Any],
    role_bindings: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Check role/set, milestone order and surviving terminal-goal declarations.

    Args:
        graph: Existing v38 semantic graph, without additional graph fields.
        task_spec: Normative TaskSpec, not reconstructed from calls.
        candidate: Source solution draft.
        role_bindings: Exact Scene Adapter result.
        lineage: Source-step members emitted by the same planner expansion.

    Returns:
        Detached declaration coverage, not a physical-success certificate.
    """
    selected = validate_semantic_task_graph(graph)
    spec = TaskSpec.from_dict(task_spec)
    validate_task_spec(spec)
    witness = validate_task_candidate(candidate)
    bindings = validate_role_bindings(role_bindings)
    declared = task_spec_from_candidate(witness)
    members = json.loads(json.dumps(lineage, allow_nan=False))
    issues: list[dict[str, Any]] = []

    def issue(code: str, owner: str) -> None:
        issues.append({"code": code, "owner": owner})

    if (
        selected["task_id"] != witness["draft"]["task_id"]
        or bindings["task_id"] != selected["task_id"]
        or spec.task_id != selected["task_id"]
    ):
        issue("task_identity_mismatch", "task_id")
    if bindings["candidate_id"] != witness["candidate_id"]:
        issue("candidate_identity_mismatch", "candidate_id")
    if [r.to_dict() for r in spec.roles] != [r.to_dict() for r in declared.roles]:
        issue("role_declaration_not_covered", "roles")
    for name in ("init", "goal", "requirements"):
        signature = (
            _signature
            if name != "requirements"
            else lambda p: canonical_hash(p.to_dict())
        )
        if {signature(p) for p in getattr(declared, name)} - {
            signature(p) for p in getattr(spec, name)
        }:
            issue("missing_normative_declaration", name)
    observation_keys = {}
    for scope, predicates in [
        ("goal", spec.goal),
        *((f"milestone:{m.id}", m.achieve) for m in spec.milestones),
    ]:
        for index, predicate in enumerate(predicates):
            key = predicate.observation or predicate.id or f"{scope}.{index}"
            if key in observation_keys:
                issue("ambiguous_observation_key", key)
            observation_keys[key] = scope
    steps = {step["id"]: step for step in witness["draft"]["steps"]}
    groups = {group["id"]: group for group in selected["task_groups"]}
    nodes = {node["id"]: node for node in selected["nodes"]}
    if type(members) is not dict or set(members) != set(steps):
        raise ValueError("Expansion lineage must cover exactly every source step.")
    role_members: dict[str, list[dict[str, Any]]] = {}
    for role in declared.roles:
        selections = []
        for ref in declared.metadata["role_references"][role.name]:
            values = bindings["reference_bindings"].get(ref, [])
            selections.append(
                [
                    {
                        "uid": (
                            uid.rpartition("::")[0]
                            if role.kind == "articulation" and "::" in uid
                            else uid
                        ),
                        "part_id": (
                            uid.rpartition("::")[2]
                            if role.kind == "articulation" and "::" in uid
                            else bindings["role_bindings"].get(ref)
                        ),
                    }
                    for uid in values
                ]
            )
        values = selections[0] if selections else []
        if not values or any(
            {(v["uid"], v["part_id"]) for v in values}
            != {(v["uid"], v["part_id"]) for v in item}
            for item in selections
        ):
            issue("missing_or_inconsistent_role_binding", role.name)
        identities = [(v["uid"], v["part_id"]) for v in values]
        if len(set(identities)) != len(identities):
            issue("duplicate_role_member", role.name)
        if (role.quantifier == "one" and len(values) != 1) or (
            role.quantifier == "count" and len(values) != role.count
        ):
            issue("role_cardinality_mismatch", role.name)
        role_members[role.name] = values
    assigned: list[str] = []
    milestone_records = []
    for step_id, step in steps.items():
        entries = members[step_id]
        if type(entries) is not list or not entries:
            issue("missing_expanded_members", step_id)
            continue
        role = declared.metadata["step_roles"][step_id]["object"]
        expected = {(v["uid"], v["part_id"]) for v in role_members[role]}
        observed = set()
        for entry in entries:
            if type(entry) is not dict or set(entry) != {
                "group_id",
                "object_uid",
                "object_part_id",
            }:
                raise ValueError("Malformed expansion lineage member.")
            group_id = entry["group_id"]
            assigned.append(group_id)
            observed.add((entry["object_uid"], entry["object_part_id"]))
            group = groups.get(group_id)
            if group is None:
                issue("missing_task_group", group_id)
                continue
            if group["task_type"] != step["task_type"] or group["success"] != {
                "kind": "semantic_task_term",
                "step_id": group_id,
                "type": task_success_type(step["task_type"], step),
            }:
                issue("milestone_success_declaration_mismatch", group_id)
            calls = [nodes[node_id]["call"] for node_id in group["node_ids"]]
            subject_calls = [
                c
                for c in calls
                if c.get("object", c.get("arguments", {}).get("object"))
                == entry["object_uid"]
            ]
            if not subject_calls and step["task_type"] not in {"E6", "E9"}:
                issue("wrong_or_missing_subject_call", group_id)
            kinds = {c.get("call_id", c["kind"]) for c in subject_calls}
            required = {
                "E2": {"gen_sim.align_held", "simulation.axis_align"},
                "E3": {"simulation.pour"},
                "E4": {"hand_over"},
                "E6": {"gen_sim.articulation_slide"},
                "E8": {f"gen_sim.twist.{group_id}"},
                "E9": {f"gen_sim.press.{group_id}"},
                "E5": {
                    (
                        "simulation.coordinated_transport"
                        if step["terminal_behavior"] == "place"
                        else "simulation.coordinated_hold"
                    )
                },
            }.get(step["task_type"])
            if required is not None and not kinds.intersection(required):
                issue("missing_semantic_producer", group_id)
            if step["task_type"] == "E8":
                controls = [
                    c
                    for c in subject_calls
                    if c.get("call_id", "").startswith("gen_sim.twist")
                ]
                settings = [c["arguments"].get("setting") for c in controls]
                if step.get("twist_goal_mode") == "coarse":
                    valid_setting = bool(settings) and all(
                        type(s) is int and s == settings[0] for s in settings
                    )
                else:
                    valid_setting = any(
                        c.get("call_id") == f"gen_sim.twist.{group_id}"
                        and c["arguments"].get("setting") == step["target_setting"]
                        for c in subject_calls
                    )
                if not valid_setting:
                    issue("wrong_control_setting", group_id)
            if step["task_type"] == "E6" and not any(
                c.get("call_id") == "gen_sim.articulation_slide"
                and c["arguments"].get("part") == entry["object_part_id"]
                and c["arguments"].get("state") == step["target_state"]
                for c in subject_calls
            ):
                issue("wrong_articulation_part_or_state", group_id)
            if (
                step["task_type"] in {"E1", "E2"}
                or step["task_type"] in {"E4", "E5"}
                and step["terminal_behavior"] == "place"
            ):
                if not kinds.intersection(
                    {
                        "place",
                        "simulation.place_relative",
                        "gen_sim.place_upright",
                        "gen_sim.stack_place",
                        "simulation.coordinated_transport",
                    }
                ):
                    issue("missing_release_producer", group_id)
            for field, slot in (
                ("transfer_arm", "source"),
                ("receive_arm", "destination"),
                ("required_arm", "primary"),
            ):
                arm = step[field]
                resources = [
                    c["resources"][slot]
                    for c in subject_calls
                    if slot in c.get("resources", {})
                ]
                if arm not in {"none", "auto"} and (
                    not resources
                    or any(value != arm.removesuffix("_arm") for value in resources)
                ):
                    issue("required_resource_not_covered", f"{group_id}.{field}")
            target_role = declared.metadata["step_roles"][step_id].get("target")
            if target_role and len(role_members[target_role]) != 1:
                issue("set_reference_relation_unavailable", group_id)
            if target_role and step["task_type"] in {"E1", "E4"}:
                targets = {v["uid"] for v in role_members[target_role]}
                if not any(
                    (
                        c.get("arguments", {}).get("reference") in targets
                        and c.get("arguments", {}).get("relation") == step["relation"]
                        and any(
                            target["uid"] == c["arguments"]["reference"]
                            and target["part_id"]
                            == c["arguments"].get("reference_part")
                            for target in role_members[target_role]
                        )
                    )
                    or (
                        step["relation"] == "inside"
                        and c["kind"] == "place"
                        and c.get("inside")
                        in {
                            f"inside__{target['uid']}__{entry['object_uid']}"
                            + (f"__{target['part_id']}" if target["part_id"] else "")
                            for target in role_members[target_role]
                        }
                    )
                    or (
                        step["relation"] == "on"
                        and c["kind"] == "place"
                        and entry["object_part_id"] is None
                        and any(
                            target["part_id"] is None
                            and c.get("on")
                            == f"on__{target['uid']}__{entry['object_uid']}"
                            for target in role_members[target_role]
                        )
                    )
                    for c in subject_calls
                ):
                    issue("required_relation_not_covered", group_id)
            if target_role and step["task_type"] == "E3":
                targets = {v["uid"] for v in role_members[target_role]}
                if not any(
                    c.get("call_id") == "simulation.move_held_object"
                    and c.get("arguments", {}).get("reference") in targets
                    for c in subject_calls
                ):
                    issue("required_pour_target_not_covered", group_id)
        if observed != expected or len(entries) != len(expected):
            issue("incomplete_set_coverage", step_id)
    if set(assigned) != set(groups) or len(assigned) != len(set(assigned)):
        issue("unassigned_or_duplicate_group", "lineage")

    def ancestors(group_id: str) -> set[str]:
        result = set(groups[group_id]["depends_on"])
        for parent in tuple(result):
            result.update(ancestors(parent))
        return result

    source_milestones = {m.id: m for m in declared.milestones}
    if set(source_milestones) - {m.id for m in spec.milestones}:
        issue("missing_normative_declaration", "milestones")
    for milestone in spec.milestones:
        source = source_milestones.get(milestone.id)
        if source is None or {_signature(p) for p in milestone.achieve} != {
            _signature(p) for p in source.achieve
        }:
            issue("milestone_not_covered", milestone.id)
            continue
        if milestone.after != source.after:
            issue("milestone_order_declaration_mismatch", milestone.id)
        source_step = declared.metadata["milestone_sources"][milestone.id]
        entries = members[source_step]
        milestone_records.append(
            {
                "id": milestone.id,
                "source_step": source_step,
                "groups": [v["group_id"] for v in entries],
                "members": deepcopy(entries),
            }
        )
        for parent in milestone.after:
            parent_step = declared.metadata["milestone_sources"].get(parent)
            if parent_step == source_step:
                continue
            parent_groups = {v["group_id"] for v in members.get(parent_step, [])}
            for child in entries:
                if child["group_id"] in groups and not parent_groups.issubset(
                    ancestors(child["group_id"])
                ):
                    issue("milestone_order_not_covered", milestone.id)
    goal_lookup = {
        _signature(p): declared.metadata["goal_sources"][p.id] for p in declared.goal
    }
    goal_records = []
    aliased_states = {}
    for predicate in spec.goal:
        source_steps = goal_lookup.get(_signature(predicate), [])
        if not source_steps:
            issue("terminal_goal_not_covered", predicate.id)
        if predicate.name in {
            "articulation_joint_near",
            "semantic_goal",
            "object_upright",
        }:
            for member in role_members.get(predicate.arguments.get("subject"), []):
                identity = (member["uid"], member["part_id"], predicate.name)
                state = (
                    predicate.arguments.get("target_state")
                    if predicate.name == "articulation_joint_near"
                    else (
                        canonical_hash(
                            {
                                "relation": predicate.arguments.get("relation"),
                                "reference": role_members.get(
                                    predicate.arguments.get("reference"), []
                                ),
                            }
                        )
                        if predicate.name == "semantic_goal"
                        else True
                    )
                )
                if identity in aliased_states and aliased_states[identity] != state:
                    issue("conflicting_aliased_terminal_roles", predicate.id)
                aliased_states[identity] = state
        goal_records.append(
            {
                "id": predicate.id,
                "predicate": predicate.to_dict(),
                "source_steps": source_steps,
                "groups": [v["group_id"] for s in source_steps for v in members[s]],
                "status": "declared" if source_steps else "uncovered",
            }
        )
    for predicate in spec.init:
        if _signature(predicate) not in {_signature(p) for p in declared.init}:
            issue("initial_condition_not_covered", predicate.id)
    for predicate in spec.invariants:
        issue("invariant_monitor_unavailable", predicate.id)
    for condition in spec.temporal:
        issue("temporal_monitor_unavailable", condition.id)
    known_requirements = {canonical_hash(r.to_dict()) for r in declared.requirements}
    for requirement in spec.requirements:
        if canonical_hash(requirement.to_dict()) not in known_requirements:
            issue(
                "consumer_requirement_not_covered",
                f"{requirement.consumer}.{requirement.key}",
            )
    return {
        "status": "covered" if not issues else "uncovered",
        "issues": issues,
        "roles": role_members,
        "goals": goal_records,
        "milestones": milestone_records,
        "lineage": members,
        "runtime_evidence_status": "not_run",
        "instruction_coverage": "validated_draft_only",
    }
