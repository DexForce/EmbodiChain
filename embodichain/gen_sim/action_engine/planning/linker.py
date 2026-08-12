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

"""Deterministic causal and resource linking for SeedGraph v3."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    validate_seed_graph,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.protocol import SEED_GRAPH_SCHEMA

__all__ = [
    "CONTRACT_LINKER_VERSION",
    "link_seed_graph",
    "link_task_dependencies",
    "validate_persisted_contracts",
]

CONTRACT_LINKER_VERSION = "action_contract_linker_v1"
_INITIAL_PREDICATES = frozenset({"arm_free", "object_free"})
_REFERENCE_KEYS = frozenset(
    {
        "anchor",
        "container",
        "reference",
        "reference_object",
        "support",
        "support_object",
        "target",
        "target_object",
    }
)


def link_task_dependencies(
    task_spec: Mapping[str, Any],
    role_bindings: Mapping[str, str],
    *,
    registry: AtomicCapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Add the minimal stable TaskGroup dependencies implied by contracts."""
    del registry  # Reserved for task-level capability specialization.
    task = validate_task_spec(task_spec)
    bindings = {str(key): str(value) for key, value in role_bindings.items()}
    bindings_hash = hashlib.sha256(
        json.dumps(
            bindings, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    existing_metadata = task.get("metadata", {})
    existing_linker = (
        existing_metadata.get("action_contract_task_linker", {})
        if isinstance(existing_metadata, Mapping)
        else {}
    )
    if (
        isinstance(existing_linker, Mapping)
        and existing_linker.get("version") == CONTRACT_LINKER_VERSION
        and existing_linker.get("role_bindings_hash") == bindings_hash
    ):
        return task
    instances = task["task_instances"]
    order = [str(item["id"]) for item in instances]
    dependencies = {
        str(item["id"]): set(str(value) for value in item["depends_on"])
        for item in instances
    }
    dependency_order = {
        str(item["id"]): [str(value) for value in item["depends_on"]]
        for item in instances
    }
    claims = {str(item["id"]): _task_claims(item, bindings) for item in instances}
    distinct_arm_pairs = _distinct_arm_pairs(task.get("metadata", {}))
    linked: list[dict[str, str]] = []

    latest_by_object: dict[str, tuple[str, str]] = {}
    for instance in instances:
        instance_id = str(instance["id"])
        task_type = str(instance["task_type"])
        primary = _task_primary_object(instance, bindings)
        previous = latest_by_object.get(primary)
        if (
            task_type == "E4"
            and previous is not None
            and previous[1] == "E2"
            and previous[0] not in dependencies[instance_id]
            and not _reaches(dependencies, previous[0], instance_id)
        ):
            dependencies[instance_id].add(previous[0])
            dependency_order[instance_id].append(previous[0])
            linked.append(
                {
                    "from": previous[0],
                    "to": instance_id,
                    "reason": "causal",
                    "detail": f"object_held:{primary}",
                }
            )
            _assert_acyclic(dependencies, "TaskSpec causal linking")
        latest_by_object[primary] = (instance_id, task_type)

    for later_index, later_id in enumerate(order):
        for earlier_id in order[:later_index]:
            if _reaches(dependencies, later_id, earlier_id) or _reaches(
                dependencies, earlier_id, later_id
            ):
                continue
            conflicts = _claim_conflicts(claims[earlier_id], claims[later_id])
            if frozenset({earlier_id, later_id}) in distinct_arm_pairs:
                conflicts = [item for item in conflicts if item != "arm:auto"]
            if not conflicts:
                continue
            dependencies[later_id].add(earlier_id)
            dependency_order[later_id].append(earlier_id)
            linked.append(
                {
                    "from": earlier_id,
                    "to": later_id,
                    "reason": "resource",
                    "detail": ",".join(conflicts),
                }
            )
            _assert_acyclic(dependencies, "TaskSpec contract linking")

    for instance in instances:
        instance_id = str(instance["id"])
        instance["depends_on"] = dependency_order[instance_id]
    metadata = dict(task.get("metadata", {}))
    metadata["action_contract_task_linker"] = {
        "version": CONTRACT_LINKER_VERSION,
        "role_bindings_hash": bindings_hash,
        "linked_dependencies": linked,
    }
    task["metadata"] = metadata
    return validate_task_spec(task)


def link_seed_graph(
    draft: Mapping[str, Any],
    *,
    registry: AtomicCapabilityRegistry | None = None,
    task_order: Sequence[str] = (),
    completed_nodes: Collection[str] = (),
    known_objects: Collection[str] | None = None,
) -> dict[str, Any]:
    """Resolve contracts, link a draft graph, and return validated SeedGraph v3."""
    if not isinstance(draft, Mapping):
        raise TypeError("SeedGraph draft must be a mapping.")
    if draft.get("schema_version") != SEED_GRAPH_SCHEMA:
        raise ValueError(f"Contract linker accepts only {SEED_GRAPH_SCHEMA!r} drafts.")
    capabilities = registry or build_atomic_capability_registry()
    graph = deepcopy(dict(draft))
    nodes = graph.get("nodes")
    groups = graph.get("task_groups")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("SeedGraph draft nodes must be a non-empty list.")
    if not isinstance(groups, list) or not groups:
        raise ValueError("SeedGraph draft task_groups must be a non-empty list.")

    already_linked = _already_linked(graph)
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise TypeError(f"SeedGraph draft node {index} must be a mapping.")
        action = str(node.get("atomic_action", ""))
        expected = capabilities.get(action).resolve_contract(node).as_mapping()
        persisted = node.get("contract")
        if persisted is not None and persisted != expected:
            raise ValueError(
                f"SeedGraph node {node.get('id')!r} persisted Action Contract "
                "does not match the current capability resolver."
            )
        node["contract"] = expected
        node.pop("resources", None)
    if already_linked:
        linked = validate_seed_graph(
            graph,
            known_objects=known_objects,
            known_actions=capabilities.names(),
        )
        validate_persisted_contracts(linked, capabilities)
        return linked

    completed = {str(item) for item in completed_nodes}
    node_by_id = _unique_by_id(nodes, "SeedGraph draft nodes")
    group_by_id = _unique_by_id(groups, "SeedGraph draft task_groups")
    ordered_groups = _ordered_group_ids(groups, task_order)
    node_reasons: list[dict[str, str]] = []
    group_reasons: list[dict[str, str]] = []

    for group in groups:
        group_id = str(group.get("id", ""))
        node_ids = [str(item) for item in group.get("node_ids", ())]
        if not node_ids or any(node_id not in node_by_id for node_id in node_ids):
            raise ValueError(
                f"SeedGraph TaskGroup {group_id!r} has missing or unknown node IDs."
            )
        _link_internal_nodes(
            node_ids,
            node_by_id,
            completed=completed,
            reasons=node_reasons,
        )
        _validate_internal_symbolic_state(node_ids, node_by_id)
        group.pop("contract", None)

    group_dependencies = {
        group_id: set(str(item) for item in group_by_id[group_id].get("depends_on", ()))
        for group_id in ordered_groups
    }
    original_group_dependencies = {
        group_id: [str(item) for item in group_by_id[group_id].get("depends_on", ())]
        for group_id in ordered_groups
    }
    _assert_acyclic(group_dependencies, "SeedGraph TaskGroups")
    summaries = {
        group_id: _summarize_group(group_by_id[group_id], node_by_id)
        for group_id in ordered_groups
    }
    distinct_arm_pairs = _distinct_arm_pairs(graph.get("metadata", {}))

    for later_index, later_id in enumerate(ordered_groups):
        for earlier_id in ordered_groups[:later_index]:
            if _reaches(group_dependencies, later_id, earlier_id) or _reaches(
                group_dependencies, earlier_id, later_id
            ):
                continue
            conflicts = _claim_conflicts(
                summaries[earlier_id]["claims"], summaries[later_id]["claims"]
            )
            if frozenset({earlier_id, later_id}) in distinct_arm_pairs:
                conflicts = [item for item in conflicts if item != "arm:auto"]
            if conflicts:
                _add_group_dependency(
                    earlier_id,
                    later_id,
                    group_dependencies,
                    completed,
                    summaries,
                    group_reasons,
                    reason="resource",
                    detail=",".join(conflicts),
                )

    for later_index, group_id in enumerate(ordered_groups):
        for requirement in summaries[group_id]["entry_requires"]:
            if requirement["predicate"] in _INITIAL_PREDICATES:
                continue
            candidates = [
                candidate
                for candidate in ordered_groups[:later_index]
                if _adds_atom(summaries[candidate]["exit_effects"], requirement)
            ]
            if not candidates:
                raise ValueError(
                    f"SeedGraph TaskGroup {group_id!r} has no producer for state "
                    f"{requirement}."
                )
            maximal = [
                candidate
                for candidate in candidates
                if not any(
                    candidate != other
                    and _reaches(group_dependencies, other, candidate)
                    for other in candidates
                )
            ]
            if len(maximal) != 1:
                raise ValueError(
                    f"SeedGraph TaskGroup {group_id!r} has multiple unordered "
                    f"producers for state {requirement}: {maximal}."
                )
            producer = maximal[0]
            if not _reaches(group_dependencies, group_id, producer):
                _add_group_dependency(
                    producer,
                    group_id,
                    group_dependencies,
                    completed,
                    summaries,
                    group_reasons,
                    reason="causal",
                    detail=_atom_key(requirement),
                )

    _assert_acyclic(group_dependencies, "SeedGraph contract linking")
    for group_id in ordered_groups:
        group = group_by_id[group_id]
        group["depends_on"] = original_group_dependencies[group_id] + [
            candidate
            for candidate in ordered_groups
            if candidate in group_dependencies[group_id]
            and candidate not in original_group_dependencies[group_id]
        ]

    _link_group_boundaries(
        ordered_groups,
        group_dependencies,
        summaries,
        node_by_id,
        completed,
        node_reasons,
    )
    for group_id in ordered_groups:
        summaries[group_id] = _summarize_group(group_by_id[group_id], node_by_id)
        group_by_id[group_id]["contract"] = summaries[group_id]

    _validate_symbolic_state(ordered_groups, group_dependencies, summaries, group_by_id)
    metadata = dict(graph.get("metadata", {}))
    metadata["action_contract_linker"] = {
        "version": CONTRACT_LINKER_VERSION,
        "group_dependencies": _sorted_reasons(group_reasons),
        "node_dependencies": _sorted_reasons(node_reasons),
    }
    graph["metadata"] = metadata
    graph["schema_version"] = SEED_GRAPH_SCHEMA
    graph["nodes"] = nodes
    graph["task_groups"] = groups
    return validate_seed_graph(
        graph,
        known_objects=known_objects,
        known_actions=capabilities.names(),
    )


def validate_persisted_contracts(
    graph: Mapping[str, Any], registry: AtomicCapabilityRegistry
) -> None:
    """Reject persisted contracts that differ from the active capability catalog."""
    metadata = graph.get("metadata", {})
    linker = (
        metadata.get("action_contract_linker", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if (
        not isinstance(linker, Mapping)
        or linker.get("version") != CONTRACT_LINKER_VERSION
    ):
        raise ValueError(
            "SeedGraph was not produced by the current deterministic Contract Linker; "
            "regenerate the configuration bundle."
        )
    for node in graph.get("nodes", ()):
        expected = (
            registry.get(str(node["atomic_action"])).resolve_contract(node).as_mapping()
        )
        if node.get("contract") != expected:
            raise ValueError(
                f"SeedGraph node {node.get('id')!r} persisted Action Contract "
                "does not match the current capability resolver."
            )
    node_by_id = {
        str(node["id"]): node
        for node in graph.get("nodes", ())
        if isinstance(node, Mapping) and "id" in node
    }
    for group in graph.get("task_groups", ()):
        expected = _summarize_group(group, node_by_id)
        if group.get("contract") != expected:
            raise ValueError(
                f"SeedGraph TaskGroup {group.get('id')!r} persisted contract "
                "does not match its linked AtomicAction topology."
            )


def _task_claims(
    instance: Mapping[str, Any], bindings: Mapping[str, str]
) -> list[dict[str, str]]:
    task_type = str(instance["task_type"])
    params = _resolve_roles(instance.get("params", {}), bindings)
    primary_key = "source_role" if task_type == "E3" else "object_role"
    primary = params.get(primary_key)
    claims: list[dict[str, str]] = []
    if isinstance(primary, str) and primary:
        claims.append(_claim(f"object:{primary}", "exclusive"))
    target = params.get("target_role")
    if isinstance(target, str) and target and target != primary:
        claims.append(_claim(f"object:{target}", "shared_read"))
    payloads = params.get("payload_roles", [])
    if isinstance(payloads, Sequence) and not isinstance(
        payloads, (str, bytes, bytearray)
    ):
        for payload in payloads:
            if isinstance(payload, str) and payload and payload != primary:
                claims.append(_claim(f"object:{payload}", "exclusive"))
    if task_type == "E4":
        transfer = str(params.get("transfer_arm", ""))
        receive = str(params.get("receive_arm", ""))
        if transfer not in {"left_arm", "right_arm"} or receive not in {
            "left_arm",
            "right_arm",
        }:
            raise ValueError(
                "E4 contract linking requires explicit transfer/receive arms."
            )
        if transfer == receive:
            raise ValueError("E4 transfer_arm and receive_arm must be distinct.")
        claims.extend((_claim(f"arm:{transfer}"), _claim(f"arm:{receive}")))
    elif task_type == "E5":
        claims.extend((_claim("arm:left_arm"), _claim("arm:right_arm")))
    else:
        required_arm = params.get("required_arm")
        if required_arm in {"left_arm", "right_arm"}:
            claims.append(_claim(f"arm:{required_arm}"))
        else:
            claims.append(_claim("arm:auto"))
    return _merge_claims(claims)


def _task_primary_object(
    instance: Mapping[str, Any], bindings: Mapping[str, str]
) -> str:
    task_type = str(instance["task_type"])
    params = _resolve_roles(instance.get("params", {}), bindings)
    key = "source_role" if task_type == "E3" else "object_role"
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"TaskGroup {instance.get('id')!r} requires a resolved {key!r}."
        )
    return value


def _link_internal_nodes(
    node_ids: Sequence[str],
    node_by_id: Mapping[str, dict[str, Any]],
    *,
    completed: set[str],
    reasons: list[dict[str, str]],
) -> None:
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for later_index, later_id in enumerate(node_ids):
        later = node_by_id[later_id]
        for requirement in later["contract"]["requires"]:
            producers = [
                earlier_id
                for earlier_id in node_ids[:later_index]
                if _adds_atom(
                    node_by_id[earlier_id]["contract"]["effects"], requirement
                )
            ]
            if producers:
                _add_node_dependency(
                    producers[-1],
                    later_id,
                    node_by_id,
                    completed,
                    reasons,
                    "causal",
                    _atom_key(requirement),
                )
        for earlier_id in node_ids[:later_index]:
            earlier = node_by_id[earlier_id]
            if earlier.get("sync_group") is not None and earlier.get(
                "sync_group"
            ) == later.get("sync_group"):
                continue
            if _node_reaches(node_by_id, later_id, earlier_id) or _node_reaches(
                node_by_id, earlier_id, later_id
            ):
                continue
            conflicts = _claim_conflicts(
                earlier["contract"]["claims"], later["contract"]["claims"]
            )
            if conflicts:
                _add_node_dependency(
                    earlier_id,
                    later_id,
                    node_by_id,
                    completed,
                    reasons,
                    "resource",
                    ",".join(conflicts),
                )
    dependencies = {
        node_id: {
            str(parent)
            for parent in node_by_id[node_id].get("depends_on", ())
            if str(parent) in positions
        }
        for node_id in node_ids
    }
    _assert_acyclic(dependencies, "AtomicAction contract linking")


def _summarize_group(
    group: Mapping[str, Any], node_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    node_ids = [str(item) for item in group["node_ids"]]
    node_set = set(node_ids)
    entries = [
        node_id
        for node_id in node_ids
        if not any(
            str(parent) in node_set for parent in node_by_id[node_id]["depends_on"]
        )
    ]
    depended = {
        str(parent)
        for node_id in node_ids
        for parent in node_by_id[node_id]["depends_on"]
        if str(parent) in node_set
    }
    terminals = [node_id for node_id in node_ids if node_id not in depended]
    entry_requires: list[dict[str, str]] = []
    for node_id in node_ids:
        node = node_by_id[node_id]
        for requirement in node["contract"]["requires"]:
            if any(
                producer_id in node_set
                and _node_reaches(node_by_id, node_id, producer_id)
                and _adds_atom(
                    node_by_id[producer_id]["contract"]["effects"], requirement
                )
                for producer_id in node_ids
            ):
                continue
            if requirement not in entry_requires:
                entry_requires.append(deepcopy(requirement))

    last_effect: dict[str, dict[str, Any]] = {}
    effect_order: list[str] = []
    for node_id in node_ids:
        for effect in node_by_id[node_id]["contract"]["effects"]:
            key = _atom_key(effect["atom"])
            if key not in last_effect:
                effect_order.append(key)
            last_effect[key] = deepcopy(effect)
    claims = [
        deepcopy(claim)
        for node_id in node_ids
        for claim in node_by_id[node_id]["contract"]["claims"]
    ]
    claims.extend(_goal_read_claims(group.get("goal", {})))
    merged_claims = _merge_claims(claims)
    free_resources = {
        (
            f"arm:{effect['atom']['arm']}"
            if effect["atom"]["predicate"] == "arm_free"
            else f"object:{effect['atom']['object_uid']}"
        )
        for effect in last_effect.values()
        if effect["op"] == "add"
        and effect["atom"]["predicate"] in {"arm_free", "object_free"}
    }
    for claim in merged_claims:
        if claim["resource"] in free_resources:
            claim["lifetime"] = "action"
    completion = (
        "terminal_barrier"
        if terminals
        and all(
            node_by_id[node_id]["contract"]["completion"] == "terminal_barrier"
            for node_id in terminals
        )
        else "ordinary"
    )
    return {
        "entry_requires": entry_requires,
        "exit_effects": [last_effect[key] for key in effect_order],
        "claims": merged_claims,
        "entry_node_ids": entries,
        "terminal_node_ids": terminals,
        "completion": completion,
    }


def _validate_internal_symbolic_state(
    node_ids: Sequence[str], node_by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    node_set = set(node_ids)
    dependencies = {
        node_id: {
            str(parent)
            for parent in node_by_id[node_id].get("depends_on", ())
            if str(parent) in node_set
        }
        for node_id in node_ids
    }
    entry_atoms = set()
    for node_id in node_ids:
        for requirement in node_by_id[node_id]["contract"]["requires"]:
            has_prior_producer = any(
                producer_id != node_id
                and _node_reaches(node_by_id, node_id, producer_id)
                and _adds_atom(
                    node_by_id[producer_id]["contract"]["effects"], requirement
                )
                for producer_id in node_ids
            )
            if not has_prior_producer:
                entry_atoms.add(_atom_key(requirement))
    state = set(entry_atoms)
    for node_id in _stable_topological(node_ids, dependencies):
        contract = node_by_id[node_id]["contract"]
        for requirement in contract["requires"]:
            if _atom_key(requirement) not in state:
                raise ValueError(
                    f"SeedGraph node {node_id!r} requires unavailable state "
                    f"{requirement}."
                )
        for effect in contract["effects"]:
            key = _atom_key(effect["atom"])
            if effect["op"] == "delete":
                if key not in state:
                    raise ValueError(
                        f"SeedGraph node {node_id!r} deletes unavailable state "
                        f"{effect['atom']}."
                    )
                state.remove(key)
            else:
                state.add(key)


def _add_group_dependency(
    parent: str,
    child: str,
    dependencies: dict[str, set[str]],
    completed: set[str],
    summaries: Mapping[str, Mapping[str, Any]],
    reasons: list[dict[str, str]],
    *,
    reason: str,
    detail: str,
) -> None:
    if any(node_id in completed for node_id in summaries[child]["entry_node_ids"]):
        raise ValueError(
            f"Contract linking cannot add dependency into completed TaskGroup {child!r}."
        )
    dependencies[child].add(parent)
    _assert_acyclic(dependencies, "SeedGraph contract linking")
    reasons.append({"from": parent, "to": child, "reason": reason, "detail": detail})


def _link_group_boundaries(
    ordered_groups: Sequence[str],
    dependencies: Mapping[str, set[str]],
    summaries: Mapping[str, Mapping[str, Any]],
    node_by_id: Mapping[str, dict[str, Any]],
    completed: set[str],
    reasons: list[dict[str, str]],
) -> None:
    for child in ordered_groups:
        for parent in ordered_groups:
            if parent not in dependencies[child]:
                continue
            for child_node in summaries[child]["entry_node_ids"]:
                for parent_node in summaries[parent]["terminal_node_ids"]:
                    _add_node_dependency(
                        parent_node,
                        child_node,
                        node_by_id,
                        completed,
                        reasons,
                        "cleanup",
                        f"TaskGroup {parent} terminal barrier",
                    )


def _add_node_dependency(
    parent: str,
    child: str,
    node_by_id: Mapping[str, dict[str, Any]],
    completed: set[str],
    reasons: list[dict[str, str]],
    reason: str,
    detail: str,
) -> None:
    if parent == child:
        raise ValueError(f"Contract linker cannot add self-dependency {child!r}.")
    dependencies = node_by_id[child].setdefault("depends_on", [])
    if parent in dependencies or _node_reaches(node_by_id, child, parent):
        return
    if _node_reaches(node_by_id, parent, child):
        raise ValueError(
            f"Contract dependency {parent!r} -> {child!r} would create a cycle."
        )
    if child in completed:
        raise ValueError(f"Contract linker cannot modify completed node {child!r}.")
    dependencies.append(parent)
    reasons.append({"from": parent, "to": child, "reason": reason, "detail": detail})


def _validate_symbolic_state(
    ordered_groups: Sequence[str],
    dependencies: Mapping[str, set[str]],
    summaries: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Any]],
) -> None:
    atoms = [
        atom
        for summary in summaries.values()
        for atom in [
            *summary["entry_requires"],
            *(effect["atom"] for effect in summary["exit_effects"]),
        ]
    ]
    state = {
        _atom_key({"predicate": "arm_free", "arm": str(atom["arm"])})
        for atom in atoms
        if "arm" in atom
    }
    state.update(
        _atom_key({"predicate": "object_free", "object_uid": str(atom["object_uid"])})
        for atom in atoms
        if "object_uid" in atom
    )
    for group_id in _stable_topological(ordered_groups, dependencies):
        if groups[group_id].get("role") == "recovery":
            for requirement in summaries[group_id]["entry_requires"]:
                if requirement["predicate"] == "object_free":
                    object_uid = str(requirement["object_uid"])
                    state = {
                        item
                        for item in state
                        if not (
                            item.startswith("object_held|")
                            or item.startswith("object_coordinated_held|")
                        )
                        or f"|{object_uid}|" not in f"|{item}|"
                    }
                state.add(_atom_key(requirement))
        for requirement in summaries[group_id]["entry_requires"]:
            key = _atom_key(requirement)
            if key not in state:
                raise ValueError(
                    f"SeedGraph TaskGroup {group_id!r} requires unavailable state "
                    f"{requirement}."
                )
        for effect in summaries[group_id]["exit_effects"]:
            key = _atom_key(effect["atom"])
            if effect["op"] == "add":
                state.add(key)
            else:
                state.discard(key)


def _goal_read_claims(value: Any) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _REFERENCE_KEYS and isinstance(child, str):
                if child not in {"table_center", "world"}:
                    claims.append(_claim(f"object:{child}", "shared_read"))
            claims.extend(_goal_read_claims(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            claims.extend(_goal_read_claims(child))
    return claims


def _claim(
    resource: str, access: str = "exclusive", lifetime: str = "action"
) -> dict[str, str]:
    return {"resource": resource, "access": access, "lifetime": lifetime}


def _merge_claims(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for claim in claims:
        resource = str(claim["resource"])
        if resource not in merged:
            order.append(resource)
            merged[resource] = _claim(
                resource,
                str(claim.get("access", "exclusive")),
                str(claim.get("lifetime", "action")),
            )
            continue
        current = merged[resource]
        if claim.get("access") == "exclusive":
            current["access"] = "exclusive"
        if claim.get("lifetime") == "until_release":
            current["lifetime"] = "until_release"
    return [merged[resource] for resource in order]


def _claim_conflicts(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> list[str]:
    first_by_resource = {str(item["resource"]): str(item["access"]) for item in first}
    second_by_resource = {str(item["resource"]): str(item["access"]) for item in second}
    conflicts = {
        resource
        for resource in set(first_by_resource) & set(second_by_resource)
        if "exclusive" in {first_by_resource[resource], second_by_resource[resource]}
    }
    first_arms = {item for item in first_by_resource if item.startswith("arm:")}
    second_arms = {item for item in second_by_resource if item.startswith("arm:")}
    if "arm:auto" in first_arms and second_arms:
        conflicts.add("arm:auto")
    if "arm:auto" in second_arms and first_arms:
        conflicts.add("arm:auto")
    return sorted(conflicts)


def _distinct_arm_pairs(value: Any) -> set[frozenset[str]]:
    if not isinstance(value, Mapping):
        return set()
    groups = value.get("legacy_allocation_groups", value.get("allocation_groups", ()))
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes, bytearray)):
        return set()
    result: set[frozenset[str]] = set()
    for group in groups:
        if (
            not isinstance(group, Mapping)
            or group.get("arm_constraint") != "distinct_arms"
        ):
            continue
        members = group.get("semantic_step_ids", group.get("task_instance_ids", ()))
        if not isinstance(members, Sequence) or isinstance(
            members, (str, bytes, bytearray)
        ):
            continue
        member_ids = [str(item) for item in members]
        for index, first in enumerate(member_ids):
            for second in member_ids[index + 1 :]:
                result.add(frozenset({first, second}))
    return result


def _resolve_roles(value: Any, bindings: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_roles(child, bindings) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_roles(child, bindings) for child in value]
    if isinstance(value, tuple):
        return tuple(_resolve_roles(child, bindings) for child in value)
    if isinstance(value, str):
        return bindings.get(value, value)
    return value


def _adds_atom(effects: Sequence[Mapping[str, Any]], atom: Mapping[str, Any]) -> bool:
    return any(
        effect.get("op") == "add" and effect.get("atom") == atom for effect in effects
    )


def _atom_key(atom: Mapping[str, Any]) -> str:
    return "|".join(
        str(atom.get(key, "")) for key in ("predicate", "object_uid", "arm")
    )


def _unique_by_id(items: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id:
            raise ValueError(f"{context} require non-empty IDs.")
        if item_id in result:
            raise ValueError(f"{context} contain duplicate ID {item_id!r}.")
        result[item_id] = item
    return result


def _ordered_group_ids(
    groups: Sequence[Mapping[str, Any]], task_order: Sequence[str]
) -> list[str]:
    available = [str(group["id"]) for group in groups]
    requested = [str(item) for item in task_order]
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(
            f"task_order references unknown TaskGroups: {sorted(unknown)}."
        )
    return requested + [item for item in available if item not in set(requested)]


def _reaches(dependencies: Mapping[str, set[str]], child: str, parent: str) -> bool:
    pending = list(dependencies.get(child, ()))
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == parent:
            return True
        if current not in visited:
            visited.add(current)
            pending.extend(dependencies.get(current, ()))
    return False


def _node_reaches(
    node_by_id: Mapping[str, Mapping[str, Any]], child: str, parent: str
) -> bool:
    pending = [str(item) for item in node_by_id[child].get("depends_on", ())]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == parent:
            return True
        if current not in visited and current in node_by_id:
            visited.add(current)
            pending.extend(
                str(item) for item in node_by_id[current].get("depends_on", ())
            )
    return False


def _assert_acyclic(dependencies: Mapping[str, set[str]], context: str) -> None:
    for item_id in dependencies:
        if _reaches(dependencies, item_id, item_id):
            raise ValueError(f"{context} produced a dependency cycle at {item_id!r}.")


def _stable_topological(
    order: Sequence[str], dependencies: Mapping[str, set[str]]
) -> list[str]:
    remaining = set(order)
    result: list[str] = []
    while remaining:
        ready = [
            item
            for item in order
            if item in remaining and not (dependencies[item] & remaining)
        ]
        if not ready:
            raise ValueError("SeedGraph TaskGroups contain a dependency cycle.")
        result.extend(ready)
        remaining.difference_update(ready)
    return result


def _already_linked(graph: Mapping[str, Any]) -> bool:
    metadata = graph.get("metadata", {})
    linker = (
        metadata.get("action_contract_linker", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    return (
        isinstance(linker, Mapping)
        and linker.get("version") == CONTRACT_LINKER_VERSION
        and all("contract" in node for node in graph.get("nodes", ()))
        and all("contract" in group for group in graph.get("task_groups", ()))
    )


def _sorted_reasons(reasons: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    unique = {
        (item["from"], item["to"], item["reason"], item["detail"]) for item in reasons
    }
    return [
        {"from": source, "to": target, "reason": reason, "detail": detail}
        for source, target, reason, detail in sorted(unique)
    ]
