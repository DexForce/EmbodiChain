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

"""Bounded retry decisions and auditable RuntimeGraph revisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
    capability_precondition,
)
from embodichain.gen_sim.action_engine.domain import motion_policy, validate_seed_graph

__all__ = [
    "FAILURE_TYPES",
    "GraphRevision",
    "RetryDecision",
    "RuntimeGraph",
    "build_upright_recovery",
    "classify_failure",
]

FAILURE_TYPES = frozenset(
    {
        "plan_failed",
        "search_exhausted",
        "grasp_missed",
        "object_fallen",
        "object_dropped",
        "postcondition_failed",
    }
)


@dataclass(frozen=True)
class RetryDecision:
    """Per-environment result of one failed full-AtomicAction attempt."""

    retry: torch.Tensor
    recover: torch.Tensor
    exhausted: torch.Tensor
    attempts: tuple[int, ...]


@dataclass(frozen=True)
class GraphRevision:
    """One immutable patch record over the original SeedGraph."""

    revision: int
    kind: str
    reason: str
    failed_node_id: str | None
    inserted_group_ids: tuple[str, ...]
    replaced_group_ids: tuple[str, ...]
    active_env_ids: tuple[int, ...] = ()


class RuntimeGraph:
    """Keep SeedGraph immutable while applying bounded, validated revisions."""

    def __init__(
        self,
        seed_graph: Mapping[str, Any],
        *,
        num_envs: int,
        max_retries: int = 2,
        max_revisions: int = 8,
        max_recovery_actions: int = 12,
        registry: AtomicCapabilityRegistry | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("RuntimeGraph num_envs must be positive.")
        self.registry = registry or build_atomic_capability_registry()
        self.seed_graph = validate_seed_graph(
            seed_graph,
            known_actions=self.registry.names(),
        )
        from embodichain.gen_sim.action_engine.planning.linker import (
            validate_persisted_contracts,
        )

        validate_persisted_contracts(self.seed_graph, self.registry)
        self._graph = deepcopy(self.seed_graph)
        self.num_envs = int(num_envs)
        self.max_retries = int(max_retries)
        self.max_revisions = int(max_revisions)
        self.max_recovery_actions = int(max_recovery_actions)
        if min(self.max_retries, self.max_revisions, self.max_recovery_actions) < 0:
            raise ValueError("RuntimeGraph budgets must be non-negative.")
        self._attempts: dict[str, list[int]] = {}
        self._recovery_action_count = 0
        self.revisions: list[GraphRevision] = []

    @property
    def graph(self) -> dict[str, Any]:
        """Return the current detached RuntimeGraph snapshot."""
        return deepcopy(self._graph)

    def record_failure(
        self,
        node_id: str,
        failed: torch.Tensor,
        *,
        precondition_holds: torch.Tensor,
    ) -> RetryDecision:
        """Consume attempt budgets and distinguish retry from recovery."""
        failed = _mask(failed, self.num_envs)
        precondition_holds = _mask(precondition_holds, self.num_envs)
        attempts = self._attempts.setdefault(node_id, [1] * self.num_envs)
        retry = torch.zeros_like(failed)
        recover = torch.zeros_like(failed)
        exhausted = torch.zeros_like(failed)
        node = _node(self._graph, node_id)
        capability = self.registry.get(str(node["atomic_action"]))
        for env_id in torch.nonzero(failed, as_tuple=False).flatten().tolist():
            attempts[env_id] += 1
            can_retry = (
                capability.retry_mode != "non_retryable"
                and bool(precondition_holds[env_id])
                and attempts[env_id] <= self.max_retries + 1
            )
            if can_retry:
                retry[env_id] = True
            elif capability.retry_mode != "non_retryable":
                recover[env_id] = True
            else:
                exhausted[env_id] = True
        return RetryDecision(retry, recover, exhausted, tuple(attempts))

    def insert_recovery_subgraph(
        self,
        *,
        failed_node_id: str,
        recovery_nodes: Sequence[Mapping[str, Any]],
        recovery_group: Mapping[str, Any],
        failure_type: str,
        active_env_ids: Sequence[int] | None = None,
        preserve_failed_group_suffix: bool = False,
    ) -> dict[str, Any]:
        """Insert a complete recovery TaskGroup and rewire the unfinished suffix."""
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"Unknown failure type {failure_type!r}.")
        if len(self.revisions) >= self.max_revisions:
            raise RuntimeError("RuntimeGraph revision budget exhausted.")
        if (
            self._recovery_action_count + len(recovery_nodes)
            > self.max_recovery_actions
        ):
            raise RuntimeError("RuntimeGraph recovery-action budget exhausted.")
        env_ids = tuple(
            sorted(
                set(
                    range(self.num_envs)
                    if active_env_ids is None
                    else (int(env_id) for env_id in active_env_ids)
                )
            )
        )
        if not env_ids or env_ids[0] < 0 or env_ids[-1] >= self.num_envs:
            raise ValueError(
                "Recovery active_env_ids are outside the environment range."
            )
        failed_node = _node(self._graph, failed_node_id)
        failed_group_id = str(failed_node["task_instance_id"])
        group = deepcopy(dict(recovery_group))
        if group.get("role") != "recovery":
            raise ValueError("Inserted recovery TaskGroup must use role='recovery'.")
        group_id = str(group.get("id", ""))
        if not group_id:
            raise ValueError("Inserted recovery TaskGroup requires an ID.")
        if any(item["id"] == group_id for item in self._graph["task_groups"]):
            raise ValueError(f"RuntimeGraph already contains TaskGroup {group_id!r}.")
        nodes = [deepcopy(dict(node)) for node in recovery_nodes]
        if not nodes:
            raise ValueError("Recovery subgraph must contain at least one node.")
        for node in nodes:
            node.pop("contract", None)
            node.pop("resources", None)
            node["role"] = "cleanup" if node.get("role") == "cleanup" else "recovery"
            node["task_instance_id"] = group_id
            node["task_type"] = group["task_type"]
        recovery_ids = {str(node["id"]) for node in nodes}
        if len(recovery_ids) != len(nodes):
            raise ValueError("Recovery node IDs must be unique.")
        node_by_id = {str(node["id"]): node for node in self._graph["nodes"]}
        children_by_id = {node_id: [] for node_id in node_by_id}
        for node_id, node in node_by_id.items():
            for dependency in node["depends_on"]:
                children_by_id[str(dependency)].append(node_id)
        descendants: set[str] = set()
        pending = list(children_by_id[failed_node_id])
        while pending:
            node_id = pending.pop()
            if node_id in descendants:
                continue
            descendants.add(node_id)
            pending.extend(children_by_id[node_id])
        same_group_descendants = {
            node_id
            for node_id in descendants
            if str(node_by_id[node_id]["task_instance_id"]) == failed_group_id
        }
        cleanup_suffix_ids: set[str] = set()
        if (
            str(failed_node["atomic_action"]) == "HandOver"
            and not preserve_failed_group_suffix
        ):
            # A failed handover leaves ownership indeterminate.  Its
            # transfer-arm retreat/home tail must not execute from a stale
            # handover pose; recovery owns the cleanup before replanning.
            cleanup_suffix_ids = {
                node_id
                for node_id in same_group_descendants
                if node_by_id[node_id]["role"] == "cleanup"
            }
            non_cleanup_dependents = [
                node_id
                for node_id in same_group_descendants - cleanup_suffix_ids
                if any(
                    dependency in cleanup_suffix_ids
                    for dependency in node_by_id[node_id]["depends_on"]
                )
            ]
            if non_cleanup_dependents:
                raise ValueError(
                    "Cannot remove the HandOver cleanup suffix because it feeds "
                    f"same-group non-cleanup nodes: {sorted(non_cleanup_dependents)}."
                )
        first = [
            node
            for node in nodes
            if not any(dep in recovery_ids for dep in node.get("depends_on", []))
        ]
        if not first:
            raise ValueError("Recovery subgraph has no entry node.")
        for node in first:
            node["depends_on"] = list(
                dict.fromkeys([*node.get("depends_on", []), failed_node_id])
            )
        terminal_ids = _terminal_ids(nodes)

        patched = deepcopy(self._graph)
        for node in patched["nodes"]:
            if node["id"] in recovery_ids:
                raise ValueError(f"RuntimeGraph already contains node {node['id']!r}.")
            node_id = str(node["id"])
            if (
                preserve_failed_group_suffix
                or node_id not in descendants
                or node_id in same_group_descendants
            ):
                continue
            node["depends_on"] = list(
                dict.fromkeys(
                    [
                        dependency
                        for dependency in node["depends_on"]
                        if dependency != failed_node_id
                        and dependency not in cleanup_suffix_ids
                    ]
                    + terminal_ids
                )
            )
        if cleanup_suffix_ids:
            patched["nodes"] = [
                node
                for node in patched["nodes"]
                if str(node["id"]) not in cleanup_suffix_ids
            ]
            failed_group = next(
                item
                for item in patched["task_groups"]
                if str(item["id"]) == failed_group_id
            )
            failed_group["node_ids"] = [
                node_id
                for node_id in failed_group["node_ids"]
                if node_id not in cleanup_suffix_ids
            ]
        group["depends_on"] = list(
            dict.fromkeys([failed_group_id, *group.get("depends_on", [])])
        )
        group.pop("contract", None)
        group["node_ids"] = [str(node["id"]) for node in nodes]
        for downstream in patched["task_groups"]:
            if (
                not preserve_failed_group_suffix
                and failed_group_id in downstream["depends_on"]
            ):
                downstream["depends_on"] = [
                    dependency
                    for dependency in downstream["depends_on"]
                    if dependency != failed_group_id
                ] + [group_id]
        patched["nodes"].extend(nodes)
        patched["task_groups"].append(group)
        patched["metadata"] = {
            **patched.get("metadata", {}),
            "runtime_revision": len(self.revisions) + 1,
        }
        from embodichain.gen_sim.action_engine.planning.linker import link_seed_graph

        self._graph = link_seed_graph(
            patched,
            registry=self.registry,
        )
        self._recovery_action_count += len(nodes)
        self.revisions.append(
            GraphRevision(
                revision=len(self.revisions) + 1,
                kind="insert_recovery",
                reason=failure_type,
                failed_node_id=failed_node_id,
                inserted_group_ids=(group_id,),
                replaced_group_ids=(),
                active_env_ids=env_ids,
            )
        )
        return self.graph

    def insert_default_recovery(
        self,
        *,
        failed_node_id: str,
        failure_type: str,
        active_env_ids: Sequence[int] | None = None,
        resume_failed_group: bool = False,
    ) -> dict[str, Any]:
        """Insert one of the deliberately small built-in recovery strategies."""
        if failure_type != "object_fallen":
            raise ValueError(
                f"No bounded default recovery is registered for {failure_type!r}."
            )
        nodes, group = build_upright_recovery(
            self._graph,
            failed_node_id=failed_node_id,
            revision=len(self.revisions) + 1,
            resume_failed_group=resume_failed_group,
        )
        return self.insert_recovery_subgraph(
            failed_node_id=failed_node_id,
            recovery_nodes=nodes,
            recovery_group=group,
            failure_type=failure_type,
            active_env_ids=active_env_ids,
            preserve_failed_group_suffix=resume_failed_group,
        )

    def replace_unfinished_suffix(
        self,
        replacement: Mapping[str, Any],
        *,
        completed_group_ids: Sequence[str],
        reason: str,
    ) -> dict[str, Any]:
        """Install a fully replanned suffix while preserving completed groups."""
        if len(self.revisions) >= self.max_revisions:
            raise RuntimeError("RuntimeGraph revision budget exhausted.")
        candidate = validate_seed_graph(
            replacement,
            known_actions=self.registry.names(),
        )
        from embodichain.gen_sim.action_engine.planning.linker import (
            validate_persisted_contracts,
        )

        validate_persisted_contracts(candidate, self.registry)
        if candidate["task_id"] != self.seed_graph["task_id"]:
            raise ValueError("Suffix replanning cannot change the task_id.")
        if candidate["capability_catalog_hash"] != self.registry.catalog_hash():
            raise ValueError(
                "Replanned suffix capability catalog does not match runtime."
            )
        current_groups = {group["id"]: group for group in self._graph["task_groups"]}
        replacement_groups = {group["id"]: group for group in candidate["task_groups"]}
        current_nodes = {node["id"]: node for node in self._graph["nodes"]}
        replacement_nodes = {node["id"]: node for node in candidate["nodes"]}
        completed = set(completed_group_ids)
        for group_id in completed:
            current_group = current_groups.get(group_id)
            replacement_group = replacement_groups.get(group_id)
            if current_group is None or replacement_group != current_group:
                raise ValueError(
                    f"Replanning changed completed TaskGroup {group_id!r}."
                )
            if any(
                replacement_nodes.get(node_id) != current_nodes[node_id]
                for node_id in current_group["node_ids"]
            ):
                raise ValueError(
                    f"Replanning changed nodes of completed TaskGroup {group_id!r}."
                )
        replaced = tuple(sorted(set(current_groups) - completed))
        self._graph = candidate
        self.revisions.append(
            GraphRevision(
                revision=len(self.revisions) + 1,
                kind="replan_suffix",
                reason=str(reason),
                failed_node_id=None,
                inserted_group_ids=tuple(
                    sorted(set(replacement_groups) - set(current_groups))
                ),
                replaced_group_ids=replaced,
                active_env_ids=tuple(range(self.num_envs)),
            )
        )
        return self.graph


def classify_failure(
    action_name: str,
    *,
    planning_succeeded: bool,
    postcondition_succeeded: bool | None = None,
    object_fallen: bool = False,
    held_before: bool = False,
    held_after: bool = False,
    registry: AtomicCapabilityRegistry | None = None,
) -> str:
    """Classify only the bounded common recovery cases supported by v2."""
    capability = (registry or build_atomic_capability_registry()).get(action_name)
    if capability.failure_classifier_hook is not None:
        result = capability.failure_classifier_hook(
            action_name=action_name,
            planning_succeeded=planning_succeeded,
            postcondition_succeeded=postcondition_succeeded,
            object_fallen=object_fallen,
            held_before=held_before,
            held_after=held_after,
        )
        if result not in FAILURE_TYPES:
            raise ValueError(
                f"AtomicAction {action_name!r} failure classifier returned {result!r}."
            )
        return result
    if not planning_succeeded:
        return "search_exhausted"
    if object_fallen:
        return "object_fallen"
    if held_before and not held_after:
        return "object_dropped"
    if capability.failure_classifier == "grasp" and not held_after:
        return "grasp_missed"
    if postcondition_succeeded is False:
        return "postcondition_failed"
    return "postcondition_failed"


def build_upright_recovery(
    graph: Mapping[str, Any],
    *,
    failed_node_id: str,
    revision: int,
    resume_failed_group: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a coordinate-free E2 recovery group for a fallen rigid object."""
    failed = _node(graph, failed_node_id)
    object_uid = str(failed["object_uid"])
    group_id = f"recovery_e2_{int(revision):02d}_{failed_node_id}"
    actor = _recovery_actor(graph, failed)
    held_consumer_arm = None
    if not resume_failed_group:
        held_consumer_arm = _downstream_held_consumer_arm(graph, failed, object_uid)
    if held_consumer_arm is not None and not (
        actor.get("mode") == "required" and actor.get("arm") == held_consumer_arm
    ):
        raise ValueError(
            "Recovery cannot satisfy the downstream held-object contract without "
            "changing the failed TaskGroup actor; resume and replay the failed "
            "TaskGroup instead."
        )
    hold_for_downstream = (
        held_consumer_arm is not None
        and actor.get("mode") == "required"
        and actor.get("arm") == held_consumer_arm
    )
    upright = motion_policy(("orientation", "upright"))
    full_specs = (
        ("PickUp", {"kind": "object", "object": object_uid}, upright),
        (
            "MoveHeldObject",
            {"kind": "semantic_goal", "semantic_step": group_id, "phase": "final"},
            upright,
        ),
        ("Place", {"kind": "current_held_pose"}, upright),
        (
            "MoveEndEffector",
            {"kind": "policy_pose", "source": "release", "operation": "retreat"},
            upright,
        ),
        (
            "MoveJoints",
            {"kind": "joint_state", "source": "initial"},
            motion_policy(),
        ),
    )
    specs = full_specs[:2] if hold_for_downstream else full_specs
    nodes = []
    registry = build_atomic_capability_registry()
    dependencies: list[str] = []
    for index, (action, binding, policy_spec) in enumerate(specs, start=1):
        node_id = f"{group_id}__a{index:02d}"
        node = {
            "id": node_id,
            "atomic_action": action,
            "object_uid": object_uid,
            "actor": actor,
            "control": "arm",
            "target_binding": binding,
            "depends_on": dependencies,
            "task_instance_id": group_id,
            "task_type": "E2",
            "role": "recovery" if index <= 3 else "cleanup",
            "precondition": {},
            "postcondition": {},
            "motion_policy": deepcopy(dict(policy_spec)),
        }
        node["precondition"] = capability_precondition(
            registry.get(action),
            object_uid=object_uid,
            actor=actor,
            target_binding=binding,
        )
        nodes.append(node)
        dependencies = [node_id]
    group = {
        "id": group_id,
        "task_type": "E2",
        "role": "recovery",
        "operator": "orient_object",
        "object_uid": object_uid,
        "actor": actor,
        "goal": {
            "relation": "none",
            "reference_state": "live",
            "orientation_goal": "upright",
            "orientation_axis": "none",
            "position_anchor": "live_xy",
            "support_object": "table",
            "upright_local_axis": "long_axis",
            "terminal_behavior": "hold" if hold_for_downstream else "place",
        },
        "depends_on": [],
        "parent_task_instance_id": str(failed["task_instance_id"]),
        "node_ids": [node["id"] for node in nodes],
        "success": {"type": "object_upright", "object": object_uid},
    }
    return nodes, group


def _recovery_actor(
    graph: Mapping[str, Any],
    failed: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the failed TaskGroup's arm-selection contract."""
    group_id = str(failed["task_instance_id"])
    group = next(
        (item for item in graph["task_groups"] if str(item["id"]) == group_id),
        None,
    )
    source = (group or failed).get("actor", {"mode": "auto"})
    if not isinstance(source, Mapping):
        raise ValueError(f"Failed TaskGroup {group_id!r} has an invalid actor.")
    actor = deepcopy(dict(source))
    mode = str(actor.get("mode", "auto"))
    if mode == "required":
        if actor.get("arm") not in {"left_arm", "right_arm"}:
            raise ValueError(
                f"Failed TaskGroup {group_id!r} has an invalid required arm."
            )
    elif mode == "auto":
        actor = {"mode": "auto"}
    elif mode == "coordinated":
        raise ValueError(
            "The single-arm upright recovery cannot inherit a coordinated actor."
        )
    else:
        raise ValueError(f"The upright recovery cannot inherit actor mode {mode!r}.")
    return actor


def _downstream_held_consumer_arm(
    graph: Mapping[str, Any],
    failed: Mapping[str, Any],
    object_uid: str,
) -> str | None:
    failed_group_id = str(failed["task_instance_id"])
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    for group in graph["task_groups"]:
        if failed_group_id not in {str(item) for item in group.get("depends_on", ())}:
            continue
        if str(group.get("object_uid")) != object_uid:
            continue
        node_ids = {str(item) for item in group["node_ids"]}
        for node_id in group["node_ids"]:
            node = nodes[str(node_id)]
            if any(str(parent) in node_ids for parent in node["depends_on"]):
                continue
            for requirement in node.get("contract", {}).get("requires", ()):
                if (
                    requirement.get("predicate") == "object_held"
                    and requirement.get("object_uid") == object_uid
                    and requirement.get("arm") in {"left_arm", "right_arm"}
                ):
                    return str(requirement["arm"])
    return None


def _node(graph: Mapping[str, Any], node_id: str) -> Mapping[str, Any]:
    try:
        return next(node for node in graph["nodes"] if node["id"] == node_id)
    except StopIteration as error:
        raise ValueError(f"RuntimeGraph contains no node {node_id!r}.") from error


def _terminal_ids(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    depended = {
        dependency for node in nodes for dependency in node.get("depends_on", [])
    }
    return [str(node["id"]) for node in nodes if node["id"] not in depended]


def _mask(value: torch.Tensor, count: int) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    if result.numel() != count:
        raise ValueError(f"Expected a mask with {count} values.")
    return result
