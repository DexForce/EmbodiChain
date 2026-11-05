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

"""Thin Task Engine projection of frozen declarations into TaskSpec evidence IDs.

This module reads no simulator, pose buffer, sensor, or execution provider. The
host supplies plain descriptors from its already compiled execution occurrences.
TaskSpec owns physical qualification and certificate decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash, json_snapshot
from embodichain.gen_sim.task_spec.condition_evidence import (
    ConditionEvidenceBinding,
    evaluate_condition_evidence,
)
from embodichain.gen_sim.task_spec.evidence import (
    SupportedPlacementEvidenceBinding,
    evaluate_supported_placement_evidence,
)

__all__ = [
    "project_placement_evidence_binding",
    "project_configured_placement_evidence_binding",
    "project_condition_evidence_binding",
]


def _project_placement_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    condition: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    *,
    predicate_id: str,
    release_node_id: str,
    boundary_node_id: str,
    evidence_scope: str,
    stability_preset: str,
    observation_kind: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> SupportedPlacementEvidenceBinding:
    """Project one single-member placement declaration without evaluating it.

    Args:
        spec: Frozen normative specification.
        graph: Frozen graph whose role/group selection was validated.
        validation: Matching graph-validation content identities and coverage.
        program: Exact serialized program snapshot, without removing validators.
        condition: Plain resolved owner condition declaration.
        occurrences: Plain compiled descriptors keyed by graph node ID. Each
            contains ``segment_name`` and ``source_path``. The selected release
            additionally contains its exact ``invocation_id``, resolved
            ``object_id``, ``reference_id`` and ``resource_id``. This adapter never
            invents invocation IDs or infers native capabilities from names.
        predicate_id: Goal or milestone predicate to bind.
        release_node_id: Selected release node in the declared member's group.
        boundary_node_id: Explicit observed final or recorded boundary node.
        evidence_scope: Selected source evidence scope; not inferred from data.
        stability_preset: Explicit named constraint referenced by the support
            occurrence. Its meaning is never inferred from the preset name.
        observation_kind: ``final`` or ``recorded``.
        evidence_format: Explicit raw_snapshot or historical_record contract.
        episode_ids: Optional historical identity, required for new Lab snapshots.
        num_envs: Ordered native row count.

    Returns:
        Independently owned, hashed normative binding. It is an evidence
        declaration, not a second runtime-policy configuration center.

    Raises:
        ValueError: Source identities, coverage, roles or occurrences disagree.
        KeyError: A required compiled occurrence or declaration is absent.
    """
    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, condition, occurrences = (
        json_snapshot(value)
        for value in (graph, validation, program, condition, occurrences)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError(f"Frozen evidence projection has a different {key}.")
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("Placement projection requires complete declared coverage.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    predicates = [(item, None) for item in normalized.goal] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
    ]
    matches = [
        (item, milestone) for item, milestone in predicates if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Projection requires one exact declared predicate.")
    predicate, milestone = matches[0]
    if predicate.name != "semantic_goal" or predicate.arguments.get("relation") not in (
        {"on", "inside"} if evidence_scope == "native_supported_placement" else {"on"}
    ):
        raise ValueError(
            "This projection supports the declared rigid on-relation slice."
        )
    selected_coverage = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected_coverage) != 1 or len(selected_coverage[0]["groups"]) != 1:
        raise ValueError("This projection requires one fully declared placement group.")
    group_id = selected_coverage[0]["groups"][0]
    groups = [item for item in graph["task_groups"] if item["id"] == group_id]
    if len(groups) != 1:
        raise ValueError("Placement group identity is missing or ambiguous.")
    group = groups[0]
    references = []
    role_definitions = {item.name: item for item in normalized.roles}
    for key in ("subject", "reference"):
        role_id = predicate.arguments[key]
        values = coverage["roles"][role_id]
        if (
            role_definitions[role_id].quantifier != "one"
            or len(values) != 1
            or values[0].get("part_id") is not None
        ):
            raise ValueError("Projection requires complete single rigid-role bindings.")
        references.append(values[0]["uid"])
    object_id, reference_id = references
    lineage = [
        item
        for values in coverage["lineage"].values()
        for item in values
        if item["group_id"] == group_id
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != object_id
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("Placement lineage differs from its normative subject.")
    if release_node_id not in group["node_ids"]:
        raise ValueError("Release does not belong to the declared placement group.")
    support_node_id = group["node_ids"][-1]
    if observation_kind == "recorded" and boundary_node_id != support_node_id:
        raise ValueError(
            "Recorded placement must use its declared terminal group boundary."
        )
    nodes = {item["id"]: item for item in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate graph nodes cannot identify execution occurrences.")

    declared_segments = {}

    def occurrence(node_id: str) -> Mapping[str, Any]:
        value = occurrences[node_id]
        if value.get("segment_name") != node_id:
            raise ValueError("Compiled occurrence names another graph node.")
        segment: Any = program
        for component in value["source_path"]:
            if type(component) not in (str, int):
                raise ValueError("Compiled source path has an invalid component.")
            segment = segment[component]
        if segment.get("kind") != "segment" or segment.get("name") != node_id:
            raise ValueError(
                "Compiled source path does not select its declared segment."
            )
        if segment["steps"].get("kind") != "invoke" or canonical_hash(
            segment["steps"]["call"]
        ) != canonical_hash(nodes[node_id]["call"]):
            raise ValueError(
                "Compiled program call differs from its frozen graph node."
            )
        declared_segments[node_id] = segment
        return value

    release = occurrence(release_node_id)
    support = occurrence(support_node_id)
    boundary = occurrence(boundary_node_id)
    if (
        release.get("object_id") != object_id
        or release.get("reference_id") != reference_id
    ):
        raise ValueError("Resolved release endpoints differ from the grounded roles.")
    if (
        release.get("resource_id")
        != nodes[release_node_id]["call"]["resources"]["primary"]
    ):
        raise ValueError("Resolved release resource differs from the declared call.")
    if not any(
        policy.get("kind") == "wait_stable"
        and policy.get("preset") == stability_preset
        and policy.get("entity") == object_id
        for policy in declared_segments[support_node_id].get("post", [])
    ):
        raise ValueError(
            "Support occurrence does not declare the selected named preset."
        )
    if condition and (
        condition.get("entity") != object_id
        or condition.get("reference") != reference_id
    ):
        raise ValueError("Placement condition differs from its grounded endpoints.")
    required = (
        tuple(
            {"segment_id": node_id, "source_path": occurrence(node_id)["source_path"]}
            for node_id in group["node_ids"]
        )
        if milestone is not None
        else ()
    )
    result = SupportedPlacementEvidenceBinding(
        task_spec_digest=validation["task_spec_digest"],
        program_digest=validation["program_digest"],
        predicate_id=predicate_id,
        object_id=object_id,
        reference_id=reference_id,
        resource_id=release["resource_id"],
        release_segment_id=release_node_id,
        release_invocation_id=release["invocation_id"],
        release_source_path=tuple(release["source_path"]),
        support_segment_id=support_node_id,
        support_source_path=tuple(support["source_path"]),
        stability_preset=stability_preset,
        boundary_segment_id=boundary_node_id,
        boundary_source_path=tuple(boundary["source_path"]),
        evidence_scope=evidence_scope,
        observation_kind=observation_kind,
        evidence_format=evidence_format,
        condition=condition,
        required_execution=required,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
    )
    # TaskSpec owns goal/milestone association rules. Empty observations exercise
    # that static selection without collecting or evaluating physical samples.
    evaluate_supported_placement_evidence(normalized, result, {}, program=program)
    return result


def project_configured_placement_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    *,
    condition: Mapping[str, Any],
    predicate_id: str,
    release_node_id: str,
    boundary_node_id: str,
    evidence_scope: str,
    stability_preset: str,
    observation_kind: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> SupportedPlacementEvidenceBinding:
    """Project configured owner declarations without a second constraint file.

    Args:
        spec: Frozen normative TaskSpec.
        graph: Validated semantic graph and role lineage.
        validation: Matching program/spec/graph digests and complete coverage.
        program: Exact serialized program, retaining policies and validators.
        occurrences: Plain compiled invocation/segment descriptors keyed by node.
        condition: Detached public declaration from the configured owner's named
            settle preset. The full condition participates in the evidence binding
            hash; the configured deployment separately identifies all owner
            declarations. No old constraints payload or digest is required.
        predicate_id: Exact goal or milestone predicate identity.
        release_node_id: Compiled release selection in the declared group.
        boundary_node_id: Explicit final or recorded boundary selection.
        evidence_scope: Explicit supported physical proof contract.
        stability_preset: Exact configured name referenced by the support policy.
        observation_kind: Final or recorded observation, matching the predicate.
        evidence_format: Raw snapshot or explicit historical record format.
        episode_ids: Initialized row identities, required for raw evaluation.
        num_envs: Ordered native environment count.

    Returns:
        Owned normative declaration with unchanged condition values and exact
        invocation/occurrence identities. No simulator is created or sampled.
    """
    return _project_placement_evidence_binding(
        spec,
        graph,
        validation,
        program,
        condition,
        occurrences,
        predicate_id=predicate_id,
        release_node_id=release_node_id,
        boundary_node_id=boundary_node_id,
        evidence_scope=evidence_scope,
        stability_preset=stability_preset,
        observation_kind=observation_kind,
        evidence_format=evidence_format,
        episode_ids=episode_ids,
        num_envs=num_envs,
    )


def project_placement_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    *,
    predicate_id: str,
    release_node_id: str,
    boundary_node_id: str,
    evidence_scope: str,
    stability_preset: str,
    observation_kind: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> SupportedPlacementEvidenceBinding:
    """Project a frozen B0 declaration for source-comparison re-evaluation.

    Args:
        spec: Frozen normative TaskSpec.
        graph: Frozen validated semantic graph.
        validation: Matching source program/spec/graph/constraints digests.
        program: Complete serialized source program.
        constraints: Frozen B0 source declaration whose original identity is
            verified. New bundles use configured owner declarations through
            :func:`project_configured_placement_evidence_binding` instead.
        occurrences: Plain compiled invocation and segment descriptors.
        predicate_id: Exact goal or milestone predicate.
        release_node_id: Explicit release occurrence in the declared group.
        boundary_node_id: Explicit observation boundary.
        evidence_scope: Explicit physical qualification contract.
        stability_preset: Exact source preset referenced by its policy.
        observation_kind: Final or recorded observation.
        evidence_format: Explicit raw or historical snapshot format.
        episode_ids: Optional static binding, required for raw evaluation.
        num_envs: Ordered native environment count.

    Returns:
        The same normalized evidence binding as the configured-owner entry point.
        This source-comparison adapter does not read or emit bundle files.
    """
    source = json_snapshot(constraints)
    if validation.get("constraints_digest") != canonical_hash(source):
        raise ValueError(
            "Frozen evidence projection has a different constraints_digest."
        )
    return project_configured_placement_evidence_binding(
        spec,
        graph,
        validation,
        program,
        occurrences,
        condition=source["presets"][stability_preset],
        predicate_id=predicate_id,
        release_node_id=release_node_id,
        boundary_node_id=boundary_node_id,
        evidence_scope=evidence_scope,
        stability_preset=stability_preset,
        observation_kind=observation_kind,
        evidence_format=evidence_format,
        episode_ids=episode_ids,
        num_envs=num_envs,
    )


def project_condition_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    resolved_state: Mapping[str, Any],
    *,
    predicate_id: str,
    boundary_node_id: str,
    policy_node_id: str,
    stability_preset: str,
    evidence_kind: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> ConditionEvidenceBinding:
    """Project grounded orientation/joint declarations into normative identities.

    Args:
        spec: Frozen TaskSpec.
        graph: Validated semantic graph, including exact group lineage.
        validation: Matching program/spec/graph digests and coverage declaration.
        program: Complete serialized program, retaining validators and policies.
        occurrences: Compiled segment names and full source paths keyed by graph
            node. Repeated executions must have distinct compiled occurrences.
        resolved_state: Plain descriptor from the configured owner, containing
            entity_id, native_uid, part_id and condition. Condition values are
            already resolved public policy declarations; this adapter never
            selects or changes controller thresholds or native calibration.
        predicate_id: Exact declared goal or milestone predicate.
        boundary_node_id: Explicit final/recorded observation boundary.
        policy_node_id: Explicit occurrence selecting the named condition.
        stability_preset: Exact configured policy name, without name inference.
        evidence_kind: Explicit orientation or joint_endpoint qualification.
        evidence_format: Explicit raw_snapshot or historical_record contract.
        episode_ids: Optional static binding, required for raw evaluation.
        num_envs: Native ordered row count.

    Returns:
        An independently owned TaskSpec binding. Goal/milestone association and
        supported normative predicates are validated by TaskSpec itself.
    """
    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, occurrences, state = (
        json_snapshot(value)
        for value in (graph, validation, program, occurrences, resolved_state)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError(f"Frozen state projection has a different {key}.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("State projection requires complete declared coverage.")
    predicates = [(item, None) for item in normalized.goal] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
    ]
    matches = [
        (item, milestone) for item, milestone in predicates if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("State projection requires one exact predicate identity.")
    predicate, milestone = matches[0]
    selected = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected) != 1 or len(selected[0]["groups"]) != 1:
        raise ValueError("State projection requires one fully declared semantic group.")
    group_id = selected[0]["groups"][0]
    groups = [item for item in graph["task_groups"] if item["id"] == group_id]
    if len(groups) != 1:
        raise ValueError("State group is missing or ambiguous.")
    group = groups[0]
    role_id = predicate.arguments["subject"]
    role = next(item for item in normalized.roles if item.name == role_id)
    members = coverage["roles"][role_id]
    if (
        role.quantifier != "one"
        or len(members) != 1
        or members[0] != {"uid": state["entity_id"], "part_id": state.get("part_id")}
    ):
        raise ValueError(
            "State descriptor differs from its complete single-role binding."
        )
    lineage = [
        item
        for entries in coverage["lineage"].values()
        for item in entries
        if item["group_id"] == group_id
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != state["entity_id"]
        or lineage[0].get("object_part_id") != state.get("part_id")
    ):
        raise ValueError("State descriptor differs from its grounded lineage.")
    nodes = {item["id"]: item for item in graph["nodes"]}
    if (
        len(nodes) != len(graph["nodes"])
        or not group["node_ids"]
        or len(set(group["node_ids"])) != len(group["node_ids"])
    ):
        raise ValueError(
            "State execution occurrences must be explicit and unambiguous."
        )

    def occurrence(node_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        descriptor = occurrences[node_id]
        declared = program
        for component in descriptor["source_path"]:
            if type(component) not in (str, int):
                raise ValueError("State source path has an invalid component.")
            declared = declared[component]
        if (
            declared.get("kind") != "segment"
            or declared.get("name") != descriptor["segment_name"]
            or declared["steps"].get("kind") != "invoke"
            or canonical_hash(declared["steps"]["call"])
            != canonical_hash(nodes[node_id]["call"])
        ):
            raise ValueError(
                "Compiled state occurrence differs from its declared graph call."
            )
        return descriptor, declared

    if policy_node_id not in group["node_ids"]:
        raise ValueError("State policy does not belong to the declared group.")
    if milestone is not None and boundary_node_id != group["node_ids"][-1]:
        raise ValueError(
            "Recorded state must use the declared terminal group boundary."
        )
    boundary, _ = occurrence(boundary_node_id)
    policy, declared_policy = occurrence(policy_node_id)
    if (
        sum(
            item.get("kind") == "wait_stable"
            and item.get("entity") == state["entity_id"]
            and item.get("preset") == stability_preset
            for item in declared_policy.get("post", [])
        )
        != 1
    ):
        raise ValueError("The selected state occurrence lacks its exact named policy.")
    condition = dict(state["condition"])
    if condition.get("entity") != state["entity_id"]:
        raise ValueError("State condition names another entity.")
    if evidence_kind == "joint_endpoint":
        if (
            "target_state" in condition
            and condition["target_state"] != predicate.arguments["target_state"]
        ):
            raise ValueError("State condition names another semantic endpoint.")
        if "part_id" in condition and condition["part_id"] != state.get("part_id"):
            raise ValueError("State condition names another grounded part.")
        condition.update(
            target_state=predicate.arguments["target_state"],
            part_id=state.get("part_id"),
        )
    required = []
    if milestone is not None:
        for node_id in group["node_ids"]:
            descriptor, _ = occurrence(node_id)
            required.append(
                {
                    "segment_id": descriptor["segment_name"],
                    "source_path": descriptor["source_path"],
                }
            )
    result = ConditionEvidenceBinding(
        task_spec_digest=validation["task_spec_digest"],
        program_digest=validation["program_digest"],
        predicate_id=predicate_id,
        entity_id=state["entity_id"],
        native_uid=state["native_uid"],
        evidence_kind=evidence_kind,
        observation_kind="final" if milestone is None else "recorded",
        condition=condition,
        evidence_format=evidence_format,
        boundary_segment_id=boundary["segment_name"],
        boundary_source_path=tuple(boundary["source_path"]),
        policy_segment_id=policy["segment_name"],
        policy_source_path=tuple(policy["source_path"]),
        stability_preset=stability_preset,
        required_execution=tuple(required),
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
    )
    evaluate_condition_evidence(normalized, result, {}, program=program)
    return result
