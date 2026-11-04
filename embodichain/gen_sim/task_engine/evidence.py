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
from embodichain.gen_sim.task_spec.evidence import (
    SupportedPlacementEvidenceBinding,
    evaluate_supported_placement_evidence,
)

__all__ = ["project_placement_evidence_binding"]


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
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> SupportedPlacementEvidenceBinding:
    """Project one single-member placement declaration without evaluating it.

    Args:
        spec: Frozen normative specification.
        graph: Frozen graph whose role/group selection was validated.
        validation: Matching graph-validation content identities and coverage.
        program: Exact serialized program snapshot, without removing validators.
        constraints: Frozen source constraint declaration.
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
    graph, validation, program, constraints, occurrences = (
        json_snapshot(value)
        for value in (graph, validation, program, constraints, occurrences)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
        ("constraints_digest", constraints),
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
    condition = constraints["presets"][stability_preset]
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
        condition=condition,
        required_execution=required,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
    )
    # TaskSpec owns goal/milestone association rules. Empty observations exercise
    # that static selection without collecting or evaluating physical samples.
    evaluate_supported_placement_evidence(normalized, result, {}, program=program)
    return result
