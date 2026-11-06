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

from embodichain.gen_sim.task_spec.placement_group_evidence import (
    PlacementEvidenceGroupBinding,
    evaluate_placement_group_evidence,
)

from embodichain.gen_sim.task_spec.condition_evidence import (
    ConditionEvidenceBinding,
    evaluate_condition_evidence,
)

from embodichain.gen_sim.task_spec.condition_group_evidence import (
    ConditionEvidenceGroupBinding,
    evaluate_condition_group_evidence,
)

from embodichain.lab.task_program.integrations.evidence import HandOverEvidenceBinding
from embodichain.lab.task_program.integrations.evidence import (
    CoordinatedHoldEvidenceBinding,
)

from embodichain.gen_sim.task_spec.pour_evidence import (
    PourEvidenceBinding,
    evaluate_pour_evidence,
)

from embodichain.lab.task_program.integrations.evidence import (
    CoordinatedTransportEvidenceBinding,
)
from embodichain.gen_sim.task_spec.manipulation_evidence import (
    HoldAcceptanceCriteria,
    evaluate_manipulation_evidence,
)
from .evidence_binding import TaskEngineManipulationBinding, _acceptance

from embodichain.gen_sim.task_spec.control_evidence import (
    ControlEvidenceBinding,
    evaluate_control_evidence,
)

__all__ = [
    "project_placement_evidence_binding",
    "project_control_evidence_binding",
    "project_configured_placement_evidence_binding",
    "project_configured_placement_evidence_group_binding",
    "project_condition_evidence_binding",
    "project_condition_evidence_group_binding",
    "project_pour_evidence_binding",
    "project_handover_evidence_binding",
    "project_coordinated_hold_evidence_binding",
    "project_coordinated_transport_evidence_binding",
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
    operation_evidence: Mapping[str, Any] | None = None,
    sampled_contact_identity: Mapping[str, Any] | None = None,
    planar_geometry: Mapping[str, Any] | None = None,
    directional_geometry: Mapping[str, Any] | None = None,
    member_group_id: str | None = None,
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
        directional_geometry: Explicit rigid/link spatial route and native owners.
        planar_geometry: Explicit native planar mode and owner identities.
        sampled_contact_identity: Actual collector program/native identity for raw sampled support.
        operation_evidence: Explicit E4 transfer or E5 transport linkage; never inferred.
        num_envs: Ordered native row count.

    Returns:
        Independently owned, hashed normative binding. It is an evidence
        declaration, not a second runtime-policy configuration center.

    Raises:
        ValueError: Source identities, coverage, roles or occurrences disagree.
        KeyError: A required compiled occurrence or declaration is absent.
    """
    if directional_geometry is not None:
        directional_geometry = json_snapshot(directional_geometry)
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
    permitted = (
        {directional_geometry["relation"]}
        if directional_geometry is not None
        else (
            {"on", "inside"}
            if evidence_scope
            in {"native_supported_placement", "measured_planar_landing_region"}
            else {"on"}
        )
    )
    if (
        predicate.name != "semantic_goal"
        or predicate.arguments.get("relation") not in permitted
    ):
        raise ValueError(
            "This projection requires an explicitly supported physical relation."
        )
    selected_coverage = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected_coverage) != 1:
        raise ValueError("This projection requires one fully declared placement group.")
    group_ids = selected_coverage[0]["groups"]
    if member_group_id is None:
        if len(group_ids) != 1:
            raise ValueError(
                "This projection requires one fully declared placement group."
            )
        group_id = group_ids[0]
    else:
        group_id = member_group_id
        role = next(
            value
            for value in normalized.roles
            if value.name == predicate.arguments["subject"]
        )
        members = coverage["roles"][role.name]
        full_lineage = [
            value
            for values in coverage["lineage"].values()
            for value in values
            if value["group_id"] in group_ids
        ]
        if (
            role.quantifier != "count"
            or role.count != len(members)
            or role.count != len(group_ids)
            or role.count != len(full_lineage)
            or len(set(group_ids)) != len(group_ids)
            or group_id not in group_ids
            or len({(value["uid"], value.get("part_id")) for value in members})
            != len(members)
            or {value["group_id"] for value in full_lineage} != set(group_ids)
            or {
                (value["object_uid"], value.get("object_part_id"))
                for value in full_lineage
            }
            != {(value["uid"], value.get("part_id")) for value in members}
        ):
            raise ValueError(
                "Placement count projection requires its complete member/group bijection."
            )
    groups = [item for item in graph["task_groups"] if item["id"] == group_id]
    if len(groups) != 1:
        raise ValueError("Placement group identity is missing or ambiguous.")
    group = groups[0]
    if member_group_id is not None and group.get("task_type") != "E1":
        raise ValueError("Placement grouping retains only the source E1 scope.")
    selected_member = [
        value
        for values in coverage["lineage"].values()
        for value in values
        if value["group_id"] == group_id
    ]
    if len(selected_member) != 1:
        raise ValueError("Placement count has ambiguous member lineage.")
    references = []
    role_definitions = {item.name: item for item in normalized.roles}
    for key in ("subject", "reference"):
        role_id = predicate.arguments[key]
        values = coverage["roles"][role_id]
        if member_group_id is not None and key == "subject":
            current = {
                "uid": selected_member[0]["object_uid"],
                "part_id": selected_member[0].get("object_part_id"),
            }
            if current not in values or current["part_id"] is not None:
                raise ValueError(
                    "Placement count requires its exact rigid subject member."
                )
            values = [current]
        elif not role_definitions[role_id].is_single_member:
            raise ValueError("Projection requires complete single rigid-role bindings.")
        if len(values) != 1 or values[0].get("part_id") != (
            directional_geometry.get("reference_part")
            if key == "reference" and directional_geometry is not None
            else None
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

    if operation_evidence is not None:
        operation_evidence = json_snapshot(operation_evidence)
        family = operation_evidence["family"]
        operation = operation_evidence["operation"]
        if family != group.get("task_type") or operation["operation_id"] != group_id:
            raise ValueError(
                "Placement operation linkage differs from its grounded group."
            )
        for key in (("transfer", "boundary") if family == "E4" else ("transport",)):
            descriptor = operation[key]
            node_id = descriptor["segment_name"]
            if node_id not in group["node_ids"]:
                raise ValueError(
                    "Linked physical operation is outside the placement group."
                )
            compiled = occurrence(node_id)
            if any(
                descriptor[name] != compiled[name]
                for name in (
                    "segment_id",
                    "segment_name",
                    "source_path",
                    "invocation_id",
                )
            ):
                raise ValueError(
                    "Linked physical operation has another compiled identity."
                )
    release = occurrence(release_node_id)
    support = occurrence(support_node_id)
    boundary = occurrence(boundary_node_id)
    if (
        release.get("object_id") != object_id
        or release.get("reference_id") != reference_id
    ):
        raise ValueError("Resolved release endpoints differ from the grounded roles.")
    resources = nodes[release_node_id]["call"]["resources"]
    if operation_evidence is not None and operation_evidence["family"] == "E5":
        matched = release.get("resource_id") in resources.values()
    else:
        matched = release.get("resource_id") == resources.get("primary")
    if not matched:
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
        or condition.get("reference")
        != (
            (directional_geometry.get("reference_link_id") or reference_id)
            if directional_geometry is not None
            else reference_id
        )
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
        operation_evidence=operation_evidence,
        sampled_contact_identity=sampled_contact_identity,
        planar_geometry=planar_geometry,
        directional_geometry=directional_geometry,
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
    operation_evidence: Mapping[str, Any] | None = None,
    sampled_contact_identity: Mapping[str, Any] | None = None,
    planar_geometry: Mapping[str, Any] | None = None,
    directional_geometry: Mapping[str, Any] | None = None,
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
        directional_geometry: Explicit rigid/link spatial route and native owners.
        planar_geometry: Explicit native planar mode and owner identities.
        sampled_contact_identity: Actual collector program/native identity for raw sampled support.
        operation_evidence: Explicit E4 transfer or E5 transport linkage; never inferred.
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
        operation_evidence=operation_evidence,
        sampled_contact_identity=sampled_contact_identity,
        planar_geometry=planar_geometry,
        directional_geometry=directional_geometry,
    )


def project_configured_placement_evidence_group_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    resolved_members: Sequence[Mapping[str, Any]],
    *,
    predicate_id: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> PlacementEvidenceGroupBinding:
    """Bind every finite E1 subject to its original independent physical proof.

    Args:
        spec: Original complete normative TaskSpec.
        graph: Original ordered semantic groups and role lineage.
        validation: Matching identities and complete source coverage.
        program: Exact original serialized program.
        occurrences: Compiled descriptors with all resolved release endpoints.
        resolved_members: Ordered group_id, condition, release_node_id,
            boundary_node_id, evidence_scope and stability_preset plus existing
            optional planar/directional/sampled-contact proof declarations.
        predicate_id: One unchanged original goal or milestone predicate.
        evidence_format: Existing explicit raw or historical input format.
        episode_ids: Exact native episode identities.
        num_envs: Ordered native row count.

    Returns:
        Complete subject count proof retaining original member bindings/hashes.
        No reduced child specification or sampled physical verdict is created.
    """
    normalized = TaskSpec.from_dict(spec)
    values = json_snapshot(resolved_members)
    required = {
        "group_id",
        "condition",
        "release_node_id",
        "boundary_node_id",
        "evidence_scope",
        "stability_preset",
    }
    optional = {"planar_geometry", "directional_geometry", "sampled_contact_identity"}
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, dict)
            or not required <= set(value)
            or set(value) - required - optional
            for value in values
        )
    ):
        raise ValueError(
            "Placement count requires complete explicit member declarations."
        )
    predicates = [(value, None) for value in normalized.goal] + [
        (value, milestone.id)
        for milestone in normalized.milestones
        for value in milestone.achieve
    ]
    selected = [
        (value, milestone)
        for value, milestone in predicates
        if value.id == predicate_id
    ]
    if len(selected) != 1:
        raise ValueError("Placement count must select one original predicate.")
    predicate, milestone = selected[0]
    covered = [
        value
        for value in validation["coverage"][
            "goals" if milestone is None else "milestones"
        ]
        if value["id"] == (predicate_id if milestone is None else milestone)
    ]
    if (
        len(covered) != 1
        or [value["group_id"] for value in values] != covered[0]["groups"]
    ):
        raise ValueError("Placement count must retain every covered group in order.")
    members = tuple(
        _project_placement_evidence_binding(
            normalized,
            graph,
            validation,
            program,
            value["condition"],
            occurrences,
            predicate_id=predicate_id,
            release_node_id=value["release_node_id"],
            boundary_node_id=value["boundary_node_id"],
            evidence_scope=value["evidence_scope"],
            stability_preset=value["stability_preset"],
            observation_kind="final" if milestone is None else "recorded",
            evidence_format=evidence_format,
            episode_ids=episode_ids,
            num_envs=num_envs,
            member_group_id=value["group_id"],
            **{key: value[key] for key in optional if key in value},
        )
        for value in values
    )
    result = PlacementEvidenceGroupBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        predicate.arguments["subject"],
        tuple(value["group_id"] for value in values),
        members,
    )
    evaluate_placement_group_evidence(normalized, result, {}, program=program)
    return result


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
    operation_evidence: Mapping[str, Any] | None = None,
    sampled_contact_identity: Mapping[str, Any] | None = None,
    planar_geometry: Mapping[str, Any] | None = None,
    directional_geometry: Mapping[str, Any] | None = None,
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
        directional_geometry: Explicit rigid/link spatial route and native owners.
        planar_geometry: Explicit native planar mode and owner identities.
        sampled_contact_identity: Actual collector program/native identity for raw sampled support.
        operation_evidence: Explicit E4 transfer or E5 transport linkage; never inferred.
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
        operation_evidence=operation_evidence,
        sampled_contact_identity=sampled_contact_identity,
        planar_geometry=planar_geometry,
        directional_geometry=directional_geometry,
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
    return _project_condition_member_binding(
        spec,
        graph,
        validation,
        program,
        occurrences,
        resolved_state,
        predicate_id=predicate_id,
        boundary_node_id=boundary_node_id,
        policy_node_id=policy_node_id,
        stability_preset=stability_preset,
        evidence_kind=evidence_kind,
        evidence_format=evidence_format,
        episode_ids=episode_ids,
        num_envs=num_envs,
    )


def project_condition_evidence_group_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    resolved_members: Sequence[Mapping[str, Any]],
    *,
    predicate_id: str,
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
    evidence_format: str = "raw_snapshot",
    evidence_kind: str = "orientation",
) -> ConditionEvidenceGroupBinding:
    """Project finite E1/E2 orientation or E6 parts without a child TaskSpec.

    Args:
        spec: Original normative TaskSpec whose subject has a finite count.
        graph: Original complete semantic group declaration.
        validation: Matching identities, exact role membership and full lineage.
        program: Complete original serialized program.
        occurrences: Compiled occurrence descriptors keyed by graph node identity.
        resolved_members: Ordered mappings of group_id, state, boundary_node_id,
            policy_node_id and stability_preset from their existing owners.
        predicate_id: One unchanged original goal or milestone predicate.
        episode_ids: Exact native episode identities for raw acquisition.
        num_envs: Ordered environment rows.
        evidence_format: Explicit raw or historical contract, shared by all members.
        evidence_kind: Original orientation or explicit articulation-joint proof.

    Returns:
        Sealed complete count binding with independent member occurrence identities.
        The original spec and existing member binding format are unchanged.
    """
    normalized = TaskSpec.from_dict(spec)
    values = json_snapshot(resolved_members)
    required = {
        "group_id",
        "state",
        "boundary_node_id",
        "policy_node_id",
        "stability_preset",
    }
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, dict) or set(v) != required for v in values)
    ):
        raise ValueError(
            "State count projection requires explicit complete member descriptors."
        )
    predicates = [(p, None) for p in normalized.goal] + [
        (p, m.id) for m in normalized.milestones for p in m.achieve
    ]
    matches = [(p, milestone) for p, milestone in predicates if p.id == predicate_id]
    if len(matches) != 1:
        raise ValueError("State group projection requires one original predicate.")
    predicate, milestone = matches[0]
    covered = [
        v
        for v in validation["coverage"]["goals" if milestone is None else "milestones"]
        if v["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(covered) != 1 or [v["group_id"] for v in values] != covered[0]["groups"]:
        raise ValueError(
            "State count projection must retain every covered group in order."
        )
    members = tuple(
        _project_condition_member_binding(
            normalized,
            graph,
            validation,
            program,
            occurrences,
            value["state"],
            predicate_id=predicate_id,
            boundary_node_id=value["boundary_node_id"],
            policy_node_id=value["policy_node_id"],
            stability_preset=value["stability_preset"],
            evidence_kind=evidence_kind,
            evidence_format=evidence_format,
            episode_ids=episode_ids,
            num_envs=num_envs,
            member_group_id=value["group_id"],
        )
        for value in values
    )
    result = ConditionEvidenceGroupBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        predicate.arguments["subject"],
        tuple(v["group_id"] for v in values),
        members,
    )
    evaluate_condition_group_evidence(normalized, result, {}, program=program)
    return result


def _project_condition_member_binding(
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
    member_group_id: str | None = None,
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
    if len(selected) != 1:
        raise ValueError("State projection requires one fully declared semantic group.")
    role_id = predicate.arguments["subject"]
    role = next(item for item in normalized.roles if item.name == role_id)
    members = coverage["roles"][role_id]
    target_member = {"uid": state["entity_id"], "part_id": state.get("part_id")}
    if member_group_id is None:
        if len(selected[0]["groups"]) != 1:
            raise ValueError(
                "State projection requires one fully declared semantic group."
            )
        group_id = selected[0]["groups"][0]
        if (
            not role.is_single_member
            or len(members) != 1
            or members[0] != target_member
        ):
            raise ValueError(
                "State descriptor differs from its complete single-role binding."
            )
    else:
        group_id = member_group_id
        group_ids = selected[0]["groups"]
        full_lineage = [
            item
            for entries in coverage["lineage"].values()
            for item in entries
            if item["group_id"] in group_ids
        ]
        if (
            role.quantifier != "count"
            or role.count != len(members)
            or role.count != len(group_ids)
            or role.count != len(full_lineage)
            or len(set(group_ids)) != len(group_ids)
            or group_id not in group_ids
            or len({(v["uid"], v.get("part_id")) for v in members}) != len(members)
            or {v["group_id"] for v in full_lineage} != set(group_ids)
            or {(v["object_uid"], v.get("object_part_id")) for v in full_lineage}
            != {(v["uid"], v.get("part_id")) for v in members}
            or target_member not in members
        ):
            raise ValueError(
                "State count projection requires the complete member/group bijection."
            )
    groups = [item for item in graph["task_groups"] if item["id"] == group_id]
    if len(groups) != 1:
        raise ValueError("State group is missing or ambiguous.")
    group = groups[0]
    if member_group_id is not None and (group.get("task_type"), evidence_kind) not in {
        ("E1", "orientation"),
        ("E2", "orientation"),
        ("E6", "joint_endpoint"),
    }:
        raise ValueError("Count state grouping retains the source E1/E2/E6 scope.")
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


def project_pour_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    resolved_operation: Mapping[str, Any],
    *,
    predicate_id: str,
    stage_node_id: str,
    pour_node_id: str,
    release_node_id: str,
    boundary_node_id: str,
    evidence_format: str = "raw_snapshot",
    step_dt: float | None = None,
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
) -> PourEvidenceBinding:
    """Project a validated operation into exact rigid Pour evidence identities.

    Args:
        spec: Frozen normative TaskSpec.
        graph: Validated semantic graph and group lineage.
        validation: Matching source program/spec/graph digests and coverage.
        program: Complete serialized program.
        occurrences: Explicit compiled segment ID, display name, source path and
            invocation ID for each graph node. No ID is manufactured from names.
        resolved_operation: Plain configured-owner descriptor with object_id,
            reference_id, resource_id, object_native_uid/reference_native_uid and
            the compiled program_fingerprint for new raw evidence.
        predicate_id: Exact goal or milestone predicate.
        stage_node_id: Explicit reference-relative staging occurrence.
        pour_node_id: Explicit Pour occurrence.
        release_node_id: Explicit release occurrence.
        boundary_node_id: This operation's terminal boundary, including for goals.
        evidence_format: Raw snapshot or explicit historical replay contract.
        step_dt: Resolved recording cadence; required for raw qualification.
        episode_ids: Initialized run identities; required for raw qualification.
        num_envs: Ordered native environment row count.

    Returns:
        Owned normative binding. This adapter chooses no motion/geometry values
        and acquires no state; TaskSpec verifies the supported proof contract.
    """
    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, occurrences, resolved = (
        json_snapshot(value)
        for value in (graph, validation, program, occurrences, resolved_operation)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError(f"Frozen Pour projection has a different {key}.")
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("Pour projection requires complete declared coverage.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    matches = [(item, None) for item in normalized.goal if item.id == predicate_id] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Pour projection requires one exact predicate.")
    predicate, milestone_id = matches[0]
    selected = [
        item
        for item in coverage["goals" if milestone_id is None else "milestones"]
        if item["id"] == (predicate_id if milestone_id is None else milestone_id)
    ]
    if len(selected) != 1 or len(selected[0]["groups"]) != 1:
        raise ValueError("Pour projection requires one complete grounded operation.")
    group_id = selected[0]["groups"][0]
    groups = [item for item in graph["task_groups"] if item["id"] == group_id]
    if len(groups) != 1:
        raise ValueError("Pour operation identity is missing or ambiguous.")
    group = groups[0]
    if (
        group.get("task_type") != "E3"
        or not group["node_ids"]
        or boundary_node_id != group["node_ids"][-1]
        or any(
            node not in group["node_ids"]
            for node in (stage_node_id, pour_node_id, release_node_id)
        )
    ):
        raise ValueError("Pour projection has another operation or terminal boundary.")
    roles = {role.name: role for role in normalized.roles}
    for argument, field in (("subject", "object_id"), ("reference", "reference_id")):
        role_id = predicate.arguments[argument]
        if not roles[role_id].is_single_member or coverage["roles"][role_id] != [
            {"uid": resolved[field], "part_id": None}
        ]:
            raise ValueError(
                "Pour descriptor differs from its complete rigid-role binding."
            )
    lineage = [
        entry
        for entries in coverage["lineage"].values()
        for entry in entries
        if entry["group_id"] == group_id
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != resolved["object_id"]
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("Pour descriptor differs from its object lineage.")
    nodes = {item["id"]: item for item in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate graph nodes cannot identify Pour occurrences.")
    selected_occurrences = []
    for node_id in group["node_ids"]:
        node = nodes[node_id]
        descriptor = occurrences[node_id]
        item = program
        for component in descriptor["source_path"]:
            if type(component) not in (str, int):
                raise ValueError(
                    "Pour source paths require exact string/integer components."
                )
            item = item[component]
        if (
            node.get("task_type") != "E3"
            or node.get("task_instance_id") != group_id
            or item.get("name") != descriptor["segment_name"]
            or item["steps"]["call"] != node["call"]
        ):
            raise ValueError(
                "Compiled Pour occurrence differs from its semantic graph declaration."
            )
        selected_occurrences.append(
            {
                name: descriptor[name]
                for name in (
                    "segment_id",
                    "segment_name",
                    "source_path",
                    "invocation_id",
                )
            }
        )
    result = PourEvidenceBinding(
        task_spec_digest=validation["task_spec_digest"],
        program_digest=validation["program_digest"],
        predicate_id=predicate_id,
        operation_id=group_id,
        object_id=resolved["object_id"],
        reference_id=resolved["reference_id"],
        resource_id=resolved["resource_id"],
        object_native_uid=resolved["object_native_uid"],
        reference_native_uid=resolved["reference_native_uid"],
        occurrences=tuple(selected_occurrences),
        stage_segment_id=occurrences[stage_node_id]["segment_id"],
        pour_segment_id=occurrences[pour_node_id]["segment_id"],
        release_segment_id=occurrences[release_node_id]["segment_id"],
        boundary_segment_id=occurrences[boundary_node_id]["segment_id"],
        observation_kind="final" if milestone_id is None else "recorded",
        evidence_format=evidence_format,
        program_fingerprint=resolved.get("program_fingerprint"),
        step_dt=step_dt,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
    )
    evaluate_pour_evidence(normalized, result, {}, program=program)
    return result


def project_handover_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    predicate_id: str,
    transfer_node_id: str,
    boundary_node_id: str,
    step_dt: float,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
    program_fingerprint: str | None = None,
    object_native_uid: str | None = None,
    robot_uid: str | None = None,
    acceptance_criteria: HoldAcceptanceCriteria | Mapping[str, Any] | None = None,
) -> TaskEngineManipulationBinding:
    """Project a frozen E4 declaration without reading physical state.

    Args:
        spec: Frozen normative specification.
        graph: Semantic graph with exact grounded group ownership.
        validation: Matching graph/spec/program digests and complete coverage.
        program: Complete serialized compiled program source.
        operation: Explicit transfer/boundary descriptors and owner hold condition.
        predicate_id: One exact goal or milestone predicate.
        transfer_node_id: Explicit graph transfer node.
        boundary_node_id: Explicit graph terminal boundary node.
        step_dt: Original recording cadence.
        evidence_format: Explicit raw or historical input contract.
        episode_ids: Exact native episode identities.
        num_envs: Native row count.
        program_fingerprint: Actual compiled program fingerprint.
        object_native_uid: Resolved native object identity.
        robot_uid: Resolved native robot identity.
        acceptance_criteria: Explicit normative limits; otherwise use the complete frozen hold condition.

    Returns:
        Sealed TaskSpec HandOver evidence binding.
    """

    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, operation = (
        json_snapshot(value) for value in (graph, validation, program, operation)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError("HandOver projection has another frozen content identity.")
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("HandOver projection requires complete declared coverage.")
    matches = [(item, None) for item in normalized.goal if item.id == predicate_id] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("HandOver projection requires one exact predicate.")
    predicate, milestone = matches[0]
    selected = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected) != 1 or selected[0]["groups"] != [operation["operation_id"]]:
        raise ValueError("HandOver descriptor differs from its covered operation.")
    groups = [
        group
        for group in graph["task_groups"]
        if group["id"] == operation["operation_id"]
    ]
    if len(groups) != 1 or groups[0].get("task_type") != "E4":
        raise ValueError("HandOver requires an exact E4 group.")
    group = groups[0]
    role = next(
        role for role in normalized.roles if role.name == predicate.arguments["subject"]
    )
    if not role.is_single_member or coverage["roles"][role.name] != [
        {"uid": operation["object_id"], "part_id": None}
    ]:
        raise ValueError(
            "HandOver participant differs from its complete rigid-role binding."
        )
    lineage = [
        entry
        for entries in coverage["lineage"].values()
        for entry in entries
        if entry["group_id"] == group["id"]
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != operation["object_id"]
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("HandOver descriptor differs from its object lineage.")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate HandOver graph nodes.")
    for key, node_id in (
        ("transfer", transfer_node_id),
        ("boundary", boundary_node_id),
    ):
        descriptor = operation[key]
        node = nodes[node_id]
        item = program
        for component in descriptor["source_path"]:
            item = item[component]
        if (
            node_id not in group["node_ids"]
            or node.get("task_type") != "E4"
            or node.get("task_instance_id") != group["id"]
            or item.get("name") != descriptor["segment_name"]
            or item["steps"]["call"] != node["call"]
        ):
            raise ValueError("HandOver occurrence differs from its graph declaration.")
    if boundary_node_id != group["node_ids"][-1]:
        raise ValueError("HandOver boundary is not its terminal group occurrence.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    result = HandOverEvidenceBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        operation,
        "final" if milestone is None else "recorded",
        step_dt,
        query_kind=predicate.name,
        evidence_format=evidence_format,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
        program_fingerprint=program_fingerprint,
        object_native_uid=object_native_uid,
        robot_uid=robot_uid,
    )
    deployment = _acceptance(
        result,
        {
            "handover_complete": "handover_complete",
            "held": "receiver_hold",
            "orientation": "receiver_orientation",
        }[predicate.name],
        acceptance_criteria,
    )
    evaluate_manipulation_evidence(
        normalized, deployment.acceptance, {}, program=program
    )
    return deployment


def project_coordinated_hold_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    predicate_id: str,
    boundary_node_id: str,
    step_dt: float,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    num_envs: int = 1,
    program_fingerprint: str | None = None,
    object_native_uid: str | None = None,
    robot_uid: str | None = None,
    acceptance_criteria: HoldAcceptanceCriteria | Mapping[str, Any] | None = None,
) -> TaskEngineManipulationBinding:
    """Project a frozen E5 declaration without reading physical state.

    Args:
        spec: Frozen normative specification.
        graph: Semantic graph with exact grounded group ownership.
        validation: Matching graph/spec/program digests and complete coverage.
        program: Complete serialized compiled program source.
        operation: Explicit transfer/boundary descriptors and owner hold condition.
        predicate_id: One exact goal or milestone predicate.
        boundary_node_id: Explicit graph terminal boundary node.
        step_dt: Original recording cadence.
        evidence_format: Explicit raw or historical input contract.
        episode_ids: Exact native episode identities.
        num_envs: Native row count.
        program_fingerprint: Actual compiled program fingerprint.
        object_native_uid: Resolved native object identity.
        robot_uid: Resolved native robot identity.
        acceptance_criteria: Explicit normative limits; otherwise use the complete frozen hold condition.

    Returns:
        Sealed TaskSpec Coordinated hold evidence binding.
    """

    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, operation = (
        json_snapshot(value) for value in (graph, validation, program, operation)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError(
                "Coordinated hold projection has another frozen content identity."
            )
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError(
            "Coordinated hold projection requires complete declared coverage."
        )
    matches = [(item, None) for item in normalized.goal if item.id == predicate_id] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Coordinated hold projection requires one exact predicate.")
    predicate, milestone = matches[0]
    selected = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected) != 1 or selected[0]["groups"] != [operation["operation_id"]]:
        raise ValueError(
            "Coordinated hold descriptor differs from its covered operation."
        )
    groups = [
        group
        for group in graph["task_groups"]
        if group["id"] == operation["operation_id"]
    ]
    if len(groups) != 1 or groups[0].get("task_type") != "E5":
        raise ValueError("Coordinated hold requires an exact E5 group.")
    group = groups[0]
    role = next(
        role for role in normalized.roles if role.name == predicate.arguments["subject"]
    )
    if not role.is_single_member or coverage["roles"][role.name] != [
        {"uid": operation["object_id"], "part_id": None}
    ]:
        raise ValueError(
            "Coordinated hold participant differs from its complete rigid-role binding."
        )
    lineage = [
        entry
        for entries in coverage["lineage"].values()
        for entry in entries
        if entry["group_id"] == group["id"]
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != operation["object_id"]
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("Coordinated hold descriptor differs from its object lineage.")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate Coordinated hold graph nodes.")
    for key, node_id in (("boundary", boundary_node_id),):
        descriptor = operation[key]
        node = nodes[node_id]
        item = program
        for component in descriptor["source_path"]:
            item = item[component]
        if (
            node_id not in group["node_ids"]
            or node.get("task_type") != "E5"
            or node.get("task_instance_id") != group["id"]
            or item.get("name") != descriptor["segment_name"]
            or item["steps"]["call"] != node["call"]
        ):
            raise ValueError(
                "Coordinated hold occurrence differs from its graph declaration."
            )
    if group["node_ids"] != [boundary_node_id]:
        raise ValueError(
            "Coordinated hold boundary is not its terminal group occurrence."
        )
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    result = CoordinatedHoldEvidenceBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        operation,
        "final" if milestone is None else "recorded",
        step_dt,
        evidence_format=evidence_format,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
        program_fingerprint=program_fingerprint,
        object_native_uid=object_native_uid,
        robot_uid=robot_uid,
    )
    deployment = _acceptance(result, "coordinated_hold", acceptance_criteria)
    evaluate_manipulation_evidence(
        normalized, deployment.acceptance, {}, program=program
    )
    return deployment


def project_coordinated_transport_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    operation: Mapping[str, Any],
    *,
    predicate_id: str,
    transport_node_id: str,
    boundary_node_id: str,
    step_dt: float,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    sampled_contact_identity: Mapping[str, Any] | None = None,
) -> TaskEngineManipulationBinding:
    """Project a recorded E5 acquisition event without treating it as final holding.

    Args:
        spec: Frozen normative specification.
        graph: Validated semantic graph and operation groups.
        validation: Exact spec/graph/program content identities and full coverage.
        program: Complete serialized program.
        occurrences: Actual compiled descriptors keyed by graph node IDs.
        operation: Explicit transport participant/reference/target declaration.
        predicate_id: One recorded held-by-both milestone predicate.
        transport_node_id: Exact coordinated transport node in its group.
        boundary_node_id: Terminal group boundary associated with the event report.
        step_dt: Original control recording cadence.
        evidence_format: Explicit raw or historical contract.
        episode_ids: Native episode identity, required for raw observations.
        sampled_contact_identity: Actual Lab collector identity, required for raw data.

    Returns:
        Data-only single-row event binding; no physical contact or pose is read.
    """
    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, occurrences, operation = (
        json_snapshot(value)
        for value in (graph, validation, program, occurrences, operation)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError(
                "Transport event projection has another frozen content identity."
            )
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("Transport event requires complete declared coverage.")
    matches = [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == predicate_id
    ]
    if len(matches) != 1 or any(item.id == predicate_id for item in normalized.goal):
        raise ValueError("Transport acquisition projects only a recorded milestone.")
    predicate, milestone = matches[0]
    selected = [item for item in coverage["milestones"] if item["id"] == milestone]
    if len(selected) != 1 or selected[0]["groups"] != [operation["operation_id"]]:
        raise ValueError("Transport acquisition differs from its grounded operation.")
    groups = [
        group
        for group in graph["task_groups"]
        if group["id"] == operation["operation_id"]
    ]
    if (
        len(groups) != 1
        or groups[0].get("task_type") != "E5"
        or groups[0]["node_ids"][-1] != boundary_node_id
        or transport_node_id not in groups[0]["node_ids"]
    ):
        raise ValueError(
            "Transport acquisition has another group or terminal boundary."
        )
    group = groups[0]
    role = next(
        role for role in normalized.roles if role.name == predicate.arguments["subject"]
    )
    if not role.is_single_member or coverage["roles"][role.name] != [
        {"uid": operation["object_id"], "part_id": None}
    ]:
        raise ValueError(
            "Transport acquisition differs from its complete rigid-role binding."
        )
    lineage = [
        item
        for items in coverage["lineage"].values()
        for item in items
        if item["group_id"] == group["id"]
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != operation["object_id"]
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("Transport acquisition differs from its object lineage.")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Transport projection has duplicate graph nodes.")
    required = []
    for node_id in group["node_ids"]:
        node = nodes[node_id]
        descriptor = occurrences[node_id]
        item = program
        for component in descriptor["source_path"]:
            item = item[component]
        if (
            node.get("task_type") != "E5"
            or node.get("task_instance_id") != group["id"]
            or item.get("name") != descriptor["segment_name"]
            or item["steps"]["call"] != node["call"]
        ):
            raise ValueError(
                "Transport compiled occurrence differs from its graph declaration."
            )
        required.append(
            dict(
                segment_id=descriptor["segment_name"],
                source_path=descriptor["source_path"],
            )
        )
    if any(
        operation["transport"][key] != occurrences[transport_node_id][key]
        for key in ("segment_id", "segment_name", "source_path", "invocation_id")
    ):
        raise ValueError(
            "Transport operation differs from the selected compiled invocation."
        )
    result = CoordinatedTransportEvidenceBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        operation,
        tuple(required),
        occurrences[boundary_node_id]["segment_name"],
        tuple(occurrences[boundary_node_id]["source_path"]),
        step_dt,
        evidence_format,
        episode_ids=None if episode_ids is None else tuple(episode_ids),
        sampled_contact_identity=sampled_contact_identity,
    )
    deployment = _acceptance(result, "coordinated_acquisition", None)
    evaluate_manipulation_evidence(
        normalized, deployment.acceptance, {}, program=program
    )
    return deployment


def project_control_evidence_binding(
    spec: TaskSpec | Mapping[str, Any],
    graph: Mapping[str, Any],
    validation: Mapping[str, Any],
    program: Mapping[str, Any],
    occurrences: Mapping[str, Mapping[str, Any]],
    route_payload: Mapping[str, Any],
    *,
    predicate_id: str,
    control_kind: str,
    terminal_node_id: str,
    preset: str,
    evidence_format: str = "raw_snapshot",
    episode_ids: Sequence[int] | None = None,
    control_identity: Mapping[str, Any] | None = None,
    policy_occurrences: Sequence[Mapping[str, Any]] = (),
    num_envs: int = 1,
) -> ControlEvidenceBinding:
    """Project exact control declarations without sampling or selecting criteria.

    Args:
        spec: Frozen normative TaskSpec.
        graph: Validated graph and grounded group lineage.
        validation: Matching spec/graph/program digests and complete coverage.
        program: Complete serialized program.
        occurrences: Actual compiled descriptors keyed by graph node identity.
        route_payload: Plain configured owner's original control route.
        predicate_id: Exact Press/Twist goal or recorded milestone predicate.
        control_kind: Explicit press or twist family.
        terminal_node_id: Original terminal cleanup/park occurrence.
        preset: Actual named terminal post-policy.
        evidence_format: Explicit raw acquisition or historical replay contract.
        episode_ids: Native episode identities, required for raw acquisition.
        control_identity: Actual runtime instance/native scope for raw evidence.
        policy_occurrences: Exact compiled policy selectors from the configured owner.
        num_envs: Native row count; raw source controls require one.

    Returns:
        Sealed TaskSpec declaration, without a live factory or physical lookup.
    """
    normalized = TaskSpec.from_dict(spec)
    graph, validation, program, occurrences, route = (
        json_snapshot(value)
        for value in (graph, validation, program, occurrences, route_payload)
    )
    for key, value in (
        ("task_spec_digest", normalized.to_dict()),
        ("graph_digest", graph),
        ("program_digest", program),
    ):
        if validation.get(key) != canonical_hash(value):
            raise ValueError("Control projection has another frozen content identity.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("Control projection requires a positive native row count.")
    coverage = validation["coverage"]
    if coverage.get("status") != "covered" or coverage.get("issues"):
        raise ValueError("Control projection requires complete declared coverage.")
    matches = [(item, None) for item in normalized.goal if item.id == predicate_id] + [
        (item, milestone.id)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Control projection requires one exact predicate.")
    predicate, milestone = matches[0]
    selected = [
        item
        for item in coverage["goals" if milestone is None else "milestones"]
        if item["id"] == (predicate_id if milestone is None else milestone)
    ]
    if len(selected) != 1 or selected[0]["groups"] != [route["operation_id"]]:
        raise ValueError("Control projection differs from its grounded operation.")
    groups = [
        group for group in graph["task_groups"] if group["id"] == route["operation_id"]
    ]
    family = (
        "E9" if control_kind == "press" else "E8" if control_kind == "twist" else None
    )
    if (
        len(groups) != 1
        or family is None
        or groups[0].get("task_type") != family
        or groups[0]["node_ids"][-1] != terminal_node_id
    ):
        raise ValueError(
            "Control projection has another source family or terminal boundary."
        )
    group = groups[0]
    uid = route["binding"]["object_id"]
    role = next(
        role for role in normalized.roles if role.name == predicate.arguments["subject"]
    )
    if not role.is_single_member or coverage["roles"][role.name] != [
        {"uid": uid, "part_id": None}
    ]:
        raise ValueError(
            "Control projection differs from its complete parent-object role binding."
        )
    lineage = [
        item
        for entries in coverage["lineage"].values()
        for item in entries
        if item["group_id"] == group["id"]
    ]
    if (
        len(lineage) != 1
        or lineage[0]["object_uid"] != uid
        or lineage[0].get("object_part_id") is not None
    ):
        raise ValueError("Control projection has another object lineage.")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Control graph nodes are ambiguous.")
    required = []
    declared_names = {}
    for node_id in group["node_ids"]:
        node = nodes[node_id]
        descriptor = occurrences[node_id]
        item = program
        for component in descriptor["source_path"]:
            item = item[component]
        if (
            node.get("task_type") != family
            or node.get("task_instance_id") != group["id"]
            or item.get("name") != descriptor["segment_name"]
            or item["steps"]["call"] != node["call"]
        ):
            raise ValueError(
                "Control occurrence differs from its exact compiled declaration."
            )
        required.append(
            dict(
                segment_id=descriptor["segment_name"],
                source_path=descriptor["source_path"],
            )
        )
        declared_names[descriptor["segment_name"]] = descriptor
    for selector in policy_occurrences:
        descriptor = declared_names[selector["segment_name"]]
        if (
            selector["segment_id"] != descriptor["segment_id"]
            or selector["source_path"] != descriptor["source_path"]
        ):
            raise ValueError(
                "Control policy selector has another compiled segment identity."
            )
    terminal = occurrences[terminal_node_id]
    result = ControlEvidenceBinding(
        validation["task_spec_digest"],
        validation["program_digest"],
        predicate_id,
        uid,
        control_kind,
        "final" if milestone is None else "recorded",
        dict(
            route_payload=route,
            terminal_segment_id=terminal["segment_name"],
            terminal_source_path=terminal["source_path"],
            preset=preset,
            required_execution=required,
        ),
        evidence_format,
        env_ids=tuple(range(num_envs)),
        episode_ids=None if episode_ids is None else tuple(episode_ids),
        control_identity=control_identity,
        policy_occurrences=tuple(policy_occurrences),
    )
    evaluate_control_evidence(normalized, result, {}, program=program)
    return result
