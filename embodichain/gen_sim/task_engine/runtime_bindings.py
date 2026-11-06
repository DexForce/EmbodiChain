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

"""Project sealed semantic declarations onto actual episode evidence owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any, TYPE_CHECKING
import yaml

from embodichain.gen_sim.task_spec import canonical_hash
from .bundle_io import BundleSnapshot
from .bundle_spec import decode_bundle_task_spec
from .semantic_graph_validation import validate_graph_coverage
from .evidence import (
    project_condition_evidence_binding,
    project_condition_evidence_group_binding,
    project_configured_placement_evidence_binding,
    project_configured_placement_evidence_group_binding,
    project_control_evidence_binding,
    project_coordinated_hold_evidence_binding,
    project_coordinated_transport_evidence_binding,
    project_handover_evidence_binding,
    project_pour_evidence_binding,
)

if TYPE_CHECKING:
    from .bundle_deployment import LoadedTaskProgramBundle
    from .runtime_evidence import TaskEngineEvidenceSession
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult

__all__ = ["BundleEvidenceProjector"]


def _one(values: Sequence[Any], description: str) -> Any:
    if len(values) != 1:
        raise ValueError("Evidence requires one exact " + description + ".")
    return values[0]


class BundleEvidenceProjector:
    """Bind existing normative proof contracts to a new configured execution.

    Args:
        bundle: Publicly verified immutable declaration bundle.
        step_dt: Actual environment control interval.
        robot_uid: Actual configured native robot identity.

    The projector reads declarations and compiled occurrences only. Lab acquires
    measurements and TaskSpec selects their verdicts. Unsupported, incomplete or
    ambiguous proof declarations remain explicitly unavailable per predicate;
    they never fall back to a weaker physical contract.
    """

    def __init__(
        self, bundle: LoadedTaskProgramBundle, *, step_dt: float, robot_uid: str
    ) -> None:
        from .bundle_deployment import LoadedTaskProgramBundle
        from embodichain.lab.task_program.integrations import task_program_fingerprint

        if type(bundle) is not LoadedTaskProgramBundle:
            raise TypeError("Evidence projection requires a publicly loaded bundle.")
        snapshot = BundleSnapshot(bundle.snapshot.identity, bundle.snapshot.files)
        identity = snapshot.identity.to_dict()
        self._fingerprint = task_program_fingerprint(bundle.program)
        if self._fingerprint != identity["program_fingerprint"]:
            raise ValueError("Evidence projection has another compiled program.")
        self._registration = bundle.deployment.integration.registration
        self._registration.assert_unchanged()
        from embodichain.lab.task_program.integrations import BoundPostPolicyPresets

        self._post_policy_presets = (
            BoundPostPolicyPresets(
                bundle.program, self._registration.post_policy_bindings
            )
            if self._registration.post_policy_bindings
            else None
        )
        if (
            canonical_hash(self._registration.fingerprint_payload())
            != identity["registration_fingerprint"]
        ):
            raise ValueError("Evidence projection has another runtime registration.")
        self._spec = decode_bundle_task_spec(
            json.loads(snapshot.files["task_spec.json"])
        )
        self._graph = json.loads(snapshot.files["semantic_task_graph.json"])
        self._program = yaml.safe_load(snapshot.files["task_program/program.yaml"])
        self._integration = yaml.safe_load(
            snapshot.files["task_program/integration.yaml"]
        )
        self._step_dt, self._robot_uid = step_dt, robot_uid
        self._nodes = {node["id"]: node for node in self._graph["nodes"]}
        self._groups = {group["id"]: group for group in self._graph["task_groups"]}
        self._occurrences = {}
        self._compiled_segments = {}
        for segment in bundle.program.iter_segments():
            if len(segment.calls) != 1 or segment.name in self._occurrences:
                raise ValueError(
                    "Evidence requires unique single-call execution occurrences."
                )
            self._compiled_segments[segment.name] = segment
            self._occurrences[segment.name] = dict(
                segment_name=segment.name,
                segment_id=segment.segment_id,
                source_path=list(segment.source_path),
                invocation_id=f"{bundle.program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}",
            )
        scene = self._registration.scene_binding
        self._native = {
            v.entity_id: v.simulation_uid
            for v in (*scene.rigid_objects, *scene.articulations)
        }
        self._validation = None
        self._coverage_error = "No complete sealed semantic coverage declaration."
        self._unavailable: dict[str, str] = {}
        self._control_ids = tuple(
            g["id"]
            for g in self._graph["task_groups"]
            if g["task_type"] in {"E8", "E9"}
        )
        if (
            self._spec is not None
            and "semantic_graph_validation.json" in snapshot.files
        ):
            audit = json.loads(snapshot.files["semantic_graph_validation.json"])
            source = audit["source"]
            coverage = validate_graph_coverage(
                self._graph,
                self._spec,
                source["candidate"],
                source["role_bindings"],
                audit["coverage"]["lineage"],
            )
            if (
                coverage != audit["coverage"]
                or coverage["status"] != "covered"
                or coverage.get("issues")
            ):
                raise ValueError("Sealed evidence coverage is incomplete or stale.")
            self._validation = dict(
                task_spec_digest=canonical_hash(self._spec.to_dict()),
                graph_digest=canonical_hash(self._graph),
                program_digest=canonical_hash(self._program),
                coverage=coverage,
            )

    @property
    def control_operation_ids(self) -> tuple[str, ...]:
        """Return exact declared control instances to acquire before the first step."""
        return self._control_ids

    @property
    def unavailable(self) -> dict[str, str]:
        """Return detached reasons for proof declarations that cannot be projected."""
        return dict(self._unavailable)

    def prepare(self, session: TaskEngineEvidenceSession) -> list[dict[str, Any]]:
        """Project recorded predicates before execution can cross their boundaries.

        Args:
            session: Actual selected bridge/episode evidence owner at step zero.

        Returns:
            Owned raw recorded bindings; missing scopes are recorded separately.
        """
        return self._project(session, recorded=True)

    def finish(
        self, session: TaskEngineEvidenceSession, result: DemoEpisodeResult
    ) -> list[dict[str, Any]]:
        """Project final goals using the actual final boundary after execution.

        Args:
            session: Same actual episode owner that captured completed boundaries.
            result: Original executor result; no success field is rewritten.

        Returns:
            Owned final bindings without replacing historical predicates.
        """
        del result
        return self._project(session, recorded=False)

    def _project(self, session, *, recorded):
        if self._spec is None:
            return []
        predicates = (
            [
                (predicate, milestone.id)
                for milestone in self._spec.milestones
                for predicate in milestone.achieve
            ]
            if recorded
            else [(predicate, None) for predicate in self._spec.goal]
        )
        result = []
        for predicate, milestone in predicates:
            try:
                if self._validation is None:
                    raise ValueError(self._coverage_error)
                if not recorded and session.final_boundary is None:
                    raise ValueError("No actual final execution boundary was observed.")
                value = self._binding(session, predicate, milestone)
                result.append(value.to_dict())
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                StopIteration,
            ) as error:
                reason = str(error) or type(error).__name__
                self._unavailable[predicate.id] = reason
                session.record_unavailable(predicate.id, reason)
        return result

    def _select_group(self, predicate, milestone):
        coverage = self._validation["coverage"]
        selected = _one(
            [
                v
                for v in coverage["milestones" if milestone else "goals"]
                if v["id"] == (milestone or predicate.id)
            ],
            "covered predicate",
        )
        group_id = _one(selected["groups"], "fully bound semantic group")
        return self._groups[group_id]

    def _declared(self, node_id):
        result = self._program
        for key in self._occurrences[node_id]["source_path"]:
            result = result[key]
        return result

    def _policies(self, group, *, entity=None, kind=None):
        result = []
        for node_id in group["node_ids"]:
            for policy in self._declared(node_id).get("post", []):
                if policy["kind"] != "wait_stable":
                    continue
                cfg = self._registration.settle_presets[policy["preset"]]
                if (entity is None or getattr(cfg, "entity", None) == entity) and (
                    kind is None or getattr(cfg, "kind", None) == kind
                ):
                    result.append((node_id, policy["preset"], cfg))
        return result

    def _descriptor(self, node_id):
        return deepcopy(self._occurrences[node_id])

    def _transfer(self, group):
        transfer = _one(
            [
                name
                for name in group["node_ids"]
                if self._nodes[name]["call"]["kind"] == "hand_over"
            ],
            "handover occurrence",
        )
        call = self._nodes[transfer]["call"]
        _, preset, condition = _one(
            self._policies(group, entity=call["object"], kind="hold"),
            "handover hold policy",
        )
        return dict(
            operation_id=group["id"],
            object_id=call["object"],
            source_resource=call["resources"]["source"],
            destination_resource=call["resources"]["destination"],
            transfer=self._descriptor(transfer),
            boundary=self._descriptor(group["node_ids"][-1]),
            hold_condition=condition.to_condition_dict(),
            hold_preset=preset,
        )

    def _transport(self, group):
        transport = _one(
            [
                name
                for name in group["node_ids"]
                if self._nodes[name]["call"].get("call_id")
                == "simulation.coordinated_transport"
            ],
            "coordinated transport occurrence",
        )
        call = self._nodes[transport]["call"]
        args = call["arguments"]
        return dict(
            operation_id=group["id"],
            object_id=args["object"],
            reference_id=args["reference"],
            target_id=args["target"],
            resources=dict(call["resources"]),
            transport=self._descriptor(transport),
        )

    def _count_state(self, session, predicate, milestone):
        coverage = self._validation["coverage"]
        selected = _one(
            [
                v
                for v in coverage["milestones" if milestone else "goals"]
                if v["id"] == (milestone or predicate.id)
            ],
            "covered finite predicate",
        )
        members = []
        for group_id in selected["groups"]:
            group = self._groups[group_id]
            lineage = _one(
                [
                    v
                    for entries in coverage["lineage"].values()
                    for v in entries
                    if v["group_id"] == group_id
                ],
                "finite member lineage",
            )
            entity = lineage["object_uid"]
            family = group["task_type"]
            matches = self._policies(
                group, entity=entity, kind="upright" if family in {"E1", "E2"} else None
            )
            if family == "E6":
                source = _one(
                    [
                        value.source
                        for factory in self._registration.registered_semantic_lowerer_factories
                        if factory.call_id == "gen_sim.articulation_slide"
                        for value in factory.bindings
                        if value.source.object_id == entity
                        and value.source.part_id == lineage.get("object_part_id")
                    ],
                    "declared articulation part",
                )
                matches = [
                    value
                    for value in matches
                    if getattr(value[2], "joint", None) == source.joint
                ]
            policy_node, preset, cfg = _one(matches[-1:], "finite member state policy")
            members.append(
                dict(
                    group_id=group_id,
                    state=dict(
                        entity_id=entity,
                        native_uid=self._native[entity],
                        part_id=lineage.get("object_part_id"),
                        condition=cfg.to_condition_dict(),
                    ),
                    boundary_node_id=(
                        group["node_ids"][-1]
                        if milestone
                        else session.final_boundary["segment_name"]
                    ),
                    policy_node_id=policy_node,
                    stability_preset=preset,
                )
            )
        return project_condition_evidence_group_binding(
            self._spec,
            self._graph,
            self._validation,
            self._program,
            self._occurrences,
            members,
            predicate_id=predicate.id,
            episode_ids=session.episode_ids,
            evidence_kind=(
                "joint_endpoint"
                if predicate.arguments.get("task_type") == "E6"
                else "orientation"
            ),
        )

    def _binding(self, session, predicate, milestone):
        role = next(
            (
                value
                for value in self._spec.roles
                if value.name == predicate.arguments.get("subject")
            ),
            None,
        )
        if (
            role is not None
            and role.quantifier == "count"
            and role.count > 1
            and predicate.arguments.get("task_type") in {"E1", "E2", "E6"}
            and predicate.name
            in {"object_upright", "orientation", "articulation_joint_near"}
        ):
            return self._count_state(session, predicate, milestone)
        if (
            role is not None
            and role.quantifier == "count"
            and role.count > 1
            and predicate.arguments.get("task_type") == "E1"
            and predicate.name == "semantic_goal"
        ):
            return self._count_placement(session, predicate, milestone)
        group = self._select_group(predicate, milestone)
        family, terminal = group["task_type"], group["node_ids"][-1]
        boundary = terminal if milestone else session.final_boundary["segment_name"]
        common = dict(predicate_id=predicate.id, episode_ids=session.episode_ids)
        inputs = (self._spec, self._graph, self._validation, self._program)
        if family in {"E8", "E9"}:
            selectors = []
            for name in group["node_ids"]:
                segment = self._compiled_segments[name]
                for index, policy in enumerate(segment.post_policies):
                    if not hasattr(policy.cfg, "preset"):
                        continue
                    preset_id = policy.cfg.preset
                    if self._post_policy_presets is not None:
                        preset_id = (
                            self._post_policy_presets.preset_for(
                                policy, segment=segment
                            )
                            or preset_id
                        )
                    cfg = self._registration.settle_presets[preset_id]
                    if getattr(cfg, "operation_id", None) == group["id"]:
                        selectors.append(
                            dict(
                                segment_name=segment.name,
                                segment_id=segment.segment_id,
                                source_path=list(segment.source_path),
                                policy_index=index,
                                preset=policy.cfg.preset,
                                phase=cfg.phase,
                                terminal=cfg.terminal,
                            )
                        )
            selected = _one(
                [v for v in selectors if v["terminal"]], "terminal control policy"
            )
            return project_control_evidence_binding(
                *inputs,
                self._occurrences,
                session.control_route_payload(group["id"]),
                control_kind="press" if family == "E9" else "twist",
                terminal_node_id=terminal,
                preset=selected["preset"],
                control_identity=session.control_identity(group["id"]),
                policy_occurrences=selectors,
                **common,
            )
        if predicate.name in {
            "object_upright",
            "orientation",
            "articulation_joint_near",
        } and family in {"E1", "E2", "E6"}:
            member = _one(
                self._validation["coverage"]["roles"][predicate.arguments["subject"]],
                "state participant",
            )
            entity = member["uid"]
            matches = self._policies(
                group, entity=entity, kind="upright" if family in {"E1", "E2"} else None
            )
            policy_node, preset, cfg = _one(matches[-1:], "state policy")
            return project_condition_evidence_binding(
                *inputs,
                self._occurrences,
                dict(
                    entity_id=entity,
                    native_uid=self._native[entity],
                    part_id=member.get("part_id"),
                    condition=cfg.to_condition_dict(),
                ),
                boundary_node_id=boundary,
                policy_node_id=policy_node,
                stability_preset=preset,
                evidence_kind=("joint_endpoint" if family == "E6" else "orientation"),
                **common,
            )
        if family == "E3" and predicate.name == "pour_motion":
            pour = _one(
                [
                    name
                    for name in group["node_ids"]
                    if self._nodes[name]["call"].get("call_id") == "simulation.pour"
                ],
                "Pour occurrence",
            )
            stage = _one(
                [
                    name
                    for name in group["node_ids"]
                    if self._nodes[name]["call"].get("call_id")
                    == "simulation.move_held_object"
                    and "reference" in self._nodes[name]["call"]["arguments"]
                ],
                "Pour stage",
            )
            release = _one(
                [
                    name
                    for name in group["node_ids"]
                    if self._nodes[name]["call"]["kind"] == "place"
                ],
                "Pour release",
            )
            subject = self._nodes[pour]["call"]["arguments"]["object"]
            reference = self._nodes[stage]["call"]["arguments"]["reference"]
            return project_pour_evidence_binding(
                *inputs,
                self._occurrences,
                dict(
                    object_id=subject,
                    reference_id=reference,
                    resource_id=self._nodes[pour]["call"]["resources"]["primary"],
                    object_native_uid=self._native[subject],
                    reference_native_uid=self._native[reference],
                    program_fingerprint=self._fingerprint,
                ),
                stage_node_id=stage,
                pour_node_id=pour,
                release_node_id=release,
                boundary_node_id=terminal,
                step_dt=self._step_dt,
                **common,
            )
        if family == "E4" and predicate.name in {
            "held",
            "orientation",
            "handover_complete",
        }:
            operation = self._transfer(group)
            return project_handover_evidence_binding(
                *inputs,
                operation,
                transfer_node_id=operation["transfer"]["segment_name"],
                boundary_node_id=terminal,
                step_dt=self._step_dt,
                program_fingerprint=self._fingerprint,
                object_native_uid=self._native[operation["object_id"]],
                robot_uid=self._robot_uid,
                **common,
            )
        if family == "E5" and predicate.name == "held_by_both_grippers":
            if any(
                self._nodes[name]["call"].get("call_id")
                == "simulation.coordinated_transport"
                for name in group["node_ids"]
            ):
                operation = self._transport(group)
                return project_coordinated_transport_evidence_binding(
                    *inputs,
                    self._occurrences,
                    operation,
                    transport_node_id=operation["transport"]["segment_name"],
                    boundary_node_id=terminal,
                    step_dt=self._step_dt,
                    sampled_contact_identity=session.contact_identity,
                    **common,
                )
            call = self._nodes[terminal]["call"]
            args = call["arguments"]
            _, preset, cfg = _one(
                self._policies(group, entity=args["object"], kind="hold"),
                "coordinated hold policy",
            )
            operation = dict(
                operation_id=group["id"],
                object_id=args["object"],
                resources=dict(call["resources"]),
                target_id=args["target"],
                boundary=self._descriptor(terminal),
                hold_condition=cfg.to_condition_dict(),
                hold_preset=preset,
            )
            return project_coordinated_hold_evidence_binding(
                *inputs,
                operation,
                boundary_node_id=terminal,
                step_dt=self._step_dt,
                program_fingerprint=self._fingerprint,
                object_native_uid=self._native[args["object"]],
                robot_uid=self._robot_uid,
                **common,
            )
        if predicate.name == "semantic_goal":
            return self._placement(session, predicate, milestone, group, boundary)
        raise ValueError(
            "No source physical proof contract for this predicate declaration."
        )

    def _placement(self, session, predicate, milestone, group, boundary):
        declaration, occurrences = self._placement_declaration(
            session, predicate, milestone, group, boundary
        )
        return project_configured_placement_evidence_binding(
            self._spec,
            self._graph,
            self._validation,
            self._program,
            occurrences,
            predicate_id=predicate.id,
            observation_kind="recorded" if milestone else "final",
            episode_ids=session.episode_ids,
            **declaration,
        )

    def _count_placement(self, session, predicate, milestone):
        coverage = self._validation["coverage"]
        selected = _one(
            [
                value
                for value in coverage["milestones" if milestone else "goals"]
                if value["id"] == (milestone or predicate.id)
            ],
            "covered placement count",
        )
        members = []
        occurrences = deepcopy(self._occurrences)
        for group_id in selected["groups"]:
            group = self._groups[group_id]
            boundary = (
                group["node_ids"][-1]
                if milestone
                else session.final_boundary["segment_name"]
            )
            declaration, annotated = self._placement_declaration(
                session, predicate, milestone, group, boundary
            )
            occurrences[declaration["release_node_id"]] = annotated[
                declaration["release_node_id"]
            ]
            members.append(dict(group_id=group_id, **declaration))
        return project_configured_placement_evidence_group_binding(
            self._spec,
            self._graph,
            self._validation,
            self._program,
            occurrences,
            members,
            predicate_id=predicate.id,
            episode_ids=session.episode_ids,
        )

    def _placement_declaration(self, session, predicate, milestone, group, boundary):
        coverage = self._validation["coverage"]
        subjects = coverage["roles"][predicate.arguments["subject"]]
        if len(subjects) == 1:
            subject = subjects[0]["uid"]
        else:
            member = _one(
                [
                    value
                    for values in coverage["lineage"].values()
                    for value in values
                    if value["group_id"] == group["id"]
                ],
                "placement member lineage",
            )
            subject = member["object_uid"]
        reference = _one(
            coverage["roles"][predicate.arguments["reference"]], "placement reference"
        )
        relation = predicate.arguments["relation"]
        _, preset, cfg = _one(
            self._policies(group, entity=subject)[-1:], "placement policy"
        )
        condition = cfg.to_condition_dict()
        release = _one(
            [
                name
                for name in group["node_ids"]
                if self._nodes[name]["call"]["kind"] == "place"
                or self._nodes[name]["call"].get("call_id")
                in {
                    "simulation.place_relative",
                    "gen_sim.place_upright",
                    "gen_sim.stack_place",
                    "simulation.coordinated_transport",
                }
            ],
            "placement release",
        )
        call = self._nodes[release]["call"]
        occurrences = deepcopy(self._occurrences)
        occurrences[release].update(
            object_id=subject,
            reference_id=reference["uid"],
            resource_id=next(iter(call["resources"].values())),
        )
        scope, extra = "native_contact_support", {}
        if getattr(cfg, "support_contract_revision", None) == 1:
            scope = "native_supported_placement"
        elif group["task_type"] == "E5":
            scope = "native_stable_support"
            extra.update(
                operation_evidence=dict(
                    family="E5", operation=self._transport(group), step_dt=self._step_dt
                ),
                sampled_contact_identity=session.contact_identity,
            )
        elif relation == "inside":
            scope = "measured_planar_landing_region"
            extra["planar_geometry"] = dict(
                exact_silhouette=True,
                program_fingerprint=self._fingerprint,
                object_native_uid=self._native[subject],
                reference_native_uid=self._native[reference["uid"]],
            )
        elif relation != "on":
            routes = [
                route
                for lowerer in self._integration["runtime_services"][
                    "registered_semantic_lowerers"
                ]
                if lowerer["kind"] == "place_relative"
                for route in lowerer["routes"]
                if route["object_id"] == subject
                and route["reference_entity_id"] == reference["uid"]
                and route.get("reference_part") == reference.get("part_id")
                and route["relation"] == relation
            ]
            route = _one(routes, "directional placement route")
            scope = (
                "bound_link_world_axes"
                if route.get("reference_link_id")
                else "bound_rigid_world_axes"
            )
            link = next(
                (
                    v
                    for v in self._registration.scene_binding.links
                    if v.entity_id == route.get("reference_link_id")
                ),
                None,
            )
            extra["directional_geometry"] = dict(
                relation=relation,
                program_fingerprint=self._fingerprint,
                object_native_uid=self._native[subject],
                reference_native_uid=self._native[reference["uid"]],
                table_id="table",
                table_native_uid=self._native["table"],
                reference_part=reference.get("part_id"),
                reference_link_id=route.get("reference_link_id"),
                native_link_name=None if link is None else link.native_link_name,
                world_displacement=route["world_displacement"],
            )
        elif (
            predicate.arguments.get("task_type") == "E1"
            and "orientation_goal" not in predicate.arguments
            and call.get("call_id") == "simulation.place_relative"
            and call["arguments"].get("reference") == "table"
            and call["arguments"].get("relation") == "on"
            and set(call["arguments"])
            <= {"object", "reference", "relation", "stage_id"}
            and condition.get("kind") == "supported_region"
        ):
            scope = "native_supported_landing_region"
        return (
            dict(
                condition=condition,
                release_node_id=release,
                boundary_node_id=boundary,
                evidence_scope=scope,
                stability_preset=preset,
                **extra,
            ),
            occurrences,
        )
