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

"""Bind Task Agent candidates to a redacted, authoritative scene inventory."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.task_engine.orchestration.grounding import (
    GroundingCaller,
    ground_scene_references,
)
from embodichain.gen_sim.task_engine.orchestration.scene_inventory import (
    SceneInventory,
    validate_source_compatibility,
    validate_target_compatibility,
)
from embodichain.gen_sim.task_engine.orchestration.source_scene import (
    PreparedScene,
    prepare_scene,
    resolve_source_scene,
)
from embodichain.gen_sim.task_engine import TaskCandidate, TaskCandidateSet
from embodichain.gen_sim.task_engine.interpretation import (
    _default_instruction_caller,
)
from embodichain.gen_sim.task_engine.scene import (
    ConservativeSceneGraph,
    SceneEngineV1Adapter,
    StaticSceneManifest,
    build_conservative_scene_graph,
    validate_static_scene_manifest,
)
from embodichain.gen_sim.task_engine.scene.final_inspection import (
    apply_final_inspection,
    validate_final_scene_inspection,
)

from .contracts import (
    BINDING_REPORT_SCHEMA,
    ROLE_BINDINGS_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    BindingReport,
    RoleBindings,
    SceneManifest,
    validate_binding_report,
    validate_role_bindings,
    validate_scene_manifest,
    validate_task_candidate,
    validate_task_candidate_set,
)
from .scene_source import (
    SceneSourceRef,
    fingerprint_scene_source,
    scene_revision_id,
)

__all__ = [
    "Adjudicator",
    "CandidateSelection",
    "SceneAdaptation",
    "SceneAdapter",
    "SceneAdapterProtocolError",
]


Adjudicator = Callable[..., Mapping[str, Any]]

_REDACTED_KEYS = frozenset(
    {
        "absolute_position",
        "bbox",
        "bboxes",
        "bounding_box",
        "center",
        "centroid",
        "coordinates",
        "dimensions",
        "extrinsics",
        "grasp_pose",
        "init_local_pose",
        "init_pos",
        "init_rot",
        "intrinsics",
        "joint_positions",
        "joints",
        "keypoint",
        "keypoints",
        "location",
        "matrix",
        "pose",
        "position",
        "position_xyz",
        "qpos",
        "quaternion",
        "rotation",
        "scale",
        "target_pose",
        "trajectory",
        "transform",
        "translation",
        "waypoints",
        "world_x",
        "world_y",
        "world_z",
        "x",
        "y",
        "z",
    }
)


class SceneAdapterProtocolError(ValueError):
    """The grounding or adjudication transport violated its JSON protocol."""


@dataclass(frozen=True)
class CandidateSelection:
    """Candidate binding against semantic scene data before materialization."""

    scene_manifest: SceneManifest
    role_bindings: RoleBindings | None
    binding_report: BindingReport
    selected_candidate: TaskCandidate | None
    candidate_bindings: dict[str, RoleBindings] = field(default_factory=dict)

    @property
    def selected_candidate_id(self) -> str | None:
        """Return the chosen candidate identifier, when one was bindable."""
        return (
            str(self.selected_candidate["candidate_id"])
            if self.selected_candidate is not None
            else None
        )


@dataclass(frozen=True)
class SceneAdaptation:
    """Complete Scene Adapter result, including the reusable prepared scene."""

    scene_manifest: SceneManifest
    role_bindings: RoleBindings | None
    binding_report: BindingReport
    selected_candidate: TaskCandidate | None
    prepared_scene: PreparedScene
    source_config_path: Path
    conservative_scene_graph: ConservativeSceneGraph
    static_scene_manifest: StaticSceneManifest | None = None
    candidate_bindings: dict[str, RoleBindings] = field(default_factory=dict)

    @property
    def selected_candidate_id(self) -> str | None:
        return (
            str(self.selected_candidate["candidate_id"])
            if self.selected_candidate is not None
            else None
        )

    @property
    def reference_bindings(self) -> dict[str, list[str]]:
        if self.role_bindings is None:
            return {}
        return deepcopy(self.role_bindings["reference_bindings"])


class SceneAdapter:
    """Adapt one existing or packaged scene to a set of task candidates."""

    def __init__(
        self,
        *,
        model: str | None = None,
        grounding_caller: GroundingCaller | None = None,
        adjudicator: Adjudicator | None = None,
        robot_profile: str = "franka",
        scene_engine_adapter: SceneEngineV1Adapter | None = None,
    ) -> None:
        self.model = model
        self.grounding_caller = grounding_caller
        self.adjudicator = adjudicator
        self.robot_profile = robot_profile
        self.scene_engine_adapter = scene_engine_adapter or SceneEngineV1Adapter()

    def adapt(
        self,
        candidate_set: TaskCandidateSet | Sequence[Mapping[str, Any]],
        source: SceneSourceRef | str | Path,
        *,
        grounding_caller: GroundingCaller | None = None,
        adjudicator: Adjudicator | None = None,
        force_most_likely: bool = False,
        final_inspection: Mapping[str, Any] | None = None,
    ) -> SceneAdaptation:
        """Ground all candidates, then deterministically choose a bindable one."""
        task_id, instruction, candidates = _coerce_candidates(candidate_set)
        source_ref = self._resolve_source(source)
        source_fingerprint = fingerprint_scene_source(source_ref)
        prepared = prepare_scene(
            source_ref.path,
            z_rotation_degrees=source_ref.z_rotation_degrees,
            body_scale_policy=source_ref.body_scale_policy,
            body_scale=source_ref.body_scale,
        )
        if final_inspection is not None:
            normalized_inspection = validate_final_scene_inspection(final_inspection)
            if normalized_inspection["scene_revision_id"] != scene_revision_id(
                source_ref
            ):
                raise ValueError(
                    "FinalSceneInspection does not describe the adapted scene revision."
                )
            prepared = apply_final_inspection(prepared, normalized_inspection)
        inventory = SceneInventory(
            prepared.planner_objects,
            robot_profile=source_ref.robot_profile,
        )
        resolved_source = resolve_source_scene(source_ref.path)
        manifest = _build_manifest(
            prepared,
            inventory,
            source_format=resolved_source.source_format,
        )
        static_manifest = self.scene_engine_adapter.adapt_prepared_scene(
            prepared,
            source_format=resolved_source.source_format,
            robot_profile=inventory.profile,
        )
        static_manifest["source"]["source_fingerprint"] = source_fingerprint.to_dict()
        static_manifest = validate_static_scene_manifest(static_manifest)
        conservative_scene_graph = build_conservative_scene_graph(
            prepared,
            scene_id=static_manifest["scene_id"],
        )
        if fingerprint_scene_source(source_ref) != source_fingerprint:
            raise RuntimeError("Source Gym project changed while it was being adapted.")

        selection = self._select_candidates(
            task_id,
            instruction,
            candidates,
            manifest=manifest,
            inventory=inventory,
            scene_objects=prepared.planner_objects,
            grounding_caller=grounding_caller,
            adjudicator=adjudicator,
            force_most_likely=force_most_likely,
        )
        return SceneAdaptation(
            scene_manifest=manifest,
            role_bindings=selection.role_bindings,
            binding_report=selection.binding_report,
            selected_candidate=selection.selected_candidate,
            prepared_scene=prepared,
            source_config_path=prepared.source_config_path,
            conservative_scene_graph=conservative_scene_graph,
            static_scene_manifest=static_manifest,
            candidate_bindings=selection.candidate_bindings,
        )

    def select_objects(
        self,
        candidate_set: TaskCandidateSet | Sequence[Mapping[str, Any]],
        scene_objects: Sequence[Mapping[str, Any]],
        *,
        source_format: str = "embodichain.scene-blueprint/v2",
        robot_profile: str | None = None,
        grounding_caller: GroundingCaller | None = None,
        adjudicator: Adjudicator | None = None,
        force_most_likely: bool = False,
    ) -> CandidateSelection:
        """Bind candidates to semantic objects before assets are generated.

        Args:
            candidate_set: Validated Task Engine candidate set.
            scene_objects: Blueprint-level semantic object records.
            source_format: Provenance label included in the semantic manifest.
            robot_profile: Optional robot profile override.
            grounding_caller: Optional structured grounding transport.
            adjudicator: Optional candidate tie-breaker.
            force_most_likely: Resolve ranked UID hypotheses instead of rejecting
                low-confidence or ambiguous responses.

        Returns:
            Audited candidate selection without requiring generated assets.
        """
        task_id, instruction, candidates = _coerce_candidates(candidate_set)
        inventory = SceneInventory(
            scene_objects,
            robot_profile=robot_profile or self.robot_profile,
        )
        manifest = _build_semantic_manifest(
            inventory,
            source_format=source_format,
        )
        return self._select_candidates(
            task_id,
            instruction,
            candidates,
            manifest=manifest,
            inventory=inventory,
            scene_objects=scene_objects,
            grounding_caller=grounding_caller,
            adjudicator=adjudicator,
            force_most_likely=force_most_likely,
        )

    def _select_candidates(
        self,
        task_id: str,
        instruction: str,
        candidates: Sequence[TaskCandidate],
        *,
        manifest: SceneManifest,
        inventory: SceneInventory,
        scene_objects: Sequence[Mapping[str, Any]],
        grounding_caller: GroundingCaller | None,
        adjudicator: Adjudicator | None,
        force_most_likely: bool,
    ) -> CandidateSelection:
        invoke = grounding_caller or self.grounding_caller
        use_default_adjudicator = invoke is None
        if invoke is None:
            invoke = _default_grounding_caller()
        choose = adjudicator or self.adjudicator
        if choose is None and use_default_adjudicator:
            choose = _default_adjudicator(model=self.model)
        audits: list[dict[str, Any]] = []
        bindings_by_candidate: dict[str, dict[str, tuple[str, ...]]] = {}
        for candidate in candidates:
            audit, bindings = _ground_candidate(
                candidate,
                instruction=instruction,
                inventory=inventory,
                scene_objects=scene_objects,
                model=self.model,
                caller=invoke,
                force_most_likely=force_most_likely,
            )
            audits.append(audit)
            if bindings is not None:
                bindings_by_candidate[str(candidate["candidate_id"])] = bindings

        selected_id, status, reason = _select_candidate(
            candidates,
            audits,
            manifest=manifest,
            instruction=instruction,
            adjudicator=choose,
        )
        report = validate_binding_report(
            {
                "schema_version": BINDING_REPORT_SCHEMA,
                "task_id": task_id,
                "status": status,
                "selected_candidate_id": selected_id or "",
                "selection_reason": reason,
                "candidates": audits,
            }
        )
        selected = next(
            (
                deepcopy(candidate)
                for candidate in candidates
                if candidate["candidate_id"] == selected_id
            ),
            None,
        )
        candidate_bindings = {
            candidate_id: validate_role_bindings(
                {
                    "schema_version": ROLE_BINDINGS_SCHEMA,
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "reference_bindings": {
                        key: list(value) for key, value in sorted(raw_bindings.items())
                    },
                    "role_bindings": {},
                }
            )
            for candidate_id, raw_bindings in bindings_by_candidate.items()
        }
        role_bindings = None if selected_id is None else candidate_bindings[selected_id]
        return CandidateSelection(
            scene_manifest=manifest,
            role_bindings=role_bindings,
            binding_report=report,
            selected_candidate=selected,
            candidate_bindings=candidate_bindings,
        )

    def _resolve_source(
        self,
        source: SceneSourceRef | str | Path,
    ) -> SceneSourceRef:
        if isinstance(source, SceneSourceRef):
            return source
        return SceneSourceRef(source, robot_profile=self.robot_profile)


def _coerce_candidates(
    value: TaskCandidateSet | Sequence[Mapping[str, Any]],
) -> tuple[str, str, list[TaskCandidate]]:
    if isinstance(value, Mapping):
        normalized = validate_task_candidate_set(value)
        return (
            normalized["task_id"],
            normalized["instruction"],
            normalized["candidates"],
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = [validate_task_candidate(candidate) for candidate in value]
        if not candidates:
            raise ValueError("SceneAdapter requires at least one TaskCandidate.")
        task_ids = {candidate["draft"]["task_id"] for candidate in candidates}
        instructions = {candidate["draft"]["instruction"] for candidate in candidates}
        if len(task_ids) != 1 or len(instructions) != 1:
            raise ValueError("All TaskCandidates must describe the same task.")
        return task_ids.pop(), instructions.pop(), candidates
    raise TypeError("candidate_set must be a TaskCandidateSet or candidate sequence.")


def _default_grounding_caller() -> GroundingCaller:
    # Keep provider setup lazy so package import and offline tests never load an
    # LLM client. This is the same structured transport used by interpretation.
    return _default_instruction_caller


def _default_adjudicator(*, model: str | None) -> Adjudicator:
    caller = _default_grounding_caller()

    def adjudicate(**kwargs: Any) -> Mapping[str, Any]:
        candidates = [
            {
                key: deepcopy(candidate[key])
                for key in (
                    "candidate_id",
                    "draft",
                    "scene_request",
                    "success_spec",
                    "vote_count",
                )
            }
            for candidate in kwargs["candidates"]
        ]
        allowed = [str(candidate["candidate_id"]) for candidate in candidates]
        schema = {
            "title": "ActionEngineTaskAdjudication",
            "type": "object",
            "additionalProperties": False,
            "required": ["candidate_id"],
            "properties": {
                "candidate_id": {"type": "string", "enum": allowed},
            },
        }
        prompt = (
            "Select exactly one already verified, fully bindable task candidate "
            "that best matches the instruction and redacted scene manifest. Do "
            "not alter a candidate or invent a new interpretation. Return only "
            "candidate_id.\n\n"
            f"Instruction:\n{kwargs['instruction']}\n\n"
            "Candidates:\n"
            f"{json.dumps(candidates, ensure_ascii=False, sort_keys=True)}\n\n"
            "Redacted scene manifest:\n"
            f"{json.dumps(kwargs['scene_manifest'], ensure_ascii=False, sort_keys=True)}"
        )
        try:
            return caller(prompt=prompt, schema=schema, model=model)
        except (TypeError, ValueError) as exc:
            raise SceneAdapterProtocolError(
                f"Task adjudication returned invalid structured output: {exc}"
            ) from exc

    return adjudicate


def _ground_candidate(
    candidate: TaskCandidate,
    *,
    instruction: str,
    inventory: SceneInventory,
    scene_objects: Sequence[Mapping[str, Any]],
    model: str | None,
    caller: GroundingCaller,
    force_most_likely: bool,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]] | None]:
    responses: list[Any] = []

    def audited_caller(**kwargs: Any) -> Mapping[str, Any]:
        call_kwargs = dict(kwargs)
        if force_most_likely:
            call_kwargs["prompt"] = (
                f"{kwargs['prompt']}\n\nFINAL BINDING OVERRIDE: do not return "
                "ambiguous merely because confidence is low. Choose the most "
                "likely existing UID that satisfies the supplied structured "
                "role, affordance, state, and attribute metadata. Return "
                "candidate UIDs in descending likelihood order. Do not invent, "
                "add, delete, move, or modify any scene object. Use not_found "
                "when no structurally compatible existing object is plausible."
            )
        response = caller(**call_kwargs)
        responses.append(deepcopy(response))
        if force_most_likely:
            return _force_most_likely_response(response, candidate=candidate)
        return response

    candidate_id = str(candidate["candidate_id"])
    try:
        result = ground_scene_references(
            instruction=instruction,
            intent=candidate["draft"],
            inventory=inventory,
            scene_objects=scene_objects,
            model=model,
            caller=audited_caller,
        )
    except (TypeError, ValueError) as exc:
        if responses:
            audits = _audit_unresolved_response(
                responses[-1],
                candidate=candidate,
                inventory=inventory,
                error=str(exc),
            )
            status = _candidate_status(audits)
            return (
                _candidate_audit(candidate, status, audits, [str(exc)]),
                None,
            )
        raise SceneAdapterProtocolError(
            f"Grounding candidate {candidate_id!r} failed before returning JSON: {exc}"
        ) from exc

    raw_bindings = result.bindings
    response_by_id = _response_bindings(responses[-1], candidate=candidate)
    self_reference_reasons = _self_reference_reasons(candidate["draft"], raw_bindings)
    reference_audits = []
    incompatible: set[str] = set()
    request_by_id = {
        str(request["reference_id"]): request
        for request in candidate["scene_request"]["references"]
    }
    for reference_id, uids in raw_bindings.items():
        compatibility_reasons = _compatibility_reasons(
            request_by_id[reference_id],
            uids,
            inventory=inventory,
            draft=candidate["draft"],
        )
        compatibility_reasons.extend(self_reference_reasons.get(reference_id, ()))
        compatibility_reasons = sorted(set(compatibility_reasons))
        if compatibility_reasons:
            incompatible.add(reference_id)
        response = response_by_id[reference_id]
        audit_reasons = list(compatibility_reasons)
        if (
            force_most_likely
            and response.get("status") == "ambiguous"
            and response.get("uids")
        ):
            audit_reasons.append(
                "Forced the highest-ranked structurally compatible UID from an "
                "ambiguous low-confidence response."
            )
        reference_audits.append(
            {
                "reference_id": reference_id,
                "status": ("incompatible" if compatibility_reasons else "resolved"),
                "confidence": float(response["confidence"]),
                "candidate_uids": list(response["uids"]),
                "selected_uids": ([] if compatibility_reasons else list(uids)),
                "reasons": audit_reasons,
            }
        )
    if incompatible:
        reasons = [
            f"Reference {reference_id!r} conflicts with authoritative scene semantics."
            for reference_id in sorted(incompatible)
        ]
        return (
            _candidate_audit(candidate, "incompatible", reference_audits, reasons),
            None,
        )
    return _candidate_audit(candidate, "resolved", reference_audits, []), dict(
        raw_bindings
    )


def _force_most_likely_response(
    response: Mapping[str, Any],
    *,
    candidate: TaskCandidate,
) -> Mapping[str, Any]:
    """Turn ranked low-confidence UID hypotheses into explicit selections."""
    if not isinstance(response, Mapping) or set(response) != {"bindings"}:
        return response
    requests = {
        str(item["reference_id"]): item
        for item in candidate["scene_request"]["references"]
    }
    raw_bindings = response.get("bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
        return response
    result = deepcopy(dict(response))
    values = []
    for raw in raw_bindings:
        if not isinstance(raw, Mapping):
            return response
        item = deepcopy(dict(raw))
        request = requests.get(str(item.get("reference_id", "")))
        uids = item.get("uids")
        if (
            request is not None
            and item.get("status") in {"resolved", "ambiguous"}
            and isinstance(uids, Sequence)
            and not isinstance(uids, (str, bytes))
            and uids
        ):
            quantifier = str(request["quantifier"])
            count = int(request["count"])
            if quantifier == "one":
                item["uids"] = list(uids[:1])
            elif quantifier == "count":
                item["uids"] = list(uids[:count])
            item["status"] = "resolved"
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(
                confidence, bool
            ):
                item["confidence"] = max(0.5, float(confidence))
        values.append(item)
    result["bindings"] = values
    return result


def _response_bindings(
    response: Any,
    *,
    candidate: TaskCandidate,
) -> dict[str, Mapping[str, Any]]:
    expected = {
        str(request["reference_id"])
        for request in candidate["scene_request"]["references"]
    }
    if not isinstance(response, Mapping) or set(response) != {"bindings"}:
        raise SceneAdapterProtocolError(
            "Grounding response must contain only bindings."
        )
    values = response["bindings"]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise SceneAdapterProtocolError("Grounding response bindings must be a list.")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise SceneAdapterProtocolError(
                "Every grounding binding must be a mapping."
            )
        reference_id = raw.get("reference_id")
        if not isinstance(reference_id, str) or reference_id not in expected:
            raise SceneAdapterProtocolError(
                "Grounding response contains an unknown reference ID."
            )
        if reference_id in result:
            raise SceneAdapterProtocolError(
                "Grounding response contains duplicate reference IDs."
            )
        result[reference_id] = raw
    if set(result) != expected:
        raise SceneAdapterProtocolError(
            "Grounding response omitted requested reference IDs."
        )
    return result


def _audit_unresolved_response(
    response: Any,
    *,
    candidate: TaskCandidate,
    inventory: SceneInventory,
    error: str,
) -> list[dict[str, Any]]:
    by_id = _response_bindings(response, candidate=candidate)
    audits: list[dict[str, Any]] = []
    for request in candidate["scene_request"]["references"]:
        reference_id = str(request["reference_id"])
        raw = by_id[reference_id]
        if set(raw) != {"reference_id", "status", "uids", "confidence"}:
            raise SceneAdapterProtocolError(
                f"Grounding binding {reference_id!r} has unsupported fields."
            )
        status = raw["status"]
        if status not in {"resolved", "ambiguous", "not_found"}:
            raise SceneAdapterProtocolError(
                f"Grounding binding {reference_id!r} has invalid status."
            )
        uids = raw["uids"]
        confidence = raw["confidence"]
        if (
            not isinstance(uids, Sequence)
            or isinstance(uids, (str, bytes))
            or any(
                not isinstance(uid, str) or uid not in inventory.by_uid for uid in uids
            )
            or len(set(uids)) != len(uids)
        ):
            raise SceneAdapterProtocolError(
                f"Grounding binding {reference_id!r} has invalid candidate UIDs."
            )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise SceneAdapterProtocolError(
                f"Grounding binding {reference_id!r} has invalid confidence."
            )
        audit_status = status
        reasons: list[str] = []
        if status == "not_found" and uids:
            raise SceneAdapterProtocolError(
                f"Grounding binding {reference_id!r} status=not_found requires no UIDs."
            )
        if status == "resolved":
            audit_status = "incompatible"
            reasons.append(error)
        else:
            reasons.append(f"Grounding returned status={status}.")
        audits.append(
            {
                "reference_id": reference_id,
                "status": audit_status,
                "confidence": float(confidence),
                "candidate_uids": list(uids),
                "selected_uids": [],
                "reasons": reasons,
            }
        )
    return audits


def _compatibility_reasons(
    request: Mapping[str, Any],
    uids: Sequence[str],
    *,
    inventory: SceneInventory,
    draft: Mapping[str, Any],
) -> list[str]:
    entities = [inventory.by_uid[uid] for uid in uids]
    reasons: list[str] = []
    role = str(request["role"])
    step = next(item for item in draft["steps"] if item["id"] == request["step_id"])
    try:
        if role == "object":
            validate_source_compatibility(str(step["task_type"]), entities)
        else:
            for entity in entities:
                validate_target_compatibility(
                    str(step["task_type"]),
                    entity,
                    relation=str(step["relation"]),
                )
    except ValueError as exc:
        reasons.append(str(exc))

    expected_structure = str(request["source_structure"])
    for entity in entities:
        # Source structure is strict for manipulated objects. Target structure
        # is relation-dependent and is already checked by
        # validate_target_compatibility; a table support surface must not be
        # rejected merely because it is passive rather than a rigid object.
        if role == "object":
            if expected_structure == "articulation" and entity.role != "articulation":
                reasons.append(
                    f"UID {entity.uid!r} is not an articulation as requested."
                )
            if expected_structure in {
                "rigid_object",
                "movable",
            } and entity.role not in {
                "object",
                "rigid_object",
            }:
                reasons.append(
                    f"UID {entity.uid!r} is not a movable rigid object as requested."
                )
        required_affordances = set(request["affordances"])
        if entity.affordances:
            missing = required_affordances - set(entity.affordances)
            if missing:
                reasons.append(
                    f"UID {entity.uid!r} explicitly lacks affordances {sorted(missing)}."
                )
        for key, expected in request["initial_state"].items():
            if key in entity.initial_state and entity.initial_state[key] != expected:
                reasons.append(
                    f"UID {entity.uid!r} state {key!r} conflicts with the request."
                )
        for key, expected in request["attributes"].items():
            if key in entity.attributes and entity.attributes[key] != expected:
                reasons.append(
                    f"UID {entity.uid!r} attribute {key!r} conflicts with the request."
                )
    return sorted(set(reasons))


def _self_reference_reasons(
    draft: Mapping[str, Any],
    bindings: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Reject object/target identity overlap, including step_result selectors."""
    objects_by_step: dict[str, tuple[str, ...]] = {}
    reasons: dict[str, list[str]] = {}
    for step in draft["steps"]:
        step_id = str(step["id"])
        object_uids = _selector_uids(
            step["object"],
            reference_id=f"{step_id}.object",
            bindings=bindings,
            objects_by_step=objects_by_step,
        )
        target_uids = _selector_uids(
            step["target"],
            reference_id=f"{step_id}.target",
            bindings=bindings,
            objects_by_step=objects_by_step,
        )
        overlap = sorted(set(object_uids) & set(target_uids))
        if overlap:
            reason = (
                f"Grounding step {step_id!r} uses the same UID as object and "
                f"target: {overlap}."
            )
            for role in ("object", "target"):
                selector = step[role]
                if selector["kind"] == "scene_ref":
                    reasons.setdefault(f"{step_id}.{role}", []).append(reason)
        objects_by_step[step_id] = object_uids
    return reasons


def _selector_uids(
    selector: Mapping[str, Any],
    *,
    reference_id: str,
    bindings: Mapping[str, Sequence[str]],
    objects_by_step: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    kind = str(selector["kind"])
    if kind == "scene_ref":
        return tuple(str(uid) for uid in bindings[reference_id])
    if kind == "step_result":
        return objects_by_step[str(selector["step_id"])]
    return ()


def _candidate_audit(
    candidate: TaskCandidate,
    status: str,
    references: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "semantic_hash": candidate["semantic_hash"],
        "status": status,
        "references": [deepcopy(dict(reference)) for reference in references],
        "reasons": list(reasons),
    }


def _candidate_status(references: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(reference["status"]) for reference in references}
    if statuses == {"resolved"}:
        return "resolved"
    if "ambiguous" in statuses:
        return "ambiguous"
    if "incompatible" in statuses:
        return "incompatible"
    return "not_found"


def _select_candidate(
    candidates: Sequence[TaskCandidate],
    audits: Sequence[Mapping[str, Any]],
    *,
    manifest: SceneManifest,
    instruction: str,
    adjudicator: Adjudicator | None,
) -> tuple[str | None, str, str]:
    audit_by_id = {str(audit["candidate_id"]): audit for audit in audits}
    bound = [
        candidate
        for candidate in candidates
        if audit_by_id[str(candidate["candidate_id"])]["status"] == "resolved"
    ]
    majority = [candidate for candidate in bound if int(candidate["vote_count"]) >= 2]
    if len(majority) == 1:
        return str(majority[0]["candidate_id"]), "bound", "majority_bindable"
    if not majority and len(bound) == 1:
        return str(bound[0]["candidate_id"]), "bound", "unique_bindable"

    choices = majority if majority else bound
    if len(choices) > 1:
        if adjudicator is None:
            return None, "ambiguous", "multiple_conflicting_bindable_candidates"
        raw = adjudicator(
            instruction=instruction,
            candidates=deepcopy(list(choices)),
            scene_manifest=deepcopy(manifest),
        )
        if not isinstance(raw, Mapping) or set(raw) != {"candidate_id"}:
            raise SceneAdapterProtocolError(
                "Adjudicator response must contain only candidate_id."
            )
        selected_id = raw["candidate_id"]
        allowed = {str(candidate["candidate_id"]) for candidate in choices}
        if not isinstance(selected_id, str) or selected_id not in allowed:
            raise SceneAdapterProtocolError(
                "Adjudicator must select the candidate_id of a verified bindable candidate."
            )
        return selected_id, "bound", "adjudicated_bindable"
    if any(audit["status"] == "ambiguous" for audit in audits):
        return None, "ambiguous", "no_fully_bound_candidate"
    return None, "unsatisfied", "no_fully_bound_candidate"


def _build_manifest(
    prepared: PreparedScene,
    inventory: SceneInventory,
    *,
    source_format: str,
) -> SceneManifest:
    objects = [
        {
            "uid": entity.uid,
            "role": entity.role,
            "name": entity.name,
            "description": entity.description,
            "category": entity.category,
            "color": entity.color,
            "affordances": sorted(entity.affordances),
            "initial_state": _redact_semantics(entity.initial_state),
            "attributes": _redact_semantics(entity.attributes),
        }
        for entity in sorted(inventory.entities, key=lambda item: item.uid)
    ]
    scene_id = _canonical_hash(
        {
            "source_format": source_format,
            "objects": objects,
            "asset_hashes": prepared.asset_hashes,
            "rotation": prepared.z_rotation_degrees,
            "xy_translation": list(prepared.source_scene_xy_translation),
            "body_scale_policy": prepared.body_scale_policy,
            "body_scale": prepared.body_scale,
        }
    )
    return validate_scene_manifest(
        {
            "schema_version": SCENE_MANIFEST_SCHEMA,
            "scene_id": scene_id,
            "source_format": source_format,
            "robot_profile": inventory.profile,
            "objects": objects,
        }
    )


def _build_semantic_manifest(
    inventory: SceneInventory,
    *,
    source_format: str,
) -> SceneManifest:
    objects = [
        {
            "uid": entity.uid,
            "role": entity.role,
            "name": entity.name,
            "description": entity.description,
            "category": entity.category,
            "color": entity.color,
            "affordances": sorted(entity.affordances),
            "initial_state": _redact_semantics(entity.initial_state),
            "attributes": _redact_semantics(entity.attributes),
        }
        for entity in sorted(inventory.entities, key=lambda item: item.uid)
    ]
    return validate_scene_manifest(
        {
            "schema_version": SCENE_MANIFEST_SCHEMA,
            "scene_id": _canonical_hash(
                {"source_format": source_format, "objects": objects}
            ),
            "source_format": source_format,
            "robot_profile": inventory.profile,
            "objects": objects,
        }
    )


def _redact_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        name = str(key)
        normalized = name.strip().lower().replace("-", "_")
        if normalized in _REDACTED_KEYS:
            continue
        if isinstance(child, Mapping):
            nested = _redact_semantics(child)
            if nested:
                result[name] = nested
        elif isinstance(child, (str, int, float, bool)) and not isinstance(
            child, complex
        ):
            result[name] = child
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            simple = [item for item in child if isinstance(item, (str, bool))]
            if len(simple) == len(child):
                result[name] = simple
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
