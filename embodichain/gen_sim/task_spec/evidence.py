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

"""Versioned, data-only bindings for independently checked placement evidence.

Importing this module loads no numerical or simulation backend. Physical
re-evaluation imports shared compute algorithms only when their payload is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import math
from typing import Any, TYPE_CHECKING

from .canonicalization import canonical_hash, json_snapshot

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation
    from .validation import EvaluationReport

__all__ = [
    "SupportedPlacementEvidenceBinding",
    "evaluate_supported_placement_evidence",
    "evaluate_placement_evidence_report",
    "sampled_support_window",
    "directional_observation_axes",
]

_SCOPES = frozenset(
    {
        "native_contact_support",
        "native_stable_support",
        "native_supported_placement",
        "measured_planar_landing_region",
        "bound_rigid_world_axes",
        "bound_link_world_axes",
        "native_supported_landing_region",
    }
)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be an exact nonempty identifier.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _path(value: Any, name: str) -> tuple[str | int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or not value
        or any(
            type(item) not in (str, int)
            or (type(item) is str and not item)
            or (type(item) is int and item < 0)
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain exact nonempty source-path components.")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class SupportedPlacementEvidenceBinding:
    """Sealed normative identity for one selected placement observation.

    The frontend projects its workflow/role declarations into this plain value.
    This value does not contain a Task Program compiler, scene object or callback.
    Actual release, episode, sample and support identities are verified against
    supplied snapshots by the evidence evaluator. Scope is explicit: the legacy
    contact observation cannot acquire a stability-window qualification merely
    because a timeline happens to be available.

    Nested JSON data is independently owned. Every public evaluation re-decodes
    the binding and verifies ``binding_hash``; in-place mutation invalidates the
    original declaration rather than preserving a stale identity.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    object_id: str
    reference_id: str
    resource_id: str
    release_segment_id: str
    release_invocation_id: str
    release_source_path: tuple[str | int, ...]
    support_segment_id: str
    stability_preset: str
    support_source_path: tuple[str | int, ...]
    boundary_segment_id: str
    boundary_source_path: tuple[str | int, ...]
    evidence_scope: str
    observation_kind: str
    evidence_format: str = "raw_snapshot"
    condition: Mapping[str, Any] = field(default_factory=dict)
    required_execution: tuple[Mapping[str, Any], ...] = ()
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    binding_hash: str = ""
    operation_evidence: Mapping[str, Any] | None = None
    sampled_contact_identity: Mapping[str, Any] | None = None
    planar_geometry: Mapping[str, Any] | None = None
    directional_geometry: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in (
            "predicate_id",
            "object_id",
            "reference_id",
            "resource_id",
            "release_segment_id",
            "release_invocation_id",
            "support_segment_id",
            "stability_preset",
            "boundary_segment_id",
        ):
            _identifier(getattr(self, name), name)
        for name in (
            "release_source_path",
            "support_source_path",
            "boundary_source_path",
        ):
            object.__setattr__(self, name, _path(getattr(self, name), name))
        if self.object_id == self.reference_id:
            raise ValueError(
                "Placement subject and reference must be different entities."
            )
        if self.evidence_scope not in _SCOPES:
            raise ValueError("Unsupported placement evidence scope.")
        if self.observation_kind not in {"recorded", "final"}:
            raise ValueError("observation_kind must be recorded or final.")
        if self.evidence_format not in {"raw_snapshot", "historical_record"}:
            raise ValueError(
                "evidence_format must explicitly select raw_snapshot or historical_record."
            )
        if (
            self.evidence_scope == "native_stable_support"
            and self.evidence_format != "historical_record"
            and self.sampled_contact_identity is None
        ):
            raise ValueError(
                "Raw sampled-timeline collection requires its explicit acquisition identity; it cannot use the COM-patch acquisition contract."
            )
        ids = tuple(self.env_ids)
        if (
            not ids
            or any(type(value) is not int for value in ids)
            or ids != tuple(range(len(ids)))
        ):
            raise ValueError(
                "The source placement contract requires ordered native row IDs."
            )
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(value) is not int or value < 0 for value in episodes
            ):
                raise ValueError(
                    "episode_ids must contain one declared EpisodeContext identity per row."
                )
            object.__setattr__(self, "episode_ids", episodes)
        if not isinstance(self.condition, Mapping):
            raise TypeError("condition must be a normalized JSON declaration.")
        object.__setattr__(self, "condition", json_snapshot(dict(self.condition)))
        if self.condition and (
            self.condition.get("entity") != self.object_id
            or self.condition.get("reference")
            != (
                (
                    self.directional_geometry.get("reference_link_id")
                    or self.reference_id
                )
                if isinstance(self.directional_geometry, Mapping)
                else self.reference_id
            )
        ):
            raise ValueError("Condition endpoints differ from the normative binding.")
        if self.evidence_scope not in {
            "native_contact_support",
            "measured_planar_landing_region",
        }:
            for name in ("duration", "translation_drift", "rotation_drift"):
                value = self.condition.get(name)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"The selected support scope requires normalized {name}."
                    )
        requirements = tuple(json_snapshot(item) for item in self.required_execution)
        for item in requirements:
            if not isinstance(item, Mapping) or set(item) != {
                "segment_id",
                "source_path",
            }:
                raise ValueError(
                    "Required execution declarations contain a segment ID and source path."
                )
            _identifier(item["segment_id"], "required execution segment")
            _path(item["source_path"], "required execution source path")
        if len({item["segment_id"] for item in requirements}) != len(requirements):
            raise ValueError("Required execution occurrences must be unique.")
        object.__setattr__(self, "required_execution", requirements)
        revision = self.condition.get("support_contract_revision")
        if self.evidence_scope == "native_supported_placement":
            from ._strong_support import _condition_values

            _condition_values(self.condition)
        elif self.evidence_scope == "native_supported_landing_region":
            from ._legacy_region_evidence import validate_legacy_region

            validate_legacy_region(self)
        elif revision is not None or self.condition.get("kind") == "supported_region":
            raise ValueError(
                "Strong/full-footprint declarations require their explicit evidence scope; they cannot use a contact-only or version-2 verifier."
            )
        if self.evidence_scope == "native_stable_support" and len(ids) != 1:
            raise ValueError(
                "The version-2 stable support contract has one native row."
            )
        if self.evidence_scope in {"bound_rigid_world_axes", "bound_link_world_axes"}:
            if self.directional_geometry is None:
                raise ValueError(
                    "Directional evidence requires its exact spatial route declaration."
                )
            object.__setattr__(
                self, "directional_geometry", json_snapshot(self.directional_geometry)
            )
            from ._directional_evidence import validate_directional_geometry

            validate_directional_geometry(self)
        elif self.directional_geometry is not None:
            raise ValueError(
                "Directional geometry cannot change another evidence scope."
            )
        if self.evidence_scope == "measured_planar_landing_region":
            if self.planar_geometry is None:
                raise ValueError(
                    "Planar evidence requires its explicit geometry declaration."
                )
            object.__setattr__(
                self, "planar_geometry", json_snapshot(self.planar_geometry)
            )
            from ._planar_evidence import validate_planar_geometry

            validate_planar_geometry(self)
        elif self.planar_geometry is not None:
            raise ValueError("Planar geometry cannot change another evidence scope.")
        if self.sampled_contact_identity is not None:
            if (
                self.evidence_scope != "native_stable_support"
                or self.evidence_format != "raw_snapshot"
                or self.episode_ids is None
            ):
                raise ValueError(
                    "Sampled acquisition identities require raw sampled support and explicit episodes."
                )
            object.__setattr__(
                self,
                "sampled_contact_identity",
                json_snapshot(self.sampled_contact_identity),
            )
            from ._sampled_contact_evidence import validate_sampled_identity

            validate_sampled_identity(self)
        if self.operation_evidence is not None:
            object.__setattr__(
                self, "operation_evidence", json_snapshot(self.operation_evidence)
            )
            from ._placement_linkage import validate_linkage

            validate_linkage(self)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Placement evidence binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        version = (
            "v7"
            if self.evidence_scope == "native_supported_landing_region"
            else (
                "v6"
                if self.directional_geometry is not None
                else (
                    "v5"
                    if self.planar_geometry is not None
                    else (
                        "v4"
                        if self.sampled_contact_identity is not None
                        else "v3" if self.operation_evidence is not None else "v2"
                    )
                )
            )
        )
        return {
            "schema_version": "task_spec.placement_evidence_binding/" + version,
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
                and (
                    item.name
                    not in {
                        "operation_evidence",
                        "sampled_contact_identity",
                        "planar_geometry",
                        "directional_geometry",
                    }
                    or getattr(self, item.name) is not None
                )
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return an independently owned declaration with its original identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: SupportedPlacementEvidenceBinding | Mapping[str, Any]
    ) -> SupportedPlacementEvidenceBinding:
        """Revalidate and own a binding, including existing instances.

        Args:
            value: One closed declaration or existing instance.

        Returns:
            A validated independently owned binding preserving its valid hash.

        Raises:
            ValueError: The schema or original content identity no longer matches.
            TypeError: An unknown field or non-JSON value is supplied.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if not isinstance(raw, dict):
            raise TypeError("Placement evidence binding must be a mapping.")
        version = raw.pop("schema_version", None)
        modes = {
            "v2": None,
            "v3": "operation_evidence",
            "v4": "sampled_contact_identity",
            "v5": "planar_geometry",
            "v6": "directional_geometry",
            "v7": None,
        }
        key = (
            version.removeprefix("task_spec.placement_evidence_binding/")
            if type(version) is str
            else None
        )
        selected = modes.get(key)
        allowed = {"operation_evidence"} if key in {"v3", "v4", "v5", "v6"} else set()
        if selected is not None:
            allowed.add(selected)
        if (
            version != "task_spec.placement_evidence_binding/" + str(key)
            or key not in modes
            or key == "v7"
            and raw.get("evidence_scope") != "native_supported_landing_region"
            or key != "v7"
            and raw.get("evidence_scope") == "native_supported_landing_region"
            or selected is not None
            and raw.get(selected) is None
            or any(
                name in raw and name not in allowed
                for name in (
                    "operation_evidence",
                    "sampled_contact_identity",
                    "planar_geometry",
                    "directional_geometry",
                )
            )
        ):
            raise ValueError(
                "Unsupported placement binding schema or operation linkage; rebuild bindings from explicit declarations."
            )
        return cls(**raw)


def sampled_support_window(
    binding: SupportedPlacementEvidenceBinding | Mapping[str, Any],
    boundary_frames: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select the original inclusive sampled-support window from its declaration.

    Args:
        binding: Exact raw sampled-support binding with source recording cadence.
        boundary_frames: Actual completed native boundary slots supplied by the host.

    Returns:
        Inclusive start and end slots for the Lab recording API. Window length
        retains the original ``ceil(duration / step_dt)`` rule.

    Raises:
        ValueError: The scope, row count or completed history is insufficient.
    """
    selected = SupportedPlacementEvidenceBinding.from_dict(binding)
    ends = tuple(boundary_frames)
    if (
        selected.evidence_scope != "native_stable_support"
        or selected.sampled_contact_identity is None
        or len(ends) != 1
        or any(type(value) is not int or value <= 0 for value in ends)
    ):
        raise ValueError(
            "Sampled support requires its bound single-row completed boundary."
        )
    start = ends[0] - math.ceil(
        selected.condition["duration"] / selected.sampled_contact_identity["step_dt"]
    )
    if start < 0:
        raise ValueError("Sampled support window predates the initialized episode.")
    return (start,), ends


def directional_observation_axes(
    binding: SupportedPlacementEvidenceBinding | Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    """Return the exact numeric world-axis query selected by a side predicate.

    Args:
        binding: Sealed directional spatial-route declaration.

    Returns:
        Ordered (axis,sign) operands for Lab geometry acquisition. This helper
        performs no physical read and does not infer capability from entity names.
    """
    selected = SupportedPlacementEvidenceBinding.from_dict(binding)
    if selected.directional_geometry is None:
        raise ValueError("Directional observation requires its explicit spatial route.")
    from ._directional_evidence import AXES

    return AXES[selected.directional_geometry["relation"]]


def _bool_rows(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(type(item) is bool for item in value)
    )


def _legacy_flags(
    binding: SupportedPlacementEvidenceBinding,
    evidence: Mapping[str, Any],
    released: list[bool],
    boundary: Mapping[str, Any],
    support_boundary: Mapping[str, Any] | None,
    *,
    require_claims: bool,
) -> list[bool]:
    """Preserve the source's explicitly weak instantaneous support scope."""
    import torch
    from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

    size = len(binding.env_ids)
    record, metrics = evidence["record"], evidence["record"]["measurements"]
    supported = metrics.get("supported")
    if (
        metrics.get("scope") != "native_contact_support"
        or metrics.get("support_contract_revision") is not None
        or metrics.get("entity") != binding.object_id
        or metrics.get("reference") != binding.reference_id
        or metrics.get("reference_uid") != binding.reference_id
        or metrics.get("relation") != "on"
        or metrics.get("env_ids") != list(binding.env_ids)
        or not _bool_rows(supported, size)
        or (
            (require_claims or "released" in metrics)
            and metrics.get("released") != released
        )
    ):
        raise ValueError(
            "Legacy support differs from its explicit subject/reference/release binding."
        )
    for name, key in (
        (binding.object_id, "object_pose"),
        (binding.reference_id, "reference_pose"),
    ):
        observed = torch.tensor(evidence["boundary_poses"][name])
        recorded = torch.tensor(metrics[key]).to(observed)
        validate_rigid_pose_batch(observed, size)
        validate_rigid_pose_batch(recorded, size)
        if not torch.allclose(observed, recorded, atol=1e-4, rtol=0):
            raise ValueError("Support differs from the exact observed boundary poses.")
    if binding.observation_kind == "recorded":
        if support_boundary is None:
            raise ValueError(
                "Recorded support requires its executed policy occurrence."
            )
        if metrics.get("boundary_frames") != boundary.get("end_steps"):
            raise ValueError("Recorded support has another terminal state slot.")
        traces = [
            item
            for item in support_boundary["metadata"].get("post_policies", [])
            if item.get("kind") == "wait_stable"
            and type(item.get("policy_index")) is int
            and item.get("source_path")
            == [*binding.support_source_path, "post", item["policy_index"]]
            and item.get("result", {}).get("preset") == binding.stability_preset
        ]
        if len(traces) != 1:
            raise ValueError(
                "Support lacks its unique declared post-policy occurrence."
            )
        native = traces[0]["result"]["measurements"]["native_support"]
        if any(metrics.get(key) != value for key, value in native.items()):
            raise ValueError(
                "Recorded support differs from its own measured policy snapshot."
            )
    return [
        support and release
        for support, release in zip(supported, released, strict=True)
    ]


def evaluate_supported_placement_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: SupportedPlacementEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Check a selected placement's physical snapshots before TaskSpec evaluation.

    Args:
        spec: TaskSpec or its JSON representation.
        binding: Sealed data-only selection of normative and execution identities.
        evidence: Owned-on-entry JSON snapshots, including named execution segments,
            a historical physical record or a raw Lab support snapshot, plus
            exact boundary poses. Existing accepted/released claims are checked;
            raw acquisition carries no predicate conclusion. Missing channels remain
            unavailable; planning success is not a physical receipt.
        program: Exact program snapshot named by the binding's content identity.

    Returns:
        One PredicateEvaluation per environment row. Feed each row's explicit
        status/value into the ordinary TaskSpec evaluator; this function does not
        decide certificate readiness for unobserved predicates or milestones.

    Raises:
        ValueError: The normative binding/spec/program identity is invalid.

    Numerical re-evaluation lazily loads the shared Torch/NumPy/Shapely compute
    implementation. Importing this API does not load those dependencies or Lab.
    """
    normalized, selected, predicate = _placement_selection(
        spec, binding, program=program
    )
    try:
        roles = {role.name: role for role in normalized.roles}
        selector_error = (
            None
            if all(
                roles[predicate.arguments[key]].is_single_member
                for key in ("subject", "reference")
            )
            else "Complete selector-universe evidence was not provided."
        )
    except KeyError as error:
        selector_error = str(error) or type(error).__name__
    return _evaluate_placement_member(
        selected, predicate, evidence, program=program, selector_error=selector_error
    )


def _placement_selection(spec, binding, *, program):
    """Validate unchanged full normative/program identity and placement semantics."""
    from .spec import TaskSpec

    selected = SupportedPlacementEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    if canonical_hash(normalized.to_dict()) != selected.task_spec_digest:
        raise ValueError("Placement binding has a different TaskSpec identity.")
    if canonical_hash(program) != selected.program_digest:
        raise ValueError("Placement binding has a different program identity.")
    predicates = [
        item
        for item in (
            *normalized.goal,
            *(
                predicate
                for milestone in normalized.milestones
                for predicate in milestone.achieve
            ),
        )
        if item.id == selected.predicate_id
    ]
    if len(predicates) != 1:
        raise ValueError("Placement binding must identify one declared predicate.")
    predicate = predicates[0]
    family = predicate.arguments.get("task_type")
    if family not in {None, "E1"}:
        if (
            selected.operation_evidence is None
            or selected.operation_evidence["family"] != family
        ):
            raise ValueError(
                "Placement requires its exact handover/coordinated task-family evidence."
            )
    elif selected.operation_evidence is not None:
        raise ValueError(
            "Ordinary placement cannot borrow another operation family's proof."
        )
    if selected.evidence_scope == "native_supported_landing_region" and (
        family != "E1" or "orientation_goal" in predicate.arguments
    ):
        raise ValueError(
            "Legacy region evidence requires the original bare-table E1 predicate."
        )
    milestone_predicate = any(
        item.id == selected.predicate_id
        for milestone in normalized.milestones
        for item in milestone.achieve
    )
    expected_kind = "recorded" if milestone_predicate else "final"
    if selected.observation_kind != expected_kind:
        raise ValueError(
            "Goal evidence must be final and milestone evidence must be recorded."
        )
    if milestone_predicate and not selected.required_execution:
        raise ValueError(
            "Milestone evidence requires its explicit execution occurrence prefix."
        )
    from ._directional_evidence import AXES

    permitted_relations = (
        set(AXES)
        if selected.directional_geometry is not None
        else (
            {"on", "inside"}
            if selected.evidence_scope
            in {"native_supported_placement", "measured_planar_landing_region"}
            else {"on"}
        )
    )
    if (
        selected.directional_geometry is not None
        and predicate.arguments.get("relation")
        != selected.directional_geometry["relation"]
    ):
        raise ValueError("Directional predicate differs from its sealed spatial route.")
    if (
        predicate.name != "semantic_goal"
        or predicate.arguments.get("relation") not in permitted_relations
        or predicate.arguments.get("relation") == "inside"
        and selected.evidence_scope != "measured_planar_landing_region"
        and selected.condition.get("kind") != "supported_region"
        or predicate.operator != "eq"
        or predicate.expected is not True
    ):
        raise ValueError(
            "This placement evidence contract requires an on-relation goal."
        )
    return normalized, selected, predicate


def _evaluate_placement_member(
    selected: SupportedPlacementEvidenceBinding,
    predicate: Any,
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    selector_error: str | None = None,
) -> tuple[PredicateEvaluation, ...]:
    """Verify one physical member after its full selector is independently bound."""
    from .expressions import EvaluationStatus, PredicateEvaluation
    from ._release_evidence import measured_release

    family = predicate.arguments.get("task_type")
    size = len(selected.env_ids)
    reason = None
    availability = [True] * size
    errors: list[str | None] = [None] * size
    flags: list[bool] = [False] * size
    execution_available = [True] * size
    reference = ()
    try:
        if selector_error is not None:
            raise ValueError(selector_error)
        snapshot = json_snapshot(evidence)
        if not isinstance(snapshot, dict):
            raise TypeError("Physical evidence must be a JSON mapping.")
        reference = (selected.binding_hash, canonical_hash(snapshot))
        if "unavailable" in snapshot:
            message = snapshot["unavailable"]
            if type(message) is not str or not message.strip():
                raise ValueError(
                    "Unavailable evidence must retain a nonempty acquisition reason."
                )
            raise ValueError(message)
        historical_record = selected.evidence_format == "historical_record"
        if historical_record:
            if "record" not in snapshot or "native_support" in snapshot:
                raise ValueError(
                    "Historical evidence requires its record format, without a raw snapshot overlay."
                )
        elif "record" in snapshot:
            raise ValueError(
                "Raw evidence cannot select historical validation by adding a record."
            )
        if selected.episode_ids is not None:
            if snapshot.get("episode_ids") != list(selected.episode_ids) or any(
                type(value) is not int for value in snapshot.get("episode_ids", [])
            ):
                raise ValueError(
                    "Physical support belongs to another declared episode."
                )
        elif not historical_record:
            raise ValueError(
                "Raw Lab observations require an explicit episode binding."
            )
        acquisition = snapshot.get("support_acquisition")
        if (
            acquisition is None
            and not historical_record
            and selected.evidence_scope
            not in {
                "native_stable_support",
                "measured_planar_landing_region",
                "bound_rigid_world_axes",
                "bound_link_world_axes",
            }
        ):
            raise ValueError("Raw support acquisition channel is missing.")
        if acquisition is not None:
            if selected.evidence_scope == "native_stable_support":
                raise ValueError(
                    "The sampled-timeline scope cannot borrow a COM-patch acquisition channel."
                )
            if (
                not isinstance(acquisition, Mapping)
                or acquisition.get("evidence_id") != "support.com_patch"
                or not _bool_rows(acquisition.get("values"), size)
                or type(acquisition.get("timestamp")) not in (int, float)
                or not math.isfinite(acquisition["timestamp"])
                or acquisition["timestamp"] < 0
                or type(acquisition.get("observation_revision")) is not int
                or acquisition["observation_revision"] < 0
                or acquisition.get("clock_domain")
                != (
                    "environment_step"
                    if selected.evidence_scope == "native_supported_placement"
                    else "simulation_time"
                )
                or acquisition.get("env_ids") != list(selected.env_ids)
                or any(
                    type(value) is not int for value in acquisition.get("env_ids", [])
                )
                or not _bool_rows(acquisition.get("valid"), size)
            ):
                raise ValueError(
                    "Support acquisition validity has another row identity."
                )
            availability = acquisition["valid"]
            errors = acquisition["acquisition_errors"]
            if (
                not isinstance(errors, list)
                or len(errors) != size
                or any(
                    error is not None and (type(error) is not str or not error.strip())
                    for error in errors
                )
            ):
                raise ValueError(
                    "Support acquisition errors have another row identity."
                )
            if any(
                valid and error is not None
                for valid, error in zip(availability, errors, strict=True)
            ):
                raise ValueError(
                    "Support validity contradicts its acquisition error channel."
                )
            if not any(availability):
                raise ValueError(
                    "Physical support acquisition is unavailable: "
                    + "; ".join(
                        dict.fromkeys(
                            error or "missing acquisition" for error in errors
                        )
                    )
                )
        segments = {item["name"]: item for item in snapshot["segments"]}
        if len(segments) != len(snapshot["segments"]):
            raise ValueError("Duplicate execution segments cannot prove a placement.")
        for required in selected.required_execution:
            segment = segments[required["segment_id"]]
            metadata = segment["metadata"]
            runtime_ids = metadata["runtime"]["env_ids"]
            if (
                metadata["program_segment_source_path"] != required["source_path"]
                or runtime_ids != list(selected.env_ids)
                or any(type(value) is not int for value in runtime_ids)
            ):
                raise ValueError(
                    "Required execution evidence has another occurrence or row identity."
                )
            active, success = segment.get("active"), segment.get("successes")
            if not _bool_rows(active, size) or not _bool_rows(success, size):
                raise ValueError(
                    "Required execution activity/outcome evidence is incomplete."
                )
            execution_available = [
                previous and eligible and completed
                for previous, eligible, completed in zip(
                    execution_available, active, success, strict=True
                )
            ]
        selected_segments = []
        for index, (name, path) in enumerate(
            (
                (selected.release_segment_id, selected.release_source_path),
                (selected.support_segment_id, selected.support_source_path),
                (selected.boundary_segment_id, selected.boundary_source_path),
            )
        ):
            if (
                index == 1
                and selected.evidence_scope != "native_supported_placement"
                and selected.observation_kind == "final"
            ):
                # These source final-state/window contracts do not consume a
                # completed post-policy receipt. Revision-one strong support
                # retains its explicit successful policy requirement.
                selected_segments.append(None)
                continue
            segment = segments[name]
            if segment["metadata"]["program_segment_source_path"] != list(path):
                raise ValueError("Evidence belongs to another execution occurrence.")
            selected_segments.append(segment)
        release_segment, support_boundary, boundary = selected_segments
        if selected.observation_kind == "final" and (
            snapshot["segments"][-1]["name"] != selected.boundary_segment_id
            or "lengths" in snapshot
            and snapshot["lengths"] != boundary["end_steps"]
        ):
            raise ValueError(
                "Final placement requires its actual last runtime boundary."
            )
        if not historical_record and selected.evidence_scope == "native_stable_support":
            from ._sampled_contact_evidence import prepare_sampled_support

            metrics, timeline, poses = prepare_sampled_support(
                selected, snapshot, boundary
            )
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": metrics,
            }
            snapshot["contact_timeline"] = timeline
            snapshot["boundary_poses"] = poses
        elif (
            not historical_record
            and selected.evidence_scope == "measured_planar_landing_region"
        ):
            from ._planar_evidence import raw_planar_metrics

            metrics = raw_planar_metrics(selected, snapshot, boundary)
            metrics.update(
                reference_uid=selected.reference_id,
                relation=predicate.arguments["relation"],
                boundary_frames=snapshot["boundary_frames"],
            )
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": metrics,
            }
        elif not historical_record and selected.directional_geometry is not None:
            from ._directional_evidence import raw_directional_metrics

            metrics = raw_directional_metrics(selected, snapshot, boundary)
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": metrics,
            }
        elif (
            not historical_record
            and selected.evidence_scope == "native_supported_landing_region"
        ):
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": dict(
                    scope=selected.evidence_scope,
                    placement_node_id=selected.release_segment_id,
                    boundary_node_id=selected.boundary_segment_id,
                    observation_kind=selected.observation_kind,
                    native_support=snapshot["native_support"],
                    landing_region=snapshot["landing_region"],
                    support_gap_m=snapshot["support_gap_m"],
                    contact_tolerance_m=0.01,
                    reference_uid=selected.reference_id,
                    relation="on",
                    boundary_frames=boundary["end_steps"],
                ),
            }
        elif not historical_record:
            native = json_snapshot(snapshot["native_support"])
            metrics = {
                **native,
                "reference_uid": selected.reference_id,
                "relation": predicate.arguments["relation"],
            }
            if selected.evidence_scope == "native_supported_placement":
                metrics = dict(
                    scope=selected.evidence_scope,
                    support_contract_revision=1,
                    placement_node_id=selected.release_segment_id,
                    support_boundary_node_id=selected.support_segment_id,
                    boundary_node_id=selected.boundary_segment_id,
                    stability_preset=selected.stability_preset,
                    observation_kind=selected.observation_kind,
                    native_support=native,
                    reference_uid=selected.reference_id,
                    relation=predicate.arguments["relation"],
                )
                for name in ("landing_region", "support_gap_m"):
                    if name in snapshot:
                        metrics[name] = snapshot[name]
            if "boundary_frames" in snapshot:
                metrics["boundary_frames"] = snapshot["boundary_frames"]
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": metrics,
            }
        if not historical_record and selected.operation_evidence is not None:
            snapshot["record"]["group_id"] = selected.operation_evidence["operation"][
                "operation_id"
            ]
        record = snapshot["record"]
        if acquisition is not None:
            native = (
                record["measurements"]["native_support"]
                if selected.evidence_scope
                in {"native_supported_placement", "native_supported_landing_region"}
                else record["measurements"]
            )
            if selected.evidence_scope == "native_supported_placement" and acquisition[
                "timestamp"
            ] != native.get("observed_at"):
                raise ValueError(
                    "Strong support channel and native measurements have different observation times."
                )
            if not _bool_rows(native.get("supported"), size) or any(
                available and value != supported
                for available, value, supported in zip(
                    availability,
                    acquisition["values"],
                    native["supported"],
                    strict=True,
                )
            ):
                raise ValueError(
                    "Support channel values contradict the native measurement rows."
                )
        if record.get("uid") != selected.object_id or record.get("part_id") is not None:
            raise ValueError("Placement record names another physical subject.")
        if (
            selected.operation_evidence is not None
            and selected.operation_evidence["family"] == "E5"
        ):
            from ._placement_linkage import transport_release

            released, release_available, receipt, release_call = transport_release(
                selected, snapshot, program, boundary
            )
        else:
            released, release_available, receipt, release_call = measured_release(
                release_segment["metadata"]["runtime"],
                boundary["metadata"]["runtime"],
                object_uid=selected.object_id,
                resource=selected.resource_id,
                invocation_id=selected.release_invocation_id,
                num_envs=size,
            )
        if not historical_record and selected.evidence_scope == "native_stable_support":
            record["measurements"].update(
                released=released, release_timestamp=receipt["timestamp"]
            )
        if selected.operation_evidence is not None:
            from ._placement_linkage import qualify_linkage

            qualify_linkage(selected, snapshot, program, receipt)
        for row in range(size):
            if not release_available[row]:
                availability[row] = False
                errors[row] = "Measured release acquisition is unavailable."
        if selected.evidence_scope == "native_supported_landing_region":
            from ._legacy_region_evidence import legacy_region_flags

            flags = legacy_region_flags(
                selected, snapshot, boundary, release_call, receipt, released, program
            )
        elif selected.directional_geometry is not None:
            from ._directional_evidence import directional_flags

            metrics = record["measurements"]
            if historical_record and metrics.get("released") != released:
                raise ValueError(
                    "Directional evidence differs from its exact measured release."
                )
            flags = directional_flags(
                selected, metrics, released, boundary, program, family=family
            )
        elif selected.evidence_scope == "measured_planar_landing_region":
            from ._planar_evidence import planar_flags

            metrics = record["measurements"]
            if (
                metrics.get("reference_uid") != selected.reference_id
                or metrics.get("relation") != predicate.arguments["relation"]
                or historical_record
                and metrics.get("released") != released
                or selected.observation_kind == "recorded"
                and metrics.get("boundary_frames") != boundary["end_steps"]
            ):
                raise ValueError(
                    "Planar evidence differs from its exact release or boundary."
                )
            flags, planar_available = planar_flags(
                selected, metrics, released, predicate.arguments["relation"]
            )
            if (
                historical_record
                and record.get("available", [True] * size) != planar_available
            ):
                raise ValueError(
                    "Planar containment availability differs from its native measurements."
                )
            availability = [
                a and b for a, b in zip(availability, planar_available, strict=True)
            ]
            errors = [
                (
                    None
                    if value
                    else "Projected footprint cannot resolve this raised 3D containment."
                )
                for value in planar_available
            ]
        elif selected.evidence_scope == "native_contact_support":
            flags = _legacy_flags(
                selected,
                snapshot,
                released,
                boundary,
                support_boundary,
                require_claims=historical_record,
            )
        elif selected.evidence_scope == "native_supported_placement":
            from ._support_contract import release_window
            from ._strong_support import (
                _strong_support_trace,
                _strong_support_flags,
                _region_poses_match,
            )
            import torch

            metrics = record["measurements"]
            expected = {
                "scope": selected.evidence_scope,
                "support_contract_revision": 1,
                "placement_node_id": selected.release_segment_id,
                "support_boundary_node_id": selected.support_segment_id,
                "boundary_node_id": selected.boundary_segment_id,
                "stability_preset": selected.stability_preset,
                "observation_kind": selected.observation_kind,
                "reference_uid": selected.reference_id,
                "relation": predicate.arguments["relation"],
            }
            if any(metrics.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "Strong placement differs from its selected declaration/occurrences."
                )
            trace = _strong_support_trace(
                support_boundary,
                selected.condition,
                selected.stability_preset,
                selected.object_id,
                selected.reference_id,
                size,
            )
            window = release_window(release_call, receipt, size)
            if historical_record and (
                metrics.get("released") != released
                or metrics.get("release_window") != window
            ):
                raise ValueError("Strong placement has a foreign release phase window.")
            native = metrics["native_support"]
            if not _region_poses_match(
                native,
                torch.tensor(snapshot["boundary_poses"][selected.object_id]),
                torch.tensor(snapshot["boundary_poses"][selected.reference_id]),
                size,
            ):
                raise ValueError(
                    "Strong support differs from its exact boundary poses."
                )
            if selected.observation_kind == "recorded":
                if (
                    native != trace["measurements"]["native_support"]
                    or metrics.get("boundary_frames") != boundary.get("end_steps")
                    or selected.condition["kind"] == "supported_region"
                    and metrics.get("landing_region")
                    != trace["measurements"].get("landing_region")
                ):
                    raise ValueError(
                        "Recorded strong support differs from its own policy snapshot."
                    )
            physical = _strong_support_flags(
                native,
                metrics.get("landing_region"),
                selected.condition,
                window,
                trace,
                selected.object_id,
                selected.reference_id,
                size,
                metrics.get("support_gap_m"),
            )
            flags = [
                support and release
                for support, release in zip(physical, released, strict=True)
            ]
        elif selected.evidence_scope == "native_stable_support":
            from ._support_evidence import stable_support_flags

            metrics = record["measurements"]
            if (
                metrics.get("scope") != "native_stable_support"
                or metrics.get("entity") != selected.object_id
                or metrics.get("reference") != selected.reference_id
                or metrics.get("reference_uid") != selected.reference_id
                or metrics.get("relation") != "on"
                or metrics.get("observation_kind") != selected.observation_kind
                or metrics.get("placement_node_id") != selected.release_segment_id
                or metrics.get("released") != released
                or metrics.get("release_timestamp") != receipt.get("timestamp")
                or metrics.get("boundary_frames") != boundary.get("end_steps")
                or metrics.get("end_step") != boundary["end_steps"][0]
            ):
                raise ValueError(
                    "Stable support belongs to another release or boundary occurrence."
                )
            physical = stable_support_flags(
                metrics, snapshot["contact_timeline"], condition=selected.condition
            )
            flags = [
                support and release
                for support, release in zip(physical, released, strict=True)
            ]
        else:
            raise ValueError(
                "The selected physical support scope is not available in this evaluator."
            )
        for row in range(size):
            if not execution_available[row]:
                availability[row] = False
                errors[row] = (
                    "The declared execution prefix is not fully active and successful."
                )
        accepted = record.get("accepted")
        if (historical_record or accepted is not None) and (
            not _bool_rows(accepted, size)
            or any(
                available and actual != expected
                for actual, expected, available in zip(
                    accepted, flags, availability, strict=True
                )
            )
        ):
            raise ValueError(
                "Claimed placement values differ from measured support/release."
            )
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
    ) as error:
        reason = str(error) or type(error).__name__
    return tuple(
        PredicateEvaluation(
            predicate_id=predicate.id,
            predicate=predicate.name,
            status=(
                EvaluationStatus.UNAVAILABLE
                if reason is not None or not availability[row]
                else (
                    EvaluationStatus.SATISFIED
                    if flags[row]
                    else EvaluationStatus.FAILED
                )
            ),
            observed=(
                None if reason is not None or not availability[row] else flags[row]
            ),
            expected=predicate.expected,
            reason=reason
            or (
                errors[row] or "Physical support acquisition is unavailable."
                if not availability[row]
                else (
                    "Explicit legacy support and measured release checked; no strong window qualification is implied."
                    if selected.evidence_scope == "native_contact_support"
                    else (
                        "Explicit planar footprint/gap and measured release checked; native contact support is not inferred."
                        if selected.evidence_scope == "measured_planar_landing_region"
                        else "The selected sampled support window and measured release satisfy their declared evidence contract."
                    )
                )
            ),
            evidence_refs=reference,
        )
        for row in range(size)
    )


def evaluate_placement_evidence_report(
    spec: TaskSpec | Mapping[str, Any],
    bindings: Sequence[SupportedPlacementEvidenceBinding | Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    program: Mapping[str, Any],
    num_envs: int,
) -> tuple[EvaluationReport, ...]:
    """Build ordinary TaskSpec reports from independently verified boundaries.

    Args:
        spec: Normative specification; predicates outside this slice stay missing.
        bindings: Unique goal/milestone declarations for the same ordered rows
            and episode identities. No final observation fills a past milestone.
        evidence: Physical/execution snapshots keyed by declared predicate ID.
        program: Exact frozen serialized program.
        num_envs: Number of native environment rows being reported.

    Returns:
        One standard EvaluationReport per row. Sparse boundaries never establish
        continuous invariants or complete temporal windows. Certificate readiness
        is decided solely by the existing TaskSpec evaluator.

    Raises:
        ValueError: Binding rows, episode ownership or observation identities clash.
    """
    from .evidence_report import evaluate_physical_evidence_report

    selected = tuple(
        SupportedPlacementEvidenceBinding.from_dict(item) for item in bindings
    )
    return evaluate_physical_evidence_report(
        spec, selected, evidence, program=program, num_envs=num_envs
    )
