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
]

_SCOPES = frozenset(
    {
        "native_contact_support",
        "native_stable_support",
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
    condition: Mapping[str, Any] = field(default_factory=dict)
    required_execution: tuple[Mapping[str, Any], ...] = ()
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    binding_hash: str = ""

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
            or self.condition.get("reference") != self.reference_id
        ):
            raise ValueError("Condition endpoints differ from the normative binding.")
        if self.evidence_scope != "native_contact_support":
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
        if self.condition.get("kind") == "supported_region":
            raise ValueError(
                "Region placement requires its complete footprint evidence contract; it cannot use a contact-only scope."
            )
        revision = self.condition.get("support_contract_revision")
        if revision is not None:
            raise ValueError(
                "A revision-1 declaration requires its separate strong evidence evaluator; it cannot use a legacy or version-2 timeline scope."
            )
        if self.evidence_scope == "native_stable_support" and len(ids) != 1:
            raise ValueError(
                "The version-2 stable support contract has one native row."
            )
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Placement evidence binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.placement_evidence_binding/v1",
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
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
        if raw.pop("schema_version", None) != "task_spec.placement_evidence_binding/v1":
            raise ValueError("Unsupported placement evidence binding schema.")
        return cls(**raw)


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
    support_boundary: Mapping[str, Any],
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
    from .expressions import EvaluationStatus, PredicateEvaluation
    from .spec import TaskSpec
    from ._release_evidence import measured_release

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
    if (
        predicate.name != "semantic_goal"
        or predicate.arguments.get("relation") != "on"
        or predicate.operator != "eq"
        or predicate.expected is not True
    ):
        raise ValueError(
            "This placement evidence contract requires an on-relation goal."
        )
    roles = {role.name: role for role in normalized.roles}
    size = len(selected.env_ids)
    reason = None
    availability = [True] * size
    errors: list[str | None] = [None] * size
    flags: list[bool] = [False] * size
    execution_available = [True] * size
    reference = ()
    try:
        if any(
            roles[predicate.arguments[key]].quantifier != "one"
            for key in ("subject", "reference")
        ):
            raise ValueError("Complete selector-universe evidence was not provided.")
        snapshot = json_snapshot(evidence)
        if not isinstance(snapshot, dict):
            raise TypeError("Physical evidence must be a JSON mapping.")
        reference = (selected.binding_hash, canonical_hash(snapshot))
        if selected.episode_ids is not None:
            if snapshot.get("episode_ids") != list(selected.episode_ids) or any(
                type(value) is not int for value in snapshot.get("episode_ids", [])
            ):
                raise ValueError(
                    "Physical support belongs to another declared episode."
                )
        elif "record" not in snapshot:
            raise ValueError(
                "Raw Lab observations require an explicit episode binding."
            )
        acquisition = snapshot.get("support_acquisition")
        if acquisition is not None:
            if (
                not isinstance(acquisition, Mapping)
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
                    error is not None and (type(error) is not str or not error)
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
        for name, path in (
            (selected.release_segment_id, selected.release_source_path),
            (selected.support_segment_id, selected.support_source_path),
            (selected.boundary_segment_id, selected.boundary_source_path),
        ):
            segment = segments[name]
            if segment["metadata"]["program_segment_source_path"] != list(path):
                raise ValueError("Evidence belongs to another execution occurrence.")
            selected_segments.append(segment)
        release_segment, support_boundary, boundary = selected_segments
        historical_record = "record" in snapshot
        if not historical_record:
            native = json_snapshot(snapshot["native_support"])
            metrics = {
                **native,
                "reference_uid": selected.reference_id,
                "relation": predicate.arguments["relation"],
            }
            if "boundary_frames" in snapshot:
                metrics["boundary_frames"] = snapshot["boundary_frames"]
            snapshot["record"] = {
                "uid": selected.object_id,
                "part_id": None,
                "measurements": metrics,
            }
        record = snapshot["record"]
        if record.get("uid") != selected.object_id or record.get("part_id") is not None:
            raise ValueError("Placement record names another physical subject.")
        released, release_available, receipt, _ = measured_release(
            release_segment["metadata"]["runtime"],
            boundary["metadata"]["runtime"],
            object_uid=selected.object_id,
            resource=selected.resource_id,
            invocation_id=selected.release_invocation_id,
            num_envs=size,
        )
        for row in range(size):
            if not release_available[row]:
                availability[row] = False
                errors[row] = "Measured release acquisition is unavailable."
        if selected.evidence_scope == "native_contact_support":
            flags = _legacy_flags(
                selected,
                snapshot,
                released,
                boundary,
                support_boundary,
                require_claims=historical_record,
            )
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
                    else "The selected sampled support window and measured release satisfy their declared evidence contract."
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
    from .spec import TaskSpec
    from .validation import evaluate_task_spec

    normalized = TaskSpec.from_dict(spec)
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    selected = tuple(
        SupportedPlacementEvidenceBinding.from_dict(item) for item in bindings
    )
    if len({item.predicate_id for item in selected}) != len(selected):
        raise ValueError("Evidence bindings must identify unique predicates.")
    if any(item.env_ids != tuple(range(num_envs)) for item in selected):
        raise ValueError("Evidence bindings have another environment batch.")
    if len({item.episode_ids for item in selected}) > 1:
        raise ValueError(
            "Evidence bindings cannot mix declared or historical episodes."
        )
    goals = {item.id: item for item in normalized.goal}
    milestones = {
        item.id: item
        for milestone in normalized.milestones
        for item in milestone.achieve
    }
    goal_rows: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    events: list[dict[int, dict[str, Any]]] = [{} for _ in range(num_envs)]

    def store_observation(
        target: dict[str, Any], key: str, packet: dict[str, Any]
    ) -> None:
        previous = target.get(key)
        if previous is not None:
            if any(
                previous.get(field) != packet.get(field)
                for field in ("status", "value")
            ):
                raise ValueError(
                    "Predicate bindings assign conflicting observations to one key."
                )
            packet["evidence_refs"] = sorted(
                set(previous["evidence_refs"]) | set(packet["evidence_refs"])
            )
        target[key] = packet

    for binding in selected:
        snapshot = json_snapshot(evidence.get(binding.predicate_id, {}))
        outcomes = evaluate_supported_placement_evidence(
            normalized, binding, snapshot, program=program
        )
        predicate = goals.get(
            binding.predicate_id, milestones.get(binding.predicate_id)
        )
        assert predicate is not None  # The public verifier checked this identity.
        frame_indices = None
        if binding.predicate_id in milestones:
            frames = snapshot.get("boundary_frames")
            if frames is None:
                frames = (
                    snapshot.get("record", {})
                    .get("measurements", {})
                    .get("boundary_frames")
                )
            if (
                isinstance(frames, list)
                and len(frames) == num_envs
                and all(type(frame) is int and frame > 0 for frame in frames)
            ):
                frame_indices = frames
        for row, outcome in enumerate(outcomes):
            packet = {
                "status": outcome.status.value,
                "value": outcome.observed,
                "reason": outcome.reason,
                "evidence_refs": list(outcome.evidence_refs),
            }
            key = predicate.observation or predicate.id
            if binding.predicate_id in goals:
                store_observation(goal_rows[row], key, packet)
            elif frame_indices is not None:
                store_observation(
                    events[row].setdefault(frame_indices[row], {}), key, packet
                )
    reports = []
    for row in range(num_envs):
        trajectory = [snapshot for _, snapshot in sorted(events[row].items())]
        for condition in normalized.temporal:
            for snapshot in trajectory:
                snapshot[condition.predicate.observation or condition.predicate.id] = {
                    "status": "unavailable",
                    "reason": "No complete control-step trajectory.",
                }
        reports.append(
            evaluate_task_spec(
                normalized,
                {"init": {}, "goal": goal_rows[row], "invariants": {}},
                trajectory=trajectory,
            )
        )
    return tuple(reports)
