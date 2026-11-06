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
"""Sealed normative qualification of operation-bound Press/Twist evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, TYPE_CHECKING
from .canonicalization import canonical_hash, json_snapshot
from .evidence import _digest, _identifier, _path

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation

__all__ = ["ControlEvidenceBinding", "evaluate_control_evidence"]


@dataclass(frozen=True, slots=True)
class ControlEvidenceBinding:
    """Seal an exact control route and occurrence independently of live factories.

    The original operation-window receipt is a different contract from a final
    joint position. Historical replay must be selected explicitly. Raw control
    acquisition is rejected until its complete operation-window channel is
    implemented; adding a legacy record cannot silently select this verifier.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    entity_id: str
    control_kind: str
    observation_kind: str
    operation: Mapping[str, Any]
    evidence_format: str
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in ("predicate_id", "entity_id"):
            _identifier(getattr(self, name), name)
        if self.control_kind not in {"press", "twist"} or self.observation_kind not in {
            "final",
            "recorded",
        }:
            raise ValueError(
                "Control evidence needs an explicit supported kind and observation scope."
            )
        if self.evidence_format != "historical_record":
            raise ValueError(
                "Raw control evidence requires the complete operation-window acquisition contract, which is not yet available."
            )
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(value) is not int for value in ids)
        ):
            raise ValueError(
                "Control evidence requires ordered native environment rows."
            )
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(value) is not int or value < 0 for value in episodes
            ):
                raise ValueError(
                    "Control evidence episode identities must match native rows."
                )
            object.__setattr__(self, "episode_ids", episodes)
        operation = json_snapshot(self.operation)
        if not isinstance(operation, dict) or set(operation) != {
            "route_payload",
            "terminal_segment_id",
            "terminal_source_path",
            "preset",
            "required_execution",
        }:
            raise ValueError(
                "Control declarations require exact route, terminal policy and occurrence prefix fields."
            )
        route = operation["route_payload"]
        if (
            not isinstance(route, Mapping)
            or route["binding"]["object_id"] != self.entity_id
        ):
            raise ValueError("Control route names another grounded entity.")
        _identifier(route["operation_id"], "operation_id")
        for name in ("joint", "link"):
            _identifier(route["binding"][name], name)
        _digest(route["binding"]["source_sha256"], "source_sha256")
        criteria = (
            {"contact_backed_coarse_turn", "calibrated_setting"}
            if self.control_kind == "twist"
            else {"contact_backed_press"}
        )
        if route.get("success_criterion", "contact_backed_press") not in criteria:
            raise ValueError(
                "This control criterion has no source qualification contract."
            )
        for name in ("terminal_segment_id", "preset"):
            _identifier(operation[name], name)
        _path(operation["terminal_source_path"], "terminal_source_path")
        required = operation["required_execution"]
        if not isinstance(required, list) or not required:
            raise ValueError("Control evidence requires its full execution prefix.")
        for item in required:
            if not isinstance(item, Mapping) or set(item) != {
                "segment_id",
                "source_path",
            }:
                raise ValueError(
                    "Control occurrence declarations require exact identities."
                )
            _identifier(item["segment_id"], "segment_id")
            _path(item["source_path"], "source_path")
        if len({item["segment_id"] for item in required}) != len(required) or required[
            -1
        ] != {
            "segment_id": operation["terminal_segment_id"],
            "source_path": operation["terminal_source_path"],
        }:
            raise ValueError(
                "Control prefix must identify unique occurrences and its exact terminal boundary."
            )
        object.__setattr__(self, "operation", operation)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Control evidence binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.control_evidence_binding/v1",
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
        cls, value: ControlEvidenceBinding | Mapping[str, Any]
    ) -> ControlEvidenceBinding:
        """Rebuild and verify the original identity of a declaration or instance.

        Args:
            value: Serialized binding or an existing instance.

        Returns:
            Independently owned normalized binding; changed data is never re-signed.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None)
            != "task_spec.control_evidence_binding/v1"
        ):
            raise ValueError("Unsupported control evidence binding version.")
        return cls(**raw)


def evaluate_control_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: ControlEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Qualify original control operation receipts without inferring activation.

    Args:
        spec: Frozen normative TaskSpec.
        binding: Explicit historical format and exact source route/occurrence.
        evidence: Original record plus runtime segments, or unavailable reason.
        program: Complete serialized program matching the binding digest.

    Returns:
        Per-environment predicate outcomes. The original control qualification
        requires all active rows to complete; absent or rejected operation proof
        remains unavailable. A Press contact event is not device activation, and
        a coarse Twist is not silently promoted to calibrated-setting success.
    """
    from .spec import TaskSpec
    from .expressions import EvaluationStatus, PredicateEvaluation
    from ._control_evidence import qualified_control_record

    selected = ControlEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("Control evidence has another normative/program identity.")
    predicates = [(item, False) for item in normalized.goal] + [
        (item, True)
        for milestone in normalized.milestones
        for item in milestone.achieve
    ]
    matches = [
        (item, past) for item, past in predicates if item.id == selected.predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Control binding must identify one exact predicate.")
    predicate, past = matches[0]
    expected = (
        ("E8", "semantic_goal")
        if selected.control_kind == "twist"
        else ("E9", "pressed")
    )
    if (
        (predicate.arguments.get("task_type"), predicate.name) != expected
        or predicate.operator != "eq"
        or predicate.expected is not True
        or selected.observation_kind != ("recorded" if past else "final")
    ):
        raise ValueError(
            "Control proof differs from its declared predicate or boundary scope."
        )
    reason = None
    refs = ()
    try:
        role = next(
            role
            for role in normalized.roles
            if role.name == predicate.arguments["subject"]
        )
        if role.quantifier != "one":
            raise ValueError("Complete selector-universe evidence was not provided.")
        snapshot = json_snapshot(evidence)
        refs = (selected.binding_hash, canonical_hash(snapshot))
        if "unavailable" in snapshot:
            raise ValueError(str(snapshot["unavailable"]))
        if "record" not in snapshot or any(
            name in snapshot
            for name in ("acquisition", "control_acquisition", "physical_evidence")
        ):
            raise ValueError(
                "Historical control replay cannot consume a new raw acquisition overlay."
            )
        if selected.episode_ids is not None and snapshot.get("episode_ids") != list(
            selected.episode_ids
        ):
            raise ValueError("Control snapshots belong to another declared episode.")
        expected_record = qualified_control_record(
            {"control_kind": selected.control_kind, **selected.operation},
            {"segments": snapshot["segments"]},
            program,
            len(selected.env_ids),
        )
        if snapshot["record"] != expected_record:
            raise ValueError(
                "Historical control claims differ from their exact measured operation receipts."
            )
    except (
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        StopIteration,
    ) as error:
        reason = str(error) or type(error).__name__
    return tuple(
        PredicateEvaluation(
            predicate.id,
            predicate.name,
            EvaluationStatus.UNAVAILABLE if reason else EvaluationStatus.SATISFIED,
            observed=None if reason else True,
            expected=True,
            reason=reason
            or "The declared original operation criterion is supported by its own complete physical receipts.",
            evidence_refs=refs,
        )
        for _ in selected.env_ids
    )
