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

"""Data-only orientation and articulation endpoint evidence qualification."""

from __future__ import annotations

from embodichain.compute._evaluation import _within_target

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from typing import Any, TYPE_CHECKING

from .canonicalization import canonical_hash, json_snapshot
from .evidence import _identifier, _digest, _path, _bool_rows

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation

__all__ = ["ConditionEvidenceBinding", "evaluate_condition_evidence"]


def _axis(value: Any, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"{name} must contain three finite real values.")
    norm = math.sqrt(sum(v * v for v in value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{name} must be finite and nonzero.")
    return tuple(float(v / norm) for v in value)


@dataclass(frozen=True, slots=True)
class ConditionEvidenceBinding:
    """Sealed source selection and grounded criteria for one state predicate.

    Current state and historical policy receipts have different contracts.
    Episode identifiers are required before raw state can be evaluated. A new
    binding must be constructed for intentionally changed declarations; existing
    instances are always snapshotted and their content identity is rechecked.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    entity_id: str
    native_uid: str
    evidence_kind: str
    observation_kind: str
    condition: Mapping[str, Any]
    evidence_format: str = "raw_snapshot"
    boundary_segment_id: str | None = None
    boundary_source_path: tuple[str | int, ...] = ()
    policy_segment_id: str | None = None
    policy_source_path: tuple[str | int, ...] = ()
    stability_preset: str | None = None
    required_execution: tuple[Mapping[str, Any], ...] = ()
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in ("predicate_id", "entity_id", "native_uid"):
            _identifier(getattr(self, name), name)
        if self.evidence_kind not in {"orientation", "joint_endpoint"}:
            raise ValueError("Unsupported state evidence kind.")
        if self.observation_kind not in {"final", "recorded"}:
            raise ValueError("State observations must be final or recorded.")
        if self.evidence_format not in {"raw_snapshot", "historical_record"}:
            raise ValueError("State evidence format must be explicit.")
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(v) is not int for v in ids)
        ):
            raise ValueError("State evidence requires ordered native environment rows.")
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(v) is not int or v < 0 for v in episodes
            ):
                raise ValueError(
                    "State episode identities must match the selected rows."
                )
            object.__setattr__(self, "episode_ids", episodes)
        if not isinstance(self.condition, Mapping):
            raise TypeError("State criteria must be a plain declaration.")
        condition = json_snapshot(dict(self.condition))
        if condition.get("entity") != self.entity_id:
            raise ValueError("State criteria refer to another entity.")
        if self.evidence_kind == "orientation":
            if condition.get("kind") not in {"upright", "hold"}:
                raise ValueError(
                    "Orientation evidence requires an explicit axis condition."
                )
            _axis(condition.get("local_axis"), "local_axis")
            if condition.get("world_axis") is not None:
                _axis(condition["world_axis"], "world_axis")
            if type(condition.get("axis_direction_free", False)) is not bool:
                raise ValueError("Axis-direction freedom must be a boolean.")
            threshold = condition.get("minimum_alignment", 0.8660254037844386)
            if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("Minimum alignment must be in [0,1].")
        else:
            _identifier(condition.get("joint"), "joint")
            if condition.get("target_state") not in {"open", "closed"}:
                raise ValueError(
                    "An endpoint criterion must retain its semantic target state."
                )
            for name in ("target", "tolerance"):
                if type(condition.get(name)) not in (int, float) or not math.isfinite(
                    condition[name]
                ):
                    raise ValueError(
                        "Endpoint criteria must contain finite target/tolerance values."
                    )
            if condition["tolerance"] < 0:
                raise ValueError("Endpoint tolerance cannot be negative.")
            if self.observation_kind == "recorded" and any(
                condition.get(name, expected) != expected
                for name, expected in (
                    ("duration", 1.0),
                    ("timeout", 5.0),
                    ("position_drift", 0.001),
                )
            ):
                raise ValueError(
                    "Endpoint milestone qualification retains the original one-second, five-second timeout and 0.001 joint-drift contract."
                )
        object.__setattr__(self, "condition", condition)
        for name in ("boundary_segment_id", "policy_segment_id", "stability_preset"):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, name)
        for name in ("boundary_source_path", "policy_source_path"):
            value = getattr(self, name)
            object.__setattr__(self, name, _path(value, name) if value else ())
        required = tuple(json_snapshot(item) for item in self.required_execution)
        for item in required:
            if not isinstance(item, Mapping) or set(item) != {
                "segment_id",
                "source_path",
            }:
                raise ValueError(
                    "Execution requirements need exact occurrence identities."
                )
            _identifier(item["segment_id"], "execution segment")
            _path(item["source_path"], "execution path")
        if len({item["segment_id"] for item in required}) != len(required):
            raise ValueError("Execution occurrence identities must be unique.")
        object.__setattr__(self, "required_execution", required)
        if self.observation_kind == "recorded" and (
            not required
            or any(
                getattr(self, name) is None
                for name in (
                    "boundary_segment_id",
                    "policy_segment_id",
                    "stability_preset",
                )
            )
            or not self.boundary_source_path
            or not self.policy_source_path
        ):
            raise ValueError(
                "A recorded state requires its boundary, policy and execution prefix."
            )
        if self.observation_kind == "recorded":
            occurrence_paths = {
                item["segment_id"]: item["source_path"] for item in required
            }
            if any(
                occurrence_paths.get(name) != list(path)
                for name, path in (
                    (self.boundary_segment_id, self.boundary_source_path),
                    (self.policy_segment_id, self.policy_source_path),
                )
            ):
                raise ValueError(
                    "Recorded boundary and policy must belong to the complete required execution prefix."
                )
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "State evidence binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.condition_evidence_binding/v1",
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return an owned declaration while retaining its original identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: ConditionEvidenceBinding | Mapping[str, Any]
    ) -> ConditionEvidenceBinding:
        """Re-decode and verify an instance or serialized state evidence binding.

        Args:
            value: Exact declaration or previously constructed binding.

        Returns:
            Independently owned normalized declaration with the original valid hash.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None)
            != "task_spec.condition_evidence_binding/v1"
        ):
            raise ValueError("Unsupported state evidence binding schema.")
        return cls(**raw)


def _current_values(
    binding: ConditionEvidenceBinding, snapshot: Mapping[str, Any], size: int
) -> tuple[list[Any], list[bool], list[str | None]]:
    if binding.evidence_format == "historical_record":
        if "record" not in snapshot or "state_acquisition" in snapshot:
            raise ValueError("Historical state observations cannot use a raw overlay.")
        record = snapshot["record"]
        if record.get("uid") != binding.entity_id:
            raise ValueError("Historical state belongs to another entity.")
        key = "pose" if binding.evidence_kind == "orientation" else "joint_position"
        return record["measurements"][key], [True] * size, [None] * size
    if "record" in snapshot:
        raise ValueError("Raw state cannot downgrade to a historical record.")
    if binding.episode_ids is None or snapshot.get("episode_ids") != list(
        binding.episode_ids
    ):
        raise ValueError("Raw state belongs to an unbound or different episode.")
    if (
        snapshot.get("schema_version") != "simulation.state_observation/v1"
        or snapshot.get("entity_id") != binding.entity_id
        or snapshot.get("native_uid") != binding.native_uid
        or snapshot.get("observation_kind") != binding.observation_kind
    ):
        raise ValueError("Raw state has another declaration or observation kind.")
    channel = snapshot["state_acquisition"]
    name = (
        "rigid.pose"
        if binding.evidence_kind == "orientation"
        else "articulation.joint_position"
    )
    if (
        channel.get("evidence_id") != name
        or channel.get("value_dtype") not in {"torch.float32", "torch.float64"}
        or channel.get("env_ids") != list(binding.env_ids)
        or not _bool_rows(channel.get("valid"), size)
        or type(channel.get("timestamp")) not in (int, float)
        or not math.isfinite(channel["timestamp"])
        or channel["timestamp"] < 0
        or type(channel.get("observation_revision")) is not int
        or channel["observation_revision"] < 0
        or channel.get("clock_domain") != "simulation_time"
        or channel.get("revision_kind") != "collector_acquisition"
    ):
        raise ValueError("State acquisition has another channel, clock or row scope.")
    errors = channel.get("acquisition_errors")
    values = channel.get("values")
    if (
        not isinstance(values, list)
        or len(values) != size
        or not isinstance(errors, list)
        or len(errors) != size
        or any(
            error is not None and (type(error) is not str or not error.strip())
            for error in errors
        )
        or any(
            valid and error is not None
            for valid, error in zip(channel["valid"], errors, strict=True)
        )
    ):
        raise ValueError(
            "State values/validity/errors are incomplete or contradictory."
        )
    if (
        binding.evidence_kind == "joint_endpoint"
        and snapshot.get("joint") != binding.condition["joint"]
    ):
        raise ValueError("State acquisition names another joint.")
    return values, channel["valid"], errors


def _successful_prefix(
    binding: ConditionEvidenceBinding, snapshot: Mapping[str, Any]
) -> tuple[dict[str, Any], list[bool]]:
    size = len(binding.env_ids)
    segments = {item["name"]: item for item in snapshot["segments"]}
    if len(segments) != len(snapshot["segments"]):
        raise ValueError("Duplicate state evidence occurrences.")
    eligible = [True] * size
    for required in binding.required_execution:
        segment = segments[required["segment_id"]]
        if segment["metadata"]["program_segment_source_path"] != required[
            "source_path"
        ] or segment["metadata"]["runtime"]["env_ids"] != list(binding.env_ids):
            raise ValueError(
                "State execution prefix has another occurrence or row scope."
            )
        if not _bool_rows(segment.get("active"), size) or not _bool_rows(
            segment.get("successes"), size
        ):
            raise ValueError("State execution prefix lacks explicit activity/outcomes.")
        eligible = [
            old and active and success
            for old, active, success in zip(
                eligible, segment["active"], segment["successes"], strict=True
            )
        ]
    boundary = segments[binding.boundary_segment_id]
    if boundary["metadata"]["program_segment_source_path"] != list(
        binding.boundary_source_path
    ) or snapshot.get("boundary_frames") != boundary.get("end_steps"):
        raise ValueError("Recorded state has another exact boundary slot.")
    return segments, eligible


def _orientation_values(
    binding: ConditionEvidenceBinding,
    values: list[Any],
    valid: list[bool],
    *,
    value_dtype: str = "torch.float32",
) -> tuple[list[bool], list[float | None]]:
    import torch
    from embodichain.compute.geometry.rigid import (
        pose_axis_alignment,
        validate_rigid_pose_batch,
    )

    cfg = binding.condition
    axis = _axis(cfg["local_axis"], "local_axis")
    world = (
        None
        if cfg.get("world_axis") is None
        else _axis(cfg["world_axis"], "world_axis")
    )
    result, measured = [], []
    for value, available in zip(values, valid, strict=True):
        if not available:
            result.append(False)
            measured.append(None)
            continue
        pose = torch.tensor(
            value,
            dtype=torch.float64 if value_dtype == "torch.float64" else torch.float32,
        )[None]
        validate_rigid_pose_batch(pose, 1)
        alignment = pose_axis_alignment(
            pose,
            axis,
            world_axis=world,
            direction_free=cfg.get("axis_direction_free", False),
        )[0]
        measured.append(float(alignment))
        result.append(
            bool(alignment >= cfg.get("minimum_alignment", 0.8660254037844386))
        )
    return result, measured


def _at_path(program: Mapping[str, Any], path: Sequence[str | int]) -> Any:
    value = program
    for part in path:
        value = value[part]
    return value


def _recorded_values(
    binding: ConditionEvidenceBinding,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
) -> tuple[list[bool], list[bool], list[str | None]]:
    """Check the declared original policy/validator occurrence, per native row."""
    size = len(binding.env_ids)
    raw = binding.evidence_format == "raw_snapshot"
    if raw:
        if (
            "record" in snapshot
            or binding.episode_ids is None
            or snapshot.get("episode_ids") != list(binding.episode_ids)
        ):
            raise ValueError(
                "Raw policy observations require their bound episode and cannot downgrade."
            )
    elif "record" not in snapshot or "state_acquisition" in snapshot:
        raise ValueError(
            "Historical policy observations require the explicit historical envelope."
        )
    if raw and (
        snapshot.get("schema_version") != "simulation.state_observation/v1"
        or snapshot.get("entity_id") != binding.entity_id
        or snapshot.get("native_uid") != binding.native_uid
        or snapshot.get("observation_kind") != "recorded"
    ):
        raise ValueError(
            "Recorded state has another native declaration or observation kind."
        )
    if not raw and snapshot["record"].get("uid") != binding.entity_id:
        raise ValueError("Historical state belongs to another entity.")
    segments, eligible = _successful_prefix(binding, snapshot)
    segment = segments[binding.policy_segment_id]
    path = list(binding.policy_source_path)
    if segment["metadata"]["program_segment_source_path"] != path or segment[
        "metadata"
    ]["runtime"]["env_ids"] != list(binding.env_ids):
        raise ValueError("Recorded policy has another occurrence or environment scope.")
    declared = _at_path(program, path)
    matches = [
        (index, item)
        for index, item in enumerate(declared.get("post", []))
        if item.get("kind") == "wait_stable"
        and item.get("entity") == binding.entity_id
        and item.get("preset") == binding.stability_preset
    ]
    if len(matches) != 1:
        raise ValueError(
            "The selected occurrence must declare exactly one matching policy."
        )
    index, _ = matches[0]
    policies = [
        item
        for item in segment["metadata"]["post_policies"]
        if item.get("policy_index") == index
    ]
    if len(policies) != 1:
        raise ValueError("Recorded policy receipt is missing or ambiguous.")
    policy = policies[0]
    if policy.get("kind") != "wait_stable" or policy.get("source_path") != path + [
        "post",
        index,
    ]:
        raise ValueError("Recorded policy has another source path or kind.")
    stable = policy["result"]
    masks = [policy.get("result_mask")]
    if binding.evidence_kind == "orientation":
        if (
            stable.get("kind") != "upright"
            or stable.get("preset") != binding.stability_preset
        ):
            raise ValueError(
                "Recorded orientation requires its original upright policy."
            )
        alignments = stable["measurements"]["alignment"]
        if len(alignments) != size or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in alignments
        ):
            raise ValueError("Recorded alignment measurements are unavailable.")
        masks.append(stable.get("accepted_mask"))
        flags = [
            value >= binding.condition.get("minimum_alignment", 0.8660254037844386)
            for value in alignments
        ]
        available = [True] * size
        errors = [None] * size
        if raw:
            values, available, errors = _current_values(binding, snapshot, size)
            actual, measured = _orientation_values(
                binding,
                values,
                available,
                value_dtype=snapshot["state_acquisition"]["value_dtype"],
            )
            for row in range(size):
                if available[row] and (
                    actual[row] != flags[row]
                    or not math.isclose(
                        measured[row], alignments[row], rel_tol=1e-6, abs_tol=1e-12
                    )
                ):
                    available[row] = False
                    errors[row] = (
                        "Recorded boundary pose differs from the measured policy alignment."
                    )
    else:
        operation = _at_path(program, binding.required_execution[0]["source_path"])
        call = operation["steps"]["call"]
        if (
            call.get("call_id") != "gen_sim.articulation_slide"
            or call.get("arguments", {}).get("object") != binding.entity_id
            or call["arguments"].get("part") != binding.condition.get("part_id")
            or call["arguments"].get("state") != binding.condition["target_state"]
        ):
            raise ValueError(
                "Endpoint milestone has another declared semantic operation or target state."
            )
        validators = declared.get("validators", [])
        validation = segment["metadata"]["validation"]
        traces = validation["validators"]
        if (
            len(validators) != 1
            or len(traces) != 1
            or len(declared.get("post", [])) != 1
            or len(segment["metadata"]["post_policies"]) != 1
        ):
            raise ValueError(
                "Endpoint qualification requires the original single validator/policy pair."
            )
        validator, trace = validators[0], traces[0]
        result = trace["result"]
        joint = binding.condition["joint"]
        target, tolerance = binding.condition["target"], binding.condition["tolerance"]
        if (
            validator.get("kind") != "articulation_joint_position"
            or validator.get("articulation") != binding.entity_id
            or validator.get("joint") != joint
            or validation.get("env_ids") != list(binding.env_ids)
            or trace.get("validator_index") != 0
            or trace.get("kind") != validator["kind"]
            or trace.get("source_path") != path + ["validators", 0]
            or result.get("source_path") != trace["source_path"]
            or result.get("kind") != validator["kind"]
            or result.get("articulation_id") != binding.entity_id
            or result.get("joint") != joint
            or result.get("env_ids") != list(binding.env_ids)
            or stable.get("joint") != joint
            or stable.get("evidence") != "joint_position_and_stability"
            or stable.get("stable_seconds") != 1.0
            or stable.get("target") != target
            or stable.get("tolerance") != tolerance
        ):
            raise ValueError(
                "Endpoint receipts differ from their declared joint/window/target."
            )
        minimum, maximum = target - tolerance, target + tolerance
        if (
            validator.get("minimum_position") != minimum
            or result.get("minimum_position") != minimum
            or validator.get("maximum_position") != maximum
            or result.get("maximum_position") != maximum
        ):
            raise ValueError(
                "Endpoint validator bounds differ from the original policy target."
            )
        positions, observed = result["joint_position"], stable["observed"]
        if len(positions) != size or len(observed) != size:
            raise ValueError("Endpoint measurements have another row count.")
        available = [
            all(
                type(value) in (int, float) and math.isfinite(value)
                for value in (positions[row], observed[row])
            )
            for row in range(size)
        ]
        errors = [
            None if value else "Joint observations are unavailable."
            for value in available
        ]
        flags = [
            ok
            and minimum <= positions[row] <= maximum
            and _within_target(observed[row], target, tolerance)
            for row, ok in enumerate(available)
        ]
        masks.extend(
            [
                validation.get("accepted_mask"),
                trace.get("result_mask"),
                result.get("accepted_mask"),
                stable.get("accepted"),
            ]
        )
        if raw:
            if (
                snapshot.get("joint") != joint
                or snapshot.get("policy_result") != stable
            ):
                raise ValueError(
                    "Recorded joint collector and runtime receipts disagree."
                )
            channel = stable["joint_observation"]
            start, end, dt = (
                channel.get("start_step"),
                channel.get("end_step"),
                channel.get("step_dt"),
            )
            if (
                channel.get("evidence_id") != "articulation.joint_position"
                or channel.get("env_ids") != list(binding.env_ids)
                or channel.get("clock") != "environment_step"
                or channel.get("revision_scope")
                != "policy_window_qpos_acquisition_sequence"
                or type(start) is not int
                or type(end) is not int
                or start < 0
                or end < start
                or type(dt) not in (int, float)
                or not math.isfinite(dt)
                or dt <= 0
                or type(channel.get("observation_revision")) is not int
                or channel["observation_revision"] != end - start + 1
                or not _bool_rows(channel.get("valid"), size)
                or not isinstance(channel.get("acquisition_errors"), list)
                or len(channel["acquisition_errors"]) != size
                or channel.get("values") != observed
                or any(frame != end for frame in segment.get("end_steps", []))
                or len(segment.get("end_steps", [])) != size
            ):
                raise ValueError(
                    "Joint policy acquisition has another channel, clock, boundary or row scope."
                )
            times = (channel.get("timestamp"), channel.get("started_at"))
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in times
            ) or not math.isclose(
                times[0], times[1] + (end - start) * dt, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(
                    "Joint policy acquisition time differs from acknowledged steps."
                )
            if any(stable["accepted"]) and end - start < math.ceil(1.0 / dt):
                raise ValueError(
                    "Joint success lacks the original complete stability window."
                )
            for row in range(size):
                error = channel["acquisition_errors"][row]
                if error is not None and (type(error) is not str or not error.strip()):
                    raise ValueError(
                        "Joint acquisition error must describe unavailable observations."
                    )
                if channel["valid"][row] and error is not None:
                    raise ValueError(
                        "Joint acquisition validity contradicts its error."
                    )
                available[row] &= channel["valid"][row]
                errors[row] = error or errors[row]
    if any(not _bool_rows(mask, size) for mask in masks):
        raise ValueError("Policy/validator acceptance masks are incomplete.")
    for row in range(size):
        # B0 omits a milestone whose group or physical receipt failed; absence
        # remains unavailable instead of being promoted to a negative fact.
        if not eligible[row] or not flags[row] or not all(mask[row] for mask in masks):
            available[row] = False
            errors[row] = (
                errors[row]
                or "The complete execution prefix and original physical receipts did not establish this milestone."
            )
    return flags, available, errors


def _final_values(
    binding: ConditionEvidenceBinding, snapshot: Mapping[str, Any], size: int
) -> tuple[list[bool], list[bool], list[str | None]]:
    values, valid, errors = _current_values(binding, snapshot, size)
    flags = [False] * size
    if binding.evidence_kind == "orientation":
        flags, alignments = _orientation_values(
            binding,
            values,
            valid,
            value_dtype=snapshot.get("state_acquisition", {}).get(
                "value_dtype", "torch.float32"
            ),
        )
        if binding.evidence_format == "historical_record":
            claimed = snapshot["record"]["measurements"].get("alignment")
            if (
                not isinstance(claimed, list)
                or len(claimed) != size
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not math.isclose(value, measured, rel_tol=1e-6, abs_tol=1e-12)
                    for value, measured in zip(claimed, alignments, strict=True)
                )
            ):
                raise ValueError(
                    "Historical alignment differs from its pose/axis measurements."
                )
    else:
        if binding.evidence_format == "historical_record" and (
            snapshot["record"]["measurements"].get("joint_target")
            != binding.condition["target"]
            or snapshot["record"].get("part_id") != binding.condition.get("part_id")
        ):
            raise ValueError(
                "Historical joint values belong to another grounded target/part."
            )
        for row, (value, available) in enumerate(zip(values, valid, strict=True)):
            if not available:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Joint position measurement must be finite.")
        import torch

        # Native final capture used tensor subtraction/comparison, including its
        # dtype rounding at tolerance boundaries. Recorded receipts below retain
        # their distinct Python scalar comparison, exactly as B0 did.
        dtype = (
            torch.float64
            if snapshot.get("state_acquisition", {}).get("value_dtype")
            == "torch.float64"
            else torch.float32
        )
        measured = torch.tensor(
            [
                value if available else 0.0
                for value, available in zip(values, valid, strict=True)
            ],
            dtype=dtype,
        )
        accepted = _within_target(
            measured, binding.condition["target"], binding.condition["tolerance"]
        )
        flags = [
            bool(value) and available
            for value, available in zip(accepted.tolist(), valid, strict=True)
        ]
    if binding.evidence_format == "historical_record":
        accepted = snapshot["record"].get("accepted")
        if not _bool_rows(accepted, size) or accepted != flags:
            raise ValueError("Historical state claims differ from its measured values.")
    return flags, valid, errors


def evaluate_condition_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: ConditionEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Qualify one current or recorded orientation/joint predicate from raw facts.

    Args:
        spec: Normative TaskSpec snapshot.
        binding: Sealed state criteria and exact execution selection.
        evidence: Owned current state or original recorded policy snapshots.
        program: Exact serialized program identified by the binding.

    Returns:
        Per-environment predicate results, preserving unavailable rows. This does
        not infer program completion or certify unrelated task predicates.
    """
    normalized, selected, predicate = _condition_selection(
        spec, binding, program=program
    )
    try:
        role = {item.name: item for item in normalized.roles}[
            predicate.arguments["subject"]
        ]
        selector_error = (
            None
            if role.is_single_member
            else "Complete selector-universe evidence was not provided."
        )
    except (KeyError, StopIteration) as error:
        selector_error = str(error) or type(error).__name__
    return _evaluate_condition_member(
        selected, predicate, evidence, program=program, selector_error=selector_error
    )


def _condition_selection(spec, binding, *, program):
    """Validate original normative/program identity and state-predicate semantics."""
    from .spec import TaskSpec

    selected = ConditionEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("State evidence has another normative/program identity.")
    matches = [
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
    if len(matches) != 1:
        raise ValueError("State binding must select one declared predicate.")
    predicate = matches[0]
    milestone = any(
        item.id == predicate.id
        for part in normalized.milestones
        for item in part.achieve
    )
    if selected.observation_kind != ("recorded" if milestone else "final"):
        raise ValueError(
            "Current goals and recorded milestones have distinct evidence contracts."
        )
    if predicate.operator != "eq" or predicate.expected is not True:
        raise ValueError("State proof requires the declared positive predicate.")
    if selected.evidence_kind == "orientation":
        vertical = (
            predicate.name == "orientation"
            and predicate.arguments.get("orientation_goal") == "vertical"
        )
        if (
            predicate.name != "object_upright" and not vertical
        ) or predicate.arguments.get("task_type") not in {None, "E1", "E2"}:
            raise ValueError(
                "This orientation proof does not verify transfer or other task-family evidence."
            )
        if (
            predicate.arguments.get("task_type") == "E1"
            and milestone
            and predicate.name != "object_upright"
        ):
            raise ValueError(
                "The source recorded E1 orientation proof requires its upright predicate."
            )
        if selected.condition.get("axis_direction_free", False) is not vertical:
            raise ValueError("State proof has another axis-direction contract.")
    elif (
        predicate.name != "articulation_joint_near"
        or predicate.arguments.get("task_type") not in {None, "E6"}
        or predicate.arguments.get("target_state") != selected.condition["target_state"]
    ):
        raise ValueError("This endpoint proof has another task family or target state.")
    return normalized, selected, predicate


def _evaluate_condition_member(
    selected: ConditionEvidenceBinding,
    predicate: Any,
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    selector_error: str | None = None,
) -> tuple[PredicateEvaluation, ...]:
    """Qualify acquired state after the caller validates its complete selector."""
    from .expressions import PredicateEvaluation, EvaluationStatus

    size = len(selected.env_ids)
    flags = [False] * size
    valid = [True] * size
    errors = [None] * size
    reason = None
    refs = ()
    try:
        if selector_error is not None:
            raise ValueError(selector_error)
        snapshot = json_snapshot(evidence)
        refs = (selected.binding_hash, canonical_hash(snapshot))
        if "unavailable" in snapshot:
            raise ValueError(str(snapshot["unavailable"]))
        if selected.observation_kind == "recorded":
            flags, valid, errors = _recorded_values(selected, snapshot, program)
        else:
            flags, valid, errors = _final_values(selected, snapshot, size)
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
            predicate.id,
            predicate.name,
            (
                EvaluationStatus.UNAVAILABLE
                if reason or not valid[row]
                else (
                    EvaluationStatus.SATISFIED
                    if flags[row]
                    else EvaluationStatus.FAILED
                )
            ),
            observed=None if reason or not valid[row] else flags[row],
            expected=predicate.expected,
            reason=reason
            or errors[row]
            or (
                "Physical state acquisition is unavailable."
                if not valid[row]
                else "Measured state checked against its declared criterion."
            ),
            evidence_refs=refs,
        )
        for row in range(size)
    )
