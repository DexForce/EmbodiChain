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
"""Recompute source control criteria from occurrence-bound physical channels."""

from __future__ import annotations
from collections.abc import Mapping
import math
from types import SimpleNamespace
from typing import Any
from .canonicalization import canonical_hash
from .evidence import _identifier, _digest, _path

__all__: list[str] = []
_SELECTOR = {
    "segment_id",
    "segment_name",
    "source_path",
    "policy_index",
    "preset",
    "phase",
    "terminal",
}
_IDENTITY = {
    "program_fingerprint",
    "episode_id",
    "operation_id",
    "entity_id",
    "env_ids",
    "clock",
    "legacy_time_origin",
    "physics_dt",
    "control_dt",
    "joint_index",
    "family",
    "sensor_uid",
    "observer_id",
    "topology_revision",
    "contact_actors",
    "startup_actors",
}


def validate_control_acquisition(binding: Any) -> None:
    identity = binding.control_identity
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY:
        raise ValueError(
            "Raw control requires its complete actual instance/native identity."
        )
    _digest(identity["program_fingerprint"], "program_fingerprint")
    for key in ("episode_id", "operation_id", "entity_id", "sensor_uid", "observer_id"):
        _identifier(identity[key], key)
    if (
        binding.env_ids != (0,)
        or binding.episode_ids is None
        or identity["env_ids"] != [0]
        or any(type(x) is not int for x in identity["env_ids"])
        or identity["family"] != binding.control_kind
        or identity["operation_id"]
        != binding.operation["route_payload"]["operation_id"]
        or identity["clock"] != "simulation_time"
    ):
        raise ValueError(
            "Raw control requires the declared B=1 operation/episode/clock scope."
        )
    for key in ("control_dt", "physics_dt", "legacy_time_origin"):
        value = identity[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or key != "legacy_time_origin"
            and value == 0
        ):
            raise ValueError("Raw control clocks are invalid.")
    for key in ("joint_index", "topology_revision"):
        if type(identity[key]) is not int or identity[key] < 0:
            raise ValueError("Raw control native indices are invalid.")
    actors = identity["contact_actors"]
    if (
        not isinstance(actors, Mapping)
        or set(actors) != {"target", "parent", "hand"}
        or not isinstance(actors["hand"], list)
        or not actors["hand"]
    ):
        raise ValueError("Control actor roster is incomplete.")
    roster = [actors["target"], actors["parent"], *actors["hand"]]
    if len(set(roster)) != len(roster) or any(
        type(x) is not int or x < 0 for x in roster
    ):
        raise ValueError("Control target/parent/hand actors must be disjoint.")
    startup = identity["startup_actors"]
    if not isinstance(startup, Mapping) or set(startup) != {"table", "robot"}:
        raise ValueError("Control startup actor declaration is incomplete.")
    if binding.control_kind == "twist" and (
        type(startup["table"]) is not int
        or startup["table"] < 0
        or not isinstance(startup["robot"], list)
        or not startup["robot"]
        or any(type(x) is not int or x < 0 for x in startup["robot"])
        or len(set(startup["robot"])) != len(startup["robot"])
    ):
        raise ValueError("Twist requires its complete table/robot startup roster.")
    selectors = binding.policy_occurrences
    if not selectors or any(
        not isinstance(item, Mapping) or set(item) != _SELECTOR for item in selectors
    ):
        raise ValueError("Control policies require complete compiled selectors.")
    if (
        not selectors
        or sum(value["phase"] == "ready" for value in selectors) != 1
        or selectors[0]["phase"] != "ready"
        or not selectors[-1]["terminal"]
    ):
        raise ValueError(
            "Control acquisition requires its ready and exact terminal policy sequence."
        )
    required = {
        item["segment_id"]: item["source_path"]
        for item in binding.operation["required_execution"]
    }
    keys = []
    for selector in selectors:
        if not isinstance(selector, Mapping) or set(selector) != _SELECTOR:
            raise ValueError("Control policies require complete compiled selectors.")
        for key in ("segment_id", "segment_name", "preset"):
            _identifier(selector[key], key)
        _path(selector["source_path"], "source_path")
        if (
            type(selector["policy_index"]) is not int
            or selector["policy_index"] < 0
            or type(selector["terminal"]) is not bool
            or required.get(selector["segment_name"]) != selector["source_path"]
        ):
            raise ValueError(
                "Control policy differs from its required occurrence prefix."
            )
        phases = (
            {"ready", "released"}
            if binding.control_kind == "press"
            else {"ready", "chunk", "released", "park"}
        )
        if (
            selector["phase"] not in phases
            or selector["terminal"]
            and selector["phase"] not in {"released", "park"}
        ):
            raise ValueError("Control policy phase is unsupported.")
        keys.append((selector["segment_id"], selector["policy_index"]))
    last = selectors[-1]
    if (
        len(set(keys)) != len(keys)
        or any(item["terminal"] for item in selectors[:-1])
        or last["segment_name"] != binding.operation["terminal_segment_id"]
        or last["source_path"] != binding.operation["terminal_source_path"]
        or last["preset"] != binding.operation["preset"]
    ):
        raise ValueError("Control acquisition has another terminal occurrence.")
    if binding.control_kind == "press" and not any(
        item["phase"] == "released" and not item["terminal"] for item in selectors
    ):
        raise ValueError(
            "Press terminal qualification requires its original retreat occurrence."
        )


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Control measurements must be finite numbers.")
    return float(value)


def _meta(frame: Mapping[str, Any], kind: str) -> None:
    if (
        frame.get("env_ids") != [0]
        or any(type(x) is not int for x in frame["env_ids"])
        or frame.get("clock_domain") != "simulation_time"
        or frame.get("revision_kind") != kind
        or type(frame.get("observation_revision")) is not int
        or frame["observation_revision"] < 0
        or _number(frame.get("timestamp")) < 0
    ):
        raise ValueError(
            "Control physical frame has another row/clock/revision identity."
        )


def _tensor(frame: Mapping[str, Any], key: str) -> Any:
    import torch

    batch = frame["batches"][key]
    if (
        batch.get("evidence_id") != key
        or batch.get("env_ids") != [0]
        or any(type(x) is not int for x in batch["env_ids"])
        or batch.get("timestamp") != frame["timestamp"]
        or type(batch.get("observation_revision")) is not int
        or batch["observation_revision"] != frame["observation_revision"]
        or batch.get("valid") != [True]
        or any(type(x) is not bool for x in batch["valid"])
        or batch.get("acquisition_errors") != [None]
    ):
        raise ValueError("Control physical channel is missing or inconsistent.")
    dtype = batch.get("value_dtype")
    kinds = {
        "torch.bool": torch.bool,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    if dtype not in kinds:
        raise ValueError("Control physical channel has an unsupported dtype.")

    def leaves(value):
        if isinstance(value, list):
            for item in value:
                yield from leaves(item)
        else:
            yield value

    values = list(leaves(batch["values"]))
    expected = (
        (bool,)
        if dtype == "torch.bool"
        else (int,) if dtype in {"torch.int32", "torch.int64"} else (int, float)
    )
    if any(type(value) not in expected for value in values):
        raise ValueError("Control physical channel values contradict their dtype.")
    result = torch.tensor(batch["values"], dtype=kinds[dtype])
    shape = batch.get("value_shape")
    if "value_shape" in batch:
        if (
            type(shape) is not list
            or not shape
            or shape[0] != 1
            or any(type(size) is not int or size < 0 for size in shape)
            or math.prod(shape) != result.numel()
            or result.numel()
            and tuple(result.shape) != tuple(shape)
        ):
            raise ValueError(
                "Control physical shape metadata contradicts channel values."
            )
        result = result.reshape(shape)
    if (
        result.ndim == 0
        or result.shape[0] != 1
        or result.is_floating_point()
        and not torch.isfinite(result).all()
    ):
        raise ValueError("Control physical values are unavailable or malformed.")
    return result


def _window(
    value: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[list[Any], list[bool], list[Mapping[str, Any]]]:
    import torch

    if (
        type(value["generation"]) is not int
        or value["generation"] < 1
        or value["legacy_time_origin"] != identity["legacy_time_origin"]
        or value["physics_dt"] != identity["physics_dt"]
        or value["errors"]
        or not value["records"]
    ):
        raise ValueError(
            "Control window is incomplete or has another source clock/generation."
        )
    samples = []
    parents = []
    details = []
    actors = identity["contact_actors"]
    previous = None
    for record in value["records"]:
        frame = record["frame"]
        _meta(frame, "joint_contact_acquisition")
        if (
            type(record["native_revision"]) is not int
            or record["native_revision"] < 0
            or type(record["topology_revision"]) is not int
            or record["topology_revision"] != identity["topology_revision"]
            or type(record["dropped_contacts"]) is not int
            or record["dropped_contacts"] != 0
            or type(record["phase"]) is not str
            or not record["phase"]
        ):
            raise ValueError(
                "Control native source stamp, phase or contact completeness is unavailable."
            )
        mask = _tensor(frame, "contact.is_valid")
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError(
                "Control contact mask must preserve the source native row."
            )
        count = mask.shape[1]
        channels = {
            key: _tensor(frame, key)
            for key in (
                "contact.user_ids",
                "contact.impulse",
                "contact.position",
                "contact.normal",
                "contact.distance",
                "articulation.joint_position",
            )
        }
        shapes = {
            "contact.user_ids": (1, count, 2),
            "contact.impulse": (1, count),
            "contact.position": (1, count, 3),
            "contact.normal": (1, count, 3),
            "contact.distance": (1, count),
            "articulation.joint_position": (1,),
        }
        if (
            any(channels[key].shape != shape for key, shape in shapes.items())
            or channels["contact.user_ids"].dtype not in (torch.int32, torch.int64)
            or any(
                not tensor.is_floating_point()
                for key, tensor in channels.items()
                if key != "contact.user_ids"
            )
        ):
            raise ValueError("Control channels have another physical shape or dtype.")
        pairs = channels["contact.user_ids"][0, mask[0]].tolist()
        impulses = channels["contact.impulse"][0, mask[0]].tolist()
        timestamp = frame["timestamp"] - value["legacy_time_origin"]
        if (
            timestamp < 0
            or previous is not None
            and not math.isclose(
                timestamp - previous, value["physics_dt"], rel_tol=0, abs_tol=1e-8
            )
        ):
            raise ValueError(
                "Control window has a missing, repeated or stale physics sample."
            )
        previous = timestamp

        def contact(actor):
            return any(
                actor in pair
                and any(hand in pair for hand in actors["hand"])
                and impulse > 0
                for pair, impulse in zip(pairs, impulses, strict=True)
            )

        sample = SimpleNamespace(
            timestamp=timestamp,
            qpos=float(channels["articulation.joint_position"][0]),
            phase=record["phase"],
            target_contact=contact(actors["target"]),
            valid=True,
        )
        parent = contact(actors["parent"])
        trace = {
            **vars(sample),
            "contact_pairs": pairs,
            "impulses": impulses,
            "dropped_contacts": 0,
            "parent_contact": parent,
        }
        samples.append(sample)
        parents.append(parent)
        details.append(dict(frame=frame, pairs=pairs, impulses=impulses, trace=trace))
    if canonical_hash(value["samples"]) != canonical_hash(
        [vars(sample) for sample in samples]
    ) or canonical_hash(value["trace"]) != canonical_hash(
        [item["trace"] for item in details]
    ):
        raise ValueError(
            "Control projected samples differ from their raw native channels."
        )
    return samples, parents, details


def _state_frame(
    frame: Any, identity: Mapping[str, Any], step: int
) -> Mapping[str, Any]:
    if frame is None:
        raise ValueError(
            "Required original control state/geometry acquisition is missing."
        )
    _meta(frame, "control_policy_read")
    expected = identity["legacy_time_origin"] + step * identity["control_dt"]
    if not math.isclose(frame["timestamp"], expected, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Control policy read has another completed clock boundary.")
    return frame


def _field(frame: Mapping[str, Any], name: str) -> float:
    value = _tensor(frame, name)
    if value.shape != (1,) or not value.is_floating_point():
        raise ValueError("Control scalar measurement has another physical shape.")
    return float(value[0])


def _match(result: Mapping[str, Any], expected: Mapping[str, Any], keys: Any) -> None:
    mismatches = [
        key
        for key in keys
        if canonical_hash(result[key]) != canonical_hash(expected[key])
    ]
    if mismatches:
        raise ValueError(
            "Control receipt differs from raw physical measurements: "
            + ", ".join(mismatches)
        )


def verify_raw_control(
    binding: Any, snapshot: Mapping[str, Any], program: Mapping[str, Any]
) -> None:
    import torch
    from embodichain.compute.articulation import (
        evaluate_press_event,
        evaluate_twist_event,
        initial_twist_state_stable,
    )
    from ._control_evidence import one

    packet = snapshot["control_acquisition"]
    identity = binding.control_identity
    if (
        packet.get("schema_version") != "simulation.control_observation/v1"
        or canonical_hash(packet["control_identity"]) != canonical_hash(identity)
        or packet.get("episode_ids") != list(binding.episode_ids)
        or any(type(x) is not int for x in packet["episode_ids"])
        or packet["terminal_selector"] != binding.policy_occurrences[-1]
    ):
        raise ValueError(
            "Raw control acquisition belongs to another actual instance/program/episode."
        )
    samples, parents, details = _window(packet["window"], identity)
    policies = packet["policies"]
    if len(policies) != len(binding.policy_occurrences):
        raise ValueError("Raw control policy history is incomplete.")
    segments = {segment["name"]: segment for segment in snapshot["segments"]}
    if len(segments) != len(snapshot["segments"]):
        raise ValueError("Raw control has duplicate runtime occurrences.")
    route = binding.operation["route_payload"]
    physical = route["binding"]
    if binding.control_kind == "press":
        released = _number(physical["released_position"])
        pressed = _number(physical["pressed_position"])
        if released == pressed:
            raise ValueError("Press physical endpoints must differ.")
        target = SimpleNamespace(
            released_position=released,
            pressed_position=pressed,
            stroke=abs(pressed - released),
        )
    else:
        target = SimpleNamespace(
            target_qpos=_number(physical["target_qpos"]),
            limits=tuple(_number(x) for x in physical["limits"]),
            success_criterion=route["success_criterion"],
            angle_tolerance=math.radians(5.0),
            coarse_turn_threshold=math.radians(15.0),
            path_deadband=math.radians(0.5),
        )
    retreat = False
    reference_qpos = None
    chunk_start = 0
    previous_contact_path = 0.0
    for selector, entry in zip(binding.policy_occurrences, policies, strict=True):
        if entry["selector"] != selector:
            raise ValueError(
                "Control physical reads have another compiled policy selector."
            )
        scope = entry["scope"]
        if (
            any(
                canonical_hash(scope.get(key)) != canonical_hash(value)
                for key, value in identity.items()
            )
            or canonical_hash(scope["route_payload"]) != canonical_hash(route)
            or scope["source_path"]
            != [*selector["source_path"], "post", selector["policy_index"]]
            or any(
                scope.get(key) != selector[key]
                for key in ("policy_index", "preset", "phase", "terminal")
            )
        ):
            raise ValueError(
                "Control policy source differs from its sealed actual owner."
            )
        segment = segments[selector["segment_name"]]
        meta = segment["metadata"]
        if (
            meta["program_segment_id"] != selector["segment_id"]
            or meta["program_segment_source_path"] != selector["source_path"]
            or meta["task_program_id"] != program["program_id"]
        ):
            raise ValueError(
                "Control physical reads have another actual invocation occurrence."
            )
        item = program
        for component in selector["source_path"]:
            item = item[component]
        declared_policy = item["post"][selector["policy_index"]]
        # A semantic operation may name its sealed parent while acquisition
        # retains the explicitly bound native child. Neither identity is inferred.
        if (
            item["name"] != selector["segment_name"]
            or declared_policy["entity"]
            not in (binding.entity_id, identity["entity_id"])
            or declared_policy
            != {
                "kind": "wait_stable",
                "entity": declared_policy["entity"],
                "preset": selector["preset"],
            }
        ):
            raise ValueError("Control policy has another declared entity or preset.")
        receipt = one(
            value
            for value in meta["post_policies"]
            if value.get("policy_index") == selector["policy_index"]
            and type(value.get("policy_index")) is int
            and value.get("source_path") == scope["source_path"]
        )
        result = receipt["result"]
        if canonical_hash(result) != canonical_hash(entry["receipt"]):
            raise ValueError(
                "Control acquisition differs from its original runtime policy receipt."
            )
        facts = entry["measurements"]
        if (
            facts["phase"] != selector["phase"]
            or facts["terminal"] is not selector["terminal"]
            or facts["generation"] != packet["window"]["generation"]
            or any(
                facts[key] != identity[key]
                for key in (
                    "control_dt",
                    "joint_index",
                    "legacy_time_origin",
                    "physics_dt",
                )
            )
            or type(facts["sample_count"]) is not int
            or not 1 <= facts["sample_count"] <= len(samples)
        ):
            raise ValueError(
                "Control measurements have another window or phase identity."
            )
        n = facts["sample_count"]
        prefix = samples[:n]
        parent_prefix = parents[:n]
        if (
            facts["window"] != [prefix[0].timestamp, prefix[-1].timestamp]
            or result["control"]["window"] != facts["window"]
            or result["control"]["sample_count"] != n
        ):
            raise ValueError("Control policy cannot borrow a later observation window.")
        start, end = facts["hold_start_step"], facts["hold_end_step"]
        if any(
            type(facts[key]) is not int
            for key in ("generation", "joint_index", "hold_start_step", "hold_end_step")
        ) or not math.isclose(
            facts["window"][-1], end * identity["control_dt"], rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(
                "Control policy window belongs to another actual hold boundary."
            )
        duration = (
            0.2
            if binding.control_kind == "press" and selector["phase"] != "ready"
            else 0.5
        )
        ticks = math.ceil(duration / identity["control_dt"])
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end - start != ticks
            or selector["policy_index"] == len(item["post"]) - 1
            and segment["end_steps"] != [end]
        ):
            raise ValueError(
                "Control hold differs from the original acknowledged control-step grid."
            )
        measured = binding.control_kind == "twist" or selector["phase"] == "ready"
        if len(facts["hold_frames"]) != (ticks if measured else 0):
            raise ValueError("Control hold has missing or foreign state reads.")
        positions = []
        velocities = []
        for offset, frame in enumerate(facts["hold_frames"], 1):
            _state_frame(frame, identity, start + offset)
            positions.append(_field(frame, "articulation.joint_position"))
            if binding.control_kind == "twist":
                velocities.append(_field(frame, "articulation.joint_velocity"))
        tail = max(1, math.ceil(0.15 / identity["physics_dt"]))
        no_contact = bool(prefix) and all(
            sample.target_contact is False and sample.valid for sample in prefix[-tail:]
        )
        no_parent = bool(prefix) and all(
            value is False for value in parent_prefix[-tail:]
        )
        if selector["phase"] == "ready":
            if positions[-1] != prefix[0].qpos or prefix[0].target_contact is not False:
                raise ValueError("Control ready window lacks its exact released onset.")
            if binding.control_kind == "press":
                limits = _tensor(
                    _state_frame(facts["joint_limits"], identity, end),
                    "articulation.joint_limits",
                )
                good = (
                    all(
                        abs(value - target.released_position) <= target.stroke * 0.1
                        for value in positions
                    )
                    and limits.shape == (1, 2)
                    and bool(
                        torch.allclose(
                            limits[0].cpu(),
                            torch.tensor(physical["limits"]),
                            atol=1e-7,
                            rtol=1e-5,
                        )
                    )
                )
                _match(
                    result,
                    {"accepted": bool(good), "joint_positions": positions},
                    ("accepted", "joint_positions"),
                )
                retreat = False
            else:
                startup, _, startup_details = _window(entry["startup_window"], identity)
                if not math.isclose(
                    startup[-1].timestamp, prefix[0].timestamp, rel_tol=0, abs_tol=1e-8
                ):
                    raise ValueError("Twist startup belongs to another ready boundary.")
                speeds = []
                table_contacts = robot_contacts = 0
                for sample, detail in zip(startup, startup_details, strict=True):
                    speeds.append(
                        _field(detail["frame"], "articulation.joint_velocity")
                    )
                    table_contacts += any(
                        identity["contact_actors"]["target"] in pair
                        and identity["startup_actors"]["table"] in pair
                        and impulse > 0
                        for pair, impulse in zip(
                            detail["pairs"], detail["impulses"], strict=True
                        )
                    )
                    robot_contacts += any(
                        identity["contact_actors"]["target"] in pair
                        and any(
                            actor in identity["startup_actors"]["robot"]
                            for actor in pair
                        )
                        and impulse > 0
                        for pair, impulse in zip(
                            detail["pairs"], detail["impulses"], strict=True
                        )
                    )
                stable = initial_twist_state_stable(
                    [sample.qpos for sample in startup],
                    speeds,
                    position_tolerance=math.radians(2.0),
                    velocity_tolerance=math.radians(2.0),
                )
                startup_good = (
                    len(startup) >= 2
                    and not table_contacts
                    and not robot_contacts
                    and stable
                )
                expected = dict(
                    accepted=bool(startup_good),
                    sample_count=len(startup),
                    table_contact_samples=table_contacts,
                    robot_contact_samples=robot_contacts,
                    max_abs_qpos=max(abs(sample.qpos) for sample in startup),
                    max_abs_qvel=max(abs(value) for value in speeds),
                    initial_angle_tolerance=math.radians(2.0),
                    initial_velocity_tolerance=math.radians(2.0),
                )
                _match(result["startup_check"], expected, expected)
                good = startup_good and initial_twist_state_stable(
                    positions,
                    velocities,
                    position_tolerance=math.radians(2.0),
                    velocity_tolerance=math.radians(2.0),
                )
                _match(
                    result,
                    dict(
                        accepted=bool(good),
                        joint_positions=positions,
                        joint_velocities=velocities,
                    ),
                    ("accepted", "joint_positions", "joint_velocities"),
                )
                chunk_start = n
                previous_contact_path = 0.0
                reference_qpos = _tensor(
                    _state_frame(facts["reference_qpos"], identity, end),
                    "articulation.joint_positions",
                )
        elif binding.control_kind == "press":
            event = evaluate_press_event(target, prefix)
            geometry = _state_frame(facts["clearance"], identity, end)
            clearance = _field(
                geometry,
                (
                    "control.terminal_clearance"
                    if selector["terminal"]
                    else "control.retreat_clearance"
                ),
            )
            observed = _field(
                _state_frame(facts["current_qpos"], identity, end),
                "articulation.joint_position",
            )
            good = (
                event["accepted"]
                and no_contact
                and (not selector["terminal"] or retreat and no_parent)
                and clearance >= 0.04
            )
            if not selector["terminal"]:
                retreat = bool(good)
            expected = {
                **event,
                "accepted": bool(good),
                "event_accepted": event["accepted"],
                "contact_released": no_contact,
                "housing_contact_released": no_parent,
                "retreat_verified": retreat,
                "retreat_clearance": _field(geometry, "control.retreat_clearance"),
                "tcp_clearance": _field(geometry, "control.tcp_clearance"),
                "terminal_clearance": (
                    _field(geometry, "control.terminal_clearance")
                    if selector["terminal"]
                    else None
                ),
                "observed_joint_position": observed,
            }
            _match(result, expected, tuple(key for key in expected if key != "reason"))
        elif selector["phase"] == "chunk":
            observed = _field(
                _state_frame(facts["current_qpos"], identity, end),
                "articulation.joint_position",
            )
            if prefix[-1].qpos != observed:
                raise ValueError(
                    "Twist chunk joint read differs from its current native acquisition."
                )
            event = evaluate_twist_event(target, prefix, final_qpos=observed)
            stable = max(positions) - min(positions) <= math.radians(1.0) and all(
                abs(value) <= math.radians(2.0) for value in velocities
            )
            chunk = prefix[chunk_start:]
            chunk_parents = parent_prefix[chunk_start:]
            path = event.get("contact_path", 0.0)
            expected = dict(
                sample_count=len(chunk),
                target_contact=any(sample.target_contact for sample in chunk),
                parent_contact_fraction=(
                    sum(chunk_parents) / len(chunk_parents) if chunk_parents else 1.0
                ),
                observed_qpos=observed,
                target_error=abs(observed - target.target_qpos),
                already_reached=abs(observed - target.target_qpos)
                <= target.angle_tolerance,
                contact_released=no_contact,
                stable=bool(stable),
                recovery_safe=bool(
                    stable and no_contact and len(prefix[-tail:]) == tail
                ),
                chunk_contact_progress=max(0.0, path - previous_contact_path),
            )
            _match(result, expected, expected)
            chunk_start = n
            previous_contact_path = path
        elif selector["terminal"] or selector["phase"] == "released":
            event = evaluate_twist_event(target, prefix, final_qpos=positions[-1])
            geometry = _state_frame(facts["clearance"], identity, end)
            clearance = _field(geometry, "control.park_clearance")
            current = _tensor(
                _state_frame(facts["current_qpos"], identity, end),
                "articulation.joint_positions",
            )
            if (
                reference_qpos is None
                or current.shape != reference_qpos.shape
                or current.ndim != 2
                or not 0 <= identity["joint_index"] < current.shape[1]
            ):
                raise ValueError(
                    "Twist lacks exact non-target joint baseline/current rows."
                )
            current = current.clone()
            current[:, identity["joint_index"]] = reference_qpos[
                :, identity["joint_index"]
            ]
            unchanged = bool(
                torch.isfinite(current).all()
                and torch.allclose(
                    current, reference_qpos, atol=math.radians(2.0), rtol=0
                )
            )
            stable = max(positions) - min(positions) <= math.radians(1.0) and all(
                abs(value) <= math.radians(2.0) for value in velocities
            )
            good = (
                event["accepted"]
                and stable
                and no_contact
                and (not selector["terminal"] or clearance >= 0.04)
                and unchanged
            )
            expected = {
                **event,
                "accepted": bool(good),
                "stable": bool(stable),
                "contact_released": no_contact,
                "clearance": clearance,
                "others_unchanged": unchanged,
                "joint_velocities": velocities,
                "parent_contact_fraction": sum(parent_prefix) / len(parent_prefix),
            }
            _match(result, expected, tuple(key for key in expected if key != "reason"))
        if result.get("accepted") is not True:
            raise ValueError(
                "Control source operation did not physically complete its required policy."
            )
    terminal_step = policies[-1]["measurements"]["hold_end_step"]
    if (
        packet.get("boundary_frames") != [terminal_step]
        or snapshot.get("boundary_frames", [terminal_step]) != [terminal_step]
        or policies[-1]["measurements"]["sample_count"] != len(samples)
    ):
        raise ValueError("Control report has another exact terminal boundary/window.")
