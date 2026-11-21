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

"""Pure source Pour occurrence and geometry-channel qualification."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any
from .evidence import _bool_rows

__all__: list[str] = []
_CRITERION = "measured_tilt_restore/v1"
_MINIMUM_TILT = math.radians(30.0)
_MAXIMUM_RESTORE_ERROR = math.radians(15.0)
_GEOMETRY_COLUMNS = [
    "maximum_tilt_rad",
    "restore_error_rad",
    "anchor_world_x_m",
    "anchor_world_y_m",
    "anchor_world_z_m",
    "receiver_projection_distance_m",
    "receiver_top_clearance_m",
]


def one(values: Any) -> Any:
    selected = list(values)
    if len(selected) != 1:
        raise ValueError("Pour evidence requires one exact execution occurrence.")
    return selected[0]


def frames(segment: Mapping[str, Any], count: int) -> list[int]:
    value = segment["end_steps"]
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(frame) is not int or frame <= 0 for frame in value)
    ):
        raise ValueError("Pour evidence requires exact positive completed boundaries.")
    return value


def declared_pour_calls(
    binding: Any, program: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    calls = {}
    for occurrence in binding.occurrences:
        item = program
        for component in occurrence["source_path"]:
            item = item[component]
        if item.get("name") != occurrence["segment_name"]:
            raise ValueError(
                "Pour declaration selects another named program occurrence."
            )
        calls[occurrence["segment_id"]] = item["steps"]["call"]
    stage, pour, release = (
        calls[key]
        for key in (
            binding.stage_segment_id,
            binding.pour_segment_id,
            binding.release_segment_id,
        )
    )
    if (
        stage.get("call_id") != "simulation.move_held_object"
        or stage["arguments"].get("object") != binding.object_id
        or stage["arguments"].get("reference") != binding.reference_id
        or pour.get("call_id") != "simulation.pour"
        or pour["arguments"].get("object") != binding.object_id
        or release.get("kind") != "place"
        or release.get("object") != binding.object_id
        or any(
            call["resources"].get("primary") != binding.resource_id
            for call in (stage, pour, release)
        )
    ):
        raise ValueError(
            "Pour declaration has another stage, source/receiver, resource or release."
        )
    if (
        sum(call.get("call_id") == "simulation.pour" for call in calls.values()) != 1
        or sum(
            call.get("kind") == "place" and call.get("object") == binding.object_id
            for call in calls.values()
        )
        != 1
        or sum(
            call.get("call_id") == "simulation.move_held_object"
            and call.get("arguments", {}).get("object") == binding.object_id
            and call["arguments"].get("reference") is not None
            for call in calls.values()
        )
        != 1
    ):
        raise ValueError(
            "Pour declaration requires unambiguous stage/tilt/release calls."
        )
    return calls


def measured_pour_occurrences(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    declared: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[bool]]:
    count = len(binding.env_ids)
    segments = {}
    for segment in snapshot["segments"]:
        identity = segment["metadata"]["program_segment_id"]
        if identity in segments:
            raise ValueError(
                "Duplicate compiled execution receipts cannot qualify Pour."
            )
        segments[identity] = segment
    eligible = [True] * count
    calls = {}
    for occurrence in binding.occurrences:
        segment = segments[occurrence["segment_id"]]
        metadata = segment["metadata"]
        runtime = metadata["runtime"]
        call = one(runtime["calls"])
        expected = declared[occurrence["segment_id"]]
        if (
            segment["name"] != occurrence["segment_name"]
            or metadata["task_program_id"] != program["program_id"]
            or metadata["program_segment_source_path"] != occurrence["source_path"]
            or runtime["env_ids"] != list(binding.env_ids)
            or any(type(value) is not int for value in runtime["env_ids"])
            or call["invocation_id"] != occurrence["invocation_id"]
            or type(call["invocation_revision"]) is not int
            or call["invocation_revision"] < 0
            or call["status"] != "completed"
            or type(call["command_count"]) is not int
            or call["command_count"] <= 0
            or call["semantic_id"] != expected.get("call_id", expected.get("kind"))
        ):
            raise ValueError("Pour receipt differs from its exact compiled occurrence.")
        for mask in (
            segment["successes"],
            segment["active"],
            metadata["validation"]["accepted_mask"],
        ):
            if not _bool_rows(mask, count):
                raise ValueError("Pour eligibility requires aligned boolean masks.")
            eligible = [
                old and value for old, value in zip(eligible, mask, strict=True)
            ]
        calls[occurrence["segment_id"]] = call
    if calls[binding.pour_segment_id].get("skill_id") != "pour":
        raise ValueError("Pour motion requires the actual Pour skill.")
    return segments, calls, eligible


def classify_pour_metrics(metrics: Mapping[str, Any], count: int) -> list[bool]:
    rows = metrics["rows"]
    if (
        metrics["scope"] != "measured_pour_motion"
        or metrics["criterion"] != _CRITERION
        or metrics["fluid_transfer_verified"] is not False
        or metrics["minimum_tilt_rad"] != _MINIMUM_TILT
        or metrics["maximum_restore_error_rad"] != _MAXIMUM_RESTORE_ERROR
        or len(rows) != count
    ):
        raise ValueError("Pour motion evidence has another criterion or batch.")
    accepted = []
    for row in rows:
        angles = (row["maximum_tilt_rad"], row["restore_error_rad"])
        checks = (row["above_receiver"], row["released"])
        distance, clearance = (
            row["receiver_projection_distance_m"],
            row["receiver_top_clearance_m"],
        )
        if (
            any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= math.pi
                for v in angles
            )
            or any(type(v) is not bool for v in checks)
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (distance, clearance)
            )
            or distance < 0
            or checks[0] != (distance == 0 and clearance >= 0)
        ):
            raise ValueError("Pour motion contains invalid native measurements.")
        accepted.append(
            angles[0] >= _MINIMUM_TILT
            and angles[1] <= _MAXIMUM_RESTORE_ERROR
            and all(checks)
        )
    return accepted


def raw_pour_geometry(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    starts: list[int],
    ends: list[int],
) -> tuple[list[dict[str, Any]], list[bool], list[str | None]]:
    count = len(binding.env_ids)
    if "record" in snapshot:
        raise ValueError(
            "Raw Pour geometry cannot downgrade to historical record qualification."
        )
    if (
        binding.episode_ids is None
        or snapshot.get("episode_ids") != list(binding.episode_ids)
        or any(type(value) is not int for value in snapshot.get("episode_ids", ()))
    ):
        raise ValueError("Pour geometry belongs to an unbound or different episode.")
    if (
        snapshot.get("schema_version") != "simulation.pour_geometry_observation/v1"
        or snapshot.get("program_id") != program["program_id"]
        or snapshot.get("program_fingerprint") != binding.program_fingerprint
        or snapshot.get("object_id") != binding.object_id
        or snapshot.get("reference_id") != binding.reference_id
        or snapshot.get("object_native_uid") != binding.object_native_uid
        or snapshot.get("reference_native_uid") != binding.reference_native_uid
        or snapshot.get("recording_clock") != "environment_step"
        or binding.step_dt is None
        or type(snapshot.get("recording_step_dt")) not in (int, float)
        or snapshot.get("recording_step_dt") != binding.step_dt
        or snapshot.get("geometry_columns") != _GEOMETRY_COLUMNS
        or snapshot.get("start_frames") != starts
        or snapshot.get("end_frames") != ends
        or snapshot.get("sample_counts")
        != [end - start + 1 for start, end in zip(starts, ends, strict=True)]
    ):
        raise ValueError(
            "Pour geometry has another native/program/recording declaration."
        )
    frame = snapshot["geometry_frame"]
    if (
        frame.get("env_ids") != list(binding.env_ids)
        or any(type(value) is not int for value in frame.get("env_ids", ()))
        or type(frame.get("timestamp")) not in (int, float)
        or not math.isfinite(frame["timestamp"])
        or frame["timestamp"] < 0
        or type(frame.get("observation_revision")) is not int
        or frame["observation_revision"] < 0
        or frame.get("clock_domain") != "simulation_time"
        or frame.get("revision_kind") != "recorded_geometry_acquisition"
    ):
        raise ValueError(
            "Pour acquisition frame has another row, clock or revision scope."
        )
    batches = frame["batches"]
    channels = []
    for name, dtype in (
        ("pour.motion_geometry", "torch.float64"),
        ("pour.anchor_projected_inside", "torch.bool"),
    ):
        batch = batches[name]
        if (
            batch.get("evidence_id") != name
            or batch.get("value_dtype") != dtype
            or batch.get("env_ids") != list(binding.env_ids)
            or any(type(value) is not int for value in batch.get("env_ids", ()))
            or type(batch.get("timestamp")) not in (int, float)
            or batch.get("timestamp") != frame["timestamp"]
            or type(batch.get("observation_revision")) is not int
            or batch["observation_revision"] != frame["observation_revision"]
            or not _bool_rows(batch.get("valid"), count)
            or not isinstance(batch.get("values"), list)
            or len(batch["values"]) != count
            or not isinstance(batch.get("acquisition_errors"), list)
            or len(batch["acquisition_errors"]) != count
        ):
            raise ValueError(
                "Pour geometry acquisition channel is missing or inconsistent."
            )
        if any(
            error is not None
            and (type(error) is not str or not error.strip())
            or available
            and error is not None
            for available, error in zip(
                batch["valid"], batch["acquisition_errors"], strict=True
            )
        ):
            raise ValueError(
                "Pour geometry validity contradicts its acquisition reason."
            )
        channels.append(batch)
    rows = []
    valid = []
    errors = []
    for index in range(count):
        available = all(batch["valid"][index] for batch in channels)
        valid.append(available)
        errors.append(
            next(
                (
                    batch["acquisition_errors"][index]
                    for batch in channels
                    if batch["acquisition_errors"][index]
                ),
                None,
            )
        )
        if not available:
            rows.append({})
            continue
        values, inside = (batch["values"][index] for batch in channels)
        if (
            not isinstance(values, list)
            or len(values) != 7
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values
            )
            or type(inside) is not bool
        ):
            raise ValueError("Pour geometry values are nonfinite or malformed.")
        rows.append(
            dict(
                maximum_tilt_rad=values[0],
                restore_error_rad=values[1],
                anchor_world_m=values[2:5],
                receiver_projection_distance_m=values[5],
                receiver_top_clearance_m=values[6],
                above_receiver=inside and values[6] >= 0,
            )
        )
    return rows, valid, errors
