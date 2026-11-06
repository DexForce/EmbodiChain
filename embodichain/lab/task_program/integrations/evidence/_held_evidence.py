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


"""Pure receipt, participant and pose checks for measured held-object evidence."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


def grip_flags(
    *,
    uid: str,
    resource: str,
    sample: Mapping[str, Any],
    terminal: Mapping[str, Any],
    frames: list[int],
    dt: float,
    count: int,
) -> list[bool]:
    """Reclassify exact raw pad contacts instead of trusting cached booleans."""
    import torch
    from embodichain.compute.geometry.grip_contacts import classify_contacts

    query = sample["query"]
    expectation = one(
        value
        for value in terminal["effect_spec"]["state_expectations"]
        if value["object_id"] == uid
        and value["resource_id"] == resource
        and value["relation"] == "attached"
    )
    clause = one(
        value
        for value in terminal["effect_spec"]["clauses"]
        if value["expectation_id"] == expectation["expectation_id"]
        and value.get("evidence_kind") == "constraint"
    )
    if (
        query["object_id"] != uid
        or query["control_part"] != clause["source"]["address"]["control_part"]
        or query["expected"] is not True
        or query["dropped_contacts"] != 0
        or query["timestamp"] != max(frames) * dt
        or query["observation_revision"] != terminal["observation_revision"] + 1
        or sample["timestamp_scope"]
        != "collection_query_clock_not_native_impulse_refresh"
    ):
        raise ValueError("Final held grip query has another participant or boundary.")
    actors, pads = sample["object_actor_ids"], sample["pad_actor_ids"]
    if len(actors) != count or len(pads) != count or len(query["rows"]) != count:
        raise ValueError("Final held grip actor inventory has another batch.")
    flags = []
    for env_id, row in enumerate(query["rows"]):
        inventory = [*actors[env_id], *pads[env_id]]
        if (
            len(actors[env_id]) != 1
            or len(pads[env_id]) != 2
            or len(set(inventory)) != 3
            or any(type(value) is not int for value in inventory)
            or type(row["env_id"]) is not int
            or row["env_id"] != env_id
        ):
            raise ValueError("Final held grip lacks exact object/pad actors.")
        raw = row["raw_contacts"]
        if any(type(value) is not int for pair in raw["pairs"] for value in pair):
            raise ValueError("Final held grip contact actor IDs are invalid.")
        pairs = torch.tensor(raw["pairs"], dtype=torch.long).reshape(-1, 2)
        normals = torch.tensor(raw["normals"], dtype=torch.float64).reshape(-1, 3)
        impulses = torch.tensor(raw["impulses"], dtype=torch.float64)
        if len(pairs) != len(normals) or impulses.shape != (len(pairs),):
            raise ValueError("Final held grip contact arrays disagree.")
        value, valid = classify_contacts(
            pairs, normals, impulses, target=actors[env_id][0], pads=tuple(pads[env_id])
        )
        if (
            not valid
            or row["finite"] is not True
            or row["valid"] is not True
            or row["value"] is not value
        ):
            raise ValueError("Final held native grip constraint is unavailable.")
        flags.append(value)
    return flags


def one(values: Any) -> Any:
    values = list(values)
    if len(values) != 1:
        raise ValueError("Held evidence requires one exact occurrence.")
    return values[0]


def flags(values: Any, count: int, expected: bool = True) -> None:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) is not bool or value is not expected for value in values)
    ):
        raise ValueError("Held evidence masks are invalid or not accepted.")


def number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Held evidence requires finite nonnegative clocks/metrics.")
    return float(value)


def matrix(value: Any, count: int) -> torch.Tensor:
    import torch
    from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

    result = torch.as_tensor(value, dtype=torch.float64).cpu()
    validate_rigid_pose_batch(result, count)
    return result


def delta(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from embodichain.compute.geometry.rigid import rigid_pose_displacement

    return rigid_pose_displacement(second, first)


def receipt(receipt: Mapping[str, Any], call: Mapping[str, Any], count: int) -> None:
    spec, monitor = receipt["effect_spec"], receipt["monitor"]
    if (
        spec["invocation_id"] != call["invocation_id"]
        or spec["invocation_revision"] != call["invocation_revision"]
        or type(spec["invocation_revision"]) is not int
        or spec["env_ids"] != list(range(count))
        or any(type(value) is not int for value in spec["env_ids"])
        or monitor["monitor_id"] != "builtin.composite_effect"
        or monitor["revision"] != "1"
        or type(receipt["observation_revision"]) is not int
    ):
        raise ValueError("Held physical receipt has another identity or monitor.")
    number(receipt["timestamp"])
    flags(receipt["decision"]["success_mask"], count)
    flags(receipt["decision"]["failure_mask"], count, False)
    for clause in spec["clauses"]:
        expectation = one(
            value
            for value in spec["state_expectations"]
            if value["expectation_id"] == clause["expectation_id"]
        )
        is_pose = clause.get("expectation") in {"matched", "separated"}
        endpoint = one(
            value
            for value in call["resolved_core_policy"]["endpoints"]
            if value["slot_id"] == expectation["slot_id"]
            and value["endpoint_id"] == ("motion" if is_pose else "grasp")
            and value["resource_id"] == expectation["resource_id"]
            and value["task_state_key"] == expectation["task_state_key"]
        )
        address = clause["source"]["address"]
        if address["control_part"] != endpoint["target_id"] or address["channel"] != (
            "pose_relation" if is_pose else "constraint"
        ):
            raise ValueError("Held samples do not belong to the resolved participant.")


def slot(
    receipt: Mapping[str, Any], uid: str, resource: str, relation: str, count: int
) -> tuple[torch.Tensor, str]:
    spec = receipt["effect_spec"]
    expectation = one(
        value
        for value in spec["state_expectations"]
        if value["object_id"] == uid
        and value["resource_id"] == resource
        and value["relation"] == relation
        and value["task_state_key"] == resource
    )
    decision = one(
        value
        for value in receipt["decision"]["expectations"]
        if value["expectation_id"] == expectation["expectation_id"]
    )
    flags(decision["satisfied_mask"], count)
    flags(decision["contradicted_mask"], count, False)
    clauses = [
        value
        for value in spec["clauses"]
        if value["expectation_id"] == expectation["expectation_id"]
    ]
    pose = one(
        value
        for value in clauses
        if value.get("expectation")
        == ("matched" if relation == "attached" else "separated")
    )
    constraint = one(
        value
        for value in clauses
        if value.get("evidence_kind") == "constraint"
        and value.get("expected") is (relation == "attached")
    )
    samples = [receipt["evidence"][value["clause_id"]] for value in (pose, constraint)]
    for clause, sample in zip((pose, constraint), samples, strict=True):
        if (
            clause["source"]["provider_id"] != "builtin.control_part"
            or clause["source"]["revision"] != "1"
            or sample["env_ids"] != list(range(count))
            or sample["timestamp"] != receipt["timestamp"]
            or sample["observation_revision"] != receipt["observation_revision"]
            or sample.get("acquisition_errors") != [None] * count
        ):
            raise ValueError("Held physical samples differ from their receipt.")
        flags(sample["valid_mask"], count)
    flags(samples[1]["values"], count, relation == "attached")
    return (
        matrix(samples[0]["object_to_endpoint"], count),
        pose["source"]["address"]["control_part"],
    )


def frames(segment: Mapping[str, Any], count: int) -> list[int]:
    frames = segment["end_steps"]
    if (
        not isinstance(frames, list)
        or len(frames) != count
        or any(type(frame) is not int or frame <= 0 for frame in frames)
    ):
        raise ValueError("Held evidence has no exact recorded end-step boundary.")
    return frames


def raw_held_poses(
    binding: Any,
    snapshot: Mapping[str, Any],
    boundary: list[int],
    part: str,
    *,
    observation_kind: str | None = None,
) -> tuple[Any, Any]:
    """Validate exact raw body/TCP channels without accepting attached claims."""
    count = len(binding.env_ids)
    if (
        snapshot.get("schema_version") != "simulation.held_pose_observation/v1"
        or snapshot.get("program_fingerprint") != binding.program_fingerprint
        or snapshot.get("object_id") != binding.operation["object_id"]
        or snapshot.get("object_native_uid") != binding.object_native_uid
        or snapshot.get("robot_uid") != binding.robot_uid
        or snapshot.get("episode_ids") != list(binding.episode_ids)
        or any(type(value) is not int for value in snapshot["episode_ids"])
        or snapshot.get("observation_kind")
        != (observation_kind or binding.observation_kind)
        or snapshot.get("boundary_frames") != boundary
        or any(type(value) is not int for value in snapshot["boundary_frames"])
        or snapshot.get("motion_parts") != [part]
        or snapshot.get("step_dt") != binding.step_dt
    ):
        raise ValueError(
            "Held poses have another native/program/episode/boundary identity."
        )
    frame = snapshot["pose_frame"]
    number(frame["timestamp"])
    if (
        frame.get("env_ids") != list(binding.env_ids)
        or any(type(value) is not int for value in frame["env_ids"])
        or type(frame.get("observation_revision")) is not int
        or frame["observation_revision"] < 0
        or frame.get("clock_domain") != "simulation_time"
        or frame.get("revision_kind") != "held_pose_acquisition"
    ):
        raise ValueError("Held pose acquisition has another clock or row scope.")
    values = []
    for name, reported in (
        ("rigid.pose", snapshot["object_pose"]),
        ("robot.tcp_pose:" + part, snapshot["tcp_poses"][part]),
    ):
        batch = frame["batches"][name]
        if (
            batch.get("evidence_id") != name
            or batch.get("value_dtype") != "torch.float64"
            or batch.get("env_ids") != list(binding.env_ids)
            or any(type(value) is not int for value in batch["env_ids"])
            or type(batch.get("timestamp")) not in (int, float)
            or batch["timestamp"] != frame["timestamp"]
            or type(batch.get("observation_revision")) is not int
            or batch["observation_revision"] != frame["observation_revision"]
            or batch.get("acquisition_errors") != [None] * count
            or batch.get("values") != reported
        ):
            raise ValueError(
                "Held pose channel is missing or contradicts its acquisition."
            )
        flags(batch["valid"], count)
        values.append(matrix(batch["values"], count))
    return tuple(values)


def raw_grip(binding: Any, sample: Mapping[str, Any], boundary: list[int]) -> None:
    """Require a fresh, complete native grip channel for the selected episode."""
    query, batch = sample["query"], sample["grip_acquisition"]
    count = len(binding.env_ids)
    if (
        sample.get("episode_ids") != list(binding.episode_ids)
        or any(type(value) is not int for value in sample["episode_ids"])
        or sample.get("boundary_frames") != boundary
        or any(type(value) is not int for value in sample["boundary_frames"])
        or batch.get("evidence_id") != "grip.bilateral"
        or batch.get("clock_domain") != "environment_step"
        or batch.get("revision_kind") != "semantic_query_revision"
        or batch.get("env_ids") != list(binding.env_ids)
        or any(type(value) is not int for value in batch["env_ids"])
        or type(batch.get("timestamp")) not in (int, float)
        or batch["timestamp"] != query["timestamp"]
        or type(batch.get("observation_revision")) is not int
        or batch["observation_revision"] != query["observation_revision"]
        or batch.get("acquisition_errors") != [None] * count
        or batch.get("values") != [row["value"] for row in query["rows"]]
    ):
        raise ValueError("Fresh grip acquisition has another episode, row or clock.")
    flags(batch["valid"], count)
