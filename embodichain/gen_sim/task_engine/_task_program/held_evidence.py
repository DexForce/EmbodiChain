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


"""Shared native held-relation evidence primitives for GenSim observers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.utils.math import xyz_quat_to_4x4_matrix
from .placement_evidence import _valid_pose

__all__: list[str] = []


def fresh_grip(
    env: Any,
    *,
    uid: str,
    resource: str,
    terminal: Mapping[str, Any],
    frames: list[int],
    dt: float,
    count: int,
) -> dict[str, Any]:
    """Query the existing native pad observer at an explicit held boundary."""
    from embodichain.lab.task_program.semantics.evidence import (
        BinaryEffectEvidenceQuery,
        EffectEvidenceCollectionContext,
    )
    from embodichain.lab.task_program.semantics.effects import (
        BinaryEffectClause,
        BinaryEvidenceKind,
        ControlPartEvidenceAddress,
        EffectEvidenceSourceRef,
        HeldObjectRelation,
        HeldObjectStateExpectation,
    )
    from .grip_evidence import NativeGripObserver, SENSOR_UID

    target = getattr(env, "unwrapped", env)
    observer = target.sim.get_sensor(SENSOR_UID)._gen_sim_grip_observer
    if not isinstance(observer, NativeGripObserver):
        raise ValueError(
            "Final held evidence requires its existing native grip observer."
        )
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
        and value.get("expected") is True
    )
    address = clause["source"]["address"]
    part = address["control_part"]
    query = BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            clause["clause_id"],
            clause["expectation_id"],
            EffectEvidenceSourceRef(
                "builtin.control_part",
                "1",
                ControlPartEvidenceAddress(part, "constraint"),
            ),
            BinaryEvidenceKind.CONSTRAINT,
            True,
        ),
        HeldObjectStateExpectation(
            expectation["expectation_id"],
            HeldObjectRelation.ATTACHED,
            uid,
            expectation["slot_id"],
            resource,
            expectation["task_state_key"],
        ),
    )
    context = EffectEvidenceCollectionContext(
        max(frames) * dt,
        terminal["observation_revision"] + 1,
        torch.arange(count, dtype=torch.long),
    )
    observer(query, context)
    return {
        "query": deepcopy(observer.trace[-1]),
        "object_actor_ids": observer.object_actors[uid].tolist(),
        "pad_actor_ids": observer.pad_actors[part].tolist(),
        "timestamp_scope": "collection_query_clock_not_native_impulse_refresh",
    }


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
    from .grip_evidence import classify_contacts

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
    result = torch.as_tensor(value, dtype=torch.float64).cpu()
    _valid_pose(result, count)
    return result


def delta(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    translation = torch.linalg.vector_norm(first[:, :3, 3] - second[:, :3, 3], dim=-1)
    rotation = first[:, :3, :3].transpose(-1, -2) @ second[:, :3, :3]
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return translation, torch.acos(cosine)


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


def native_poses(
    env: Any,
    deployment: Any,
    uid: str,
    frames: list[int],
    motion_part: str,
    count: int,
    final: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the existing single-part observation through one shared snapshot."""
    pose, tcps = native_held_poses(
        env, deployment, uid, frames, (motion_part,), count, final
    )
    return pose, tcps[motion_part]


def native_held_poses(
    env: Any,
    deployment: Any,
    uid: str,
    frames: list[int],
    motion_parts: tuple[str, ...],
    count: int,
    final: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Read one body and robot snapshot for all participating held endpoints."""
    if not motion_parts or len(set(motion_parts)) != len(motion_parts):
        raise ValueError("Held evidence requires distinct motion participants.")
    target = getattr(env, "unwrapped", env)
    binding = one(
        value
        for value in deployment.integration.registration.scene_binding.rigid_objects
        if value.entity_id == uid
    )
    counts = target._traj_steps
    states = target._traj_buffer["states"]
    objects, joints = (
        states["rigid_objects"][binding.simulation_uid]["pose"],
        states["robot"]["qpos"],
    )
    if (
        len(counts) != count
        or objects.ndim != 3
        or objects.shape[0] != count
        or objects.shape[-1] != 7
        or joints.ndim != 3
        or joints.shape[0] != count
        or any(
            frame > int(counts[row]) or frame >= min(objects.shape[1], joints.shape[1])
            for row, frame in enumerate(frames)
        )
    ):
        raise ValueError(
            "Held evidence requires its seeded end-step pose and robot state slots."
        )
    # Recorder rows are pre-action; only the seeded end_step is post-segment.
    pose = xyz_quat_to_4x4_matrix(
        torch.stack([objects[row, frame] for row, frame in enumerate(frames)])
        .detach()
        .cpu()
    )
    qpos = torch.stack([joints[row, frame] for row, frame in enumerate(frames)])
    robot = target.robot

    def endpoints(snapshot: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            part: matrix(
                robot.compute_fk(
                    qpos=snapshot[:, robot.get_joint_ids(name=part)],
                    name=part,
                    to_matrix=True,
                )
                .detach()
                .cpu(),
                count,
            )
            for part in motion_parts
        }

    recorded_pose, recorded_tcps = matrix(pose, count), endpoints(qpos)
    if not final:
        return recorded_pose, recorded_tcps
    native = target.sim.get_rigid_object(binding.simulation_uid)
    fresh_pose = matrix(native.get_local_pose(to_matrix=True).detach().cpu(), count)
    fresh_tcps = endpoints(robot.get_qpos().detach().clone())
    if not torch.allclose(recorded_pose, fresh_pose, atol=1e-4, rtol=0) or any(
        not torch.allclose(recorded_tcps[part], fresh_tcps[part], atol=1e-4, rtol=0)
        for part in motion_parts
    ):
        raise ValueError(
            "Final native held poses do not match their recorded boundary."
        )
    return fresh_pose, fresh_tcps
