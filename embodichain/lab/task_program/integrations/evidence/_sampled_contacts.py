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

"""Raw acquisition checks for explicitly bound sampled native support."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any
from embodichain._json_contract import canonical_hash
from ._held_evidence import number, flags

__all__: list[str] = []


def validate_sampled_identity(binding: Any) -> None:
    from ._values import _digest, _identifier

    identity = binding.sampled_contact_identity
    if not isinstance(identity, Mapping) or set(identity) != {
        "program_fingerprint",
        "sensor_uid",
        "step_dt",
        "object_native_uids",
        "object_actors",
        "object_body_types",
        "pad_actors",
        "topology_revision",
        "timeline_episode_generations",
    }:
        raise ValueError(
            "Sampled contacts require a complete program/native acquisition identity."
        )
    _digest(identity["program_fingerprint"], "program_fingerprint")
    _identifier(identity["sensor_uid"], "sensor_uid")
    if (
        number(identity["step_dt"]) <= 0
        or type(identity["topology_revision"]) is not int
        or identity["topology_revision"] < 0
        or len(identity["timeline_episode_generations"]) != 1
        or any(
            type(value) is not int or value < 0
            for value in identity["timeline_episode_generations"]
        )
    ):
        raise ValueError("Sampled contact clocks/topology/generation are invalid.")
    objects = identity["object_native_uids"]
    if (
        not isinstance(objects, dict)
        or not {binding.object_id, binding.reference_id} <= set(objects)
        or set(identity["object_actors"]) != set(objects)
        or set(identity["object_body_types"]) != set(objects)
    ):
        raise ValueError(
            "Sampled contacts require their complete rigid actor universe."
        )
    for name, uid in objects.items():
        _identifier(name, "entity")
        _identifier(uid, "native_uid")
        if identity["object_body_types"][name] not in {
            "dynamic",
            "static",
            "kinematic",
        }:
            raise ValueError("Sampled contacts have an unknown rigid body type.")
    inventory = []
    for mapping, size in ((identity["object_actors"], 1), (identity["pad_actors"], 2)):
        for name, rows in mapping.items():
            _identifier(name, "actor binding")
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], list)
                or len(rows[0]) != size
                or any(type(actor) is not int for actor in rows[0])
            ):
                raise ValueError("Sampled contact actor rows are malformed.")
            inventory.extend(rows[0])
    if len(set(inventory)) != len(inventory) or len(set(objects.values())) != len(
        objects
    ):
        raise ValueError("Sampled contact native owners must be disjoint.")


def read_sampled_timeline(binding: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one raw native timeline before any operation-specific decision."""
    identity = binding.sampled_contact_identity
    if (
        raw.get("schema_version") != "simulation.recorded_contacts/v1"
        or raw.get("env_ids") != [0]
        or any(type(x) is not int for x in raw["env_ids"])
        or raw.get("episode_ids") != list(binding.episode_ids)
        or any(type(x) is not int for x in raw["episode_ids"])
        or raw.get("object_id") != binding.object_id
        or raw.get("reference_id") != binding.reference_id
    ):
        raise ValueError("Sampled support has another physical declaration or episode.")
    if any(
        canonical_hash(raw[key]) != canonical_hash(value)
        for key, value in identity.items()
    ):
        raise ValueError(
            "Sampled support has another compiled/native acquisition identity."
        )
    dt = identity["step_dt"]
    recorded = raw["timeline"]
    if (
        recorded["step_dt"] != dt
        or recorded["scope"] != "native_substep_samples_not_continuous_contact_proof"
        or len(recorded["environments"]) != 1
    ):
        raise ValueError("Sampled contact timeline has another clock or row scope.")
    row = recorded["environments"][0]
    if (
        type(row["env_id"]) is not int
        or row["env_id"] != 0
        or row["episode_generation"] != identity["timeline_episode_generations"][0]
        or type(row["episode_generation"]) is not int
        or row["errors"]
        or row["incomplete_control_step"] is not None
    ):
        raise ValueError(
            "Sampled contact timeline is incomplete or belongs to another generation."
        )
    samples = []
    for sample in row["samples"]:
        step = sample["step"]
        number(sample["time"])
        if (
            type(sample["native_revision"]) is not int
            or type(sample["topology_revision"]) is not int
            or type(sample["episode_generation"]) is not int
        ):
            raise ValueError("Recorded contact source identities must be integers.")
        if (
            type(step) is not int
            or step <= 0
            or sample["available"] is not True
            or sample["episode_generation"] != row["episode_generation"]
            or sample["topology_revision"] != identity["topology_revision"]
        ):
            raise ValueError("Recorded contact boundary has another acquisition scope.")
        substeps = []
        for sub in sample["substeps"]:
            if (
                sub["available"] is not True
                or sub["topology_revision"] != identity["topology_revision"]
                or type(sub["native_revision"]) is not int
                or sub["native_revision"] < 0
                or type(sub["observation_revision"]) is not int
                or sub["observation_revision"] < 0
                or not math.isclose(
                    number(sub["timestamp"]),
                    (step - 1) * dt + number(sub["offset"]),
                    rel_tol=0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(
                    "Recorded native substep acquisition is unavailable or misaligned."
                )
            substeps.append({**sub, "rows": [sub["contacts"]]})
        if (
            not substeps
            or sample["native_revision"] != substeps[-1]["native_revision"]
            or sample["contacts"] != substeps[-1]["contacts"]
        ):
            raise ValueError("Recorded boundary differs from its last native sample.")
        samples.append(
            {
                **sample,
                "rows": [sample["contacts"]],
                "substeps": substeps,
                "hand_opening": {
                    part: [value] for part, value in sample["hand_opening"].items()
                },
            }
        )
    timeline = dict(
        scope=recorded["scope"],
        step_dt=dt,
        object_actors=identity["object_actors"],
        object_body_types=identity["object_body_types"],
        pad_actors=identity["pad_actors"],
        errors=[],
        samples=samples,
    )
    return timeline


def prepare_sampled_support(
    binding: Any, snapshot: Mapping[str, Any], boundary: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

    raw = snapshot["recorded_contacts"]
    identity = binding.sampled_contact_identity
    dt = identity["step_dt"]
    timeline = read_sampled_timeline(binding, raw)
    ends = boundary["end_steps"]
    if (
        len(ends) != 1
        or type(ends[0]) is not int
        or ends[0] <= 0
        or raw.get("boundary_frames") != ends
        or any(type(x) is not int for x in raw["boundary_frames"])
    ):
        raise ValueError("Sampled support has another completed boundary.")
    if "boundary_frames" in snapshot and snapshot["boundary_frames"] != ends:
        raise ValueError("Sampled support report has another completed boundary.")
    end = ends[0]
    from .observation_requests import _sampled_support_window as sampled_support_window

    starts, _ = sampled_support_window(binding, ends)
    start = starts[0]
    if (
        start < 0
        or raw.get("window_start_frames") != [start]
        or type(raw["window_start_frames"][0]) is not int
    ):
        raise ValueError(
            "Sampled support must retain its original exact duration window."
        )
    frame = raw["pose_frame"]
    number(frame["timestamp"])
    if (
        frame["env_ids"] != [0]
        or any(type(x) is not int for x in frame["env_ids"])
        or type(frame["observation_revision"]) is not int
        or frame["observation_revision"] < 0
        or frame["clock_domain"] != "simulation_time"
        or frame["revision_kind"] != "recorded_contact_pose_acquisition"
    ):
        raise ValueError("Sampled support pose acquisition has another clock or row.")
    poses = []
    for entity in (binding.object_id, binding.reference_id):
        key = "rigid.pose_window:" + entity
        batch = frame["batches"][key]
        if (
            batch["evidence_id"] != key
            or batch["value_dtype"] != "torch.float64"
            or batch["env_ids"] != [0]
            or any(type(x) is not int for x in batch["env_ids"])
            or batch["timestamp"] != frame["timestamp"]
            or type(batch["observation_revision"]) is not int
            or batch["observation_revision"] != frame["observation_revision"]
            or batch["acquisition_errors"] != [None]
        ):
            raise ValueError("Sampled support pose channel is missing or inconsistent.")
        flags(batch["valid"], 1)
        pose = torch.as_tensor(batch["values"], dtype=torch.float64)
        if pose.shape != (1, end - start + 1, 4, 4):
            raise ValueError("Sampled support poses have another exact seeded window.")
        validate_rigid_pose_batch(pose[0], end - start + 1)
        poses.append(pose[0])
    relative = torch.linalg.solve(poses[1], poses[0])
    metrics = dict(
        scope="native_stable_support",
        criterion="native_stable_reference_support/v2",
        entity=binding.object_id,
        reference=binding.reference_id,
        reference_uid=binding.reference_id,
        relation="on",
        observation_kind=binding.observation_kind,
        placement_node_id=binding.release_segment_id,
        boundary_frames=ends,
        start_step=start,
        end_step=end,
        step_dt=dt,
        duration=binding.condition["duration"],
        constraint=binding.condition,
        relative_poses=relative.tolist(),
        timeline_digest=canonical_hash(timeline),
        object_pose=poses[0][-1:].tolist(),
        reference_pose=poses[1][-1:].tolist(),
    )
    return (
        metrics,
        timeline,
        {
            binding.object_id: metrics["object_pose"],
            binding.reference_id: metrics["reference_pose"],
        },
    )
