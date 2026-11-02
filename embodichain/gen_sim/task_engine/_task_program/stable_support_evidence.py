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

"""Versioned stable native support, not instantaneous contact-hull containment."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from embodichain.utils.math import xyz_quat_to_4x4_matrix
from ..contracts import canonical_hash
from .placement_evidence import _valid_pose
from .stability import StabilityConstraint

__all__: list[str] = []
CRITERION = "native_stable_reference_support/v2"


def _rooted_load(raw: dict, obj: int, ref: int, dynamic: set[int]) -> tuple[bool, bool]:
    """Find a simultaneous upward load path and exclude external cargo contacts."""
    if any(type(value) is not int for pair in raw["pairs"] for value in pair):
        raise ValueError("Stable support native actor IDs must be integers.")
    pairs = np.asarray(raw["pairs"], dtype=np.int64).reshape(-1, 2)
    normals = np.asarray(raw["normals"], dtype=float).reshape(-1, 3)
    impulse = np.asarray(raw["impulses"], dtype=float)
    if (
        normals.shape != (len(pairs), 3)
        or impulse.shape != (len(pairs),)
        or not np.isfinite(normals).all()
        or not np.isfinite(impulse).all()
        or (impulse < 0).any()
    ):
        raise ValueError("Stable support native contact arrays are invalid.")
    if (np.linalg.norm(normals, axis=-1)[impulse > 0] <= 1e-8).any():
        raise ValueError("Stable support has a zero native contact normal.")
    edges, contacts = [], []
    for (a, b), normal, value in zip(pairs, normals, impulse, strict=True):
        if value <= 0:
            continue
        contacts.append((a, b))
        if normal[2] > 0:
            edges.append((a, b))
        elif normal[2] < 0:
            edges.append((b, a))
    rooted = {ref}
    coupled = {obj}
    # Do not combine paths across substeps, or traverse the reference into its
    # table. Only known dynamic rigid cargo may extend a native load path.
    while True:
        next_rooted = rooted | {b for a, b in edges if a in rooted and b in dynamic}
        next_coupled = coupled | {
            b
            for a, b in contacts + [(b, a) for a, b in contacts]
            if a in coupled and b in dynamic and b != ref
        }
        if next_rooted == rooted and next_coupled == coupled:
            break
        rooted, coupled = next_rooted, next_coupled
    external = any(
        (a in coupled and b not in rooted) or (b in coupled and a not in rooted)
        for a, b in contacts
    )
    return obj in rooted, external


def stable_support_flags(metrics: dict, timeline: dict) -> list[bool]:
    """Recheck complete sampled windows, native counterparts and relative motion."""
    if (
        timeline.get("errors")
        or metrics.get("criterion") != CRITERION
        or metrics.get("timeline_digest") != canonical_hash(timeline)
    ):
        raise ValueError("Stable support has another criterion or incomplete source.")
    cfg = StabilityConstraint.decode(metrics["constraint"])
    if cfg.entity != metrics["entity"] or cfg.reference != metrics["reference"]:
        raise ValueError("Stable support constraint has different native endpoints.")
    dt, start, end = metrics["step_dt"], metrics["start_step"], metrics["end_step"]
    if (
        type(dt) not in (int, float)
        or not math.isfinite(dt)
        or dt <= 0
        or timeline["step_dt"] != dt
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or not math.isfinite(metrics["duration"])
        or metrics["duration"] <= 0
        or metrics["duration"] != cfg.duration
        or type(metrics["release_timestamp"]) not in (int, float)
        or not math.isfinite(metrics["release_timestamp"])
        or metrics["release_timestamp"] < 0
        or (end - start) * dt + 1e-8 < metrics["duration"]
        or start * dt + 1e-8 < metrics["release_timestamp"]
    ):
        raise ValueError("Stable support lacks its post-release time window.")
    samples = {value["step"]: value for value in timeline["samples"]}
    if len(samples) != len(timeline["samples"]):
        raise ValueError("Stable support has duplicate native samples.")
    object_ids = timeline["object_actors"][metrics["entity"]][0]
    reference_ids = timeline["object_actors"][metrics["reference"]][0]
    if len(object_ids) != 1 or len(reference_ids) != 1 or object_ids == reference_ids:
        raise ValueError("Stable support actors must be distinct exact rigid bodies.")
    obj, ref = object_ids[0], reference_ids[0]
    dynamic = {
        actor
        for name, body_type in timeline["object_body_types"].items()
        if body_type == "dynamic"
        for actor in timeline["object_actors"][name][0]
    }
    if obj not in dynamic:
        raise ValueError("Stable support payload is not a native dynamic rigid body.")
    loaded, external_loads = [], []
    for step in range(start + 1, end + 1):
        sample = samples[step]
        if (
            sample["time"] != step * dt
            or sample["dropped_contacts"] != 0
            or len(sample["rows"]) != 1
        ):
            raise ValueError(
                "Stable support has missing, stale or dropped contact data."
            )
        offset, proofs = 0.0, []
        for substep in sample["substeps"]:
            interval = substep["dt"]
            offset += interval
            if (
                not math.isfinite(interval)
                or interval <= 0
                or not math.isclose(offset, substep["offset"], abs_tol=1e-8, rel_tol=0)
                or substep["dropped_contacts"] != 0
                or len(substep["rows"]) != 1
            ):
                raise ValueError("Stable support has an incomplete native substep.")
            proofs.append(_rooted_load(substep["rows"][0], obj, ref, dynamic))
        if not math.isclose(offset, dt, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Stable support lacks a complete physics interval.")
        loaded.append(
            any(value for value, _ in proofs) and not any(other for _, other in proofs)
        )
        external_loads.append(any(other for _, other in proofs))
    relative = torch.as_tensor(metrics["relative_poses"], dtype=torch.float64)
    if relative.shape != (end - start + 1, 4, 4):
        raise ValueError("Stable support relative pose window has another length.")
    _valid_pose(relative, len(relative))
    translation = torch.linalg.vector_norm(
        relative[:, :3, 3] - relative[0, :3, 3], dim=-1
    )
    rotation = relative[0, :3, :3].T @ relative[:, :3, :3]
    angles = torch.acos(
        ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    )
    stable = bool((translation <= cfg.translation_drift).all()) and bool(
        (angles <= cfg.rotation_drift).all()
    )
    if not stable or any(external_loads):
        return [False]
    if not all(loaded):
        raise ValueError(
            "Native load path is unobserved in part of the stable window; "
            "zero active contacts cannot distinguish sleep from absent support."
        )
    return [True]


def capture_stable_support(
    target: Any,
    bindings: dict,
    uid: str,
    reference: str,
    frame: int,
    configured: StabilityConstraint,
    release_timestamp: float,
) -> tuple[list[bool], dict]:
    """Read exact recorded poses and contacts; missing windows remain unavailable."""
    reader = getattr(target, "_gen_sim_acceptance_contacts", None)
    if reader is None or target.num_envs != 1:
        raise ValueError(
            "Stable support requires its single-environment native timeline."
        )
    timeline = reader.snapshot()
    dt = float(target.step_dt)
    start = frame - math.ceil(configured.duration / dt)
    if start < 0:
        raise ValueError("Stable support window predates this episode.")
    states = target._traj_buffer["states"]["rigid_objects"]
    poses = []
    for name in (uid, reference):
        values = states[bindings[name].simulation_uid]["pose"]
        if values.shape[1] <= frame or frame > int(target._traj_steps[0]):
            raise ValueError("Stable support lacks its exact seeded terminal slot.")
        poses.append(
            xyz_quat_to_4x4_matrix(values[0, start : frame + 1].detach().cpu()).double()
        )
    relative = torch.linalg.solve(poses[1], poses[0])
    metrics = {
        "scope": "native_stable_support",
        "criterion": CRITERION,
        "entity": uid,
        "reference": reference,
        "start_step": start,
        "end_step": frame,
        "step_dt": dt,
        "duration": configured.duration,
        "release_timestamp": release_timestamp,
        "constraint": asdict(configured),
        "relative_poses": relative.tolist(),
        "timeline_digest": canonical_hash(timeline),
        "object_pose": poses[0][-1:].tolist(),
        "reference_pose": poses[1][-1:].tolist(),
    }
    return stable_support_flags(metrics, timeline), metrics
