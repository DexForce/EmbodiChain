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

"""Pure snapshot qualification of explicitly selected support evidence scopes.

Import is standard-library-only. Numerical re-evaluation loads the shared compute
implementation only when a complete physical payload requires it.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .canonicalization import canonical_hash

__all__: list[str] = []

CRITERION = "native_stable_reference_support/v2"


def stable_support_flags(
    metrics: Mapping[str, Any],
    timeline: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
) -> list[bool]:
    """Recheck complete sampled windows, native counterparts and relative motion."""
    if (
        timeline.get("errors")
        or metrics.get("criterion") != CRITERION
        or metrics.get("timeline_digest") != canonical_hash(timeline)
    ):
        raise ValueError("Stable support has another criterion or incomplete source.")
    import torch
    from embodichain.compute.geometry.contact_support import contact_load_path
    from embodichain.compute.geometry.rigid import (
        rigid_pose_displacement,
        validate_rigid_pose_batch,
    )

    if canonical_hash(metrics["constraint"]) != canonical_hash(condition):
        raise ValueError("Stable support has a different normative condition.")
    cfg = condition
    if cfg["entity"] != metrics["entity"] or cfg["reference"] != metrics["reference"]:
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
        or metrics["duration"] != cfg["duration"]
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
            raw = substep["rows"][0]
            proofs.append(
                contact_load_path(
                    raw["pairs"],
                    raw["normals"],
                    raw["impulses"],
                    object_actor=obj,
                    reference_actor=ref,
                    dynamic_actors=dynamic,
                )
            )
        if not math.isclose(offset, dt, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Stable support lacks a complete physics interval.")
        loaded.append(
            any(value for value, _ in proofs) and not any(other for _, other in proofs)
        )
        external_loads.append(any(other for _, other in proofs))
    relative = torch.as_tensor(metrics["relative_poses"], dtype=torch.float64)
    if relative.shape != (end - start + 1, 4, 4):
        raise ValueError("Stable support relative pose window has another length.")
    validate_rigid_pose_batch(relative, len(relative))
    translation, angles = rigid_pose_displacement(
        relative, relative[0].expand_as(relative)
    )
    stable = bool((translation <= cfg["translation_drift"]).all()) and bool(
        (angles <= cfg["rotation_drift"]).all()
    )
    if not stable or any(external_loads):
        return [False]
    if not all(loaded):
        raise ValueError(
            "Native load path is unobserved in part of the stable window; "
            "zero active contacts cannot distinguish sleep from absent support."
        )
    return [True]
