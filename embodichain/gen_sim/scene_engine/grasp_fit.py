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

"""Explicit source-mesh grasp-fit declarations independent of Task Engine graphs."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from embodichain.compute.geometry.grasp_fit import compute_grasp_fit_scale
from embodichain.lab.sim.scene.source_geometry import read_source_vertices

__all__ = [
    "SourceGraspFitRequest",
    "SourceGraspFitResult",
    "fit_source_grasp_declarations",
]


@dataclass(frozen=True, slots=True)
class SourceGraspFitRequest:
    """Explicit target and calibrated jaw/span inputs, without semantic task types."""

    object_id: str
    opening: float
    contact_clearance: float = 0.0
    fit_mode: Literal["minimum_axis", "single_arm_transverse"] = "single_arm_transverse"
    source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SourceGraspFitResult:
    """Owned source rigid declarations and ordered numeric adaptation receipts."""

    rigid_objects: tuple[dict[str, Any], ...]
    records: tuple[dict[str, Any], ...]


def fit_source_grasp_declarations(
    rigid_objects: Sequence[Mapping[str, Any]],
    requests: Sequence[SourceGraspFitRequest],
) -> SourceGraspFitResult:
    """Adapt explicitly acquired mesh declarations while retaining their support.

    Args:
        rigid_objects: Ordered source physical rigid declarations.
        requests: Unique target declarations with actual calibrated openings.

    Returns:
        Owned objects and receipts in source-sorted target order. Source files,
        mass and unrelated rigid objects remain unchanged. Asset normalization
        is a separate later stage so this function never writes a mesh or bundle.
    """
    owned = tuple(deepcopy(dict(item)) for item in rigid_objects)
    by_uid = {str(item["uid"]): item for item in owned}
    if len(by_uid) != len(owned):
        raise ValueError("Grasp-fit rigid identities must be unique.")
    selected = {value.object_id: value for value in requests}
    if any(type(value) is not SourceGraspFitRequest for value in requests) or len(
        selected
    ) != len(requests):
        raise ValueError("Grasp-fit requests require unique typed target declarations.")
    records = []
    for object_id, request in sorted(selected.items()):
        obj = by_uid.get(object_id)
        if obj is None:
            raise ValueError(
                f"Grasp-fit target {object_id!r} must be one bound rigid object."
            )
        shape = obj.get("shape", {})
        if shape.get("shape_type") != "Mesh" or not shape.get("fpath"):
            raise ValueError(
                f"Grasp-fit target {object_id!r} requires a GLB rigid mesh."
            )
        old_scale = np.asarray(obj.get("body_scale", [1.0] * 3), dtype=np.float64)
        unscaled = deepcopy(obj)
        unscaled["body_scale"] = [1.0] * 3
        vertices = read_source_vertices(unscaled)
        original_z = float(obj["init_pos"][2])
        fit = compute_grasp_fit_scale(
            vertices,
            body_scale=tuple(old_scale),
            rotation_xyz_degrees=tuple(obj["init_rot"]),
            origin_z=original_z,
            opening=request.opening,
            contact_clearance=request.contact_clearance,
            fit_mode=request.fit_mode,
        )
        record = dict(
            object_id=object_id,
            fit_mode=request.fit_mode,
            source_path=str(shape["fpath"]),
            source_sha256=request.source_sha256,
            calibrated_opening_m=request.opening,
            pad_margin_m=fit.pad_margin,
            contact_clearance_m=float(request.contact_clearance),
            fit_margin_m=0.001,
            usable_span_m=fit.usable_span,
            original_fit_span_m=fit.fit_span,
            scale_factor=fit.factor,
            mass_policy="preserve",
            status="unchanged" if fit.factor == 1.0 else "scaled",
        )
        if fit.factor != 1.0:
            obj["body_scale"] = list(fit.body_scale)
            obj["init_pos"][2] = fit.origin_z
            if obj.get("init_local_pose") is not None:
                pose = np.asarray(obj["init_local_pose"], dtype=float).copy()
                pose[2, 3] = fit.origin_z
                obj["init_local_pose"] = pose.tolist()
            record.update(
                original_body_scale=old_scale.tolist(),
                adapted_body_scale=list(fit.body_scale),
                original_extents_m=list(fit.extents),
                adapted_extents_m=(np.asarray(fit.extents) * fit.factor).tolist(),
                support_bottom_z=fit.support_bottom_z,
                original_origin_z=original_z,
                adapted_origin_z=fit.origin_z,
            )
        records.append(record)
    return SourceGraspFitResult(owned, tuple(records))
