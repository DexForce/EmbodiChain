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

"""Opt-in, uniform grasp-fit scaling for grasped rigid objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .source_scene import PreparedScene

__all__: list[str] = []

_POLICY = "semantic_grasp_fit/v3"
_MIN_SCALE = 0.25
_PAD_MARGIN = 0.002
_FIT_MARGIN = 0.001
_COORDINATED_CALLS = {
    "simulation.coordinated_hold",
    "simulation.coordinated_transport",
}


def fit_grasp_assets(
    scene: PreparedScene,
    graph: dict[str, Any],
    *,
    openings: dict[str, float],
    contact_clearances: dict[str, float] | None = None,
) -> tuple[PreparedScene, dict[str, Any]]:
    """Shrink acquired rigid meshes while retaining support and bounded scale.

    ``contact_clearances`` optionally supplies each target's required per-jaw
    separation from the open gripper. When provided, every acquired object must
    have a finite non-negative value. The geometric pad margin remains a floor.
    """
    requests = _grasp_requests(graph, openings)
    rigid = [deepcopy(item) for item in scene.rigid_objects]
    planner = [deepcopy(item) for item in scene.planner_objects]
    rigid_by_uid = {str(item["uid"]): item for item in rigid}
    planner_by_uid = {
        str(item["runtime_uid"]): item
        for item in planner
        if item.get("role") == "rigid_object"
    }
    records = []
    for object_id, request in sorted(requests.items()):
        if contact_clearances is not None and object_id not in contact_clearances:
            raise ValueError(
                f"Missing contact clearance for grasp-fit target {object_id!r}."
            )
        obj = rigid_by_uid.get(object_id)
        planner_obj = planner_by_uid.get(object_id)
        if obj is None or planner_obj is None:
            raise ValueError(
                f"Grasp-fit target {object_id!r} must be one bound rigid object."
            )
        records.append(
            _fit_object(
                obj,
                planner_obj,
                object_id=object_id,
                opening=request["opening"],
                fit_mode=request["fit_mode"],
                task_types=sorted(request["task_types"]),
                resources=sorted(request["resources"]),
                source_sha256=scene.asset_hashes.get(object_id),
                contact_clearance=(
                    0.0 if contact_clearances is None else contact_clearances[object_id]
                ),
            )
        )
    status = (
        "not_applicable"
        if not records
        else (
            "scaled"
            if any(record["status"] == "scaled" for record in records)
            else "unchanged"
        )
    )
    report = {
        "schema_version": "gen_sim.asset-adaptation/v3",
        "policy": _POLICY,
        "enabled": True,
        "minimum_scale": _MIN_SCALE,
        "status": status,
        "records": records,
    }
    if status != "scaled":
        return scene, report
    return (
        replace(scene, rigid_objects=tuple(rigid), planner_objects=tuple(planner)),
        report,
    )


def _grasp_requests(
    graph: dict[str, Any], openings: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Collect objects acquired by single-arm Pick or coordinated E5 calls."""
    requests: dict[str, dict[str, Any]] = {}
    for node in graph["nodes"]:
        call = node["call"]
        call_id = str(call.get("call_id", ""))
        is_pick = call.get("kind") == "pick" or call_id == "simulation.pick"
        is_pick |= call_id.startswith("gen_sim.pick.")
        coordinated = call_id in _COORDINATED_CALLS
        if not is_pick and not coordinated:
            continue
        arguments = call.get("arguments", {})
        object_id = str(call.get("object", arguments.get("object", ""))).strip()
        if not object_id:
            raise ValueError("Grasp-fit acquisition has no object identifier.")
        resources = (
            tuple(str(value) for value in call["resources"].values())
            if coordinated
            else (str(call["resources"]["primary"]),)
        )
        missing = [resource for resource in resources if resource not in openings]
        if missing:
            raise ValueError(f"Grasp-fit has no calibrated opening for {missing}.")
        opening = min(openings[resource] for resource in resources)
        task_type = str(node["task_type"])
        fit_mode = "minimum_axis" if coordinated else "single_arm_transverse"
        current = requests.setdefault(
            object_id,
            {
                "opening": opening,
                "fit_mode": fit_mode,
                "task_types": set(),
                "resources": set(),
            },
        )
        current["opening"] = min(current["opening"], opening)
        if fit_mode == "single_arm_transverse":
            current["fit_mode"] = fit_mode
        current["task_types"].add(task_type)
        current["resources"].update(resources)
    return requests


def _fit_object(
    obj: dict[str, Any],
    planner_obj: dict[str, Any],
    *,
    object_id: str,
    opening: float,
    fit_mode: str,
    task_types: list[str],
    resources: list[str],
    source_sha256: str | None,
    contact_clearance: float,
) -> dict[str, Any]:
    """Apply one uniform scale while retaining the original support bottom."""
    shape = obj.get("shape", {})
    if shape.get("shape_type") != "Mesh" or not shape.get("fpath"):
        raise ValueError(f"Grasp-fit target {object_id!r} requires a GLB rigid mesh.")
    if (
        isinstance(contact_clearance, bool)
        or not isinstance(contact_clearance, (int, float))
        or not math.isfinite(contact_clearance)
        or contact_clearance < 0
    ):
        raise ValueError("Grasp-fit contact clearance must be finite and non-negative.")
    pad_margin = max(_PAD_MARGIN, float(contact_clearance))
    if not math.isfinite(opening) or opening <= 2 * pad_margin + _FIT_MARGIN:
        raise ValueError("Grasp-fit requires a finite, positive calibrated opening.")

    import trimesh

    loaded = trimesh.load(str(shape["fpath"]), force="scene")
    geometry = (
        loaded.to_geometry()
        if hasattr(loaded, "to_geometry")
        else loaded.dump(concatenate=True)
    )
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not vertices.size
        or not np.isfinite(vertices).all()
    ):
        raise ValueError(f"Grasp-fit target {object_id!r} has invalid vertices.")
    sim_vertices = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    old_scale = np.asarray(obj.get("body_scale", [1.0] * 3), dtype=np.float64)
    if (
        old_scale.shape != (3,)
        or not np.isfinite(old_scale).all()
        or (old_scale <= 0).any()
    ):
        raise ValueError(f"Grasp-fit target {object_id!r} has invalid body_scale.")
    sim_vertices *= old_scale
    extents = np.ptp(sim_vertices, axis=0)
    if fit_mode == "single_arm_transverse":
        span = float(np.delete(extents, int(np.argmax(extents))).max())
    else:
        span = float(extents.min())
    if not math.isfinite(span) or span <= 0:
        raise ValueError(f"Grasp-fit target {object_id!r} has no measurable span.")
    usable = opening - 2 * pad_margin
    factor = min(1.0, (usable - _FIT_MARGIN) / span)
    if factor < _MIN_SCALE:
        raise ValueError(
            f"Grasp-fit target {object_id!r} requires scale {factor:.6f} "
            f"below minimum {_MIN_SCALE:.2f}."
        )

    record = {
        "object_id": object_id,
        "task_types": task_types,
        "resources": resources,
        "fit_mode": fit_mode,
        "source_path": str(shape["fpath"]),
        "source_sha256": source_sha256,
        "calibrated_opening_m": opening,
        "pad_margin_m": pad_margin,
        "contact_clearance_m": float(contact_clearance),
        "fit_margin_m": _FIT_MARGIN,
        "usable_span_m": usable,
        "original_fit_span_m": span,
        "scale_factor": factor,
        "mass_policy": "preserve",
        "status": "unchanged" if factor == 1.0 else "scaled",
    }
    if factor == 1.0:
        return record

    rotation = Rotation.from_euler("XYZ", obj["init_rot"], degrees=True)
    original_z = float(obj["init_pos"][2])
    old_bottom = float(rotation.apply(sim_vertices)[:, 2].min() + original_z)
    new_bottom_offset = float(rotation.apply(sim_vertices * factor)[:, 2].min())
    new_z = old_bottom - new_bottom_offset
    new_scale = (old_scale * factor).tolist()
    obj["body_scale"] = new_scale
    obj["init_pos"][2] = new_z
    planner_obj["body_scale"] = new_scale.copy()
    planner_obj["init_pos"][2] = new_z
    record.update(
        original_body_scale=old_scale.tolist(),
        adapted_body_scale=new_scale,
        original_extents_m=extents.tolist(),
        adapted_extents_m=(extents * factor).tolist(),
        support_bottom_z=old_bottom,
        original_origin_z=original_z,
        adapted_origin_z=new_z,
    )
    return record
