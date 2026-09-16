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

"""Opt-in, uniform grasp-fit scaling for one E2 tabletop rigid object."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .source_scene import PreparedScene

__all__: list[str] = []

_POLICY = "e2_single_rigid_grasp_fit/v1"
_MIN_SCALE = 0.25
_PAD_MARGIN = 0.002
_FIT_MARGIN = 0.001


def fit_e2_grasp_asset(
    scene: PreparedScene,
    graph: dict[str, Any],
    *,
    opening: float,
) -> tuple[PreparedScene, dict[str, Any]]:
    """Shrink only the bound E2 target; never change a source mesh or tolerance."""
    groups = graph["task_groups"]
    if len(groups) != 1 or groups[0]["task_type"] != "E2":
        raise ValueError("Grasp-fit scaling supports one E2 task group only.")
    nodes = graph["nodes"]
    picks = [
        node["call"]
        for node in nodes
        if node["call"]["kind"] == "pick"
        or node["call"].get("call_id", "").startswith("gen_sim.pick.")
        or node["call"].get("call_id") == "simulation.pick"
    ]
    if len(picks) != 1:
        raise ValueError("Grasp-fit scaling requires exactly one E2 Pick.")
    pick = picks[0]
    object_id = str(pick.get("object", pick.get("arguments", {}).get("object")))
    if any(node["task_type"] != "E2" for node in nodes):
        raise ValueError("Grasp-fit scaling cannot modify a mixed task graph.")
    if any(
        node["call"].get("arguments", {}).get("reference") not in {None, "table"}
        for node in nodes
    ):
        raise ValueError("Grasp-fit scaling supports only tabletop E2 relations.")
    rigid = [deepcopy(item) for item in scene.rigid_objects]
    planner = [deepcopy(item) for item in scene.planner_objects]
    matches = [item for item in rigid if item["uid"] == object_id]
    if len(matches) != 1:
        raise ValueError("Grasp-fit target must be one rigid object.")
    obj = matches[0]
    shape = obj.get("shape", {})
    if shape.get("shape_type") != "Mesh" or not shape.get("fpath"):
        raise ValueError("Grasp-fit target requires a GLB rigid mesh.")
    if not math.isfinite(opening) or opening <= 2 * _PAD_MARGIN + _FIT_MARGIN:
        raise ValueError("Grasp-fit requires a finite, positive calibrated opening.")

    import trimesh

    loaded = trimesh.load(str(shape["fpath"]), force="scene")
    geometry = loaded.to_geometry()
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not vertices.size
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Grasp-fit target has invalid mesh vertices.")
    # Match the runtime GLB Y-up to DexSim Z-up conversion before applying body scale.
    sim_vertices = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    old_scale = np.asarray(obj.get("body_scale", [1.0] * 3), dtype=np.float64)
    if (
        old_scale.shape != (3,)
        or not np.isfinite(old_scale).all()
        or (old_scale <= 0).any()
    ):
        raise ValueError("Grasp-fit target has invalid body_scale.")
    sim_vertices *= old_scale
    extents = np.ptp(sim_vertices, axis=0)
    major = int(np.argmax(extents))
    transverse = np.delete(extents, major)
    span = float(transverse.max())
    if not math.isfinite(span) or span <= 0:
        raise ValueError("Grasp-fit target has no measurable transverse span.")
    usable = opening - 2 * _PAD_MARGIN
    factor = min(1.0, (usable - _FIT_MARGIN) / span)
    if factor < _MIN_SCALE:
        raise ValueError(
            f"Grasp-fit requires scale {factor:.6f} below minimum {_MIN_SCALE:.2f}."
        )

    record = {
        "policy": _POLICY,
        "object_id": object_id,
        "source_path": str(shape["fpath"]),
        "source_sha256": scene.asset_hashes.get(object_id),
        "calibrated_opening_m": opening,
        "pad_margin_m": _PAD_MARGIN,
        "fit_margin_m": _FIT_MARGIN,
        "usable_span_m": usable,
        "original_transverse_span_m": span,
        "scale_factor": factor,
        "minimum_scale": _MIN_SCALE,
        "mass_policy": "preserve",
        "status": "unchanged" if factor == 1.0 else "scaled",
    }
    if factor == 1.0:
        return scene, record

    rotation = Rotation.from_euler("XYZ", obj["init_rot"], degrees=True)
    original_z = float(obj["init_pos"][2])
    old_bottom = float(rotation.apply(sim_vertices)[:, 2].min() + original_z)
    new_bottom_offset = float(rotation.apply(sim_vertices * factor)[:, 2].min())
    new_z = old_bottom - new_bottom_offset
    new_scale = (old_scale * factor).tolist()
    obj["body_scale"] = new_scale
    obj["init_pos"][2] = new_z
    for item in planner:
        if item["runtime_uid"] == object_id:
            item["body_scale"] = new_scale.copy()
            item["init_pos"][2] = new_z
    record.update(
        original_body_scale=old_scale.tolist(),
        adapted_body_scale=new_scale,
        original_extents_m=extents.tolist(),
        adapted_extents_m=(extents * factor).tolist(),
        support_bottom_z=old_bottom,
        original_origin_z=original_z,
        adapted_origin_z=new_z,
    )
    return (
        replace(scene, rigid_objects=tuple(rigid), planner_objects=tuple(planner)),
        record,
    )
