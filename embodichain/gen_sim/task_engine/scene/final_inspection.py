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

"""Geometry-derived evidence from one completed scene revision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np
from embodichain.lab.sim.scene.source_measurement import measure_source_geometry
from embodichain.compute.geometry.source_relations import (
    source_orientation_label,
    source_support_proposal,
)

from embodichain.gen_sim.task_engine.orchestration.source_scene import (
    PreparedScene,
    prepare_scene,
    resolve_source_scene,
)

__all__ = [
    "FINAL_SCENE_INSPECTION_SCHEMA",
    "FinalSceneInspection",
    "apply_final_inspection",
    "inspect_final_scene",
    "validate_final_scene_inspection",
]

FINAL_SCENE_INSPECTION_SCHEMA: Final = "embodichain.final-scene-inspection/v1"
FinalSceneInspection: TypeAlias = dict[str, Any]


def inspect_final_scene(
    source: str | Path | PreparedScene,
    *,
    revision_id: str,
    contact_tolerance_m: float = 0.03,
) -> FinalSceneInspection:
    """Measure final AABBs, orientation, and support from exported geometry.

    Args:
        source: Completed scene project, configuration path, or an explicitly
            selected normalized Z-up physical scene. Prepared inputs are
            measured directly without reloading their original source model.
        revision_id: Content identity already assigned to the completed revision.
        contact_tolerance_m: Maximum support-surface contact gap in meters.

    Returns:
        Strict geometry-derived final inspection document.
    """
    tolerance = float(contact_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("contact_tolerance_m must be positive and finite.")
    normalized_revision_id = str(revision_id)
    if len(normalized_revision_id) != 64:
        raise ValueError("revision_id must be a SHA-256 hexadecimal digest.")
    try:
        int(normalized_revision_id, 16)
    except ValueError as exc:
        raise ValueError("revision_id must be a SHA-256 hexadecimal digest.") from exc
    if isinstance(source, PreparedScene):
        prepared = source
        convert_y_up = False
    else:
        resolved = resolve_source_scene(source)
        prepared = prepare_scene(source)
        convert_y_up = resolved.is_prompt2scene
    runtime = {
        str(item.get("uid")): item
        for item in (
            *prepared.background,
            *prepared.rigid_objects,
            *prepared.articulations,
        )
        if isinstance(item, Mapping) and item.get("uid")
    }
    measured: dict[str, dict[str, Any]] = {}
    for raw in prepared.planner_objects:
        uid = str(raw.get("uid", ""))
        role = str(raw.get("role", ""))
        geometry = measure_source_geometry(
            runtime.get(uid, raw),
            convert_y_up=convert_y_up,
        )
        measured[uid] = {
            "uid": uid,
            "role": role,
            "orientation": source_orientation_label(geometry),
            "support": {
                "parent_uid": None if uid == "table" else "unknown",
                "relation": "root" if uid == "table" else "unknown",
                "confidence": 1.0 if uid == "table" else None,
                "gap_m": None,
                "xy_overlap_ratio": None,
            },
            "world_aabb": (
                None
                if geometry is None
                else {
                    "min": geometry["bounds"][0].tolist(),
                    "max": geometry["bounds"][1].tolist(),
                }
            ),
            "evidence": {
                "source": "final_geometry" if geometry is not None else "unmeasured",
                "method": "world_aabb_and_dominant_axis",
            },
        }

    for uid, item in measured.items():
        child_geometry = _geometry_from_record(item)
        if uid == "table" or child_geometry is None:
            continue
        support = source_support_proposal(
            uid,
            child_geometry,
            measured,
            tolerance=tolerance,
        )
        if support is not None:
            item["support"] = support

    return validate_final_scene_inspection(
        {
            "schema_version": FINAL_SCENE_INSPECTION_SCHEMA,
            "scene_revision_id": normalized_revision_id,
            "source_config_path": prepared.source_config_path.as_posix(),
            "contact_tolerance_m": tolerance,
            "objects": [measured[uid] for uid in sorted(measured)],
        }
    )


def apply_final_inspection(
    prepared_scene: PreparedScene,
    inspection: Mapping[str, Any],
) -> PreparedScene:
    """Return a detached PreparedScene enriched with measured final evidence.

    Args:
        prepared_scene: Normalized scene to enrich without mutation.
        inspection: Validated or raw final inspection mapping.

    Returns:
        Prepared scene whose semantic state reflects measured final geometry.
    """
    normalized = validate_final_scene_inspection(inspection)
    by_uid = {str(item["uid"]): item for item in normalized["objects"]}
    planner_objects = []
    for raw in prepared_scene.planner_objects:
        item = deepcopy(raw)
        evidence = by_uid.get(str(item.get("uid")))
        if evidence is not None:
            initial_state = deepcopy(dict(item.get("initial_state", {})))
            initial_state.pop("orientation", None)
            if evidence["orientation"] == "standing":
                initial_state["orientation"] = "upright"
            elif evidence["orientation"] == "lying":
                initial_state["orientation"] = "fallen"
            attributes = deepcopy(dict(item.get("attributes", {})))
            attributes["final_support"] = deepcopy(evidence["support"])
            attributes["final_world_aabb"] = deepcopy(evidence["world_aabb"])
            item["initial_state"] = initial_state
            item["attributes"] = attributes
        planner_objects.append(item)
    return replace(prepared_scene, planner_objects=tuple(planner_objects))


def validate_final_scene_inspection(
    value: Mapping[str, Any],
) -> FinalSceneInspection:
    """Validate and detach one final scene inspection document.

    Args:
        value: Inspection mapping to validate.

    Returns:
        Detached, normalized inspection document.
    """
    if not isinstance(value, Mapping):
        raise TypeError("FinalSceneInspection must be a mapping.")
    result = deepcopy(dict(value))
    expected = {
        "schema_version",
        "scene_revision_id",
        "source_config_path",
        "contact_tolerance_m",
        "objects",
    }
    if set(result) != expected:
        raise ValueError("FinalSceneInspection fields are invalid.")
    if result.get("schema_version") != FINAL_SCENE_INSPECTION_SCHEMA:
        raise ValueError("FinalSceneInspection schema version is invalid.")
    revision_id = result.get("scene_revision_id")
    if not isinstance(revision_id, str) or len(revision_id) != 64:
        raise ValueError("FinalSceneInspection.scene_revision_id is invalid.")
    try:
        int(revision_id, 16)
    except ValueError as exc:
        raise ValueError("FinalSceneInspection.scene_revision_id is invalid.") from exc
    source_path = result.get("source_config_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("FinalSceneInspection.source_config_path is invalid.")
    tolerance = result.get("contact_tolerance_m")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError("FinalSceneInspection.contact_tolerance_m is invalid.")
    objects = result.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise TypeError("FinalSceneInspection.objects must be a sequence.")
    normalized = [_validate_object(item, index) for index, item in enumerate(objects)]
    if len({item["uid"] for item in normalized}) != len(normalized):
        raise ValueError("FinalSceneInspection object UIDs must be unique.")
    result["objects"] = normalized
    result["contact_tolerance_m"] = float(tolerance)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _validate_object(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FinalSceneInspection.objects[{index}] must be a mapping.")
    item = deepcopy(dict(value))
    expected = {"uid", "role", "orientation", "support", "world_aabb", "evidence"}
    if set(item) != expected:
        raise ValueError(f"FinalSceneInspection.objects[{index}] fields are invalid.")
    if not isinstance(item["uid"], str) or not item["uid"]:
        raise ValueError(f"FinalSceneInspection.objects[{index}].uid is invalid.")
    if not isinstance(item["role"], str) or not item["role"]:
        raise ValueError(f"FinalSceneInspection.objects[{index}].role is invalid.")
    if item["orientation"] not in {"standing", "lying", "unknown"}:
        raise ValueError(
            f"FinalSceneInspection.objects[{index}].orientation is invalid."
        )
    if not isinstance(item["support"], Mapping) or not isinstance(
        item["evidence"], Mapping
    ):
        raise TypeError("FinalSceneInspection support and evidence must be mappings.")
    support = deepcopy(dict(item["support"]))
    if set(support) != {
        "parent_uid",
        "relation",
        "confidence",
        "gap_m",
        "xy_overlap_ratio",
    }:
        raise ValueError("FinalSceneInspection support fields are invalid.")
    if support["parent_uid"] is not None and not isinstance(support["parent_uid"], str):
        raise TypeError("FinalSceneInspection support parent_uid is invalid.")
    if support["relation"] not in {"root", "on", "unknown"}:
        raise ValueError("FinalSceneInspection support relation is invalid.")
    for field_name in ("confidence", "gap_m", "xy_overlap_ratio"):
        field_value = support[field_name]
        if field_value is not None and (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not np.isfinite(float(field_value))
        ):
            raise ValueError(f"FinalSceneInspection support {field_name} is invalid.")
    if (
        support["confidence"] is not None
        and not 0.0 <= float(support["confidence"]) <= 1.0
    ):
        raise ValueError("FinalSceneInspection support confidence is invalid.")
    if (
        support["xy_overlap_ratio"] is not None
        and not 0.0 <= float(support["xy_overlap_ratio"]) <= 1.0 + 1.0e-6
    ):
        raise ValueError("FinalSceneInspection support overlap is invalid.")
    item["support"] = support
    aabb = item["world_aabb"]
    if aabb is not None:
        if not isinstance(aabb, Mapping) or set(aabb) != {"min", "max"}:
            raise ValueError("FinalSceneInspection world_aabb is invalid.")
        if aabb["min"] is None or aabb["max"] is None:
            raise ValueError("FinalSceneInspection world_aabb vectors are invalid.")
        minimum = _vector(aabb["min"], default=(0.0, 0.0, 0.0))
        maximum = _vector(aabb["max"], default=(0.0, 0.0, 0.0))
        if np.any(np.asarray(maximum) < np.asarray(minimum)):
            raise ValueError("FinalSceneInspection world_aabb bounds are inverted.")
        item["world_aabb"] = {"min": minimum, "max": maximum}
    evidence = deepcopy(dict(item["evidence"]))
    if set(evidence) != {"source", "method"} or any(
        not isinstance(evidence[key], str) or not evidence[key] for key in evidence
    ):
        raise ValueError("FinalSceneInspection evidence is invalid.")
    item["evidence"] = evidence
    return item


def _geometry_from_record(item: Mapping[str, Any]) -> np.ndarray | None:
    aabb = item.get("world_aabb")
    if not isinstance(aabb, Mapping):
        return None
    return np.asarray([aabb["min"], aabb["max"]], dtype=float)


def _vector(value: Any, *, default: tuple[float, float, float]) -> list[float]:
    raw = default if value is None else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("Scene geometry vectors must be sequences.")
    result = [float(item) for item in raw]
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("Scene geometry vectors must contain three finite values.")
    return result
