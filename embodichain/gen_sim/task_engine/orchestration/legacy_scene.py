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

"""Read-only conversion of legacy Gym projects into editable scene revisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Final

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .source_scene import prepare_scene, resolve_source_scene

from .scene_source import (
    SceneSourceFingerprint,
    fingerprint_scene_source,
    verify_scene_source_fingerprint,
)

__all__ = [
    "LEGACY_SCENE_CONVERSION_SCHEMA",
    "LegacySceneRevision",
    "convert_legacy_gym_project",
    "restore_locked_scene_entities",
]

LEGACY_SCENE_CONVERSION_SCHEMA: Final = "embodichain.legacy-scene-conversion/v1"
_CONVERSION_MANIFEST = "legacy_conversion.json"


@dataclass(frozen=True)
class LegacySceneRevision:
    """A new editable revision derived without modifying its legacy source."""

    output_root: Path
    scene_config_path: Path
    scene_graph_path: Path
    manifest_path: Path
    source_fingerprint: SceneSourceFingerprint
    locked_entity_uids: tuple[str, ...]


def convert_legacy_gym_project(
    source: str | Path,
    output_root: str | Path,
) -> LegacySceneRevision:
    """Convert a supported legacy Gym project into a Scene Engine revision.

    Args:
        source: Legacy Gym project directory or explicit configuration path.
        output_root: Empty destination owned by the new scene revision.

    Returns:
        Paths and provenance for the converted revision.

    Raises:
        ValueError: If the source is not legacy or the destination already exists.
        FileNotFoundError: If a referenced source asset is missing.
    """
    resolved = resolve_source_scene(source)
    if resolved.source_format != "legacy_gym_config":
        raise ValueError("Legacy conversion requires a legacy Gym configuration.")
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("Legacy scene revision output_root must be empty.")
    source_fingerprint = fingerprint_scene_source(source)
    prepared = prepare_scene(source)
    export_root = destination / "scene_export"
    assets_root = export_root / "mesh_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    semantics = {str(item.get("uid")): item for item in prepared.planner_objects}

    background = [
        _editable_entry(item, semantics=semantics, assets_root=assets_root)
        for item in prepared.background
    ]
    rigid_objects = [
        _editable_entry(item, semantics=semantics, assets_root=assets_root)
        for item in prepared.rigid_objects
    ]
    articulations = [
        _locked_articulation(
            item,
            source_root=resolved.path.parent,
            destination_root=export_root / "locked_assets",
        )
        for item in prepared.articulations
    ]
    table = next((item for item in background if item.get("uid") == "table"), None)
    if table is None:
        raise ValueError("Legacy conversion requires one table support object.")
    _measure_support_metadata(table, export_root=export_root)
    for item in rigid_objects:
        _measure_center(item, export_root=export_root)

    scene_config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": f"legacy-revision-{source_fingerprint.config_sha256[:16]}",
        "background": background,
        "rigid_object": rigid_objects,
        "articulation": articulations,
    }
    scene_config_path = export_root / "scene_config.json"
    _write_json(scene_config_path, scene_config)
    scene_graph = {
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
                "orientation_state": None,
            },
            *[
                {
                    "object_id": str(item["uid"]),
                    "parent_id": "table",
                    "parent_relation": "on",
                    "table_region": None,
                    "orientation_state": None,
                }
                for item in rigid_objects
            ],
        ],
        "relations": [],
    }
    scene_graph_path = export_root / "scene_graph.json"
    _write_json(scene_graph_path, scene_graph)
    locked_uids = tuple(
        sorted(str(item["uid"]) for item in [*background, *articulations])
    )
    manifest = {
        "schema_version": LEGACY_SCENE_CONVERSION_SCHEMA,
        "source": source_fingerprint.to_dict(),
        "scene_config": scene_config_path.as_posix(),
        "audit_hierarchy": "unknown",
        "operational_hierarchy": "assumed_on_table",
        "assumptions": [
            {
                "uid": str(item["uid"]),
                "relation": "on",
                "parent_uid": "table",
                "confidence": None,
                "source": "operational_assumption",
            }
            for item in rigid_objects
        ],
        "locked_entity_uids": list(locked_uids),
        "locked_articulations": deepcopy(articulations),
        "locked_background": deepcopy(
            [item for item in background if item.get("uid") != "table"]
        ),
    }
    manifest_path = destination / _CONVERSION_MANIFEST
    _write_json(manifest_path, manifest)
    verify_scene_source_fingerprint(source_fingerprint.to_dict())
    return LegacySceneRevision(
        output_root=destination,
        scene_config_path=scene_config_path,
        scene_graph_path=scene_graph_path,
        manifest_path=manifest_path,
        source_fingerprint=source_fingerprint,
        locked_entity_uids=locked_uids,
    )


def restore_locked_scene_entities(revision_root: str | Path) -> Path:
    """Restore collision-only legacy entities after Scene Engine export.

    Args:
        revision_root: Converted revision root containing ``legacy_conversion.json``.

    Returns:
        Updated scene configuration path.

    Raises:
        FileNotFoundError: If the conversion manifest or scene config is absent.
        ValueError: If a generated scene attempts to reuse a locked UID.
    """
    root = Path(revision_root).expanduser().resolve()
    manifest_path = root / _CONVERSION_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Legacy conversion manifest not found: {manifest_path}"
        )
    manifest = _read_mapping(manifest_path)
    if manifest.get("schema_version") != LEGACY_SCENE_CONVERSION_SCHEMA:
        raise ValueError("Legacy conversion manifest schema is invalid.")
    config_path = root / "scene_export" / "scene_config.json"
    config = _read_mapping(config_path)
    existing = {
        str(item.get("uid"))
        for section in ("background", "rigid_object", "articulation")
        for item in config.get(section, ())
        if isinstance(item, Mapping) and item.get("uid")
    }
    for section, key in (
        ("background", "locked_background"),
        ("articulation", "locked_articulations"),
    ):
        values = manifest.get(key, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"Legacy conversion manifest {key} must be a sequence.")
        target = list(config.get(section, ()))
        for raw in values:
            item = deepcopy(dict(raw))
            uid = str(item.get("uid", ""))
            if uid in existing:
                raise ValueError(f"Generated scene reused locked entity UID {uid!r}.")
            existing.add(uid)
            target.append(item)
        config[section] = target
    _write_json(config_path, config)
    verify_scene_source_fingerprint(manifest["source"])
    return config_path


def _editable_entry(
    value: Mapping[str, Any],
    *,
    semantics: Mapping[str, Mapping[str, Any]],
    assets_root: Path,
) -> dict[str, Any]:
    item = deepcopy(dict(value))
    uid = str(item.get("uid", "")).strip()
    if not uid:
        raise ValueError("Converted scene entities require a UID.")
    semantic = semantics.get(uid, {})
    for key in ("category", "name", "description"):
        item[key] = str(semantic.get(key) or item.get(key) or uid)
    shape = item.get("shape")
    if not isinstance(shape, Mapping):
        raise ValueError(f"Legacy scene entity {uid!r} has no supported shape.")
    destination = assets_root / uid / f"{uid}.glb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _shape_to_glb(shape, destination)
    item["shape"] = {
        "shape_type": "Mesh",
        "fpath": destination.relative_to(assets_root.parent).as_posix(),
        "compute_uv": False,
    }
    item.setdefault("body_scale", [1.0, 1.0, 1.0])
    item.setdefault("init_pos", [0.0, 0.0, 0.0])
    item.setdefault("init_rot", [0.0, 0.0, 0.0])
    item.setdefault("attrs", {"mass": 1.0})
    item.setdefault("body_type", "kinematic" if uid == "table" else "dynamic")
    item.setdefault("max_convex_hull_num", 1 if uid == "table" else 16)
    return item


def _shape_to_glb(shape: Mapping[str, Any], destination: Path) -> None:
    shape_type = str(shape.get("shape_type", ""))
    if shape_type == "Mesh":
        source = Path(str(shape.get("fpath", ""))).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Legacy mesh asset not found: {source}")
        mesh = trimesh.load(source, force="scene")
    elif shape_type == "Cube":
        size = _vector(shape.get("size", [1.0, 1.0, 1.0]), length=3)
        mesh = trimesh.Scene(trimesh.creation.box(extents=size))
    elif shape_type == "Sphere":
        radius = float(shape.get("radius", 1.0))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Legacy sphere radius must be positive and finite.")
        mesh = trimesh.Scene(trimesh.creation.icosphere(radius=radius))
    else:
        raise ValueError(f"Unsupported legacy shape_type {shape_type!r}.")
    mesh.export(destination, file_type="glb")


def _locked_articulation(
    value: Mapping[str, Any],
    *,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    item = deepcopy(dict(value))
    uid = str(item.get("uid", "")).strip()
    raw = Path(str(item.get("fpath", ""))).expanduser()
    source = raw.resolve() if raw.is_absolute() else (source_root / raw).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Legacy articulation asset not found: {source}")
    target_root = destination_root / uid
    shutil.copytree(source.parent, target_root, dirs_exist_ok=True)
    copied = target_root / source.name
    item["fpath"] = copied.resolve().as_posix()
    return item


def _measure_support_metadata(entry: dict[str, Any], *, export_root: Path) -> None:
    bounds = _world_bounds(entry, export_root=export_root)
    entry["support_surface_z"] = float(bounds[1, 2])
    rectangle = [
        [float(bounds[0, 0]), float(bounds[0, 1])],
        [float(bounds[1, 0]), float(bounds[0, 1])],
        [float(bounds[1, 0]), float(bounds[1, 1])],
        [float(bounds[0, 0]), float(bounds[1, 1])],
    ]
    entry["support_contour_xy"] = rectangle
    entry["support_optimization_rect_xy"] = deepcopy(rectangle)
    entry["center_xy"] = [
        float((bounds[0, 0] + bounds[1, 0]) / 2.0),
        float((bounds[0, 1] + bounds[1, 1]) / 2.0),
    ]


def _measure_center(entry: dict[str, Any], *, export_root: Path) -> None:
    bounds = _world_bounds(entry, export_root=export_root)
    entry["center_xy"] = [
        float((bounds[0, 0] + bounds[1, 0]) / 2.0),
        float((bounds[0, 1] + bounds[1, 1]) / 2.0),
    ]


def _world_bounds(entry: Mapping[str, Any], *, export_root: Path) -> np.ndarray:
    shape = dict(entry["shape"])
    mesh_path = (export_root / str(shape["fpath"])).resolve()
    loaded = trimesh.load(mesh_path, force="scene")
    mesh = loaded.to_geometry()
    scale = np.asarray(_vector(entry.get("body_scale", [1.0] * 3), length=3))
    mesh.apply_scale(scale)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "XYZ",
        _vector(entry.get("init_rot", [0.0] * 3), length=3),
        degrees=True,
    ).as_matrix()
    transform[:3, 3] = _vector(entry.get("init_pos", [0.0] * 3), length=3)
    mesh.apply_transform(transform)
    return np.asarray(mesh.bounds, dtype=float)


def _vector(value: Any, *, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Legacy scene vector must be a sequence.")
    result = [float(item) for item in value]
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"Legacy scene vector must contain {length} finite values.")
    return result


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must contain an object: {path}")
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
