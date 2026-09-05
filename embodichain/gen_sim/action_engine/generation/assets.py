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

"""Normalize GLB node transforms and body scale into reusable runtime assets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import PreparedScene

__all__ = ["normalize_scene_assets"]

_POLICY = "action_engine_glb_geometry_v2"


def normalize_scene_assets(
    scene: PreparedScene,
    output_dir: str | Path,
) -> PreparedScene:
    """Return a scene whose valid GLB meshes have flattened runtime geometry.

    Source files are never modified. Cache names derive from source bytes,
    object scale, and the normalization policy, so repeated generation reuses
    identical assets.
    """
    sections = {
        "background": [deepcopy(value) for value in scene.background],
        "rigid_object": [deepcopy(value) for value in scene.rigid_objects],
        "articulation": [deepcopy(value) for value in scene.articulations],
    }
    cache_dir = Path(output_dir).expanduser().resolve() / "mesh_assets" / "normalized"
    reports: list[dict[str, Any]] = []
    hashes = dict(scene.asset_hashes)
    normalized_by_uid: dict[str, dict[str, Any]] = {}
    for section in ("background", "rigid_object"):
        for config in sections[section]:
            report = _normalize_object(config, cache_dir)
            if report is not None:
                reports.append(report)
                hashes[str(config["uid"])] = str(report["runtime_sha256"])
            normalized_by_uid[str(config["uid"])] = config

    planner = [deepcopy(value) for value in scene.planner_objects]
    for item in planner:
        runtime = normalized_by_uid.get(str(item["runtime_uid"]))
        if runtime is None:
            continue
        item["shape"] = deepcopy(runtime.get("shape", {}))
        item["body_scale"] = list(runtime.get("body_scale", [1.0, 1.0, 1.0]))
    return replace(
        scene,
        planner_objects=tuple(planner),
        background=tuple(sections["background"]),
        rigid_objects=tuple(sections["rigid_object"]),
        articulations=tuple(sections["articulation"]),
        asset_hashes=hashes,
        asset_provenance=tuple(reports),
    )


def _normalize_object(
    config: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any] | None:
    shape = config.get("shape")
    if not isinstance(shape, dict) or not shape.get("fpath"):
        return None
    source = Path(str(shape["fpath"])).expanduser().resolve()
    if source.suffix.lower() not in {".glb", ".gltf"}:
        return None
    source_hash = _file_hash(source)
    scale = [float(value) for value in config.get("body_scale", [1.0, 1.0, 1.0])]
    key = hashlib.sha256(
        json.dumps(
            {"source": source_hash, "scale": scale, "policy": _POLICY},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    destination = cache_dir / f"{source.stem[:32]}_{key[:16]}.glb"
    status = "reused" if destination.is_file() else "generated"
    if status == "generated":
        try:
            _bake_glb(source, destination, scale)
        except Exception as exc:
            return {
                "uid": str(config.get("uid", "")),
                "source_path": source.as_posix(),
                "source_sha256": source_hash,
                "runtime_path": source.as_posix(),
                "runtime_sha256": source_hash,
                "body_scale": scale,
                "status": "preserved_invalid_source",
                "error": f"{type(exc).__name__}: {exc}",
                "policy_version": _POLICY,
            }
    shape["fpath"] = destination.as_posix()
    config["body_scale"] = [1.0, 1.0, 1.0]
    return {
        "uid": str(config.get("uid", "")),
        "source_path": source.as_posix(),
        "source_sha256": source_hash,
        "runtime_path": destination.as_posix(),
        "runtime_sha256": _file_hash(destination),
        "body_scale": scale,
        "status": status,
        "policy_version": _POLICY,
    }


def _bake_glb(source: Path, destination: Path, sim_scale: list[float]) -> None:
    import trimesh

    source_scene = trimesh.load(source.as_posix(), force="scene")
    baked = trimesh.Scene()
    scale = np.diag([sim_scale[0], sim_scale[2], sim_scale[1], 1.0])
    for node_name in source_scene.graph.nodes_geometry:
        node_transform, geometry_name = source_scene.graph.get(node_name)
        mesh = source_scene.geometry[geometry_name].copy()
        mesh.apply_transform(scale @ node_transform)
        baked.add_geometry(
            mesh,
            node_name=str(node_name),
            geom_name=f"geometry_{len(baked.geometry)}",
        )
    if not baked.geometry:
        raise ValueError(f"GLB contains no mesh geometry: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    baked.export(destination.as_posix(), file_type="glb")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
