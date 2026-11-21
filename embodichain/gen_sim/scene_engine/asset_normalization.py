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

"""Prepare content-addressed source mesh assets independently of bundle publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from embodichain.compute.geometry.shading import has_inconsistent_shading_normals

__all__ = ["NormalizedSceneAssets", "normalize_source_scene_assets"]

_POLICY = "action_engine_glb_geometry_v3"


@dataclass(frozen=True, slots=True)
class NormalizedSceneAssets:
    """Owned physical declarations and original source/runtime provenance."""

    background: tuple[dict[str, Any], ...]
    rigid_objects: tuple[dict[str, Any], ...]
    articulations: tuple[dict[str, Any], ...]
    asset_hashes: dict[str, str]
    provenance: tuple[dict[str, Any], ...]


def normalize_source_scene_assets(
    *,
    background: Sequence[Mapping[str, Any]],
    rigid_objects: Sequence[Mapping[str, Any]],
    articulations: Sequence[Mapping[str, Any]],
    cache_root: str | Path,
) -> NormalizedSceneAssets:
    """Bake source GLB transforms/scale and split only inconsistent normals.

    Args:
        background: Ordered source background declarations.
        rigid_objects: Ordered source rigid declarations.
        articulations: Source articulation declarations, passed through unchanged.
        cache_root: Separate derived-asset preparation directory. This function
            does not write a Task Program bundle or modify authored source files.

    Returns:
        Independent declarations and content-addressed assets using the original
        v3 policy. Runtime scale is one after baking. Invalid source geometry is
        retained with its original explicit diagnostic rather than substituted.
    """
    sections = {
        "background": [deepcopy(dict(value)) for value in background],
        "rigid_object": [deepcopy(dict(value)) for value in rigid_objects],
        "articulation": [deepcopy(dict(value)) for value in articulations],
    }
    cache_dir = Path(cache_root).expanduser().resolve() / "mesh_assets" / "normalized"
    reports = []
    hashes = {}
    for section in ("background", "rigid_object"):
        for config in sections[section]:
            report = _normalize_object(config, cache_dir)
            if report is not None:
                reports.append(report)
                hashes[str(config["uid"])] = str(report["runtime_sha256"])
    return NormalizedSceneAssets(
        tuple(sections["background"]),
        tuple(sections["rigid_object"]),
        tuple(sections["articulation"]),
        hashes,
        tuple(reports),
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
        if has_inconsistent_shading_normals(mesh.vertices, mesh.faces):
            # DexSim rejects ray-traced meshes when a smoothed vertex normal
            # points away from an adjacent triangle. Split only affected
            # meshes; preserving safe topology keeps grasp sampling stable.
            mesh.unmerge_vertices()
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
