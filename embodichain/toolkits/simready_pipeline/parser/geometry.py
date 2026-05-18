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

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from embodichain.toolkits.simready_pipeline.parser.base import AssetParser
from embodichain.toolkits.simready_pipeline.core.asset import Asset
from embodichain.toolkits.simready_pipeline.utils.geometry_utils import process_obj


def _load_geometry_cleanup_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "gen_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("geometry_cleanup", {})


GEOMETRY_CLEANUP_CONFIG = _load_geometry_cleanup_config()


class GeometryParser(AssetParser):
    name = "geometry"

    def __init__(self):
        super().__init__()

    def _topology_stats(self, mesh: trimesh.Trimesh) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "is_empty": bool(mesh.is_empty),
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "is_volume": bool(mesh.is_volume),
            "euler_number": None,
            "body_count": int(mesh.body_count) if hasattr(mesh, "body_count") else None,
            "face_component_count": None,
            "broken_face_count": None,
            "boundary_edge_count": None,
            "manifold_edge_count": None,
            "nonmanifold_edge_count": None,
            "edge_incidence_hist": None,
        }

        if mesh.is_empty:
            return stats

        try:
            tmp = mesh.copy(include_visual=False)
            tmp.remove_unreferenced_vertices()
            stats["euler_number"] = int(tmp.euler_number)
        except Exception:
            try:
                stats["euler_number"] = int(mesh.euler_number)
            except Exception:
                stats["euler_number"] = None

        stats["face_component_count"] = None

        try:
            broken = trimesh.repair.broken_faces(mesh)
            stats["broken_face_count"] = int(len(broken))
        except Exception:
            stats["broken_face_count"] = None

        try:
            edges = mesh.edges_unique
            if len(edges) > 0:
                counts = np.bincount(mesh.edges_unique_inverse)
                stats["boundary_edge_count"] = int(np.sum(counts == 1))
                stats["manifold_edge_count"] = int(np.sum(counts == 2))
                stats["nonmanifold_edge_count"] = int(np.sum(counts > 2))
        except Exception:
            pass

        return stats

    def parse(self, asset: Asset, asset_root: Path) -> None:
        asset.parsed.setdefault("geometry", {})

        if asset.asset_data.get("type") != "mesh":
            asset.parsed["geometry"] = {"asset dont have a mesh": "skipped"}
            return

        mesh_path = asset_root / asset.asset_data.get("path")
        process_obj(
            input_path=str(mesh_path),
            output_path=str(mesh_path),
            ratio=GEOMETRY_CLEANUP_CONFIG.get("ratio", 0.5),
            weld_distance=GEOMETRY_CLEANUP_CONFIG.get("weld_distance", 0.0001),
            merge_dist=GEOMETRY_CLEANUP_CONFIG.get("merge_dist", 1e-5),
            remove_non_manifold=GEOMETRY_CLEANUP_CONFIG.get(
                "remove_non_manifold", True
            ),
            triangulate=GEOMETRY_CLEANUP_CONFIG.get("triangulate", False),
            fill_hole_sides=GEOMETRY_CLEANUP_CONFIG.get("fill_hole_sides", 8),
        )

        try:

            mesh = trimesh.load(
                mesh_path, force="mesh", skip_materials=True, process=False
            )

            geom_info = {
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
                "extents": mesh.extents.tolist() if mesh.extents is not None else None,
                "area": float(mesh.area),
            }

            geom_info.update(self._topology_stats(mesh))
            asset.parsed["geometry"] = geom_info

        except Exception as e:
            print(f"[GEOMETRY PARSER FAILED] {mesh_path}: {str(e)}")
            asset.parsed["geometry"] = {"error": str(e)}
