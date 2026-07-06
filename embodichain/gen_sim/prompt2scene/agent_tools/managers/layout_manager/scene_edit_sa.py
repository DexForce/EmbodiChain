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

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.sa_node3_5 import (
    run_node_3_5,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.sa_state import (
    Tempo_SceneState,
)
from embodichain.gen_sim.prompt2scene.utils.io import write_json

__all__ = ["optimize_scene_edit_layout_with_sa_node3_5"]


def _glb_scale_to_sim(scale: list[float]) -> list[float]:
    if len(scale) != 3:
        raise ValueError("GLB scale must be a 3-vector.")
    return [
        float(scale[0]),
        float(scale[2]),
        float(scale[1]),
    ]


def _grid_to_region(grid_name: str) -> str:
    grid = str(grid_name or "").strip()
    mapping = {
        "left_front": "front_left_area",
        "center_front": "front_area",
        "right_front": "front_right_area",
        "left_center": "left_area",
        "center": "center_area",
        "right_center": "right_area",
        "left_back": "back_left_area",
        "center_back": "back_area",
        "right_back": "back_right_area",
        "front": "front_area",
        "back": "back_area",
        "left": "left_area",
        "right": "right_area",
    }
    return mapping.get(grid, "unspecified")


def _support_region_origin_xy(
    support_region: dict[str, Any],
) -> tuple[float, float]:
    aabb_xy = support_region.get("aabb_xy")
    if (
        isinstance(aabb_xy, list)
        and len(aabb_xy) == 2
        and all(isinstance(item, list) and len(item) == 2 for item in aabb_xy)
    ):
        return float(aabb_xy[0][0]), float(aabb_xy[0][1])
    center_xy = support_region.get("center_xy")
    size_xy = support_region.get("size_xy")
    if (
        isinstance(center_xy, list)
        and len(center_xy) == 2
        and isinstance(size_xy, list)
        and len(size_xy) == 2
    ):
        return (
            float(center_xy[0]) - 0.5 * float(size_xy[0]),
            float(center_xy[1]) - 0.5 * float(size_xy[1]),
        )
    return 0.0, 0.0


def _resolve_asset_path(value: Any, *, output_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _generated_body_scale(generated_asset: dict[str, Any]) -> list[float]:
    metric_scale = generated_asset.get("metric_scale")
    scale_factor = 1.0
    if isinstance(metric_scale, dict):
        try:
            scale_factor = float(metric_scale.get("scale_factor", 1.0))
        except (TypeError, ValueError):
            scale_factor = 1.0
    if scale_factor <= 0.0:
        scale_factor = 1.0
    body_scale = _glb_scale_to_sim([scale_factor, scale_factor, scale_factor])
    return [float(value) for value in body_scale]


def _build_objects_config_scaled(
    *,
    output_root: Path,
    runtime_root: Path,
    layout_items: dict[str, dict[str, Any]],
    rigid_by_id: dict[str, dict[str, Any]],
    generated_asset_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rigid_objects: list[dict[str, Any]] = []
    gym_export_root = output_root / "gym_export"
    for object_id in sorted(layout_items):
        if object_id in rigid_by_id:
            rigid = copy.deepcopy(rigid_by_id[object_id])
            shape = rigid.get("shape")
            if not isinstance(shape, dict):
                raise ValueError(f"Existing rigid_object has no shape for {object_id}")
            mesh_path = _resolve_asset_path(
                gym_export_root / str(shape.get("fpath", "")),
                output_root=output_root,
            )
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Mesh not found for {object_id}: {mesh_path}")
            shape["fpath"] = os.path.relpath(mesh_path, runtime_root)
            rigid["shape"] = shape
            rigid_objects.append(rigid)
            continue

        generated_asset = generated_asset_by_id.get(object_id)
        if generated_asset is None:
            raise ValueError(f"Missing generated asset for {object_id}")
        mesh_path = _resolve_asset_path(
            generated_asset.get("simready_geometry_path")
            or generated_asset.get("mesh_path"),
            output_root=output_root,
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Generated mesh not found for {object_id}: {mesh_path}"
            )
        rigid_objects.append(
            {
                "uid": object_id,
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": os.path.relpath(mesh_path, runtime_root),
                    "compute_uv": False,
                },
                "body_scale": _generated_body_scale(generated_asset),
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
            }
        )
    return {"rigid_object": rigid_objects}


def optimize_scene_edit_layout_with_sa_node3_5(
    *,
    output_root: Path,
    support_region: dict[str, Any],
    layout_items: dict[str, dict[str, Any]],
    updated_relations: list[dict[str, Any]],
    updated_grids: dict[str, str],
    fixed_object_ids: list[str],
    rigid_by_id: dict[str, dict[str, Any]],
    generated_asset_by_id: dict[str, dict[str, Any]],
    runtime_root: Path,
) -> dict[str, Any]:
    size_xy = support_region.get("size_xy")
    if not (isinstance(size_xy, list) and len(size_xy) == 2):
        raise ValueError(
            "support_region.size_xy is required for SA scene-edit optimization."
        )
    origin_x, origin_y = _support_region_origin_xy(support_region)

    runtime_root.mkdir(parents=True, exist_ok=True)
    objects_cfg = _build_objects_config_scaled(
        output_root=output_root,
        runtime_root=runtime_root,
        layout_items=layout_items,
        rigid_by_id=rigid_by_id,
        generated_asset_by_id=generated_asset_by_id,
    )
    write_json(runtime_root / "objects_config_scaled.json", objects_cfg)

    state = Tempo_SceneState()
    state.table_size = (float(size_xy[0]) * 100.0, float(size_xy[1]) * 100.0)
    state.raw_object_dict = {}
    state.init_layout = {}

    for object_id, item in sorted(layout_items.items()):
        center_xy = item.get("center_xy")
        if not (isinstance(center_xy, list) and len(center_xy) == 2):
            continue
        relation: dict[str, list[Any]] = {
            "left_of": [],
            "right_of": [],
            "front_of": [],
            "back_of": [],
        }
        state.raw_object_dict[object_id] = {
            "region": _grid_to_region(updated_grids.get(object_id, "")),
            "contact": {},
            "relation": relation,
        }
        if object_id in fixed_object_ids:
            state.raw_object_dict[object_id]["coordinate"] = [
                (float(center_xy[0]) - origin_x) * 100.0,
                (float(center_xy[1]) - origin_y) * 100.0,
            ]

        init_rot = [0.0, 0.0, 0.0]
        rigid = rigid_by_id.get(object_id)
        if isinstance(rigid, dict):
            value = rigid.get("init_rot")
            if isinstance(value, list) and len(value) >= 3:
                init_rot = value
        rot_deg = 0.0
        try:
            rot_deg = float(init_rot[2])
        except (TypeError, ValueError, IndexError):
            rot_deg = 0.0

        state.init_layout[object_id] = {
            "init_coordinate": [
                (float(center_xy[0]) - origin_x) * 100.0,
                (float(center_xy[1]) - origin_y) * 100.0,
                rot_deg,
            ]
        }

    for rel in updated_relations:
        subject = str(rel.get("subject", "")).strip()
        relation_name = str(rel.get("relation", "")).strip()
        object_id = str(rel.get("object", "")).strip()
        if (
            subject not in state.raw_object_dict
            or object_id not in state.raw_object_dict
        ):
            continue
        relation = state.raw_object_dict[subject]["relation"]
        if relation_name == "left_of":
            relation["left_of"].append(object_id)
        elif relation_name == "right_of":
            relation["right_of"].append(object_id)
        elif relation_name == "front_of":
            relation["front_of"].append(object_id)
        elif relation_name in {"back_of", "behind"}:
            relation["back_of"].append(object_id)

    state = run_node_3_5(state, runtime_root)
    centers: dict[str, list[float]] = {}
    for object_id, item in (state.optimized_layout or {}).items():
        center_2d = item.get("center_2d")
        if not (isinstance(center_2d, list) and len(center_2d) == 2):
            continue
        centers[object_id] = [
            float(center_2d[0]) + origin_x,
            float(center_2d[1]) + origin_y,
        ]

    return {
        "status": "ok",
        "centers": centers,
        "metadata": {
            "messages": list(getattr(state, "messages", []) or []),
            "optimization_model": getattr(state, "optimization_model", {}),
            "stack_groups": getattr(state, "stack_groups", {}),
        },
    }
