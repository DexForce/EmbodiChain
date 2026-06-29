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

from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.io import (
    relative_path,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager.scene_geometry import (
    _compose_json_matrices,
    _compose_simready_to_aligned_matrix,
    _decompose_transform_matrix,
)
from embodichain.gen_sim.prompt2scene.utils.io import write_json

__all__ = ["_write_multi_object_layout_manifests"]


def _write_multi_object_layout_manifests(
    *,
    glb_gen_dir: Path,
    output_root: Path,
    table: dict[str, Any] | None,
    objects: list[dict[str, Any]],
    alignment: dict[str, Any] | None,
) -> dict[str, str]:
    simready_to_aligned_path = glb_gen_dir / "simready_to_aligned_manifest.json"

    write_json(
        simready_to_aligned_path,
        _simready_to_aligned_manifest(
            table=table,
            items=objects,
            alignment=alignment,
            output_root=output_root,
        ),
    )
    return {
        "simready_to_aligned_manifest_path": relative_path(
            str(simready_to_aligned_path),
            output_root,
        ),
    }


def _simready_to_aligned_manifest(
    *,
    table: dict[str, Any] | None,
    items: list[dict[str, Any]],
    alignment: dict[str, Any] | None,
    output_root: Path,
) -> dict[str, Any]:
    alignment = alignment or {}
    alignment_matrix = alignment.get("alignment_matrix", [])
    glb_output_axis_transform = alignment.get("glb_output_axis_transform", [])
    object_alignment_matrices = alignment.get("object_alignment_matrices", {})
    aligned_by_id = _aligned_outputs_by_id(alignment)
    return {
        "note": (
            "Aligned GLBs are generated from raw_downloads plus SAM3D layout "
            "matrices in memory; simready paths are recorded here as the "
            "simulation-ready counterpart for each raw GLB."
        ),
        "alignment_status": alignment.get("status", ""),
        "alignment_reason": alignment.get("reason", ""),
        "selected_up_down_variant": alignment.get("selected_up_down_variant", ""),
        "applied_up_down_flip": alignment.get("applied_up_down_flip", False),
        "alignment_matrix": alignment_matrix,
        "global_metric_scale": alignment.get("global_metric_scale"),
        "final_clutter_2d_aabb_cm": alignment.get("final_clutter_2d_aabb_cm"),
        "glb_output_axis_transform": glb_output_axis_transform,
        "table": (
            _simready_manifest_table_item(table, output_root=output_root)
            if table is not None
            else None
        ),
        "items": [
            _simready_to_aligned_manifest_item(
                item,
                aligned_by_id=aligned_by_id,
                alignment_matrix=alignment_matrix,
                object_alignment_matrices=object_alignment_matrices,
                glb_output_axis_transform=glb_output_axis_transform,
                output_root=output_root,
            )
            for item in items
        ],
    }


def _aligned_outputs_by_id(alignment: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for item in alignment.get("objects", []) or []:
        if isinstance(item, dict) and item.get("id"):
            outputs[str(item["id"])] = str(item.get("aligned_geometry_path", ""))
    return outputs


def _simready_manifest_table_item(
    item: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "kind": item.get("kind", "table"),
        "status": item.get("status", ""),
        "simready_geometry_path": (
            relative_path(
                str(
                    _resolve_generated_path(
                        item.get("simready_geometry_path"), output_root
                    )
                ),
                output_root,
            )
            if item.get("simready_geometry_path")
            else ""
        ),
        "support_reference_geometry_path": (
            relative_path(
                str(
                    _resolve_generated_path(
                        item.get("support_reference_geometry_path"),
                        output_root,
                    )
                ),
                output_root,
            )
            if item.get("support_reference_geometry_path")
            else ""
        ),
        "table_asset_source": item.get("table_asset_source", ""),
        "support_normal_source": item.get("support_normal_source", ""),
        "is_complete_visible_table": item.get("is_complete_visible_table", False),
        "complete_table_description": item.get("complete_table_description", ""),
    }


def _simready_to_aligned_manifest_item(
    item: dict[str, Any],
    *,
    aligned_by_id: dict[str, str],
    alignment_matrix: Any,
    object_alignment_matrices: Any,
    glb_output_axis_transform: Any,
    output_root: Path,
) -> dict[str, Any]:
    item_id = str(item.get("id", ""))
    sam3d_transform = item.get("transform_matrix", [])
    item_alignment_matrix = alignment_matrix
    if isinstance(object_alignment_matrices, dict):
        item_alignment_matrix = object_alignment_matrices.get(
            item_id,
            alignment_matrix,
        )
    raw_to_aligned_matrix = _compose_json_matrices(
        glb_output_axis_transform,
        item_alignment_matrix,
        sam3d_transform,
    )
    simready_to_aligned_matrix = _compose_simready_to_aligned_matrix(
        raw_to_aligned_matrix=raw_to_aligned_matrix,
        raw_to_simready_matrix=item.get("raw_to_simready_glb_matrix", []),
    )
    decomposed = _decompose_transform_matrix(simready_to_aligned_matrix)
    return {
        "id": item_id,
        "name": item.get("name", ""),
        "kind": item.get("kind", ""),
        "simready_geometry_path": item.get("simready_geometry_path", ""),
        "aligned_geometry_path": aligned_by_id.get(item_id, ""),
        "metric_scale": _trim_metric_scale(item.get("metric_scale")),
        "simready_to_aligned_matrix": simready_to_aligned_matrix,
        "translation": decomposed["translation"],
        "rotation_matrix": decomposed["rotation_matrix"],
        "scale": decomposed["scale"],
    }


def _trim_metric_scale(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    metric_scale = dict(value)
    for key in ["result_path", "raw_model_output_path"]:
        metric_scale.pop(key, None)
    return metric_scale


def _resolve_generated_path(value: Any, output_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()
