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

import traceback
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.tools.table_clutter_fit import (
    fit_table_to_clutter,
)
from embodichain.gen_sim.prompt2scene.utils.io import relative_path, write_json
from embodichain.gen_sim.prompt2scene.utils.log import log_info, log_warning

__all__ = ["fit_image_scene_table", "fit_text_scene_table"]


def _table_fit_output_paths(
    *,
    output_dir: Path,
    object_ids: list[str],
) -> tuple[Path, dict[str, Path]]:
    return (
        output_dir / "table_fit_to_clutter.glb",
        {
            object_id: output_dir / f"{object_id}_on_table.glb"
            for object_id in object_ids
        },
    )


def fit_text_scene_table(
    *,
    table_result: dict[str, Any],
    clutter_layout_result: dict[str, Any],
    output_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit the text-scene table and convert failures to result data."""
    try:
        object_ids = [
            str(item["id"])
            for item in clutter_layout_result.get("objects", [])
            if isinstance(item, dict) and item.get("id") and item.get("status") == "ok"
        ]
        table_output_path, object_output_paths = _table_fit_output_paths(
            output_dir=output_dir,
            object_ids=object_ids,
        )
        manifest = fit_table_to_clutter(
            table_result=table_result,
            clutter_result=clutter_layout_result,
            output_root=output_root,
            output_dir=output_dir,
            table_output_path=table_output_path,
            object_output_paths=object_output_paths,
            object_coverage_percent=table_result.get("object_coverage_percent"),
        )
        manifest_path = output_dir / "table_fit_to_clutter_manifest.json"
        write_json(manifest_path, manifest)
        result = {
            "status": "ok",
            "manifest_path": relative_path(str(manifest_path), output_root),
        }
        log_info(f"text table fit completed status={result.get('status')}")
        return result
    except Exception as exc:
        log_warning(f"text table fit failed error={exc}")
        return {
            "status": "failed",
            "reason": traceback.format_exc(),
        }


def fit_image_scene_table(
    *,
    layout_result: dict[str, Any],
    fallback_table_result: dict[str, Any] | None,
    output_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit the image-scene table or return a structured skipped result."""
    generated_table = layout_result.get("table") or fallback_table_result
    generated_objects = layout_result.get("objects") or []
    alignment_result = layout_result.get("alignment")
    if (
        generated_table is None
        or not generated_objects
        or not isinstance(alignment_result, dict)
    ):
        return {
            "status": "skipped",
            "reason": "missing_table_objects_or_alignment",
        }

    try:
        object_ids = [
            str(item["id"])
            for item in generated_objects
            if isinstance(item, dict)
            and item.get("id")
            and item.get("aligned_geometry_path")
        ]
        table_output_path, object_output_paths = _table_fit_output_paths(
            output_dir=output_dir,
            object_ids=object_ids,
        )
        clutter_result = {
            "clutter_2d_aabb_cm": alignment_result.get(
                "final_clutter_2d_aabb_cm"
            ),
            "objects": [
                {
                    "id": item["id"],
                    "status": "ok",
                    "laid_out_glb_path": item["aligned_geometry_path"],
                }
                for item in generated_objects
                if item.get("id") and item.get("aligned_geometry_path")
            ],
        }
        manifest = fit_table_to_clutter(
            table_result=generated_table,
            clutter_result=clutter_result,
            output_root=output_root,
            output_dir=output_dir,
            table_output_path=table_output_path,
            object_output_paths=object_output_paths,
            object_coverage_percent=generated_table.get("object_coverage_percent"),
        )
        manifest_path = output_dir / "table_fit_to_clutter_manifest.json"
        write_json(manifest_path, manifest)
        result = {
            "status": "ok",
            "manifest_path": relative_path(str(manifest_path), output_root),
        }
        log_info(f"image table fit completed status={result.get('status')}")
        return result
    except Exception as exc:
        log_warning(f"image table fit failed error={exc}")
        return {
            "status": "failed",
            "reason": traceback.format_exc(),
        }
