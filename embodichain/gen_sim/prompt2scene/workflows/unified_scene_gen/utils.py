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

from embodichain.gen_sim.prompt2scene.utils.io import relative_path

__all__ = ["update_unified_scene"]


def update_unified_scene(
    unified_scene: dict[str, Any],
    table_result: dict[str, Any],
    object_results: list[dict[str, Any]],
    output_root: Path,
) -> None:
    """Write generated asset references back into a unified-scene payload."""
    table = unified_scene.setdefault("table", {})
    metadata_keys = (
        "table_asset_source",
        "support_normal_source",
        "is_complete_visible_table",
        "complete_table_description",
    )
    path_keys = (
        "image_path",
        "raw_geometry_path",
        "support_reference_geometry_path",
        "generated_table_raw_geometry_path",
        "transformed_geometry_path",
        "simready_geometry_path",
        "aligned_geometry_path",
        "mesh_path",
    )
    for key in metadata_keys:
        if key in table_result:
            table[key] = table_result[key]
    for key in path_keys:
        if table_result.get(key):
            table[key] = relative_path(table_result[key], output_root)

    objects_by_id = {
        str(item.get("id", "")): item
        for item in unified_scene.setdefault("objects", [])
        if isinstance(item, dict)
    }
    for result in object_results:
        target = objects_by_id.get(str(result.get("id", "")))
        if target is None:
            continue
        for key in ("image_path", "mesh_path", "aligned_geometry_path"):
            if result.get(key):
                target[key] = relative_path(result[key], output_root)
        metric_scale = result.get("metric_scale")
        if isinstance(metric_scale, dict):
            target["metric_scale"] = {
                key: value
                for key, value in metric_scale.items()
                if key not in {"result_path", "raw_model_output_path"}
            }
