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

import shutil
import traceback
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_generation_manager import (
    GeometryGenerationManager,
    RgbaImageToGeometryRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_generation_manager import (
    ImageGenerationManager,
    TextToAssetImageRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_segmentation_manager import (
    AssetImageToRgbaRequest,
    ImageSegmentationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager import (
    MakeAssetSimreadyRequest,
    MakeTableSimreadyRequest,
    SimreadyManager,
)
from embodichain.gen_sim.prompt2scene.utils.log import log_info, log_warning

__all__ = [
    "generate_text_object_asset",
    "generate_text_object_assets",
    "generate_text_table_asset",
]


def generate_text_object_asset(
    *,
    object_spec: dict[str, Any],
    image_gen_dir: Path,
    glb_gen_dir: Path,
    debug_dir: Path,
) -> dict[str, Any]:
    """Generate one object asset from a text-origin object spec."""
    object_id = str(object_spec.get("id", "object"))
    object_name = str(object_spec.get("name", ""))
    description = str(object_spec.get("description", ""))
    class_candidates = [
        str(candidate).replace("_", " ")
        for candidate in object_spec.get("class_candidate", [])
        if isinstance(candidate, str) and candidate.strip()
    ]
    status = "ok"
    image_path = ""
    raw_geometry_path = ""
    mesh_path = ""
    raw_to_simready_matrix: list[list[float]] = []

    debug_subdir = debug_dir / object_id
    debug_subdir.mkdir(parents=True, exist_ok=True)
    log_info(f"text object generation started id={object_id} name={object_name}")

    image_manager = ImageGenerationManager()
    segmentation_manager = ImageSegmentationManager()
    geometry_manager = GeometryGenerationManager()
    simready_manager = SimreadyManager()

    try:
        image_prompt = f"{object_name}, {description}".strip(", ")
        raw_image_path = str(
            image_manager.generate_asset_image_from_text(
                TextToAssetImageRequest(
                    prompt=image_prompt,
                    output_path=debug_subdir / f"{object_id}.png",
                )
            )
        )

        rgba_prompts: list[str] = []
        if description.strip():
            rgba_prompts.append(description.strip())
        for candidate in class_candidates:
            candidate_prompt = f"The entire {candidate} on the center of the image"
            if candidate_prompt not in rgba_prompts:
                rgba_prompts.append(candidate_prompt)
        if not rgba_prompts:
            rgba_prompts.append(
                f"the entire single isolated object {object_name}"
                if object_name
                else "the entire single isolated object"
            )

        rgba_path = ""
        last_rgba_error: Exception | None = None
        for prompt in rgba_prompts:
            try:
                rgba_path = str(
                    segmentation_manager.convert_asset_image_to_rgba(
                        AssetImageToRgbaRequest(
                            image_path=Path(raw_image_path),
                            prompt=prompt,
                            output_path=image_gen_dir / f"{object_id}.png",
                        )
                    )
                )
                break
            except Exception as exc:
                last_rgba_error = exc
                log_warning(
                    "text object segmentation prompt failed "
                    f"id={object_id} prompt={prompt!r} error={exc}"
                )
        if not rgba_path:
            raise last_rgba_error or RuntimeError(
                f"No RGBA prompt succeeded for {object_id}"
            )

        raw_glb_path = str(
            geometry_manager.convert_rgba_image_to_geometry(
                RgbaImageToGeometryRequest(
                    image_path=Path(rgba_path),
                    output_path=debug_subdir / f"{object_id}_raw.glb",
                )
            )
        )
        raw_geometry_dir = glb_gen_dir / "raw_downloads"
        raw_geometry_dir.mkdir(parents=True, exist_ok=True)
        object_raw_path = raw_geometry_dir / f"{object_id}_raw.glb"
        shutil.copy2(raw_glb_path, object_raw_path)
        raw_geometry_path = str(object_raw_path)

        simready_result = simready_manager.make_asset_simready(
            MakeAssetSimreadyRequest(
                input_path=Path(raw_glb_path),
                output_path=glb_gen_dir
                / "text_objects_simready"
                / f"{object_id}_simready.glb",
            )
        )
        mesh_path = str(simready_result.output_path)
        raw_to_simready_matrix = simready_result.transform_matrix

        image_path = rgba_path
        log_info(f"text object generation completed id={object_id} mesh={mesh_path}")
    except Exception as exc:
        status = f"failed: {traceback.format_exc()}"
        log_warning(f"text object generation failed id={object_id} error={exc}")

    return {
        "id": object_id,
        "name": object_name,
        "description": description,
        "status": status,
        "image_path": image_path,
        "raw_geometry_path": raw_geometry_path,
        "mesh_path": mesh_path,
        "simready_geometry_path": mesh_path,
        "raw_to_simready_glb_matrix": raw_to_simready_matrix,
        "metric_scale": None,
    }


def generate_text_object_assets(
    *,
    object_specs: list[dict[str, Any]],
    image_gen_dir: Path,
    glb_gen_dir: Path,
    debug_dir: Path,
) -> list[dict[str, Any]]:
    """Generate all object assets for a text-origin unified scene."""
    log_info(f"text object batch generation started count={len(object_specs)}")
    results = [
        generate_text_object_asset(
            object_spec=object_spec,
            image_gen_dir=image_gen_dir,
            glb_gen_dir=glb_gen_dir,
            debug_dir=debug_dir,
        )
        for object_spec in object_specs
    ]
    succeeded = sum(result.get("status") == "ok" for result in results)
    log_info(
        f"text object batch generation completed "
        f"succeeded={succeeded} failed={len(results) - succeeded}"
    )
    return results


def generate_text_table_asset(
    *,
    table_spec: dict[str, Any],
    image_gen_dir: Path,
    glb_gen_dir: Path,
    debug_dir: Path,
) -> dict[str, Any]:
    """Generate the table asset for a text-origin unified scene."""
    table_id = str(table_spec.get("id", "table"))
    description = str(
        table_spec.get("complete_table_description")
        or table_spec.get("description", "")
    ).strip()
    status = "ok"
    image_path = ""
    raw_geometry_path = ""
    generated_table_raw_geometry_path = ""
    mesh_path = ""

    debug_subdir = debug_dir / table_id
    debug_subdir.mkdir(parents=True, exist_ok=True)
    log_info(f"text table generation started id={table_id}")

    image_manager = ImageGenerationManager()
    segmentation_manager = ImageSegmentationManager()
    geometry_manager = GeometryGenerationManager()
    simready_manager = SimreadyManager()

    try:
        raw_image_path = str(
            image_manager.generate_asset_image_from_text(
                TextToAssetImageRequest(
                    prompt=description,
                    output_path=debug_subdir / f"{table_id}.png",
                )
            )
        )
        rgba_path = str(
            segmentation_manager.convert_asset_image_to_rgba(
                AssetImageToRgbaRequest(
                    image_path=Path(raw_image_path),
                    prompt=description if description.strip() else "whole table",
                    output_path=image_gen_dir / f"{table_id}.png",
                )
            )
        )
        raw_glb_path = str(
            geometry_manager.convert_rgba_image_to_geometry(
                RgbaImageToGeometryRequest(
                    image_path=Path(rgba_path),
                    output_path=debug_subdir / f"{table_id}_raw.glb",
                )
            )
        )
        generated_table_raw_geometry_path = raw_glb_path
        raw_geometry_dir = glb_gen_dir / "raw_downloads"
        raw_geometry_dir.mkdir(parents=True, exist_ok=True)
        table_raw_path = raw_geometry_dir / "table_raw.glb"
        shutil.copy2(raw_glb_path, table_raw_path)
        raw_geometry_path = str(table_raw_path)
        mesh_path = str(
            simready_manager.make_table_simready(
                MakeTableSimreadyRequest(
                    input_path=Path(raw_geometry_path),
                    output_path=glb_gen_dir
                    / "text_objects_simready"
                    / f"{table_id}_simready.glb",
                )
            ).output_path
        )
        image_path = rgba_path
        log_info(f"text table generation completed id={table_id} mesh={mesh_path}")
    except Exception as exc:
        status = f"failed: {traceback.format_exc()}"
        log_warning(f"text table generation failed id={table_id} error={exc}")

    return {
        "id": table_id,
        "name": str(table_spec.get("name", "table")),
        "description": str(table_spec.get("description", "")),
        "complete_table_description": description,
        "is_complete_visible_table": bool(
            table_spec.get("is_complete_visible_table", False)
        ),
        "status": status,
        "image_path": image_path,
        "raw_geometry_path": raw_geometry_path,
        "generated_table_raw_geometry_path": generated_table_raw_geometry_path,
        "support_reference_geometry_path": "",
        "table_asset_source": "description_generated",
        "support_normal_source": "",
        "mesh_path": mesh_path,
        "simready_geometry_path": mesh_path,
    }
