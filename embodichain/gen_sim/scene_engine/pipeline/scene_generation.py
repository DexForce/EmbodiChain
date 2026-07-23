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

import json
from pathlib import Path
import shutil

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    export_baked_layout_object_glbs,
    quaternion_wxyz_to_euler_xyz_degrees,
    simready_object_glb,
)

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def generate_geometries_and_coarse_layout(
    image_path: str | Path,
    output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
    geometry_generation_client: GeometryGenerationClient,
) -> Scene:

    resolved_image_path = _validate_image_path(image_path)
    # Create stage output directory.
    stage_output_root = Path(output_root).expanduser().resolve() / "scene_generation"
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)
    # Create debug folder and the sim-ready geometry folder.
    debug_output_root = stage_output_root / "debug" # Keeps the other files for debugging.
    intermediate_output_root = stage_output_root / "intermediate" # Save some intermediate results.
    coarse_geometry_output_root = stage_output_root / "coarse_geometry" # Keeps the coarse geometries.
    simready_geometry_output_root = stage_output_root / "simready_geometry" # Keeps the final-used geometries.
    debug_output_root.mkdir()
    intermediate_output_root.mkdir()
    coarse_geometry_output_root.mkdir()
    simready_geometry_output_root.mkdir()

    # Coarse geometry generation and coarse layout generation.
    _generate_coarse_results_from_masks(
        image_path=resolved_image_path, 
        debug_output_root=debug_output_root,
        coarse_geometry_output_root=coarse_geometry_output_root,
        scene=scene, # Use the masks which are kept in the scene data structure.
        vlm_client=vlm_client,
        geometry_generation_client=geometry_generation_client,
    )

    # Geometries refinement and layout refinement.
    _refine_geometries_and_layout(
        image_path=resolved_image_path,
        debug_output_root=debug_output_root,
        intermediate_output_root=intermediate_output_root,
        coarse_geometry_output_root=coarse_geometry_output_root,
        simready_geometry_output_root=simready_geometry_output_root,
        scene=scene,
        vlm_client=vlm_client,
    )

    # Write the Updated scene JSON for debugging.
    (stage_output_root / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return scene

def _generate_coarse_results_from_masks(
    image_path: str | Path,
    debug_output_root: str | Path,
    coarse_geometry_output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
    geometry_generation_client: GeometryGenerationClient,
) -> None:

    # Parse whether the scene has each assets' binary masks.
    # The original image has already been validated.
    # The table must exist, for it is the base of the scene.
    if scene.table is None:
        raise ValueError("Scene must contain a table before geometry generation.")

    scene_objects = [scene.table, *scene.assets]
    object_masks: list[tuple[str, Path]] = []
    for scene_object in scene_objects:
        if scene_object.mask_path is None:
            raise ValueError(
                f"Scene object {scene_object.id!r} has no binary mask path."
            )
        mask_path = Path(scene_object.mask_path).expanduser().resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Binary mask for scene object {scene_object.id!r} not found: "
                f"{mask_path}"
            )
        object_masks.append((scene_object.id, mask_path)) # id + mask, for avoiding the download glbs order confusion.

    # Sent the request, wait, then save the intermediate results.
    response_data, response_objects = geometry_generation_client.generate_multiple_objects(
        image_path=image_path,
        object_masks=object_masks,
        output_root=coarse_geometry_output_root, # Keep the coarse geometries
    )
    # Write the response JSON which contains all the layout info the server gave us.
    # Keep original response for getting the sam3d coarse layout matrix.
    (Path(debug_output_root) / "geometry_generation_response.json").write_text(
        json.dumps(response_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Write the coarse layout JSON as one of the results in this step.
    coarse_layout = [
        {
            "id": object_id,
            "rot": quaternion_wxyz_to_euler_xyz_degrees(
                response_object["rotation_quaternion_wxyz"]
            ),
            "pos": response_object["translation"],
            "scale": response_object["scale"],
        }
        for (object_id, _), response_object in zip(object_masks, response_objects)
    ]
    (Path(coarse_geometry_output_root) / "coarse_layout.json").write_text(
        json.dumps(coarse_layout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Nothing to be returned.
    return None

def _refine_geometries_and_layout(
    image_path: str | Path,
    debug_output_root: str | Path,
    intermediate_output_root: str | Path,
    coarse_geometry_output_root: str | Path,
    simready_geometry_output_root: str | Path,
    scene: Scene,
    *,
    vlm_client: OpenAICompatibleVLM,
) -> None:

    # Simready all the assets(includes table).
    # Treat table and assets seperately.
    # Notice that, currently the simready process is only
    # scale + canonicalize the glb (no real-world scale, no physical attributes).

    # Load the coarse layout.
    coarse_layout = _load_layout(Path(coarse_geometry_output_root) / "coarse_layout.json")
    coarse_layout_by_id = {
        layout_object["id"]: layout_object for layout_object in coarse_layout
    }

    # Simready all the assets.
    simready_assets_layout = _simready_assets(
        scene=scene,
        coarse_layout_by_id=coarse_layout_by_id,
        coarse_geometry_output_root=coarse_geometry_output_root,
        simready_geometry_output_root=simready_geometry_output_root,
    )

    # Simready the table.
    simready_table_layout = _simready_table(
        scene=scene,
        coarse_layout_by_id=coarse_layout_by_id,
        coarse_geometry_output_root=coarse_geometry_output_root,
        simready_geometry_output_root=simready_geometry_output_root,
    )
    # Concat then save the table info and the assets info in one JSON file.
    simready_layout = [simready_table_layout, *simready_assets_layout]
    (Path(intermediate_output_root) / "simready_layout.json").write_text(
        json.dumps(simready_layout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    

    # # Test whether the simreadyed assets can still construct the same scene layout
    # # in contrast to the original raw returned value.
    # saved_simready_layout = _load_layout(
    #     Path(intermediate_output_root) / "simready_layout.json"
    # )
    # export_baked_layout_object_glbs(
    #     layout=saved_simready_layout,
    #     geometry_root=simready_geometry_output_root,
    #     output_root=Path(debug_output_root) / "simready_baked_geometries",
    # )

    # saved_coarse_layout = _load_layout(
    #     Path(coarse_geometry_output_root) / "coarse_layout.json"
    # )
    # export_baked_layout_object_glbs(
    #     layout=saved_coarse_layout,
    #     geometry_root=coarse_geometry_output_root,
    #     output_root=Path(debug_output_root) / "coarse_baked_geometries",
    # )

    # Layout refinement will start with the table.
    # _layout_refinement()

    return None

def _simready_assets(
    *,
    scene: Scene,
    coarse_layout_by_id: dict[str, dict[str, object]],
    coarse_geometry_output_root: str | Path,
    simready_geometry_output_root: str | Path,
) -> list[dict[str, object]]:
    # Batch process all the assets in the scene.
    return [
        _simready_asset(
            asset_id=asset.id,
            coarse_layout=coarse_layout_by_id.get(asset.id),
            coarse_geometry_output_root=coarse_geometry_output_root,
            simready_geometry_output_root=simready_geometry_output_root,
        )
        for asset in scene.assets
    ]


def _simready_asset(
    *,
    asset_id: str,
    coarse_layout: dict[str, object] | None,
    coarse_geometry_output_root: str | Path,
    simready_geometry_output_root: str | Path,
) -> dict[str, object]:
    # Hard code some asset like bottle, treat their z-axis carefully.
    # For the table, treat it with the same strategy for now.
    # Add asset-id-specific SimReady processing here before the generic path.
    return _simready_object(
        asset_id=asset_id,
        coarse_layout=coarse_layout,
        coarse_geometry_output_root=coarse_geometry_output_root,
        simready_geometry_output_root=simready_geometry_output_root,
    )


def _simready_object(
    *,
    asset_id: str,
    coarse_layout: dict[str, object] | None,
    coarse_geometry_output_root: str | Path,
    simready_geometry_output_root: str | Path,
) -> dict[str, object]:
    if coarse_layout is None:
        raise ValueError(f"Coarse layout does not contain object {asset_id!r}.")
    simready_mesh, simready_transform = simready_object_glb(
        Path(coarse_geometry_output_root) / f"{asset_id}.glb",
        object_id=asset_id,
        rot=coarse_layout.get("rot"),
        pos=coarse_layout.get("pos"),
        scale=coarse_layout.get("scale"),
    )
    output_path = Path(simready_geometry_output_root) / f"{asset_id}.glb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simready_mesh.export(output_path, file_type="glb")
    if not output_path.is_file():
        raise FileNotFoundError(
            f"SimReady object geometry was not written: {output_path}"
        )
    return {"id": asset_id, **simready_transform}


def _simready_table(
    *,
    scene: Scene,
    coarse_layout_by_id: dict[str, dict[str, object]],
    coarse_geometry_output_root: str | Path,
    simready_geometry_output_root: str | Path,
) -> dict[str, object]:
    # There must be a table in one scene.
    if scene.table is None:
        raise ValueError("Cannot SimReady a scene without a table.")

    # Using the same strategy as the normal assets first.
    return _simready_object(
        asset_id=scene.table.id,
        coarse_layout=coarse_layout_by_id.get(scene.table.id),
        coarse_geometry_output_root=coarse_geometry_output_root,
        simready_geometry_output_root=simready_geometry_output_root,
    )


def _load_layout(layout_path: str | Path) -> list[dict[str, object]]:
    # Load and check the coarse layout JSON file.
    resolved_layout_path = Path(layout_path).expanduser().resolve()
    if not resolved_layout_path.is_file():
        raise FileNotFoundError(f"Layout not found: {resolved_layout_path}")
    try:
        layout = json.loads(resolved_layout_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Layout is not valid JSON: {resolved_layout_path}") from exc
    if not isinstance(layout, list) or not all(isinstance(item, dict) for item in layout):
        raise ValueError("Layout must be a JSON array of objects.")
    for layout_object in layout:
        if not isinstance(layout_object.get("id"), str):
            raise ValueError("Each layout object must have a string id.")
    return layout


def _validate_image_path(image_path: str | Path) -> Path:
    resolved_image_path = Path(image_path).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Image input not found: {resolved_image_path}")
    if resolved_image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Image input must be one of the supported formats: {_SUPPORTED_IMAGE_SUFFIXES}.")
    return resolved_image_path
