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

from dataclasses import dataclass
from pathlib import Path
import shutil

from PIL import Image

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_generation import (
    ImageGenerationClient,
)
from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import SceneEditPlan
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    MaskCandidate,
    build_mask_candidates,
    invert_mask_if_foreground_is_off_center,
    save_binary_mask,
    union_overlapping_mask_candidates,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)


@dataclass(frozen=True)
class _AddedAssetInfo:
    """Semantic information needed while preparing one newly added asset."""

    object_id: str
    category: str
    name: str
    description: str


def prepare_scene_edit_assets(
    *,
    scene_edit_plan: SceneEditPlan,
    output_root: str | Path,
    image_generation_client: ImageGenerationClient,
    geometry_generation_client: GeometryGenerationClient,
    image_segmentation_client: ImageSegmentationClient,
    vlm_client: OpenAICompatibleVLM | None = None,
) -> list[SceneObject]:
    """Generate canonical SimReady assets for a scene edit's add operations.

    Move-only and delete-only plans return immediately without modifying an
    existing asset-preparation directory. For add operations, the function
    generates and segments one image per object, creates coarse geometry,
    processes it into SimReady geometry, and resets the returned objects to
    identity edit-time poses.

    Args:
        scene_edit_plan: Validated edit plan whose add operations define the
            objects to generate.
        output_root: Scene Engine output root. Intermediate artifacts are
            written below ``scene_editing/asset_preparation``.
        image_generation_client: Client used to render object images from the
            operation descriptions.
        geometry_generation_client: Client used to create coarse GLB geometry
            from each generated image and mask.
        image_segmentation_client: Client used to isolate the generated object
            in each image.
        vlm_client: Optional VLM used by SimReady processing to estimate asset
            scale and orientation.

    Returns:
        Added ``SceneObject`` assets in edit-plan order, or an empty list when
        the plan contains no add operations.

    Raises:
        ValueError: If add metadata or generated image, mask, and geometry
            mappings are incomplete or inconsistent.
        FileNotFoundError: If geometry generation does not produce an expected
            GLB file.
    """
    # Prepare descriptions for all newly added objects.
    added_asset_descriptions = _collect_added_asset_descriptions(scene_edit_plan)
    # Skip asset generation when the edit plan only moves or deletes existing objects.
    if not added_asset_descriptions:
        return []

    # Recreate this stage only when new assets need image, segmentation, and geometry outputs.
    stage_output_root = (
        Path(output_root).expanduser().resolve() / "scene_editing" / "asset_preparation"
    )
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)

    generated_asset_images = _generate_added_asset_images(
        added_asset_descriptions=added_asset_descriptions,
        stage_output_root=stage_output_root,
        image_generation_client=image_generation_client,
    )
    generated_asset_masks = _segment_generated_added_asset_images(
        added_asset_descriptions=added_asset_descriptions,
        generated_asset_images=generated_asset_images,
        stage_output_root=stage_output_root,
        image_segmentation_client=image_segmentation_client,
    )

    generated_asset_glbs = _generate_added_assets_coarse_geometry(
        generated_asset_images=generated_asset_images,
        generated_asset_masks=generated_asset_masks,
        stage_output_root=stage_output_root,
        geometry_generation_client=geometry_generation_client,
    )
    # Build a list of added SceneObjects.
    added_assets = _build_added_scene_objects(
        added_asset_descriptions=added_asset_descriptions,
        generated_asset_glbs=generated_asset_glbs,
    )
    # The temporary scene contains only new assets because the existing table is reused.
    tmp_scene = Scene(objects=added_assets)
    simready_processor = SimReadyProcessor(
        scene=tmp_scene,
        coarse_layout_by_id=_coarse_layouts_by_id(generated_asset_glbs),
        coarse_geometry_root=stage_output_root / "coarse_geometry",
        simready_geometry_root=stage_output_root / "simready_geometry",
        # Every added asset uses the VLM's pose and post-pose XY footprint scale.
        config=SimReadyProcessorConfig(
            use_vlm_scale=vlm_client is not None,
            use_vlm_rotation=vlm_client is not None,
            # Explicit edit pose descriptions override the default stable pose.
            pose_descriptions_by_id={
                operation.object_id: operation.pose_description
                for operation in scene_edit_plan.operations
                if (
                    operation.op == "add"
                    and operation.object_id is not None
                    and operation.pose_description is not None
                )
            },
        ),
        vlm_client=vlm_client,
    )
    # process_assets() validates and processes assets only; it does not require a table.
    simready_processor.process_assets()
    # Canonical GLBs use identity edit-time poses; layout editing sets them later.
    _reset_added_asset_layouts(added_assets)
    return added_assets


def _build_added_scene_objects(
    *,
    added_asset_descriptions: list[_AddedAssetInfo],
    generated_asset_glbs: list[tuple[str, Path]],
) -> list[SceneObject]:
    """Build temporary SceneObjects from generated coarse GLBs."""
    glbs_by_id = dict(generated_asset_glbs)
    if len(glbs_by_id) != len(generated_asset_glbs):
        raise ValueError("Generated asset GLBs must use unique object ids.")
    assets: list[SceneObject] = []
    for asset_info in added_asset_descriptions:
        glb_path = glbs_by_id.get(asset_info.object_id)
        if glb_path is None:
            raise ValueError(f"Generated asset {asset_info.object_id!r} has no GLB.")
        assets.append(
            SceneObject(
                id=asset_info.object_id,
                kind="asset",
                category=asset_info.category,
                name=asset_info.name,
                description=asset_info.description,
                simready_glb_path=str(glb_path),
            )
        )
    return assets


def _reset_added_asset_layouts(added_assets: list[SceneObject]) -> None:
    """Reset added asset poses after SimReady canonicalization."""
    for asset in added_assets:
        asset.rot = [0.0, 0.0, 0.0]
        asset.pos = [0.0, 0.0, 0.0]
        asset.scale = [1.0, 1.0, 1.0]


def _coarse_layouts_by_id(
    generated_asset_glbs: list[tuple[str, Path]],
) -> dict[str, dict[str, object]]:
    """Build edit-time layouts with fixed identity poses and scale."""
    return {
        object_id: {
            "rot": [0.0, 0.0, 0.0],
            "pos": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        for object_id, _ in generated_asset_glbs
    }


def _collect_added_asset_descriptions(
    scene_edit_plan: SceneEditPlan,
) -> list[_AddedAssetInfo]:
    """Return complete semantic information for add operations in plan order."""
    # Existing assets already have SimReady GLBs, so only add operations need assets.
    added_asset_descriptions: list[_AddedAssetInfo] = []
    for operation in scene_edit_plan.operations:
        if operation.op != "add":
            continue
        if (
            operation.object_id is None
            or operation.category is None
            or operation.name is None
            or operation.description is None
        ):
            raise ValueError(
                "Add operations must have an object_id, category, name, and description."
            )
        added_asset_descriptions.append(
            _AddedAssetInfo(
                object_id=operation.object_id,
                category=operation.category,
                name=operation.name,
                description=operation.description,
            )
        )
    return added_asset_descriptions


def _generate_added_asset_images(
    *,
    added_asset_descriptions: list[_AddedAssetInfo],
    stage_output_root: Path,
    image_generation_client: ImageGenerationClient,
) -> list[tuple[str, Path]]:
    """Generate one stable PNG for each new object description."""
    # Prepare a list.
    generated_asset_images: list[tuple[str, Path]] = []
    # Create a subdir.
    image_output_root = stage_output_root / "generated_images"
    image_output_root.mkdir(parents=True, exist_ok=True)

    for asset_info in added_asset_descriptions:
        object_id = asset_info.object_id
        # Stable object IDs preserve the image-to-asset mapping across later stages.
        image_path = image_generation_client.generate_image_by_prompt(
            prompt=asset_info.description,
            output_path=image_output_root / f"{object_id}.png",
        )
        generated_asset_images.append((object_id, image_path))
    return generated_asset_images


def _segment_generated_added_asset_images(
    *,
    added_asset_descriptions: list[_AddedAssetInfo],
    generated_asset_images: list[tuple[str, Path]],
    stage_output_root: Path,
    image_segmentation_client: ImageSegmentationClient,
) -> list[tuple[str, Path]]:
    """Segment each generated image with its description and return binary masks."""
    asset_info_by_id = {
        asset_info.object_id: asset_info for asset_info in added_asset_descriptions
    }
    if len(asset_info_by_id) != len(added_asset_descriptions):
        raise ValueError("Added asset descriptions must use unique object ids.")

    masks_output_root = stage_output_root / "generated_masks"
    masks_output_root.mkdir(parents=True, exist_ok=True)
    generated_asset_masks: list[tuple[str, Path]] = []
    for object_id, image_path in generated_asset_images:
        asset_info = asset_info_by_id.get(object_id)
        if asset_info is None:
            raise ValueError(f"Generated image {object_id!r} has no description.")

        candidates: list[MaskCandidate] = []
        # Retry with simpler semantic prompts when the detailed description is not found.
        for prompt in (asset_info.description, asset_info.name, asset_info.category):
            candidates = union_overlapping_mask_candidates(
                build_mask_candidates(
                    image_segmentation_client.segment_single_object(
                        image_path=image_path,
                        prompt=prompt,
                    )
                ),
                min_iou=0.8,
            )
            if candidates:
                break
        # A single generated object may still yield multiple SAM3 candidates; use the first one.
        if not candidates:
            raise ValueError(
                f"Generated asset {object_id!r} produced no segmentation candidates."
            )
        with Image.open(image_path) as image:
            image_size = image.size
        mask_path = save_binary_mask(
            invert_mask_if_foreground_is_off_center(candidates[0]),
            image_size=image_size,
            output_path=masks_output_root / f"{object_id}_mask.png",
        )
        generated_asset_masks.append((object_id, mask_path))
    return generated_asset_masks


def _generate_added_assets_coarse_geometry(
    *,
    generated_asset_images: list[tuple[str, Path]],
    generated_asset_masks: list[tuple[str, Path]],
    stage_output_root: Path,
    geometry_generation_client: GeometryGenerationClient,
) -> list[tuple[str, Path]]:
    """Generate one coarse GLB for each generated image and binary mask."""
    masks_by_id = dict(generated_asset_masks)
    if len(masks_by_id) != len(generated_asset_masks):
        raise ValueError("Generated asset masks must use unique object ids.")
    if set(masks_by_id) != {object_id for object_id, _ in generated_asset_images}:
        raise ValueError(
            "Generated asset images and masks must have matching object ids."
        )

    geometry_output_root = stage_output_root / "coarse_geometry"
    geometry_output_root.mkdir(parents=True, exist_ok=True)
    generated_asset_glbs: list[tuple[str, Path]] = []
    for object_id, image_path in generated_asset_images:
        # Each generated object has its own color image, so it needs an individual request.
        geometry_generation_client.generate_objects(
            image_path=image_path,
            object_masks=[(object_id, masks_by_id[object_id])],
            output_root=geometry_output_root,
        )
        glb_path = geometry_output_root / f"{object_id}.glb"
        if not glb_path.is_file():
            raise FileNotFoundError(
                f"Geometry generation did not produce a GLB for {object_id!r}: {glb_path}"
            )
        generated_asset_glbs.append((object_id, glb_path))
    return generated_asset_glbs
