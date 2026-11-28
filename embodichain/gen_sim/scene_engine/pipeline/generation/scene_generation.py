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

__all__: list[str] = []

from embodichain.gen_sim.services.compat import reconstruct
from embodichain.gen_sim.services.contracts import GeometryRequest


from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from shapely.geometry import Polygon

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from embodichain.gen_sim.services.compat import GeometryClient
from embodichain.gen_sim.services.compat import ArticulationClient
from embodichain.gen_sim.services.compat import ImageClient, SegmentationClient
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    TABLE_OBJECT_ID,
    SceneGraph,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.services.compat import VisionClient
from embodichain.gen_sim.scene_engine.pipeline.utils._asset_posture import (
    review_asset_posture,
)
from embodichain.gen_sim.scene_engine.pipeline.utils._degraded_table_asset import (
    generate_degraded_table_asset,
)
from embodichain.gen_sim.scene_engine.pipeline.utils._degraded_table_fit import (
    fit_degraded_table,
    tabletop_level_rotation,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp import (
    AssetsGroupSupportClamp,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner import (
    AssetsGroupTableAligner,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer import (
    AssetsSupportLayoutOptimizer,
)
from embodichain.gen_sim.asset_engine.utils.articulated_usdc_utils import (
    _canonicalize_articulated_usdc_bottom_center,
)
from embodichain.gen_sim.asset_engine.pipeline.articulation import (
    generate_articulated_usdcs,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.floor_placement import (
    FloorSide,
    floor_placement_delta,
    identify_floor_placements,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler import (
    GravitySettleBody,
    GravitySettler,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer import (
    ParentSurfaceLayoutOptimizer,
    ParentSurfaceLayoutProblem,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    apply_rest_pose_preserving_yaw,
    measure_scene_object_z_up_world_aabb,
    rotate_scene_z_up_world,
    scene_object_y_up_layout,
    translate_scene_object_y_up_by_z_up_delta,
    y_up_to_z_up_matrix,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_heading import (
    review_scene_heading,
    review_settled_scene_heading,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
    SupportSurfaceConfig,
    TableSupportSurfaceDetector,
    store_table_support,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer import (
    VisualYawOptimizer,
)
from embodichain.utils.logger import log_info

if TYPE_CHECKING:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_heading import (
        _SourceImageAudit,
    )

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_DEGRADED_TABLETOP_HEIGHT_M = (
    0.75  # Metric height prior when the floor is outside the image.
)
_FLOOR_RELEASE_CLEARANCE_M = 0.02


def generate_scene_and_refine(
    image_path: str | Path,
    output_root: str | Path,
    scene: Scene,
    scene_graph: SceneGraph,
    *,
    geometry_generation_client: GeometryClient,
    vlm_client: VisionClient,
    articulated_generation_client: ArticulationClient | None = None,
    image_generation_client: ImageClient | None = None,
    image_segmentation_client: SegmentationClient | None = None,
    source_image_audit: _SourceImageAudit | None = None,
) -> Scene:

    resolved_image_path = _validate_image_path(image_path)
    # Validate the scene graph before layout refinement consumes it.
    scene_graph.validate()
    table = scene.table
    recover_table = table is not None and table.table_state == "degraded"
    if recover_table and (
        image_generation_client is None or image_segmentation_client is None
    ):
        raise ValueError("Degraded table recovery requires image and mask clients.")
    # Create stage output directory.
    stage_output_root = Path(output_root).expanduser().resolve() / "scene_generation"
    if stage_output_root.exists():
        shutil.rmtree(stage_output_root)
    stage_output_root.mkdir(parents=True, exist_ok=True)
    # Create debug folder and the sim-ready geometry folder.
    debug_output_root = (
        stage_output_root / "debug"
    )  # Keeps the other files for debugging.
    coarse_geometry_output_root = (
        stage_output_root / "coarse_geometry"
    )  # Keeps the coarse geometries.
    simready_geometry_output_root = (
        stage_output_root / "simready_geometry"
    )  # Keeps the final-used geometries.
    debug_output_root.mkdir()
    coarse_geometry_output_root.mkdir()
    simready_geometry_output_root.mkdir()

    # Coarse geometry generation and coarse layout generation.
    coarse_scales_y_up_by_id = _generate_coarse_results_from_masks(
        image_path=resolved_image_path,
        debug_output_root=debug_output_root,
        coarse_geometry_output_root=coarse_geometry_output_root,
        scene=scene,  # Use the masks which are kept in the scene data structure.
        geometry_generation_client=geometry_generation_client,
    )
    # Simready all the assets(includes table).
    # Treat table and assets seperately.
    coarse_layout = _load_layout(coarse_geometry_output_root / "coarse_layout.json")
    coarse_layout_by_id = {
        layout_object["id"]: layout_object for layout_object in coarse_layout
    }
    # Direct tabletop assets receive their reviewed pose during layout, when the
    # whole base-and-descendants subtree can follow the adjustment.
    table_root_ids = _table_on_asset_ids(
        scene_graph=scene_graph, table_id=TABLE_OBJECT_ID
    )
    pose_descriptions_by_id = {
        node.object_id: node.pose_description
        for node in scene_graph.nodes
        if node.object_id != TABLE_OBJECT_ID
        and node.object_id not in table_root_ids
        and node.pose_description is not None
    }
    simready_processor = SimReadyProcessor(
        scene=scene,
        coarse_layout_by_id=coarse_layout_by_id,
        coarse_geometry_root=coarse_geometry_output_root,
        simready_geometry_root=simready_geometry_output_root,
        debug_output_root=debug_output_root,
        # Keep geometry-server scale while applying graph self-pose semantics.
        config=SimReadyProcessorConfig(
            use_vlm_scale=False,
            use_vlm_rotation=False,
            pose_descriptions_by_id=pose_descriptions_by_id,
        ),
        vlm_client=vlm_client,
    )
    simready_table_layout = simready_processor.process_table(
        reference_plane=recover_table
    )
    ground_shift = 0.0
    if recover_table:
        assert (
            table is not None
            and image_generation_client is not None
            and image_segmentation_client is not None
        )
        ground_shift = _recover_degraded_table(
            table=table,
            output_root=stage_output_root / "table_recovery",
            image_generation_client=image_generation_client,
            image_segmentation_client=image_segmentation_client,
            geometry_generation_client=geometry_generation_client,
            vlm_client=vlm_client,
        )
    _generate_articulated_usdcs(
        scene=scene,
        output_root=stage_output_root / "articulated_geometry",
        coarse_scales_y_up_by_id=coarse_scales_y_up_by_id,
        articulated_generation_client=articulated_generation_client,
    )
    simready_assets_layout = simready_processor.process_assets()
    if ground_shift:
        simready_assets_layout = _lift_table_subtrees(
            scene_graph=scene_graph,
            table_layout=simready_table_layout,
            assets_layout=simready_assets_layout,
            table_root_ids=table_root_ids,
            ground_shift=ground_shift,
        )
    # Defer each stacked object's existing visual yaw to its own placement layer.
    visual_yaws_by_id = _optimize_simready_asset_visual_yaws(
        scene=scene,
        simready_assets_layout=simready_assets_layout,
        coarse_layout_by_id=coarse_layout_by_id,
        vlm_client=vlm_client,
        debug_output_root=debug_output_root / "visual_yaw",
        excluded_ids=table_root_ids,
    )
    # Concat then save the table info and the assets info in one JSON file.
    simready_layout = [simready_table_layout, *simready_assets_layout]
    (simready_geometry_output_root / "simready_layout.json").write_text(
        json.dumps(simready_layout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Layout refinement will start with the table.
    refined_table_layout, refined_assets_layout = _layout_refinement(
        scene=scene,  # Update this data structure internally.
        scene_graph=scene_graph,
        image_path=resolved_image_path,
        source_image_audit=source_image_audit,
        simready_geometry_output_root=simready_geometry_output_root,  # Contains simready assets and their current coarse layout JSON.
        debug_output_root=debug_output_root,  # Keep the table support surface info + optimized layout info (render with matplotlib) for debugging.
        z_up_yaws_degrees_by_id=visual_yaws_by_id,
        vlm_client=vlm_client,
    )

    # Write the Updated scene JSON for debugging.
    (stage_output_root / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return scene


def _recover_degraded_table(
    *,
    table: SceneObject,
    output_root: Path,
    image_generation_client: ImageClient,
    image_segmentation_client: SegmentationClient,
    geometry_generation_client: GeometryClient,
    vlm_client: VisionClient,
) -> float:
    """Replace the leveled reference with a complete, grounded table."""
    if (
        table.simready_glb_path is None
        or table.support_surface_z is None
        or table.support_contour_xy is None
    ):
        raise ValueError("Table recovery requires a leveled reference surface.")
    if not table.visible_rgba_path or not Path(table.visible_rgba_path).is_file():
        raise ValueError("Table recovery requires the observed table RGB mask image.")
    old_height = table.support_surface_z
    reference_xy = np.asarray(table.support_contour_xy, dtype=float)
    reference_bounds = np.stack([reference_xy.min(axis=0), reference_xy.max(axis=0)])
    width, depth = (reference_bounds[1] - reference_bounds[0]).tolist()
    table_path = Path(table.simready_glb_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(table_path, output_root / "old_plane.glb")
    log_info("Recovering a complete table from the degraded tabletop.")
    generated_path = generate_degraded_table_asset(
        reference_image_path=Path(table.visible_rgba_path),
        description=table.description,
        support_width_depth=(width, depth),
        output_root=output_root,
        image_generation_client=image_generation_client,
        image_segmentation_client=image_segmentation_client,
        geometry_generation_client=geometry_generation_client,
    )
    basis = y_up_to_z_up_matrix()
    complete_mesh = load_glb_mesh(generated_path)
    complete_mesh.apply_transform(basis)
    level = np.eye(4)
    level[:3, :3] = tabletop_level_rotation(complete_mesh, complete=True)
    complete_mesh.apply_transform(level)
    detector = TableSupportSurfaceDetector(
        table_world_mesh=complete_mesh,
        debug_output_root=output_root,
        config=SupportSurfaceConfig(preserve_holes=True),
    )
    new_support = detector.select_with_vision(
        vlm_client=vlm_client, table_state="normal", require_complete_base=True
    )
    fit = fit_degraded_table(
        replacement_mesh_z_up=complete_mesh,
        replacement_support=new_support,
        reference_bounds_xy=reference_bounds,
        reference_top_z=old_height,
        grounded_tabletop_height_m=_DEGRADED_TABLETOP_HEIGHT_M,
    )
    detector.table_world_mesh = fit.mesh_z_up
    detector.support_region = fit.support_region
    detector.save_support_surface_debug_images()
    grounded_path = output_root / "grounded_table.glb"
    grounded_mesh_y_up = fit.mesh_z_up.copy()
    grounded_mesh_y_up.apply_transform(np.linalg.inv(basis))
    grounded_mesh_y_up.export(grounded_path, file_type="glb")
    (output_root / "alignment.json").write_text(
        json.dumps(
            {
                "reference_bounds_xy_m": reference_bounds.tolist(),
                "reference_height_m": old_height,
                "new_support_bounds_xy_m": np.asarray(
                    fit.support_region.optimization_rectangle.bounds
                )
                .reshape(2, 2)
                .tolist(),
                "new_support_contour_xy_m": [
                    list(point)
                    for point in fit.support_region.support_polygon.exterior.coords
                ],
                "scale_xyz": fit.scale_xyz,
                "aligned_table_bottom_m": fit.aligned_bottom_z,
                "shared_ground_shift_m": fit.ground_shift_z,
                "grounded_table_bottom_m": float(fit.mesh_z_up.bounds[0, 2]),
                "grounded_tabletop_height_m": fit.support_region.top_z,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(grounded_path, table_path)
    store_table_support(table, fit.support_region)
    log_info(
        f"Restored table {table.id!r}, with tabletop at {fit.support_region.top_z:.3f} m "
        f"and shared vertical shift {fit.ground_shift_z:+.3f} m."
    )
    return fit.ground_shift_z


def _lift_table_subtrees(
    *,
    scene_graph: SceneGraph,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    table_root_ids: set[str],
    ground_shift: float,
) -> list[dict[str, object]]:
    """Apply the table-local ground shift in the current SimReady layout frame."""
    table_matrix = layout_object_to_transform_matrix(table_layout)
    world_shift = table_matrix[:3, 1] * ground_shift
    roots = _select_asset_layouts(assets_layout=assets_layout, asset_ids=table_root_ids)
    return _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=assets_layout,
        updated_root_layouts=[
            {**layout, "pos": (np.asarray(layout["pos"]) + world_shift).tolist()}
            for layout in roots
        ],
        root_matrices_before_update=_layout_matrices_by_id(roots),
    )


def _optimize_simready_asset_visual_yaws(
    *,
    scene: Scene,
    simready_assets_layout: list[dict[str, object]],
    coarse_layout_by_id: dict[str, dict[str, object]],
    vlm_client: VisionClient,
    debug_output_root: str | Path,
    excluded_ids: set[str] | None = None,
) -> dict[str, float]:
    """Query an absolute Z-up yaw for each included observed SimReady asset."""
    assets_by_id = {asset.id: asset for asset in scene.assets}
    layout_ids = [layout.get("id") for layout in simready_assets_layout]
    if not all(isinstance(layout_id, str) and layout_id for layout_id in layout_ids):
        raise ValueError("Every SimReady asset layout must contain a non-empty id.")
    if len(layout_ids) != len(set(layout_ids)):
        raise ValueError("SimReady asset layouts must have unique ids.")
    if set(layout_ids) != set(assets_by_id):
        raise ValueError("SimReady asset layouts must match the scene asset ids.")

    yaws_degrees_by_id: dict[str, float] = {}
    excluded = excluded_ids or set()
    for asset_layout in simready_assets_layout:
        layout_id = asset_layout["id"]
        assert isinstance(layout_id, str)
        if layout_id in excluded:
            continue
        coarse_layout = coarse_layout_by_id.get(layout_id)
        if coarse_layout is None:
            raise ValueError(f"Coarse layout does not contain asset {layout_id!r}.")
        coarse_scale = coarse_layout.get("scale")
        if not isinstance(coarse_scale, list):
            raise ValueError(
                f"Coarse layout scale for asset {layout_id!r} must be a list."
            )
        yaws_degrees_by_id[layout_id] = VisualYawOptimizer(
            scene_object=assets_by_id[layout_id],
            baked_scale_y_up=coarse_scale,
            vlm_client=vlm_client,
            debug_output_root=debug_output_root,
        ).optimize_z_up_yaw_degrees()
    return yaws_degrees_by_id


def _apply_visual_yaws_to_simready_asset_layouts(
    *,
    simready_assets_layout: list[dict[str, object]],
    z_up_yaws_degrees_by_id: dict[str, float],
) -> list[dict[str, object]]:
    """Keep table-frame positions but replace each coarse rotation with canonical yaw."""
    layout_ids = [layout.get("id") for layout in simready_assets_layout]
    if not all(isinstance(layout_id, str) and layout_id for layout_id in layout_ids):
        raise ValueError("Every SimReady asset layout must contain a non-empty id.")
    typed_layout_ids = [str(layout_id) for layout_id in layout_ids]
    if len(typed_layout_ids) != len(set(typed_layout_ids)):
        raise ValueError("SimReady asset layouts must have unique ids.")
    if set(z_up_yaws_degrees_by_id) != set(typed_layout_ids):
        raise ValueError("Visual yaws must match the SimReady asset layout ids.")

    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = Rotation.from_euler(
        "x", 90.0, degrees=True
    ).as_matrix()
    z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)
    yawed_layouts: list[dict[str, object]] = []
    for asset_layout, asset_id in zip(simready_assets_layout, typed_layout_ids):
        original_matrix = layout_object_to_transform_matrix(asset_layout)
        z_up_yaw_matrix = np.eye(4)
        z_up_yaw_matrix[:3, :3] = Rotation.from_euler(
            "z", z_up_yaws_degrees_by_id[asset_id], degrees=True
        ).as_matrix()
        # The canonical SimReady pose replaces the coarse layout rotation.
        canonical_y_up_matrix = (
            z_up_to_y_up_matrix @ z_up_yaw_matrix @ y_up_to_z_up_matrix
        )
        canonical_y_up_matrix[:3, 3] = original_matrix[:3, 3]
        canonical_y_up_matrix[:3, :3] = canonical_y_up_matrix[:3, :3] @ np.diag(
            np.linalg.norm(original_matrix[:3, :3], axis=0)
        )
        yawed_layouts.append(
            transform_matrix_to_layout_object(asset_id, canonical_y_up_matrix)
        )
    return yawed_layouts


def _generate_coarse_results_from_masks(
    image_path: str | Path,
    debug_output_root: str | Path,
    coarse_geometry_output_root: str | Path,
    scene: Scene,
    *,
    geometry_generation_client: GeometryClient,
) -> dict[str, list[float]]:

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
        object_masks.append(
            (scene_object.id, mask_path)
        )  # id + mask, for avoiding the download glbs order confusion.

    # Sent the request, wait, then save the intermediate results.
    result = reconstruct(
        geometry_generation_client,
        GeometryRequest(
            Path(image_path), tuple(object_masks), Path(coarse_geometry_output_root)
        ),
    )
    if result.audit_path is not None:
        import shutil

        destination = Path(debug_output_root) / "geometry_generation_response.json"
        if result.audit_path.resolve() != destination.resolve():
            shutil.copyfile(result.audit_path, destination)
    coarse_root = Path(coarse_geometry_output_root).resolve()
    coarse_layout = []
    for obj in result.objects:
        rotation = Rotation.from_quat(obj.quaternion_xyzw)
        local_to_scene = np.eye(4)
        local_to_scene[:3, :3] = rotation.as_matrix() @ np.diag(obj.scale)
        local_to_scene[:3, 3] = obj.position
        mesh_path = obj.mesh_path.resolve()
        if mesh_path.is_relative_to(coarse_root):
            mesh_path = mesh_path.relative_to(coarse_root)
        source = {"geometry_path": mesh_path.as_posix()}
        if obj.source_quaternion_wxyz is not None:
            source.update(
                rotation_quaternion_wxyz=list(obj.source_quaternion_wxyz),
                translation=list(obj.position),
                scale=list(obj.scale),
                pose_coordinate_system=obj.pose_coordinate_system,
                pose_convention=obj.pose_convention,
                pose_convention_source=obj.pose_convention_source,
            )
        coarse_layout.append(
            {
                "id": obj.object_id,
                "rot": rotation.as_euler("xyz", degrees=True).tolist(),
                "pos": list(obj.position),
                "scale": list(obj.scale),
                "matrix": local_to_scene.tolist(),
                "source": source,
            }
        )
    (coarse_root / "coarse_layout.json").write_text(
        json.dumps(coarse_layout, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        str(layout_object["id"]): [float(value) for value in layout_object["scale"]]
        for layout_object in coarse_layout
    }


def _generate_articulated_usdcs(
    *,
    scene: Scene,
    output_root: str | Path,
    coarse_scales_y_up_by_id: dict[str, list[float]],
    articulated_generation_client: ArticulationClient | None,
) -> None:
    """Delegate articulated asset generation to the Asset Engine."""
    generate_articulated_usdcs(
        asset_objects=scene.objects,
        output_root=output_root,
        coarse_scales_y_up_by_id=coarse_scales_y_up_by_id,
        articulated_generation_client=articulated_generation_client,
        canonicalize_fn=_canonicalize_articulated_usdc_bottom_center,
    )


def _update_scene_final_y_up_layout_and_z_up_centers(
    *,
    scene: Scene,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
) -> None:
    """Write final y-up layouts and z-up XY centers into the scene."""
    if scene.table is None:
        raise ValueError("Cannot update a final layout without a table.")

    # Keep final poses in the y-up layout convention used by exported GLBs.
    _copy_y_up_layout_to_scene_object(scene.table, table_layout)
    assets_by_id = {asset.id: asset for asset in scene.assets}
    layout_ids = set()
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or asset_id not in assets_by_id:
            raise ValueError(f"Final layout contains unknown asset {asset_id!r}.")
        if asset_id in layout_ids:
            raise ValueError(f"Final layout contains duplicate asset {asset_id!r}.")
        _copy_y_up_layout_to_scene_object(assets_by_id[asset_id], asset_layout)
        layout_ids.add(asset_id)

    missing_assets = set(assets_by_id) - layout_ids
    if missing_assets:
        raise ValueError(
            f"Final layout is missing scene assets: {sorted(missing_assets)}."
        )

    # Measure final geometry in z-up so scene edits can compare tabletop XY positions.
    table_mesh, assets_aabb_corners_by_id = _measure_table_and_assets_in_z_up_world(
        table_layout=table_layout,
        assets_layout=assets_layout,
        geometry_root=geometry_root,
    )
    # Persist AABB centers for future scene-edit object disambiguation.
    scene.table.center_xy = table_mesh.bounds[:, :2].mean(axis=0).tolist()
    # The support is already in the refined table frame. The mesh AABB center
    # can differ from that frame's origin, especially after tabletop recovery.
    for asset in scene.assets:
        asset.center_xy = assets_aabb_corners_by_id[asset.id].mean(axis=0).tolist()


def _copy_y_up_layout_to_scene_object(
    scene_object: SceneObject,
    layout_object: dict[str, object],
) -> None:
    """Copy one y-up layout object after validating its id and numeric vectors."""
    if layout_object.get("id") != scene_object.id:
        raise ValueError(
            f"Layout id {layout_object.get('id')!r} does not match scene object "
            f"{scene_object.id!r}."
        )

    for field_name in ("rot", "pos", "scale"):
        values = layout_object.get(field_name)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(
                f"Layout object {scene_object.id!r} has invalid {field_name!r}."
            )
        vector = [float(value) for value in values]
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Layout object {scene_object.id!r} has non-finite {field_name!r}."
            )
        setattr(scene_object, field_name, vector)


def _layout_refinement(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    image_path: Path,
    simready_geometry_output_root: str | Path,
    debug_output_root: str | Path,
    z_up_yaws_degrees_by_id: dict[str, float],
    vlm_client: VisionClient,
    source_image_audit: _SourceImageAudit | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:

    # 1. All layouts and geometries below are SimReady outputs. Do not mix a
    # coarse layout with a SimReady GLB (or vice versa), because each object's
    # SimReady canonicalization may include its own local pose compensation.
    simready_layout = _load_layout(
        Path(simready_geometry_output_root) / "simready_layout.json"
    )
    if scene.table is None:
        raise ValueError("Cannot refine a layout without a table.")
    table_id = scene.table.id
    table_layout = next(
        (
            layout_object
            for layout_object in simready_layout
            if layout_object["id"] == table_id
        ),
        None,
    )
    if table_layout is None:
        raise ValueError(f"SimReady layout does not contain table {table_id!r}.")

    # Keep the intermediate layout y-up; the simulator converts final GLBs to
    # z-up. Left multiplication expresses every complete asset pose (position
    # and rotation) in the SimReady table frame.
    simready_table_to_world_matrix = layout_object_to_transform_matrix(table_layout)
    world_to_simready_table_matrix = np.linalg.inv(simready_table_to_world_matrix)

    # 2. The table defines the refined world frame, so its transform is exact
    # identity instead of a numerically reconstructed inverse(table) @ table.
    refined_table_layout = transform_matrix_to_layout_object(
        table_layout["id"],
        np.eye(4),
    )
    refined_assets_layout: list[dict[str, object]] = []
    for asset_layout in simready_layout:
        if asset_layout["id"] == table_layout["id"]:
            continue

        simready_asset_to_world_matrix = layout_object_to_transform_matrix(asset_layout)
        simready_asset_to_table_matrix = (
            world_to_simready_table_matrix @ simready_asset_to_world_matrix
        )

        # Converting an asset back through the table pose must reconstruct its
        # original SimReady world pose. This catches missing rotations, wrong
        # matrix order, and coarse/SimReady coordinate-system mixing early.
        if not np.allclose(
            simready_table_to_world_matrix @ simready_asset_to_table_matrix,
            simready_asset_to_world_matrix,
            atol=1e-6,
        ):
            raise ValueError(
                "SimReady table-frame conversion failed for asset "
                f"{asset_layout['id']!r}."
            )

        refined_assets_layout.append(
            transform_matrix_to_layout_object(
                asset_layout["id"],
                simready_asset_to_table_matrix,
            )
        )

    # Only direct on-table children participate in the table-level layout
    # stages. Their descendants follow each solved root transform until their
    # own parent-surface optimization is introduced in a later BFS pass.
    table_root_ids = _table_on_asset_ids(scene_graph=scene_graph, table_id=table_id)
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )
    geometry_root = Path(simready_geometry_output_root)
    debug_root = Path(debug_output_root)
    if not table_root_layouts:
        log_info("Scene has no on-table assets; skipping table layout refinement.")
        heading = review_scene_heading(
            image_path=image_path,
            source_image_audit=source_image_audit,
            table_layout=refined_table_layout,
            assets_layout=refined_assets_layout,
            geometry_root=geometry_root,
            debug_output_root=debug_root / "reference_heading",
        )
        initial_layout = [refined_table_layout, *refined_assets_layout]
        return _finalize_refined_layout(
            scene=scene,
            scene_graph=scene_graph,
            image_path=image_path,
            source_image_audit=source_image_audit,
            table_layout=refined_table_layout,
            assets_layout=refined_assets_layout,
            geometry_root=geometry_root,
            debug_root=debug_root,
            vlm_client=vlm_client,
            heading=heading,
            table_root_ids=table_root_ids,
            z_up_yaws_degrees_by_id=z_up_yaws_degrees_by_id,
            pose_frames=dict.fromkeys(
                ("before_posture", "before_y_align", "after_y_align"), initial_layout
            ),
        )

    if (
        scene.table.support_surface_z is None
        or scene.table.support_contour_xy is None
        or scene.table.support_optimization_rect_xy is None
    ):
        raise ValueError("Scene table has no persisted support geometry.")

    pose_frames = {
        "before_posture": deepcopy([refined_table_layout, *refined_assets_layout])
    }
    assets_by_id = {asset.id: asset for asset in scene.assets}
    floor_placements = identify_floor_placements(
        image_path=image_path,
        table=scene.table,
        assets=[asset for asset in scene.assets if asset.id in table_root_ids],
        vlm_client=vlm_client,
        debug_root=debug_root,
    )
    floor_parents = floor_placements.keys() & scene_graph._children_by_parent().keys()
    if floor_parents:
        names = ", ".join(repr(object_id) for object_id in sorted(floor_parents))
        raise ValueError(
            f"Floor-supported objects with on children are not supported: {names}."
        )
    graph_nodes_by_id = scene_graph.node_by_id()
    prepared_roots = []
    for layout in table_root_layouts:
        object_id = str(layout["id"])
        glb_path = Path(simready_geometry_output_root) / f"{object_id}.glb"
        rest_rotation = review_asset_posture(
            glb_path=glb_path,
            description=assets_by_id[object_id].description,
            pose_description=graph_nodes_by_id[object_id].pose_description,
            vlm_client=vlm_client,
            debug_output_root=Path(debug_output_root) / "asset_posture" / object_id,
        )
        prepared_roots.append(
            layout
            if rest_rotation is None
            else apply_rest_pose_preserving_yaw(
                layout,
                rest_rotation_y_up=rest_rotation,
                local_center=load_glb_mesh(glb_path).bounds.mean(axis=0),
            )
        )
    refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=refined_assets_layout,
        updated_root_layouts=prepared_roots,
        root_matrices_before_update=_layout_matrices_by_id(table_root_layouts),
    )
    pose_frames["before_y_align"] = deepcopy(
        [refined_table_layout, *refined_assets_layout]
    )

    table_root_ids.difference_update(floor_placements)
    # Prepare the tabletop bases and their descendants at the selected surface. The
    # heading scorer compares all four projections from these release poses.
    refined_table_layout, refined_assets_layout = _align_table_roots_individually(
        scene_graph=scene_graph,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        table_root_ids=table_root_ids,
        geometry_root=simready_geometry_output_root,
        support_top_z=scene.table.support_surface_z,
    )
    pose_frames["after_y_align"] = deepcopy(
        [refined_table_layout, *refined_assets_layout]
    )
    heading_degrees, heading_pivot_xy = review_scene_heading(
        image_path=image_path,
        source_image_audit=source_image_audit,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=geometry_root,
        debug_output_root=debug_root / "reference_heading",
        mask_paths={
            uid: path
            for uid in table_root_ids
            if (path := assets_by_id[uid].mask_path) is not None
        },
    )
    if not table_root_ids:
        return _finalize_refined_layout(
            scene=scene,
            scene_graph=scene_graph,
            image_path=image_path,
            source_image_audit=source_image_audit,
            table_layout=refined_table_layout,
            assets_layout=refined_assets_layout,
            geometry_root=geometry_root,
            debug_root=debug_root,
            vlm_client=vlm_client,
            heading=(heading_degrees, heading_pivot_xy),
            table_root_ids=table_root_ids,
            z_up_yaws_degrees_by_id=z_up_yaws_degrees_by_id,
            pose_frames=pose_frames,
            floor_placements=floor_placements,
        )

    # Re-select direct table children after posture and vertical placement.
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )

    # 5. Reuse the selected support geometry for planar refinement.
    table_support_polygon = Polygon(scene.table.support_contour_xy)
    table_optimization_rectangle = Polygon(scene.table.support_optimization_rect_xy)
    if not table_support_polygon.is_valid or table_support_polygon.is_empty:
        raise ValueError("Scene table support contour is not a valid polygon.")
    if (
        not table_optimization_rectangle.is_valid
        or table_optimization_rectangle.is_empty
    ):
        raise ValueError("Scene table optimization rectangle is not valid.")
    _, table_root_aabb_2d_z_up_world_corners_by_id = (
        _measure_table_and_assets_in_z_up_world(
            table_layout=refined_table_layout,
            assets_layout=table_root_layouts,
            geometry_root=simready_geometry_output_root,
        )
    )

    # 6. Keep the direct on-table clutter rigid in the table plane. A successful
    # result applies one shared z-up XY delta to every root AABB, and the same
    # transform is propagated to each root's descendants. It is *not* an asset
    # packing pass: pre-existing overlap is deliberately left to a later optimizer.
    table_root_matrices_before_clamp = _layout_matrices_by_id(table_root_layouts)
    group_clamp = AssetsGroupSupportClamp(
        support_region=table_support_polygon,
        assets_aabb_2d_z_up_world_corners_by_id=(
            table_root_aabb_2d_z_up_world_corners_by_id
        ),
        assets_layout=table_root_layouts,
        debug_output_root=debug_output_root,
    )
    # Clamp.
    clamped_table_root_layouts = group_clamp.clamp()
    # Save debug images.
    group_clamp.save_group_clamp_debug_images()
    # Update to descendants.
    refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=refined_assets_layout,
        updated_root_layouts=clamped_table_root_layouts,
        root_matrices_before_update=table_root_matrices_before_clamp,
    )
    # Re-select.
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )

    # The clamp returns y-up layouts; measure their resulting z-up AABBs again
    # so the following independent optimizer consumes the same world-frame
    # geometry as every other stage.
    _, clamped_table_root_aabb_2d_z_up_world_corners_by_id = (
        _measure_table_and_assets_in_z_up_world(
            table_layout=refined_table_layout,
            assets_layout=table_root_layouts,
            geometry_root=simready_geometry_output_root,
        )
    )

    # 7. Optimize independent on-table root positions inside the conservative
    # rectangle.
    # The clamp above already used the exact outer contour for the shared shift.
    table_root_matrices_before_optimization = _layout_matrices_by_id(table_root_layouts)
    overlap_optimizer = AssetsSupportLayoutOptimizer(
        support_region=table_optimization_rectangle,
        assets_aabb_2d_z_up_world_corners_by_id=(
            clamped_table_root_aabb_2d_z_up_world_corners_by_id
        ),
        assets_layout=table_root_layouts,
        debug_output_root=debug_output_root,
    )
    # Render this stage separately from the rigid group clamp.  The latter
    # intentionally preserves pre-existing overlaps, while this figure shows
    # whether independent AABB separation actually resolved them.
    optimized_table_root_layouts = overlap_optimizer.optimize()
    # Save debug image.
    overlap_optimizer.save_overlap_optimization_debug_images()
    # Update.
    refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=refined_assets_layout,
        updated_root_layouts=optimized_table_root_layouts,
        root_matrices_before_update=table_root_matrices_before_optimization,
    )
    # Re-select.
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )

    # 8. Settle only the direct on-table roots. Their unoptimized descendants
    # stay outside this simulation and inherit each root's final pose delta.
    table_root_matrices_before_settle = _layout_matrices_by_id(table_root_layouts)
    # Settle.
    settled_pose_by_id = GravitySettler(
        table_body=GravitySettleBody(
            scene_object=scene.table,
            y_up_layout=refined_table_layout,
        ),
        participant_bodies=[
            GravitySettleBody(
                scene_object=assets_by_id[str(asset_layout["id"])],
                y_up_layout=asset_layout,
            )
            for asset_layout in table_root_layouts
        ],
        dynamic_asset_ids=set(table_root_ids),
        static_asset_ids=set(),
    ).settle()
    settled_table_root_layouts: list[dict[str, object]] = []
    for asset_layout in table_root_layouts:
        asset_id = str(asset_layout["id"])
        settled_pose = settled_pose_by_id[asset_id]
        settled_table_root_layouts.append(
            {
                **asset_layout,
                "pos": settled_pose["pos"],
                "rot": settled_pose["rot"],
            }
        )
    # Update the descendants.
    refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
        scene_graph=scene_graph,
        assets_layout=refined_assets_layout,
        updated_root_layouts=settled_table_root_layouts,
        root_matrices_before_update=table_root_matrices_before_settle,
    )

    return _finalize_refined_layout(
        scene=scene,
        scene_graph=scene_graph,
        image_path=image_path,
        source_image_audit=source_image_audit,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=geometry_root,
        debug_root=debug_root,
        vlm_client=vlm_client,
        heading=(heading_degrees, heading_pivot_xy),
        table_root_ids=table_root_ids,
        z_up_yaws_degrees_by_id=z_up_yaws_degrees_by_id,
        pose_frames=pose_frames,
        floor_placements=floor_placements,
    )


def _finalize_refined_layout(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    image_path: Path,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: Path,
    debug_root: Path,
    vlm_client: VisionClient,
    heading: tuple[int, list[float]],
    table_root_ids: set[str],
    z_up_yaws_degrees_by_id: dict[str, float],
    pose_frames: dict[str, list[dict[str, object]]],
    floor_placements: dict[str, FloorSide] | None = None,
    source_image_audit: _SourceImageAudit | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Resolve the shared heading, place supported children, and publish the poses."""
    _sync_scene_y_up_poses_from_layouts(
        scene=scene,
        table_layout=table_layout,
        assets_layout=assets_layout,
    )
    program_heading, pivot_xy = heading
    tabletop_preview_layouts = assets_layout
    if floor_placements:
        tabletop_preview_layouts = [
            layout for layout in assets_layout if layout["id"] not in floor_placements
        ]
    heading_degrees = review_settled_scene_heading(
        image_path=image_path,
        source_image_audit=source_image_audit,
        table_layout=table_layout,
        assets_layout=tabletop_preview_layouts,
        geometry_root=geometry_root,
        debug_output_root=debug_root / "reference_heading",
        vision_client=vlm_client,
        proposed_yaw_degrees=program_heading,
        pivot_xy=pivot_xy,
    )
    if table_root_ids:
        # Check support bounds and settle children using the orientation they
        # will have once the selected heading is applied to the whole scene.
        assets_layout = _refine_on_children_bfs(
            scene=scene,
            scene_graph=scene_graph,
            table_layout=table_layout,
            assets_layout=assets_layout,
            table_root_ids=table_root_ids,
            debug_output_root=debug_root,
            z_up_yaws_degrees_by_id={
                asset_id: yaw - heading_degrees
                for asset_id, yaw in z_up_yaws_degrees_by_id.items()
            },
        )
    _update_scene_final_y_up_layout_and_z_up_centers(
        scene=scene,
        table_layout=table_layout,
        assets_layout=assets_layout,
        geometry_root=geometry_root,
    )
    pose_frames["before_heading"] = [table_layout, *assets_layout]
    if heading_degrees:
        rotate_scene_z_up_world(
            scene=scene,
            scene_graph=scene_graph,
            rotation_degrees=heading_degrees,
            pivot_xy=pivot_xy,
        )
        table_layout = scene_object_y_up_layout(scene.table)
        assets_layout = [scene_object_y_up_layout(asset) for asset in scene.assets]
    if floor_placements:
        assets_layout = _place_floor_roots(
            scene=scene,
            assets_layout=assets_layout,
            floor_placements=floor_placements,
            debug_root=debug_root,
        )
        _update_scene_final_y_up_layout_and_z_up_centers(
            scene=scene,
            table_layout=table_layout,
            assets_layout=assets_layout,
            geometry_root=geometry_root,
        )
    pose_frames["final"] = [table_layout, *assets_layout]
    (debug_root / "layout_pose_snapshots.json").write_text(
        json.dumps({"up_axis": "Y", "frames": pose_frames}, indent=2) + "\n",
        encoding="utf-8",
    )
    return table_layout, assets_layout


def _place_floor_roots(
    *,
    scene: Scene,
    assets_layout: list[dict[str, object]],
    floor_placements: dict[str, FloorSide],
    debug_root: Path,
) -> list[dict[str, object]]:
    """Release standalone ground assets beside the finalized table."""
    assert scene.table is not None
    table_bounds = np.asarray(
        measure_scene_object_z_up_world_aabb(scene_object=scene.table)
    )
    ground_z = float(table_bounds[0, 2])
    assets_by_id = {asset.id: asset for asset in scene.assets}
    root_layouts = _select_asset_layouts(
        assets_layout=assets_layout, asset_ids=set(floor_placements)
    )
    original_poses = _layout_matrices_by_id(root_layouts)
    y_up_to_z_up = y_up_to_z_up_matrix()[:3, :3]
    z_up_to_y_up = y_up_to_z_up.T
    ground_offset_y_up = z_up_to_y_up @ np.array([0.0, 0.0, ground_z])
    release_layouts = {}
    for object_id, side in floor_placements.items():
        asset = assets_by_id[object_id]
        delta = floor_placement_delta(
            table_bounds,
            np.asarray(measure_scene_object_z_up_world_aabb(scene_object=asset)),
            side,
        )
        # The temporary world has its ground at Z=0. Release above that plane.
        delta[2] += _FLOOR_RELEASE_CLEARANCE_M - ground_z
        pose = original_poses[object_id].copy()
        pose[:3, 3] += z_up_to_y_up @ delta
        release_layouts[object_id] = transform_matrix_to_layout_object(object_id, pose)

    table_layout = scene_object_y_up_layout(scene.table)
    table_layout["pos"] = (
        np.asarray(table_layout["pos"]) - ground_offset_y_up
    ).tolist()
    settled = GravitySettler(
        table_body=GravitySettleBody(
            scene_object=scene.table, y_up_layout=table_layout
        ),
        participant_bodies=[
            GravitySettleBody(
                scene_object=assets_by_id[object_id], y_up_layout=release_layout
            )
            for object_id, release_layout in release_layouts.items()
        ],
        dynamic_asset_ids=set(release_layouts),
        static_asset_ids=set(),
    ).settle()

    updated_layouts = {}
    audit = {}
    for object_id, side in floor_placements.items():
        asset = assets_by_id[object_id]
        settled_layout = {**release_layouts[object_id], **settled[object_id]}
        settled_pos = np.asarray(settled_layout["pos"]) + ground_offset_y_up
        settled_layout["pos"] = settled_pos.tolist()
        settled_asset = replace(asset)
        _copy_y_up_layout_to_scene_object(settled_asset, settled_layout)
        # Rolling or tipping may change the footprint. Only move it farther
        # from the table if needed, keeping the height and rotation from physics.
        correction = floor_placement_delta(
            table_bounds,
            np.asarray(
                measure_scene_object_z_up_world_aabb(scene_object=settled_asset)
            ),
            side,
        )
        correction[2] = 0.0
        if side in {"left", "front"}:
            correction = np.minimum(correction, 0.0)
        else:
            correction = np.maximum(correction, 0.0)
        final_pos = settled_pos + z_up_to_y_up @ correction
        settled_layout["pos"] = final_pos.tolist()
        updated_layouts[object_id] = settled_layout
        asset.placement_support = "floor"
        audit[object_id] = {
            "side": side,
            "release_clearance_m": _FLOOR_RELEASE_CLEARANCE_M,
            "ground_z_up_m": ground_z,
            "world_delta_z_up_m": (
                y_up_to_z_up @ (final_pos - original_poses[object_id][:3, 3])
            ).tolist(),
        }
    (debug_root / "floor_placement_final.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return [updated_layouts.get(str(layout["id"]), layout) for layout in assets_layout]


def _align_table_roots_individually(
    *,
    scene_graph: SceneGraph,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    table_root_ids: set[str],
    geometry_root: str | Path,
    support_top_z: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Place each direct on-table root above the table and move its subtree."""
    refined_table_layout = table_layout
    refined_assets_layout = assets_layout
    # Preserve layout order while each root receives an independent z correction.
    initial_table_root_layouts = _select_asset_layouts(
        assets_layout=assets_layout,
        asset_ids=table_root_ids,
    )
    for initial_root_layout in initial_table_root_layouts:
        root_id = initial_root_layout.get("id")
        if not isinstance(root_id, str) or not root_id:
            raise ValueError(
                "Every direct on-table layout must contain a non-empty id."
            )
        current_root_layouts = _select_asset_layouts(
            assets_layout=refined_assets_layout,
            asset_ids={root_id},
        )
        root_matrices_before_align = _layout_matrices_by_id(current_root_layouts)
        aligned_table_layout, aligned_root_layouts = AssetsGroupTableAligner(
            table_layout=refined_table_layout,
            assets_layout=current_root_layouts,
            geometry_root=geometry_root,
            support_top_z=support_top_z,
        ).align()
        refined_table_layout = aligned_table_layout
        # Propagating the complete root delta keeps descendants attached to it.
        refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
            scene_graph=scene_graph,
            assets_layout=refined_assets_layout,
            updated_root_layouts=aligned_root_layouts,
            root_matrices_before_update=root_matrices_before_align,
        )
    return refined_table_layout, refined_assets_layout


def _refine_on_children_bfs(
    *,
    scene: Scene,
    scene_graph: SceneGraph,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    table_root_ids: set[str],
    debug_output_root: str | Path,
    z_up_yaws_degrees_by_id: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Refine every non-table ``on`` group after its parent has settled."""
    if scene.table is None:
        raise ValueError("Cannot refine parent-surface groups without a table.")
    _sync_scene_y_up_poses_from_layouts(
        scene=scene,
        table_layout=table_layout,
        assets_layout=assets_layout,
    )
    assets_by_id = {asset.id: asset for asset in scene.assets}
    children_by_parent: dict[str, list[str]] = {}
    for node in scene_graph.nodes:
        if node.parent_id is not None and node.parent_relation == "on":
            children_by_parent.setdefault(node.parent_id, []).append(node.object_id)

    parent_surface_optimizer = ParentSurfaceLayoutOptimizer()
    pending_parent_ids = deque(table_root_ids)
    refined_assets_layout = assets_layout
    while pending_parent_ids:
        parent_id = pending_parent_ids.popleft()
        child_ids = children_by_parent.get(parent_id, [])
        if not child_ids:
            continue
        parent = assets_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Non-table parent {parent_id!r} is not an asset.")

        # Keep inherited poses through the parent's placement. Apply the existing
        # child-level yaw only when this support group reaches its own turn.
        if z_up_yaws_degrees_by_id is not None:
            child_layouts = _select_asset_layouts(
                assets_layout=refined_assets_layout, asset_ids=set(child_ids)
            )
            refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
                scene_graph=scene_graph,
                assets_layout=refined_assets_layout,
                updated_root_layouts=_apply_visual_yaws_to_simready_asset_layouts(
                    simready_assets_layout=child_layouts,
                    z_up_yaws_degrees_by_id={
                        key: z_up_yaws_degrees_by_id[key] for key in child_ids
                    },
                ),
                root_matrices_before_update=_layout_matrices_by_id(child_layouts),
            )
            for child_layout in _select_asset_layouts(
                assets_layout=refined_assets_layout, asset_ids=set(child_ids)
            ):
                _copy_y_up_layout_to_scene_object(
                    assets_by_id[str(child_layout["id"])], child_layout
                )

        child_id_set = set(child_ids)
        parent_aabb = np.asarray(
            measure_scene_object_z_up_world_aabb(scene_object=parent), dtype=float
        )
        if parent_aabb.shape != (2, 3) or not np.isfinite(parent_aabb).all():
            raise ValueError(f"Parent {parent_id!r} has no finite mesh bounds.")
        parent_aabb_xy = parent_aabb[:, :2].tolist()
        parent_top_z = float(parent_aabb[1, 2])
        child_aabbs_by_id = {
            child_id: np.asarray(
                measure_scene_object_z_up_world_aabb(
                    scene_object=assets_by_id[child_id]
                ),
                dtype=float,
            )
            for child_id in child_ids
        }
        child_aabbs_xy_by_id = {
            child_id: bounds[:, :2] for child_id, bounds in child_aabbs_by_id.items()
        }

        # Establish release height at the inherited XZ, then carry the Y change
        # to descendants before seeking a horizontal position in the parent AABB.
        child_layouts = _select_asset_layouts(
            assets_layout=refined_assets_layout, asset_ids=child_id_set
        )
        for child_id in child_ids:
            child = assets_by_id[child_id]
            child.center_xy = child_aabbs_xy_by_id[child_id].mean(axis=0).tolist()
            assert (
                child.pos is not None
            )  # Synchronized from the inherited layout above.
            x, y, z = child.pos
            delta_y = parent_top_z + 0.02 - child_aabbs_by_id[child_id][0, 2]
            child.pos = [x, y + float(delta_y), z]
        refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
            scene_graph=scene_graph,
            assets_layout=refined_assets_layout,
            updated_root_layouts=[
                scene_object_y_up_layout(assets_by_id[child_id])
                for child_id in child_ids
            ],
            root_matrices_before_update=_layout_matrices_by_id(child_layouts),
        )
        _sync_scene_y_up_poses_from_layouts(
            scene=scene, table_layout=table_layout, assets_layout=refined_assets_layout
        )

        child_seed_xy_by_id, projected_child_aabbs_xy_by_id = (
            _project_child_aabb_centers_into_parent_aabb(
                parent_aabb_xy=parent_aabb_xy,
                child_aabbs_xy_by_id=child_aabbs_xy_by_id,
            )
        )
        projected_child_ids = [
            child_id
            for child_id in child_ids
            if not np.allclose(
                child_aabbs_xy_by_id[child_id],
                projected_child_aabbs_xy_by_id[child_id],
            )
        ]
        if projected_child_ids:
            log_info(
                "Projected "
                f"{len(projected_child_ids)} child AABBs into support bounds on {parent_id!r}: "
                f"{projected_child_ids}."
            )
        else:
            log_info(
                f"All direct children are already inside their support bounds on {parent_id!r}."
            )
        _render_parent_child_aabb_transition(
            parent_id=parent_id,
            parent_aabb_xy=parent_aabb_xy,
            before_child_aabbs_xy_by_id=child_aabbs_xy_by_id,
            after_child_aabbs_xy_by_id=projected_child_aabbs_xy_by_id,
            before_title="Before support projection",
            after_title="After support projection",
            output_path=(
                Path(debug_output_root)
                / f"parent_{parent_id}_child_aabb_projection_2d.png"
            ),
        )
        # All image-observed children are movable and start from their nearest
        # position within the parent's horizontal AABB.
        parent_surface_problem = ParentSurfaceLayoutProblem(
            assets_by_id=assets_by_id,
            child_ids=child_ids,
            child_seed_xy_by_id=child_seed_xy_by_id,
            imported_child_ids=child_id_set,
            fixed_child_xy_by_id={child_id: None for child_id in child_ids},
            parent_aabb_xy=parent_aabb_xy,
            parent_top_z=parent_top_z,
            half_extents_xy_by_id={
                child_id: (bounds[1] - bounds[0]) / 2.0
                for child_id, bounds in child_aabbs_xy_by_id.items()
            },
            child_relations=[
                relation
                for relation in scene_graph.relations
                if relation.source_id in child_id_set
                and relation.target_id in child_id_set
            ],
        )
        solved_child_xy_by_id = parent_surface_optimizer.optimize(
            parent_surface_problem
        )
        optimized_child_aabbs_xy_by_id = {
            child_id: _translated_aabb_xy(
                aabb_xy=projected_child_aabbs_xy_by_id[child_id],
                delta_xy=(
                    np.asarray(solved_child_xy_by_id[child_id], dtype=float)
                    - np.asarray(child_seed_xy_by_id[child_id], dtype=float)
                ),
            )
            for child_id in child_ids
        }
        _render_parent_child_aabb_transition(
            parent_id=parent_id,
            parent_aabb_xy=parent_aabb_xy,
            before_child_aabbs_xy_by_id=projected_child_aabbs_xy_by_id,
            after_child_aabbs_xy_by_id=optimized_child_aabbs_xy_by_id,
            before_title="Before parent-child AABB optimization",
            after_title="After parent-child AABB optimization",
            output_path=(
                Path(debug_output_root)
                / f"parent_{parent_id}_child_aabb_optimization_2d.png"
            ),
        )

        child_matrices_before_translation = _layout_matrices_by_id(
            _select_asset_layouts(
                assets_layout=refined_assets_layout,
                asset_ids=child_id_set,
            )
        )
        for child_id, solved_xy in solved_child_xy_by_id.items():
            delta_xy = np.asarray(solved_xy) - child_aabbs_xy_by_id[child_id].mean(
                axis=0
            )
            # Keep the release height and rotation during this XZ adjustment.
            translate_scene_object_y_up_by_z_up_delta(
                scene_object=assets_by_id[child_id],
                delta_xy=delta_xy.tolist(),
            )
        refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
            scene_graph=scene_graph,
            assets_layout=refined_assets_layout,
            updated_root_layouts=[
                scene_object_y_up_layout(assets_by_id[child_id])
                for child_id in child_ids
            ],
            root_matrices_before_update=child_matrices_before_translation,
        )
        _sync_scene_y_up_poses_from_layouts(
            scene=scene,
            table_layout=table_layout,
            assets_layout=refined_assets_layout,
        )

        child_matrices_before_settle = _layout_matrices_by_id(
            _select_asset_layouts(
                assets_layout=refined_assets_layout,
                asset_ids=child_id_set,
            )
        )
        settled_pose_by_id = parent_surface_optimizer.settle_dynamic_children(
            table=scene.table,
            parent=parent,
            problem=parent_surface_problem,
            dynamic_child_ids=child_id_set,
        )
        for child_id, settled_pose in settled_pose_by_id.items():
            assets_by_id[child_id].pos = settled_pose["pos"]
            assets_by_id[child_id].rot = settled_pose["rot"]
        refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
            scene_graph=scene_graph,
            assets_layout=refined_assets_layout,
            updated_root_layouts=[
                scene_object_y_up_layout(assets_by_id[child_id])
                for child_id in child_ids
            ],
            root_matrices_before_update=child_matrices_before_settle,
        )
        _sync_scene_y_up_poses_from_layouts(
            scene=scene,
            table_layout=table_layout,
            assets_layout=refined_assets_layout,
        )
        pending_parent_ids.extend(child_ids)

    return refined_assets_layout


def _sync_scene_y_up_poses_from_layouts(
    *,
    scene: Scene,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
) -> None:
    """Synchronize current y-up poses without rewriting final spatial metadata."""
    if scene.table is None:
        raise ValueError("Cannot synchronize layouts without a table.")
    _copy_y_up_layout_to_scene_object(scene.table, table_layout)
    assets_by_id = {asset.id: asset for asset in scene.assets}
    synced_asset_ids: set[str] = set()
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        asset = assets_by_id.get(asset_id)
        if asset is None:
            raise ValueError(f"Current layout contains unknown asset {asset_id!r}.")
        if asset_id in synced_asset_ids:
            raise ValueError(f"Current layout contains duplicate asset {asset_id!r}.")
        _copy_y_up_layout_to_scene_object(asset, asset_layout)
        synced_asset_ids.add(asset_id)
    if synced_asset_ids != set(assets_by_id):
        raise ValueError("Current layouts do not cover every scene asset.")


def _project_child_aabb_centers_into_parent_aabb(
    *,
    parent_aabb_xy: list[list[float]],
    child_aabbs_xy_by_id: dict[str, np.ndarray],
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    """Project child centers to the nearest positions inside the parent AABB."""
    parent_bounds = np.asarray(parent_aabb_xy, dtype=float)
    if parent_bounds.shape != (2, 2) or not np.all(np.isfinite(parent_bounds)):
        raise ValueError("Parent AABB must contain two finite XY corners.")
    support_minimum, support_maximum = parent_bounds
    projected_center_xy_by_id: dict[str, list[float]] = {}
    projected_aabbs_xy_by_id: dict[str, np.ndarray] = {}
    for child_id, child_aabb_xy in child_aabbs_xy_by_id.items():
        child_bounds = np.asarray(child_aabb_xy, dtype=float)
        if child_bounds.shape != (2, 2) or not np.all(np.isfinite(child_bounds)):
            raise ValueError(f"Child {child_id!r} has an invalid XY AABB.")
        child_minimum, child_maximum = child_bounds
        half_extents_xy = (child_maximum - child_minimum) / 2.0
        center_xy = (child_minimum + child_maximum) / 2.0
        legal_minimum = support_minimum + half_extents_xy
        legal_maximum = support_maximum - half_extents_xy
        if np.any(legal_minimum > legal_maximum):
            raise ValueError(
                f"Asset {child_id!r} cannot fit inside its support bounds."
            )
        projected_center_xy = np.clip(center_xy, legal_minimum, legal_maximum)
        projected_center_xy_by_id[child_id] = projected_center_xy.tolist()
        projected_aabbs_xy_by_id[child_id] = _translated_aabb_xy(
            aabb_xy=child_bounds,
            delta_xy=projected_center_xy - center_xy,
        )
    return projected_center_xy_by_id, projected_aabbs_xy_by_id


def _translated_aabb_xy(*, aabb_xy: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    """Translate one validated XY AABB by a finite two-dimensional delta."""
    bounds = np.asarray(aabb_xy, dtype=float)
    delta = np.asarray(delta_xy, dtype=float)
    if bounds.shape != (2, 2) or delta.shape != (2,):
        raise ValueError(
            "AABB translation requires two XY corners and a two-value delta."
        )
    if not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(delta)):
        raise ValueError("AABB translation requires finite values.")
    return bounds + delta


def _render_parent_child_aabb_transition(
    *,
    parent_id: str,
    parent_aabb_xy: list[list[float]],
    before_child_aabbs_xy_by_id: dict[str, np.ndarray],
    after_child_aabbs_xy_by_id: dict[str, np.ndarray],
    before_title: str,
    after_title: str,
    output_path: str | Path,
) -> None:
    """Render one parent AABB and its direct-child AABBs before and after a stage."""
    parent_bounds = np.asarray(parent_aabb_xy, dtype=float)
    if parent_bounds.shape != (2, 2) or not np.all(np.isfinite(parent_bounds)):
        raise ValueError("Parent AABB must contain two finite XY corners.")
    if set(before_child_aabbs_xy_by_id) != set(after_child_aabbs_xy_by_id):
        raise ValueError("Parent-child debug AABB states must contain identical IDs.")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=160, constrained_layout=True)
    for axis, child_aabbs_xy_by_id, title, color in (
        (axes[0], before_child_aabbs_xy_by_id, before_title, "tab:blue"),
        (axes[1], after_child_aabbs_xy_by_id, after_title, "tab:green"),
    ):
        _draw_parent_child_aabb_state(
            axis=axis,
            parent_id=parent_id,
            parent_bounds=parent_bounds,
            child_aabbs_xy_by_id=child_aabbs_xy_by_id,
            title=title,
            child_color=color,
        )
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _draw_parent_child_aabb_state(
    *,
    axis: object,
    parent_id: str,
    parent_bounds: np.ndarray,
    child_aabbs_xy_by_id: dict[str, np.ndarray],
    title: str,
    child_color: str,
) -> None:
    """Draw one parent AABB and direct children in a single z-up XY panel."""
    parent_minimum, parent_maximum = parent_bounds
    axis.add_patch(
        Rectangle(
            parent_minimum,
            *(parent_maximum - parent_minimum),
            facecolor="tab:orange",
            edgecolor="saddlebrown",
            alpha=0.25,
            linewidth=2.0,
        )
    )
    axis.text(
        *parent_bounds.mean(axis=0),
        parent_id,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    all_bounds = [parent_bounds]
    for child_id, child_aabb_xy in child_aabbs_xy_by_id.items():
        child_bounds = np.asarray(child_aabb_xy, dtype=float)
        if child_bounds.shape != (2, 2) or not np.all(np.isfinite(child_bounds)):
            raise ValueError(f"Child {child_id!r} has an invalid debug XY AABB.")
        child_minimum, child_maximum = child_bounds
        axis.add_patch(
            Rectangle(
                child_minimum,
                *(child_maximum - child_minimum),
                facecolor=child_color,
                edgecolor=child_color,
                alpha=0.3,
                linewidth=1.5,
            )
        )
        axis.text(
            *child_bounds.mean(axis=0),
            child_id,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        all_bounds.append(child_bounds)
    combined_bounds = np.vstack(all_bounds)
    padding = max(float(np.ptp(combined_bounds, axis=0).max()) * 0.1, 0.02)
    axis.set_xlim(
        combined_bounds[:, 0].min() - padding, combined_bounds[:, 0].max() + padding
    )
    axis.set_ylim(
        combined_bounds[:, 1].min() - padding, combined_bounds[:, 1].max() + padding
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (z-up world)")
    axis.set_ylabel("y (z-up world)")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)


def _table_on_asset_ids(*, scene_graph: SceneGraph, table_id: str) -> set[str]:
    """Return the IDs of assets that directly rest on the table."""
    return {
        node.object_id
        for node in scene_graph.nodes
        if node.parent_id == table_id and node.parent_relation == "on"
    }


def _select_asset_layouts(
    *,
    assets_layout: list[dict[str, object]],
    asset_ids: set[str],
) -> list[dict[str, object]]:
    """Return the requested asset layouts without discarding other scene assets."""
    selected_assets_layout = [
        asset_layout
        for asset_layout in assets_layout
        if asset_layout.get("id") in asset_ids
    ]
    selected_ids = {
        asset_id
        for asset_layout in selected_assets_layout
        if isinstance(asset_id := asset_layout.get("id"), str)
    }
    if selected_ids != asset_ids:
        raise ValueError(
            "Scene graph on-table assets and refined asset layouts do not match."
        )
    return selected_assets_layout


def _layout_matrices_by_id(
    assets_layout: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Return complete layout transforms keyed by their validated asset IDs."""
    matrices_by_id: dict[str, np.ndarray] = {}
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("Each asset layout must contain a non-empty string id.")
        if asset_id in matrices_by_id:
            raise ValueError(f"Asset layouts repeat id {asset_id!r}.")
        matrices_by_id[asset_id] = layout_object_to_transform_matrix(asset_layout)
    return matrices_by_id


def _apply_root_layout_updates_to_descendant_subtrees(
    *,
    scene_graph: SceneGraph,
    assets_layout: list[dict[str, object]],
    updated_root_layouts: list[dict[str, object]],
    root_matrices_before_update: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Replace solved root layouts and propagate each complete pose delta downward."""
    assets_layout_by_id = {
        asset_id: asset_layout
        for asset_layout in assets_layout
        if isinstance(asset_id := asset_layout.get("id"), str) and asset_id
    }
    if len(assets_layout_by_id) != len(assets_layout):
        raise ValueError(
            "Each asset layout must contain one unique non-empty string id."
        )
    updated_root_layouts_by_id = {
        asset_id: asset_layout
        for asset_layout in updated_root_layouts
        if isinstance(asset_id := asset_layout.get("id"), str) and asset_id
    }
    if len(updated_root_layouts_by_id) != len(updated_root_layouts):
        raise ValueError("Updated root layouts must have unique non-empty string ids.")
    if set(updated_root_layouts_by_id) != set(root_matrices_before_update):
        raise ValueError("Updated root layouts do not match the saved root poses.")

    children_by_parent: dict[str, list[str]] = {}
    for node in scene_graph.nodes:
        if node.parent_id is not None:
            children_by_parent.setdefault(node.parent_id, []).append(node.object_id)

    for root_id, root_layout in updated_root_layouts_by_id.items():
        if root_id not in assets_layout_by_id:
            raise ValueError(f"Updated root layout {root_id!r} is not an asset layout.")
        # Compute the world-frame delta from the previous root pose to the updated root pose.
        root_matrix_after_update = layout_object_to_transform_matrix(root_layout)
        root_delta = root_matrix_after_update @ np.linalg.inv(
            root_matrices_before_update[root_id]
        )
        assets_layout_by_id[root_id] = root_layout

        # A parent pose update moves every descendant in the same world frame.
        pending_descendant_ids = list(children_by_parent.get(root_id, []))
        while pending_descendant_ids:
            descendant_id = pending_descendant_ids.pop(0)
            descendant_layout = assets_layout_by_id.get(descendant_id)
            if descendant_layout is None:
                raise ValueError(
                    f"Scene graph descendant {descendant_id!r} has no asset layout."
                )
            assets_layout_by_id[descendant_id] = transform_matrix_to_layout_object(
                descendant_id,
                root_delta @ layout_object_to_transform_matrix(descendant_layout),
            )
            # Add the next generation of descendants to the pending list.
            pending_descendant_ids.extend(children_by_parent.get(descendant_id, []))

    return [
        assets_layout_by_id[str(asset_layout["id"])] for asset_layout in assets_layout
    ]


def _measure_table_and_assets_in_z_up_world(
    *,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray]]:
    """Measure a table mesh and asset AABBs in one shared z-up world frame.

    Scene layouts and SimReady GLBs are y-up.  The support detector and the
    group clamp both operate in z-up world XY, so this conversion is performed
    once here and the exact same measured AABBs are passed to the clamp.
    """
    table_id = table_layout.get("id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("Table layout must contain a non-empty string id.")

    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)
    resolved_geometry_root = Path(geometry_root).expanduser().resolve()

    def _mesh_in_z_up_world(layout_object: dict[str, object]) -> trimesh.Trimesh:
        object_id = layout_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Layout object must contain a non-empty string id.")
        mesh = load_glb_mesh(resolved_geometry_root / f"{object_id}.glb")
        z_up_layout = transform_matrix_to_layout_object(
            object_id,
            y_up_to_z_up_matrix
            @ layout_object_to_transform_matrix(layout_object)
            @ z_up_to_y_up_matrix,
        )
        mesh.apply_transform(y_up_to_z_up_matrix)
        mesh.apply_transform(layout_object_to_transform_matrix(z_up_layout))
        return mesh

    table_world_mesh_z_up = _mesh_in_z_up_world(table_layout)
    asset_aabbs_by_id: dict[str, np.ndarray] = {}
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("Each asset layout must contain a non-empty string id.")
        if asset_id in asset_aabbs_by_id:
            raise ValueError(f"Asset layouts contain duplicate id {asset_id!r}.")
        asset_bounds_xy = _mesh_in_z_up_world(asset_layout).bounds[:, :2]
        asset_aabbs_by_id[asset_id] = np.array(
            [
                [asset_bounds_xy[0, 0], asset_bounds_xy[0, 1]],
                [asset_bounds_xy[1, 0], asset_bounds_xy[0, 1]],
                [asset_bounds_xy[1, 0], asset_bounds_xy[1, 1]],
                [asset_bounds_xy[0, 0], asset_bounds_xy[1, 1]],
            ],
            dtype=float,
        )
    return table_world_mesh_z_up, asset_aabbs_by_id


def _load_layout(layout_path: str | Path) -> list[dict[str, object]]:
    # Load and check the coarse layout JSON file.
    resolved_layout_path = Path(layout_path).expanduser().resolve()
    if not resolved_layout_path.is_file():
        raise FileNotFoundError(f"Layout not found: {resolved_layout_path}")
    try:
        layout = json.loads(resolved_layout_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Layout is not valid JSON: {resolved_layout_path}") from exc
    if not isinstance(layout, list) or not all(
        isinstance(item, dict) for item in layout
    ):
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
        raise ValueError(
            f"Image input must be one of the supported formats: {_SUPPORTED_IMAGE_SUFFIXES}."
        )
    return resolved_image_path
