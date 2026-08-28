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

from collections import deque
import json
from pathlib import Path
import shutil

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from shapely.geometry import Polygon

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    TABLE_OBJECT_ID,
    SceneGraph,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
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
    quaternion_wxyz_to_euler_xyz_degrees,
    transform_matrix_to_layout_object,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_utils import (
    measure_scene_object_z_up_world_aabb,
    scene_object_y_up_layout,
    update_scene_object_y_up_pose_from_z_up_support,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer import (
    VisualYawOptimizer,
)
from embodichain.utils.logger import log_info

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def generate_scene_and_refine(
    image_path: str | Path,
    output_root: str | Path,
    scene: Scene,
    scene_graph: SceneGraph,
    *,
    geometry_generation_client: GeometryGenerationClient,
    vlm_client: OpenAICompatibleVLM,
) -> Scene:

    resolved_image_path = _validate_image_path(image_path)
    # Validate the scene graph before layout refinement consumes it.
    scene_graph.validate()
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
    _generate_coarse_results_from_masks(
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
    # Only an explicit graph pose description requires a VLM pose adjustment.
    pose_descriptions_by_id = {
        node.object_id: node.pose_description
        for node in scene_graph.nodes
        if node.object_id != TABLE_OBJECT_ID and node.pose_description is not None
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
    simready_assets_layout = simready_processor.process_assets()
    # Replace unreliable coarse rotations with VLM-observed canonical z-up yaw.
    visual_yaws_by_id = _optimize_simready_asset_visual_yaws(
        scene=scene,
        simready_assets_layout=simready_assets_layout,
        coarse_layout_by_id=coarse_layout_by_id,
        vlm_client=vlm_client,
        debug_output_root=debug_output_root / "visual_yaw",
    )
    simready_table_layout = simready_processor.process_table()
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
        simready_geometry_output_root=simready_geometry_output_root,  # Contains simready assets and their current coarse layout JSON.
        debug_output_root=debug_output_root,  # Keep the table support surface info + optimized layout info (render with matplotlib) for debugging.
        z_up_yaws_degrees_by_id=visual_yaws_by_id,
    )

    # Write the Updated scene JSON for debugging.
    (stage_output_root / "scene.json").write_text(
        json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return scene


def _optimize_simready_asset_visual_yaws(
    *,
    scene: Scene,
    simready_assets_layout: list[dict[str, object]],
    coarse_layout_by_id: dict[str, dict[str, object]],
    vlm_client: OpenAICompatibleVLM,
    debug_output_root: str | Path,
) -> dict[str, float]:
    """Query one absolute canonical z-up yaw for every observed SimReady asset."""
    assets_by_id = {asset.id: asset for asset in scene.assets}
    layout_ids = [layout.get("id") for layout in simready_assets_layout]
    if not all(isinstance(layout_id, str) and layout_id for layout_id in layout_ids):
        raise ValueError("Every SimReady asset layout must contain a non-empty id.")
    if len(layout_ids) != len(set(layout_ids)):
        raise ValueError("SimReady asset layouts must have unique ids.")
    if set(layout_ids) != set(assets_by_id):
        raise ValueError("SimReady asset layouts must match the scene asset ids.")

    yaws_degrees_by_id: dict[str, float] = {}
    for asset_layout in simready_assets_layout:
        layout_id = asset_layout["id"]
        assert isinstance(layout_id, str)
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
        object_masks.append(
            (scene_object.id, mask_path)
        )  # id + mask, for avoiding the download glbs order confusion.

    # Sent the request, wait, then save the intermediate results.
    response_data, response_objects = geometry_generation_client.generate_objects(
        image_path=image_path,
        object_masks=object_masks,
        output_root=coarse_geometry_output_root,  # Keep the coarse geometries
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
    if scene.table.support_contour_xy is not None:
        # Move SimReady-local support geometry into the final table-frame position.
        table_center_xy = np.asarray(scene.table.center_xy, dtype=float)
        scene.table.support_contour_xy = [
            (np.asarray(point, dtype=float) + table_center_xy).tolist()
            for point in scene.table.support_contour_xy
        ]
        if scene.table.support_optimization_rect_xy is not None:
            scene.table.support_optimization_rect_xy = [
                (np.asarray(point, dtype=float) + table_center_xy).tolist()
                for point in scene.table.support_optimization_rect_xy
            ]
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
    simready_geometry_output_root: str | Path,
    debug_output_root: str | Path,
    z_up_yaws_degrees_by_id: dict[str, float],
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

    # Table-frame conversion retains the coarse relative positions but can also
    # transfer an inverted coarse-table orientation; replace only that rotation
    # with the canonical SimReady pose and its observed z-up yaw.
    refined_assets_layout = _apply_visual_yaws_to_simready_asset_layouts(
        simready_assets_layout=refined_assets_layout,
        z_up_yaws_degrees_by_id=z_up_yaws_degrees_by_id,
    )

    # 4. Only direct on-table children participate in the table-level layout
    # stages. Their descendants follow each solved root transform until their
    # own parent-surface optimization is introduced in a later BFS pass.
    table_root_ids = _table_on_asset_ids(scene_graph=scene_graph, table_id=table_id)
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )
    if not table_root_layouts:
        log_info("Scene has no on-table assets; skipping table layout refinement.")
        return refined_table_layout, refined_assets_layout

    # Each on-table root needs its own support height; its descendants follow it.
    refined_table_layout, refined_assets_layout = _align_table_roots_individually(
        scene_graph=scene_graph,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        table_root_ids=table_root_ids,
        geometry_root=simready_geometry_output_root,
    )
    # Re-select direct table children after their independent vertical placement.
    table_root_layouts = _select_asset_layouts(
        assets_layout=refined_assets_layout,
        asset_ids=table_root_ids,
    )

    # 5. Reuse support geometry detected during SimReady processing.
    if (
        scene.table is None
        or scene.table.support_contour_xy is None
        or scene.table.support_optimization_rect_xy is None
    ):
        raise ValueError("Scene table has no persisted support geometry.")
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
    assets_by_id = {asset.id: asset for asset in scene.assets}
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

    # 9. The table roots are now stable, so refine each non-table on-parent
    # group in BFS order. Every child begins only after its parent is current.
    refined_assets_layout = _refine_on_children_bfs(
        scene=scene,
        scene_graph=scene_graph,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        table_root_ids=table_root_ids,
        debug_output_root=debug_output_root,
    )

    # Update the scene data structure with the final layout and spatial metadata.
    _update_scene_final_y_up_layout_and_z_up_centers(
        scene=scene,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=simready_geometry_output_root,
    )
    return refined_table_layout, refined_assets_layout


def _align_table_roots_individually(
    *,
    scene_graph: SceneGraph,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    table_root_ids: set[str],
    geometry_root: str | Path,
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

        parent_aabb = measure_scene_object_z_up_world_aabb(scene_object=parent)
        parent_aabb_xy = [
            [float(parent_aabb[0][0]), float(parent_aabb[0][1])],
            [float(parent_aabb[1][0]), float(parent_aabb[1][1])],
        ]
        child_aabbs_xy_by_id = {
            child_id: _scene_object_z_up_aabb_xy(scene_object=assets_by_id[child_id])
            for child_id in child_ids
        }
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
                f"{len(projected_child_ids)} child AABBs into parent {parent_id!r}: "
                f"{projected_child_ids}."
            )
        else:
            log_info(
                f"All direct children are already inside parent {parent_id!r}'s AABB."
            )
        _render_parent_child_aabb_transition(
            parent_id=parent_id,
            parent_aabb_xy=parent_aabb_xy,
            before_child_aabbs_xy_by_id=child_aabbs_xy_by_id,
            after_child_aabbs_xy_by_id=projected_child_aabbs_xy_by_id,
            before_title="Before parent-AABB projection",
            after_title="After parent-AABB projection",
            output_path=(
                Path(debug_output_root)
                / f"parent_{parent_id}_child_aabb_projection_2d.png"
            ),
        )
        child_id_set = set(child_ids)
        # All image-observed children are movable and start from their nearest
        # parent-AABB-valid image seed.
        parent_surface_problem = ParentSurfaceLayoutProblem(
            assets_by_id=assets_by_id,
            child_ids=child_ids,
            child_seed_xy_by_id=child_seed_xy_by_id,
            imported_child_ids=child_id_set,
            fixed_child_xy_by_id={child_id: None for child_id in child_ids},
            parent_aabb_xy=parent_aabb_xy,
            parent_top_z=float(parent_aabb[1][2]),
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

        child_matrices_before_placement = _layout_matrices_by_id(
            _select_asset_layouts(
                assets_layout=refined_assets_layout,
                asset_ids=child_id_set,
            )
        )
        for child_id, solved_xy in solved_child_xy_by_id.items():
            # Place each child above the parent's current top before gravity settles it.
            update_scene_object_y_up_pose_from_z_up_support(
                scene_object=assets_by_id[child_id],
                support_region_z=parent_surface_problem.parent_top_z,
                center_xy=solved_xy,
                clearance_m=0.02,
            )
        refined_assets_layout = _apply_root_layout_updates_to_descendant_subtrees(
            scene_graph=scene_graph,
            assets_layout=refined_assets_layout,
            updated_root_layouts=[
                scene_object_y_up_layout(assets_by_id[child_id])
                for child_id in child_ids
            ],
            root_matrices_before_update=child_matrices_before_placement,
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


def _scene_object_z_up_aabb_xy(*, scene_object: SceneObject) -> np.ndarray:
    """Measure one current scene object's z-up XY AABB bounds."""
    aabb = measure_scene_object_z_up_world_aabb(scene_object=scene_object)
    return np.asarray(
        [
            [float(aabb[0][0]), float(aabb[0][1])],
            [float(aabb[1][0]), float(aabb[1][1])],
        ]
    )


def _project_child_aabb_centers_into_parent_aabb(
    *,
    parent_aabb_xy: list[list[float]],
    child_aabbs_xy_by_id: dict[str, np.ndarray],
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    """Project child AABB centers to their nearest parent-AABB-valid positions."""
    parent_bounds = np.asarray(parent_aabb_xy, dtype=float)
    if parent_bounds.shape != (2, 2) or not np.all(np.isfinite(parent_bounds)):
        raise ValueError("Parent AABB must contain two finite XY corners.")
    parent_minimum, parent_maximum = parent_bounds
    projected_center_xy_by_id: dict[str, list[float]] = {}
    projected_aabbs_xy_by_id: dict[str, np.ndarray] = {}
    for child_id, child_aabb_xy in child_aabbs_xy_by_id.items():
        child_bounds = np.asarray(child_aabb_xy, dtype=float)
        if child_bounds.shape != (2, 2) or not np.all(np.isfinite(child_bounds)):
            raise ValueError(f"Child {child_id!r} has an invalid XY AABB.")
        child_minimum, child_maximum = child_bounds
        half_extents_xy = (child_maximum - child_minimum) / 2.0
        center_xy = (child_minimum + child_maximum) / 2.0
        legal_minimum = parent_minimum + half_extents_xy
        legal_maximum = parent_maximum - half_extents_xy
        if np.any(legal_minimum > legal_maximum):
            raise ValueError(f"Asset {child_id!r} cannot fit inside its parent AABB.")
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
