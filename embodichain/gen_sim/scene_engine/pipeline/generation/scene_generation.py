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

import numpy as np
import trimesh

from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
    GeometryGenerationClient,
)
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp import (
    AssetsGroupSupportClamp,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner import (
    AssetsGroupTableAligner,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer import (
    AssetsSupportLayoutOptimizer,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_gravity_settler import (
    AssetsGravitySettler,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    quaternion_wxyz_to_euler_xyz_degrees,
    transform_matrix_to_layout_object,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
    TableSupportSurfaceDetector,
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
    simready_processor = SimReadyProcessor(
        scene=scene,
        coarse_layout_by_id=coarse_layout_by_id,
        coarse_geometry_root=coarse_geometry_output_root,
        simready_geometry_root=simready_geometry_output_root,
    )
    simready_assets_layout = simready_processor.process_assets()
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
        simready_geometry_output_root=simready_geometry_output_root,  # Contains simready assets and their current coarse layout JSON.
        debug_output_root=debug_output_root,  # Keep the table support surface info + optimized layout info (render with matplotlib) for debugging.
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
    simready_geometry_output_root: str | Path,
    debug_output_root: str | Path,
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

    # 3. Move all assets as one rigid group so its lowest AABB point is 2cm above
    # the table. This preserves the initial relative poses for the later
    # gravity simulation, which can settle individual assets physically.

    group_table_aligner = AssetsGroupTableAligner(
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=simready_geometry_output_root,
    )
    refined_table_layout, refined_assets_layout = group_table_aligner.align()
    if not refined_assets_layout:
        log_info("Scene has no movable assets; skipping support-region clamping.")
        return refined_table_layout, []

    # 4. Detect the actual upward support triangles instead of projecting the
    # entire table mesh to one convex hull.  The result retains concavities
    # (for example, an L-shaped tabletop) and is the only boundary used for
    # placement below.
    (
        table_world_mesh_z_up,
        assets_aabb_2d_z_up_world_corners_by_id,
    ) = _measure_table_and_assets_in_z_up_world(
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=simready_geometry_output_root,
    )
    support_detector = TableSupportSurfaceDetector(
        table_world_mesh=table_world_mesh_z_up,
        debug_output_root=debug_output_root,
    )
    table_support_region = support_detector.detect()
    support_detector.save_support_surface_debug_images()

    # 5. Keep the complete clutter rigid in the table plane.  A successful
    # result applies one shared z-up XY delta to every AABB, so it preserves
    # all existing asset-to-asset relations.  It is *not* an asset packing
    # pass: pre-existing overlap is deliberately left to a later optimizer.
    group_clamp = AssetsGroupSupportClamp(
        support_region=table_support_region.support_polygon,
        assets_aabb_2d_z_up_world_corners_by_id=(
            assets_aabb_2d_z_up_world_corners_by_id
        ),
        assets_layout=refined_assets_layout,
        debug_output_root=debug_output_root,
    )
    refined_assets_layout = group_clamp.clamp()
    group_clamp.save_group_clamp_debug_images()

    # The clamp returns y-up layouts; measure their resulting z-up AABBs again
    # so the following independent optimizer consumes the same world-frame
    # geometry as every other stage.
    _, clamped_assets_aabb_2d_z_up_world_corners_by_id = (
        _measure_table_and_assets_in_z_up_world(
            table_layout=refined_table_layout,
            assets_layout=refined_assets_layout,
            geometry_root=simready_geometry_output_root,
        )
    )

    # 6. Restore the previous pairwise AABB separation stage, but constrain
    # every candidate with the actual support polygon rather than the legacy
    # largest internal rectangle.  Assets may now move independently only as
    # much as needed to remove overlap; every resulting AABB remains on the
    # L-shaped, circular, or otherwise non-convex support region.
    overlap_optimizer = AssetsSupportLayoutOptimizer(
        support_region=table_support_region.support_polygon,
        assets_aabb_2d_z_up_world_corners_by_id=(
            clamped_assets_aabb_2d_z_up_world_corners_by_id
        ),
        assets_layout=refined_assets_layout,
        debug_output_root=debug_output_root,
    )
    # Render this stage separately from the rigid group clamp.  The latter
    # intentionally preserves pre-existing overlaps, while this figure shows
    # whether independent AABB separation actually resolved them.
    refined_assets_layout = overlap_optimizer.optimize()
    overlap_optimizer.save_overlap_optimization_debug_images()

    # 7. Gravity simulation, to let all the assets to be stable and placed well on the table's support surface.
    # Notice that: we do not consider the assets like a bottle, which should be standing on the table but laid down
    # after the simulation.
    gravity_settler = AssetsGravitySettler(
        scene=scene,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=simready_geometry_output_root,
    )
    refined_assets_layout = gravity_settler.settle()

    # Update the scene data structure with the final layout and spatial metadata.
    _update_scene_final_y_up_layout_and_z_up_centers(
        scene=scene,
        table_layout=refined_table_layout,
        assets_layout=refined_assets_layout,
        geometry_root=simready_geometry_output_root,
    )
    return refined_table_layout, refined_assets_layout


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
