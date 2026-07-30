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
import re
from typing import Sequence

from embodichain.lab.sim import SimulationManager as _EmbodiSimManager
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
import matplotlib
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import trimesh

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator

_UPRIGHT_CONTAINER_ID_TOKENS = frozenset({"bottle", "can", "jar", "flask", "thermos"})


def quaternion_wxyz_to_euler_xyz_degrees(
    quaternion_wxyz: Sequence[float],
) -> list[float]:
    """Convert a ``[w, x, y, z]`` quaternion to [roll_x, pitch_y, yaw_z] degrees."""
    if len(quaternion_wxyz) != 4:
        raise ValueError("Rotation quaternion must contain exactly four values.")

    w, x, y, z = quaternion_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True).tolist()


def _layout_rotation_to_simulation_euler_xyz_degrees(
    layout_object: dict[str, object],
) -> list[float]:
    """Convert a layout's lowercase-``xyz`` Euler rotation for SimulationManager.

    Scene layouts use ``Rotation.from_euler("xyz", ...)``, whereas
    ``RigidObjectCfg.init_rot`` is interpreted with uppercase ``"XYZ"``.
    Convert through the rotation matrix so both represent exactly the same pose.
    """
    layout_rotation = Rotation.from_euler(
        "xyz",
        _three_floats(layout_object.get("rot"), field_name="rot"),
        degrees=True,
    )
    return layout_rotation.as_euler("XYZ", degrees=True).tolist()


def layout_object_to_transform_matrix(
    layout_object: dict[str, object],
) -> np.ndarray:
    """Return the matrix that maps an object's local coordinates to world coordinates."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = Rotation.from_euler(
        "xyz",
        _three_floats(layout_object.get("rot"), field_name="rot"),
        degrees=True,
    ).as_matrix() @ np.diag(
        _three_floats(layout_object.get("scale"), field_name="scale")
    )
    transform_matrix[:3, 3] = _three_floats(layout_object.get("pos"), field_name="pos")
    return transform_matrix


def transform_matrix_to_layout_object(
    object_id: str,
    transform_matrix: np.ndarray,
) -> dict[str, object]:
    """Convert a non-sheared 4x4 transform matrix into one layout object."""
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Layout object id must be a non-empty string.")
    matrix = np.asarray(transform_matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Transform matrix must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Transform matrix must be affine.")

    linear_matrix = matrix[:3, :3]
    scale = np.linalg.norm(linear_matrix, axis=0)
    if np.any(scale <= 1e-8):
        raise ValueError("Transform matrix has a zero scale axis.")
    rotation_matrix = linear_matrix / scale
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-6):
        raise ValueError("Transform matrix contains shear and cannot be decomposed.")
    if np.linalg.det(rotation_matrix) <= 0:
        raise ValueError(
            "Transform matrix contains a reflection and cannot be decomposed."
        )

    return {
        "id": object_id,
        "rot": Rotation.from_matrix(rotation_matrix)
        .as_euler("xyz", degrees=True)
        .tolist(),
        "pos": matrix[:3, 3].tolist(),
        "scale": scale.tolist(),
    }


def load_glb_mesh(glb_path: str | Path) -> trimesh.Trimesh:
    """Load one GLB as a single trimesh mesh."""
    resolved_glb_path = Path(glb_path).expanduser().resolve()
    if not resolved_glb_path.is_file():
        raise FileNotFoundError(f"GLB geometry not found: {resolved_glb_path}")
    loaded_mesh = trimesh.load(resolved_glb_path, process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
        return loaded_mesh.dump(concatenate=True)
    if isinstance(loaded_mesh, trimesh.Trimesh):
        return loaded_mesh
    raise ValueError(f"GLB geometry is not a mesh: {resolved_glb_path}")


def align_assets_to_table_aabb_top(
    *,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
    clearance: float = 0.02,  # 2cm.
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Place assets above a table using temporary z-up AABB height calculations.

    Input and output layouts use y-up, matching the GLBs on disk. The geometry
    and layouts are converted to z-up only while measuring and changing height.

    Notice:
        - The refinement pipeline currently uses the group version so it preserves
        the assets' relative vertical arrangement before gravity simulation.
    """
    if clearance < 0:
        raise ValueError("Table clearance must be non-negative.")

    # Prepare y-up and z-up conversion matrices.
    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)

    z_up_table_layout = _convert_layout_coordinate_system(
        table_layout,
        source_to_target_matrix=y_up_to_z_up_matrix,
    )
    z_up_assets_layout = [
        _convert_layout_coordinate_system(
            asset_layout,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        for asset_layout in assets_layout
    ]

    # Get the table's top z position in z-up coordinates, and add the clearance to it.
    resolved_geometry_root = Path(geometry_root).expanduser().resolve()
    table_mesh = load_glb_mesh(
        resolved_geometry_root / f"{z_up_table_layout['id']}.glb"
    )
    table_mesh.apply_transform(y_up_to_z_up_matrix)
    table_mesh.apply_transform(layout_object_to_transform_matrix(z_up_table_layout))
    target_asset_bottom_z = table_mesh.bounds[1, 2] + clearance

    # Iterate through each asset and adjust its z position to sit above the table.
    for asset_layout in z_up_assets_layout:
        asset_mesh = load_glb_mesh(resolved_geometry_root / f"{asset_layout['id']}.glb")
        asset_mesh.apply_transform(y_up_to_z_up_matrix)
        asset_mesh.apply_transform(layout_object_to_transform_matrix(asset_layout))
        asset_bottom_z = asset_mesh.bounds[0, 2]
        asset_layout["pos"][2] += target_asset_bottom_z - asset_bottom_z

    return (
        _convert_layout_coordinate_system(
            z_up_table_layout,
            source_to_target_matrix=z_up_to_y_up_matrix,
        ),
        [
            _convert_layout_coordinate_system(
                asset_layout,
                source_to_target_matrix=z_up_to_y_up_matrix,
            )
            for asset_layout in z_up_assets_layout
        ],
    )


def align_assets_group_to_table_aabb_top(
    *,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
    clearance: float = 0.02,  # 2cm.
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Place all assets as one rigid vertical group above the table.

    Input and output layouts use y-up, matching the GLBs on disk.  The group
    is temporarily measured in z-up coordinates and every asset receives the
    same vertical translation.  This preserves all asset-to-asset relative
    poses;
    """
    if clearance < 0:
        raise ValueError("Table clearance must be non-negative.")
    if not assets_layout:
        return table_layout, []

    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)

    z_up_table_layout = _convert_layout_coordinate_system(
        table_layout,
        source_to_target_matrix=y_up_to_z_up_matrix,
    )
    z_up_assets_layout = [
        _convert_layout_coordinate_system(
            asset_layout,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        for asset_layout in assets_layout
    ]

    resolved_geometry_root = Path(geometry_root).expanduser().resolve()
    table_mesh = load_glb_mesh(
        resolved_geometry_root / f"{z_up_table_layout['id']}.glb"
    )
    table_mesh.apply_transform(y_up_to_z_up_matrix)
    table_mesh.apply_transform(layout_object_to_transform_matrix(z_up_table_layout))
    target_group_bottom_z = table_mesh.bounds[1, 2] + clearance

    group_bottom_z = np.inf
    for asset_layout in z_up_assets_layout:
        asset_mesh = load_glb_mesh(resolved_geometry_root / f"{asset_layout['id']}.glb")
        asset_mesh.apply_transform(y_up_to_z_up_matrix)
        asset_mesh.apply_transform(layout_object_to_transform_matrix(asset_layout))
        group_bottom_z = min(
            group_bottom_z, float(asset_mesh.bounds[0, 2])
        )  # Find the lowest z among all the assets.

    group_vertical_translation_z = target_group_bottom_z - group_bottom_z
    for asset_layout in z_up_assets_layout:
        asset_layout["pos"][2] += group_vertical_translation_z

    return (
        _convert_layout_coordinate_system(
            z_up_table_layout,
            source_to_target_matrix=z_up_to_y_up_matrix,
        ),
        [
            _convert_layout_coordinate_system(
                asset_layout,
                source_to_target_matrix=z_up_to_y_up_matrix,
            )
            for asset_layout in z_up_assets_layout
        ],
    )


def _prepare_gravity_sim_body(
    *,
    layout_object: dict[str, object],
    geometry_root: Path,
    y_up_to_z_up_matrix: np.ndarray,
) -> tuple[
    Path,
    trimesh.Trimesh,
    dict[str, object],
    list[float],
    list[float],
]:
    """Load one y-up GLB and derive its z-up rigid pose for gravity simulation."""
    object_id = str(layout_object["id"])
    source_mesh_path = geometry_root / f"{object_id}.glb"
    source_mesh = load_glb_mesh(source_mesh_path)
    z_up_layout = _convert_layout_coordinate_system(
        layout_object,
        source_to_target_matrix=y_up_to_z_up_matrix,
    )
    y_up_scale = _three_floats(layout_object.get("scale"), field_name="scale")
    z_up_scale = _three_floats(z_up_layout.get("scale"), field_name="scale")
    z_up_rigid_layout = {
        "id": object_id,
        "rot": _three_floats(z_up_layout.get("rot"), field_name="rot"),
        "pos": _three_floats(z_up_layout.get("pos"), field_name="pos"),
        "scale": [1.0, 1.0, 1.0],
    }
    return (
        source_mesh_path,
        source_mesh,
        z_up_rigid_layout,
        y_up_scale,
        z_up_scale,
    )


def _mesh_to_z_up_world_for_aabb(
    *,
    y_up_mesh: trimesh.Trimesh,
    z_up_rigid_layout: dict[str, object],
    z_up_scale: Sequence[float],
    y_up_to_z_up_matrix: np.ndarray,
) -> trimesh.Trimesh:
    """Transform a y-up mesh into its z-up world pose for AABB measurement."""
    y_up_mesh.apply_transform(y_up_to_z_up_matrix)
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] = np.diag(z_up_scale)
    y_up_mesh.apply_transform(scale_matrix)
    y_up_mesh.apply_transform(layout_object_to_transform_matrix(z_up_rigid_layout))
    return y_up_mesh


def gravity_settle_assets_on_table(
    *,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
    clearance: float = 0.02,
    settle_steps: int = 300,
    physics_dt: float = 1.0 / 100.0,
    sim_device: str = "cpu",
    max_convex_hull_num: int = 32,
) -> list[dict[str, object]]:
    """Settle all assets together on a static table with z-up gravity.

    Layouts and source GLBs are y-up. The simulator automatically converts its
    y-up GLB inputs to z-up, while its gravity poses are expressed in z-up.
    This function therefore keeps the source meshes y-up and converts only the
    layout poses for measurement and simulation. Before all dynamic assets are
    added to one simulation, each asset's own lowest AABB z is placed
    ``clearance`` above the table AABB top. The final rigid-body poses are
    converted back to y-up layouts, with their original scales preserved.
    """

    # Check.
    if clearance < 0.0:
        raise ValueError("Gravity-settle clearance must be non-negative.")
    if settle_steps <= 0:
        raise ValueError("Gravity-settle steps must be positive.")
    if physics_dt <= 0.0:
        raise ValueError("Gravity-settle physics_dt must be positive.")
    if max_convex_hull_num <= 0:
        raise ValueError("Gravity-settle max_convex_hull_num must be positive.")
    if not assets_layout:
        return []

    table_id = table_layout.get("id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("Table layout must contain a non-empty string id.")
    asset_ids: set[str] = set()
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("Each asset layout must contain a non-empty string id.")
        if asset_id in asset_ids:
            raise ValueError(f"Asset layouts contain duplicate id {asset_id!r}.")
        asset_ids.add(asset_id)

    # The source GLBs/layouts are y-up, while the gravity service uses z-up.
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

    (
        table_mesh_path,
        table_mesh,
        table_rigid_layout,
        table_y_up_scale,
        table_z_up_scale,
    ) = _prepare_gravity_sim_body(
        layout_object=table_layout,
        geometry_root=resolved_geometry_root,
        y_up_to_z_up_matrix=y_up_to_z_up_matrix,
    )
    # Match the simulator's automatic y-up-GLB conversion while measuring the
    # physical z-up table top.
    table_world_mesh = _mesh_to_z_up_world_for_aabb(
        y_up_mesh=table_mesh,
        z_up_rigid_layout=table_rigid_layout,
        z_up_scale=table_z_up_scale,
        y_up_to_z_up_matrix=y_up_to_z_up_matrix,
    )
    table_top_z = float(table_world_mesh.bounds[1, 2])

    prepared_assets: dict[str, dict[str, object]] = {}
    for asset_layout in assets_layout:
        asset_id = str(asset_layout["id"])
        (
            asset_mesh_path,
            asset_mesh,
            asset_rigid_layout,
            asset_y_up_scale,
            asset_z_up_scale,
        ) = _prepare_gravity_sim_body(
            layout_object=asset_layout,
            geometry_root=resolved_geometry_root,
            y_up_to_z_up_matrix=y_up_to_z_up_matrix,
        )
        asset_world_mesh = _mesh_to_z_up_world_for_aabb(
            y_up_mesh=asset_mesh,
            z_up_rigid_layout=asset_rigid_layout,
            z_up_scale=asset_z_up_scale,
            y_up_to_z_up_matrix=y_up_to_z_up_matrix,
        )
        asset_bottom_z = float(asset_world_mesh.bounds[0, 2])
        asset_rigid_layout["pos"][2] += table_top_z + clearance - asset_bottom_z
        prepared_assets[asset_id] = {
            "mesh_path": asset_mesh_path,
            "rigid_layout": asset_rigid_layout,
            "y_up_scale": asset_y_up_scale,
            "z_up_scale": asset_z_up_scale,
        }

    sim = _EmbodiSimManager(
        SimulationManagerCfg(
            headless=True,
            physics_dt=physics_dt,
            sim_device=sim_device,
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid=table_id,
                shape=MeshCfg(fpath=str(table_mesh_path)),
                init_pos=tuple(table_rigid_layout["pos"]),
                init_rot=tuple(
                    _layout_rotation_to_simulation_euler_xyz_degrees(table_rigid_layout)
                ),
                body_scale=tuple(table_y_up_scale),
                body_type="static",
                max_convex_hull_num=max_convex_hull_num,
                acd_method="vhacd",  # Use vhacd by default.
            )
        )
        simulated_assets: dict[str, object] = {}
        for asset_id, asset_info in prepared_assets.items():
            rigid_layout = asset_info["rigid_layout"]
            simulated_assets[asset_id] = sim.add_rigid_object(
                RigidObjectCfg(
                    uid=asset_id,
                    shape=MeshCfg(fpath=str(asset_info["mesh_path"])),
                    init_pos=tuple(rigid_layout["pos"]),
                    init_rot=tuple(
                        _layout_rotation_to_simulation_euler_xyz_degrees(rigid_layout)
                    ),
                    body_scale=tuple(asset_info["y_up_scale"]),
                    body_type="dynamic",
                    max_convex_hull_num=max_convex_hull_num,
                    acd_method="vhacd",  # Use vhacd by default.
                )
            )

        # All assets share this one simulation, so they can collide with the
        # table and with one another while settling.
        sim.update(step=settle_steps)

        settled_layout_by_id: dict[str, dict[str, object]] = {}
        for asset_id, simulated_asset in simulated_assets.items():
            final_rigid_pose_z_up = np.asarray(
                simulated_asset.get_local_pose(to_matrix=True)[0]
                .detach()
                .cpu()
                .numpy(),
                dtype=float,
            )
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] = np.diag(prepared_assets[asset_id]["z_up_scale"])
            final_z_up_layout_matrix = final_rigid_pose_z_up @ scale_matrix
            settled_layout_by_id[asset_id] = transform_matrix_to_layout_object(
                asset_id,
                z_up_to_y_up_matrix @ final_z_up_layout_matrix @ y_up_to_z_up_matrix,
            )
    finally:
        sim.destroy(exit_process=False)
        _EmbodiSimManager.flush_cleanup_queue()

    settled_assets_layout = [
        settled_layout_by_id[str(asset_layout["id"])] for asset_layout in assets_layout
    ]
    return settled_assets_layout


def heuristic_table_support_surface(
    *,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: str | Path,
    debug_output_root: str | Path,
) -> tuple[
    list[list[float]],
    dict[str, list[list[float]]],
    dict[str, list[list[float]] | list[list[int]]],
]:
    """Return the table support boundary, asset AABBs, and table 2D mesh.

    The input table layout and its GLB use y-up. This function will convert
    both to temporary z-up coordinates before extracting the support surface.
    The returned convex-hull boundary is ordered counter-clockwise in the z-up
    world x-y plane. Each projected rectangle is keyed by asset id and contains
    four counter-clockwise x-y corners. The projected table mesh contains 2D
    vertices and triangle faces, so later stages do not need to recompute it.
    """
    table_id = table_layout.get("id")
    if not isinstance(table_id, str) or not table_id:
        raise ValueError("Table layout must contain a non-empty string id.")

    resolved_geometry_root = Path(geometry_root).expanduser().resolve()
    table_glb_path = resolved_geometry_root / f"{table_id}.glb"
    if not table_glb_path.is_file():
        raise FileNotFoundError(f"Table geometry not found: {table_glb_path}")

    resolved_debug_output_root = Path(debug_output_root).expanduser().resolve()
    resolved_debug_output_root.mkdir(parents=True, exist_ok=True)

    # 1. Load the y-up table GLB, convert its vertices and layout to z-up, then
    #    apply the z-up world transform to obtain the table world geometry.
    y_up_to_z_up_matrix = np.eye(4)
    y_up_to_z_up_matrix[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    z_up_table_layout = _convert_layout_coordinate_system(
        table_layout,
        source_to_target_matrix=y_up_to_z_up_matrix,
    )
    table_world_mesh = load_glb_mesh(table_glb_path)
    table_world_mesh.apply_transform(y_up_to_z_up_matrix)
    table_world_mesh.apply_transform(
        layout_object_to_transform_matrix(z_up_table_layout)
    )

    # Prepare every asset's z-up world x-y AABB for the debug rendering.
    # To check if any asset's AABB is outside the table's support surface.
    assets_2d_aabbs: list[tuple[str, np.ndarray]] = (
        []
    )  # id + 2D AABB infos in z-up world x-y plane.
    projected_rectangles_by_id: dict[str, list[list[float]]] = {}
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("Each asset layout must contain a non-empty string id.")
        asset_glb_path = resolved_geometry_root / f"{asset_id}.glb"
        if not asset_glb_path.is_file():
            raise FileNotFoundError(f"Asset geometry not found: {asset_glb_path}")

        z_up_asset_layout = _convert_layout_coordinate_system(
            asset_layout,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        asset_world_mesh = load_glb_mesh(asset_glb_path)
        asset_world_mesh.apply_transform(y_up_to_z_up_matrix)
        asset_world_mesh.apply_transform(
            layout_object_to_transform_matrix(z_up_asset_layout)
        )
        asset_bounds_xy = asset_world_mesh.bounds[:, :2]
        asset_2d_aabb = np.array(
            [
                [asset_bounds_xy[0, 0], asset_bounds_xy[0, 1]],
                [asset_bounds_xy[1, 0], asset_bounds_xy[0, 1]],
                [asset_bounds_xy[1, 0], asset_bounds_xy[1, 1]],
                [asset_bounds_xy[0, 0], asset_bounds_xy[1, 1]],
            ]
        )
        assets_2d_aabbs.append((asset_id, asset_2d_aabb))
        projected_rectangles_by_id[asset_id] = asset_2d_aabb.tolist()

    # 2. Project every table triangle into the z-up world's x-y plane.
    if len(table_world_mesh.vertices) < 3 or len(table_world_mesh.faces) == 0:
        raise ValueError("Table geometry must contain at least one triangle.")
    projected_vertices = table_world_mesh.vertices[
        :, :2
    ]  # Ignore z, for we wanna get the x-y plane projection.
    try:
        projected_hull = ConvexHull(
            projected_vertices
        )  # Compute the convex hull for the 2D projection.
        # Notice that: for the L-shape table, this will return a bad result.
    except QhullError as exc:
        raise ValueError("Table's x-y projection is degenerate.") from exc
    support_region_boundary = projected_vertices[projected_hull.vertices]

    projected_triangles = projected_vertices[table_world_mesh.faces]
    # 3. Render the full projected mesh and its outer boundary for debugging.
    _render_table_xy_projection(
        projected_triangles=projected_triangles,  # All the projection triangles, draw with blue color.
        support_region_boundary=support_region_boundary,  # The convex hull boundary, draw with red line.
        assets_2d_aabbs=assets_2d_aabbs,  # Render together for debugging.
        table_id=table_id,
        output_path=resolved_debug_output_root / "table_xy_projection.png",
    )

    # 4. Return the convex-hull boundary, each asset's AABB, and the table 2D mesh.
    table_projected_mesh_2d: dict[str, list[list[float]] | list[list[int]]] = {
        "vertices": projected_vertices.tolist(),
        "faces": table_world_mesh.faces.tolist(),
    }
    return (
        support_region_boundary.tolist(),
        projected_rectangles_by_id,
        table_projected_mesh_2d,
    )


def _render_table_xy_projection(
    *,
    projected_triangles: np.ndarray,
    support_region_boundary: np.ndarray,
    assets_2d_aabbs: list[tuple[str, np.ndarray]],
    largest_internal_rectangle: np.ndarray | None = None,
    table_id: str,
    output_path: str | Path,
) -> Path:
    """Render a table's z-up world x-y projection with axes and tick marks."""
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(8, 8), dpi=160)
    axes.add_collection(
        PolyCollection(
            projected_triangles,
            facecolor="steelblue",
            alpha=0.08,
            edgecolor="none",
        )
    )
    closed_boundary = np.vstack(
        [support_region_boundary, support_region_boundary[0]]
    )  # Close the convex hull boundary by adding the first point to the end of the array.
    axes.plot(
        closed_boundary[:, 0],
        closed_boundary[:, 1],
        color="crimson",
        linewidth=2.0,
        label="2D convex-hull boundary",
    )
    if largest_internal_rectangle is not None:
        closed_largest_internal_rectangle = np.vstack(
            [largest_internal_rectangle, largest_internal_rectangle[0]]
        )
        axes.fill(
            closed_largest_internal_rectangle[:, 0],
            closed_largest_internal_rectangle[:, 1],
            color="seagreen",
            alpha=0.25,
            label="largest internal x-y AABB",
        )
        axes.plot(
            closed_largest_internal_rectangle[:, 0],
            closed_largest_internal_rectangle[:, 1],
            color="seagreen",
            linewidth=2.0,
        )
    # Render each asset's 2D AABB with its own id for debugging.
    for index, (asset_id, asset_aabb) in enumerate(assets_2d_aabbs):
        closed_asset_aabb = np.vstack([asset_aabb, asset_aabb[0]])
        axes.fill(
            closed_asset_aabb[:, 0],
            closed_asset_aabb[:, 1],
            color="darkorange",
            alpha=0.16,
            label="asset 2D AABB" if index == 0 else None,
        )
        axes.plot(
            closed_asset_aabb[:, 0],
            closed_asset_aabb[:, 1],
            color="darkorange",
            linewidth=1.5,
        )
        asset_aabb_center = asset_aabb.mean(axis=0)
        axes.text(
            asset_aabb_center[0],
            asset_aabb_center[1],
            asset_id,
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axes.scatter(
        0.0,
        0.0,
        color="black",
        marker="+",
        s=100,
        label="world origin",
    )
    axes.update_datalim(np.array([[0.0, 0.0]]))
    axes.autoscale_view()
    axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    axes.axvline(0.0, color="black", linewidth=0.8, alpha=0.55)

    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    axes.annotate(
        "+x",
        xy=(x_max, 0.0),
        xytext=(x_max - (x_max - x_min) * 0.12, (y_max - y_min) * 0.03),
        arrowprops={"arrowstyle": "->", "color": "black"},
        ha="right",
        va="bottom",
    )
    axes.annotate(
        "+y",
        xy=(0.0, y_max),
        xytext=((x_max - x_min) * 0.03, y_max - (y_max - y_min) * 0.12),
        arrowprops={"arrowstyle": "->", "color": "black"},
        ha="left",
        va="top",
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x (z-up world)")
    axes.set_ylabel("y (z-up world)")
    axes.set_title(f"Table 2D Projection: {table_id}")
    axes.xaxis.set_major_locator(MaxNLocator(nbins=8))
    axes.yaxis.set_major_locator(MaxNLocator(nbins=8))
    axes.tick_params(axis="both", which="major", labelsize=9)
    axes.legend(loc="best")
    axes.grid(True, alpha=0.25)
    figure.savefig(resolved_output_path, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path


def heuristic_table_largest_internal_rectangle(
    *,
    table_support_surface_2d_z_up_world_boundary: Sequence[Sequence[float]],
    assets_aabb_2d_z_up_world_corners_by_id: dict[str, list[list[float]]],
    table_mesh_2d_z_up_world_projection: dict[str, list[list[float]] | list[list[int]]],
    debug_output_root: str | Path,
) -> list[list[float]]:
    """Return the largest centered, x/y-aligned AABB with the table AABB aspect ratio.

    The table boundary is used to binary-search a safe uniform scale. Asset
    AABBs and the table mesh projection are only reused for debug rendering.
    """
    # The boundary is already in the z-up world x-y plane.
    boundary = np.asarray(table_support_surface_2d_z_up_world_boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError(
            "Table support-region boundary must contain at least three 2D points."
        )
    if not np.all(np.isfinite(boundary)):
        raise ValueError(
            "Table support-region boundary must contain only finite values."
        )
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    # The support-surface stage has already returned this as a counter-clockwise
    # convex-hull boundary, so do not compute another convex hull here.
    convex_boundary = boundary

    boundary_min = convex_boundary.min(axis=0)
    boundary_max = convex_boundary.max(axis=0)
    # Build the smallest origin-centered 2D AABB that contains the red boundary.
    boundary_half_extents = np.maximum(
        np.abs(boundary_min),
        np.abs(boundary_max),
    )
    boundary_size = boundary_half_extents * 2.0
    if np.any(boundary_size <= 0):
        raise ValueError(
            "Table support-region boundary must have non-zero width and height."
        )

    # Keep the internal rectangle centered at the table/world origin.
    # rectangle_center = convex_boundary.mean(axis=0) # The mean is not always 0,0.
    rectangle_center = np.array([0.0, 0.0])
    coordinate_scale = max(float(boundary_size.max()), 1.0)
    containment_tolerance = coordinate_scale * 1e-8
    edge_starts = convex_boundary
    edge_vectors = np.roll(convex_boundary, -1, axis=0) - edge_starts

    def _rectangle_at_scale(scale: float) -> np.ndarray:
        half_extents = boundary_size * scale / 2.0
        return np.array(
            [
                rectangle_center - half_extents,
                rectangle_center + [half_extents[0], -half_extents[1]],
                rectangle_center + half_extents,
                rectangle_center + [-half_extents[0], half_extents[1]],
            ]
        )

    def _is_inside_boundary(rectangle: np.ndarray) -> bool:
        corner_offsets = rectangle[None, :, :] - edge_starts[:, None, :]
        cross_products = (
            edge_vectors[:, 0, None] * corner_offsets[:, :, 1]
            - edge_vectors[:, 1, None] * corner_offsets[:, :, 0]
        )
        return bool(np.all(cross_products >= -containment_tolerance))

    # Binary-search the largest safe uniform scale in [0, 1].
    largest_safe_scale = 0.0
    smallest_unsafe_scale = 1.0
    for _ in range(32):
        candidate_scale = (largest_safe_scale + smallest_unsafe_scale) / 2.0
        if _is_inside_boundary(_rectangle_at_scale(candidate_scale)):
            largest_safe_scale = candidate_scale
        else:
            smallest_unsafe_scale = candidate_scale
    if largest_safe_scale <= 1e-8:
        raise ValueError("Table support-region boundary has no usable interior area.")
    largest_internal_rectangle = _rectangle_at_scale(largest_safe_scale)

    # These values were created by heuristic_table_support_surface in this
    # pipeline, so convert them for rendering without validating them again.
    projected_vertices = np.asarray(
        table_mesh_2d_z_up_world_projection["vertices"], dtype=float
    )
    projected_faces = np.asarray(
        table_mesh_2d_z_up_world_projection["faces"], dtype=int
    )
    assets_2d_aabbs = [
        (asset_id, np.asarray(asset_aabb, dtype=float))
        for asset_id, asset_aabb in assets_aabb_2d_z_up_world_corners_by_id.items()
    ]
    _render_table_xy_projection(
        projected_triangles=projected_vertices[projected_faces],
        support_region_boundary=convex_boundary,
        assets_2d_aabbs=assets_2d_aabbs,
        largest_internal_rectangle=largest_internal_rectangle,
        table_id="table",
        output_path=(
            Path(debug_output_root).expanduser().resolve()
            / "table_largest_internal_rectangle.png"
        ),
    )
    return largest_internal_rectangle.tolist()


def make_assets_2d_aabb_inside_table_largest_rectangle(
    *,
    table_id: str,
    table_support_surface_2d_z_up_world_boundary: Sequence[Sequence[float]],
    table_mesh_2d_z_up_world_projection: dict[str, list[list[float]] | list[list[int]]],
    table_largest_internal_rectangle_2d_z_up_world: Sequence[Sequence[float]],
    assets_aabb_2d_z_up_world_corners_by_id: dict[str, list[list[float]]],
    debug_output_root: str | Path,
    assets_layout: list[dict[str, object]],
    boundary_margin: float = 1e-6,
    aabb_clearance: float = 1e-6,
) -> list[dict[str, object]]:
    """Center the asset AABB union, then pack the AABBs inside the table.

    All AABB inputs are in the z-up world's x-y plane. Layouts remain y-up, so
    a z-up planar offset ``(dx, dy)`` is written back as ``pos.x += dx`` and
    ``pos.z -= dy``. ``boundary_margin`` and ``aabb_clearance`` are deliberately
    near zero by default, but remain explicit so callers can request a gap.
    The table projection inputs are used only to render the final debug image.
    """
    if not assets_layout:
        return []

    # Get the table's largest internal rectangle's min and max corners in the z-up world x-y plane.
    rectangle_min, rectangle_max = _aabb_2d_bounds_from_corners(
        table_largest_internal_rectangle_2d_z_up_world,
        name="Table largest internal rectangle",
        require_nonzero_extent=True,
    )

    # Prepare asset layouts by id for validation and later lookup.
    layout_by_id: dict[str, dict[str, object]] = {}
    for asset_layout in assets_layout:
        asset_id = asset_layout.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("Each asset layout must contain a non-empty string id.")
        if asset_id in layout_by_id:
            raise ValueError(f"Asset layouts contain duplicate id {asset_id!r}.")
        layout_by_id[asset_id] = asset_layout

    aabb_ids = set(assets_aabb_2d_z_up_world_corners_by_id)
    layout_ids = set(layout_by_id)
    if aabb_ids != layout_ids:
        missing_aabbs = sorted(layout_ids - aabb_ids)
        missing_layouts = sorted(aabb_ids - layout_ids)
        raise ValueError(
            "Asset layouts and 2D AABBs must have the same ids: "
            f"missing AABBs={missing_aabbs}, missing layouts={missing_layouts}."
        )

    aabb_corners_by_id: dict[str, np.ndarray] = {}
    aabb_bounds_by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for asset_id, corners in assets_aabb_2d_z_up_world_corners_by_id.items():
        corner_array = np.asarray(corners, dtype=float)
        asset_min, asset_max = _aabb_2d_bounds_from_corners(
            corner_array,
            name=f"Asset {asset_id!r} 2D AABB",
            require_nonzero_extent=False,
        )
        aabb_corners_by_id[asset_id] = corner_array
        aabb_bounds_by_id[asset_id] = (asset_min, asset_max)

    # Union all the assets' AABBs to find the center of the group, then offset all AABBs to be centered at the origin.
    # A heuristic implementation.
    union_min = np.min(
        np.stack([bounds[0] for bounds in aabb_bounds_by_id.values()]), axis=0
    )
    union_max = np.max(
        np.stack([bounds[1] for bounds in aabb_bounds_by_id.values()]), axis=0
    )
    union_center = (union_min + union_max) / 2.0
    union_to_origin_offset = -union_center
    # Center all the AABBs by subtracting the union center from each corner.
    centered_aabb_corners_by_id = {
        asset_id: corners + union_to_origin_offset
        for asset_id, corners in aabb_corners_by_id.items()
    }
    # Optimize all the asset AABBs:
    # 1. Do not collide with each other.
    # 2. Inside the table's region.
    optimizer_offsets_by_id = _optimize_assets_2d_aabbs_in_rectangle(
        rectangle_min=rectangle_min,
        rectangle_max=rectangle_max,
        aabb_corners_by_id=centered_aabb_corners_by_id,
        boundary_margin=boundary_margin,
        aabb_clearance=aabb_clearance,
    )

    # Render the final packed AABBs using the original table support-surface
    # projection rather than approximating the table with its internal rectangle.
    projected_vertices = np.asarray(
        table_mesh_2d_z_up_world_projection["vertices"], dtype=float
    )
    projected_faces = np.asarray(
        table_mesh_2d_z_up_world_projection["faces"], dtype=int
    )
    final_assets_2d_aabbs = [
        (
            asset_id,
            centered_aabb_corners_by_id[asset_id] + optimizer_offsets_by_id[asset_id],
        )
        for asset_id in sorted(centered_aabb_corners_by_id)
    ]
    _render_table_xy_projection(
        projected_triangles=projected_vertices[projected_faces],
        support_region_boundary=np.asarray(
            table_support_surface_2d_z_up_world_boundary,
            dtype=float,
        ),
        assets_2d_aabbs=final_assets_2d_aabbs,
        largest_internal_rectangle=np.asarray(
            table_largest_internal_rectangle_2d_z_up_world,
            dtype=float,
        ),
        table_id=table_id,
        output_path=(
            Path(debug_output_root).expanduser().resolve()
            / "assets_2d_aabb_optimization.png"
        ),
    )

    # Update each asset layout's planar position only: z-up (x, y) maps to
    # y-up (x, -z), so update layout pos.x and pos.z while preserving pos.y,
    # rotation, and scale.
    refined_assets_layout: list[dict[str, object]] = []
    for asset_layout in assets_layout:
        asset_id = str(asset_layout["id"])
        final_z_up_xy_offset = (
            union_to_origin_offset + optimizer_offsets_by_id[asset_id]
        )
        refined_layout = dict(asset_layout)
        refined_pos = _three_floats(asset_layout.get("pos"), field_name="pos")
        refined_pos[0] += float(final_z_up_xy_offset[0])
        refined_pos[2] -= float(final_z_up_xy_offset[1])
        refined_layout["pos"] = refined_pos
        refined_assets_layout.append(refined_layout)

    return refined_assets_layout


def _aabb_2d_bounds_from_corners(
    corners: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    require_nonzero_extent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2D AABB corners and return their minimum and maximum corners."""
    corner_array = np.asarray(corners, dtype=float)
    if corner_array.shape != (4, 2) or not np.all(np.isfinite(corner_array)):
        raise ValueError(f"{name} must be four finite [x, y] corners.")
    minimum = corner_array.min(axis=0)
    maximum = corner_array.max(axis=0)
    if require_nonzero_extent and np.any(maximum <= minimum):
        raise ValueError(f"{name} must have non-zero width and height.")
    return minimum, maximum


def _aabb_pair_overlap_depths(
    *,
    current_mins: np.ndarray,
    current_maxs: np.ndarray,
    first_index: int,
    second_index: int,
    aabb_clearance: float,
    tolerance: float,
) -> tuple[float, float] | None:
    """Return x/y overlap depths, or ``None`` when two AABBs do not overlap."""
    overlap_x = (
        min(current_maxs[first_index, 0], current_maxs[second_index, 0])
        - max(current_mins[first_index, 0], current_mins[second_index, 0])
        + aabb_clearance
    )
    overlap_y = (
        min(current_maxs[first_index, 1], current_maxs[second_index, 1])
        - max(current_mins[first_index, 1], current_mins[second_index, 1])
        + aabb_clearance
    )
    if overlap_x <= tolerance or overlap_y <= tolerance:
        return None
    return overlap_x, overlap_y


def _find_overlapping_2d_aabb_pairs(
    *,
    current_mins: np.ndarray,
    current_maxs: np.ndarray,
    aabb_clearance: float,
    tolerance: float,
) -> list[tuple[float, int, int]]:
    """Return overlapping pairs, most constrained pair first."""
    overlaps: list[tuple[float, int, int]] = []
    for first_index in range(len(current_mins)):
        for second_index in range(first_index + 1, len(current_mins)):
            overlap_depths = _aabb_pair_overlap_depths(
                current_mins=current_mins,
                current_maxs=current_maxs,
                first_index=first_index,
                second_index=second_index,
                aabb_clearance=aabb_clearance,
                tolerance=tolerance,
            )
            if overlap_depths is not None:
                overlaps.append((min(overlap_depths), first_index, second_index))
    return sorted(overlaps, reverse=True)


def _aabb_pair_push_candidates(
    *,
    current_mins: np.ndarray,
    current_maxs: np.ndarray,
    first_index: int,
    second_index: int,
    allowed_min: np.ndarray,
    allowed_max: np.ndarray,
    aabb_clearance: float,
    tolerance: float,
) -> list[tuple[float, int, float, float, float]] | None:
    """Return feasible opposite-direction pushes, or ``None`` if already separate."""
    if (
        _aabb_pair_overlap_depths(
            current_mins=current_mins,
            current_maxs=current_maxs,
            first_index=first_index,
            second_index=second_index,
            aabb_clearance=aabb_clearance,
            tolerance=tolerance,
        )
        is None
    ):
        return None

    candidates: list[tuple[float, int, float, float, float]] = []
    for axis in (0, 1):
        for first_direction in (-1.0, 1.0):
            second_direction = -first_direction
            if first_direction < 0.0:
                required_distance = (
                    current_maxs[first_index, axis]
                    + aabb_clearance
                    - current_mins[second_index, axis]
                )
                first_capacity = max(
                    0.0,
                    current_mins[first_index, axis] - allowed_min[axis],
                )
                second_capacity = max(
                    0.0,
                    allowed_max[axis] - current_maxs[second_index, axis],
                )
            else:
                required_distance = (
                    current_maxs[second_index, axis]
                    + aabb_clearance
                    - current_mins[first_index, axis]
                )
                first_capacity = max(
                    0.0,
                    allowed_max[axis] - current_maxs[first_index, axis],
                )
                second_capacity = max(
                    0.0,
                    current_mins[second_index, axis] - allowed_min[axis],
                )
            if first_capacity + second_capacity < required_distance - tolerance:
                continue

            # Split the required movement as evenly as possible, constrained by
            # each AABB's remaining distance to the table boundary.
            first_move = float(
                np.clip(
                    required_distance / 2.0,
                    max(0.0, required_distance - second_capacity),
                    min(required_distance, first_capacity),
                )
            )
            second_move = required_distance - first_move
            candidates.append(
                (
                    first_move**2 + second_move**2,
                    axis,
                    first_direction,
                    first_move,
                    second_move,
                )
            )
    return candidates


def _optimize_assets_2d_aabbs_in_rectangle(
    *,
    rectangle_min: np.ndarray,
    rectangle_max: np.ndarray,
    aabb_corners_by_id: dict[str, np.ndarray],
    boundary_margin: float,
    aabb_clearance: float,
    max_rounds: int = 64,
) -> dict[str, np.ndarray]:
    """Greedily pack 2D AABBs with minimum local squared displacement."""

    # Check the inputs for validity.
    if not np.isfinite(boundary_margin) or boundary_margin < 0.0:
        raise ValueError("boundary_margin must be a finite non-negative number.")
    if not np.isfinite(aabb_clearance) or aabb_clearance < 0.0:
        raise ValueError("aabb_clearance must be a finite non-negative number.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")

    asset_ids = sorted(aabb_corners_by_id)
    if not asset_ids:
        return {}

    asset_mins: list[np.ndarray] = []
    asset_maxs: list[np.ndarray] = []
    for asset_id in asset_ids:
        corners = aabb_corners_by_id[asset_id]
        # Get all the asset's AABB min and max corners in the z-up world x-y plane.
        asset_min, asset_max = _aabb_2d_bounds_from_corners(
            corners,
            name=f"Asset {asset_id!r} centered 2D AABB",
            require_nonzero_extent=False,
        )
        asset_mins.append(asset_min)
        asset_maxs.append(asset_max)

    base_mins = np.stack(asset_mins)
    base_maxs = np.stack(asset_maxs)
    # Get table support surface's largest internal rectangle's min and max corners in the z-up world x-y plane.
    allowed_min = rectangle_min + boundary_margin
    allowed_max = rectangle_max - boundary_margin
    # Compute the least and greatest offsets for each asset's AABB to stay inside the table's largest internal rectangle.
    lower_offset_bounds = allowed_min - base_mins
    upper_offset_bounds = allowed_max - base_maxs

    # Check if any asset's AABB is larger than the table's largest internal rectangle after applying the boundary margin. If so, raise an error.
    if np.any(lower_offset_bounds > upper_offset_bounds + 1e-9):
        too_large_index = int(
            np.argwhere(lower_offset_bounds > upper_offset_bounds)[0, 0]
        )
        asset_id = asset_ids[too_large_index]
        raise ValueError(
            f"Asset {asset_id!r} is larger than the table packing rectangle "
            "after applying boundary_margin."
        )

    # The zero vector keeps the centered initial layout. Clamp it only when an
    # AABB starts outside the table; this is the smallest boundary-only move.
    offsets = np.clip(
        np.zeros_like(base_mins),
        lower_offset_bounds,
        upper_offset_bounds,
    )
    tolerance = 1e-9

    for _ in range(max_rounds):
        current_mins = base_mins + offsets
        current_maxs = base_maxs + offsets
        overlaps = _find_overlapping_2d_aabb_pairs(
            current_mins=current_mins,
            current_maxs=current_maxs,
            aabb_clearance=aabb_clearance,
            tolerance=tolerance,
        )
        if not overlaps:
            return {
                asset_id: offsets[index].copy()
                for index, asset_id in enumerate(asset_ids)
            }

        # Process every pair found at the start of this round. A preceding pair
        # move may already resolve a later pair, so recheck it before moving.
        for _, first_index, second_index in overlaps:
            current_mins = base_mins + offsets
            current_maxs = base_maxs + offsets
            candidates = _aabb_pair_push_candidates(
                current_mins=current_mins,
                current_maxs=current_maxs,
                first_index=first_index,
                second_index=second_index,
                allowed_min=allowed_min,
                allowed_max=allowed_max,
                aabb_clearance=aabb_clearance,
                tolerance=tolerance,
            )
            if candidates is None:
                continue
            if not candidates:
                # Both AABBs are already blocked by the table boundary on every
                # separating axis. Keep the current boundary-safe layout and
                # let the later gravity simulation handle this residual overlap.
                return {
                    asset_id: offsets[index].copy()
                    for index, asset_id in enumerate(asset_ids)
                }

            _, axis, first_direction, first_move, second_move = min(candidates)
            offsets[first_index, axis] += first_direction * first_move
            offsets[second_index, axis] -= first_direction * second_move
            offsets = np.clip(offsets, lower_offset_bounds, upper_offset_bounds)

    # The bounded greedy search may leave overlaps in densely packed scenes.
    # Return its best boundary-safe result instead of aborting scene generation;
    # the following gravity simulation can resolve remaining physical contacts.
    return {asset_id: offsets[index].copy() for index, asset_id in enumerate(asset_ids)}


def _convert_layout_coordinate_system(
    layout_object: dict[str, object],
    *,
    source_to_target_matrix: np.ndarray,
) -> dict[str, object]:
    """A helper to convert a layout object between coordinate systems using a 4x4 transform."""
    target_to_source_matrix = np.linalg.inv(source_to_target_matrix)
    return transform_matrix_to_layout_object(
        str(layout_object["id"]),
        source_to_target_matrix
        @ layout_object_to_transform_matrix(layout_object)
        @ target_to_source_matrix,
    )


def export_baked_layout_object_glbs(
    layout: list[dict[str, object]],
    geometry_root: str | Path,
    output_root: str | Path,
) -> list[Path]:
    """Bake a layout into each object GLB and export them separately."""
    if not layout:
        raise ValueError("Cannot export objects without layout objects.")

    resolved_geometry_root = Path(geometry_root).expanduser().resolve()
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for layout_object in layout:
        object_id = layout_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Layout object id must be a non-empty string.")
        mesh_path = resolved_geometry_root / f"{object_id}.glb"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Geometry not found: {mesh_path}")

        loaded_mesh = trimesh.load(mesh_path, process=False)
        if isinstance(loaded_mesh, trimesh.Scene):
            mesh = loaded_mesh.dump(concatenate=True)
        elif isinstance(loaded_mesh, trimesh.Trimesh):
            mesh = loaded_mesh
        else:
            raise ValueError(f"Coarse geometry is not a mesh: {mesh_path}")

        mesh.apply_transform(layout_object_to_transform_matrix(layout_object))
        output_path = resolved_output_root / f"{object_id}.glb"
        mesh.export(output_path, file_type="glb")
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Baked coarse object was not written: {output_path}"
            )
        output_paths.append(output_path)
    return output_paths


def export_baked_coarse_object_glbs(
    coarse_layout: list[dict[str, object]],
    coarse_geometry_root: str | Path,
    output_root: str | Path,
) -> list[Path]:
    """Bake the coarse layout into each object GLB and export them separately."""
    return export_baked_layout_object_glbs(
        layout=coarse_layout,
        geometry_root=coarse_geometry_root,
        output_root=output_root,
    )


def simready_object_glb(
    coarse_glb_path: str | Path,
    *,
    object_id: str,
    rot: object,
    pos: object,
    scale: object,
) -> tuple[trimesh.Trimesh, dict[str, list[float]]]:
    """Bake an object's coarse scale (from the coarse layout currently)
    and canonicalize its AABB bottom center to the world's x-y plane (0, 0).

     Return the processed mesh and its updated layout transform without writing a
     GLB file. The caller owns the output path and export.
    """

    resolved_coarse_glb_path = Path(coarse_glb_path).expanduser().resolve()
    if not resolved_coarse_glb_path.is_file():
        raise FileNotFoundError(
            f"Coarse object geometry not found: {resolved_coarse_glb_path}"
        )

    loaded_mesh = trimesh.load(resolved_coarse_glb_path, process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
        mesh = loaded_mesh.dump(concatenate=True)
    elif isinstance(loaded_mesh, trimesh.Trimesh):
        mesh = loaded_mesh
    else:
        raise ValueError(
            f"Coarse object geometry is not a mesh: {resolved_coarse_glb_path}"
        )

    coarse_rot = _three_floats(rot, field_name="rot")
    coarse_pos = np.asarray(_three_floats(pos, field_name="pos"), dtype=float)
    coarse_scale = np.asarray(_three_floats(scale, field_name="scale"), dtype=float)
    if np.any(coarse_scale <= 0):
        raise ValueError("Coarse object scale values must be positive.")
    # We need the object id to determine whether it is a bottle-like object.
    # If it does, then we will do a special standardization. (Hard code)
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Scene object id must be a non-empty string.")

    # GLB uses y-up. Convert its vertices to z-up while processing the geometry.
    y_up_to_z_up_rotation = Rotation.from_euler("x", 90.0, degrees=True)
    y_up_to_z_up_matrix = y_up_to_z_up_rotation.as_matrix()
    y_up_to_z_up_transform = np.eye(4)
    y_up_to_z_up_transform[:3, :3] = y_up_to_z_up_matrix
    mesh.apply_transform(y_up_to_z_up_transform)

    # Standardize upright containers in temporary z-up coordinates before the
    # shared center, scale, and bottom-center preprocessing.
    # This is to ensure the action agent can pick up the bottle or can-like objects.
    bottle_alignment_matrix = np.eye(3)
    if _is_upright_container_id(object_id):
        bottle_alignment_matrix = _standardize_bottle_z_up(mesh)
        bottle_alignment_transform = np.eye(4)
        bottle_alignment_transform[:3, :3] = bottle_alignment_matrix
        mesh.apply_transform(bottle_alignment_transform)

    # First make the object's AABB center at the origin.
    original_aabb_center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-original_aabb_center)

    # Scale the object with the value in the coarse layout.
    scale_transform = np.eye(4)
    scale_transform[
        :3, :3
    ] = (  # Actually there's no need to do so, for the scale factor is all equal in x, y, z axes.
        bottle_alignment_matrix
        @ y_up_to_z_up_matrix
        @ np.diag(coarse_scale)
        @ y_up_to_z_up_matrix.T
        @ bottle_alignment_matrix.T
    )
    mesh.apply_transform(scale_transform)

    # Move the scaled object's AABB bottom center to the world's x-y plane (z=0).
    scaled_bounds = mesh.bounds
    scaled_aabb_bottom_center = np.array(
        [
            (scaled_bounds[0, 0] + scaled_bounds[1, 0]) / 2,
            (scaled_bounds[0, 1] + scaled_bounds[1, 1]) / 2,
            scaled_bounds[0, 2],
        ]
    )
    mesh.apply_translation(-scaled_aabb_bottom_center)

    # Convert the processed GLB back to its standard y-up coordinate system.
    z_up_to_y_up_transform = np.eye(4)
    z_up_to_y_up_transform[:3, :3] = y_up_to_z_up_matrix.T
    mesh.apply_transform(z_up_to_y_up_transform)

    # Compensate the bottle's local rotation so that its coarse world pose does not change.
    local_bottle_rotation = Rotation.from_matrix(
        y_up_to_z_up_matrix.T @ bottle_alignment_matrix @ y_up_to_z_up_matrix
    )
    coarse_rotation_matrix = Rotation.from_euler(
        "xyz", coarse_rot, degrees=True
    ).as_matrix()
    rotation = Rotation.from_matrix(
        coarse_rotation_matrix @ local_bottle_rotation.inv().as_matrix()
    )
    # Update the pos.
    position_offset = y_up_to_z_up_matrix.T @ (
        scale_transform[:3, :3] @ original_aabb_center + scaled_aabb_bottom_center
    )
    return mesh, {
        "rot": rotation.as_euler("xyz", degrees=True).tolist(),
        "pos": (coarse_pos + rotation.apply(position_offset)).tolist(),
        "scale": [1.0, 1.0, 1.0],
    }


def _is_upright_container_id(object_id: str) -> bool:
    """Return True if the object id contains tokens that indicate it is a bottle-like upright container."""
    # Example: soda_can_0
    # tokens: {"soda", "can", "0"}
    # _UPRIGHT_CONTAINER_ID_TOKENS: {"bottle", "can", "jar"}
    # So this would return True because "can" is in the set of upright container tokens.
    tokens = set(re.findall(r"[a-z0-9]+", object_id.lower()))
    return bool(tokens & _UPRIGHT_CONTAINER_ID_TOKENS)


def _standardize_bottle_z_up(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return a proper rotation that maps a bottle-like mesh's long axis to z-up.
    Thanks to chenjian for this idea!
    """
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError(
            "Bottle standardization requires a non-degenerate triangle mesh."
        )

    open3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    sampled_points = np.asarray(
        open3d_mesh.sample_points_uniformly(number_of_points=10_000).points
    )  # (10000, 3) x (x, y, z)

    # Check the number of the points again, and check whether have some non-finite values.
    if sampled_points.shape[0] < 4 or not np.all(np.isfinite(sampled_points)):
        raise ValueError("Bottle standardization could not sample valid mesh points.")

    centered_points = sampled_points - sampled_points.mean(axis=0)
    # SVD find the longest axis.
    _, _, principal_axes = np.linalg.svd(centered_points, full_matrices=False)
    if np.linalg.det(principal_axes) < 0:
        principal_axes[2, :] *= -1  # in case the SVD returns a reflection.

    bottle_rotation = Rotation.from_euler(
        "y", 90.0, degrees=True
    ).as_matrix()  # 3x3 matrix
    # The first PCA axis is the longest axis; rotate it onto the temporary z axis.
    bottle_rotation = bottle_rotation @ principal_axes
    standardized_points = (bottle_rotation @ centered_points.T).T

    axis_min = standardized_points[:, 2].min()
    axis_max = standardized_points[:, 2].max()
    axis_range = axis_max - axis_min
    upper_points = standardized_points[
        standardized_points[:, 2] > axis_min + axis_range * 0.8
    ]
    lower_points = standardized_points[
        standardized_points[:, 2] < axis_min + axis_range * 0.2
    ]
    upper_volume = _convex_hull_volume(upper_points)
    lower_volume = _convex_hull_volume(lower_points)

    # Bottles usually have a smaller top (neck) than bottom; flip if necessary.
    if upper_volume > lower_volume:
        bottle_rotation = (
            Rotation.from_euler("x", 180.0, degrees=True).as_matrix() @ bottle_rotation
        )
    return bottle_rotation


def _convex_hull_volume(points: np.ndarray) -> float:
    """Return the volume of a non-degenerate point set's convex hull."""
    if points.shape[0] < 4:
        raise ValueError("Bottle standardization needs at least four points per end.")
    try:
        return float(ConvexHull(points).volume)
    except QhullError as exc:
        raise ValueError(
            "Bottle standardization found a degenerate end volume."
        ) from exc


def _three_floats(value: object, *, field_name: str) -> list[float]:

    # Validate whether the value is a list of three numeric values.
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Coarse layout field {field_name} must contain three values.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Coarse layout field {field_name} must contain numeric values."
        ) from exc
