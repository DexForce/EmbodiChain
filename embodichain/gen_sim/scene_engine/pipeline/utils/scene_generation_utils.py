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

from embodichain.lab.sim import SimulationManagerCfg, SimulationManager
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import trimesh

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

    sim = SimulationManager(
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
        SimulationManager.flush_cleanup_queue()

    settled_assets_layout = [
        settled_layout_by_id[str(asset_layout["id"])] for asset_layout in assets_layout
    ]
    return settled_assets_layout


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
