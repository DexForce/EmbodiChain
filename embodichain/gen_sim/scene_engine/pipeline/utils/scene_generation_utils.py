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

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import trimesh


_UPRIGHT_CONTAINER_ID_TOKENS = frozenset(
    {"bottle", "can", "jar", "flask", "thermos"}
)


def quaternion_wxyz_to_euler_xyz_degrees(
    quaternion_wxyz: Sequence[float],
) -> list[float]:
    """Convert a ``[w, x, y, z]`` quaternion to [roll_x, pitch_y, yaw_z] degrees."""
    if len(quaternion_wxyz) != 4:
        raise ValueError("Rotation quaternion must contain exactly four values.")

    w, x, y, z = quaternion_wxyz
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True).tolist()


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

        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "xyz",
            _three_floats(layout_object.get("rot"), field_name="rot"),
            degrees=True,
        ).as_matrix() @ np.diag(
            _three_floats(layout_object.get("scale"), field_name="scale")
        )
        transform[:3, 3] = _three_floats(layout_object.get("pos"), field_name="pos")
        mesh.apply_transform(transform)
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
        raise FileNotFoundError(f"Coarse object geometry not found: {resolved_coarse_glb_path}")

    loaded_mesh = trimesh.load(resolved_coarse_glb_path, process=False)
    if isinstance(loaded_mesh, trimesh.Scene):
        mesh = loaded_mesh.dump(concatenate=True)
    elif isinstance(loaded_mesh, trimesh.Trimesh):
        mesh = loaded_mesh
    else:
        raise ValueError(f"Coarse object geometry is not a mesh: {resolved_coarse_glb_path}")

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
    scale_transform[:3, :3] = ( # Actually there's no need to do so, for the scale factor is all equal in x, y, z axes.
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
        y_up_to_z_up_matrix.T
        @ bottle_alignment_matrix
        @ y_up_to_z_up_matrix
    )
    coarse_rotation_matrix = Rotation.from_euler(
        "xyz", coarse_rot, degrees=True
    ).as_matrix()
    rotation = Rotation.from_matrix(
        coarse_rotation_matrix @ local_bottle_rotation.inv().as_matrix()
    )
    # Update the pos.
    position_offset = y_up_to_z_up_matrix.T @ (
        scale_transform[:3, :3] @ original_aabb_center
        + scaled_aabb_bottom_center
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
        raise ValueError("Bottle standardization requires a non-degenerate triangle mesh.")

    open3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    sampled_points = np.asarray(
        open3d_mesh.sample_points_uniformly(number_of_points=10_000).points
    ) # (10000, 3) x (x, y, z)

    # Check the number of the points again, and check whether have some non-finite values.
    if sampled_points.shape[0] < 4 or not np.all(np.isfinite(sampled_points)):
        raise ValueError("Bottle standardization could not sample valid mesh points.")

    centered_points = sampled_points - sampled_points.mean(axis=0)
    # SVD find the longest axis.
    _, _, principal_axes = np.linalg.svd(centered_points, full_matrices=False)
    if np.linalg.det(principal_axes) < 0:
        principal_axes[2, :] *= -1 # in case the SVD returns a reflection.

    bottle_rotation = Rotation.from_euler("y", 90.0, degrees=True).as_matrix() # 3x3 matrix
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
            Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
            @ bottle_rotation
        )
    return bottle_rotation


def _convex_hull_volume(points: np.ndarray) -> float:
    """Return the volume of a non-degenerate point set's convex hull."""
    if points.shape[0] < 4:
        raise ValueError("Bottle standardization needs at least four points per end.")
    try:
        return float(ConvexHull(points).volume)
    except QhullError as exc:
        raise ValueError("Bottle standardization found a degenerate end volume.") from exc


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
