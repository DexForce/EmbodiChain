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
import re

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.utils.logger import log_info

_TABLE_PHYSICS_ATTRS = {
    "mass": 10.0,  # Keep the table heavy if a simulator treats it as movable.
    "static_friction": 0.95,  # Resist lateral sliding at table contacts.
    "dynamic_friction": 0.9,  # Maintain high friction during sliding contacts.
    "restitution": 0.01,  # Prevent a table contact from producing visible bounce.
}
_ASSET_PHYSICS_ATTRS = {
    "mass": 0.01,  # Use a lightweight default for unconstrained generated assets.
    "contact_offset": 0.003,  # Start contact detection slightly before mesh contact.
    "rest_offset": 0.001,  # Keep a small stable separation after contact resolution.
    "restitution": 0.01,  # Prevent generated assets from bouncing on the table.
    "max_depenetration_velocity": 10.0,  # Cap corrective separation speed.
    "min_position_iters": 32,  # Use extra position iterations for stable contacts.
    "min_velocity_iters": 8,  # Use extra velocity iterations for stable contacts.
}
_FIXED_MAX_CONVEX_HULL_NUM = 16  # Shared VHACD hull budget for settling and export.


@dataclass(frozen=True)
class SimReadySceneProcessorConfig:
    """Object-category policy for SimReady mesh canonicalization."""

    upright_container_id_tokens: frozenset[str] = frozenset(
        {"bottle", "can", "jar", "flask", "thermos"}
    )  # Object-id tokens that enable upright-container standardization.


class SimReadySceneProcessor:
    """Create SimReady GLBs and layouts for one table and its scene assets."""

    def __init__(
        self,
        *,
        scene: Scene,
        coarse_layout_by_id: dict[str, dict[str, object]],
        coarse_geometry_root: str | Path,
        simready_geometry_root: str | Path,
        config: SimReadySceneProcessorConfig | None = None,
    ) -> None:
        self.scene = scene
        self.coarse_layout_by_id = coarse_layout_by_id
        self.coarse_geometry_root = Path(coarse_geometry_root).expanduser().resolve()
        self.simready_geometry_root = (
            Path(simready_geometry_root).expanduser().resolve()
        )
        self.simready_table_layout: dict[str, object] | None = None
        self.simready_assets_layout: list[dict[str, object]] | None = None
        self.config = config if config is not None else SimReadySceneProcessorConfig()
        if not self.config.upright_container_id_tokens:
            raise ValueError("upright_container_id_tokens must not be empty.")

    def process_table(self) -> dict[str, object]:
        """Process the required scene table and return its SimReady layout."""
        if self.scene.table is None:
            raise ValueError("Cannot SimReady a scene without a table.")
        self.simready_table_layout = self._process_object(self.scene.table)
        return self.simready_table_layout

    def process_assets(self) -> list[dict[str, object]]:
        """Process every scene asset and return SimReady layouts in scene order."""
        asset_ids: set[str] = set()
        processed_assets: list[dict[str, object]] = []
        for asset in self.scene.assets:
            if asset.id in asset_ids:
                raise ValueError(f"Scene assets contain duplicate id {asset.id!r}.")
            asset_ids.add(asset.id)
            processed_assets.append(self._process_object(asset))
        self.simready_assets_layout = processed_assets
        return self.simready_assets_layout

    def _process_object(self, scene_object: SceneObject) -> dict[str, object]:
        """Canonicalize one coarse object and write its SimReady GLB."""
        object_id = scene_object.id
        object_role = scene_object.kind
        if object_role not in {"table", "asset"}:
            raise ValueError(f"Unsupported SimReady object role {object_role!r}.")
        coarse_layout = self.coarse_layout_by_id.get(object_id)
        if coarse_layout is None:
            raise ValueError(f"Coarse layout does not contain object {object_id!r}.")
        simready_mesh, simready_transform = self._canonicalize_object_mesh(
            coarse_glb_path=self.coarse_geometry_root / f"{object_id}.glb",
            object_id=object_id,
            rot=coarse_layout.get("rot"),
            pos=coarse_layout.get("pos"),
            scale=coarse_layout.get("scale"),
        )
        output_path = self.simready_geometry_root / f"{object_id}.glb"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simready_mesh.export(output_path, file_type="glb")
        if not output_path.is_file():
            raise FileNotFoundError(
                f"SimReady {object_role} geometry was not written: {output_path}"
            )
        scene_object.simready_glb_path = str(output_path)
        scene_object.physics = self._fixed_physics_for_kind(object_role)
        log_info(f"Created SimReady {object_role}: {object_id!r}.")
        return {"id": object_id, **simready_transform}

    @staticmethod
    def _fixed_physics_for_kind(kind: str) -> ObjectPhysics:
        """Create the fixed initial physics profile for one SimReady object."""
        if kind == "table":
            return ObjectPhysics(
                body_type="kinematic",
                attrs=dict(_TABLE_PHYSICS_ATTRS),
                max_convex_hull_num=_FIXED_MAX_CONVEX_HULL_NUM,
            )
        if kind == "asset":
            return ObjectPhysics(
                body_type="dynamic",
                attrs=dict(_ASSET_PHYSICS_ATTRS),
                max_convex_hull_num=_FIXED_MAX_CONVEX_HULL_NUM,
            )
        raise ValueError(f"Unsupported SceneObject kind {kind!r} for physics.")

    def _canonicalize_object_mesh(
        self,
        *,
        coarse_glb_path: str | Path,
        object_id: str,
        rot: object,
        pos: object,
        scale: object,
    ) -> tuple[trimesh.Trimesh, dict[str, list[float]]]:
        """Bake coarse scale and canonicalize one mesh's AABB bottom centre.

        Return the processed mesh and its updated layout transform without writing
        a GLB file. The caller owns the output path and export.
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

        coarse_rot = self._three_floats(rot, field_name="rot")
        coarse_pos = np.asarray(self._three_floats(pos, field_name="pos"), dtype=float)
        coarse_scale = np.asarray(
            self._three_floats(scale, field_name="scale"), dtype=float
        )
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
        if self._is_upright_container_id(object_id):
            bottle_alignment_matrix = self._standardize_bottle_z_up(mesh)
            bottle_alignment_transform = np.eye(4)
            bottle_alignment_transform[:3, :3] = bottle_alignment_matrix
            mesh.apply_transform(bottle_alignment_transform)

        # First make the object's AABB center at the origin.
        original_aabb_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-original_aabb_center)

        # Scale the object with the value in the coarse layout.
        scale_transform = np.eye(4)
        scale_transform[:3, :3] = (
            # Actually there's no need to do so, for the scale factor is all equal
            # in x, y, z axes.
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

        # Compensate the bottle's local rotation so that its coarse world pose does
        # not change.
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

    def _is_upright_container_id(self, object_id: str) -> bool:
        """Return whether object-id tokens indicate a bottle-like container."""
        # Example: soda_can_0
        # tokens: {"soda", "can", "0"}
        # upright_container_id_tokens: {"bottle", "can", "jar"}
        # So this returns True because "can" is in the configured token set.
        tokens = set(re.findall(r"[a-z0-9]+", object_id.lower()))
        return bool(tokens & self.config.upright_container_id_tokens)

    @staticmethod
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

        # Check the number of the points again, and check whether have some
        # non-finite values.
        if sampled_points.shape[0] < 4 or not np.all(np.isfinite(sampled_points)):
            raise ValueError(
                "Bottle standardization could not sample valid mesh points."
            )

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
        upper_volume = SimReadySceneProcessor._convex_hull_volume(upper_points)
        lower_volume = SimReadySceneProcessor._convex_hull_volume(lower_points)

        # Bottles usually have a smaller top (neck) than bottom; flip if necessary.
        if upper_volume > lower_volume:
            bottle_rotation = (
                Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
                @ bottle_rotation
            )
        return bottle_rotation

    @staticmethod
    def _convex_hull_volume(points: np.ndarray) -> float:
        """Return the volume of a non-degenerate point set's convex hull."""
        if points.shape[0] < 4:
            raise ValueError(
                "Bottle standardization needs at least four points per end."
            )
        try:
            return float(ConvexHull(points).volume)
        except QhullError as exc:
            raise ValueError(
                "Bottle standardization found a degenerate end volume."
            ) from exc

    @staticmethod
    def _three_floats(value: object, *, field_name: str) -> list[float]:
        """Validate and convert a three-value layout field to floats."""
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(
                f"Coarse layout field {field_name} must contain three values."
            )
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Coarse layout field {field_name} must contain numeric values."
            ) from exc
