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

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.llms.openai_compatible_client import (
    OpenAICompatibleVLM,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    query_vlm_object_rotation_and_target_size,
    compute_uniform_xy_scale_for_target,
    render_object_front_top_views,
    rotate_glb_about_x_axis,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
    TableSupportSurfaceDetector,
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
class SimReadyProcessorConfig:
    """SceneGraph-conditioned policy for SimReady mesh canonicalization."""

    use_vlm_scale: bool = False  # Use the VLM-selected asset scale.
    use_vlm_rotation: bool = False  # Use the VLM-selected asset rotation.

    long_axis_object_ids: frozenset[str] = frozenset()


class SimReadyProcessor:
    """Create SimReady GLBs and layouts for scene objects."""

    def __init__(
        self,
        *,
        scene: Scene,
        coarse_layout_by_id: dict[str, dict[str, object]],
        coarse_geometry_root: str | Path,
        simready_geometry_root: str | Path,
        debug_output_root: str | Path | None = None,
        config: SimReadyProcessorConfig | None = None,
        vlm_client: OpenAICompatibleVLM | None = None,
    ) -> None:
        self.scene = scene
        self.coarse_layout_by_id = coarse_layout_by_id
        self.coarse_geometry_root = Path(coarse_geometry_root).expanduser().resolve()
        self.simready_geometry_root = (
            Path(simready_geometry_root).expanduser().resolve()
        )
        # Save rendered debug images.
        self.debug_output_root = (
            Path(debug_output_root).expanduser().resolve()
            if debug_output_root is not None
            else None
        )
        self.simready_table_layout: dict[str, object] | None = None
        self.simready_assets_layout: list[dict[str, object]] | None = None
        self.config = config if config is not None else SimReadyProcessorConfig()
        self.vlm_client = vlm_client
        if (
            self.config.use_vlm_scale or self.config.use_vlm_rotation
        ) and vlm_client is None:
            raise ValueError("vlm_client is required when VLM transforms are enabled.")

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

    def _process_object(
        self,
        scene_object: SceneObject,
        *,
        scale: object | None = None,
        rot: object | None = None,
    ) -> dict[str, object]:
        """Canonicalize one coarse object and write its SimReady GLB."""
        object_id = scene_object.id
        object_role = scene_object.kind
        if object_role not in {"table", "asset"}:
            raise ValueError(f"Unsupported SimReady object role {object_role!r}.")
        coarse_layout = self.coarse_layout_by_id.get(object_id)
        if coarse_layout is None:
            raise ValueError(f"Coarse layout does not contain object {object_id!r}.")
        prepared_glb_path, vlm_scale = self._prepare_vlm_rotated_glb(scene_object)
        selected_scale = scale
        if selected_scale is None:
            selected_scale = vlm_scale or coarse_layout.get("scale")
        simready_mesh, simready_transform = self._canonicalize_object_mesh(
            coarse_glb_path=prepared_glb_path,
            object_id=object_id,
            # An enabled external rotation replaces the coarse-layout rotation.
            rot=coarse_layout.get("rot") if rot is None else rot,
            pos=coarse_layout.get("pos"),
            # An enabled VLM scale replaces the coarse-layout scale.
            scale=selected_scale,
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
        # For table. (currently the id is fixed into table)
        if object_role == "table":
            # Detect and persist all reusable tabletop support geometry at SimReady time.
            support_detector = TableSupportSurfaceDetector(
                table_world_mesh=self._z_up_table_mesh(simready_mesh),
                debug_output_root=self.debug_output_root,
            )
            support_region = support_detector.detect()
            scene_object.support_surface_z = support_region.top_z
            scene_object.support_contour_xy = [
                [float(x), float(y)]
                for x, y in support_region.support_polygon.exterior.coords[:-1]
            ]
            scene_object.support_optimization_rect_xy = [
                [float(x), float(y)]
                for x, y in support_region.optimization_rectangle.exterior.coords[:-1]
            ]
            if self.debug_output_root is not None:
                # Keep the 3D selected surface and 2D contour diagnostics beside SimReady output.
                support_detector.save_support_surface_debug_images()
        log_info(f"Created SimReady {object_role}: {object_id!r}.")
        return {"id": object_id, **simready_transform}

    @staticmethod
    def _z_up_table_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Convert one canonical y-up GLB mesh into the detector's z-up frame."""
        y_up_to_z_up = np.eye(4)
        y_up_to_z_up[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
        z_up_mesh = mesh.copy()
        z_up_mesh.apply_transform(y_up_to_z_up)
        return z_up_mesh

    def _prepare_vlm_rotated_glb(
        self, scene_object: SceneObject
    ) -> tuple[Path, list[float] | None]:
        """Render, query, and optionally bake the VLM-selected x-axis rotation."""
        coarse_path = self.coarse_geometry_root / f"{scene_object.id}.glb"
        if not (self.config.use_vlm_scale or self.config.use_vlm_rotation):
            return coarse_path, None
        decision = self._vlm_transform_for_object(
            scene_object,
            use_scale=self.config.use_vlm_scale,
            use_rotation=self.config.use_vlm_rotation,
        )
        rotate_about_x = bool(decision["rotate_about_x"])
        vlm_scale = compute_uniform_xy_scale_for_target(
            glb_path=coarse_path,
            target_xy_size_cm=decision["target_xy_size_cm"],
            rotate_about_x=rotate_about_x,
        )
        rotated_path = rotate_glb_about_x_axis(
            input_path=coarse_path,
            output_path=self.simready_geometry_root
            / "vlm_rotated"
            / f"{scene_object.id}.glb",
            rotate=rotate_about_x,
        )
        # The scale flag controls whether this VLM-derived isotropic scale is used.
        # Apply the same factor on x, y, and z to preserve the asset's proportions.
        return (
            rotated_path,
            [vlm_scale, vlm_scale, vlm_scale] if self.config.use_vlm_scale else None,
        )

    def _vlm_transform_for_object(
        self,
        scene_object: SceneObject,
        *,
        use_scale: bool,
        use_rotation: bool,
    ) -> dict[str, object]:
        """Render the object and return the validated VLM pose decision."""
        del use_scale, use_rotation
        assert self.vlm_client is not None
        coarse_path = self.coarse_geometry_root / f"{scene_object.id}.glb"
        needed_layout = "This asset needs to be place on the table that will not move a lot after simulation."
        debug_root = self.simready_geometry_root.parent / "debug"
        rendered_path = render_object_front_top_views(
            glb_path=coarse_path,
            output_path=debug_root / "vlm_views" / f"{scene_object.id}.png",
        )
        # Both semantic questions are always answered in one multimodal call.
        return query_vlm_object_rotation_and_target_size(
            scene_object_description=scene_object.description,
            needed_layout=needed_layout,
            rendered_views_path=rendered_path,
            vlm_client=self.vlm_client,
            debug_output_path=debug_root / "vlm_outputs" / f"{scene_object.id}.json",
        )

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
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Scene object id must be a non-empty string.")

        # GLB uses y-up. Convert its vertices to z-up while processing the geometry.
        y_up_to_z_up_rotation = Rotation.from_euler("x", 90.0, degrees=True)
        y_up_to_z_up_matrix = y_up_to_z_up_rotation.as_matrix()
        y_up_to_z_up_transform = np.eye(4)
        y_up_to_z_up_transform[:3, :3] = y_up_to_z_up_matrix
        mesh.apply_transform(y_up_to_z_up_transform)

        # Standardize graph-marked elongated assets before shared mesh processing.
        # This makes local z their primary axis, so later scene-graph calibration
        # can reliably recover the image-observed standing or lying orientation.
        long_axis_alignment_matrix = np.eye(3)
        if self._requires_long_axis_standardization(object_id):
            long_axis_alignment_matrix = self._standardize_long_axis_z_up(mesh)
            long_axis_alignment_transform = np.eye(4)
            long_axis_alignment_transform[:3, :3] = long_axis_alignment_matrix
            mesh.apply_transform(long_axis_alignment_transform)

        # First make the object's AABB center at the origin.
        original_aabb_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-original_aabb_center)

        # Scale the object with the value in the coarse layout.
        scale_transform = np.eye(4)
        scale_transform[:3, :3] = (
            # Actually there's no need to do so, for the scale factor is all equal
            # in x, y, z axes.
            long_axis_alignment_matrix
            @ y_up_to_z_up_matrix
            @ np.diag(coarse_scale)
            @ y_up_to_z_up_matrix.T
            @ long_axis_alignment_matrix.T
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

        # Compensate the local canonicalization so its coarse world pose does not
        # change until layout refinement applies the image-observed correction.
        local_long_axis_rotation = Rotation.from_matrix(
            y_up_to_z_up_matrix.T @ long_axis_alignment_matrix @ y_up_to_z_up_matrix
        )
        coarse_rotation_matrix = Rotation.from_euler(
            "xyz", coarse_rot, degrees=True
        ).as_matrix()
        rotation = Rotation.from_matrix(
            coarse_rotation_matrix @ local_long_axis_rotation.inv().as_matrix()
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

    def _requires_long_axis_standardization(self, object_id: str) -> bool:
        """Return whether graph semantics identified one asset with a long axis."""
        return object_id in self.config.long_axis_object_ids

    @staticmethod
    def _standardize_long_axis_z_up(mesh: trimesh.Trimesh) -> np.ndarray:
        """Return a proper rotation that maps a mesh's primary axis to z-up.
        Thanks to chanjian's idea.
        """
        if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
            raise ValueError(
                "Long-axis standardization requires a non-degenerate triangle mesh."
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
                "Long-axis standardization could not sample valid mesh points."
            )

        centered_points = sampled_points - sampled_points.mean(axis=0)
        # SVD find the longest axis.
        _, _, principal_axes = np.linalg.svd(centered_points, full_matrices=False)
        if np.linalg.det(principal_axes) < 0:
            principal_axes[2, :] *= -1  # in case the SVD returns a reflection.

        long_axis_rotation = Rotation.from_euler(
            "y", 90.0, degrees=True
        ).as_matrix()  # 3x3 matrix
        # The first PCA axis is the longest axis; rotate it onto the temporary z axis.
        long_axis_rotation = long_axis_rotation @ principal_axes
        return long_axis_rotation

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
