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

"""Private single-mesh backend for antipodal grasp generation."""

from __future__ import annotations

import os
import open3d as o3d
import time
import torch
import numpy as np
import trimesh
import hashlib
import torch.nn.functional as F

import viser
import viser.transforms as tf
from pathlib import Path
from typing import Any, cast

from embodichain.utils import logger
from embodichain.utils.nms import pose_nms
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSampler,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionChecker,
    GripperCollisionCfg,
)

GRASP_ANNOTATOR_CACHE_DIR = (
    Path.home() / ".cache" / "embodichain" / "grasp_annotator_cache"
)
VERSION_TAG = "v0.0.2"


__all__: list[str] = []


class _AntipodalMeshBackend:
    """Implement antipodal generation for one mesh.

    This backend is an implementation detail of
    :class:`AntipodalGraspPoseGenerator`. It owns mesh-specific sampling,
    annotation, collision checking, and disk-cache state, but it deliberately
    does not define a second public generator or configuration API.
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        *,
        sampler_cfg: AntipodalSamplerCfg,
        collision_cfg: GripperCollisionCfg,
        max_deviation_angle: float,
        approach_direction_samples: int,
        max_candidates: int,
        interactive_annotation: bool,
        viser_port: int,
        use_largest_connected_component: bool,
        filter_ground_collision: bool,
    ) -> None:
        """Initialize the private backend for one target-local mesh.

        Args:
            vertices: Vertex positions with shape ``(V, 3)``.
            triangles: Triangle indices with shape ``(F, 3)``.
            sampler_cfg: Configuration for antipodal contact sampling.
            collision_cfg: Physical collision-checker configuration.
            max_deviation_angle: Maximum approach-axis deviation in radians.
            approach_direction_samples: Number of approach variants per pair.
            max_candidates: Maximum number of ranked candidates to retain.
            interactive_annotation: Whether to select a region through Viser.
            viser_port: Port for interactive annotation.
            use_largest_connected_component: Whether to retain only the largest
                component of an interactive selection.
            filter_ground_collision: Whether to reject ground collisions.
        """
        self.device = vertices.device
        self.vertices = vertices
        self.triangles = triangles
        self.mesh = trimesh.Trimesh(
            vertices=vertices.to("cpu").numpy(),
            faces=triangles.to("cpu").numpy(),
            process=False,
            force="mesh",
        )
        self._collision_checker = GripperCollisionChecker(
            object_mesh_verts=vertices,
            object_mesh_faces=triangles,
            cfg=collision_cfg,
        )
        self._sampler_cfg = sampler_cfg
        self._max_deviation_angle = max_deviation_angle
        self._approach_direction_samples = approach_direction_samples
        self._max_candidates = max_candidates
        self._interactive_annotation = interactive_annotation
        self._viser_port = viser_port
        self._use_largest_connected_component = use_largest_connected_component
        self._filter_ground_collision = filter_ground_collision
        self._antipodal_sampler = AntipodalSampler(cfg=sampler_cfg)
        self._hit_point_pairs: torch.Tensor | None = None

        # Load cached antipodal pairs for the whole mesh if available.
        cache_path = self._get_cache_dir(self.vertices, self.triangles)
        if os.path.exists(cache_path):
            logger.log_info(f"Found cached antipodal pairs at {cache_path}. Loading.")
            self._hit_point_pairs = torch.tensor(
                np.load(cache_path), dtype=torch.float32, device=self.device
            )

    @property
    def is_prepared(self) -> bool:
        """Whether antipodal pairs are ready for pose generation."""
        return self._hit_point_pairs is not None

    @property
    def antipodal_pairs(self) -> torch.Tensor:
        """Return an owned snapshot of the prepared antipodal pairs."""
        if self._hit_point_pairs is None:
            raise RuntimeError("The mesh backend has not been prepared.")
        return self._hit_point_pairs.clone()

    def annotate(self) -> torch.Tensor:
        """Annotate antipodal grasp region on the mesh and return sampled antipodal point pairs.

        Returns:
            torch.Tensor: A tensor of shape (N, 2, 3) representing N antipodal point pairs.
                Each pair consists of a hit point and its corresponding surface point.
        """
        if not self._interactive_annotation:
            hit_point_pairs = self._generate_hit_point_pairs(
                self.vertices, self.triangles
            )
            self._cache_hit_point_pairs(hit_point_pairs)
            return hit_point_pairs
        logger.log_info(
            f"[Viser] *****Annotate grasp region in http://localhost:{self._viser_port}"
        )

        server = viser.ViserServer(port=self._viser_port)
        server.gui.configure_theme(brand_color=(130, 0, 150))
        server.scene.set_up_direction("+z")

        mesh_handle = server.scene.add_mesh_trimesh(name="/mesh", mesh=self.mesh)
        selected_overlay: viser.GlbHandle | None = None
        sel_vertex_indices: np.ndarray | None = None
        sel_face_indices: np.ndarray | None = None
        sel_vertices: np.ndarray | None = None
        sel_faces: np.ndarray | None = None

        hit_point_pairs = None
        return_flag = False

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            nonlocal mesh_handle
            nonlocal selected_overlay
            nonlocal sel_vertex_indices
            nonlocal sel_face_indices
            nonlocal sel_vertices
            nonlocal sel_faces

            # client.camera.position = np.array([0.0, 0.0, -0.5])
            # client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            select_button = client.gui.add_button(
                "Rect Select Region", icon=viser.Icon.PAINT
            )
            confirm_button = client.gui.add_button("Confirm Selection")

            @select_button.on_click
            def _(_evt: viser.GuiEvent) -> None:
                select_button.disabled = True

                @client.scene.on_pointer_event(event_type="rect-select")
                def _(event: viser.ScenePointerEvent) -> None:
                    nonlocal mesh_handle
                    nonlocal selected_overlay
                    nonlocal sel_vertex_indices
                    nonlocal sel_face_indices
                    nonlocal sel_vertices
                    nonlocal sel_faces
                    nonlocal hit_point_pairs
                    client.scene.remove_pointer_callback()

                    proj, depth = _AntipodalMeshBackend._project_vertices_to_screen(
                        cast(np.ndarray, self.mesh.vertices),
                        mesh_handle,
                        event.client.camera,
                    )

                    lower = np.minimum(
                        np.array(event.screen_pos[0]), np.array(event.screen_pos[1])
                    )
                    upper = np.maximum(
                        np.array(event.screen_pos[0]), np.array(event.screen_pos[1])
                    )
                    vertex_mask = ((proj >= lower) & (proj <= upper)).all(axis=1) & (
                        depth > 1e-6
                    )

                    (
                        sel_vertex_indices,
                        sel_face_indices,
                        sel_vertices,
                        sel_faces,
                    ) = _AntipodalMeshBackend._extract_selection_from_vertex_mask(
                        self.mesh,
                        vertex_mask,
                        self._use_largest_connected_component,
                    )
                    if sel_vertices is None:
                        logger.log_warning("[Selection] No vertices selected.")
                        return

                    color_mesh = self.mesh.copy()
                    vertex_colors = np.tile(
                        np.array([[0.85, 0.85, 0.85, 1.0]]),
                        (self.mesh.vertices.shape[0], 1),
                    )
                    vertex_colors[sel_vertex_indices] = np.array(
                        [0.56, 0.17, 0.92, 1.0]
                    )
                    color_mesh.visual.vertex_colors = vertex_colors  # type: ignore
                    mesh_handle = server.scene.add_mesh_trimesh(
                        name="/mesh", mesh=color_mesh
                    )

                    if selected_overlay is not None:
                        selected_overlay.remove()
                    selected_mesh = trimesh.Trimesh(
                        vertices=sel_vertices,
                        faces=sel_faces,
                        process=False,
                    )
                    selected_mesh.visual.face_colors = (0.9, 0.2, 0.2, 0.65)  # type: ignore
                    selected_overlay = server.scene.add_mesh_trimesh(
                        name="/selected", mesh=selected_mesh
                    )
                    logger.log_info(
                        f"[Selection] Selected {sel_vertex_indices.size} vertices and {sel_face_indices.size} faces."
                    )

                    hit_point_pairs = self._generate_hit_point_pairs(
                        torch.tensor(sel_vertices, device=self.device),
                        torch.tensor(sel_faces, device=self.device),
                    )

                    # for visualization only
                    extended_hit_point_pairs = (
                        _AntipodalMeshBackend._extend_hit_point_pairs(hit_point_pairs)
                    )
                    server.scene.add_line_segments(
                        name="/antipodal_pairs",
                        points=extended_hit_point_pairs.to("cpu").numpy(),
                        colors=(20, 200, 200),
                        line_width=1.5,
                    )

                @client.scene.on_pointer_callback_removed
                def _() -> None:
                    select_button.disabled = False

            @confirm_button.on_click
            def _(_evt: viser.GuiEvent) -> None:
                nonlocal return_flag
                if sel_vertices is None:
                    logger.log_warning("[Selection] No vertex selected.")
                    return
                else:
                    logger.log_info(
                        f"[Selection] {sel_vertices.shape[0]}vertices selected. Generating antipodal point pairs."
                    )
                    return_flag = True

        while True:
            if return_flag:
                if hit_point_pairs is not None:
                    self._cache_hit_point_pairs(hit_point_pairs)
                break
            time.sleep(0.5)
        if self._hit_point_pairs is None:
            raise RuntimeError("Interactive annotation completed without point pairs.")
        return self._hit_point_pairs

    def _generate_hit_point_pairs(
        self, vertices: torch.Tensor, triangles: torch.Tensor
    ) -> torch.Tensor:
        return self._antipodal_sampler.sample(
            vertices=vertices,
            faces=triangles,
        )

    def _cache_hit_point_pairs(self, hit_point_pairs: torch.Tensor):
        self._hit_point_pairs = hit_point_pairs
        cache_path = self._get_cache_dir(self.vertices, self.triangles)
        self._save_cache(cache_path, hit_point_pairs)

    def _get_cache_dir(self, vertices: torch.Tensor, triangles: torch.Tensor):
        vert_bytes = vertices.to("cpu").numpy().tobytes()
        face_bytes = triangles.to("cpu").numpy().tobytes()
        sampler_cfg = self._sampler_cfg
        sampler_signature = (
            f"{sampler_cfg.n_sample}|{sampler_cfg.max_angle:.17g}|"
            f"{sampler_cfg.min_length:.17g}|{sampler_cfg.max_length:.17g}|"
            f"partial={self._interactive_annotation}|"
            f"largest={self._use_largest_connected_component}"
        ).encode("utf-8")
        md5_hash = hashlib.md5(vert_bytes + face_bytes + sampler_signature).hexdigest()
        cache_path = os.path.join(
            GRASP_ANNOTATOR_CACHE_DIR, f"antipodal_cache_{VERSION_TAG}_{md5_hash}.npy"
        )
        return cache_path

    def _save_cache(self, cache_path: str, hit_point_pairs: torch.Tensor):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, hit_point_pairs.cpu().numpy().astype(np.float32))

    @staticmethod
    def _extend_hit_point_pairs(hit_point_pairs: torch.Tensor):
        origin_points = hit_point_pairs[:, 0, :]
        hit_points = hit_point_pairs[:, 1, :]
        mid_points = (origin_points + hit_points) / 2
        point_diff = hit_points - origin_points
        extended_origin = mid_points - 0.8 * point_diff
        extended_hit = mid_points + 0.8 * point_diff
        extended_point_pairs = torch.cat(
            [extended_origin[:, None, :], extended_hit[:, None, :]], dim=1
        )
        return extended_point_pairs

    @staticmethod
    def _project_vertices_to_screen(
        vertices_mesh: np.ndarray,
        mesh_handle: viser.GlbHandle,
        camera: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        T_world_mesh = tf.SE3.from_rotation_and_translation(
            tf.SO3(np.asarray(mesh_handle.wxyz)),
            np.asarray(mesh_handle.position),
        )
        vertices_world_h = (
            T_world_mesh.as_matrix()
            @ np.hstack([vertices_mesh, np.ones((vertices_mesh.shape[0], 1))]).T
        ).T
        vertices_world = vertices_world_h[:, :3]

        T_camera_world = tf.SE3.from_rotation_and_translation(
            tf.SO3(np.asarray(camera.wxyz)),
            np.asarray(camera.position),
        ).inverse()
        vertices_camera_h = (
            T_camera_world.as_matrix()
            @ np.hstack([vertices_world, np.ones((vertices_world.shape[0], 1))]).T
        ).T
        vertices_camera = vertices_camera_h[:, :3]

        fov = float(camera.fov)
        aspect = float(camera.aspect)
        projected = vertices_camera[:, :2] / np.maximum(vertices_camera[:, 2:3], 1e-8)
        projected /= np.tan(fov / 2.0)
        projected[:, 0] /= aspect
        projected = (1.0 + projected) / 2.0
        return projected, vertices_camera[:, 2]

    @staticmethod
    def _extract_selection_from_vertex_mask(
        mesh: trimesh.Trimesh,
        vertex_mask: np.ndarray,
        largest_component: bool,
    ) -> tuple[
        np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]:
        """Extract a sub-mesh from *mesh* using a per-vertex boolean mask.

        Args:
            mesh: The source mesh.
            vertex_mask: Boolean array of shape ``(V,)`` indicating which
                vertices are selected.
            largest_component: If ``True``, keep only the largest connected
                component among the selected faces.

        Returns:
            A tuple ``(vertex_indices, face_indices, sub_vertices, sub_faces)``
            where ``sub_vertices`` and ``sub_faces`` define the extracted
            sub-mesh with remapped indices.  Returns ``(None, None, None, None)``
            if no faces are selected.
        """
        faces = cast(np.ndarray, mesh.faces)
        face_mask = np.all(vertex_mask[faces], axis=1)
        face_indices = np.flatnonzero(face_mask)
        if face_indices.size == 0:
            return None, None, None, None
        if largest_component:
            face_indices = _AntipodalMeshBackend._largest_connected_face_component(
                mesh, face_indices
            )
            if face_indices.size == 0:
                return None, None, None, None
        return _AntipodalMeshBackend._build_sub_mesh(mesh, face_indices)

    @staticmethod
    def _build_sub_mesh(
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a sub-mesh with remapped vertex indices from selected faces.

        Returns:
            ``(vertex_indices, face_indices, sub_vertices, sub_faces)``
        """
        faces = cast(np.ndarray, mesh.faces)
        selected_face_vertices = faces[face_indices]
        vertex_indices = np.unique(selected_face_vertices.reshape(-1))

        old_to_new = np.full(mesh.vertices.shape[0], -1, dtype=np.int32)
        old_to_new[vertex_indices] = np.arange(vertex_indices.size, dtype=np.int32)

        sub_vertices = np.asarray(mesh.vertices)[vertex_indices]
        sub_faces = np.asarray(old_to_new)[selected_face_vertices]

        return vertex_indices, face_indices, sub_vertices, sub_faces

    @staticmethod
    def _largest_connected_face_component(
        mesh: trimesh.Trimesh,
        face_ids: np.ndarray,
    ) -> np.ndarray:
        """Return the face indices of the largest connected component."""
        if face_ids.size <= 1:
            return face_ids

        face_id_set = set(face_ids.tolist())
        parent: dict[int, int] = {int(face_id): int(face_id) for face_id in face_ids}

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != x:
                x_parent = parent[x]
                parent[x] = root
                x = x_parent
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        face_adjacency = cast(np.ndarray, mesh.face_adjacency)
        for face_a, face_b in face_adjacency:
            if int(face_a) in face_id_set and int(face_b) in face_id_set:
                union(int(face_a), int(face_b))

        groups: dict[int, list[int]] = {}
        for face_id in face_ids:
            root = find(int(face_id))
            groups.setdefault(root, []).append(int(face_id))

        largest_group = max(groups.values(), key=len)
        return np.array(largest_group, dtype=np.int32)

    @staticmethod
    def _apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        r = transform[:3, :3]
        t = transform[:3, 3]
        return points @ r.T + t

    def get_valid_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool = True,
        visualize_collision: bool = False,
    ):
        """Filter valid grasps, optionally to one projected half of the object.

        Args:
            object_pose: Current object pose with shape ``(4, 4)``.
            approach_direction: World-frame gripper approach direction.
            obj_longest_axis: Optional world-frame object axis. When ``None``,
                all annotated antipodal pairs remain eligible (center mode).
            is_positive_part: When an axis is supplied, select the positive
                projected half if true and the negative half otherwise.
            visualize_collision: Whether to visualize collision checks.

        Returns:
            Success, grasp poses, opening lengths, and grasp costs.
        """
        if self._hit_point_pairs is None:
            logger.log_warning(
                "No antipodal point pairs available. "
                "Prepare the mesh before requesting grasp poses."
            )
            return (
                False,
                torch.eye(4, device=self.device),
                0.0,
                torch.zeros(1, device=self.device),
            )
        origin_points = self._hit_point_pairs[:, 0, :]
        hit_points = self._hit_point_pairs[:, 1, :]
        origin_points_ = self._apply_transform(origin_points, object_pose)
        hit_points_ = self._apply_transform(hit_points, object_pose)
        mesh_vert_transformed = self._apply_transform(self.vertices, object_pose)

        if obj_longest_axis is None:
            origin_points_masked = origin_points_
            hit_points_masked = hit_points_
        else:
            axis = torch.as_tensor(
                obj_longest_axis,
                dtype=torch.float32,
                device=self.device,
            )
            if axis.shape != (3,) or not torch.isfinite(axis).all():
                raise ValueError("obj_longest_axis must be a finite (3,) tensor.")
            axis_norm = torch.linalg.vector_norm(axis)
            if axis_norm <= 1.0e-8:
                raise ValueError("obj_longest_axis must be non-zero.")
            if not isinstance(is_positive_part, bool):
                raise TypeError("is_positive_part must be a bool.")
            axis = axis / axis_norm
            mesh_projection = torch.matmul(mesh_vert_transformed, axis)
            mesh_projection_range = mesh_projection.max() - mesh_projection.min()
            projection_posi_threshold = (
                mesh_projection.min() + 0.65 * mesh_projection_range
            )
            projection_nega_threshold = (
                mesh_projection.min() + 0.35 * mesh_projection_range
            )
            pair_centers = 0.5 * (origin_points_ + hit_points_)
            pair_projection = torch.matmul(pair_centers, axis)
            if is_positive_part:
                part_mask = pair_projection > projection_posi_threshold
            else:
                part_mask = pair_projection < projection_nega_threshold
            origin_points_masked = origin_points_[part_mask]
            hit_points_masked = hit_points_[part_mask]
        return self._filter_valid_grasp_poses(
            origin_points_=origin_points_masked,
            hit_points_=hit_points_masked,
            object_pose=object_pose,
            approach_direction=approach_direction,
            mesh_vert_transformed=mesh_vert_transformed,
            visualize_collision=visualize_collision,
        )

    def get_dual_arm_valid_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        visualize_collision: bool = False,
    ) -> dict | None:
        if self._hit_point_pairs is None:
            logger.log_warning(
                "No antipodal point pairs available. "
                "Prepare the mesh before requesting grasp poses."
            )
            return None
        origin_points = self._hit_point_pairs[:, 0, :]
        hit_points = self._hit_point_pairs[:, 1, :]
        origin_points_ = self._apply_transform(origin_points, object_pose)
        hit_points_ = self._apply_transform(hit_points, object_pose)

        mesh_vert_transformed = self._apply_transform(self.vertices, object_pose)

        # project mesh_vert_transformed to left_to_right_arm_direction and get the min and max value
        n_vert = mesh_vert_transformed.shape[0]
        projected = (
            mesh_vert_transformed * left_to_right_arm_direction.repeat(n_vert, 1)
        ).sum(dim=-1)
        min_proj, max_proj = projected.min(), projected.max()
        left_threshold = min_proj + (max_proj - min_proj) * (
            0.5 - middle_empty_ratio / 2
        )
        right_threshold = max_proj - (max_proj - min_proj) * (
            0.5 - middle_empty_ratio / 2
        )

        origin_projected = (
            origin_points_
            * left_to_right_arm_direction.repeat(origin_points_.shape[0], 1)
        ).sum(dim=-1)
        hit_projected = (
            hit_points_ * left_to_right_arm_direction.repeat(hit_points_.shape[0], 1)
        ).sum(dim=-1)
        left_mask = (origin_projected < left_threshold) | (
            hit_projected < left_threshold
        )
        right_mask = (origin_projected > right_threshold) | (
            hit_projected > right_threshold
        )

        origin_left = origin_points_[left_mask]
        hit_left = hit_points_[left_mask]
        origin_right = origin_points_[right_mask]
        hit_right = hit_points_[right_mask]
        is_succes_left, grasp_poses_left, open_lengths_left, total_cost_left = (
            self._filter_valid_grasp_poses(
                hit_points_=hit_left,
                origin_points_=origin_left,
                object_pose=object_pose,
                approach_direction=approach_direction,
                mesh_vert_transformed=mesh_vert_transformed,
                visualize_collision=visualize_collision,
            )
        )
        is_succes_right, grasp_poses_right, open_lengths_right, total_cost_right = (
            self._filter_valid_grasp_poses(
                hit_points_=hit_right,
                origin_points_=origin_right,
                object_pose=object_pose,
                approach_direction=approach_direction,
                mesh_vert_transformed=mesh_vert_transformed,
                visualize_collision=visualize_collision,
            )
        )
        result = {
            "left": {
                "is_success": is_succes_left,
                "grasp_poses": grasp_poses_left,
                "open_lengths": open_lengths_left,
                "total_cost": total_cost_left,
            },
            "right": {
                "is_success": is_succes_right,
                "grasp_poses": grasp_poses_right,
                "open_lengths": open_lengths_right,
                "total_cost": total_cost_right,
            },
        }
        # self.visualize_grasp_poses(
        #     obj_pose=object_pose,
        #     grasp_poses=torch.vstack([grasp_poses_left, grasp_poses_right]),
        #     open_lengths=torch.cat([open_lengths_left, open_lengths_right]),
        # )
        return result

    def _filter_valid_grasp_poses(
        self,
        origin_points_: torch.Tensor,
        hit_points_: torch.Tensor,
        approach_direction: torch.Tensor,
        mesh_vert_transformed: torch.Tensor,
        object_pose: torch.Tensor,
        visualize_collision: bool = False,
    ):
        grasp_x = F.normalize(hit_points_ - origin_points_, dim=-1)
        cos_angle = torch.clamp((grasp_x * approach_direction).sum(dim=-1), -1.0, 1.0)
        positive_angle = torch.abs(torch.acos(cos_angle))
        valid_mask = (positive_angle - torch.pi / 2).abs() <= self._max_deviation_angle
        if valid_mask.sum() == 0:
            logger.log_warning("No valid antipodal pairs after angle filtering.")
            return (
                False,
                torch.eye(4, device=self.device),
                0.0,
                torch.zeros(1, device=self.device),
            )

        centers = (origin_points_ + hit_points_) / 2
        mesh_center = mesh_vert_transformed.mean(dim=0)

        valid_grasp_x = grasp_x[valid_mask]
        valid_centers = centers[valid_mask]
        valid_open_lengths = torch.norm(
            origin_points_[valid_mask] - hit_points_[valid_mask], dim=-1
        )

        # compute grasp poses using antipodal point pairs and approach direction
        approach_directions = [approach_direction]
        for _ in range(self._approach_direction_samples - 1):
            rota_direction = AntipodalSampler._random_rotate_unit_vectors(
                approach_direction.unsqueeze(0), self._max_deviation_angle
            )
            approach_directions.append(rota_direction[0])
        valid_grasp_poses_list = []
        for direct in approach_directions:
            valid_grasp_poses = (
                _AntipodalMeshBackend._grasp_pose_from_approach_direction(
                    valid_grasp_x,
                    direct,
                    valid_centers,
                )
            )
            valid_grasp_poses_list.append(valid_grasp_poses)
        valid_grasp_poses = torch.vstack(valid_grasp_poses_list)
        valid_grasp_x = valid_grasp_x.repeat(self._approach_direction_samples, 1)
        valid_centers = valid_centers.repeat(self._approach_direction_samples, 1)
        valid_open_lengths = valid_open_lengths.repeat(self._approach_direction_samples)

        # Compress near-identical candidates before the more expensive
        # collision query. Keep the per-pose metadata aligned with NMS output.
        valid_grasp_poses, nms_indices = pose_nms(
            valid_grasp_poses,
            angle_th=np.pi / 36,
            dist_th=0.005,
        )
        valid_open_lengths = valid_open_lengths[nms_indices]
        valid_centers = valid_centers[nms_indices]

        is_colliding, max_penetration = self._collision_checker.query(
            object_pose,
            valid_grasp_poses,
            valid_open_lengths,
            is_filter_ground_collision=self._filter_ground_collision,
            is_visual=visualize_collision,
            collision_threshold=0.0,
        )
        if is_colliding.logical_not().sum() == 0:
            logger.log_warning("No valid antipodal pairs after collision filtering.")
            return (
                False,
                torch.eye(4, device=self.device),
                0.0,
                torch.zeros(1, device=self.device),
            )

        # get best grasp pose
        valid_grasp_poses = valid_grasp_poses[~is_colliding]
        valid_open_lengths = valid_open_lengths[~is_colliding]
        valid_centers = valid_centers[~is_colliding]
        valid_grasp_x = F.normalize(valid_grasp_poses[:, :3, 0], dim=-1)

        cos_angle = torch.clamp(
            (valid_grasp_x * approach_direction).sum(dim=-1), -1.0, 1.0
        )
        positive_angle = torch.abs(torch.acos(cos_angle))
        angle_cost = torch.abs(positive_angle - 0.5 * torch.pi) / (0.5 * torch.pi)
        center_distance = torch.norm(valid_centers - mesh_center, dim=-1)
        center_cost = center_distance / center_distance.max()
        length_cost = 1 - valid_open_lengths / valid_open_lengths.max()
        total_cost = 0.25 * angle_cost + 0.25 * length_cost + 0.5 * center_cost

        n_valid = valid_grasp_poses.shape[0]
        if n_valid == 0:
            # no valid grasp pose
            return False, valid_grasp_poses, valid_open_lengths, total_cost
        if n_valid > self._max_candidates:
            # select only top-k grasps
            topk_indices = torch.topk(
                total_cost, self._max_candidates, largest=False
            ).indices
            top_grasp_poses = valid_grasp_poses[topk_indices]
            top_open_lengths = valid_open_lengths[topk_indices]
            top_total_cost = total_cost[topk_indices]
        else:
            top_grasp_poses = valid_grasp_poses
            top_open_lengths = valid_open_lengths
            top_total_cost = total_cost
        # self.visualize_grasp_poses(
        #     obj_pose=object_pose,
        #     grasp_poses=top_grasp_poses,
        #     open_lengths=top_open_lengths,
        # )
        return True, top_grasp_poses, top_open_lengths, top_total_cost

    def get_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        visualize_collision: bool = False,
        visualize_pose: bool = False,
    ) -> tuple[bool, torch.Tensor, float]:
        """Get grasp pose given approach direction.

        Uses the antipodal point pairs stored in ``self._hit_point_pairs``
        (populated by :meth:`annotate`).

        TODO:
            1. Support Top-k grasp poses selection.
            2. Support more selection criteria.

        Args:
            object_pose: ``(4, 4)`` homogeneous transformation matrix
                representing the pose of the object in the world frame.
            approach_direction: ``(3,)`` unit vector representing the desired
                approach direction of the gripper in the world frame.
            visualize_collision: If ``True``, enable visual collision checking.
            visualize_pose: If ``True``, visualize the best grasp pose using Open3D
                after computation.

        Returns:
            is_success (bool): Whether a valid grasp pose is found.
            best_grasp_pose (torch.Tensor): If a valid grasp pose is found, a tensor of shape (4, 4) representing the homogeneous transformation matrix of the best grasp pose in the world frame. Otherwise, an identity matrix.
            best_open_length (float): If a valid grasp pose is found, a scalar representing the optimal gripper opening length. Otherwise, a zero tensor.

        Raises:
            RuntimeError: If :meth:`annotate` has not been called yet.
        """
        is_success, valid_grasp_poses, valid_open_lengths, total_cost = (
            self.get_valid_grasp_poses(
                object_pose,
                approach_direction,
                visualize_collision=visualize_collision,
            )
        )
        if not is_success:
            return False, torch.eye(4, device=self.device), 0.0
        best_idx = torch.argmin(total_cost)
        best_grasp_pose = valid_grasp_poses[best_idx]
        best_open_length = valid_open_lengths[best_idx]
        if visualize_pose:
            self.visualize_grasp_pose(
                obj_pose=object_pose,
                grasp_pose=best_grasp_pose,
                open_length=best_open_length.item(),
            )
        return True, best_grasp_pose, best_open_length

    @staticmethod
    def _grasp_pose_from_approach_direction(
        grasp_x: torch.Tensor, approach_direction: torch.Tensor, center: torch.Tensor
    ):
        approach_direction_repeat = approach_direction[None, :].repeat(
            grasp_x.shape[0], 1
        )
        grasp_y = torch.cross(approach_direction_repeat, grasp_x, dim=-1)
        grasp_y = F.normalize(grasp_y, dim=-1)
        grasp_z = torch.cross(grasp_x, grasp_y, dim=-1)
        grasp_z = F.normalize(grasp_z, dim=-1)
        grasp_poses = (
            torch.eye(4, device=grasp_x.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(grasp_x.shape[0], 1, 1)
        )
        grasp_poses[:, :3, 0] = grasp_x
        grasp_poses[:, :3, 1] = grasp_y
        grasp_poses[:, :3, 2] = grasp_z
        grasp_poses[:, :3, 3] = center
        return grasp_poses

    def visualize_grasp_pose(
        self,
        obj_pose: torch.Tensor,
        grasp_pose: torch.Tensor,
        open_length: float,
    ):
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices.to("cpu").numpy()),
            triangles=o3d.utility.Vector3iVector(self.triangles.to("cpu").numpy()),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.6, 0.3])
        mesh.transform(obj_pose.to("cpu").numpy())
        vertices_ = torch.tensor(
            np.asarray(mesh.vertices),
            device=self.vertices.device,
            dtype=self.vertices.dtype,
        )
        mesh_scale = (vertices_.max(dim=0)[0] - vertices_.min(dim=0)[0]).max().item()
        groud_plane = o3d.geometry.TriangleMesh.create_cylinder(
            radius=mesh_scale, height=0.01 * mesh_scale
        )
        groud_plane.compute_vertex_normals()
        center = vertices_.mean(dim=0)
        z_sim = vertices_.min(dim=0)[0][2].item()
        groud_plane.translate(
            (center[0].item(), center[1].item(), z_sim - 0.005 * mesh_scale)
        )

        draw_thickness = 0.02 * mesh_scale
        draw_length = 0.3 * mesh_scale
        grasp_finger1 = o3d.geometry.TriangleMesh.create_box(
            draw_thickness, draw_thickness, draw_length
        )
        grasp_finger1.translate(
            (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
        )
        grasp_finger2 = o3d.geometry.TriangleMesh.create_box(
            draw_thickness, draw_thickness, draw_length
        )
        grasp_finger2.translate(
            (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
        )
        grasp_finger1.translate((-open_length / 2, 0, -0.25 * draw_length))
        grasp_finger2.translate((open_length / 2, 0, -0.25 * draw_length))
        grasp_root1 = o3d.geometry.TriangleMesh.create_box(
            open_length, draw_thickness, draw_thickness
        )
        grasp_root1.translate(
            (-open_length / 2, -0.5 * draw_thickness, -0.5 * draw_thickness)
        )
        grasp_root1.translate((0, 0, -0.75 * draw_length))
        grasp_root2 = o3d.geometry.TriangleMesh.create_box(
            draw_thickness, draw_thickness, draw_length
        )
        grasp_root2.translate(
            (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
        )
        grasp_root2.translate((0, 0, -1.25 * draw_length))

        grasp_visual = grasp_finger1 + grasp_finger2 + grasp_root1 + grasp_root2
        grasp_visual.paint_uniform_color([0.8, 0.2, 0.8])
        grasp_visual.transform(grasp_pose.to("cpu").numpy())
        o3d.visualization.draw_geometries(
            [grasp_visual, mesh, groud_plane],
            window_name="Grasp Pose Visualization",
            mesh_show_back_face=True,
        )

    def visualize_grasp_poses(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        open_lengths: torch.Tensor,
    ):
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices.to("cpu").numpy()),
            triangles=o3d.utility.Vector3iVector(self.triangles.to("cpu").numpy()),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.6, 0.3])
        mesh.transform(obj_pose.to("cpu").numpy())
        vertices_ = torch.tensor(
            np.asarray(mesh.vertices),
            device=self.vertices.device,
            dtype=self.vertices.dtype,
        )
        mesh_scale = (vertices_.max(dim=0)[0] - vertices_.min(dim=0)[0]).max().item()
        groud_plane = o3d.geometry.TriangleMesh.create_cylinder(
            radius=mesh_scale, height=0.01 * mesh_scale
        )
        groud_plane.compute_vertex_normals()
        center = vertices_.mean(dim=0)
        z_sim = vertices_.min(dim=0)[0][2].item()
        groud_plane.translate(
            (center[0].item(), center[1].item(), z_sim - 0.005 * mesh_scale)
        )
        draw_thickness = 0.02 * mesh_scale
        draw_length = 0.3 * mesh_scale
        visual_mesh_list = [mesh, groud_plane]
        for i in range(grasp_poses.shape[0]):
            grasp_finger1 = o3d.geometry.TriangleMesh.create_box(
                draw_thickness, draw_thickness, draw_length
            )
            grasp_finger1.translate(
                (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
            )
            grasp_finger2 = o3d.geometry.TriangleMesh.create_box(
                draw_thickness, draw_thickness, draw_length
            )
            grasp_finger2.translate(
                (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
            )
            grasp_finger1.translate((-open_lengths[i] / 2, 0, -0.25 * draw_length))
            grasp_finger2.translate((open_lengths[i] / 2, 0, -0.25 * draw_length))
            grasp_root1 = o3d.geometry.TriangleMesh.create_box(
                open_lengths[i], draw_thickness, draw_thickness
            )
            grasp_root1.translate(
                (-open_lengths[i] / 2, -0.5 * draw_thickness, -0.5 * draw_thickness)
            )
            grasp_root1.translate((0, 0, -0.75 * draw_length))
            grasp_root2 = o3d.geometry.TriangleMesh.create_box(
                draw_thickness, draw_thickness, draw_length
            )
            grasp_root2.translate(
                (-0.5 * draw_thickness, -0.5 * draw_thickness, -0.5 * draw_length)
            )
            grasp_root2.translate((0, 0, -1.25 * draw_length))

            grasp_visual = grasp_finger1 + grasp_finger2 + grasp_root1 + grasp_root2
            grasp_visual.paint_uniform_color([0.8, 0.2, 0.8])
            grasp_visual.transform(grasp_poses[i].to("cpu").numpy())
            visual_mesh_list.append(grasp_visual)
        o3d.visualization.draw_geometries(
            visual_mesh_list,
            window_name="Grasp Pose Visualization",
            mesh_show_back_face=True,
        )
