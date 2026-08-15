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

import os
import argparse
from collections.abc import Callable
from copy import deepcopy
import json
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
from embodichain.utils import configclass
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSampler,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    GripperCollisionChecker,
    GripperCollisionCfg,
)

GRASP_ANNOTATOR_CACHE_DIR = (
    Path.home() / ".cache" / "embodichain" / "grasp_annotator_cache"
)
GRASP_ANNOTATOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VERSION_TAG = "v0.0.2"


__all__ = ["GraspGenerator", "GraspGeneratorCfg", "antipodal_cache_key"]


def antipodal_cache_key(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    cfg: AntipodalSamplerCfg,
) -> str:
    """Return the stage-aware identity for raw antipodal point pairs.

    Raw pairs depend on object/submesh content and antipodal sampling policy,
    but not on end-effector collision geometry or downstream approach-pose
    deviations. Keeping those stages out of this key avoids invalidating an
    expensive mesh sample for unrelated planner changes.

    Args:
        vertices: Mesh vertices consumed by the sampler.
        triangles: Mesh triangle indices consumed by the sampler.
        cfg: Raw antipodal sampling policy.

    Returns:
        Stable cache key containing the algorithm version and content hashes.
    """
    mesh_hash = hashlib.sha256(
        vertices.detach().to("cpu").contiguous().numpy().tobytes()
        + triangles.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()
    policy_payload = json.dumps(
        {
            "max_angle": float(cfg.max_angle),
            "max_length": float(cfg.max_length),
            "min_length": float(cfg.min_length),
            "n_sample": int(cfg.n_sample),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    policy_hash = hashlib.sha256(policy_payload).hexdigest()
    return f"{VERSION_TAG}_{mesh_hash}_{policy_hash}"


@configclass
class GraspGeneratorCfg:
    """Configuration for :class:`GraspGenerator`.

    Controls the interactive grasp region annotation workflow, including the
    browser-based visualizer settings, antipodal sampling parameters, and
    grasp-pose filtering thresholds.
    """

    viser_port: int = 15531
    """Port used by the Viser browser-based visualizer for interactive grasp
    region annotation."""

    use_largest_connected_component: bool = False
    """When ``True``, only the largest connected component of the selected mesh
    region is retained. Useful for meshes that contain disconnected fragments
    or when selecting a local feature such as a handle."""

    antipodal_sampler_cfg: AntipodalSamplerCfg = AntipodalSamplerCfg()
    """Nested configuration for the antipodal point sampler. Controls the
    number of sampled surface points, ray perturbation angle, and gripper jaw
    distance limits. See :class:`AntipodalSamplerCfg` for details."""

    max_deviation_angle: float = np.pi / 6
    """Maximum allowed angle (in radians) between the specified approach
    direction and the axis connecting an antipodal point pair. Pairs that
    deviate more than this threshold from perpendicular to the approach are
    discarded during grasp pose computation."""

    n_deviated_approach_directions: int = 4
    """Number of approach directions with evenly deviated angles when sampling grasp poses."""

    n_top_grasps: int = 50
    """Number of top-ranked grasp poses to return based on the scoring cost."""

    is_partial_annotate: bool = False
    """When ``True``, the annotator allows selecting a partial region of the 
    mesh for grasp sampling. If ``False``, the entire mesh is used."""

    is_filter_ground_collision: bool = True
    """Whether to filter out grasp poses that would cause the gripper to 
    collide."""


class GraspGenerator:
    """Antipodal grasp-pose generator for parallel-jaw grippers.

    Given an object mesh, ``GraspGenerator`` produces feasible grasp poses
    through a three-stage pipeline:

    1. **Antipodal sampling** — Surface points are uniformly sampled and
       rays are cast along (and near) the inward normal to find antipodal
       point pairs on opposite sides of the mesh (:meth:`generate`).
       Alternatively, an interactive Viser-based annotator lets a human
       select the graspable region (:meth:`annotate`).
    2. **Pose construction** — For each antipodal pair, a 6-DoF grasp
       frame is built so that the gripper opening aligns with the pair axis
       and the approach direction is consistent with a user-specified
       vector (:meth:`get_grasp_poses`).
    3. **Filtering & ranking** — Grasp candidates that would cause the
       gripper to collide with the object are discarded.  Surviving poses
       are scored by a weighted cost that penalises angular deviation from
       the approach direction, narrow opening length, and distance to the
       mesh centroid.

    Typical usage::

        generator = GraspGenerator(vertices, triangles, cfg=cfg)

        # Programmatic: sample on the whole mesh or a sub-region
        generator.generate()                       # whole mesh
        generator.generate(face_indices=some_idx)  # specific faces

        # Interactive: pick region in a browser UI
        generator.annotate()

        # Then compute the best grasp pose
        pose, open_length = generator.get_grasp_poses(object_pose, approach_dir)
    """

    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        cfg: GraspGeneratorCfg = GraspGeneratorCfg(),
        gripper_collision_cfg: GripperCollisionCfg = GripperCollisionCfg(),
    ) -> None:
        """Initialize the GraspGenerator with the given mesh vertices, triangles, and configuration.
        Args:
            vertices (torch.Tensor): A tensor of shape (V, 3) representing the vertex positions of the mesh.
            triangles (torch.Tensor): A tensor of shape (F, 3) representing the triangle indices of the mesh.
            cfg (GraspGeneratorCfg, optional): Configuration for the grasp annotator. Defaults to GraspGeneratorCfg().
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
            cfg=gripper_collision_cfg,
        )
        self.cfg = cfg
        self._antipodal_sampler = AntipodalSampler(cfg=cfg.antipodal_sampler_cfg)
        self._hit_point_pairs: torch.Tensor | None = None
        self._last_filter_diagnostics: dict[str, Any] = {}

        # Load cached antipodal pairs for the whole mesh if available.
        cache_path = self._get_cache_dir(self.vertices, self.triangles)
        if os.path.exists(cache_path):
            logger.log_info(f"Found cached antipodal pairs at {cache_path}. Loading.")
            self._hit_point_pairs = torch.tensor(
                np.load(cache_path), dtype=torch.float32, device=self.device
            )

    def generate(
        self,
        vertex_indices: torch.Tensor | None = None,
        face_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate antipodal point pairs for grasping on the given mesh region.

        Exactly one of ``vertex_indices`` or ``face_indices`` must be provided
        to define the grasp region.  When both are ``None``, the whole mesh is
        used.

        Results are cached to disk.

        Args:
            vertex_indices: 1-D ``torch.Tensor`` of vertex indices defining the
                grasp region.
            face_indices: 1-D ``torch.Tensor`` of face indices defining the
                grasp region.

        Raises:
            ValueError: If both ``vertex_indices`` and ``face_indices`` are
                provided at the same time.

        Returns:
            torch.Tensor: A tensor of shape ``(N, 2, 3)`` representing N
                antipodal point pairs.  Each pair consists of a hit point and
                its corresponding surface point.
        """
        if vertex_indices is not None and face_indices is not None:
            raise ValueError(
                "Only one of vertex_indices or face_indices should be provided, not both."
            )

        if vertex_indices is None and face_indices is None:
            sub_vertices = self.vertices
            sub_faces = self.triangles
        else:
            if face_indices is not None:
                face_idx_np = face_indices.cpu().numpy()
            else:
                vertex_idx_np = vertex_indices.cpu().numpy()
                vertex_mask = np.zeros(self.mesh.vertices.shape[0], dtype=bool)
                vertex_mask[vertex_idx_np] = True
                face_all = cast(np.ndarray, self.mesh.faces)
                face_idx_np = np.flatnonzero(np.all(vertex_mask[face_all], axis=1))
            (
                _,
                _,
                sub_vertices_np,
                sub_faces_np,
            ) = GraspGenerator._extract_selection_from_faces(
                self.mesh, face_idx_np, self.cfg.use_largest_connected_component
            )
            if sub_vertices_np is None:
                return torch.empty(0, 2, 3, dtype=torch.float32, device=self.device)
            sub_vertices = torch.as_tensor(
                sub_vertices_np, dtype=torch.float32, device=self.device
            )
            sub_faces = torch.as_tensor(
                sub_faces_np, dtype=torch.int64, device=self.device
            )

        cache_path = self._get_cache_dir(sub_vertices, sub_faces)
        if os.path.exists(cache_path):
            logger.log_info(f"Found cached antipodal pairs at {cache_path}")
            return torch.tensor(
                np.load(cache_path), dtype=torch.float32, device=self.device
            )

        self._hit_point_pairs = self._antipodal_sampler.sample(sub_vertices, sub_faces)
        self._save_cache(cache_path, self._hit_point_pairs)
        return self._hit_point_pairs

    def annotate(self) -> torch.Tensor:
        """Annotate antipodal grasp region on the mesh and return sampled antipodal point pairs.

        Returns:
            torch.Tensor: A tensor of shape (N, 2, 3) representing N antipodal point pairs.
                Each pair consists of a hit point and its corresponding surface point.
        """
        if self.cfg.is_partial_annotate == False:
            hit_point_pairs = self._generate_hit_point_pairs(
                self.vertices, self.triangles
            )
            self._cache_hit_point_pairs(hit_point_pairs)
            return self._hit_point_pairs
        logger.log_info(
            f"[Viser] *****Annotate grasp region in http://localhost:{self.cfg.viser_port}"
        )

        server = viser.ViserServer(port=self.cfg.viser_port)
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

                    proj, depth = GraspGenerator._project_vertices_to_screen(
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
                    ) = GraspGenerator._extract_selection_from_vertex_mask(
                        self.mesh, vertex_mask, self.cfg.use_largest_connected_component
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
                    extended_hit_point_pairs = GraspGenerator._extend_hit_point_pairs(
                        hit_point_pairs
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
        key = antipodal_cache_key(
            vertices,
            triangles,
            self.cfg.antipodal_sampler_cfg,
        )
        cache_path = os.path.join(
            GRASP_ANNOTATOR_CACHE_DIR,
            f"antipodal_cache_{key}.npy",
        )
        return cache_path

    @property
    def last_filter_diagnostics(self) -> dict[str, Any]:
        """Return a detached trace for the most recent grasp-filtering call."""
        return deepcopy(self._last_filter_diagnostics)

    def _save_cache(self, cache_path: str, hit_point_pairs: torch.Tensor):
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
            face_indices = GraspGenerator._largest_connected_face_component(
                mesh, face_indices
            )
            if face_indices.size == 0:
                return None, None, None, None
        return GraspGenerator._build_sub_mesh(mesh, face_indices)

    @staticmethod
    def _extract_selection_from_faces(
        mesh: trimesh.Trimesh,
        face_indices: np.ndarray,
        largest_component: bool,
    ) -> tuple[
        np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
    ]:
        """Extract a sub-mesh from *mesh* using face indices.

        Args:
            mesh: The source mesh.
            face_indices: Array of face indices to include.
            largest_component: If ``True``, keep only the largest connected
                component among the selected faces.

        Returns:
            Same as :meth:`_extract_selection_from_vertex_mask`.
        """
        if face_indices.size == 0:
            return None, None, None, None
        if largest_component:
            face_indices = GraspGenerator._largest_connected_face_component(
                mesh, face_indices
            )
            if face_indices.size == 0:
                return None, None, None, None
        return GraspGenerator._build_sub_mesh(mesh, face_indices)

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

    @staticmethod
    def _deterministic_approach_directions(
        direction: torch.Tensor,
        *,
        count: int,
        max_angle: float,
        attempt_id: int,
    ) -> list[torch.Tensor]:
        """Enumerate a reproducible low-discrepancy cone around ``direction``."""
        if count <= 0:
            raise ValueError("count must be positive.")
        if attempt_id < 0:
            raise ValueError("attempt_id must be non-negative.")
        base = F.normalize(direction, dim=0)
        approaches = [base] if attempt_id == 0 else []
        if count == 1 or max_angle <= 0.0:
            return [base]

        reference_index = int(torch.argmin(torch.abs(base)).item())
        reference = torch.zeros_like(base)
        reference[reference_index] = 1.0
        tangent = F.normalize(torch.cross(base, reference, dim=0), dim=0)
        bitangent = torch.cross(base, tangent, dim=0)
        golden_ratio_conjugate = (5.0**0.5 - 1.0) / 2.0
        per_attempt = count - len(approaches)
        sequence_start = (
            0 if attempt_id == 0 else (count - 1) + (attempt_id - 1) * count
        )
        for offset in range(per_attempt):
            sequence_index = sequence_start + offset + 1
            fraction = (sequence_index * golden_ratio_conjugate) % 1.0
            polar = float(max_angle) * fraction**0.5
            azimuth = base.new_tensor(2.0 * torch.pi * fraction)
            radial = (
                torch.cos(azimuth) * tangent
                + torch.sin(azimuth) * bitangent
            )
            approaches.append(
                torch.cos(base.new_tensor(polar)) * base
                + torch.sin(base.new_tensor(polar)) * radial
            )
        return approaches

    def get_valid_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        object_part: str = "center",
        visualize_collision: bool = False,
        pose_cost_fn: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        approach_attempt_id: int = 0,
    ):
        self._last_filter_diagnostics = {
            "mode": "single_arm",
            "raw_pair_count": (
                0 if self._hit_point_pairs is None else len(self._hit_point_pairs)
            ),
            "approach_attempt_id": int(approach_attempt_id),
        }
        if self._hit_point_pairs is None:
            logger.log_warning(
                "No antipodal point pairs available. "
                "Call generate() or annotate() first."
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

        if object_part == "bottom":
            z_max = mesh_vert_transformed[:, 2].max()
            z_min = mesh_vert_transformed[:, 2].min()
            z_threshold = z_min + (z_max - z_min) * 0.45
            z_mask = (origin_points_[:, 2] < z_threshold) | (
                hit_points_[:, 2] < z_threshold
            )
            origin_points_masked = origin_points_[z_mask]
            hit_points_masked = hit_points_[z_mask]
        elif object_part == "top":
            z_max = mesh_vert_transformed[:, 2].max()
            z_min = mesh_vert_transformed[:, 2].min()
            z_threshold = z_min + (z_max - z_min) * 0.6
            z_mask = (origin_points_[:, 2] > z_threshold) | (
                hit_points_[:, 2] > z_threshold
            )
            origin_points_masked = origin_points_[z_mask]
            hit_points_masked = hit_points_[z_mask]
        else:
            origin_points_masked = origin_points_
            hit_points_masked = hit_points_
        return self._filter_valid_grasp_poses(
            origin_points_=origin_points_masked,
            hit_points_=hit_points_masked,
            object_pose=object_pose,
            approach_direction=approach_direction,
            mesh_vert_transformed=mesh_vert_transformed,
            visualize_collision=visualize_collision,
            pose_cost_fn=pose_cost_fn,
            stage_name=str(object_part),
            approach_attempt_id=approach_attempt_id,
        )

    def get_dual_arm_valid_grasp_poses(
        self,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        visualize_collision: bool = False,
        approach_attempt_id: int = 0,
    ) -> dict | None:
        self._last_filter_diagnostics = {
            "mode": "dual_arm",
            "raw_pair_count": (
                0 if self._hit_point_pairs is None else len(self._hit_point_pairs)
            ),
            "approach_attempt_id": int(approach_attempt_id),
        }
        if self._hit_point_pairs is None:
            logger.log_warning(
                "No antipodal point pairs available. "
                "Call generate() or annotate() first."
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
        self._last_filter_diagnostics["partition"] = {
            "middle_empty_ratio": float(middle_empty_ratio),
            "left_pair_count": int(left_mask.sum().item()),
            "right_pair_count": int(right_mask.sum().item()),
        }
        is_succes_left, grasp_poses_left, open_lengths_left, total_cost_left = (
            self._filter_valid_grasp_poses(
                hit_points_=hit_left,
                origin_points_=origin_left,
                object_pose=object_pose,
                approach_direction=approach_direction,
                mesh_vert_transformed=mesh_vert_transformed,
                visualize_collision=visualize_collision,
                stage_name="left",
                approach_attempt_id=approach_attempt_id,
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
                stage_name="right",
                approach_attempt_id=approach_attempt_id,
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
        pose_cost_fn: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        stage_name: str = "grasp",
        approach_attempt_id: int = 0,
    ):
        stage_trace: dict[str, Any] = {
            "input_pair_count": int(origin_points_.shape[0]),
        }
        self._last_filter_diagnostics[stage_name] = stage_trace
        grasp_x = F.normalize(hit_points_ - origin_points_, dim=-1)
        cos_angle = torch.clamp((grasp_x * approach_direction).sum(dim=-1), -1.0, 1.0)
        positive_angle = torch.abs(torch.acos(cos_angle))
        valid_mask = (
            positive_angle - torch.pi / 2
        ).abs() <= self.cfg.max_deviation_angle
        stage_trace["angle_valid_pair_count"] = int(valid_mask.sum().item())
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
        approach_directions = self._deterministic_approach_directions(
            approach_direction,
            count=self.cfg.n_deviated_approach_directions,
            max_angle=self.cfg.max_deviation_angle,
            attempt_id=approach_attempt_id,
        )
        valid_grasp_poses_list = []
        for direct in approach_directions:
            valid_grasp_poses = GraspGenerator._grasp_pose_from_approach_direction(
                valid_grasp_x, direct, valid_centers
            )
            valid_grasp_poses_list.append(valid_grasp_poses)
        valid_grasp_poses = torch.vstack(valid_grasp_poses_list)
        valid_grasp_x = valid_grasp_x.repeat(self.cfg.n_deviated_approach_directions, 1)
        valid_centers = valid_centers.repeat(self.cfg.n_deviated_approach_directions, 1)
        valid_open_lengths = valid_open_lengths.repeat(
            self.cfg.n_deviated_approach_directions
        )
        stage_trace["pose_candidate_count"] = int(valid_grasp_poses.shape[0])

        # TODO: too slow
        # # remove near grasp poses using non-maximum suppression
        # nms_indices = pose_nms_indices(
        #     valid_grasp_poses,
        #     angle_th=np.pi / 18,
        #     dist_th=0.01,
        # )
        # valid_grasp_poses = valid_grasp_poses[nms_indices]
        # valid_open_lengths = valid_open_lengths[nms_indices]
        # valid_centers = valid_centers[nms_indices]

        is_colliding, max_penetration = self._collision_checker.query(
            object_pose,
            valid_grasp_poses,
            valid_open_lengths,
            is_filter_ground_collision=self.cfg.is_filter_ground_collision,
            is_visual=visualize_collision,
            collision_threshold=0.0,
        )
        stage_trace["collision"] = self._collision_checker.last_query_diagnostics
        stage_trace["collision_free_pose_count"] = int(
            is_colliding.logical_not().sum().item()
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
        total_cost = 0.2 * angle_cost + 0.2 * length_cost + 0.6 * center_cost
        if pose_cost_fn is not None:
            adjusted_cost = pose_cost_fn(valid_grasp_poses, total_cost)
            if adjusted_cost.shape != total_cost.shape:
                logger.log_error(
                    "pose_cost_fn must preserve the grasp cost shape.",
                    ValueError,
                )
            total_cost = adjusted_cost.to(
                device=total_cost.device,
                dtype=total_cost.dtype,
            )

        n_valid = valid_grasp_poses.shape[0]
        if n_valid == 0:
            # no valid grasp pose
            return False, valid_grasp_poses, valid_open_lengths, total_cost
        if n_valid > self.cfg.n_top_grasps:
            # select only top-k grasps
            topk_indices = torch.topk(
                total_cost, self.cfg.n_top_grasps, largest=False
            ).indices
            top_grasp_poses = valid_grasp_poses[topk_indices]
            top_open_lengths = valid_open_lengths[topk_indices]
            top_total_cost = total_cost[topk_indices]
        else:
            top_grasp_poses = valid_grasp_poses
            top_open_lengths = valid_open_lengths
            top_total_cost = total_cost
        stage_trace["returned_pose_count"] = int(top_grasp_poses.shape[0])
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
        (populated by :meth:`generate` or :meth:`annotate`).

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
            RuntimeError: If :meth:`generate` or :meth:`annotate` has not
                been called yet.
        """
        is_success, valid_grasp_poses, valid_open_lengths, total_cost = (
            self.get_valid_grasp_poses(
                object_pose, approach_direction, visualize_collision
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
