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

import os
import argparse
import open3d as o3d
import time
from pathlib import Path
from typing import Any, cast
import torch
import numpy as np
import trimesh

import viser
import viser.transforms as tf
from embodichain.utils import logger
from dataclasses import dataclass
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSampler,
    AntipodalSamplerCfg,
)
from .gripper_collision_checker import (
    SimpleGripperCollisionChecker,
    SimpleGripperCollisionCfg,
)
import hashlib
import torch.nn.functional as F
import tempfile


@dataclass
class GraspAnnotatorCfg:
    viser_port: int = 15531
    use_largest_connected_component: bool = False
    antipodal_sampler_cfg: AntipodalSamplerCfg = AntipodalSamplerCfg()
    force_regenerate: bool = False
    max_deviation_angle: float = np.pi / 12


@dataclass
class SelectResult:
    vertex_indices: np.ndarray | None = None
    face_indices: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


class GraspAnnotator:
    """GraspAnnotator provides functionality to annotate antipodal grasp regions on a given object mesh. It allows users to interactively select regions on the mesh and generates antipodal point pairs for grasping based on the selected region. The annotator also includes a collision checker to filter out infeasible grasp poses and can visualize the generated grasp poses in a 3D viewer.
    """
    def __init__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        cfg: GraspAnnotatorCfg = GraspAnnotatorCfg(),
    ) -> None:
        """Initialize the GraspAnnotator with the given mesh vertices, triangles, and configuration.
        Args:
            vertices (torch.Tensor): A tensor of shape (V, 3) representing the vertex positions of the mesh.
            triangles (torch.Tensor): A tensor of shape (F, 3) representing the triangle indices of the mesh.
            cfg (GraspAnnotatorCfg, optional): Configuration for the grasp annotator. Defaults to GraspAnnotatorCfg().
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
        self._collision_checker = SimpleGripperCollisionChecker(
            object_mesh_verts=vertices,
            object_mesh_faces=triangles,
            cfg=SimpleGripperCollisionCfg(),
        )
        self.cfg = cfg
        self.antipodal_sampler = AntipodalSampler(cfg=cfg.antipodal_sampler_cfg)

    def annotate(self) -> torch.Tensor:
        """Annotate antipodal grasp region on the mesh and return sampled antipodal point pairs.
        Returns:
            torch.Tensor: A tensor of shape (N, 2, 3) representing N antipodal point pairs. Each pair consists of a hit point and its corresponding surface point.
        """
        cache_path = self._get_cache_dir(self.vertices, self.triangles)
        if os.path.exists(cache_path) and not self.cfg.force_regenerate:
            logger.log_info(
                f"Found existing antipodal retult. Loading cached antipodal pairs from {cache_path}"
            )
            hit_point_pairs = torch.tensor(
                np.load(cache_path), dtype=torch.float32, device=self.device
            )
            return hit_point_pairs
        else:
            logger.log_info(
                f"[Viser] *****Annotate grasp region in http://localhost:{self.cfg.viser_port}"
            )

        server = viser.ViserServer(port=self.cfg.viser_port)
        server.gui.configure_theme(brand_color=(130, 0, 150))
        server.scene.set_up_direction("+z")

        mesh_handle = server.scene.add_mesh_trimesh(name="/mesh", mesh=self.mesh)
        selected_overlay: viser.GlbHandle | None = None
        selection: SelectResult = SelectResult()

        hit_point_pairs = None
        return_flag = False

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            nonlocal mesh_handle
            nonlocal selected_overlay
            nonlocal selection

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
                    nonlocal selection
                    nonlocal hit_point_pairs
                    client.scene.remove_pointer_callback()

                    proj, depth = GraspAnnotator._project_vertices_to_screen(
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

                    selection = GraspAnnotator._extract_selection(
                        self.mesh, vertex_mask, self.cfg.use_largest_connected_component
                    )
                    if selection.vertices is None:
                        logger.log_warning("[Selection] No vertices selected.")
                        return

                    color_mesh = self.mesh.copy()
                    used_vertex_indices = selection.vertex_indices
                    vertex_colors = np.tile(
                        np.array([[0.85, 0.85, 0.85, 1.0]]),
                        (self.mesh.vertices.shape[0], 1),
                    )
                    vertex_colors[used_vertex_indices] = np.array(
                        [0.56, 0.17, 0.92, 1.0]
                    )
                    color_mesh.visual.vertex_colors = vertex_colors  # type: ignore
                    mesh_handle = server.scene.add_mesh_trimesh(
                        name="/mesh", mesh=color_mesh
                    )

                    if selected_overlay is not None:
                        selected_overlay.remove()
                    selected_mesh = trimesh.Trimesh(
                        vertices=selection.vertices,
                        faces=selection.faces,
                        process=False,
                    )
                    selected_mesh.visual.face_colors = (0.9, 0.2, 0.2, 0.65)  # type: ignore
                    selected_overlay = server.scene.add_mesh_trimesh(
                        name="/selected", mesh=selected_mesh
                    )
                    logger.log_info(
                        f"[Selection] Selected {selection.vertex_indices.size} vertices and {selection.face_indices.size} faces."
                    )

                    hit_point_pairs = self.antipodal_sampler.sample(
                        torch.tensor(selection.vertices, device=self.device),
                        torch.tensor(selection.faces, device=self.device),
                    )
                    extended_hit_point_pairs = GraspAnnotator._extend_hit_point_pairs(
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
                if selection.vertices is None:
                    logger.log_warning("[Selection] No vertex selected.")
                    return
                else:
                    logger.log_info(
                        f"[Selection] {selection.vertices.shape[0]}vertices selected. Generating antipodal point pairs."
                    )
                    return_flag = True

        while True:
            if return_flag:
                # save result to cache
                if hit_point_pairs is not None:
                    self._save_cache(cache_path, hit_point_pairs)
                break
            time.sleep(0.5)
        return hit_point_pairs

    def _get_cache_dir(self, vertices: torch.Tensor, triangles: torch.Tensor):
        vert_bytes = vertices.to("cpu").numpy().tobytes()
        face_bytes = triangles.to("cpu").numpy().tobytes()
        md5_hash = hashlib.md5(vert_bytes + face_bytes).hexdigest()
        cache_path = os.path.join(
            tempfile.gettempdir(), f"antipodal_cache_{md5_hash}.npy"
        )
        return cache_path

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

    def _extract_selection(
        mesh: trimesh.Trimesh,
        vertex_mask: np.ndarray,
        largest_component: bool,
    ) -> SelectResult:
        def _largest_connected_face_component(face_ids: np.ndarray) -> np.ndarray:
            if face_ids.size <= 1:
                return face_ids

            face_id_set = set(face_ids.tolist())
            parent: dict[int, int] = {
                int(face_id): int(face_id) for face_id in face_ids
            }

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

        faces = cast(np.ndarray, mesh.faces)
        face_mask = np.all(vertex_mask[faces], axis=1)

        face_indices = np.flatnonzero(face_mask)
        if face_indices.size == 0:
            return SelectResult()
        if largest_component:
            face_indices = _largest_connected_face_component(face_indices)
            if face_indices.size == 0:
                return SelectResult()

        selected_face_vertices = faces[face_indices]
        vertex_indices = np.unique(selected_face_vertices.reshape(-1))

        old_to_new = np.full(mesh.vertices.shape[0], -1, dtype=np.int32)
        old_to_new[vertex_indices] = np.arange(vertex_indices.size, dtype=np.int32)

        sub_vertices = np.asarray(mesh.vertices)[vertex_indices]
        sub_faces = np.asarray(old_to_new)[selected_face_vertices]

        return SelectResult(
            vertex_indices=vertex_indices,
            face_indices=face_indices,
            vertices=sub_vertices,
            faces=sub_faces,
        )

    @staticmethod
    def _apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        r = transform[:3, :3]
        t = transform[:3, 3]
        return points @ r.T + t

    def get_grasp_poses(
        self,
        hit_point_pairs: torch.Tensor,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        is_visual: bool = False,
    ) -> torch.Tensor:
        """Get grasp pose given approach direction

        Args:
            hit_point_pairs (torch.Tensor): (N, 2, 3) tensor of N antipodal point pairs. Each pair consists of a hit point and its corresponding surface point.
            object_pose (torch.Tensor): (4, 4) homogeneous transformation matrix representing the pose of the object in the world frame.
            approach_direction (torch.Tensor): (3,) unit vector representing the desired approach direction of the gripper in the world frame.

        Returns:
            torch.Tensor: (4, 4) homogeneous transformation matrix representing the grasp pose in the world frame that aligns the gripper's approach direction with the given approach_direction. Returns None if no valid grasp pose can be found.
        """
        origin_points = hit_point_pairs[:, 0, :]
        hit_points = hit_point_pairs[:, 1, :]
        origin_points_ = self._apply_transform(origin_points, object_pose)
        hit_points_ = self._apply_transform(hit_points, object_pose)
        centers = (origin_points_ + hit_points_) / 2

        mesh_vert_transformed = self._apply_transform(self.vertices, object_pose)
        mesh_center = mesh_vert_transformed.mean(dim=0)

        # filter perpendicular antipodal point
        grasp_x = F.normalize(hit_points_ - origin_points_, dim=-1)
        cos_angle = torch.clamp((grasp_x * approach_direction).sum(dim=-1), -1.0, 1.0)
        positive_angle = torch.abs(torch.acos(cos_angle))
        valid_mask = (
            positive_angle - torch.pi / 2
        ).abs() <= self.cfg.max_deviation_angle
        valid_grasp_x = grasp_x[valid_mask]
        valid_centers = centers[valid_mask]

        # compute grasp poses using antipodal point pairs and approach direction
        valid_grasp_poses = GraspAnnotator._grasp_pose_from_approach_direction(
            valid_grasp_x, approach_direction, valid_centers
        )
        valid_open_lengths = torch.norm(
            origin_points_[valid_mask] - hit_points_[valid_mask], dim=-1
        )
        # select non-collide grasp poses
        is_colliding, max_penetration = self._collision_checker.query(
            object_pose,
            valid_grasp_poses,
            valid_open_lengths,
            is_visual=is_visual,
            collision_threshold=0.0,
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
        total_cost = 0.3 * angle_cost + 0.3 * length_cost + 0.4 * center_cost
        best_idx = torch.argmin(total_cost)
        best_grasp_pose = valid_grasp_poses[best_idx]
        best_open_length = valid_open_lengths[best_idx]
        return best_grasp_pose, best_open_length

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viser mesh 标注工具：框选并导出对应顶点与三角面"
    )
    parser.add_argument(
        "--mesh", type=Path, required=True, help="输入 mesh 文件路径，例如 mug.obj"
    )
    parser.add_argument("--scale", type=float, default=1.0, help="加载后整体缩放系数")
    parser.add_argument("--port", type=int, default=12151, help="viser 服务端口")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mesh_annotations"),
        help="标注结果导出目录",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="只保留框选结果中的最大连通块（常用于稳定提取把手等局部）",
    )
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    vertices = mesh.vertices * args.scale
    triangles = mesh.faces
    cfg = GraspAnnotatorCfg(
        force_regenerate=True,
    )
    tool = GraspAnnotator(cfg=cfg)
    hit_point_pairs = tool.annotate(
        vertices=torch.from_numpy(vertices).float(),
        triangles=torch.from_numpy(triangles).long(),
    )
    logger.log_info(f"Sample {hit_point_pairs.shape[0]} antipodal point pairs.")


if __name__ == "__main__":
    main()
