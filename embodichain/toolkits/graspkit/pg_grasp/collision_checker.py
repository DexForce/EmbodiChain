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

import trimesh
import numpy as np
import torch
import warp as wp
import time
import hashlib
import os
import pickle
import open3d as o3d

from typing import List, Tuple, Union
from dexsim.kit.meshproc import convex_decomposition_coacd

from embodichain.utils.warp import convex_signed_distance_kernel
from embodichain.utils.device_utils import standardize_device_string
from embodichain.utils.math import transform_points_mat
from embodichain.utils import configclass

__all__ = ["ConvexCollisionCheckerCfg", "ConvexCollisionChecker"]


@configclass
class ConvexCollisionCheckerCfg:
    """Configuration for ConvexCollisionChecker."""

    collision_threshold: float = 0.0
    """ Collision threshold in meters. A point is considered colliding if its signed distance to the hull interior is <= this threshold. This allows for a margin of error in collision checking, where a small positive threshold can be used to consider points near the surface as colliding, and a small negative threshold can be used to allow for slight penetration without considering it a collision."""

    n_query_mesh_samples: int = 4096
    """ Number of points to sample from the query mesh surface for collision checking. A higher number of samples can provide a more accurate collision check at the cost of increased computation time. The optimal number may depend on the complexity of the mesh and the required precision of collision detection."""

    debug: bool = False
    """ Whether to visualize the collision checking results for debugging purposes. If set to True, the code will generate visualizations of the query points colored by their collision status (e.g., red for colliding points and green for non-colliding points) along with the original mesh. This can help in understanding and verifying the collision checking process, especially during development and testing."""


class ConvexCollisionChecker:
    """ConvexCollisionChecker performs efficient collision checking between a batch of query point clouds and a convex decomposition of a mesh. The convex decomposition is represented by plane equations of the convex hulls, which are precomputed and cached for efficiency. The collision checking is done by computing the signed distance from each query point to the convex hulls using the plane equations, and determining if any points are colliding based on a specified collision threshold. This class can be used"""

    def __init__(
        self,
        base_mesh_verts: torch.Tensor,
        base_mesh_faces: torch.Tensor,
        max_decomposition_hulls: int = 32,
    ):
        """Initialize the ConvexCollisionChecker by performing convex decomposition on the input mesh and extracting plane equations for the convex hulls. The plane equations are cached to disk to avoid redundant computation in future runs.

        Args:
            base_mesh_verts: [N, 3] vertex positions of the input mesh.
            base_mesh_faces: [M, 3] triangle indices of the input mesh.
            max_decomposition_hulls: maximum number of convex hulls to decompose into. A higher number allows for a more accurate approximation of the original mesh but increases computation time and memory usage. The optimal number may depend on the complexity of the mesh and the required precision of collision checking.
        """
        from embodichain.lab.sim import CONVEX_DECOMP_DIR

        if not os.path.isdir(CONVEX_DECOMP_DIR):
            os.makedirs(CONVEX_DECOMP_DIR, exist_ok=True)
        self.device = base_mesh_verts.device
        base_mesh_verts_np = base_mesh_verts.cpu().numpy()
        base_mesh_faces_np = base_mesh_faces.cpu().numpy()
        mesh_hash = hashlib.md5(
            (base_mesh_verts_np.tobytes() + base_mesh_faces_np.tobytes())
        ).hexdigest()

        # for visualization
        self.mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(base_mesh_verts_np),
            triangles=o3d.utility.Vector3iVector(base_mesh_faces_np),
        )
        self.mesh.compute_vertex_normals()

        self.cache_path = os.path.join(
            CONVEX_DECOMP_DIR, f"{mesh_hash}_{max_decomposition_hulls}.pkl"
        )

        if not os.path.isfile(self.cache_path):
            # [n_convex, n_max_faces, 4]: plane equations, normals(3) and offsets(1), padded with zeros if a hull has less than n_max_faces
            # [n_convex, ]: number of faces for each convex hull

            # generate convex hulls and extract plane equations, then cache to disk
            plane_equations_np = ConvexCollisionChecker._compute_plane_equations(
                base_mesh_verts_np, base_mesh_faces_np, max_decomposition_hulls
            )
            # pack as a single tensor
            n_convex = len(plane_equations_np)
            n_max_equation = max(len(normals) for normals, _ in plane_equations_np)
            plane_equations = torch.zeros(
                size=(n_convex, n_max_equation, 4),
                dtype=torch.float32,
                device=self.device,
            )
            plane_equations_counts = torch.zeros(
                n_convex, dtype=torch.int32, device=self.device
            )
            for i in range(n_convex):
                n_equation = plane_equations_np[i][0].shape[0]
                # plane normals
                plane_equations[i, :n_equation, :3] = torch.tensor(
                    plane_equations_np[i][0], device=self.device
                )
                # plane offsets
                plane_equations[i, :n_equation, 3] = torch.tensor(
                    plane_equations_np[i][1], device=self.device
                )
                plane_equations_counts[i] = n_equation
            self.plane_equations = {
                "plane_equations": plane_equations,
                "plane_equation_counts": plane_equations_counts,
            }
            pickle.dump(self.plane_equations, open(self.cache_path, "wb"))
        else:
            self.plane_equations = pickle.load(open(self.cache_path, "rb"))
            self.plane_equations["plane_equations"] = self.plane_equations[
                "plane_equations"
            ].to(self.device)
            self.plane_equations["plane_equation_counts"] = self.plane_equations[
                "plane_equation_counts"
            ].to(self.device)

    @staticmethod
    def batch_point_convex_query(
        plane_equations: torch.Tensor,
        plane_equation_counts: torch.Tensor,
        batch_points: torch.Tensor,
        device: torch.device,
        collision_threshold: float = -0.003,
    ):
        # always use cuda for batch grasp pose query
        is_cpu = device == torch.device("cpu")
        if is_cpu:
            plane_equations_wp = wp.from_torch(plane_equations.to("cuda"))
            plane_equation_counts_wp = wp.from_torch(plane_equation_counts.to("cuda"))
            batch_points_wp = wp.from_torch(batch_points.to("cuda"))
        else:
            plane_equations_wp = wp.from_torch(plane_equations)
            plane_equation_counts_wp = wp.from_torch(plane_equation_counts)
            batch_points_wp = wp.from_torch(batch_points)

        if is_cpu:
            wp_device = standardize_device_string(torch.device("cuda"))
        else:
            wp_device = standardize_device_string(device)
        n_pose = batch_points.shape[0]
        n_point = batch_points.shape[1]
        n_convex = plane_equations.shape[0]
        point_convex_signed_distance_wp = wp.full(
            shape=(n_pose, n_point, n_convex),
            value=-float("inf"),
            dtype=float,
            device=wp_device,
        )  # [n_pose, n_point, n_convex]
        wp.launch(
            kernel=convex_signed_distance_kernel,
            dim=(n_pose, n_point, n_convex),
            inputs=(batch_points_wp, plane_equations_wp, plane_equation_counts_wp),
            outputs=(point_convex_signed_distance_wp,),
            device=wp_device,
        )
        point_convex_signed_distance = wp.to_torch(point_convex_signed_distance_wp)
        point_signed_distance = point_convex_signed_distance.min(
            dim=-1
        ).values  # [n_pose, n_point]
        is_point_collide = point_signed_distance <= collision_threshold
        if is_cpu:
            return point_signed_distance.to("cpu"), is_point_collide.to("cpu")
        else:
            return point_signed_distance, is_point_collide

    def query_batch_points(
        self,
        batch_points: torch.Tensor,
        collision_threshold: float = 0.0,
        is_visual: bool = False,
    ) -> torch.Tensor:
        """Query collision status for a batch of point clouds. The collision status is determined by checking if the signed distance from any point in the cloud to the convex hulls is less than or equal to the specified collision threshold.
        Args:
            batch_points: [B, n_point, 3] batch of point clouds to query for collision status.
            collision_threshold: Collision threshold in meters. A point is considered colliding if its signed distance to the hull interior is <= this threshold. This allows for a margin of error in collision checking, where a small positive threshold can be used to consider points near the surface as colliding, and a small negative threshold can be used to allow for slight penetration without considering it a collision.
            is_visual: Whether to visualize the collision checking results for debugging purposes. If set to True, the code will generate visualizations of the query points colored by their collision status (e.g., red for colliding points and green for non-colliding points) along with the original mesh. This can help in understanding and verifying the collision checking process, especially during development and testing.
        Returns:
            is_pose_collide: [B, ] boolean tensor indicating whether each point cloud in the
        """
        n_batch = batch_points.shape[0]
        point_signed_distance, is_point_collide = (
            ConvexCollisionChecker.batch_point_convex_query(
                self.plane_equations["plane_equations"],
                self.plane_equations["plane_equation_counts"],
                batch_points,
                device=self.device,
                collision_threshold=collision_threshold,
            )
        )
        is_pose_collide = is_point_collide.any(dim=-1)  # [B]
        pose_surface_distance = point_signed_distance.min(dim=-1).values  # [B]
        if is_visual:
            # visualize result
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            for i in range(n_batch):
                query_points_o3d = o3d.geometry.PointCloud()
                query_points_np = batch_points[i].cpu().numpy()
                query_points_o3d.points = o3d.utility.Vector3dVector(query_points_np)
                query_points_color = np.zeros_like(query_points_np)
                query_points_color[is_point_collide[i].cpu().numpy()] = [
                    1.0,
                    0,
                    0,
                ]  # red for colliding points
                query_points_color[~is_point_collide[i].cpu().numpy()] = [
                    0,
                    1.0,
                    0,
                ]  # green for non-colliding points
                query_points_o3d.colors = o3d.utility.Vector3dVector(query_points_color)
                o3d.visualization.draw_geometries(
                    [self.mesh, query_points_o3d, frame], mesh_show_back_face=True
                )
        return is_pose_collide, pose_surface_distance

    def query(
        self,
        query_mesh_verts: torch.Tensor,
        query_mesh_faces: torch.Tensor,
        poses: torch.Tensor,
        cfg: ConvexCollisionCheckerCfg = ConvexCollisionCheckerCfg(),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_mesh = trimesh.Trimesh(
            vertices=query_mesh_verts.to("cpu").numpy(),
            faces=query_mesh_faces.to("cpu").numpy(),
        )
        n_query = cfg.n_query_mesh_samples
        n_batch = poses.shape[0]
        query_points_np = query_mesh.sample(n_query).astype(np.float32)
        query_points = torch.tensor(
            query_points_np, device=poses.device
        )  # [n_query, 3]
        penetration_result = torch.zeros(size=(n_batch, n_query), device=poses.device)
        penetration_result.fill_(-float("inf"))
        collision_result = torch.zeros(
            size=(n_batch, n_query), dtype=torch.bool, device=poses.device
        )
        collision_result.fill_(False)
        for normals, offsets in self.plane_equations:
            normals_torch = torch.tensor(normals, device=poses.device)
            offsets_torch = torch.tensor(offsets, device=poses.device)
            penetration, collides = check_collision_single_hull(
                normals_torch,
                offsets_torch,
                transform_points_mat(query_points, poses),
                cfg.collision_threshold,
            )
            penetration_result = torch.max(penetration_result, penetration)
            collision_result = torch.logical_or(collision_result, collides)
        is_colliding = collision_result.any(dim=-1)  # [B]
        max_penetration = penetration_result.max(dim=-1)[0]  # [B]

        if cfg.debug:
            # visualize result
            for i in range(n_batch):
                query_points_o3d = o3d.geometry.PointCloud()
                query_points_o3d.points = o3d.utility.Vector3dVector(query_points_np)
                query_points_o3d.transform(poses[i].to("cpu").numpy())
                query_points_color = np.zeros_like(query_points_np)
                query_points_color[collision_result[i].cpu().numpy()] = [
                    1.0,
                    0,
                    0,
                ]  # red for colliding points
                query_points_color[~collision_result[i].cpu().numpy()] = [
                    0,
                    1.0,
                    0,
                ]  # green for non-colliding points
                query_points_o3d.colors = o3d.utility.Vector3dVector(query_points_color)
                o3d.visualization.draw_geometries(
                    [self.mesh, query_points_o3d], mesh_show_back_face=True
                )
        return is_colliding, max_penetration

    @staticmethod
    def _compute_plane_equations(
        vertices: np.ndarray, faces: np.ndarray, max_decomposition_hulls: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convex decomposition and extract plane equations given mesh vertices and triangles.
        Each convex hull is represented by its outward-facing face normals and offsets.
        No padding is applied; each hull can have a different number of faces.

        Args:
            vertices: [N, 3] vertex positions of the input mesh.
            faces: [M, 3] triangle indices of the input mesh.
            max_decomposition_hulls: maximum number of convex hulls to decompose into.

        Returns:
            List of (normals_i [Ki, 3], offsets_i [Ki]) tuples, one per convex hull.
            Ki is the number of faces of the i-th hull and can differ across hulls.
        """
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
        mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
        is_success, out_mesh_list = convex_decomposition_coacd(
            mesh, max_convex_hull_num=max_decomposition_hulls
        )
        convex_vert_face_list = []
        for out_mesh in out_mesh_list:
            verts = out_mesh.vertex.positions.numpy()
            faces = out_mesh.triangle.indices.numpy()
            convex_vert_face_list.append((verts, faces))
        return extract_plane_equations(convex_vert_face_list)


def extract_plane_equations(
    convex_meshes: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract plane equations from a list of convex hull meshes.
    Each convex hull is represented by its outward-facing face normals and offsets.
    No padding is applied; each hull can have a different number of faces.

    Args:
        convex_meshes: List of convex hull data.
            - tuple of (vertices [N,3], faces [M,3])

    Returns:
        List of (normals_i [Ki, 3], offsets_i [Ki]) tuples, one per convex hull.
        Ki is the number of faces of the i-th hull and can differ across hulls.
    """
    convex_plane_data = []

    for i, convex_mesh_data in enumerate(convex_meshes):
        vertices, faces = convex_mesh_data
        hull = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
        )
        # Outward-facing face normals [Ki, 3]
        face_normals = hull.face_normals
        # One vertex per face to compute offset [Ki, 3]
        face_origins = hull.triangles[:, 0, :]
        # Plane equation: n · x + d = 0  =>  d = -(n · p)
        offsets_i = -np.sum(face_normals * face_origins, axis=1)

        convex_plane_data.append(
            (face_normals.astype(np.float32), offsets_i.astype(np.float32))
        )
    return convex_plane_data


def sample_surface_points(mesh_path: str, num_points: int = 4096) -> np.ndarray:
    """
    Sample surface points from a mesh file.

    Args:
        mesh_path: Path to the mesh file.
        num_points: Number of surface points to sample.

    Returns:
        points: [P, 3] numpy array of sampled surface points.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    points = mesh.sample(num_points)
    return points.astype(np.float32)


def check_collision_single_hull(
    normals: torch.Tensor,  # [K, 3]
    offsets: torch.Tensor,  # [K]
    transformed_points: torch.Tensor,  # [B, P, 3]
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check collision between a batch of transformed point clouds and a single convex hull.

    A point p is inside the convex hull iff:
        max_k (n_k · p + d_k) <= 0

    Penetration depth for a point is defined as:
        penetration = -(max_k (n_k · p + d_k))
    Positive penetration means the point is inside the hull.

    Args:
        normals: [K, 3] outward face normals of the convex hull.
        offsets: [K] plane offsets of the convex hull.
        transformed_points: [B, P, 3] point cloud already transformed by batch poses.
        threshold: collision threshold. A point is considered colliding if
                   its signed distance to the hull interior is <= threshold.

    Returns:
        penetration: [B, P] penetration depth for each point.
                     Positive values indicate the point is inside the hull.
        collides: [B, P] boolean mask, True if the point collides with this hull.
    """
    # signed_dist: [B, P, K] = einsum([B,P,3], [K,3]) + [K]
    signed_dist = torch.einsum("bpj, kj -> bpk", transformed_points, normals) + offsets

    # For each point, the maximum signed distance across all planes
    # If max <= 0, the point satisfies all half-plane constraints => inside the hull
    max_over_planes, _ = signed_dist.max(dim=-1)  # [B, P]

    # Penetration depth: negate so that positive = inside
    penetration = -max_over_planes  # [B, P]

    # A point collides if its penetration exceeds the threshold
    collides = penetration > threshold  # [B, P]

    return penetration, collides
