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

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import open3d.core as o3c
from embodichain.utils import configclass
from embodichain.utils import logger

__all__ = ["AntipodalSamplerCfg", "AntipodalSampler"]


@configclass
class AntipodalSamplerCfg:
    """Configuration for AntipodalSampler."""

    n_sample: int = 20000
    """surface point sample number"""

    max_angle: float = np.pi / 12
    """maximum angle (in radians) to randomly disturb the ray direction for antipodal point sampling, 
    used to increase the diversity of sampled antipodal points. Note that setting max_angle to 0 will 
    disable the random disturbance and sample antipodal points strictly along the surface normals, 
    which may result in less diverse antipodal points and may not be ideal for all objects or grasping 
    scenarios.
    """

    max_length: float = 0.1
    """maximum gripper open width, used to filter out antipodal points that are too far apart to be grasped"""

    min_length: float = 0.001
    """minimum gripper open width, used to filter out antipodal points that are too close to be grasped"""


class AntipodalSampler:
    """AntipodalSampler samples antipodal point pairs on a given mesh. It uses Open3D's raycasting functionality to find points on the mesh that are visible along the negative normal direction from uniformly sampled points on the mesh surface. The sampler can also apply a random disturbance to the ray direction to increase the diversity of sampled antipodal points. The resulting antipodal point pairs can be used for grasp generation and annotation tasks."""

    def __init__(
        self,
        cfg: AntipodalSamplerCfg = AntipodalSamplerCfg(),
    ):
        self.mesh: o3d.t.geometry.TriangleMesh | None = None
        self.cfg = cfg

    def sample(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Get sample Antipodal point pair

        Args:
            vertices: [V, 3] vertex positions of the mesh
            faces: [F, 3] triangle indices of the mesh

        Returns:
            hit_point_pairs: [N, 2, 3] tensor of N antipodal point pairs. Each pair consists of a hit point and its corresponding surface point.
        """
        # update mesh
        self.mesh = o3d.t.geometry.TriangleMesh()
        self.mesh.vertex.positions = o3c.Tensor(
            vertices.to("cpu").numpy(), dtype=o3c.float32
        )
        self.mesh.triangle.indices = o3c.Tensor(
            faces.to("cpu").numpy(), dtype=o3c.int32
        )
        # Sample surface points and normals by raycasting Fibonacci-distributed
        # rays from outside the mesh toward its centroid. Each contact point
        # replaces the previous uniform surface sample and keeps its face normal.
        sample_points, sample_normals = self._sample_surface_by_fibonacci_raycast(
            vertices, self.cfg.n_sample
        )

        vertices_x_range = vertices[:, 0].max() - vertices[:, 0].min()
        vertices_y_range = vertices[:, 1].max() - vertices[:, 1].min()
        vertices_z_range = vertices[:, 2].max() - vertices[:, 2].min()
        max_range = max(vertices_x_range, vertices_y_range, vertices_z_range)
        # generate rays
        ray_direc = sample_normals
        ray_origin = (
            sample_points - 2.0 * max_range * ray_direc
        )  # ray origin in the other side of the mesh
        disturb_direc = AntipodalSampler._random_rotate_unit_vectors(
            ray_direc, max_angle=self.cfg.max_angle
        )
        ray_origin = torch.vstack([ray_origin, ray_origin])
        ray_direc = torch.vstack([ray_direc, disturb_direc])
        # casting
        return self._get_raycast_result(
            ray_origin,
            ray_direc,
            surface_origin=torch.vstack([sample_points, sample_points]),
        )

    def _sample_surface_by_fibonacci_raycast(
        self,
        vertices: torch.Tensor,
        n_sample: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample surface points with normals via Fibonacci-sphere raycasting.

        Instead of sampling points directly on the mesh surface, rays are
        distributed uniformly over the unit sphere using the Fibonacci spiral
        and cast from a sphere enclosing the mesh toward its centroid. The
        first contact point of each ray with the mesh is the sample, and the
        face normal at the contact (oriented against the ray) is its normal.

        Args:
            vertices: ``[V, 3]`` vertex positions of the mesh, used to build the
                enclosing raycast sphere and to recover the sample device/dtype.
            n_sample: Number of Fibonacci-distributed rays to cast.

        Returns:
            A tuple ``(sample_points, sample_normals)`` of ``[n_sample, 3]``
            tensors. For a closed mesh the enclosing sphere guarantees every ray
            intersects the surface. Rays that miss (open meshes) yield a
            zero-length hit at the ray origin with a zero normal and are filtered
            out downstream during antipodal-pair construction.
        """
        if n_sample <= 0:
            empty = torch.empty((0, 3), device=vertices.device, dtype=vertices.dtype)
            return empty, empty.clone()

        # Fibonacci spiral directions on the unit sphere.
        index = (
            torch.arange(n_sample, device=vertices.device, dtype=vertices.dtype) + 0.5
        )
        golden_angle = torch.tensor(
            np.pi * (1.0 + np.sqrt(5.0)), device=vertices.device, dtype=vertices.dtype
        )
        theta = golden_angle * index
        z = 1.0 - 2.0 * index / n_sample
        z = torch.clamp(z, -1.0, 1.0)
        rho = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        directions = torch.stack(
            [rho * torch.cos(theta), rho * torch.sin(theta), z], dim=-1
        )

        # Raycast from a sphere enclosing the mesh toward its centroid.
        vertices_np = vertices.detach().to("cpu").numpy()
        centroid = vertices_np.mean(axis=0)
        extent = np.linalg.norm(vertices_np - centroid, axis=1)
        max_radius = float(extent.max()) if vertices_np.shape[0] > 0 else 0.0
        ray_distance = 2.0 * max_radius + 1.0  # safely outside the mesh

        directions_np = directions.detach().to("cpu").numpy().astype(np.float32)
        ray_origins_np = (centroid[None, :] - ray_distance * directions_np).astype(
            np.float32
        )
        rays_np = np.concatenate([ray_origins_np, directions_np], axis=-1)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(self.mesh)
        ans = scene.cast_rays(o3c.Tensor(rays_np, dtype=o3c.float32))

        t_hit = torch.from_numpy(np.asarray(ans["t_hit"].numpy())).to(
            device=vertices.device, dtype=vertices.dtype
        )
        normals = torch.from_numpy(np.asarray(ans["primitive_normals"].numpy())).to(
            device=vertices.device, dtype=vertices.dtype
        )

        # No hit -> cast_rays reports infinite t_hit; clamp those to a zero-length hit.
        t_hit = torch.where(torch.isfinite(t_hit), t_hit, torch.zeros_like(t_hit))

        sample_points = (
            torch.as_tensor(
                ray_origins_np, device=vertices.device, dtype=vertices.dtype
            )
            + t_hit[:, None] * directions
        )
        sample_normals = normals

        # Orient normals to oppose the ray direction (toward the ray origin).
        dot = (sample_normals * directions).sum(dim=-1, keepdim=True)
        sample_normals = torch.where(dot > 0, -sample_normals, sample_normals)
        return sample_points, sample_normals

    def _get_raycast_result(
        self,
        ray_origin: torch.Tensor,
        ray_direc: torch.Tensor,
        surface_origin: torch.Tensor,
    ):
        if ray_origin.ndim != 2 or ray_origin.shape[-1] != 3:
            raise ValueError("ray_origin must have shape [N, 3]")
        if ray_direc.ndim != 2 or ray_direc.shape[-1] != 3:
            raise ValueError("ray_direc must have shape [N, 3]")
        if ray_origin.shape[0] != ray_direc.shape[0]:
            raise ValueError(
                "ray_origin and ray_direc must have the same number of rays"
            )
        if ray_origin.shape[0] != surface_origin.shape[0]:
            raise ValueError(
                "ray_origin and surface_origin must have the same number of rays"
            )

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(self.mesh)

        rays = torch.cat([ray_origin, ray_direc], dim=-1)
        rays_o3d = o3c.Tensor(rays.detach().to("cpu").numpy(), dtype=o3c.float32)

        ans = scene.cast_rays(rays_o3d)
        t_hit = torch.from_numpy(ans["t_hit"].numpy()).to(
            device=ray_origin.device, dtype=ray_origin.dtype
        )
        hit_points = ray_origin + t_hit[:, None] * ray_direc
        antipodal_len = torch.norm(hit_points - surface_origin, dim=-1)
        hit_mask = torch.logical_and(
            antipodal_len > self.cfg.min_length, antipodal_len < self.cfg.max_length
        )

        hit_points = hit_points[hit_mask]
        hit_origins = surface_origin[hit_mask]
        hit_point_pairs = torch.cat(
            [hit_points[:, None, :], hit_origins[:, None, :]], dim=1
        )
        hit_point_pairs = hit_point_pairs.to(dtype=torch.float32)
        return hit_point_pairs

    @staticmethod
    def _random_rotate_unit_vectors(
        vectors: torch.Tensor,
        max_angle: float,
        degrees: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Apply random small rotations to a batch of unit vectors [N, 3].

        Args:
            vectors: [N, 3], unit vectors
            max_angle: Maximum rotation angle
            degrees: If True, `max_angle` is given in degrees
            eps: Numerical stability constant

        Returns:
            rotated: [N, 3], rotated unit vectors
        """
        assert vectors.ndim == 2 and vectors.shape[-1] == 3, "vectors must be [N, 3]"

        v = F.normalize(vectors, dim=-1)

        if degrees:
            max_angle = torch.deg2rad(
                torch.tensor(max_angle, dtype=v.dtype, device=v.device)
            ).item()

        n = v.shape[0]

        # 1) Generate a random direction for each vector
        #   then project it onto the plane perpendicular to v to get the rotation axis k
        rand_dir = torch.randn_like(v) + eps
        proj = (rand_dir * v).sum(dim=-1, keepdim=True) * v
        k = rand_dir - proj
        k = F.normalize(k, dim=-1)

        # 2) Sample rotation angles in the range [eps, max_angle]
        theta = (
            torch.rand(n, 1, device=v.device, dtype=v.dtype) * (max_angle - eps) + eps
        )

        # 3) Rodrigues' rotation formula
        # R(v) = v*cosθ + (k×v)*sinθ + k*(k·v)*(1-cosθ)
        # Since k ⟂ v, the last term is theoretically 0, but keeping the general formula is more robust
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        kv = (k * v).sum(dim=-1, keepdim=True)
        rotated = v * cos_t + torch.cross(k, v, dim=-1) * sin_t + k * kv * (1.0 - cos_t)

        return F.normalize(rotated, dim=-1)

    def visualize(self, hit_point_pairs: torch.Tensor):
        if self.mesh is None:
            logger.log_warning("Mesh is not initialized. Cannot visualize.")
            return

        if hit_point_pairs.shape[0] == 0:
            raise ValueError("No point pairs to visualize")
        origin_points = hit_point_pairs[:, 0, :]
        hit_points = hit_point_pairs[:, 1, :]

        origin_points_np = origin_points.to("cpu").numpy()
        hit_points_np = hit_points.detach().to("cpu").numpy()

        n_pairs = hit_point_pairs.shape[0]
        line_indices = np.stack(
            [np.arange(n_pairs), np.arange(n_pairs) + n_pairs], axis=1
        )

        mesh_legacy = self.mesh.to_legacy()
        mesh_legacy.compute_vertex_normals()
        mesh_legacy.paint_uniform_color([0.8, 0.8, 0.8])

        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points = o3d.utility.Vector3dVector(origin_points_np)
        origin_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.1, 0.4, 1.0]]), (n_pairs, 1))
        )

        hit_pcd = o3d.geometry.PointCloud()
        hit_pcd.points = o3d.utility.Vector3dVector(hit_points_np)
        hit_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.2, 0.2]]), (n_pairs, 1))
        )

        line_set = o3d.geometry.LineSet()
        mid_points = (origin_points_np + hit_points_np) / 2
        point_diff = hit_points_np - origin_points_np
        draw_origin = mid_points - 0.6 * point_diff
        draw_end = mid_points + 0.6 * point_diff
        draw_pointpair = np.concatenate([draw_origin, draw_end], axis=0)
        line_set.points = o3d.utility.Vector3dVector(draw_pointpair)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.2, 0.9, 0.2]]), (n_pairs, 1))
        )

        o3d.visualization.draw_geometries(
            [mesh_legacy, origin_pcd, hit_pcd, line_set],
            window_name="Antipodal Point Pairs",
            mesh_show_back_face=True,
        )

    def visualize_antipodal_pairs(self, hit_point_pairs: torch.Tensor) -> None:
        """Visualize sampled antipodal point pairs on the mesh with Open3D.

        Temporary debug helper that draws the mesh, a ground plane, and the
        antipodal point pairs (surface origin connected to its antipodal hit)
        in the style of :meth:`GraspGenerator.visualize_grasp_poses`.

        .. attention::
            ``self.mesh`` must have been populated by a prior call to
            :meth:`sample`.

        Args:
            hit_point_pairs: ``[N, 2, 3]`` tensor of antipodal point pairs as
                returned by :meth:`sample`. ``[:, 0]`` is the antipodal contact
                (hit) point and ``[:, 1]`` is the surface origin (sample) point.
        """
        if self.mesh is None:
            logger.log_warning("Mesh is not initialized. Cannot visualize.")
            return
        if hit_point_pairs.shape[0] == 0:
            raise ValueError("No point pairs to visualize")

        hit_points_np = hit_point_pairs[:, 0, :].detach().to("cpu").numpy()
        origin_points_np = hit_point_pairs[:, 1, :].detach().to("cpu").numpy()
        n_pairs = hit_point_pairs.shape[0]

        mesh_legacy = self.mesh.to_legacy()
        mesh_legacy.compute_vertex_normals()
        mesh_legacy.paint_uniform_color([0.3, 0.6, 0.3])

        verts_np = np.asarray(mesh_legacy.vertices)
        mesh_scale = float((verts_np.max(axis=0) - verts_np.min(axis=0)).max())
        ground_center = verts_np.mean(axis=0)
        z_min = float(verts_np[:, 2].min())

        ground_plane = o3d.geometry.TriangleMesh.create_cylinder(
            radius=mesh_scale, height=0.01 * mesh_scale
        )
        ground_plane.compute_vertex_normals()
        ground_plane.paint_uniform_color([0.7, 0.7, 0.7])
        ground_plane.translate(
            (ground_center[0], ground_center[1], z_min - 0.005 * mesh_scale)
        )

        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points = o3d.utility.Vector3dVector(origin_points_np)
        origin_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.1, 0.4, 1.0]]), (n_pairs, 1))
        )

        hit_pcd = o3d.geometry.PointCloud()
        hit_pcd.points = o3d.utility.Vector3dVector(hit_points_np)
        hit_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.2, 0.2]]), (n_pairs, 1))
        )

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            np.concatenate([origin_points_np, hit_points_np], axis=0)
        )
        line_set.lines = o3d.utility.Vector2iVector(
            np.stack([np.arange(n_pairs), np.arange(n_pairs) + n_pairs], axis=1)
        )
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.2, 0.9, 0.2]]), (n_pairs, 1))
        )

        o3d.visualization.draw_geometries(
            [mesh_legacy, ground_plane, origin_pcd, hit_pcd, line_set],
            window_name="Antipodal Point Pairs",
            mesh_show_back_face=True,
        )
