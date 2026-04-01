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


@configclass
class AntipodalSamplerCfg:
    """Configuration for AntipodalSampler."""

    n_sample: int = 20000
    """surface point sample number"""
    max_angle: float = np.pi / 12
    """maximum angle (in radians) to randomly disturb the ray direction for antipodal point sampling, used to increase the diversity of sampled antipodal points. Note that setting max_angle to 0 will disable the random disturbance and sample antipodal points strictly along the surface normals, which may result in less diverse antipodal points and may not be ideal for all objects or grasping scenarios."""
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
        self.mesh.compute_vertex_normals()
        # sample points and normals
        sample_pcd = self.mesh.sample_points_uniformly(
            number_of_points=self.cfg.n_sample
        )
        sample_points = torch.tensor(
            sample_pcd.point.positions.numpy(),
            device=vertices.device,
            dtype=vertices.dtype,
        )
        sample_normals = torch.tensor(
            sample_pcd.point.normals.numpy(),
            device=vertices.device,
            dtype=vertices.dtype,
        )
        # generate rays
        ray_direc = -sample_normals
        ray_origin = (
            sample_points + 1e-3 * ray_direc
        )  # Offset ray origin slightly along the normal to avoid self-intersection
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
        hit_mask = torch.logical_and(
            t_hit > self.cfg.min_length, t_hit < self.cfg.max_length
        )
        hit_points = ray_origin[hit_mask] + t_hit[hit_mask, None] * ray_direc[hit_mask]
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


if __name__ == "__main__":
    mesh_path = "/media/chenjian/_abc/project/grasp_annotator/dustpan_saa.ply"
    mesh = o3d.t.io.read_triangle_mesh(mesh_path)
    vertices = torch.from_numpy(mesh.vertex.positions.cpu().numpy())
    faces = torch.from_numpy(mesh.triangle.indices.cpu().numpy())

    sampler = AntipodalSampler()
    hit_point_pairs = sampler.sample(vertices, faces)
    sampler.visualize(hit_point_pairs)
    print(f"Sampled {hit_point_pairs.shape[0]} antipodal points")
