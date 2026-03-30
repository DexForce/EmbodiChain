import trimesh
import numpy as np
import torch
import time
from typing import List, Tuple, Union
from dexsim.kit.meshproc import convex_decomposition_coacd
import hashlib
from dataclasses import dataclass
import os
import pickle
import open3d as o3d
from embodichain.utils import logger

CONVEX_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "embodichain_cache", "convex_decomposition"
)


@dataclass
class BatchConvexCollisionCheckerCfg:
    collsion_threshold: float = 0.0
    n_query_mesh_samples: int = 4096
    debug: bool = False


class BatchConvexCollisionChecker:
    def __init__(
        self,
        base_mesh_verts: torch.Tensor,
        base_mesh_faces: torch.Tensor,
        max_decomposition_hulls: int = 32,
    ):
        if not os.path.isdir(CONVEX_CACHE_DIR):
            os.makedirs(CONVEX_CACHE_DIR, exist_ok=True)
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
            CONVEX_CACHE_DIR, f"{mesh_hash}_{max_decomposition_hulls}.pkl"
        )

        if not os.path.isfile(self.cache_path):
            # generate convex hulls and extract plane equations, then cache to disk
            self.plane_equations = BatchConvexCollisionChecker._compute_plane_equations(
                base_mesh_verts_np, base_mesh_faces_np, max_decomposition_hulls
            )
            pickle.dump(self.plane_equations, open(self.cache_path, "wb"))
        else:
            # load precomputed plane equations from cache
            self.plane_equations = pickle.load(open(self.cache_path, "rb"))

    def query_batch_points(
        self,
        batch_points: torch.Tensor,
        collision_threshold: float = 0.0,
        is_visual: bool = False,
    ):
        n_batch = batch_points.shape[0]
        n_points = batch_points.shape[1]
        penetration_result = torch.zeros(size=(n_batch, n_points), device=self.device)
        penetration_result.fill_(-float("inf"))
        collision_result = torch.zeros(
            size=(n_batch, n_points), dtype=torch.bool, device=self.device
        )
        collision_result.fill_(False)
        for normals, offsets in self.plane_equations:
            normals_torch = torch.tensor(normals, device=self.device)
            offsets_torch = torch.tensor(offsets, device=self.device)
            penetration, collides = check_collision_single_hull(
                normals_torch,
                offsets_torch,
                batch_points,
                collision_threshold,
            )
            penetration_result = torch.max(penetration_result, penetration)
            collision_result = torch.logical_or(collision_result, collides)
        is_colliding = collision_result.any(dim=-1)  # [B]
        max_penetration = penetration_result.max(dim=-1)[0]  # [B]

        if is_visual:
            # visualize result
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            for i in range(n_batch):
                query_points_o3d = o3d.geometry.PointCloud()
                query_points_np = batch_points[i].cpu().numpy()
                query_points_o3d.points = o3d.utility.Vector3dVector(query_points_np)
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
                    [self.mesh, query_points_o3d, frame], mesh_show_back_face=True
                )
        return is_colliding, max_penetration

    def query(
        self,
        query_mesh_verts: torch.Tensor,
        query_mesh_faces: torch.Tensor,
        poses: torch.Tensor,
        cfg: BatchConvexCollisionCheckerCfg = BatchConvexCollisionCheckerCfg(),
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
                transform_points_batch(query_points, poses),
                cfg.collsion_threshold,
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


def merge_collision_results(
    hull_results: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge collision detection results from multiple convex hulls.

    A pose is considered colliding if ANY point penetrates ANY convex hull.
    The reported penetration depth is the maximum across all points and all hulls.

    Args:
        hull_results: List of (penetration [B, P], collides [B, P]) tuples,
                      one per convex hull, as returned by check_collision_single_hull.
        device: torch device.

    Returns:
        overall_collisions: [B] boolean, True if the pose collides with any hull.
        overall_max_penetrations: [B] float, maximum penetration depth per pose.
    """
    if not hull_results:
        raise ValueError("hull_results is empty, nothing to merge.")

    B = hull_results[0][0].shape[0]

    overall_collisions = torch.zeros(B, dtype=torch.bool, device=device)
    overall_max_penetrations = torch.full(
        (B,), -float("inf"), dtype=torch.float32, device=device
    )

    for penetration, collides in hull_results:
        # Update collision flag: OR across hulls
        # A pose collides if any point collides with this hull
        overall_collisions |= collides.any(dim=-1)  # [B]

        # Update max penetration: take the per-pose maximum across all points for this hull,
        # then compare with the running maximum
        hull_max_pen = penetration.max(dim=-1)[0]  # [B]
        overall_max_penetrations = torch.max(overall_max_penetrations, hull_max_pen)

    return overall_collisions, overall_max_penetrations


def transform_points_batch(
    points: torch.Tensor, poses: torch.Tensor  # [P, 3]  # [B, 4, 4]
) -> torch.Tensor:
    """
    Apply a batch of rigid transforms to a point cloud.

    Args:
        points: [P, 3] source point cloud.
        poses: [B, 4, 4] batch of homogeneous transformation matrices.

    Returns:
        transformed: [B, P, 3] transformed point cloud for each pose.
    """
    R = poses[:, :3, :3]  # [B, 3, 3]
    t = poses[:, :3, 3]  # [B, 3]
    transformed = torch.einsum("bij, pj -> bpi", R, points) + t.unsqueeze(1)
    return transformed


if __name__ == "__main__":
    from embodichain.data import get_data_path

    mug_path = get_data_path("CoffeeCup/cup.ply")
    mug_path = get_data_path("ScannedBottle/moliwulong_processed.ply")
    mug_mesh = trimesh.load(mug_path, force="mesh", process=False)
    verts = torch.tensor(mug_mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mug_mesh.faces, dtype=torch.int32)
    collision_checker = BatchConvexCollisionChecker(
        verts, faces, max_decomposition_hulls=16
    )

    poses = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.05],
                [0, 0, 0, 1],
            ],
            [
                [1, 0, 0, 0.05],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
        ]
    )
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", [12, 3, 32], degrees=True).as_matrix()
    poses[0, :3, :3] = torch.tensor(rot, dtype=torch.float32)
    poses[1, :3, :3] = torch.tensor(rot, dtype=torch.float32)

    obj_path = get_data_path("ScannedBottle/yibao_processed.ply")
    obj_mesh = trimesh.load(obj_path, force="mesh", process=False)
    obj_verts = torch.tensor(obj_mesh.vertices, dtype=torch.float32)
    obj_faces = torch.tensor(obj_mesh.faces, dtype=torch.int32)
    test_pc = transform_points_batch(obj_verts, poses)

    collision_checker.query_batch_points(
        test_pc, collision_threshold=0.003, is_visual=True
    )
    collision_checker.query(
        obj_verts,
        obj_faces,
        poses,
        cfg=BatchConvexCollisionCheckerCfg(
            debug=True, n_query_mesh_samples=32768, collsion_threshold=0.000
        ),
    )
