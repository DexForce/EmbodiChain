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
            query_points_o3d = o3d.geometry.PointCloud()
            query_points_o3d.points = o3d.utility.Vector3dVector(query_points_np)
            query_points_o3d.transform(poses[-1].to("cpu").numpy())
            query_points_color = np.zeros_like(query_points_np)
            query_points_color[collision_result[-1].cpu().numpy()] = [
                1.0,
                0,
                0,
            ]  # red for colliding points
            query_points_color[~collision_result[-1].cpu().numpy()] = [
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


def batch_collision_detection(
    convex_planes: List[Tuple[torch.Tensor, torch.Tensor]],
    points_B: torch.Tensor,  # [P, 3]
    poses: torch.Tensor,  # [B, 4, 4]
    threshold: float = 0.0,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full batch collision detection pipeline.

    Iterates over convex hulls sequentially and over pose chunks to control
    GPU memory usage. Within each (hull, chunk) pair, the computation is
    fully parallelized over B_chunk * P * K.

    Args:
        convex_planes: List of (normals [Ki, 3], offsets [Ki]) tensors on device,
                       one per convex hull. Ki can differ across hulls.
        points_B: [P, 3] sampled surface points of mesh B, on device.
        poses: [B, 4, 4] batch of relative poses, on device.
        threshold: collision threshold (positive = safety margin).
        chunk_size: number of poses to process per chunk.

    Returns:
        overall_collisions: [B] bool
        overall_max_penetrations: [B] float
    """
    device = points_B.device
    B = poses.shape[0]

    all_hull_results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # Sequential iteration over convex hulls to save memory
    for hull_idx, (normals, offsets) in enumerate(convex_planes):
        hull_pen_chunks = []
        hull_col_chunks = []

        # Chunk over batch dimension to control peak memory
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            poses_chunk = poses[start:end]

            # Transform points for this chunk of poses
            transformed_chunk = transform_points_batch(
                points_B, poses_chunk
            )  # [B_chunk, P, 3]

            # Check collision against this single hull
            penetration, collides = check_collision_single_hull(
                normals, offsets, transformed_chunk, threshold
            )

            hull_pen_chunks.append(penetration)
            hull_col_chunks.append(collides)

        # Concatenate chunks for this hull
        hull_penetration = torch.cat(hull_pen_chunks, dim=0)  # [B, P]
        hull_collides = torch.cat(hull_col_chunks, dim=0)  # [B, P]

        all_hull_results.append((hull_penetration, hull_collides))

    # Merge results across all hulls
    overall_collisions, overall_max_penetrations = merge_collision_results(
        all_hull_results, device
    )

    return overall_collisions, overall_max_penetrations


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create dummy mesh files for testing ---
    box1 = trimesh.primitives.Box(extents=[0.5, 0.5, 0.5])
    box2 = trimesh.primitives.Box(
        extents=[0.4, 0.4, 0.4],
        transform=trimesh.transformations.translation_matrix([1, 0, 0]),
    )
    box1.export("mesh_hull_1.obj")
    box2.export("mesh_hull_2.obj")

    sphere_mesh = trimesh.primitives.Sphere(radius=0.3)
    sphere_mesh.export("mesh_B.obj")
    print("Created dummy mesh files.\n")

    # ==================== Preprocessing ====================

    # Load externally decomposed convex hull meshes
    convex_mesh_files = ["mesh_hull_1.obj", "mesh_hull_2.obj"]
    convex_meshes = load_convex_meshes(convex_mesh_files)
    if not convex_meshes:
        print("No convex hulls loaded. Exiting.")
        return

    # Extract plane equations (list of variable-length arrays)
    convex_plane_data_np = extract_plane_equations(convex_meshes)

    # Convert to torch tensors on device
    convex_planes_torch: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for normals_np, offsets_np in convex_plane_data_np:
        convex_planes_torch.append(
            (
                torch.tensor(normals_np, device=device),  # [Ki, 3]
                torch.tensor(offsets_np, device=device),  # [Ki]
            )
        )

    # Sample surface points from mesh B
    points_np = sample_surface_points("mesh_B.obj", num_points=2048)
    points_B = torch.tensor(points_np, device=device)  # [P, 3]

    # ==================== Generate test poses ====================
    B = 10000
    chunk_size = 1024

    # Random rotation matrices via SVD
    random_mat = torch.randn(B, 3, 3, device=device)
    U, _, Vt = torch.linalg.svd(random_mat)
    R = U @ Vt
    # Fix reflections to ensure proper rotations (det = +1)
    det = torch.det(R)
    R[det < 0] *= -1

    poses = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = torch.randn(B, 3, device=device) * 0.5

    # ==================== Run collision detection ====================
    print(
        f"\nRunning collision detection: {B} poses, {points_B.shape[0]} points, "
        f"{len(convex_planes_torch)} hulls..."
    )

    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        collisions, penetration_depths = batch_collision_detection(
            convex_planes_torch, points_B, poses, threshold=0.001, chunk_size=chunk_size
        )

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - start_time

    # ==================== Report results ====================
    print(f"\n{'='*40}")
    print(f"Total poses:       {B}")
    print(f"Collisions:        {collisions.sum().item()} / {B}")
    if collisions.any():
        print(f"Max penetration:   {penetration_depths[collisions].max().item():.6f}")
    else:
        print(f"Max penetration:   N/A (no collisions)")
    print(f"Total time:        {elapsed:.3f}s")
    print(f"Per pose:          {elapsed / B * 1e6:.2f} μs")
    print(f"{'='*40}")

    # ==================== Benchmark ====================
    num_iters = 50
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            batch_collision_detection(
                convex_planes_torch,
                points_B,
                poses,
                threshold=0.001,
                chunk_size=chunk_size,
            )
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()

    avg_ms = (t1 - t0) / num_iters * 1000
    print(
        f"\nBenchmark ({num_iters} iters): {avg_ms:.2f} ms/iter, "
        f"{avg_ms / B * 1000:.2f} μs/pose"
    )


if __name__ == "__main__":
    from embodichain.data import get_data_path

    bottle_a_path = get_data_path("ScannedBottle/moliwulong_processed.ply")
    bottle_b_path = get_data_path("ScannedBottle/yibao_processed.ply")

    bottle_a_mesh = trimesh.load(bottle_a_path)
    bottle_b_mesh = trimesh.load(bottle_b_path)
    bottle_a_verts = torch.tensor(bottle_a_mesh.vertices, dtype=torch.float32)
    bottle_a_faces = torch.tensor(bottle_a_mesh.faces, dtype=torch.int64)
    bottle_b_verts = torch.tensor(bottle_b_mesh.vertices, dtype=torch.float32)
    bottle_b_faces = torch.tensor(bottle_b_mesh.faces, dtype=torch.int64)

    collision_checker = BatchConvexCollisionChecker(bottle_a_verts, bottle_a_faces)
    poses = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1.0],
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
    check_cfg = BatchConvexCollisionCheckerCfg(
        debug=False,
        n_query_mesh_samples=32768,
        collsion_threshold=-0.003,
    )
    collisions, penetrations = collision_checker.query(
        bottle_b_verts, bottle_b_faces, poses, cfg=check_cfg
    )
    print("Collisions:", collisions)
    print("Penetrations:", penetrations)
