# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

# CUDA is mandatory for this module; there is intentionally no CPU fallback.
assert torch.cuda.is_available(), "cuda_knn requires a CUDA-capable device."


def compute_nearest_distance_to_list_cuda(
    tensor_a: torch.Tensor,
    tensor_list: List[torch.Tensor],
    batch_mode: bool = False,
    return_vectors: bool = False,
    return_tensor_indices: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    CUDA-accelerated parallel computation of nearest distances from query
    points to a list of reference point sets. All tensors must reside on the
    same device.

    Args:
        tensor_a: Query tensor
            - If batch_mode=False: shape [N, D]
            - If batch_mode=True: shape [B, N, D]
        tensor_list: List of reference tensors
            - If batch_mode=False: [M_i, D] each
            - If batch_mode=True: [B, M_i, D] each
        batch_mode: Whether to use batch processing
        return_vectors: If True, also return vectors from query to nearest points
        return_tensor_indices: If False, skip the per-list-entry loop that maps the nearest point to
            the index of its list entry (returned as None); distances and vectors are unaffected

    Returns:
        min_distances: Minimum distances [N] or [B, N]
        min_indices: Tensor list indices [N] or [B, N]
        vectors: (optional) Vectors from query to nearest points [N, D] or [B, N, D]
                 vectors = nearest_point - query_point
    """
    # All tensors must reside on the same device
    device = tensor_a.device
    for i, t in enumerate(tensor_list):
        if t.device != device:
            raise RuntimeError(
                f"All tensors must be on the same device: tensor_a is on {device}, tensor_list[{i}] is on {t.device}"
            )

    if len(tensor_list) == 0:
        raise ValueError("tensor_list cannot be empty")

    device = tensor_a.device
    original_batch_mode = batch_mode

    if not batch_mode:
        # Add batch dimension for unified processing
        tensor_a = tensor_a.unsqueeze(0)  # [1, N, D]
        tensor_list = [t.unsqueeze(0) for t in tensor_list]  # Each becomes [1, M_i, D]

    B, N, D = tensor_a.shape

    # Strategy 1: Concatenate all references for single parallel computation
    ref_sizes = [ref.shape[1] for ref in tensor_list]
    all_refs = torch.cat(tensor_list, dim=1)  # [B, total_M, D]

    # Single CUDA-parallel distance computation
    dist_matrix = torch.cdist(tensor_a, all_refs, p=2.0)  # [B, N, total_M]

    # Find minimum distances and indices in parallel
    min_dists, min_idx = dist_matrix.min(dim=2)  # Both [B, N]

    # Convert concatenated indices to tensor list indices
    min_tensor_indices = None
    if return_tensor_indices:
        min_tensor_indices = torch.zeros_like(min_idx, dtype=torch.long)
        cumsum = 0
        for i, size in enumerate(ref_sizes):
            mask = (min_idx >= cumsum) & (min_idx < cumsum + size)
            min_tensor_indices[mask] = i
            cumsum += size

    # Compute vectors if requested
    vectors = None
    if return_vectors:
        # Get the nearest points using advanced indexing
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, N)
        nearest_points = all_refs[batch_indices, min_idx]  # [B, N, D]
        # Vector from query to nearest point
        vectors = nearest_points - tensor_a  # [B, N, D]

    if not original_batch_mode:
        min_dists = min_dists.squeeze(0)
        if min_tensor_indices is not None:
            min_tensor_indices = min_tensor_indices.squeeze(0)
        if vectors is not None:
            vectors = vectors.squeeze(0)

    if return_vectors:
        return min_dists, min_tensor_indices, vectors
    else:
        return min_dists, min_tensor_indices


def compute_knn_all_objects_vectorized(
    positions: torch.Tensor,
    valid_vtx_mask: torch.Tensor,
    FAR: float = 1e6,
    valid_obj_mask: torch.Tensor = None,
    num_groups: int = 1,
    active_slots: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Efficient batch-mode KNN for multi-object collision detection.

    Args:
        positions: (B, Nobj, Nv, 3) - vertex positions for all objects
        valid_vtx_mask: (B, Nobj, Nv) - True for valid vertices
        FAR: Distance offset for invalid vertices (default 1e6)
        valid_obj_mask: (B, Nobj) - True for valid objects
        num_groups: Number of groups to split objects into for GPU-parallel processing.
                    num_groups=1: per-object loop (Nobj iterations)
                    num_groups>1: GPU-parallel within each group (num_groups iterations)
                    Example: Nobj=564, num_groups=6 -> 94 objects processed in parallel per group
        active_slots: Optional list of object slots that hold a valid object in at least one batch element
                    (per-object path only). Slots outside it are padding in every sample and their output
                    is zeroed either way, so they are not searched; valid objects still search every other slot.

    Returns:
        dist_vectors: (B, Nobj, Nv, 3) - vector from each vertex to nearest point on OTHER objects
    """
    B, Nobj, Nv, D = positions.shape
    device = positions.device

    if Nobj < 2:
        raise RuntimeError(
            "compute_knn_all_objects_vectorized requires at least 2 objects"
        )

    if valid_obj_mask is None:
        valid_obj_mask = valid_vtx_mask.any(dim=2)  # (B, Nobj)

    dist_vectors = torch.zeros(B, Nobj, Nv, D, device=device)

    # Pre-compute masked positions (FAR offset for invalid vertices)
    masked_positions = positions.clone()
    vtx_mask_3d = valid_vtx_mask.unsqueeze(-1)  # (B, Nobj, Nv, 1)
    masked_positions = torch.where(
        vtx_mask_3d, masked_positions, masked_positions + FAR
    )

    # ========== GPU-PARALLEL GROUPED PROCESSING ==========
    if num_groups > 1 and Nobj > num_groups:
        group_size = (Nobj + num_groups - 1) // num_groups

        # All reference points (all objects, masked): (B, Nobj*Nv, 3)
        all_refs_flat = masked_positions.reshape(B, Nobj * Nv, D)

        for group_idx in range(num_groups):
            g_start = group_idx * group_size
            g_end = min(g_start + group_size, Nobj)
            G = g_end - g_start  # Objects in this group

            # Query points for this group: (B, G, Nv, 3) -> (B, G*Nv, 3)
            group_queries = positions[:, g_start:g_end].reshape(B, G * Nv, D)

            # Compute full distance matrix: (B, G*Nv, Nobj*Nv)
            dist_matrix = torch.cdist(group_queries, all_refs_flat, p=2.0)

            # Create self-exclusion mask: for each query vertex, mask out vertices from same object
            # Query vertex q belongs to object (g_start + q // Nv)
            # Ref vertex r belongs to object (r // Nv)
            query_obj_idx = (
                torch.arange(G, device=device).unsqueeze(1).expand(G, Nv).reshape(-1)
                + g_start
            )  # (G*Nv,)
            ref_obj_idx = (
                torch.arange(Nobj, device=device)
                .unsqueeze(1)
                .expand(Nobj, Nv)
                .reshape(-1)
            )  # (Nobj*Nv,)

            # Self mask: (G*Nv, Nobj*Nv), True where query and ref belong to same object
            self_mask = query_obj_idx.unsqueeze(1) == ref_obj_idx.unsqueeze(
                0
            )  # (G*Nv, Nobj*Nv)

            # Apply FAR to self-distances so they're not selected as minimum
            dist_matrix = dist_matrix + self_mask.float() * FAR  # (B, G*Nv, Nobj*Nv)

            # Find minimum distance and corresponding ref index
            min_dists, min_indices = dist_matrix.min(dim=2)  # (B, G*Nv)

            # Get nearest points using advanced indexing
            batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, G * Nv)
            nearest_points = all_refs_flat[batch_idx, min_indices]  # (B, G*Nv, 3)

            # Compute vectors from query to nearest
            vectors = nearest_points - group_queries  # (B, G*Nv, 3)

            # Reshape back and store: (B, G*Nv, 3) -> (B, G, Nv, 3)
            dist_vectors[:, g_start:g_end] = vectors.reshape(B, G, Nv, D)

    else:
        # ========== PER-OBJECT PROCESSING ==========
        # The references of object i are all other slots in slot order: the flattened (B, Nobj*Nv, D)
        # tensor without object i's block, the same tensor as concatenating the other slots one by one.
        refs_flat = masked_positions.reshape(B, Nobj * Nv, D)
        for i in (range(Nobj) if active_slots is None else active_slots):
            body_pos = positions[:, i]  # (B, Nv, 3)
            ref_list = [refs_flat[:, : i * Nv], refs_flat[:, (i + 1) * Nv :]]

            _, _, body_dist_vec = compute_nearest_distance_to_list_cuda(
                body_pos,
                ref_list,
                batch_mode=True,
                return_vectors=True,
                return_tensor_indices=False,
            )
            dist_vectors[:, i] = body_dist_vec

    # Zero out invalid query vertices
    vtx_mask_expanded = valid_vtx_mask.unsqueeze(-1)
    dist_vectors = torch.where(
        vtx_mask_expanded, dist_vectors, torch.zeros_like(dist_vectors)
    )

    # Zero out invalid objects
    obj_mask_expanded = valid_obj_mask.unsqueeze(-1).unsqueeze(-1)
    dist_vectors = torch.where(
        obj_mask_expanded, dist_vectors, torch.zeros_like(dist_vectors)
    )

    return dist_vectors


def knn_pack(
    valid_vtx_mask: torch.Tensor, valid_obj_mask: Optional[torch.Tensor] = None
) -> dict:
    """Packed layout of every sample's valid vertices for compute_knn_packed.

    The masks do not change over a batch or a rollout, so this is computed once (one host sync for the packed
    width) and reused at every step. valid_vtx_mask (B, Nobj, Nv); valid_obj_mask (B, Nobj) or None.
    Packed order = slot order, then vertex order: the order in which the per-object path lists candidates.
    """
    B, Nobj, Nv = valid_vtx_mask.shape
    vm = valid_vtx_mask.bool()
    if valid_obj_mask is not None:
        vm = vm & valid_obj_mask.bool().unsqueeze(-1)
    n_obj = vm.any(dim=2).sum(dim=1)
    if bool((n_obj < 2).any()):
        raise ValueError(
            "compute_knn_packed: every sample needs at least two objects with valid vertices"
        )
    flat = vm.reshape(B, Nobj * Nv)
    counts = flat.sum(dim=1)
    n = int(counts.max())
    order = torch.argsort((~flat).to(torch.int8), dim=1, stable=True)[
        :, :n
    ]  # (B, n) valid first
    valid = torch.arange(n, device=flat.device).unsqueeze(0) < counts.unsqueeze(
        1
    )  # (B, n)
    obj = torch.where(
        valid, torch.div(order, Nv, rounding_mode="floor"), torch.full_like(order, -1)
    )
    return {"order": order, "valid": valid, "obj": obj, "shape": (B, Nobj, Nv)}


def compute_knn_packed(
    positions: torch.Tensor, pack: dict, block_elems: int = 1 << 27
) -> torch.Tensor:
    """Vector from every valid vertex to the nearest valid vertex of any OTHER object, over the packed valid
    vertices only (padded vertices and padded object slots never enter the distance matrix).

    Same candidates in the same order and the same first-minimum tie-breaking as compute_knn_all_objects_vectorized,
    and the squared distance is cdist's own construction ([-2x, |x|^2, 1] . [y, 1, |y|^2]) without the clamp and
    square root over the whole matrix (monotone, so the argmin is unchanged). Results agree with the per-object
    path up to the floating-point rounding of the distance values (a near-tie may pick a different, equally near
    vertex). positions (B, Nobj, Nv, D); pack from knn_pack; returns (B, Nobj, Nv, D), zero at invalid entries.
    """
    B, Nobj, Nv = pack["shape"]
    D = positions.shape[-1]
    order, valid, obj = pack["order"], pack["valid"], pack["obj"]
    n = order.shape[1]
    with torch.autocast(device_type="cuda", enabled=False):
        flat = positions.float().reshape(B, Nobj * Nv, D)
        P = torch.gather(flat, 1, order.unsqueeze(-1).expand(B, n, D))  # (B, n, D)
        sq = P.pow(2).sum(dim=-1, keepdim=True)
        ones = torch.ones_like(sq)
        Xq = torch.cat([P.mul(-2), sq, ones], dim=-1)  # (B, n, D+2)
        Yr = torch.cat([P, ones, sq], dim=-1).transpose(1, 2)  # (B, D+2, n)
        rows = max(1, min(n, block_elems // max(1, B * n)))
        idx = torch.empty(B, n, dtype=torch.long, device=P.device)
        for s0 in range(0, n, rows):
            s1 = min(n, s0 + rows)
            d2 = torch.bmm(Xq[:, s0:s1], Yr)  # (B, rows, n)
            d2.masked_fill_(
                (obj[:, s0:s1, None] == obj[:, None, :]) | ~valid[:, None, :],
                float("inf"),
            )
            idx[:, s0:s1] = d2.argmin(dim=2)
        nearest = torch.gather(P, 1, idx.unsqueeze(-1).expand(B, n, D))
        vec = torch.where(
            valid.unsqueeze(-1), nearest - P, torch.zeros((), device=P.device)
        )
        out = torch.zeros(B, Nobj * Nv, D, device=P.device, dtype=torch.float32)
        out.scatter_(1, order.unsqueeze(-1).expand(B, n, D), vec)
    return out.reshape(B, Nobj, Nv, D).to(positions.dtype)


def compute_nearest_distance_to_list_cuda_chunked(
    tensor_a: torch.Tensor, tensor_list: List[torch.Tensor], chunk_size: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient CUDA version using chunked processing for large tensors.
    Processes distance computation in chunks to avoid OOM on large datasets.

    Args:
        tensor_a: Query tensor on CUDA, shape [N, D]
        tensor_list: List of reference tensors on CUDA, each [M_i, D]
        chunk_size: Maximum number of reference points to process at once

    Returns:
        min_distances: shape [N]
        min_tensor_indices: shape [N]
    """
    if not tensor_a.is_cuda:
        raise RuntimeError("tensor_a must be on CUDA; CPU execution is not supported.")

    for i, t in enumerate(tensor_list):
        if not t.is_cuda:
            raise RuntimeError(
                f"tensor_list[{i}] must be on CUDA; CPU execution is not supported."
            )

    N, D = tensor_a.shape
    device = tensor_a.device

    # Initialize results
    global_min_dists = torch.full((N,), float("inf"), device=device)
    global_tensor_idx = torch.zeros(N, dtype=torch.long, device=device)

    for tensor_idx, ref_tensor in enumerate(tensor_list):
        M = ref_tensor.shape[0]

        # Process this reference tensor in chunks
        chunk_min_dists = torch.full((N,), float("inf"), device=device)

        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            ref_chunk = ref_tensor[start_idx:end_idx]

            # Compute distances for this chunk
            dist_matrix = torch.cdist(tensor_a, ref_chunk, p=2.0)  # [N, chunk_size]
            chunk_dists, _ = dist_matrix.min(dim=1)  # [N]

            # Update chunk minimums
            chunk_min_dists = torch.minimum(chunk_min_dists, chunk_dists)

        # Update global minimums
        update_mask = chunk_min_dists < global_min_dists
        global_min_dists[update_mask] = chunk_min_dists[update_mask]
        global_tensor_idx[update_mask] = tensor_idx

    return global_min_dists, global_tensor_idx


def compute_nearest_distance_to_list_cuda_optimized(
    tensor_a: torch.Tensor, tensor_list: List[torch.Tensor], max_memory_mb: float = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive CUDA implementation that chooses strategy based on memory constraints.

    Args:
        tensor_a: Query tensor on CUDA, shape [N, D]
        tensor_list: List of reference tensors on CUDA
        max_memory_mb: Maximum memory to use in MB

    Returns:
        min_distances: shape [N]
        min_tensor_indices: shape [N]
    """
    if not tensor_a.is_cuda:
        raise RuntimeError(
            "All tensors must be on CUDA; CPU execution is not supported."
        )

    N, D = tensor_a.shape

    # Estimate memory requirement
    total_ref_points = sum(t.shape[0] for t in tensor_list)
    memory_needed_mb = (N * total_ref_points * 4) / (1024 * 1024)  # float32 = 4 bytes

    if memory_needed_mb <= max_memory_mb:
        # Use fast concatenated version
        return compute_nearest_distance_to_list_cuda(
            tensor_a, tensor_list, batch_mode=False
        )
    else:
        # Use memory-efficient chunked version
        chunk_size = int(max_memory_mb * 1024 * 1024 / (N * 4))
        return compute_nearest_distance_to_list_cuda_chunked(
            tensor_a, tensor_list, chunk_size
        )


# Self-tests
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CUDA-accelerated nearest distance computation")
    print("=" * 60)

    # Create test data ON CUDA
    tensor_a = torch.rand(1000, 3).cuda()  # 1000 query points
    tensor_list = [
        torch.rand(500, 3).cuda(),  # 500 reference points
        torch.rand(800, 3).cuda(),  # 800 reference points
        torch.rand(1200, 3).cuda(),  # 1200 reference points
    ]

    print("\nTest 1: Basic CUDA version")
    try:
        min_dists, min_indices = compute_nearest_distance_to_list_cuda(
            tensor_a, tensor_list, batch_mode=False
        )
        print(f"Success. Min distances shape: {min_dists.shape}")
        print(f"  First 5 distances: {min_dists[:5]}")
        print(f"  Tensor indices: {min_indices[:5]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTest 1b: With vector output")
    try:
        min_dists, min_indices, vectors = compute_nearest_distance_to_list_cuda(
            tensor_a, tensor_list, batch_mode=False, return_vectors=True
        )
        print(f"Success. Vectors shape: {vectors.shape}")
        print(f"  First vector: {vectors[0]}")
        # Verify vector length matches distance
        vec_norm = vectors[0].norm()
        print(f"  Vector norm: {vec_norm:.6f}, Distance: {min_dists[0]:.6f}")
        print(f"  Match: {torch.allclose(vec_norm, min_dists[0], rtol=1e-5)}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTest 2: Batch mode")
    tensor_a_batch = torch.rand(4, 100, 3).cuda()  # Batch of 4
    tensor_list_batch = [
        torch.rand(4, 50, 3).cuda(),
        torch.rand(4, 80, 3).cuda(),
    ]

    try:
        min_dists_batch, min_indices_batch = compute_nearest_distance_to_list_cuda(
            tensor_a_batch, tensor_list_batch, batch_mode=True
        )
        print(f"Success. Batch distances shape: {min_dists_batch.shape}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTest 3: Chunked version for large tensors")
    try:
        min_dists_chunked, min_indices_chunked = (
            compute_nearest_distance_to_list_cuda_chunked(
                tensor_a, tensor_list, chunk_size=100
            )
        )
        print(f"Success. Chunked distances shape: {min_dists_chunked.shape}")

        # Verify chunked gives same result as full
        diff = (min_dists - min_dists_chunked).abs().max()
        print(f"  Difference from full version: {diff:.6f}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTest 4: Testing with CPU tensor (should fail)")
    tensor_cpu = torch.rand(10, 3)  # CPU tensor
    try:
        min_dists, min_indices = compute_nearest_distance_to_list_cuda(
            tensor_cpu, tensor_list, batch_mode=False
        )
        print("FAILED: expected an error for the CPU tensor")
    except RuntimeError as e:
        print(f"Correctly rejected CPU tensor: {e}")

    print("\nTest 5: Performance comparison")
    import time

    # Large test case
    large_query = torch.rand(5000, 3).cuda()
    large_refs = [torch.rand(2000, 3).cuda() for _ in range(5)]

    # Time the optimized version
    start = time.time()
    result = compute_nearest_distance_to_list_cuda_optimized(large_query, large_refs)
    cuda_time = time.time() - start

    print(f"Processed 5000 queries against 10000 references in {cuda_time:.3f}s")
    print(
        f"  Throughput: {5000 * 10000 / cuda_time / 1e6:.1f} million distance computations/second"
    )

    print("\n" + "=" * 60)
    print("Test 6: Vectorized multi-object KNN (correctness verification)")
    print("=" * 60)

    # Create test data: B batches, Nobj objects, Nv vertices each
    B, Nobj, Nv = 4, 5, 100
    FAR = 1e6

    positions = torch.rand(B, Nobj, Nv, 3).cuda()
    # Create a valid mask (80% of vertices are valid)
    valid_mask = torch.rand(B, Nobj, Nv).cuda() > 0.2

    print(f"  Test setup: B={B}, Nobj={Nobj}, Nv={Nv}")
    print(f"  Valid vertices per object: {valid_mask.float().mean(dim=(0,2)) * Nv}")

    # Method 1: loop-based reference implementation
    print("\n  Method 1: Loop-based KNN...")
    dist_vectors_loop = torch.zeros(B, Nobj, Nv, 3, device=positions.device)

    start = time.time()
    for i in range(Nobj):
        body_pos = positions[:, i]  # (B, Nv, 3)
        ref_list = []
        for j in range(Nobj):
            if j == i:
                continue
            other_pos = positions[:, j].clone()
            other_vmask = valid_mask[:, j].unsqueeze(-1)  # (B, Nv, 1)
            other_pos = torch.where(other_vmask, other_pos, other_pos + FAR)
            ref_list.append(other_pos)

        _, _, body_dist_vec = compute_nearest_distance_to_list_cuda(
            body_pos, ref_list, batch_mode=True, return_vectors=True
        )
        dist_vectors_loop[:, i] = body_dist_vec
    loop_time = time.time() - start
    print(f"  Loop-based completed in {loop_time*1000:.2f}ms")

    # Method 2: Vectorized approach
    print("\n  Method 2: Vectorized KNN...")
    start = time.time()
    dist_vectors_vec = compute_knn_all_objects_vectorized(positions, valid_mask, FAR)
    vec_time = time.time() - start
    print(f"  Vectorized completed in {vec_time*1000:.2f}ms")

    # Compare results
    print("\n  Comparing results...")

    # Only compare valid query vertices (invalid queries may have different results)
    valid_query_mask = valid_mask.unsqueeze(-1).expand_as(dist_vectors_loop)

    diff = (dist_vectors_loop - dist_vectors_vec).abs()
    diff_valid = diff[valid_query_mask]

    max_diff = diff_valid.max().item() if diff_valid.numel() > 0 else 0
    mean_diff = diff_valid.mean().item() if diff_valid.numel() > 0 else 0

    print(f"  Max difference (valid vertices): {max_diff:.6e}")
    print(f"  Mean difference (valid vertices): {mean_diff:.6e}")

    if max_diff < 1e-5:
        print(f"  PASSED: Results match within tolerance (1e-5)")
    else:
        print(f"  FAILED: Results differ by more than tolerance")
        # Debug: find where they differ
        diff_mask = diff > 1e-5
        num_diff = diff_mask.sum().item()
        print(f"    Number of differing elements: {num_diff}")

    # Speedup
    speedup = loop_time / vec_time if vec_time > 0 else float("inf")
    print(f"\n  Speedup: {speedup:.2f}x")

    # Test with larger data
    print("\n  Testing with larger data (B=8, Nobj=10, Nv=500)...")
    B2, Nobj2, Nv2 = 8, 10, 500
    positions2 = torch.rand(B2, Nobj2, Nv2, 3).cuda()
    valid_mask2 = torch.rand(B2, Nobj2, Nv2).cuda() > 0.1

    # Warmup
    _ = compute_knn_all_objects_vectorized(positions2, valid_mask2, FAR)
    torch.cuda.synchronize()

    start = time.time()
    _ = compute_knn_all_objects_vectorized(positions2, valid_mask2, FAR)
    torch.cuda.synchronize()
    vec_time2 = time.time() - start
    print(f"  Vectorized (large): {vec_time2*1000:.2f}ms")
    print(f"    Throughput: {B2 * Nobj2 * Nv2 / vec_time2 / 1e6:.2f}M vertices/sec")

    print("\n" + "=" * 60)
    print("Test 7: Invalid vertex/object masking verification")
    print("=" * 60)

    B, Nobj, Nv = 2, 4, 50
    positions = torch.rand(B, Nobj, Nv, 3).cuda()

    # Create mask with some fully invalid objects and some invalid vertices
    valid_vtx_mask = torch.ones(B, Nobj, Nv, dtype=torch.bool, device="cuda")
    valid_obj_mask = torch.ones(B, Nobj, dtype=torch.bool, device="cuda")

    # Make object 2 fully invalid (no valid vertices)
    valid_vtx_mask[:, 2, :] = False
    valid_obj_mask[:, 2] = False

    # Make some vertices invalid in valid objects
    valid_vtx_mask[0, 0, 10:20] = False  # batch 0, object 0, vertices 10-19 invalid
    valid_vtx_mask[1, 1, 30:40] = False  # batch 1, object 1, vertices 30-39 invalid

    print(f"  Setup: B={B}, Nobj={Nobj}, Nv={Nv}")
    print(f"  Object 2 is fully invalid (all vertices masked)")
    print(f"  Some vertices in valid objects are also masked")

    dist_vectors = compute_knn_all_objects_vectorized(
        positions, valid_vtx_mask, FAR, valid_obj_mask
    )

    # Check 1: Invalid object (object 2) should have ALL zero distance vectors
    obj2_dist = dist_vectors[:, 2, :, :]  # (B, Nv, 3)
    obj2_is_zero = (obj2_dist.abs() < 1e-10).all()
    print(
        f"\n  Check 1: Invalid object (obj 2) has zero dist_vec: {obj2_is_zero.item()}"
    )
    if obj2_is_zero:
        print("  PASSED: Invalid object has all zeros")
    else:
        print(
            f"  FAILED: Non-zero values found: max={obj2_dist.abs().max().item():.6e}"
        )

    # Check 2: Invalid vertices in valid objects should have zero distance vectors
    invalid_vtx_dist_0 = dist_vectors[0, 0, 10:20, :]  # batch 0, obj 0, vtx 10-19
    invalid_vtx_dist_1 = dist_vectors[1, 1, 30:40, :]  # batch 1, obj 1, vtx 30-39

    vtx0_is_zero = (invalid_vtx_dist_0.abs() < 1e-10).all()
    vtx1_is_zero = (invalid_vtx_dist_1.abs() < 1e-10).all()

    print(f"\n  Check 2: Invalid vertices have zero dist_vec:")
    print(f"    batch 0, obj 0, vtx 10-19: {vtx0_is_zero.item()}")
    print(f"    batch 1, obj 1, vtx 30-39: {vtx1_is_zero.item()}")
    if vtx0_is_zero and vtx1_is_zero:
        print("  PASSED: Invalid vertices have all zeros")
    else:
        print(f"  FAILED: Non-zero values found")

    # Check 3: Valid vertices in valid objects should have NON-zero distance vectors
    # (at least most of them, since they should find nearest neighbors on other objects)
    valid_vtx_dist_0 = dist_vectors[0, 0, :10, :]  # batch 0, obj 0, vtx 0-9 (valid)
    valid_vtx_dist_1 = dist_vectors[0, 1, :30, :]  # batch 0, obj 1, vtx 0-29 (valid)

    valid_has_nonzero_0 = (valid_vtx_dist_0.abs() > 1e-10).any()
    valid_has_nonzero_1 = (valid_vtx_dist_1.abs() > 1e-10).any()

    print(f"\n  Check 3: Valid vertices have NON-zero dist_vec (found neighbors):")
    print(f"    batch 0, obj 0, vtx 0-9: {valid_has_nonzero_0.item()}")
    print(f"    batch 0, obj 1, vtx 0-29: {valid_has_nonzero_1.item()}")
    if valid_has_nonzero_0 and valid_has_nonzero_1:
        print("  PASSED: Valid vertices have non-zero distances")
    else:
        print("  FAILED: Valid vertices should find neighbors on other objects")

    print("\n" + "=" * 60)
    print("All tests completed! CUDA acceleration is working.")
