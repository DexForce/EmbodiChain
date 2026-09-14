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

import math

import numpy as np
import pytest
import torch
import warp as wp

from embodichain.utils.nms import pose_nms, pose_nms_indices


def _pose(
    x: float = 0.0, yaw: float = 0.0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    pose = torch.eye(4, dtype=dtype)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    pose[:2, :2] = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=dtype)
    pose[0, 3] = x
    return pose


def test_pose_nms_uses_relative_translation_and_rotation() -> None:
    close_angle = math.radians(2.0)
    separate_angle = math.radians(10.0)
    poses = torch.stack(
        [
            _pose(),
            _pose(x=0.001, yaw=close_angle),
            _pose(x=0.001, yaw=separate_angle),
            _pose(x=0.01),
        ]
    )

    filtered, indices = pose_nms(
        poses, angle_th=math.radians(5.0), dist_th=0.003, chunk_size=2
    )

    assert indices.tolist() == [0, 2, 3]
    assert torch.equal(filtered, poses[indices])


def test_pose_nms_supports_float64_input_without_changing_output_dtype() -> None:
    poses = torch.stack(
        [_pose(dtype=torch.float64), _pose(x=0.001, dtype=torch.float64)]
    )

    filtered, indices = pose_nms(poses, dist_th=0.003, chunk_size=1)

    assert indices.tolist() == [0]
    assert filtered.dtype == torch.float64


def test_pose_nms_neighbor_priority_is_deterministic() -> None:
    poses = torch.stack([_pose(x=0.0), _pose(x=0.001), _pose(x=0.002)])

    indices = pose_nms_indices(
        poses,
        angle_th=math.radians(5.0),
        dist_th=0.0015,
        preserve_order=False,
        chunk_size=2,
    )

    assert indices.tolist() == [0, 2]


def test_pose_nms_empty_input_and_disabled_thresholds() -> None:
    empty = torch.empty((0, 4, 4))
    assert pose_nms(empty)[1].numel() == 0

    poses = torch.stack([_pose(), _pose()])
    _, indices = pose_nms(poses, angle_th=0.0)
    assert indices.tolist() == [0, 1]


def test_pose_nms_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="expected \\(N, 4, 4\\)"):
        pose_nms(torch.eye(4))

    with pytest.raises(ValueError, match="positive integer"):
        pose_nms(torch.eye(4).unsqueeze(0), chunk_size=0)


@pytest.mark.gpu
def test_pose_nms_cpu_input_ignores_warp_current_cuda_device() -> None:
    """Regress a CUDA-current-device launch with CPU pose buffers."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    wp.init()
    previous_device = wp.get_device()
    try:
        wp.set_device("cuda:0")
        poses = torch.stack([_pose(), _pose(x=0.001)])

        filtered, indices = pose_nms(poses, dist_th=0.003, chunk_size=1)

        assert indices.tolist() == [0]
        assert filtered.device.type == "cpu"
    finally:
        wp.set_device(previous_device)


@pytest.mark.gpu
def test_pose_nms_cuda_input_keeps_indices_on_cuda() -> None:
    """Exercise both Warp kernels with CUDA-resident pose components."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    poses = torch.stack([_pose(), _pose(x=0.001), _pose(x=0.01)]).cuda()

    indices = pose_nms_indices(
        poses,
        dist_th=0.003,
        preserve_order=False,
        chunk_size=1,
    )

    assert indices.tolist() == [2, 0]
    assert indices.device.type == "cuda"


def _reference_pose_nms_indices(
    poses: torch.Tensor,
    angle_th: float,
    dist_th: float,
    preserve_order: bool,
) -> list[int]:
    """Literal O(N^2) implementation of the documented NMS semantics."""
    from embodichain.utils.nms import _poses_to_components

    num_poses = poses.shape[0]
    positions, quaternions = _poses_to_components(poses)
    rotation_always_close = angle_th > np.pi
    cos_th = 0.0 if rotation_always_close else float(np.cos(0.5 * angle_th))
    dist_sq = float(dist_th * dist_th)

    diff = positions[:, None, :] - positions[None, :, :]
    close = diff.pow(2).sum(-1) < dist_sq
    if not rotation_always_close:
        dots = (quaternions[:, None, :] * quaternions[None, :, :]).sum(-1)
        close &= dots.abs() > cos_th
    close.fill_diagonal_(False)

    if preserve_order:
        order = list(range(num_poses))
    else:
        counts = close.sum(dim=1).cpu()
        priority = counts * (num_poses + 1) + torch.arange(num_poses)
        order = torch.argsort(priority).tolist()

    close_np = close.cpu().numpy()
    suppressed = np.zeros(num_poses, dtype=bool)
    keep: list[int] = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        suppressed |= close_np[idx]
        suppressed[idx] = True
    return keep


def _clustered_poses(n: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Random pose set with deliberate near-duplicate clusters."""
    g = torch.Generator().manual_seed(seed)
    n_centers = max(1, n // 8)
    centers_pos = torch.rand(n_centers, 3, generator=g) * 0.5
    centers_aa = torch.rand(n_centers, 3, generator=g) * 2.0 - 1.0
    assign = torch.randint(0, n_centers, (n,), generator=g)
    # Jitter spans both sides of the thresholds (3 mm / 5 deg defaults).
    pos = centers_pos[assign] + (torch.rand(n, 3, generator=g) - 0.5) * 0.01
    aa = centers_aa[assign] + (torch.rand(n, 3, generator=g) - 0.5) * 0.3

    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    k = torch.zeros(n, 3, 3)
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3).expand(n, 3, 3)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    rot = eye + sin * k + (1 - cos) * (k @ k)

    poses = torch.eye(4).repeat(n, 1, 1)
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = pos
    return poses.to(device)


@pytest.mark.parametrize("preserve_order", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 7, 128, 2048])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pose_nms_matches_reference_semantics(preserve_order, chunk_size, seed):
    """Blocked batched implementation must match the literal reference."""
    poses = _clustered_poses(400, seed)
    expected = _reference_pose_nms_indices(
        poses, angle_th=np.pi / 36, dist_th=0.003, preserve_order=preserve_order
    )
    got = pose_nms_indices(
        poses,
        angle_th=np.pi / 36,
        dist_th=0.003,
        preserve_order=preserve_order,
        chunk_size=chunk_size,
    )
    assert got.tolist() == expected


@pytest.mark.parametrize("preserve_order", [True, False])
def test_pose_nms_rotation_always_close_branch(preserve_order):
    """angle_th > pi treats every rotation as close (translation only)."""
    poses = _clustered_poses(200, seed=3)
    expected = _reference_pose_nms_indices(
        poses, angle_th=4.0, dist_th=0.02, preserve_order=preserve_order
    )
    got = pose_nms_indices(
        poses, angle_th=4.0, dist_th=0.02, preserve_order=preserve_order
    )
    assert got.tolist() == expected


def test_pose_nms_all_duplicates_keep_single():
    poses = _pose().repeat(500, 1, 1)
    indices = pose_nms_indices(poses, preserve_order=True)
    assert indices.tolist() == [0]


@pytest.mark.gpu
def test_pose_nms_cuda_matches_cpu_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    poses = _clustered_poses(400, seed=4)
    expected = _reference_pose_nms_indices(
        poses, angle_th=np.pi / 36, dist_th=0.003, preserve_order=False
    )
    got = pose_nms_indices(poses.cuda(), preserve_order=False)
    assert got.cpu().tolist() == expected


def _tight_clustered_poses(n: int, seed: int) -> torch.Tensor:
    """Heavily duplicated pose set: few survivors, exercising alive filtering."""
    g = torch.Generator().manual_seed(seed)
    n_centers = max(1, n // 100)
    base = _clustered_poses(n_centers, seed)
    assign = torch.randint(0, n_centers, (n,), generator=g)
    poses = base[assign].clone()
    # Jitter well below the 3 mm / 5 deg thresholds.
    poses[:, :3, 3] += (torch.rand(n, 3, generator=g) - 0.5) * 0.001
    return poses


@pytest.mark.parametrize("preserve_order", [True, False])
def test_pose_nms_heavy_suppression_matches_reference(preserve_order):
    """Large tight-cluster input (the realistic grasp-candidate profile):
    the CUDA-offloaded pairwise math must reproduce the literal CPU
    reference exactly."""
    poses = _tight_clustered_poses(6000, seed=5)
    expected = _reference_pose_nms_indices(
        poses, angle_th=np.pi / 36, dist_th=0.003, preserve_order=preserve_order
    )
    got = pose_nms_indices(poses, preserve_order=preserve_order)
    assert got.tolist() == expected
    # Suppression must actually be heavy for this test to mean anything.
    assert len(expected) < 6000 // 10
