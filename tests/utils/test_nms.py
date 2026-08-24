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
