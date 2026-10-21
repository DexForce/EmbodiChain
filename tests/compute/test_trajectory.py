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

"""Tests for distance-based trajectory interpolation utilities."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from embodichain.compute.trajectory import (
    interpolate_minimum_jerk,
    interpolate_with_distance,
    resample_with_distance,
)


def test_minimum_jerk_expands_sample_count_and_preserves_limits() -> None:
    keyframes = torch.tensor([[[0.0], [1.0], [0.5]]])
    step_dt = 0.05

    result = interpolate_minimum_jerk(
        keyframes,
        minimum_sample_count=5,
        step_dt=step_dt,
        velocity_limit=0.5,
        acceleration_limit=1.0,
    )

    assert result.shape[1] > 5
    assert torch.equal(result[:, :1], keyframes[:, :1])
    assert torch.equal(result[:, -1:], keyframes[:, -1:])
    assert any(
        torch.equal(result[:, index], keyframes[:, 1])
        for index in range(1, result.shape[1] - 1)
    )
    interval_velocity = result.diff(dim=1) / step_dt
    interval_acceleration = interval_velocity.diff(dim=1) / step_dt
    assert interval_velocity.abs().max() <= 0.5 + 1.0e-5
    assert interval_acceleration.abs().max() <= 1.0 + 1.0e-4


def test_minimum_jerk_uses_sample_count_as_a_lower_bound() -> None:
    result = interpolate_minimum_jerk(
        torch.tensor([[[0.0], [0.01]]]),
        minimum_sample_count=20,
        step_dt=0.1,
        velocity_limit=1.0,
        acceleration_limit=2.0,
    )

    assert result.shape == (1, 20, 1)


def test_minimum_jerk_validates_dense_float32_output_derivatives() -> None:
    step_dt = 0.001
    velocity_limit = 0.2
    acceleration_limit = 0.5

    result = interpolate_minimum_jerk(
        torch.tensor([[[0.0], [1.0]]], dtype=torch.float32),
        minimum_sample_count=3,
        step_dt=step_dt,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
    )

    interval_velocity = result.diff(dim=1) / step_dt
    boundary_acceleration = (
        torch.cat(
            [
                interval_velocity[:, :1],
                interval_velocity.diff(dim=1),
                -interval_velocity[:, -1:],
            ],
            dim=1,
        )
        / step_dt
    )
    assert torch.all(result.diff(dim=1) >= 0.0)
    assert interval_velocity.abs().max() <= velocity_limit * (1.0 + 1.0e-5)
    assert boundary_acceleration.abs().max() <= acceleration_limit * (1.0 + 1.0e-5)


def test_minimum_jerk_allows_only_stationary_zero_speed_joints() -> None:
    keyframes = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])

    result = interpolate_minimum_jerk(
        keyframes,
        minimum_sample_count=3,
        step_dt=0.05,
        velocity_limit=torch.tensor([0.0, 0.2]),
        acceleration_limit=0.5,
    )

    assert torch.equal(result[..., 0], torch.zeros_like(result[..., 0]))
    with pytest.raises(ValueError, match="zero-velocity joint"):
        interpolate_minimum_jerk(
            torch.tensor([[[0.0, 0.0], [0.1, 1.0]]]),
            minimum_sample_count=3,
            step_dt=0.05,
            velocity_limit=torch.tensor([0.0, 0.2]),
            acceleration_limit=0.5,
        )


def test_minimum_jerk_rejects_unbounded_interval_requests() -> None:
    with pytest.raises(ValueError, match="exceeding the safety limit"):
        interpolate_minimum_jerk(
            torch.tensor([[[0.0], [1.0]]]),
            minimum_sample_count=3,
            step_dt=0.01,
            velocity_limit=1.0e-12,
            acceleration_limit=1.0,
        )


def test_minimum_jerk_rejects_oversized_requested_sample_count() -> None:
    with pytest.raises(ValueError, match="exceeding the safety limit"):
        interpolate_minimum_jerk(
            torch.tensor([[[0.0], [0.0]]]),
            minimum_sample_count=10_002,
            step_dt=0.01,
            velocity_limit=1.0,
            acceleration_limit=1.0,
        )


def test_interpolate_with_distance_preserves_intermediate_keyframe() -> None:
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])

    result = interpolate_with_distance(keyframes, interp_num=5, device="cpu")

    # One interval is reserved per segment; the two remaining intervals are
    # split evenly by largest remainder for segment lengths one and two.
    expected = torch.tensor([[[0.0], [0.5], [1.0], [2.0], [3.0]]])
    assert torch.equal(result, expected)


def test_interpolate_with_distance_allocates_each_batch_independently() -> None:
    keyframes = torch.tensor(
        [
            [[0.0], [1.0], [4.0]],
            [[0.0], [3.0], [4.0]],
        ]
    )

    result = interpolate_with_distance(keyframes, interp_num=6, device="cpu")

    expected = torch.tensor(
        [
            [[0.0], [0.5], [1.0], [2.0], [3.0], [4.0]],
            [[0.0], [1.0], [2.0], [3.0], [3.5], [4.0]],
        ]
    )
    assert torch.equal(result, expected)


def test_interpolate_with_distance_retains_duplicate_keyframes() -> None:
    keyframes = torch.tensor([[[0.0], [0.0], [1.0]]])

    result = interpolate_with_distance(keyframes, interp_num=4, device="cpu")

    expected = torch.tensor([[[0.0], [0.0], [0.5], [1.0]]])
    assert torch.equal(result, expected)


def test_interpolate_with_distance_rejects_insufficient_samples() -> None:
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])

    with pytest.raises(ValueError, match="at least the number of keyframes"):
        interpolate_with_distance(keyframes, interp_num=2, device="cpu")


def test_interpolate_with_distance_repeats_single_keyframe() -> None:
    keyframes = torch.tensor([[[2.0, 3.0]]])

    result = interpolate_with_distance(keyframes, interp_num=3, device="cpu")

    expected = keyframes.expand(-1, 3, -1)
    assert torch.equal(result, expected)


@pytest.mark.gpu
def test_interpolate_with_distance_preserves_keyframes_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]], device="cuda:0")

    result = interpolate_with_distance(keyframes, interp_num=5, device="cuda:0")

    expected = torch.tensor([[[0.0], [0.5], [1.0], [2.0], [3.0]]], device="cuda:0")
    assert torch.equal(result, expected)


def test_resample_with_distance_allows_path_downsampling() -> None:
    wp.init()
    path = torch.tensor([[[0.0], [1.0], [3.0]]])

    result = resample_with_distance(path, interp_num=2, device="cpu")

    expected = torch.tensor([[[0.0], [3.0]]])
    assert torch.equal(result, expected)


def test_resample_with_distance_falls_back_when_warp_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])

    def fail_from_torch(_trajectory: torch.Tensor) -> None:
        raise RuntimeError("Warp runtime unavailable")

    monkeypatch.setattr(wp, "from_torch", fail_from_torch)

    result = resample_with_distance(path, interp_num=3, device="cpu")

    expected = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_warp_trajectory_preserves_endpoints_and_applies_batched_keyframes(
    device: str,
) -> None:
    from embodichain.compute.trajectory import warp_trajectory_qpos

    wp.init()
    trajectory = torch.tensor([[0.0], [1.0], [2.0]])
    indices = torch.tensor([1])
    keyframes = torch.tensor([[[3.0]], [[5.0]]])
    result = warp_trajectory_qpos(trajectory, indices, keyframes, device=device)

    expected = torch.tensor(
        [[[0.0], [3.0], [2.0]], [[0.0], [5.0], [2.0]]], device=device
    )
    assert torch.equal(result, expected)


def test_interpolate_with_nums_preserves_zero_sample_segment_endpoints() -> None:
    from embodichain.compute.trajectory import interpolate_with_nums

    keyframes = torch.tensor([[[0.0], [2.0], [4.0]]])
    result = interpolate_with_nums(keyframes, [0, 2], device="cpu")
    assert torch.equal(result, torch.tensor([[[0.0], [2.0], [3.0], [4.0]]]))
