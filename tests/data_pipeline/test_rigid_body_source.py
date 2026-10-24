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

"""Causal geometry window ownership and episode-boundary coverage."""

from __future__ import annotations

import pytest
import torch

from embodichain.data_pipeline.datasets.rigid_body import RigidBodySequenceSource


def test_geometry_windows_keep_point_identity_and_owned_storage():
    # Every point coordinate encodes its episode/time index independently.
    positions = torch.arange(2 * 7 * 2 * 4 * 3).reshape(2, 7, 2, 4, 3).float()
    expected = positions.clone()
    source = RigidBodySequenceSource(
        positions,
        torch.ones(2, 2, 3),
        step_dt=0.05,
        source_id="fixed-corpus",
        stride=2,
        episode_lengths=torch.tensor([5, 7]),
    )
    positions.zero_()
    batch = source.get_batch(torch.tensor([0, 1]), torch.tensor([2, 4]))
    torch.testing.assert_close(batch["object_pos_prev"], expected[[0, 1], [0, 2]])
    torch.testing.assert_close(batch["object_pos"], expected[[0, 1], [2, 4]])
    torch.testing.assert_close(batch["object_pos_next"], expected[[0, 1], [4, 6]])
    torch.testing.assert_close(
        batch["object_first_frame_pos"], batch["object_pos_prev"]
    )
    torch.testing.assert_close(batch["delta_times"], torch.full((2,), 0.1))
    batch["object_pos_next"].zero_()
    again = source.get_batch(torch.tensor([0]), torch.tensor([2]))
    torch.testing.assert_close(again["object_pos_next"][0], expected[0, 4])
    with pytest.raises(ValueError, match="episode boundary"):
        source.get_batch(torch.tensor([0]), torch.tensor([3]))


def test_geometry_sampling_replays_rng_and_preserves_valid_lengths():
    source = RigidBodySequenceSource(
        torch.randn(2, 8, 3, 5, 3),
        torch.randn(2, 3, 5, 2),
        step_dt=0.1,
        source_id="ragged",
        episode_lengths=torch.tensor([3, 8]),
        object_lens=torch.tensor([1, 3]),
        object_point_lens=torch.tensor([[4, 5, 5], [3, 4, 5]]),
    )
    first = source.sample_batch(20, generator=torch.Generator().manual_seed(3))
    second = source.sample_batch(20, generator=torch.Generator().manual_seed(3))
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert set(first["object_lens"].tolist()) == {1, 3}
    assert first["vertex_properties"].shape == (20, 3, 5, 2)


def test_geometry_memory_limits_precede_payload_delivery():
    positions, props = torch.zeros(1, 4, 2, 4, 3), torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="memory budget"):
        RigidBodySequenceSource(
            positions, props, step_dt=0.1, source_id="bounded", max_resident_bytes=1
        )
    source = RigidBodySequenceSource(
        positions, props, step_dt=0.1, source_id="bounded", max_batch_bytes=16
    )
    with pytest.raises(ValueError, match="max_batch_bytes"):
        source.sample_batch(1)
