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

from __future__ import annotations

import math
import pytest
import torch

from embodichain.compute.geometry.cable import (
    cable_endpoints,
    sample_cable_material,
    cable_grasp_slip,
)

pytestmark = pytest.mark.no_sim


def test_material_coordinates_follow_rotated_capsule_axes_and_world_origins():
    state = torch.zeros(2, 2, 13)
    state[..., 3] = -math.sqrt(0.5)
    state[..., 6] = math.sqrt(0.5)
    state[..., :3] = torch.tensor([[0, 0.1, 1], [0, 0.3, 1]])
    state[1, :, :3] += torch.tensor([3, 5, 0])
    lengths = torch.tensor([0.2, 0.2])
    point, tangent, index = sample_cable_material(state, lengths, 0.3)
    assert index == 1
    torch.testing.assert_close(point, torch.tensor([[0, 0.3, 1], [3, 5.3, 1]]))
    torch.testing.assert_close(
        tangent, torch.tensor([[0, 1, 0], [0, 1, 0.0]]), atol=1e-6, rtol=0
    )
    first, last = cable_endpoints(state, lengths)
    torch.testing.assert_close(
        first[0, 0], torch.tensor([0, 0, 1.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        last[0, -1], torch.tensor([0, 0.4, 1.0]), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("distance", [-0.01, 0.41, float("nan")])
def test_material_sampling_rejects_invalid_distance(distance):
    s = torch.zeros(1, 2, 13)
    s[..., 6] = 1
    with pytest.raises(ValueError):
        sample_cable_material(s, torch.tensor([0.2, 0.2]), distance)


def test_slip_uses_the_tool_frame_material_point():
    initial = torch.tensor([[0.01, 0, 0], [0, 0, 0.01]])
    current = initial + torch.tensor([[0, 0.003, 0.004], [0, 0, 0]])
    torch.testing.assert_close(
        cable_grasp_slip(current, initial), torch.tensor([0.005, 0])
    )


@pytest.mark.parametrize(
    "lengths",
    [
        torch.tensor([]),
        torch.tensor([-0.2]),
        torch.tensor([0.0]),
        torch.tensor([float("nan")]),
    ],
)
def test_endpoints_reject_invalid_rest_geometry(lengths):
    state = torch.zeros(2, lengths.numel(), 13)
    state[..., 6] = 1
    with pytest.raises(ValueError, match="positive vector"):
        cable_endpoints(state, lengths)


def test_material_sampling_preserves_endpoint_identity_for_unequal_segments():
    state = torch.zeros(1, 2, 13, dtype=torch.float64)
    state[..., 6] = 1
    state[0, :, 2] = torch.tensor([0.05, 0.25])
    lengths = torch.tensor([0.1, 0.3], dtype=torch.float64)
    for distance, expected_z, expected_index in [
        (0.0, 0.0, 0),
        (0.1, 0.1, 0),
        (0.4, 0.4, 1),
    ]:
        point, tangent, index = sample_cable_material(state, lengths, distance)
        assert index == expected_index
        torch.testing.assert_close(point, state.new_tensor([[0, 0, expected_z]]))
        torch.testing.assert_close(tangent, state.new_tensor([[0, 0, 1]]))
