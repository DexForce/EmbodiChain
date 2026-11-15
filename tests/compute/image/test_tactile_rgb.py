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

import pytest
import torch

from embodichain.compute.image._tactile_rgb import _render_tactile_rgb


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_planar_rgb_response_preserves_background_and_bounds(device: str) -> None:
    depth = torch.tensor(
        [[[0.0, 0.02], [0.03, 0.05]], [[10.0, 10.0], [10.0, 10.0]]],
        dtype=torch.float32,
        device=device,
    )
    hit = torch.ones_like(depth, dtype=torch.bool)
    hit[1] = False
    background = torch.full((2, 2, 3), 0.5, device=device)
    background[0, 0, 2] = 0.25
    rgb = torch.empty((2, 2, 2, 3), device=device)
    _render_tactile_rgb(
        depth,
        hit,
        background,
        spacing=(0.01, 0.03),
        x_response=(0.25, 0.0, 0.0),
        y_response=(0.0, -1.0, 0.25),
        rgb=rgb,
    )
    expected = (
        torch.tensor([1.0, 0.0, 0.75], device=device).expand_as(background).clone()
    )
    expected[0, 0, 2] = 0.5
    torch.testing.assert_close(rgb[0], expected)
    torch.testing.assert_close(rgb[1], background)


def test_rgb_ignores_stale_depth_in_missed_neighbor() -> None:
    depth = torch.tensor([[[0.01, 10.0]]])
    hit = torch.tensor([[[True, False]]])
    background = torch.full((1, 2, 3), 0.5)
    rgb = torch.empty((1, 1, 2, 3))
    _render_tactile_rgb(
        depth,
        hit,
        background,
        spacing=(0.02, 0.01),
        x_response=(0.5, 0.0, 0.0),
        y_response=(1.0, 1.0, 1.0),
        rgb=rgb,
    )
    torch.testing.assert_close(rgb[0, 0, 0], torch.tensor([0.25, 0.5, 0.5]))
    torch.testing.assert_close(rgb[0, 0, 1], background[0, 1])
