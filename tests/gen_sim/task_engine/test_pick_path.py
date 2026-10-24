# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

from embodichain.gen_sim.task_engine._task_program.pick_path import _compose_approach


@pytest.mark.parametrize("initial_hand", [0.0, 0.7, 0.700128])
def test_rebuilt_approach_preserves_opening_before_close_and_passive_joints(
    initial_hand: float,
) -> None:
    original = torch.zeros(1, 8, 4)
    original[:, :, 3] = 0.37
    original[:, :4, 2] = torch.linspace(initial_hand, 0.0, 4)
    original[:, 4:7, 2] = torch.linspace(0.0, 0.7, 3)
    original[:, 7:, 2] = 0.7
    original[:, :4, :2] = torch.linspace(0.0, 0.4, 4)[None, :, None]
    before = original.clone()
    prefix = torch.linspace(0.0, 0.2, 4)[None, :, None].expand(1, 4, 2).clone()
    contact = torch.linspace(0.2, 0.4, 3)[None, :, None].expand(1, 3, 2).clone()
    bounds = torch.tensor([[[0.0, 0.7]]])
    result = _compose_approach(original, 4, [0, 1], prefix, contact, [2], bounds)
    approach_count = len(prefix[0]) + len(contact[0]) - 1
    assert torch.equal(result[:, :4, :2], prefix)
    assert torch.equal(result[:, 4:approach_count, :2], contact[:, 1:])
    assert torch.equal(result[:, approach_count:], original[:, 4:])
    assert torch.equal(result[:, :, 3], torch.full_like(result[:, :, 3], 0.37))
    assert result[0, 0, 2] == min(initial_hand, 0.7)
    assert result[0, 0, 0] == original[0, 0, 0]
    assert result[0, approach_count - 1, 2] == result[0, approach_count, 2] == 0.0
    assert (result[0, 1:approach_count, 2] <= result[0, : approach_count - 1, 2]).all()
    assert torch.equal(original, before)
