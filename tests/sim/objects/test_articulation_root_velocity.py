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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.objects.articulation import Articulation


def _articulation() -> Articulation:
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation._entities = [object()] * 3
    articulation._data = SimpleNamespace(
        articulation_view=SimpleNamespace(apply_root_velocity=Mock())
    )
    return articulation


def test_default_selection_writes_all_root_velocities() -> None:
    articulation = _articulation()
    values = torch.arange(18, dtype=torch.float64).reshape(3, 6)
    articulation.set_root_velocity(values)
    actual, ids = (
        articulation._data.articulation_view.apply_root_velocity.call_args.args
    )
    torch.testing.assert_close(actual, values.float())
    assert ids.tolist() == [0, 1, 2]


def test_root_velocity_preserves_requested_row_order() -> None:
    articulation = _articulation()
    values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    articulation.set_root_velocity(values, env_ids=[2, 0])
    actual, ids = (
        articulation._data.articulation_view.apply_root_velocity.call_args.args
    )
    torch.testing.assert_close(actual, values)
    assert ids.tolist() == [2, 0]


@pytest.mark.parametrize("shape", [(2, 6), (3, 5)])
def test_invalid_root_velocity_shape_does_not_write(shape: tuple[int, int]) -> None:
    articulation = _articulation()
    with pytest.raises(ValueError, match="Root velocity must have shape"):
        articulation.set_root_velocity(torch.zeros(shape))
    articulation._data.articulation_view.apply_root_velocity.assert_not_called()
