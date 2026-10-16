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

pytestmark = pytest.mark.no_sim


def test_combined_state_clamps_both_positions_before_one_write() -> None:
    art = object.__new__(Articulation)
    art.device = torch.device("cpu")
    art._entities = [object()] * 3
    art.__dict__["dof"] = 2
    art._data = SimpleNamespace(
        qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]] * 3),
        articulation_view=SimpleNamespace(apply_state=Mock()),
    )
    art._stabilize_newton_mimic_target_write = Mock()
    art.set_state(
        env_ids=[2, 0],
        joint_ids=[1, 0],
        qpos=torch.tensor([[5.0, -3.0], [-5.0, 3.0]]),
        target_qpos=torch.tensor([[-4.0, 4.0], [4.0, -4.0]]),
        clear_dynamics=True,
    )
    call = art._data.articulation_view.apply_state
    call.assert_called_once()
    rows, columns = call.call_args.args
    assert torch.as_tensor(rows).tolist() == [2, 0]
    assert torch.as_tensor(columns).tolist() == [1, 0]
    torch.testing.assert_close(
        call.call_args.kwargs["qpos"], torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    )
    torch.testing.assert_close(
        call.call_args.kwargs["target_qpos"], torch.tensor([[-2.0, 1.0], [2.0, -1.0]])
    )
    call.reset_mock()
    with pytest.raises(ValueError, match="root_velocity"):
        art.set_state(
            qpos=torch.zeros((3, 2)),
            root_velocity=torch.zeros((3, 5)),
            clear_dynamics=True,
        )
    call.assert_not_called()
