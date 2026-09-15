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

from embodichain.gen_sim.task_engine._task_program.collision_validation import (
    validate_configurations,
)


@pytest.mark.parametrize("failing_cost", [0, 1, 2])
def test_all_collision_constraints_are_required(failing_cost: int) -> None:
    q = torch.zeros(1, 2, 7)
    state = SimpleNamespace(robot_spheres=torch.zeros(1, 2, 3, 4))
    costs = [torch.zeros(1, 2) for _ in range(3)]
    costs[failing_cost][0, 1] = float("nan") if failing_cost == 2 else 1.0
    scene = Mock()
    scene.forward.return_value = costs[2]
    checker = SimpleNamespace(
        self_collision_cost=object(),
        collision_constraint=scene,
        get_kinematics=Mock(return_value=state),
        setup_batch_tensors=Mock(),
        get_bound=Mock(return_value=costs[0]),
        get_self_collision=Mock(return_value=costs[1]),
    )
    assert validate_configurations(checker, q).tolist() == [[True, False]]
    scene.update_num_spheres.assert_called_once_with(3, batch_size=1, horizon=2)
    scene.forward.assert_called_once_with(state, idxs_env_query=None)
    assert not q.any()


def test_missing_collision_constraint_rejects() -> None:
    checker = SimpleNamespace(self_collision_cost=None, collision_constraint=None)
    with pytest.raises(ValueError, match="both self and scene"):
        validate_configurations(checker, torch.zeros(1, 2, 7))
