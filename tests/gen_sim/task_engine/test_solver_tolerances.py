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

import pytest

from embodichain.gen_sim.task_engine._task_program.pick_candidate_path import (
    _tolerances as pick_tolerances,
)
from embodichain.gen_sim.task_engine._task_program.place_branch import (
    _tolerances as place_tolerances,
)
from embodichain.lab.sim.motion.solvers import URSolver


@pytest.mark.parametrize("consumer", [pick_tolerances, place_tolerances])
def test_analytic_ur_pose_validation_does_not_require_iteration_api(consumer) -> None:
    solver = object.__new__(URSolver)
    assert not hasattr(solver, "get_iteration_params")
    robot = SimpleNamespace(get_solver=lambda _part: solver)
    assert consumer(robot, "arm") == (5e-4, 5e-4)


@pytest.mark.parametrize("consumer", [pick_tolerances, place_tolerances])
def test_iterative_pose_tolerances_are_not_changed(consumer) -> None:
    parameters = {"pos_eps": 0.003, "rot_eps": 0.0002}
    solver = SimpleNamespace(get_iteration_params=lambda: parameters)
    robot = SimpleNamespace(get_solver=lambda _part: solver)
    assert consumer(robot, "arm") == (0.003, 0.0002)


@pytest.mark.parametrize("consumer", [pick_tolerances, place_tolerances])
def test_unknown_solver_does_not_receive_implicit_tolerances(consumer) -> None:
    robot = SimpleNamespace(get_solver=lambda _part: object())
    with pytest.raises(ValueError, match="pose tolerances"):
        consumer(robot, "arm")
