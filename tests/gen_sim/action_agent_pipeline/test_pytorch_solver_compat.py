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

import numpy as np
import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.pytorch_solver_compat import (
    install_action_agent_pytorch_solver_compat,
)
from embodichain.lab.sim.solvers import PytorchSolver

_DUAL_FRANKA_TCP = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _make_solver(get_ik):
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver.tcp_xpos = _DUAL_FRANKA_TCP.copy()
    solver.get_ik = get_ik
    return solver


def test_compat_applies_correct_tcp_inverse_and_restores_tcp() -> None:
    received = {}

    def fake_get_ik(*, target_xpos, **kwargs):
        received["target_xpos"] = target_xpos.clone()
        received["tcp_xpos"] = np.array(solver.tcp_xpos, copy=True)
        return "result"

    solver = _make_solver(fake_get_ik)
    original_tcp = solver.tcp_xpos
    robot = SimpleNamespace(_solvers={"left_arm": solver})
    target_xpos = torch.eye(4).unsqueeze(0)

    assert install_action_agent_pytorch_solver_compat(robot) == 1
    assert solver.get_ik(target_xpos=target_xpos) == "result"

    expected = target_xpos @ torch.linalg.inv(torch.as_tensor(original_tcp))
    torch.testing.assert_close(received["target_xpos"], expected)
    np.testing.assert_array_equal(received["tcp_xpos"], np.eye(4, dtype=np.float32))
    assert solver.tcp_xpos is original_tcp


def test_compat_restores_tcp_when_original_solver_raises() -> None:
    def fail_get_ik(*, target_xpos, **kwargs):
        raise RuntimeError("IK failed")

    solver = _make_solver(fail_get_ik)
    original_tcp = solver.tcp_xpos
    robot = SimpleNamespace(_solvers={"arm": solver})
    install_action_agent_pytorch_solver_compat(robot)

    with pytest.raises(RuntimeError, match="IK failed"):
        solver.get_ik(target_xpos=torch.eye(4))

    assert solver.tcp_xpos is original_tcp


def test_compat_installation_is_idempotent_for_aliased_solver() -> None:
    calls = []

    def fake_get_ik(*, target_xpos, **kwargs):
        calls.append(target_xpos)
        return True

    solver = _make_solver(fake_get_ik)
    robot = SimpleNamespace(_solvers={"left_arm": solver, "arm": solver})

    assert install_action_agent_pytorch_solver_compat(robot) == 1
    assert install_action_agent_pytorch_solver_compat(robot) == 0
    assert solver.get_ik(target_xpos=torch.eye(4)) is True
    assert len(calls) == 1
