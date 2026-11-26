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

"""Capability checks shared by borrowed observations, attachment and writes."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.dexsim_session.robot_state import _require_joint_solver


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"solver_type": "mujoco_warp"},
        {"solver_type": "dexuni", "features": {"mujoco_solve_enabled": True}},
    ],
)
def test_supported_joint_solver(diagnostics: dict) -> None:
    _require_joint_solver(diagnostics)


@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {"solver_type": "auto"},
        {"solver_type": "dexuni"},
        {"solver_type": "dexuni", "features": {"mujoco_solve_enabled": False}},
    ],
)
def test_unsupported_joint_solver(diagnostics: dict) -> None:
    with pytest.raises(ValueError):
        _require_joint_solver(diagnostics)
