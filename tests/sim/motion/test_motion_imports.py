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

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "first_module",
    ["embodichain.lab.sim.objects", "embodichain.lab.sim.motion.planners"],
)
def test_motion_import_order_and_config_resolution(first_module: str) -> None:
    script = """
import importlib
import sys
import dexsim

importlib.import_module(sys.argv[1])
from embodichain.lab.sim import motion
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.motion.solvers import SolverCfg, URSolverCfg
from embodichain.lab.sim.motion.workspace import RobotWorkspaceCfg

solver = SolverCfg.from_dict({"class_type": "URSolver", "robot_type": "ur5"})
robot = RobotCfg.from_dict({
    "solver_cfg": {"arm": {"class_type": "URSolver", "robot_type": "ur5"}},
    "workspace_cfg": {"arm": {"cache_path": "workspace.h5"}},
})
assert isinstance(solver, URSolverCfg)
assert isinstance(robot.solver_cfg["arm"], URSolverCfg)
assert isinstance(robot.workspace_cfg["arm"], RobotWorkspaceCfg)
for name in motion.__all__:
    module = getattr(motion, name)
    assert module is importlib.import_module("embodichain.lab.sim.motion." + name)
assert "solvers" in dir(motion)
try:
    motion.unknown_motion_module
except AttributeError:
    pass
else:
    raise AssertionError("unknown motion attribute must raise AttributeError")
assert dexsim.get_world_num() == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
