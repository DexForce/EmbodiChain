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
    [
        "embodichain.lab.sim.objects",
        "embodichain.lab.sim.motion.planners",
        "embodichain.lab.sim.motion.motion_generator",
    ],
)
def test_motion_import_order_and_config_resolution(first_module: str) -> None:
    script = """
import importlib
import importlib.abc
import sys

class BlockExternalToppra(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "toppra" or fullname.startswith("toppra."):
            raise AssertionError("External toppra must not be imported")
sys.meta_path.insert(0, BlockExternalToppra())
import dexsim

importlib.import_module(sys.argv[1])
generator_path = "embodichain.lab.sim.motion.motion_generator"
if sys.argv[1] != generator_path:
    assert generator_path not in sys.modules
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenerator, MotionGenCfg, MotionGenOptions,
)
from embodichain.lab.sim.motion import planners
assert not hasattr(planners, "MotionGenerator")
assert MotionGenerator.__module__ == generator_path
assert MotionGenCfg.__module__ == generator_path
assert MotionGenOptions().strategy == "motion_gen"
from embodichain.lab.sim import motion
from embodichain.lab.sim.motion import expansion
from embodichain.lab.sim.motion.execution import (
    EpisodeSink,
    FixedSceneInitialStatePort,
    InitialStatePort,
    MeasuredExecutor,
    SingleSlotOutcome,
    SingleSlotRunner,
)
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
for name in (
    "InitialStatePort",
    "MeasuredExecutor",
    "EpisodeSink",
    "FixedSceneInitialStatePort",
    "SingleSlotOutcome",
    "SingleSlotRunner",
):
    assert not hasattr(expansion, name)
assert InitialStatePort.__module__ == "embodichain.lab.sim.motion.execution"
assert MeasuredExecutor.__module__ == "embodichain.lab.sim.motion.execution"
assert EpisodeSink.__module__ == "embodichain.lab.sim.motion.execution"
assert FixedSceneInitialStatePort.__module__ == "embodichain.lab.sim.motion.execution"
assert SingleSlotOutcome.__module__ == "embodichain.lab.sim.motion.execution"
assert SingleSlotRunner.__module__ == "embodichain.lab.sim.motion.execution"
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


def test_toppra_planning_and_backward_work_without_external_package() -> None:
    script = """
import importlib.abc
import sys
class BlockExternalToppra(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "toppra" or fullname.startswith("toppra."):
            raise AssertionError("External toppra must not be imported")
sys.meta_path.insert(0, BlockExternalToppra())
import numpy as np
import torch
from embodichain.compute.trajectory._toppra import _NumpyToppra
from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp
points = torch.tensor([[[0.0], [0.73]]], dtype=torch.float64, requires_grad=True)
acceleration = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
reference = _NumpyToppra(points.detach().numpy()[0], 1.0, 1.7)
result = _retime_toppra_warp(points, 1.0, acceleration, sample_dt=0.13)
assert result["success"].all()
assert np.isclose(result["dt"].sum().item(), reference.duration)
result["dt"].sum().backward()
assert torch.isfinite(points.grad).all() and torch.isfinite(acceleration.grad)
assert "toppra" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
