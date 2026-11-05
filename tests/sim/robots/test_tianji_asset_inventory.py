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

"""Concrete Marvin declares its effective model files without native construction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_sim


@pytest.fixture(autouse=True)
def forbid_runtime_creation(monkeypatch):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("model inventory constructed a native runtime or solver")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)


def test_standard_marvin_declares_the_actual_same_source_kinematic_inputs(tmp_path):
    from embodichain.data import get_data_path
    from embodichain.data.asset_dependencies import asset_dependency_files
    from embodichain.lab.gym.utils.gym_utils import robot_config_from_dict
    from embodichain.lab.task_program.integrations import DeclaredAssetFileProvider

    declaration = {"class_type": "TianjiMarvin", "with_gripper": True}
    before = deepcopy(declaration)
    cfg = robot_config_from_dict(declaration)
    snapshot = cfg.to_dict()
    assert isinstance(cfg, DeclaredAssetFileProvider)
    source = Path(get_data_path("TianjiMarvin/robot_with_ee_acd.urdf")).resolve()
    assert Path(cfg.fpath).resolve() == source
    files = cfg.declared_asset_files()
    assert files == asset_dependency_files(source)
    assert source in files and len(files) > 1
    assert declaration == before and cfg.to_dict() == snapshot


def test_marvin_inventory_honors_effective_source_and_solver_overrides(tmp_path):
    from embodichain.lab.gym.utils.gym_utils import robot_config_from_dict

    mesh = tmp_path / "mesh.obj"
    mesh.write_text("v 0 0 0\n")
    source = tmp_path / "selected.urdf"
    source.write_text(
        '<robot><link name="base"><collision><geometry><mesh filename="mesh.obj"/></geometry></collision></link></robot>'
    )
    solver = tmp_path / "solver.urdf"
    solver.write_text('<robot><link name="base"/></robot>')
    cfg = robot_config_from_dict(
        {"class_type": "TianjiMarvin", "with_gripper": True, "fpath": str(source)}
    )
    cfg.solver_cfg["left_arm"].urdf_path = str(solver)
    snapshot = cfg.to_dict()
    assert cfg.declared_asset_files() == tuple(sorted([source, mesh, solver]))
    assert cfg.to_dict() == snapshot
    mesh.unlink()
    with pytest.raises(FileNotFoundError):
        cfg.declared_asset_files()
