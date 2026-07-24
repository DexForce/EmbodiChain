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

"""Dependency-free unit tests for the optional cuRobo planner surface.

These tests never import the real ``curobo`` package. They cover config
validation, the public export behavior, the matrix -> position/quaternion
conversion, the dynamic-obstacle validator, the actionable error raised when
cuRobo is absent, and the fast-fail when the robot is not on a CUDA device.

The planner's planning behavior (subprocess worker, CUDA graphs, multi-env
worlds) is exercised by ``test_curobo_integration.py`` and
``test_curobo_subprocess.py`` instead.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.planners import CuroboPlannerCfg
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg as CuroboPlannerCfgDirect,
    CuroboWorldCfg,
    _matrix_to_position_quaternion,
    _require_curobo,
    _validate_dynamic_obstacles,
)


def _raise_module_not_found(*args, **kwargs):
    raise ModuleNotFoundError("curobo not installed")


def test_public_config_imports_without_curobo():
    """The planner package must export cuRobo configs without curobo installed."""
    assert CuroboPlannerCfg.__name__ == "CuroboPlannerCfg"
    assert CuroboPlannerCfgDirect is CuroboPlannerCfg
    assert CuroboPlannerCfg().planner_type == "curobo"


def test_matrix_to_position_quaternion_uses_wxyz():
    matrix = torch.eye(4).unsqueeze(0)
    position, quaternion = _matrix_to_position_quaternion(matrix)
    assert torch.equal(position, torch.zeros(1, 3))
    assert torch.equal(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert position.is_contiguous()
    assert quaternion.is_contiguous()


def test_matrix_to_position_quaternion_rejects_non_4x4_batch():
    with pytest.raises(ValueError, match="4, 4"):
        _matrix_to_position_quaternion(torch.zeros(3, 3))


def test_missing_curobo_is_actionable(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _raise_module_not_found)
    with pytest.raises(ImportError, match=r"cu12.*cu13"):
        _require_curobo()


def test_unknown_dynamic_obstacle_is_rejected():
    with pytest.raises(ValueError, match="unknown obstacle"):
        _validate_dynamic_obstacles({"unknown": torch.eye(4)}, ["known"])


def test_dynamic_obstacle_shape_is_validated():
    # (4, 4) is not batched -> rejected; the API requires (B, 4, 4).
    with pytest.raises(ValueError, match="4, 4"):
        _validate_dynamic_obstacles({"known": torch.eye(4)}, ["known"])


def test_curobo_plan_options_carries_context_fields():
    opts = CuroboPlanOptions(
        start_qpos=torch.zeros(2, 7),
        control_part="arm",
        max_attempts=3,
    )
    assert opts.control_part == "arm"
    assert opts.max_attempts == 3
    assert opts.start_qpos.shape == (2, 7)


def test_curobo_planner_cfg_defaults():
    cfg = CuroboPlannerCfg(robot_uid="franka")
    assert cfg.planner_type == "curobo"
    assert cfg.warmup_iterations == 1
    assert cfg.max_attempts == 5
    # CUDA graphs are always on in the subprocess worker (no toggle).
    assert not hasattr(cfg, "use_cuda_graph")
    assert isinstance(cfg.world, CuroboWorldCfg)
    # No external-YAML / profile config; the base-frame override defaults to None.
    assert cfg.sim_base_to_curobo_base is None
    assert not hasattr(cfg, "robot_profiles")
    assert not hasattr(cfg.world, "world_config_path")


def test_curobo_world_cfg_uses_v2_safe_default_collision_cache():
    cache = CuroboWorldCfg().collision_cache

    assert cache == {"cuboid": 8, "mesh": 2}


def test_auto_gen_defaults_keep_sphere_count_low():
    """The voxel sphere estimate must be scaled down so planning stays fast."""
    auto = CuroboPlannerCfg(robot_uid="franka").auto_gen
    assert auto.fit_type == "voxel"
    assert auto.sphere_density == 0.1


def test_curobo_planner_class_is_lazy_import_safe():
    """Referencing the class must not import curobo."""
    import sys

    sys.modules.pop("curobo", None)
    assert CuroboPlanner.__name__ == "CuroboPlanner"
    assert "curobo" not in sys.modules


def test_cpu_device_is_rejected(monkeypatch):
    """A CPU robot fails fast at construction, before any worker is spawned.

    Named without ``cuda`` so the conftest keyword marker does not auto-skip it
    as a GPU test - it exercises the CPU rejection path and needs no CUDA.
    """
    from embodichain.lab.sim.sim_manager import SimulationManager

    cpu_robot = SimpleNamespace(device=torch.device("cpu"))
    fake_sim = SimpleNamespace(get_robot=lambda uid: cpu_robot)
    monkeypatch.setattr(
        SimulationManager, "get_instance", classmethod(lambda cls: fake_sim)
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        CuroboPlanner(CuroboPlannerCfg(robot_uid="cpu_robot"))
