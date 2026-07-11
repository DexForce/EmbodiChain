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
validation, the public export behavior, the named-joint reorder helper, the
matrix -> position/quaternion conversion, the dynamic-obstacle validator, and
the actionable error raised when cuRobo is absent.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from embodichain.lab.sim.planners import CuroboPlannerCfg
from embodichain.lab.sim.planners.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg as CuroboPlannerCfgDirect,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
    _matrix_to_position_quaternion,
    _require_curobo,
    _reorder_by_names,
    _validate_dynamic_obstacles,
)


def _raise_module_not_found(*args, **kwargs):
    raise ModuleNotFoundError("curobo not installed")


def test_public_config_imports_without_curobo():
    """The planner package must export cuRobo configs without curobo installed."""
    assert CuroboPlannerCfg.__name__ == "CuroboPlannerCfg"
    assert CuroboPlannerCfgDirect is CuroboPlannerCfg
    assert CuroboPlannerCfg().planner_type == "curobo"


def test_reorder_by_names_preserves_batch_and_time_dimensions():
    values = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (1, 2, 2)
    result = _reorder_by_names(values, ["joint_b", "joint_a"], ["joint_a", "joint_b"])
    assert torch.equal(result, torch.tensor([[[20.0, 10.0], [40.0, 30.0]]]))


def test_reorder_by_names_rejects_mismatched_name_sets():
    values = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="name"):
        _reorder_by_names(values, ["joint_a", "joint_b"], ["joint_a", "joint_c"])


def test_matrix_to_position_quaternion_uses_wxyz():
    matrix = torch.eye(4).unsqueeze(0)
    position, quaternion = _matrix_to_position_quaternion(matrix)
    assert torch.equal(position, torch.zeros(1, 3))
    assert torch.equal(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


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
    assert cfg.warmup is True
    assert cfg.max_attempts == 5
    assert cfg.use_cuda_graph is True
    assert isinstance(cfg.world, CuroboWorldCfg)


def test_curobo_robot_profile_cfg_requires_joint_map():
    cfg = CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names={"a": "b"},
    )
    assert cfg.robot_config_path == "franka.yml"
    assert cfg.sim_to_curobo_joint_names == {"a": "b"}
    assert cfg.fixed_joint_positions == {}


def test_curobo_planner_class_is_lazy_import_safe():
    """Referencing the class must not import curobo."""
    import sys

    sys.modules.pop("curobo", None)
    assert CuroboPlanner.__name__ == "CuroboPlanner"
    assert "curobo" not in sys.modules
