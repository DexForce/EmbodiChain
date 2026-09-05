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

import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.solvers import DifferentialSolverCfg, OPWSolverCfg, URSolverCfg
from embodichain.lab.sim.solvers.opw_solver import OPWSolver
from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolver

UR5_DH_PARAMETERS = {
    "d1": 0.089159,
    "a2": -0.425,
    "a3": -0.39225,
    "d4": 0.10915,
    "d5": 0.09465,
    "d6": 0.0823,
}


def test_continuous_batch_ik_is_an_optional_solver_capability() -> None:
    # Capability discovery must not initialize kinematics or simulator resources.
    ordinary = object.__new__(PytorchSolver)
    opw = object.__new__(OPWSolver)
    assert not ordinary.supports_continuous_batch_ik
    assert opw.supports_continuous_batch_ik
    with pytest.raises(NotImplementedError, match="continuous batch IK"):
        ordinary._select_continuous_ik_path(
            torch.empty(1, 2, 1, 6),
            torch.empty(1, 2, 1, dtype=torch.bool),
            torch.zeros(1, 6),
        )


def assert_ur5_dh_parameters(cfg: URSolverCfg) -> None:
    """Assert that a UR solver config contains the UR5 DH parameters."""
    for field_name, expected_value in UR5_DH_PARAMETERS.items():
        assert getattr(cfg, field_name) == pytest.approx(expected_value)


def make_ur5_robot_dict() -> dict:
    """Return a minimal robot dictionary with a nested UR5 solver config."""
    return {
        "control_parts": {"arm": [f"joint_{index}" for index in range(6)]},
        "solver_cfg": {
            "arm": {
                "class_type": "URSolver",
                "ur_type": "ur5",
                "root_link_name": "base",
                "end_link_name": "tool0",
            }
        },
    }


def test_solver_cfg_from_dict_constructs_ur5_with_derived_dh_parameters():
    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": "ur5",
        }
    )

    assert isinstance(cfg, URSolverCfg)
    assert cfg.ur_type == "ur5"
    assert_ur5_dh_parameters(cfg)


def test_solver_cfg_from_dict_runs_concrete_post_init_once(monkeypatch):
    post_init_calls = 0
    original_post_init = URSolverCfg.__post_init__

    def counted_post_init(cfg: URSolverCfg) -> None:
        nonlocal post_init_calls
        post_init_calls += 1
        original_post_init(cfg)

    monkeypatch.setattr(URSolverCfg, "__post_init__", counted_post_init)

    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": "ur5",
        }
    )

    assert post_init_calls == 1
    assert_ur5_dh_parameters(cfg)


def test_robot_cfg_from_dict_constructs_nested_ur5_solver():
    cfg = RobotCfg.from_dict(make_ur5_robot_dict())

    solver_cfg = cfg.solver_cfg["arm"]
    assert isinstance(solver_cfg, URSolverCfg)
    assert solver_cfg.root_link_name == "base"
    assert solver_cfg.end_link_name == "tool0"
    assert_ur5_dh_parameters(solver_cfg)


def test_robot_cfg_solver_to_dict_from_dict_roundtrip_preserves_derived_values():
    cfg = RobotCfg.from_dict(make_ur5_robot_dict())

    restored_cfg = RobotCfg.from_dict(cfg.to_dict())

    restored_solver_cfg = restored_cfg.solver_cfg["arm"]
    assert isinstance(restored_solver_cfg, URSolverCfg)
    assert restored_solver_cfg.ur_type == "ur5"
    assert restored_solver_cfg.root_link_name == "base"
    assert restored_solver_cfg.end_link_name == "tool0"
    np.testing.assert_allclose(restored_solver_cfg.tcp, np.eye(4))
    assert_ur5_dh_parameters(restored_solver_cfg)


def test_solver_cfg_from_dict_applies_other_derived_config_logic():
    cfg = DifferentialSolverCfg.from_dict(
        {
            "class_type": "DifferentialSolver",
            "ik_method": "dls",
        }
    )

    assert isinstance(cfg, DifferentialSolverCfg)
    assert cfg.ik_method == "dls"
    assert cfg.ik_params == {"lambda_val": 0.01}


def test_solver_cfg_from_dict_preserves_unannotated_config_attributes():
    cfg = OPWSolverCfg.from_dict(
        {
            "class_type": "OPWSolver",
            "a1": 1.25,
        }
    )

    assert isinstance(cfg, OPWSolverCfg)
    assert cfg.a1 == pytest.approx(1.25)


def test_solver_cfg_from_dict_ignores_unknown_fields(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "embodichain.lab.sim.solvers.base_solver.logger.log_warning",
        warnings.append,
    )

    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": "ur5",
            "unsupported_field": "ignored",
        }
    )

    assert not hasattr(cfg, "unsupported_field")
    assert warnings == ["Key 'unsupported_field' not found in URSolverCfg."]
