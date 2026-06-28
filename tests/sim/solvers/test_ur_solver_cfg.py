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

from embodichain.lab.sim.solvers import URSolverCfg

UR5_EXPECTED_DH = {
    "d1": 0.089159,
    "d4": 0.10915,
    "d5": 0.09465,
    "d6": 0.0823,
    "a2": -0.425,
    "a3": -0.39225,
}
DUAL_UR5_TCP = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.16],
    [0.0, 0.0, 0.0, 1.0],
]


def test_ur_solver_cfg_from_dict_applies_ur5_variant_defaults() -> None:
    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": "ur5",
            "root_link_name": "left_base_link",
            "end_link_name": "left_ee_link",
            "tcp": DUAL_UR5_TCP,
        }
    )

    assert cfg.class_type == "URSolver"
    assert cfg.ur_type == "ur5"
    assert cfg.root_link_name == "left_base_link"
    assert cfg.end_link_name == "left_ee_link"
    assert np.asarray(cfg.tcp).tolist() == DUAL_UR5_TCP
    for param_name, expected_value in UR5_EXPECTED_DH.items():
        assert getattr(cfg, param_name) == pytest.approx(expected_value)
    assert cfg.urdf_path.endswith("UniversalRobots/UR5/UR5.urdf")


def test_ur_solver_cfg_from_dict_preserves_explicit_none_urdf_path() -> None:
    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": "ur5",
            "urdf_path": None,
            "root_link_name": "right_base_link",
            "end_link_name": "right_ee_link",
            "tcp": DUAL_UR5_TCP,
        }
    )

    assert cfg.ur_type == "ur5"
    assert cfg.urdf_path is None
    assert cfg.root_link_name == "right_base_link"
    assert cfg.end_link_name == "right_ee_link"
    for param_name, expected_value in UR5_EXPECTED_DH.items():
        assert getattr(cfg, param_name) == pytest.approx(expected_value)
