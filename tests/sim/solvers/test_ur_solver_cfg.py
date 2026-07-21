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

_EXPECTED_DH = {
    "ur3": {
        "d1": 0.1519,
        "d4": 0.11235,
        "d5": 0.08535,
        "d6": 0.0819,
        "a2": -0.24365,
        "a3": -0.21325,
    },
    "ur5": {
        "d1": 0.089159,
        "d4": 0.10915,
        "d5": 0.09465,
        "d6": 0.0823,
        "a2": -0.425,
        "a3": -0.39225,
    },
    "ur10": {
        "d1": 0.1273,
        "d4": 0.163941,
        "d5": 0.1157,
        "d6": 0.0922,
        "a2": -0.612,
        "a3": -0.5723,
    },
}
_TCP = np.eye(4).tolist()


@pytest.mark.parametrize("ur_type", ["ur3", "ur5", "ur10"])
def test_ur_solver_cfg_from_dict_applies_variant_defaults(ur_type: str) -> None:
    cfg = URSolverCfg.from_dict(
        {
            "class_type": "URSolver",
            "ur_type": ur_type,
            "root_link_name": "left_base_link",
            "end_link_name": "left_ee_link",
            "tcp": _TCP,
        }
    )

    assert cfg.ur_type == ur_type
    for parameter, expected in _EXPECTED_DH[ur_type].items():
        assert getattr(cfg, parameter) == pytest.approx(expected)


def test_ur_solver_cfg_from_dict_preserves_explicit_fields() -> None:
    cfg = URSolverCfg.from_dict(
        {
            "ur_type": "ur5",
            "urdf_path": None,
            "d1": 1.25,
        }
    )

    assert cfg.urdf_path is None
    assert cfg.d1 == pytest.approx(1.25)
    assert cfg.a2 == pytest.approx(_EXPECTED_DH["ur5"]["a2"])
