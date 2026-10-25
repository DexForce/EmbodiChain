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

"""Explicit SSIK selection retains the catalog X5A tool convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain_robots import ARX5Cfg

pytestmark = pytest.mark.no_sim


def test_arx5_explicit_ssik_override_preserves_chain_source_and_tcp() -> None:
    source = Path(__file__).parents[2] / "fixtures/kinematics/arx5_x5a.urdf"
    default = ARX5Cfg.from_dict({"fpath": str(source)})
    original = default.solver_cfg["arm"]
    cfg = ARX5Cfg.from_dict(
        {
            "fpath": str(source),
            "solver_cfg": {
                "arm": SSIKSolverCfg(
                    backend="reference",
                    root_link_name=original.root_link_name,
                    end_link_name=original.end_link_name,
                    tcp=original.tcp,
                    reference_artifact="prepared_x5a.py",
                )
            },
        }
    )
    assert isinstance(cfg.solver_cfg["arm"], SSIKSolverCfg)
    assert cfg.solver_cfg["arm"].reference_artifact == "prepared_x5a.py"
    assert len(cfg.build_pk_serial_chain()["arm"].get_joint_parameter_names()) == 6
    restored = ARX5Cfg.from_dict(cfg.to_dict())
    assert restored.solver_cfg["arm"].backend == "reference"
    assert restored.fpath == str(source)
    np.testing.assert_array_equal(
        restored.solver_cfg["arm"].tcp,
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0.085], [0, 0, 0, 1]],
    )
