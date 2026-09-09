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
"""Default IK seed behaviour: joint-range midpoint instead of zeros.

Franka FR3 is the regression robot on purpose: its joints 4 and 6 exclude the
zero configuration, so any code path that silently falls back to a zero seed
is caught here.
"""

from __future__ import annotations

import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers import (
    DifferentialSolverCfg,
    PytorchSolverCfg,
)
from embodichain.utils.utility import reset_all_seeds

_FRANKA_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def _franka_urdf() -> str:
    return get_data_path("Franka/Panda/PandaWithHand.urdf")


def _make_solver(cfg_cls, **kwargs):
    cfg = cfg_cls(
        urdf_path=_franka_urdf(),
        end_link_name="fr3_hand_tcp",
        root_link_name="base",
        joint_names=_FRANKA_JOINTS,
        ik_nearest_weight=[1.0] * len(_FRANKA_JOINTS),
        **kwargs,
    )
    return cfg.init_solver(device=torch.device("cpu"))


class TestDefaultQposSeed:
    def test_zero_configuration_is_infeasible_premise(self):
        """The regression robot must actually exclude the zero configuration."""
        solver = _make_solver(PytorchSolverCfg)
        zero = torch.zeros(solver.dof)
        within = (zero >= solver.lower_qpos_limits) & (zero <= solver.upper_qpos_limits)
        assert not bool(within.all()), "FR3 zero pose unexpectedly feasible"

    def test_helper_returns_midpoint_within_limits(self):
        solver = _make_solver(PytorchSolverCfg)
        seed = solver.get_default_qpos_seed()
        lo, hi = solver.lower_qpos_limits, solver.upper_qpos_limits
        assert torch.allclose(seed, (lo + hi) / 2)
        assert bool(((seed > lo) & (seed < hi)).all())

    def test_pytorch_none_seed_equals_explicit_midpoint(self):
        """The None path must behave identically to passing the midpoint."""
        solver = _make_solver(PytorchSolverCfg, num_samples=1)
        lo, hi = solver.lower_qpos_limits, solver.upper_qpos_limits
        reset_all_seeds(0)
        q_true = lo + torch.rand(8, solver.dof) * (hi - lo)
        with torch.no_grad():
            target = solver.get_fk(q_true)

        # num_samples=1 keeps only the seed slot, so the solve is
        # deterministic and the two paths must match exactly.
        ok_none, q_none = solver.get_ik(target.clone(), num_samples=1)
        mid = solver.get_default_qpos_seed()
        ok_mid, q_mid = solver.get_ik(
            target.clone(), qpos_seed=mid.clone(), num_samples=1
        )
        assert torch.equal(ok_none, ok_mid)
        assert torch.allclose(q_none, q_mid)

    def test_differential_none_seed_is_feasible_start(self):
        solver = _make_solver(DifferentialSolverCfg)
        lo, hi = solver.lower_qpos_limits, solver.upper_qpos_limits
        reset_all_seeds(0)
        q_true = lo + torch.rand(4, solver.dof) * (hi - lo)
        with torch.no_grad():
            target = solver.get_fk(q_true)
        ok, qpos = solver.get_ik(target.clone())
        assert qpos.shape[-1] == solver.dof
        assert torch.isfinite(qpos).all()
