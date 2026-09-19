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
"""Manipulability-based multi-seed IK solution selection."""

from __future__ import annotations

import os

import pytest
import torch

from embodichain.compute.kinematics import yoshikawa_manipulability
from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.utils.utility import reset_all_seeds


def _cfg_dict(selection: str) -> dict:
    urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
    assert os.path.isfile(urdf)
    return {
        "uid": f"w1_{selection}",
        "fpath": urdf,
        "control_parts": {
            "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": "PytorchSolver",
                "end_link_name": "left_ee",
                "root_link_name": "left_arm_base",
                "num_samples": 30,
                "ik_solution_selection": selection,
            },
        },
    }


class TestManipulabilitySelection:
    def setup_method(self):
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)

    def teardown_method(self):
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()

    def _add_robot(self, selection: str) -> Robot:
        return self.sim.add_robot(cfg=RobotCfg.from_dict(_cfg_dict(selection)))

    def _targets(self, solver, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        reset_all_seeds(0)
        lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
        span = upper - lower
        # Stay away from the limits so every target is comfortably reachable.
        q_true = lower + span * (0.3 + 0.4 * torch.rand(n, solver.dof))
        with torch.no_grad():
            return solver.get_fk(q_true), q_true

    def _scores(self, solver, qpos: torch.Tensor) -> torch.Tensor:
        return yoshikawa_manipulability(solver.get_jacobian(qpos))

    def test_invalid_mode_rejected(self):
        self._add_robot("bogus")
        with pytest.raises(ValueError, match="ik_solution_selection"):
            self.sim.prepare()

    def test_solutions_reach_target_and_outrank_nearest(self):
        robots = {mode: self._add_robot(mode) for mode in ("nearest", "manipulability")}
        self.sim.prepare()
        targets = None
        results = {}
        for mode in ("nearest", "manipulability"):
            solver = robots[mode].get_solver("left_arm")
            if targets is None:
                targets, seed_q = self._targets(solver, n=8)
            reset_all_seeds(1)  # identical multi-seed draws for both modes
            success, qpos = solver.get_ik(targets, qpos_seed=seed_q[0])
            assert bool(success.all()), f"{mode} selection lost IK success"
            qpos = qpos[:, 0, :]
            # The selected solution must still reach the target pose.
            fk = solver.get_fk(qpos)
            assert torch.allclose(fk[:, :3, 3], targets[:, :3, 3], atol=5e-3)
            results[mode] = self._scores(solver, qpos)

        # Same candidate pool (same RNG draws): picking by manipulability can
        # never do worse than picking the nearest-to-seed candidate.
        assert torch.all(results["manipulability"] >= results["nearest"] - 1e-9)
        # And on a 7-DOF arm it must find a strictly better posture somewhere.
        assert bool((results["manipulability"] > results["nearest"] + 1e-9).any())

    def test_deterministic_selection(self):
        robot = self._add_robot("manipulability")
        self.sim.prepare()
        solver = robot.get_solver("left_arm")
        targets, seed_q = self._targets(solver, n=4)
        reset_all_seeds(2)
        _, first = solver.get_ik(targets, qpos_seed=seed_q[0])
        reset_all_seeds(2)
        _, second = solver.get_ik(targets, qpos_seed=seed_q[0])
        assert torch.equal(first, second)
