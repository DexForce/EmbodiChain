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

import math

import numpy as np
import pytest
import torch

from embodichain.lab.sim.planners.toppra_planner import _toppra_solve_one_env
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod


# Note: TOPPRA's ParametrizeConstAccel robustly avoids returning None for smooth
# splines (verified empirically across tiny limits, huge displacements, zigzags,
# and plateaus), so the ``jnt_traj is None`` branch in ``_toppra_solve_one_env``
# is defensive and not directly unit-tested here; infeasible inputs instead
# raise during path/constraint construction and are caught by the
# ``except Exception`` -> ``_empty_failure`` path.
class TestToppraWorker:
    def test_solve_one_env_quantity(self):
        # 2-waypoint, 6-DOF
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["positions"].shape == (20, 6)
        assert out["velocities"].shape == (20, 6)
        assert out["dt"].shape == (20,)

    def test_solve_one_env_infeasible_exception(self):
        # Single waypoint -> SplineInterpolator raises -> caught, returns failure
        wp = np.array([[0.0] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=10,
        )
        assert out["success"] is False

    def test_solve_one_env_time_sampling(self):
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=0.05,
        )
        assert out["success"] is True
        assert out["positions"].shape[0] == out["n"]
        assert out["n"] >= 2

    def test_solve_one_env_same_waypoint_shortcut(self):
        wp = np.array([[0.3] * 6, [0.3] * 6])  # identical
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["n"] == 2
        assert out["duration"] == 0.0


class TestToppraCfgFields:
    def test_cfg_defaults(self):
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlannerCfg

        cfg = ToppraPlannerCfg(robot_uid="x")
        assert cfg.max_workers is None
        assert cfg.mp_context == "fork"


class TestToppraPlanBatched:
    def _make_planner(self):
        from embodichain.lab.sim.planners.toppra_planner import (
            ToppraPlanner,
            ToppraPlannerCfg,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=3)
        )
        robot = sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "t", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="t", max_workers=1))
        return planner, sim

    def test_plan_batched_quantity_uniform_N(self):
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions

        planner, sim = self._make_planner()
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.0, 0.4, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=15,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.success.all().item()
            assert r.positions.shape == (B, 15, dofs)
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()

    def test_plan_batched_time_tailpads(self):
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions

        planner, sim = self._make_planner()
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.tensor([0.1, 0.4, 0.9])  # different durations
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.TIME,
                sample_interval=0.05,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.positions.shape[0] == B
            assert r.duration.shape == (B,)

            # The env with the SHORTEST duration got tail-padded; its trailing
            # padded rows must equal its last real waypoint (held pose) and the
            # padded tail must be constant.
            shortest_env = int(r.duration.argmin().item())
            longest_env = int(r.duration.argmax().item())
            tail = r.positions[shortest_env]  # (max_n, DOF)
            max_n = tail.shape[0]
            # Reconstruct n_real for the shortest env the same way the worker does.
            n_real = max(2, int(math.ceil(r.duration[shortest_env].item() / 0.05)) + 1)
            # The longest env should not be padded (its n_real == max_n).
            assert r.positions[longest_env].shape[0] == max_n
            # Shortest env must actually have been padded (otherwise the test
            # isn't exercising the tail-pad branch).
            assert n_real < max_n, "expected shortest env to be tail-padded"
            held = tail[n_real - 1]  # last real waypoint
            padded = tail[n_real:]  # all padded rows
            assert torch.allclose(padded, held.expand(padded.shape))
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()

    def test_plan_batched_pool_path(self):
        # Exercise the real ProcessPoolExecutor branch (max_workers=2, B=3).
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim.planners.toppra_planner import (
            ToppraPlanner,
            ToppraPlannerCfg,
            ToppraPlanOptions,
        )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=3)
        )
        sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "p", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="p", max_workers=2))
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.1, 0.5, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=12,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.success.all().item()
            assert r.positions.shape == (B, 12, dofs)
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
class TestToppraNumericalRegression:
    def test_batched_equals_inline_single(self):
        from embodichain.lab.sim.planners.toppra_planner import (
            _toppra_solve_one_env,
            ToppraPlanner,
            ToppraPlannerCfg,
            ToppraPlanOptions,
        )
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg

        sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=4)
        )
        sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {"uid": "r", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}
            )
        )
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="r", max_workers=1))
        try:
            B, dofs = 4, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.1, 0.6, B)
            states = [
                PlanState.from_qpos(torch.zeros(B, dofs)),
                PlanState.from_qpos(wp),
            ]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=20,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            # Compare each env to the inline single-env solve
            for b in range(B):
                single = _toppra_solve_one_env(
                    np.stack([np.zeros(dofs), wp[b].numpy()]),
                    1.0,
                    2.0,
                    TrajectorySampleMethod.QUANTITY,
                    20,
                )
                assert np.allclose(
                    r.positions[b].cpu().numpy(), single["positions"], atol=1e-5
                )
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om

            om.SimulationManager.flush_cleanup_queue()
