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

"""Reach-equivalence e2e test for MoveEndEffector across motion sources."""

from __future__ import annotations

import torch
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.actions import (
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.lab.sim.atomic_actions.core import EndEffectorPoseTarget


@pytest.mark.requires_sim
@pytest.mark.slow
class TestMotionSourceReachEquivalence:
    """Verify MoveEndEffector reaches a reachable pose for ik_interp and motion_gen."""

    CONTROL_PART = "left_arm"
    ROBOT_UID = "cobot_e2e"
    SAMPLE_INTERVAL = 80
    POS_TOL = 0.02

    def _setup(self, motion_source: str):
        sim = SimulationManager(SimulationManagerCfg(headless=True, device="cpu"))
        robot = sim.add_robot(
            cfg=CobotMagicCfg.from_dict(
                {
                    "uid": self.ROBOT_UID,
                    "init_pos": [0.0, 0.0, 0.7775],
                    "init_qpos": [0.0] * 16,
                }
            )
        )
        mg = MotionGenerator(
            MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.ROBOT_UID))
        )
        engine = AtomicActionEngine(mg)
        cfg = MoveEndEffectorCfg(
            motion_source=motion_source,
            planner_type="toppra" if motion_source == "motion_gen" else None,
            control_part=self.CONTROL_PART,
            sample_interval=self.SAMPLE_INTERVAL,
        )
        engine.register(MoveEndEffector(mg, cfg), name="move_end_effector")
        return sim, robot, engine

    def _teardown(self, sim):
        sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()

    def _reachable_target(self, robot):
        """Return the current EE pose shifted 5 cm upward and the arm joint ids."""
        arm_ids = robot.get_joint_ids(name=self.CONTROL_PART)
        qpos = robot.get_qpos(name=self.CONTROL_PART)
        fk = robot.compute_fk(qpos=qpos, name=self.CONTROL_PART, to_matrix=True)
        target = fk[0].clone()
        target[2, 3] += 0.05
        return target, arm_ids

    def _run_reach_test(self, motion_source: str):
        sim, robot, engine = self._setup(motion_source)
        try:
            target, arm_ids = self._reachable_target(robot)
            success, traj, _ = engine.run(
                [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
            )
            assert success.all().item(), f"{motion_source} reported failure"
            final_q = traj[0, -1, arm_ids]
            fk = robot.compute_fk(
                qpos=final_q[None], name=self.CONTROL_PART, to_matrix=True
            )[0]
            err = torch.norm(fk[:3, 3] - target[:3, 3])
            assert (
                err < self.POS_TOL
            ), f"{motion_source} EE pos error {err.item():.4f} m"
        finally:
            self._teardown(sim)

    def test_ik_interp_reaches_target(self):
        self._run_reach_test("ik_interp")

    def test_motion_gen_toppra_reaches_target(self):
        self._run_reach_test("motion_gen")
