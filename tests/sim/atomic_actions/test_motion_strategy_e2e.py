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

"""Reach-equivalence e2e test for MoveEndEffector across supported strategies."""

from __future__ import annotations

import torch
import pytest

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    MotionPolicy,
)


@pytest.mark.requires_sim
@pytest.mark.slow
class TestMotionStrategyReachEquivalence:
    """Verify MoveEndEffector reaches a reachable pose for ik_interp and motion_gen."""

    CONTROL_PART = "left_arm"
    ROBOT_UID = "cobot_e2e"
    SAMPLE_INTERVAL = 80
    POS_TOL = 0.02

    def _setup(self):
        sim = SimulationManager(SimulationManagerCfg(headless=True, sim_device="cpu"))
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

    def _run_reach_test(self, strategy: str):
        sim, robot, engine = self._setup()
        try:
            target, arm_ids = self._reachable_target(robot)
            result = engine.compile(
                (
                    ActionInvocation(
                        skill_id="move_end_effector",
                        goal=EndEffectorPoseGoal(xpos=target),
                        binding=ActionBinding(
                            manipulators={"primary": self.CONTROL_PART}
                        ),
                        motion_policy=MotionPolicy(
                            strategy=strategy,
                            sample_count=self.SAMPLE_INTERVAL,
                        ),
                    ),
                ),
                engine.initial_context(control_dt=sim.sim_config.physics_dt),
            )
            assert result.plan_success.all().item(), f"{strategy} reported failure"
            final_q = result.trajectory.positions[0, -1, arm_ids]
            fk = robot.compute_fk(
                qpos=final_q[None], name=self.CONTROL_PART, to_matrix=True
            )[0]
            err = torch.norm(fk[:3, 3] - target[:3, 3])
            assert err < self.POS_TOL, f"{strategy} EE pos error {err.item():.4f} m"
        finally:
            self._teardown(sim)

    def test_ik_interp_reaches_target(self):
        self._run_reach_test("ik_interp")

    def test_motion_gen_toppra_reaches_target(self):
        self._run_reach_test("motion_gen")
