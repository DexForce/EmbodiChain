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

"""Qualify the downloaded PiPER-X arm and its assembled AGX tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg
from embodichain.robot_registry import resolve_robot_cfg

pytestmark = [pytest.mark.slow, pytest.mark.requires_sim]

_EMBODIMENT = (
    Path(__file__).resolve().parents[3]
    / "extensions/embodichain_tasks/configs/components/embodiments"
    / "piper_x_agx_gripper.yaml"
)
_ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]
_FULL_OPENING = 0.1  # Official AGX opening range is 0–100 mm.
_FLANGE_TO_TIP = 0.1425  # 4.5 mm mount offset + 138 mm to the finger tips.


class TestAgilexPiperX:
    def setup_method(self) -> None:
        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                physics_cfg=DefaultPhysicsCfg(device="cpu", physics_dt=0.001),
                startup_summary="off",
                robot_ik_gizmo=None,
            )
        )

    def teardown_method(self) -> None:
        self.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    @pytest.mark.parametrize("mounted", [False, True], ids=["bare", "agx_gripper"])
    def test_flange_and_tcp_agree_with_simulation_and_round_trip_ik(
        self, mounted: bool
    ) -> None:
        if mounted:
            cfg = config_to_cfg(
                {
                    "id": "EmbodiChain-PiPERX-AssemblyTest-v0",
                    "physics": "default",
                    "env": {},
                    "embodiment": {"component": str(_EMBODIMENT)},
                }
            ).robot
        else:
            cfg = resolve_robot_cfg("AgilexPiperX")
        cfg = type(cfg).from_dict(json.loads(json.dumps(cfg.to_dict())))
        cfg.validate()
        assert cfg.control_parts == (
            {"arm": _ARM_JOINTS, "eef": ["gripper"]}
            if mounted
            else {"arm": _ARM_JOINTS}
        )
        assert (
            cfg.build_pk_serial_chain()["arm"].get_joint_parameter_names()
            == _ARM_JOINTS
        )

        robot = self.sim.add_robot(cfg)
        self.sim.prepare()
        home = robot.get_qpos(name="arm").clone()
        qpos = home + torch.tensor([[0.14, 0.10, -0.08, 0.11, -0.12, 0.18]])
        robot.set_qpos(qpos, name="arm", target=False)
        expected = robot.get_link_pose("flange_link", to_matrix=True)
        if mounted:
            # Build the TCP from the actual gripper root pose and source finger length.
            expected = robot.get_link_pose("gripper_base", to_matrix=True)
            expected[:, :3, 3] += expected[:, :3, 2] * (_FLANGE_TO_TIP - 0.0045)

        actual = robot.compute_fk(qpos, name="arm", to_matrix=True)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
        success, solution = robot.compute_ik(actual, joint_seed=home, name="arm")
        assert success.item()
        recovered = robot.compute_fk(solution.reshape(1, 6), name="arm", to_matrix=True)
        torch.testing.assert_close(recovered, actual, atol=1e-3, rtol=0)

    def test_agx_gripper_drive_moves_both_fingers_symmetrically(self) -> None:
        cfg = config_to_cfg(
            {
                "id": "EmbodiChain-PiPERX-DriveTest-v0",
                "physics": "default",
                "env": {},
                "embodiment": {"component": str(_EMBODIMENT)},
            }
        ).robot
        robot = self.sim.add_robot(cfg)
        self.sim.prepare()
        home = robot.get_qpos(name="arm").clone()

        for opening in (0.0, _FULL_OPENING):
            robot.set_qpos(torch.tensor([[opening]]), name="eef")
            self.sim.update(step=900, render_final_step=False)
            qpos = dict(zip(robot.joint_names, robot.get_qpos()[0].tolist()))
            assert qpos["gripper"] == pytest.approx(opening, abs=0.003)
            assert qpos["gripper_joint1"] == pytest.approx(opening / 2, abs=0.002)
            assert qpos["gripper_joint2"] == pytest.approx(-opening / 2, abs=0.002)
            finger_left = robot.get_link_pose("gripper_link1")[0, :3]
            finger_right = robot.get_link_pose("gripper_link2")[0, :3]
            gap = torch.linalg.vector_norm(finger_left - finger_right).item()
            assert gap == pytest.approx(opening, abs=0.004)
        torch.testing.assert_close(robot.get_qpos(name="arm"), home, atol=0.035, rtol=0)


if __name__ == "__main__":
    from embodichain.lab.scripts.preview_robot import cli

    cli(["--robot", "AgilexPiperX"])
