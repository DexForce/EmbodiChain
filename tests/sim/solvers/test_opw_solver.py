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

import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg


# Base test class for OPWSolver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(headless=True, sim_device=sim_device)
        self.sim = SimulationManager(config)
        self.sim.set_manual_update(False)

        cfg_dict = {
            "uid": "CobotMagic",
            "init_pos": [0.0, 0.0, 0.7775],
            "init_qpos": [
                -0.3,
                0.3,
                1.0,
                1.0,
                -1.2,
                -1.2,
                0.0,
                0.0,
                0.6,
                0.6,
                0.0,
                0.0,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
            "solver_cfg": {
                "left_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "left_link6",
                    "root_link_name": "left_arm_base",
                    "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                },
                "right_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "right_link6",
                    "root_link_name": "right_arm_base",
                    "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        test_qpos = torch.tensor(
            [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]],
            dtype=torch.float32,
            device=self.robot.device
        )

        fk_xpos = self.robot.compute_fk(qpos=test_qpos, name=arm_name, to_matrix=True)

        res, ik_qpos = self.robot.compute_ik(
            pose=fk_xpos, joint_seed=test_qpos, name=arm_name
        )

        res, ik_qpos2 = self.robot.compute_ik(pose=fk_xpos, name=arm_name)

        if ik_qpos2.dim() == 3:
            ik_xpos = self.robot.compute_fk(
                qpos=ik_qpos2[0][0], name=arm_name, to_matrix=True
            )
        else:
            ik_xpos = self.robot.compute_fk(
                qpos=ik_qpos2, name=arm_name, to_matrix=True
            )

        assert torch.allclose(
            test_qpos, ik_qpos, atol=5e-3, rtol=5e-3
        ), f"FK and IK qpos do not match for {arm_name}"

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=5e-3, rtol=5e-3
        ), f"FK and IK xpos do not match for {arm_name}"
        # test for failed xpos
        invalid_pose = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 10.0],
                    [0.0, 0.0, 1.0, 10.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device=self.robot.device,
        )
        res, ik_qpos = self.robot.compute_ik(
            pose=invalid_pose, joint_seed=ik_qpos, name=arm_name
        )
        dof = ik_qpos.shape[-1]
        assert res[0] == False
        assert ik_qpos.shape == (1, dof)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestOPWSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestOPWSolverCUDA(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
