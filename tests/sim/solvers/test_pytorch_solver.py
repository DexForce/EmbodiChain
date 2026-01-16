# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import os
import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.data import get_data_path


# Base test class for CPU and CUDA
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)

        # Load robot URDF file
        urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
        assert os.path.isfile(urdf)

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
                "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
            },
            "solver_cfg": {
                "left_arm": {
                    "class_type": solver_type,
                    "end_link_name": "left_ee",
                    "root_link_name": "left_arm_base",
                    "ik_nearest_weight": [1.0, 1.0, 1.0, 0.9, 0.9, 0.1, 0.1],
                },
                "right_arm": {
                    "class_type": solver_type,
                    "end_link_name": "right_ee",
                    "root_link_name": "right_arm_base",
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

        # Wait for robot to stabilize.
        self.sim.update(step=100)

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        qpos_fk = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]], dtype=torch.float32
        )

        fk_xpos = self.robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

        res, ik_qpos = self.robot.compute_ik(pose=fk_xpos, name=arm_name)

        if ik_qpos.dim() == 3:
            ik_xpos = self.robot.compute_fk(
                qpos=ik_qpos[0][0], name=arm_name, to_matrix=True
            )
        else:
            ik_xpos = self.robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=1e-2, rtol=1e-2
        ), f"FK and IK results do not match for {arm_name}"

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


class TestPytorchSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="PytorchSolver")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    test_solver = TestPytorchSolver()
    test_solver.setup_method()
