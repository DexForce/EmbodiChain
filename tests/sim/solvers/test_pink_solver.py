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


# Base test class for differential solver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)
        self.sim.set_manual_update(False)

        # Load robot URDF file
        urdf = get_data_path("Rokae/SR5/SR5.urdf")

        assert os.path.isfile(urdf)

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "main_arm": [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ],
            },
            "solver_cfg": {
                "main_arm": {
                    "class_type": "PinkSolver",
                    "end_link_name": "ee_link",
                    "root_link_name": "base_link",
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    def test_differential_solver(self):
        # Test differential solver with a 1x4x4 homogeneous matrix pose and a joint_seed
        arm_name = "main_arm"

        qpos_fk = torch.tensor(
            [[0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0]], dtype=torch.float32
        )

        fk_xpos = self.robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

        # Define start and end poses
        start_pose = fk_xpos.clone()[0]
        end_pose = fk_xpos.clone()[0]
        end_pose[:3, 3] += torch.tensor([0.0, 0.4, 0.0], dtype=torch.float32)

        # Interpolate poses
        num_steps = 100
        interpolated_poses = [
            torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)
        ]

        ik_qpos = qpos_fk

        for i, pose in enumerate(interpolated_poses):
            res, ik_qpos = self.robot.compute_ik(
                pose=pose, joint_seed=ik_qpos, name=arm_name
            )
            assert res, f"IK failed for step {i} with pose:\n{pose}"

            # Verify forward kinematics matches the target pose
            ik_xpos = self.robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
            assert torch.allclose(
                pose, ik_xpos, atol=1e-3, rtol=1e-3
            ), f"FK result does not match target pose at step {i}."

            # Set robot joint positions
            self.robot.set_qpos(
                qpos=ik_qpos, joint_ids=self.robot.get_joint_ids(arm_name)
            )

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

    @classmethod
    def teardown_class(cls):
        if cls.sim is not None:
            try:
                cls.sim.destroy()
                print("sim destroyed successfully")
            except Exception as e:
                print(f"Error during sim.destroy(): {e}")


@pytest.mark.skip(reason="Skipping Pink tests temporarily")
class TestPinkSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="PinkSolver")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", __file__]
    pytest.main(pytest_args)
