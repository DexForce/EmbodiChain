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
        self.sim.build_multiple_arenas(1)

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
                },
                "right_arm": {
                    "class_type": solver_type,
                    "end_link_name": "right_ee",
                    "root_link_name": "right_arm_base",
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_differential_solver(self, arm_name: str):
        # Test differential solver with a 1x4x4 homogeneous matrix pose and a joint_seed

        qpos_fk = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.robot.device,
        )

        fk_xpos = self.robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

        # Define start and end poses
        start_pose = fk_xpos.clone()[0]
        end_pose = fk_xpos.clone()[0]
        end_pose[:3, 3] += torch.tensor(
            [0.0, 0.0, -0.02], dtype=torch.float32, device=self.robot.device
        )

        # Interpolate poses
        num_steps = 5
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
                pose, ik_xpos, atol=5e-3, rtol=5e-3
            ), f"FK result does not match target pose at step {i}."

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


class TestDifferentialSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="DifferentialSolver")


if __name__ == "__main__":
    torch.set_printoptions(precision=5, sci_mode=False)
    pytest_args = ["-v", __file__]
    pytest.main(pytest_args)
