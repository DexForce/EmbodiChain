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
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.data import get_data_path


# Define control parts
CONTROL_PARTS = {
    "left_arm": [
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
    ],
    "right_arm": [
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    ],
}


# Base test class for CPU and CUDA
class BaseRobotTest:
    def setup_simulation(self, sim_device):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, sim_device=sim_device, num_envs=10)
        self.sim = SimulationManager(config)

        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "version": "v021",
                "arm_kind": "anthropomorphic",
            }
        )

        self.robot: Robot = self.sim.add_robot(cfg=cfg)

        # Initialize GPU physics if needed
        if sim_device == "cuda" and getattr(self.sim, "is_use_gpu_physics", False):
            self.sim.init_gpu_physics()

    def test_get_joint_ids(self):
        left_joint_ids = self.robot.get_joint_ids("left_arm")
        right_joint_ids = self.robot.get_joint_ids("right_arm")

        assert left_joint_ids == [
            6,
            8,
            10,
            12,
            14,
            16,
            18,
        ], f"Unexpected left arm joint IDs: {left_joint_ids}"
        assert right_joint_ids == [
            7,
            9,
            11,
            13,
            15,
            17,
            19,
        ], f"Unexpected right arm joint IDs: {right_joint_ids}"

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_fk(self, arm_name: str):
        # Test forward kinematics (FK) for both to_matrix=True and to_matrix=False

        qpos = torch.randn(10, 7, device=self.sim.device)  # Random joint positions

        # Test with to_matrix=False (6D result: translation + Euler angles)
        result_7d = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=False)

        # Check result shape for 6D output (batch, 6)
        assert result_7d.shape == (
            10,
            7,
        ), f"Expected shape (10, 7), got {result_7d.shape}"

        # Test with to_matrix=True (4x4 matrix result)
        result_matrix = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
        print("result_matrix:", result_matrix)
        # Check result shape for matrix output (batch, 4, 4)
        assert result_matrix.shape == (
            10,
            4,
            4,
        ), f"Expected shape (10, 4, 4), got {result_matrix.shape}"

    def test_compute_fk(self):
        torch.set_printoptions(precision=6, sci_mode=False)
        qpos = np.zeros(40)
        result = self.robot.compute_fk(qpos=qpos, link_names=["left_ee", "right_ee"])

        # Additional checks for specific values (if known)
        expected_values = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.791],
                    [0.0, -1.0, 0.0, 1.3648],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -0.791],
                    [0.0, 1.0, 0.0, 1.3648],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze_(0)

        assert torch.allclose(
            result, expected_values, atol=1e-4, rtol=1e-4
        ), f"FK result does not match expected values. Got {result}, expected {expected_values}."

    def test_compute_jacobian(self):
        qpos = np.full(7, 10 * np.pi / 180)

        left_ee_jacobian = self.robot.compute_jacobian(
            qpos=qpos, end_link_name="left_ee", root_link_name="left_arm_base"
        )
        right_ee_jacobian = self.robot.compute_jacobian(
            qpos=qpos, end_link_name="right_ee", root_link_name="right_arm_base"
        )

        assert left_ee_jacobian.shape == (
            1,
            6,
            7,
        ), f"Expected shape (1, 6, 7) for left EE Jacobian, got {left_ee_jacobian.shape}"
        assert right_ee_jacobian.shape == (
            1,
            6,
            7,
        ), f"Expected shape (1, 6, 7) for right EE Jacobian, got {right_ee_jacobian.shape}"

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        # Define a sample target pose as a 1x4x4 homogeneous matrix
        target_pose = torch.tensor(
            [
                [-0.3490, -0.6369, -0.6874, -0.4502],
                [0.2168, -0.7685, 0.6020, -0.0639],
                [-0.9117, 0.0611, 0.4063, 0.3361],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float32,
            device=self.sim.device,
        ).unsqueeze(0)

        # Define joint_seed as a tensor of ones with shape (1, 7) for initialization
        joint_seed = torch.ones(1, 7, device=self.sim.device)
        success_tensor, qpos_tensor = self.robot.compute_ik(
            pose=target_pose, name=arm_name, joint_seed=joint_seed, env_ids=[0]
        )
        print(f"Success: {success_tensor}, Qpos: {qpos_tensor}")

        # Check output shapes robustly
        assert success_tensor.shape == (
            1,
        ), f"Expected shape (1,), got {success_tensor.shape}"
        assert isinstance(
            qpos_tensor, torch.Tensor
        ), "qpos_tensor should be a torch.Tensor"
        # Accept both (1, 7) and (1, N, 7) shapes
        if qpos_tensor.ndim == 2:
            assert qpos_tensor.shape == (
                1,
                7,
            ), f"Expected shape (1, 7), got {qpos_tensor.shape}"
        elif qpos_tensor.ndim == 3:
            assert (
                qpos_tensor.shape[2] == 7
            ), f"Expected dof 7, got {qpos_tensor.shape[2]}"
            assert (
                qpos_tensor.shape[0] == 1
            ), f"Expected batch size 1, got {qpos_tensor.shape[0]}"
            assert (
                qpos_tensor.shape[1] >= 1
            ), f"Expected at least one solution, got {qpos_tensor.shape[1]}"
        else:
            raise AssertionError(f"Unexpected qpos_tensor shape: {qpos_tensor.shape}")

        # If success, check qpos is not all zeros
        if success_tensor.item():
            assert not torch.all(
                qpos_tensor == 0
            ), "IK returned all zeros for valid solution"

    def test_mimic(self):

        assert (
            len(self.robot.mimic_ids) == 8
        ), f"Expected 8 mimic IDs, got {len(self.robot.mimic_ids)}"

        left_eef_ids_without_mimic = self.robot.get_joint_ids(
            "left_eef", remove_mimic=False
        )
        right_eef_ids_without_mimic = self.robot.get_joint_ids(
            "right_eef", remove_mimic=False
        )
        assert (
            len(left_eef_ids_without_mimic) == 6
        ), f"Expected 6 left eef joint IDs without mimic, got {len(left_eef_ids_without_mimic)}"
        assert (
            len(right_eef_ids_without_mimic) == 6
        ), f"Expected 6 right eef joint IDs without mimic, got {len(right_eef_ids_without_mimic)}"

    def test_setter_and_getter_with_control_part(self):
        left_arm_qpos = self.robot.get_qpos(name="left_arm")
        assert left_arm_qpos.shape == (10, 7)

        left_qpos_limits = self.robot.get_qpos_limits(name="left_arm")
        assert left_qpos_limits.shape == (10, 7, 2)

        dummy_qpos = torch.randn(10, 7, device=self.sim.device)
        # Clamp to limits
        dummy_qpos = torch.max(
            torch.min(dummy_qpos, left_qpos_limits[:, :, 1]), left_qpos_limits[:, :, 0]
        )
        self.robot.set_qpos(qpos=dummy_qpos, name="left_arm")

    def test_robot_cfg_merge(self):
        from copy import deepcopy
        from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg

        cfg = deepcopy(self.robot.cfg)

        cfg_dict = {
            "drive_pros": {
                "max_effort": {
                    "(RIGHT|LEFT)_[A-Z|_]+": 1.0,
                },
            },
            "solver_cfg": {
                "left_arm": {
                    "tcp": np.eye(4),
                }
            },
        }

        cfg = merge_robot_cfg(cfg, cfg_dict)

        assert (
            cfg.drive_pros.max_effort["(RIGHT|LEFT)_[A-Z|_]+"] == 1.0
        ), "Drive properties merge failed."

        assert np.allclose(
            cfg.solver_cfg["left_arm"].tcp, np.eye(4)
        ), "Solver config merge failed."

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()

    def test_set_physical_visible(self):
        self.robot.set_physical_visible(
            visible=True,
            rgba=(0.1, 0.1, 0.9, 0.4),
            control_part="left_arm",
        )
        self.robot.set_physical_visible(
            visible=True,
            control_part="left_arm",
        )
        self.robot.set_physical_visible(
            visible=False,
            control_part="left_arm",
        )


class TestRobotCPU(BaseRobotTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestRobotCUDA(BaseRobotTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    # Run tests directly
    test_cpu = TestRobotCPU()
    test_cpu.setup_method()
    test_cpu.test_fk("left_arm")
