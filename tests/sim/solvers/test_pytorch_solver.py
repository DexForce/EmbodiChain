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

import os
import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.data import get_data_path
from embodichain.utils.utility import reset_all_seeds


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device=None,
    max_samples: int = 4096,
) -> torch.Tensor:
    """Generate grid samples for qpos from qpos_limits.

    Args:
        qpos_limits: tensor of shape (1, n, 2) or (n, 2) where each row is [low, high].
        steps_per_joint: number of values per joint (defaults to 2: low and high).
        device: torch device to place the samples on.
        max_samples: cap the number of returned samples (take first N if grid is larger).

    Returns:
        Tensor of shape (N, n) where N <= max_samples.
    """
    if device is None:
        device = qpos_limits.device

    limits = qpos_limits.squeeze(0) if qpos_limits.dim() == 3 else qpos_limits
    lows = limits[:, 0].to(device) + 1e-2
    highs = limits[:, 1].to(device) - 1e-2

    # create per-joint linspaces
    grids = [
        torch.linspace(l.item(), h.item(), steps_per_joint, device=device)
        for l, h in zip(lows, highs)
    ]

    # meshgrid and stack
    mesh = torch.meshgrid(*grids, indexing="ij")
    stacked = torch.stack([m.reshape(-1) for m in mesh], dim=1)

    if stacked.shape[0] > max_samples:
        return stacked[:max_samples]
    return stacked


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
                    "num_samples": 30,
                },
                "right_arm": {
                    "class_type": solver_type,
                    "end_link_name": "right_ee",
                    "root_link_name": "right_arm_base",
                    "num_samples": 30,
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

        # Wait for robot to stabilize.
        self.sim.update(step=100)

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        reset_all_seeds(0)
        qpos_limit = torch.tensor(
            [
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.2, 0.8],
            ]
        )
        # generate a small grid of qpos samples from the joint limits (low/high)
        sample_qpos = grid_sample_qpos_from_limits(
            qpos_limit, steps_per_joint=3, device=self.robot.device, max_samples=200
        )
        sample_qpos = sample_qpos[None, :, :]

        fk_xpos = self.robot.compute_batch_fk(
            qpos=sample_qpos, name=arm_name, to_matrix=True
        )
        fk_xpos_xyzquat = self.robot.compute_batch_fk(
            qpos=sample_qpos, name=arm_name, to_matrix=False
        )

        res, ik_qpos = self.robot.compute_batch_ik(
            pose=fk_xpos, joint_seed=sample_qpos, name=arm_name
        )

        res, ik_qpos_xyzquat = self.robot.compute_batch_ik(
            pose=fk_xpos_xyzquat, joint_seed=sample_qpos, name=arm_name
        )

        ik_xpos = self.robot.compute_batch_fk(
            qpos=ik_qpos_xyzquat, name=arm_name, to_matrix=True
        )

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
            pose=invalid_pose, joint_seed=ik_qpos[:, 0, :], name=arm_name
        )
        dof = ik_qpos.shape[-1]
        assert res[0].item() == False
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
