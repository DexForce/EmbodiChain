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
import math
import torch
import pytest
import numpy as np

from embodichain.data import get_data_path
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix
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


def _franka_tcp() -> list[list[float]]:
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _pose_error(
    target_pose: torch.Tensor, actual_pose: torch.Tensor
) -> tuple[float, float]:
    pos_error = float(torch.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3]))
    target_quat = quat_from_matrix(target_pose[:3, :3].unsqueeze(0))
    actual_quat = quat_from_matrix(actual_pose[:3, :3].unsqueeze(0))
    rot_error = float(quat_error_magnitude(target_quat, actual_quat)[0])
    return pos_error, rot_error


def test_pytorch_solver_ik_respects_rotated_tcp():
    """FK->IK->FK should remain accurate when TCP contains rotation."""
    from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolverCfg

    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    cfg = PytorchSolverCfg(
        urdf_path=urdf,
        joint_names=[
            "Joint1",
            "Joint2",
            "Joint3",
            "Joint4",
            "Joint5",
            "Joint6",
            "Joint7",
        ],
        end_link_name="ee_link",
        root_link_name="base_link",
        tcp=_franka_tcp(),
        num_samples=30,
    )
    solver = cfg.init_solver(device=torch.device("cpu"))
    start_qpos = torch.tensor(
        [0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4],
        dtype=torch.float32,
    )
    target_qpos = start_qpos + torch.tensor(
        [0.12, -0.08, 0.10, 0.06, -0.07, 0.08, -0.05],
        dtype=torch.float32,
    )

    target_pose = solver.get_fk(target_qpos.unsqueeze(0))[0]
    success, ik_qpos = solver.get_ik(
        target_pose.unsqueeze(0),
        qpos_seed=start_qpos.unsqueeze(0),
    )
    final_pose = solver.get_fk(ik_qpos[:, 0, :])[0]
    pos_error, rot_error = _pose_error(target_pose, final_pose)

    assert success.all()
    assert pos_error < 1e-3
    assert rot_error < 5e-3


# Base test class for CPU and CUDA
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str):
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import RobotCfg

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

        self.robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

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
