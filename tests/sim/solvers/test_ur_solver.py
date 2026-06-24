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

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.solvers import URSolverCfg
from embodichain.lab.sim.cfg import (
    RenderCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    URDFCfg,
)


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device=None,
    max_samples: int = 4096,
    safe_margin: float = 5 / 180 * np.pi,  # 5 degrees in radians
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
    lows = limits[:, 0].to(device) + safe_margin * 1.01
    highs = limits[:, 1].to(device) - safe_margin * 1.01

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


# Base test class for OPWSolver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, sim_device):
        config = SimulationManagerCfg(headless=True, sim_device=sim_device)
        self.sim = SimulationManager(config)
        self.sim.set_manual_update(False)

        ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
        gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
        # Configure the robot with its components and control properties
        cfg = RobotCfg(
            uid="UR10",
            urdf_cfg=URDFCfg(
                components=[
                    {"component_type": "arm", "urdf_path": ur10_urdf_path},
                    {"component_type": "hand", "urdf_path": gripper_urdf_path},
                ]
            ),
            drive_pros=JointDrivePropertiesCfg(
                stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e3},
                damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e2},
                max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e4},
                drive_type="force",
            ),
            control_parts={
                "arm": ["JOINT[0-9]"],
                "hand": ["FINGER[1-2]"],
            },
            solver_cfg={
                "arm": URSolverCfg(
                    ur_type="ur10",
                    tcp=[
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.12],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
            },
            init_qpos=[
                0.0,
                -np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                0.0,
                0.0,
                0.0,
            ],
            init_pos=(0, 0, 0),
        )
        self.robot: Robot = self.sim.add_robot(cfg=cfg)

    def test_ik(self):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed
        arm_name = "arm"
        # qpos_limit = self.robot.get_qpos_limits(name=arm_name)
        qpos_limit = torch.tensor(
            [
                [
                    [-3.14, 3.14],
                    [-3.14, 3.14],
                    [-3.14, 3.14],
                    [-3.14, 3.14],
                    [-3.14, 3.14],
                    [-3.14, 3.14],
                ]
            ],
            dtype=torch.float32,
            device=self.robot.device,
        )
        # generate a small grid of qpos samples from the joint limits (low/high)
        sample_qpos = grid_sample_qpos_from_limits(
            qpos_limit, steps_per_joint=8, device=self.robot.device, max_samples=65536
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

        assert torch.allclose(
            ik_qpos, ik_qpos_xyzquat, atol=5e-3, rtol=5e-3
        ), "IK results do not match for different pose formats"

        ik_xpos = self.robot.compute_batch_fk(
            qpos=ik_qpos_xyzquat, name=arm_name, to_matrix=True
        )
        assert torch.allclose(
            sample_qpos, ik_qpos, atol=5e-3, rtol=5e-3
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
            pose=invalid_pose, joint_seed=ik_qpos[:, 0, :], name=arm_name
        )
        dof = ik_qpos.shape[-1]
        assert res[0] == False
        assert ik_qpos.shape == (1, dof)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestURSolverCUDA(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cuda")


class TestURSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cpu")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
