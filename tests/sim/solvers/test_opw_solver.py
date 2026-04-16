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
from embodichain.lab.sim.cfg import RenderCfg


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
                    "qpos_limits": [
                        [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944],
                        [2.618, 3.14159, 0.0, 1.745, 1.22, 2.0944],
                    ],
                },
                "right_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "right_link6",
                    "root_link_name": "right_arm_base",
                    "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                    "qpos_limits": [
                        [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944],
                        [2.618, 3.14159, 0.0, 1.745, 1.22, 2.0944],
                    ],
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        qpos_limit = self.robot.get_qpos_limits(name=arm_name)
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
            ik_qpos, ik_qpos_xyzquat, atol=1e-4, rtol=1e-4
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
