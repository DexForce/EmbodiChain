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
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.utils.utility import reset_all_seeds

CHECKPOINT_REPO = "dexforce/neural_ik_solver"
CHECKPOINT_FILE = "franka.pt"

IK_POS_ATOL = 0.05
IK_ROT_ATOL = 0.55

_c = math.cos(-math.pi / 4)
_s = math.sin(-math.pi / 4)
TCP = [
    [_c, -_s, 0.0, 0.0],
    [_s, _c, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.1034],
    [0.0, 0.0, 0.0, 1.0],
]


def grid_sample_qpos_from_limits(
    qpos_limits: torch.Tensor,
    steps_per_joint: int = 4,
    device=None,
    max_samples: int = 4096,
) -> torch.Tensor:
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


# Skip tests if huggingface_hub cannot reach the checkpoint repo.
pytestmark = pytest.mark.skipif(
    os.environ.get("NEURAL_IK_OFFLINE") is not None,
    reason="NEURAL_IK_OFFLINE is set, skipping NeuralIK tests",
)


class BaseSolverTest:
    sim = None

    def setup_simulation(self, solver_type: str):
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)

        urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
        assert os.path.isfile(urdf)
        checkpoint_path = hf_hub_download(
            repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE
        )

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "main_arm": [
                    "Joint1",
                    "Joint2",
                    "Joint3",
                    "Joint4",
                    "Joint5",
                    "Joint6",
                    "Joint7",
                ],
            },
            "solver_cfg": {
                "main_arm": {
                    "class_type": solver_type,
                    "end_link_name": "ee_link",
                    "root_link_name": "base_link",
                    "tcp": TCP,
                    "checkpoint_path": checkpoint_path,
                    "num_arm_joints": 7,
                    "max_steps": 30,
                    "action_scale": 0.2,
                    "hidden_dims": [256, 256],
                    "pos_eps": 0.1,
                    "rot_eps": 0.5,
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
        self.sim.update(step=100)

    def test_ik(self):
        reset_all_seeds(0)
        arm_name = "main_arm"
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

        pos_err = (fk_xpos[:, :, :3, 3] - ik_xpos[:, :, :3, 3]).norm(dim=-1)
        assert torch.all(
            pos_err < IK_POS_ATOL
        ), f"Position error too large: max {pos_err.max().item():.4f} m > {IK_POS_ATOL} m"

        rot_err = (fk_xpos[:, :, :3, :3] - ik_xpos[:, :, :3, :3]).norm(dim=(-2, -1))
        assert torch.all(
            rot_err < IK_ROT_ATOL
        ), f"Rotation error too large: max {rot_err.max().item():.4f} > {IK_ROT_ATOL}"

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


class TestNeuralIKSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="NeuralIKSolver")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    test_solver = TestNeuralIKSolver()
    test_solver.setup_method()
