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

import numpy as np
import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
from embodichain.lab.sim.solvers import NeuralIKSolverCfg
from embodichain.lab.sim.solvers.neural_ik_solver import _build_mlp
from embodichain.utils.utility import reset_all_seeds

NUM_ARM_JOINTS = 7
OBS_DIM = 2 * NUM_ARM_JOINTS + 14  # 28
HIDDEN_DIMS = [256, 256]


def _create_fake_checkpoint(tmp_path) -> str:
    """Create a minimal fake checkpoint for testing the solver interface."""
    mlp = _build_mlp(OBS_DIM, HIDDEN_DIMS, NUM_ARM_JOINTS)
    ckpt = {
        "agent": {f"actor_mean.{k}": v for k, v in mlp.state_dict().items()},
        "obs_normalizer": {
            "mean": torch.zeros(OBS_DIM),
            "var": torch.ones(OBS_DIM),
        },
    }
    ckpt_path = str(tmp_path / "fake_neural_ik.pt")
    torch.save(ckpt, ckpt_path)
    return ckpt_path


class TestNeuralIKSolver:
    sim: SimulationManager | None = None
    robot: Robot | None = None

    def _setup(self, tmp_path):
        checkpoint_path = _create_fake_checkpoint(tmp_path)
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)

        cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
        cfg.solver_cfg["arm"] = NeuralIKSolverCfg(
            end_link_name="fr3_hand_tcp",
            root_link_name="base",
            tcp=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            checkpoint_path=checkpoint_path,
            num_arm_joints=NUM_ARM_JOINTS,
            max_steps=30,
            action_scale=0.2,
            hidden_dims=HIDDEN_DIMS,
            pos_eps=0.1,
            rot_eps=0.5,
        )

        self.robot: Robot = self.sim.add_robot(cfg=cfg)
        self.sim.update(step=100)

    def teardown_method(self):
        if self.sim is not None:
            self.sim.destroy()

    def _make_solver_input(self):
        """Create a standard qpos and its FK target for solver tests."""
        arm_name = "arm"
        qpos = torch.tensor(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=torch.float32,
            device=self.robot.device,
        ).unsqueeze(0)
        target_xpos = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
        solver = self.robot.get_solver(arm_name)
        return solver, qpos, target_xpos

    def test_ik_interface(self, tmp_path):
        """Verify compute_ik returns correct shapes and types."""
        reset_all_seeds(0)
        self._setup(tmp_path)
        arm_name = "arm"

        qpos = torch.tensor(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=torch.float32,
            device=self.robot.device,
        ).unsqueeze(0)
        target_xpos = self.robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)

        res, ik_qpos = self.robot.compute_ik(
            pose=target_xpos, joint_seed=qpos, name=arm_name
        )

        assert res.shape == (1,)
        assert res.dtype == torch.bool
        dof = qpos.shape[-1]
        assert ik_qpos.shape[-1] == dof

        # test for unreachable pose
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
            pose=invalid_pose, joint_seed=qpos, name=arm_name
        )
        assert res[0].item() is False

    def test_multi_sample_shape(self, tmp_path):
        """Verify output shape when using multiple samples."""
        reset_all_seeds(0)
        self._setup(tmp_path)
        solver, qpos, target_xpos = self._make_solver_input()

        success, ik_qpos = solver.get_ik(
            target_xpos=target_xpos,
            qpos_seed=qpos,
            num_samples=5,
        )

        dof = qpos.shape[-1]
        assert success.shape == (1,)
        assert ik_qpos.shape == (1, 1, dof)

    def test_multi_sample_return_all(self, tmp_path):
        """Verify return_all_solutions returns all sampled solutions."""
        reset_all_seeds(0)
        self._setup(tmp_path)
        solver, qpos, target_xpos = self._make_solver_input()
        num_samples = 5

        success, ik_qpos = solver.get_ik(
            target_xpos=target_xpos,
            qpos_seed=qpos,
            num_samples=num_samples,
            return_all_solutions=True,
        )

        dof = qpos.shape[-1]
        assert success.shape == (1,)
        assert ik_qpos.shape == (1, num_samples, dof)


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
