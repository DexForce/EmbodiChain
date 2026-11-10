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

import torch
import pytest

from solver_support import grid_sample_qpos_from_limits, assert_pose_ik_roundtrip
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain_robots import CobotMagicCfg


# Base test class for OPWSolver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, device):
        config = SimulationManagerCfg(headless=True, device=device)
        self.sim = SimulationManager(config)

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
                    "tcp": [
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0.143],
                        [0, 0, 0, 1],
                    ],
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
        self.sim.prepare()

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_ik(self, arm_name: str) -> None:
        assert_pose_ik_roundtrip(self.robot, arm_name, safe_margin=20 / 180 * np.pi)

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_continuous_batch_ik_reconstructs_fk_path(self, arm_name: str):
        """Continuous batch IK preserves a branch across an OPW pose path."""
        qpos_limits = self.robot.get_qpos_limits(name=arm_name)
        qpos = grid_sample_qpos_from_limits(
            safe_margin=20 / 180 * np.pi,
            qpos_limits=qpos_limits,
            steps_per_joint=2,
            device=self.robot.device,
            max_samples=1,
        )
        qpos_path = qpos[None].expand(1, 3, -1).contiguous()
        target_path = self.robot.compute_batch_fk(
            qpos=qpos_path, name=arm_name, to_matrix=True
        )

        success, solved_path = self.robot.compute_batch_ik(
            pose=target_path,
            joint_seed=qpos,
            name=arm_name,
            continuous=True,
        )

        assert success.all()
        reconstructed = self.robot.compute_batch_fk(
            qpos=solved_path, name=arm_name, to_matrix=True
        )
        assert torch.allclose(target_path, reconstructed, atol=5e-3, rtol=5e-3)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()


class TestOPWSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cpu")


class TestOPWSolverCUDA(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation("cuda")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
