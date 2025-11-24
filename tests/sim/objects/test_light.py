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

import pytest
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import LightCfg


class TestLight:
    def setup_method(self):
        # Setup SimulationManager
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)
        self.sim.build_multiple_arenas(10)

        # Create batch of lights
        cfg_dict = {
            "light_type": "point",
            "color": [0.1, 0.1, 0.1],
            "radius": 10.0,
            "position": [0.0, 0.0, 2.0],
            "uid": "point_light",
        }
        self.light = self.sim.add_light(cfg=LightCfg.from_dict(cfg_dict))

    def test_set_color_with_env_ids(self):
        """Test set_color with and without env_ids."""
        base_color = torch.tensor([0.1, 0.1, 0.1], device=self.sim.device)

        # Set for all environments
        try:
            self.light.set_color(base_color)
        except Exception as e:
            pytest.fail(f"Failed to set color for all envs: {e}")

        # Set for specific envs
        env_ids = [1, 3, 5]
        new_color = torch.tensor([0.9, 0.8, 0.7], device=self.sim.device)
        try:
            self.light.set_color(new_color, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"Failed to set color for env_ids={env_ids}: {e}")

    def test_set_falloff_with_env_ids(self):
        """Test set_falloff with and without env_ids."""
        base_val = torch.tensor(100.0, device=self.sim.device)

        # Set for all
        try:
            self.light.set_falloff(base_val)
        except Exception as e:
            pytest.fail(f"Failed to set falloff for all envs: {e}")

        env_ids = [0, 7, 9]
        new_vals = torch.tensor([200.0, 300.0, 400.0], device=self.sim.device)
        try:
            self.light.set_falloff(new_vals, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"Failed to set falloff for env_ids={env_ids}: {e}")

    def test_set_and_get_local_pose_matrix_and_vector(self):
        """
        Test setting and getting local pose in both matrix and vector forms.

        1. Set all lights to identity pose (4x4 matrix)
        2. Overwrite subset of lights (env_ids) with custom pose
        3. Check both vector and matrix results match expectations
        """

        # ----------------------------
        # 1. Set all lights to identity matrix
        # ----------------------------
        pose_matrix = torch.eye(4, device=self.sim.device)
        try:
            self.light.set_local_pose(pose_matrix, to_matrix=True)
        except Exception as e:
            pytest.fail(f"Failed to set pose matrix for all envs: {e}")

        result_matrix = self.light.get_local_pose(to_matrix=True)
        assert result_matrix.shape == (
            10,
            4,
            4,
        ), "Unexpected shape from get_local_pose(to_matrix=True)"
        for i, mat in enumerate(result_matrix):
            assert torch.allclose(
                mat, pose_matrix, atol=1e-5
            ), f"Initial matrix pose mismatch at env {i}"

        # ----------------------------
        # 2. Set translation via matrix for selected env_ids
        # ----------------------------
        env_ids = [2, 4, 6]
        pose_matrix_2 = (
            torch.eye(4, device=self.sim.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )
        pose_matrix_2[:, 0, 3] = 1.0
        pose_matrix_2[:, 1, 3] = 2.0
        pose_matrix_2[:, 2, 3] = 3.0

        try:
            self.light.set_local_pose(pose_matrix_2, env_ids=env_ids, to_matrix=True)
        except Exception as e:
            pytest.fail(f"Failed to set pose matrix for env_ids={env_ids}: {e}")

        # ----------------------------
        # 3. Check vector form after env_ids modification
        # ----------------------------
        result_vec = self.light.get_local_pose(to_matrix=False)
        assert result_vec.shape == (
            10,
            3,
        ), "Unexpected shape from get_local_pose(to_matrix=False)"

        for i in range(10):
            expected = (
                torch.tensor([1.0, 2.0, 3.0], device=self.sim.device)
                if i in env_ids
                else torch.tensor([0.0, 0.0, 0.0], device=self.sim.device)
            )
            assert torch.allclose(
                result_vec[i], expected, atol=1e-5
            ), f"Translation vector mismatch at env {i}"

        # ----------------------------
        # 4. Verify matrix form translation field
        # ----------------------------
        result_matrix = self.light.get_local_pose(to_matrix=True)
        for i in range(10):
            expected = (
                torch.tensor([1.0, 2.0, 3.0], device=self.sim.device)
                if i in env_ids
                else torch.tensor([0.0, 0.0, 0.0], device=self.sim.device)
            )
            assert torch.allclose(
                result_matrix[i][:3, 3], expected, atol=1e-5
            ), f"Translation matrix mismatch at env {i}"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
