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
import gymnasium as gym

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, DEFAULT_MANAGER_MODULES
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.data import get_data_path

NUM_ENVS = 10

urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
METADATA = {
    "id": "EmbodiedEnv-v1",
    "max_episodes": 1,
    "env": {
        "events": {
            "random_light": {
                "func": "randomize_light",
                "mode": "interval",
                "interval_step": 10,
                "params": {
                    "entity_cfg": {"uid": "light_1"},
                    "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
                    "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                    "intensity_range": [500000.0, 1500000.0],
                },
            }
        }
    },
    "sensor": [
        {
            "sensor_type": "Camera",
            "width": 640,
            "height": 480,
            "enable_mask": True,
            "enable_depth": True,
            "extrinsics": {
                "eye": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
        }
    ],
    "robot": {
        "fpath": urdf_path,
        "drive_pros": {"stiffness": {"joint[1-6]": 200.0}},
        "solver_cfg": {
            "class_type": "PytorchSolver",
            "end_link_name": "ee_link",
            "root_link_name": "base_link",
        },
        "init_pos": [0.0, 0.3, 1.0],
    },
    "light": {
        "direct": [
            {
                "uid": "light_1",
                "light_type": "point",
                "color": [1.0, 1.0, 1.0],
                "intensity": 1000000.0,
                "init_pos": [0, 0, 2],
                "radius": 10.0,
            }
        ]
    },
    "background": [
        {
            "uid": "shop_table",
            "shape": {
                "shape_type": "Mesh",
                "fpath": "ShopTableSimple/shop_table_simple.ply",
            },
            "max_convex_hull_num": 2,
            "attrs": {"mass": 10.0},
            "body_scale": (2, 1.6, 1),
        }
    ],
    "rigid_object": [
        {
            "uid": "duck",
            "shape": {
                "shape_type": "Mesh",
                "fpath": "ToyDuck/toy_duck.glb",
            },
            "body_scale": (0.75, 0.75, 1.0),
            "init_pos": (0.0, 0.0, 1.0),
        }
    ],
    "articulation": [
        {
            "uid": "sliding_box_drawer",
            "fpath": "SlidingBoxDrawer/SlidingBoxDrawer.urdf",
            "init_pos": (0.5, 0.0, 0.5),
        }
    ],
}


class EmbodiedEnvTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        cfg: EmbodiedEnvCfg = config_to_cfg(
            METADATA, manager_modules=DEFAULT_MANAGER_MODULES
        )
        cfg.num_envs = NUM_ENVS
        cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=sim_device)

        self.env = gym.make(id=METADATA["id"], cfg=cfg)

    def test_env_rollout(self):
        """Test environment rollout."""
        for episode in range(2):
            print("Episode:", episode)
            obs, info = self.env.reset()

            for i in range(2):
                action = self.env.action_space.sample()
                action = torch.as_tensor(
                    action,
                    dtype=torch.float32,
                    device=self.env.get_wrapper_attr("device"),
                )

                obs, reward, done, truncated, info = self.env.step(action)

        assert reward.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected reward shape ({self.env.get_wrapper_attr('num_envs')},), got {reward.shape}"
        assert done.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected done shape ({self.env.get_wrapper_attr('num_envs')},), got {done.shape}"
        assert truncated.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected truncated shape ({self.env.get_wrapper_attr('num_envs')},), got {truncated.shape}"
        assert obs.get("robot") is not None, "Expected 'robot' info in the info dict"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.env.close()


class TestCPU(EmbodiedEnvTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestCUDA(EmbodiedEnvTest):
    def setup_method(self):
        self.setup_simulation("cuda")
