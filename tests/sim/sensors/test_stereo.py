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
# ----------------------------------------------------------------------------,

import pytest
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors import StereoCamera, SensorCfg


NUM_ENVS = 4


class StereoCameraTest:
    def setup_simulation(self, sim_device, enable_rt):
        # Setup SimulationManager
        config = SimulationManagerCfg(
            headless=True, sim_device=sim_device, enable_rt=enable_rt, num_envs=NUM_ENVS
        )
        self.sim = SimulationManager(config)
        # Create batch of cameras
        cfg_dict = {
            "sensor_type": "StereoCamera",
            "width": 640,
            "height": 480,
            "enable_mask": True,
            "enable_depth": True,
            "enable_normal": True,
            "enable_position": True,
            "enable_disparity": True,
            "left_to_right_pos": (0.1, 0.0, 0.0),
        }
        cfg = SensorCfg.from_dict(cfg_dict)
        self.camera: StereoCamera = self.sim.add_sensor(cfg)

    def test_get_data(self):

        self.camera.update()

        # Get data from the camera
        data = self.camera.get_data()

        # Check if data is a dictionary
        assert isinstance(data, dict), "Camera data should be a dictionary"

        # Check if all expected keys are present
        for key in self.camera.SUPPORTED_DATA_TYPES:
            assert key in data, f"Missing key in camera data: {key}"

        # Check if the data shape matches the expected shape
        assert data["color"].shape == (NUM_ENVS, 480, 640, 4), "RGB data shape mismatch"
        assert data["depth"].shape == (
            NUM_ENVS,
            480,
            640,
            1,
        ), "Depth data shape mismatch"
        assert data["normal"].shape == (
            NUM_ENVS,
            480,
            640,
            3,
        ), "Normal data shape mismatch"
        assert data["position"].shape == (
            NUM_ENVS,
            480,
            640,
            3,
        ), "Position data shape mismatch"
        assert data["mask"].shape == (NUM_ENVS, 480, 640, 1), "Mask data shape mismatch"
        assert data["disparity"].shape == (
            NUM_ENVS,
            480,
            640,
            1,
        ), "Disparity data shape mismatch"

        # Check if the data types are correct
        assert data["color"].dtype == torch.uint8, "Color data type mismatch"
        assert data["depth"].dtype == torch.float32, "Depth data type mismatch"
        assert data["normal"].dtype == torch.float32, "Normal data type mismatch"
        assert data["position"].dtype == torch.float32, "Position data type mismatch"
        assert data["mask"].dtype == torch.int32, "Mask data type mismatch"
        assert data["disparity"].dtype == torch.float32, "Disparity data type mismatch"

    def test_local_pose_with_env_ids(self):
        env_ids = [0, 1, 2]

        pose = (
            torch.eye(4, device=self.sim.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )
        pose[:, 2, 3] = 2.0

        self.camera.set_local_pose(pose, env_ids=env_ids)

        # Verify the local pose for specified env_ids
        assert torch.allclose(self.camera.get_local_pose(to_matrix=True)[env_ids], pose)

    def test_set_intrinsics(self):
        # Define new intrinsic parameters
        new_intrinsics = (
            torch.tensor(
                [500.0, 500.0, 320.0, 240.0],
                device=self.sim.device,
            )
            .unsqueeze(0)
            .repeat(NUM_ENVS, 1)
        )

        # Set new intrinsic parameters for all environments
        self.camera.set_intrinsics(new_intrinsics)

        right_intrinsics = (
            torch.tensor(
                [520.0, 520.0, 315.0, 235.0],
                device=self.sim.device,
            )
            .unsqueeze(0)
            .repeat(NUM_ENVS, 1)
        )

        self.camera.set_intrinsics(new_intrinsics, right_intrinsics=right_intrinsics)

        new_intrinsics = torch.tensor(
            [500.0, 500.0, 320.0, 240.0],
            device=self.sim.device,
        )
        self.camera.set_intrinsics(new_intrinsics)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestStereoCameraRaster(StereoCameraTest):
    def setup_method(self):
        self.setup_simulation("cpu", enable_rt=False)


class TestStereoCameraRaster(StereoCameraTest):
    def setup_method(self):
        self.setup_simulation("cuda", enable_rt=False)


class TestStereoCameraFastRT(StereoCameraTest):
    def setup_method(self):
        self.setup_simulation("cpu", enable_rt=True)


class TestStereoCameraFastRT(StereoCameraTest):
    def setup_method(self):
        self.setup_simulation("cuda", enable_rt=True)
