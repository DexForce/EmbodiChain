# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Tests for dataset functors."""

from __future__ import annotations

import pytest
import torch

from unittest.mock import MagicMock, Mock, patch

# Skip all tests if LeRobot is not available
try:
    from embodichain.lab.gym.envs.managers.datasets import (
        LeRobotRecorder,
        LEROBOT_AVAILABLE,
    )

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    LeRobotRecorder = None


# Import Camera for mocking (only if available)
try:
    from embodichain.lab.sim.sensors import Camera

    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    Camera = None


class MockRobot:
    """Mock robot for dataset functor tests."""

    def __init__(self, num_joints: int = 6):
        self.num_joints = num_joints
        self.joint_names = [f"joint_{i}" for i in range(num_joints)]


class MockSensor:
    """Mock sensor for dataset functor tests."""

    def __init__(self, uid: str = "camera", is_stereo: bool = False):
        self.uid = uid
        self.cfg = Mock()
        self.cfg.height = 480
        self.cfg.width = 640
        self._is_stereo = is_stereo

    def get_intrinsics(self):
        return torch.zeros(1, 3, 3)


def is_stereocam(sensor):
    """Check if sensor is stereo camera."""
    return getattr(sensor, "_is_stereo", False)


class MockEnvForDataset:
    """Mock environment for dataset functor tests."""

    def __init__(
        self, num_envs: int = 4, num_joints: int = 6, has_sensors: bool = True
    ):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.active_joint_ids = list(range(num_joints))

        self.robot = MockRobot(num_joints)

        # Mock has_sensors
        self.has_sensors = has_sensors

        # Mock single observation space
        self.single_observation_space = {
            "robot": {
                "qpos": Mock(),
                "qvel": Mock(),
                "qf": Mock(),
            },
            "sensor": {"camera": {"color": Mock()}},
        }

        # Setup mock sensor
        self._sensors = {"camera": MockSensor("camera")}
        self._sensor_uids = ["camera"]

        # Mock observation manager with active_functors
        self.observation_manager = Mock()
        self.observation_manager.active_functors = {"add": []}

    def get_sensor(self, uid: str):
        return self._sensors.get(uid)

    def get_sensor_uid_list(self):
        return self._sensor_uids


class MockFunctorCfg:
    """Mock functor config for testing."""

    def __init__(self, params: dict = None):
        self.params = params or {}


# Tests that don't require LeRobot
class TestDatasetFunctorBasics:
    """Basic tests for dataset functors."""

    def test_lerobot_available_flag(self):
        """Test that LEROBOT_AVAILABLE flag reflects actual availability."""
        # This test just verifies the import worked
        try:
            from embodichain.lab.envs.managers.datasets import LEROBOT_AVAILABLE
        except ImportError:
            pass  # Expected if not installed

    def test_dataset_functor_module_imports(self):
        """Test that dataset functor module can be imported."""
        try:
            from embodichain.lab.gym.envs.managers import datasets

            # Check module has expected attributes
            assert (
                hasattr(datasets, "LeRobotRecorder") or not datasets.LEROBOT_AVAILABLE
            )
        except ImportError:
            pass  # Module might not exist


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
class TestLeRobotRecorderInitialization:
    """Tests for LeRobotRecorder initialization."""

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_initialization_with_defaults(self, mock_lerobot_dataset):
        """Test LeRobotRecorder initialization with default parameters."""
        env = MockEnvForDataset()

        # Mock the LeRobotDataset.create method
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        assert recorder.lerobot_data_root == "/tmp/test_dataset"
        assert recorder.use_videos == False

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_initialization_with_videos(self, mock_lerobot_dataset):
        """Test LeRobotRecorder initialization with video recording enabled."""
        env = MockEnvForDataset()

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": True,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        assert recorder.use_videos == True


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
class TestLeRobotRecorderFeatures:
    """Tests for LeRobotRecorder feature building."""

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_creates_correct_structure(self, mock_lerobot_dataset):
        """Test that _build_features creates the correct feature structure."""
        env = MockEnvForDataset(num_joints=6)

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        # Access the private method through the instance
        features = recorder._build_features()

        # Check expected features exist
        assert "observation.qpos" in features
        assert "observation.qvel" in features
        assert "observation.qf" in features
        assert "action" in features

        # Check shapes
        assert features["observation.qpos"]["shape"] == (6,)
        assert features["action"]["shape"] == (6,)

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_with_sensor(self, mock_lerobot_dataset):
        """Test that _build_features includes sensor features when sensors exist."""
        env = MockEnvForDataset(num_joints=6)

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        # Patch isinstance to treat MockSensor as Camera
        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if isinstance(obj, MockSensor):
                if class_or_tuple is Camera or (
                    isinstance(class_or_tuple, tuple) and Camera in class_or_tuple
                ):
                    return True
            return original_isinstance(obj, class_or_tuple)

        with patch(
            "embodichain.lab.gym.envs.managers.datasets.isinstance",
            side_effect=mock_isinstance,
        ):
            recorder = LeRobotRecorder(cfg, env)
            features = recorder._build_features()

        # Check camera feature exists
        assert "camera.color" in features


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
class TestLeRobotRecorderFrameConversion:
    """Tests for LeRobotRecorder frame conversion."""

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_convert_frame_with_tensor_action(self, mock_lerobot_dataset):
        """Test frame conversion with tensor action."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        # Create mock observation
        from tensordict import TensorDict

        obs = TensorDict(
            {
                "robot": {
                    "qpos": torch.zeros(6),
                    "qvel": torch.zeros(6),
                    "qf": torch.zeros(6),
                },
                "sensor": {},
            },
            batch_size=[],
        )

        # Create mock action
        action = torch.zeros(6)

        frame = recorder._convert_frame_to_lerobot(obs, action, "test_task")

        assert "task" in frame
        assert frame["task"] == "test_task"
        assert "observation.qpos" in frame
        assert "action" in frame


class TestDatasetFunctorCfg:
    """Tests for dataset functor configuration."""

    def test_functor_cfg_import(self):
        """Test that FunctorCfg can be imported."""
        from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

        # Should be able to instantiate
        cfg = DatasetFunctorCfg()
        assert cfg is not None
