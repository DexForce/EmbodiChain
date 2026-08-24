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

import json
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tensordict import TensorDict
from unittest.mock import MagicMock, Mock, patch

# Skip all tests if LeRobot is not available
try:
    import pandas as pd

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from embodichain.lab.gym.envs.managers.datasets import (
        LeRobotRecorder,
        LEROBOT_AVAILABLE,
    )

    from embodichain.data.enum import LeRobotKey

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    LeRobotDataset = None
    LeRobotRecorder = None
    LeRobotKey = None

# Import Camera for mocking (only if available)

try:
    from embodichain.lab.sim.sensors import Camera

    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    Camera = None

# Depth sidecar video tests require an HEVC encoder in the bundled FFmpeg build.
try:
    from embodichain.data_pipeline.depth_video import detect_depth_encoder

    _HAS_DEPTH_CODEC = detect_depth_encoder("libx265") is not None
except ImportError:
    _HAS_DEPTH_CODEC = False


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
        self,
        num_envs: int = 4,
        num_joints: int = 6,
        has_sensors: bool = True,
        step_dt: float = 1.0 / 30.0,
    ):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.step_dt = step_dt
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
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        assert recorder.lerobot_data_root == "/tmp/test_dataset"
        assert recorder.use_videos is False
        assert recorder.dataset_fps == 30
        assert mock_lerobot_dataset.create.call_args.kwargs["fps"] == 30

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_non_integer_environment_frequency_is_rejected(self, mock_lerobot_dataset):
        """LeRobot recording rejects cadence that its integer FPS cannot encode."""
        env = MockEnvForDataset(step_dt=0.03)
        cfg = MockFunctorCfg(params={"save_path": "/tmp/test_dataset"})

        with pytest.raises(ValueError, match="requires an integer dataset FPS"):
            LeRobotRecorder(cfg, env)

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_episode_duration_uses_environment_step_dt(self, mock_lerobot_dataset):
        """Episode duration follows actual environment steps, not dataset metadata."""
        env = MockEnvForDataset(has_sensors=False, step_dt=0.04)

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 3}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(params={"save_path": "/tmp/test_dataset"})
        recorder = LeRobotRecorder(cfg, env)
        recorder._convert_frame_to_lerobot = Mock(return_value={"task": "test"})

        saved = recorder._save_single_episode(
            env_id=0,
            obs_list=[object(), object(), object()],
            action_list=[object(), object(), object()],
        )

        assert saved is True
        assert recorder.total_time == pytest.approx(0.12)

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
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": True,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        assert recorder.use_videos is True

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_finalize_is_idempotent_and_does_not_commit_live_rollout(
        self, mock_lerobot_dataset, tmp_path
    ):
        """Closing finalizes storage but never turns a partial rollout into an episode."""
        env = MockEnvForDataset(num_envs=2)
        env.current_rollout_step = 3
        mock_dataset_instance = Mock()
        mock_dataset_instance.image_writer = None
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance
        cfg = MockFunctorCfg(
            params={
                "save_path": str(tmp_path),
                "robot_meta": {"robot_type": "test_robot", "control_freq": 30},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
            }
        )
        recorder = LeRobotRecorder(cfg, env)
        recorder._save_episodes = Mock()

        first_path = recorder.finalize()
        second_path = recorder.close()

        assert first_path == second_path == recorder.dataset_path
        recorder._save_episodes.assert_not_called()
        mock_dataset_instance.finalize.assert_called_once_with()
        with pytest.raises(RuntimeError, match="already finalized"):
            recorder(env, torch.tensor([0]))


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
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        # Access the private method through the instance
        features = recorder._build_features()

        assert LeRobotKey.OBS_STATE.value in features
        assert LeRobotKey.ACTION.value in features

        # Check shapes
        assert features[LeRobotKey.OBS_STATE.value]["shape"] == (6,)
        assert features[LeRobotKey.ACTION.value]["shape"] == (6,)
        assert features["subtask_index"] == {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
        assert features["annotation.segment_id"] == {
            "dtype": "int64",
            "shape": (1,),
            "names": ["segment_id"],
        }
        assert features["annotation.segment_accepted"] == {
            "dtype": "int64",
            "shape": (1,),
            "names": ["segment_accepted"],
        }
        assert "annotation.segment_attempt_id" in features
        assert "annotation.continuity_id" in features

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
                "robot_meta": {"robot_type": "test_robot"},
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

        # Check camera feature exists (use LeRobot standard key format)
        assert f"{LeRobotKey.OBS_IMAGES.value}.camera" in features

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_with_depth_and_mask(self, mock_lerobot_dataset):
        """Test that depth and mask keep their numeric shape and dtype."""
        env = MockEnvForDataset(num_joints=6)
        env.single_observation_space["sensor"]["camera"].update(
            {
                "depth": Mock(dtype=np.dtype("float32"), shape=(480, 640)),
                "depth_right": Mock(dtype=np.dtype("float32"), shape=(480, 640)),
                "mask": Mock(dtype=np.dtype("int32"), shape=(480, 640)),
                "mask_right": Mock(dtype=np.dtype("int32"), shape=(480, 640)),
            }
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

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

        assert features["observation.depth.camera"] == {
            "dtype": "float32",
            "shape": (480, 640),
            "names": ["height", "width"],
        }
        assert features["observation.mask.camera"] == {
            "dtype": "int32",
            "shape": (480, 640),
            "names": ["height", "width"],
        }
        assert features["observation.depth.camera_right"] == {
            "dtype": "float32",
            "shape": (480, 640),
            "names": ["height", "width"],
        }
        assert features["observation.mask.camera_right"] == {
            "dtype": "int32",
            "shape": (480, 640),
            "names": ["height", "width"],
        }

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_ignores_unsupported_camera_frame(
        self, mock_lerobot_dataset
    ):
        """Test that unsupported camera data does not create an invalid feature."""
        env = MockEnvForDataset(num_joints=6)
        env.single_observation_space["sensor"]["camera"]["normal"] = Mock(
            dtype=np.dtype("float32"), shape=(480, 640, 3)
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

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

        assert "observation.normal.camera" not in features

    @pytest.mark.skipif(not _HAS_DEPTH_CODEC, reason="libx265/hevc not available")
    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_excludes_depth_when_video_enabled(
        self, mock_lerobot_dataset
    ):
        """Depth is excluded from numeric features when depth video is on."""
        env = MockEnvForDataset(num_joints=6)
        env.single_observation_space["sensor"]["camera"].update(
            {
                "depth": Mock(dtype=np.dtype("float32"), shape=(32, 48)),
                "depth_right": Mock(dtype=np.dtype("float32"), shape=(32, 48)),
                "mask": Mock(dtype=np.dtype("int32"), shape=(32, 48)),
            }
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
                "depth_video": {"enable": True, "depth_min": 0.1, "depth_max": 3.0},
            }
        )

        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if isinstance(obj, MockSensor) and (
                class_or_tuple is Camera
                or (isinstance(class_or_tuple, tuple) and Camera in class_or_tuple)
            ):
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch(
            "embodichain.lab.gym.envs.managers.datasets.isinstance",
            side_effect=mock_isinstance,
        ):
            recorder = LeRobotRecorder(cfg, env)
            features = recorder._build_features()

        # Depth is offloaded to sidecar videos -> not a numeric feature.
        assert "observation.depth.camera" not in features
        assert "observation.depth.camera_right" not in features
        # Mask remains a numeric feature.
        assert features["observation.mask.camera"]["dtype"] == "int32"
        # Both depth sensors were registered for the sidecar writer.
        assert set(recorder._depth_sensor_specs) == {"camera", "camera_right"}

    @pytest.mark.skipif(not _HAS_DEPTH_CODEC, reason="libx265/hevc not available")
    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_build_features_keeps_numeric_depth_with_fallback(
        self, mock_lerobot_dataset
    ):
        """keep_numeric_fallback retains depth as a numeric feature too."""
        env = MockEnvForDataset(num_joints=6)
        env.single_observation_space["sensor"]["camera"].update(
            {
                "depth": Mock(dtype=np.dtype("float32"), shape=(32, 48)),
                "mask": Mock(dtype=np.dtype("int32"), shape=(32, 48)),
            }
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
                "depth_video": {
                    "enable": True,
                    "depth_min": 0.1,
                    "depth_max": 3.0,
                    "keep_numeric_fallback": True,
                },
            }
        )

        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if isinstance(obj, MockSensor) and (
                class_or_tuple is Camera
                or (isinstance(class_or_tuple, tuple) and Camera in class_or_tuple)
            ):
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch(
            "embodichain.lab.gym.envs.managers.datasets.isinstance",
            side_effect=mock_isinstance,
        ):
            recorder = LeRobotRecorder(cfg, env)
            features = recorder._build_features()

        # Depth is both a numeric feature AND registered for the sidecar writer.
        assert features["observation.depth.camera"]["dtype"] == "float32"
        assert "camera" in recorder._depth_sensor_specs


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
class TestLeRobotRecorderDepthSidecar:
    """Integration tests for depth sidecar video writing during save."""

    @pytest.mark.skipif(not _HAS_DEPTH_CODEC, reason="libx265/hevc not available")
    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_save_episode_writes_depth_sidecar(self, mock_lerobot_dataset, tmp_path):
        """A real episode writes depth sidecar MP4s and metadata."""
        env = MockEnvForDataset(num_joints=6)
        env._sensors["camera"].cfg.height = 32
        env._sensors["camera"].cfg.width = 48
        env.single_observation_space["sensor"]["camera"].update(
            {
                "color": Mock(dtype=np.dtype("uint8"), shape=(32, 48, 4)),
                "depth": Mock(dtype=np.dtype("float32"), shape=(32, 48)),
                "mask": Mock(dtype=np.dtype("int32"), shape=(32, 48)),
            }
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": str(tmp_path),
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
                "depth_video": {"enable": True, "depth_min": 0.1, "depth_max": 3.0},
            }
        )

        original_isinstance = isinstance

        def mock_isinstance(obj, class_or_tuple):
            if isinstance(obj, MockSensor) and (
                class_or_tuple is Camera
                or (isinstance(class_or_tuple, tuple) and Camera in class_or_tuple)
            ):
                return True
            return original_isinstance(obj, class_or_tuple)

        with patch(
            "embodichain.lab.gym.envs.managers.datasets.isinstance",
            side_effect=mock_isinstance,
        ):
            recorder = LeRobotRecorder(cfg, env)

            depth = torch.linspace(0.2, 2.8, 32 * 48, dtype=torch.float32).reshape(
                32, 48
            )
            obs_list = []
            for i in range(3):
                obs_list.append(
                    TensorDict(
                        {
                            "robot": {
                                "qpos": torch.zeros(6),
                                "qvel": torch.zeros(6),
                                "qf": torch.zeros(6),
                            },
                            "sensor": {
                                "camera": {
                                    "color": torch.zeros(32, 48, 4, dtype=torch.uint8),
                                    "depth": depth + 0.01 * i,
                                    "mask": torch.zeros(32, 48, dtype=torch.int32),
                                }
                            },
                        },
                        batch_size=[],
                    )
                )
            action_list = [torch.zeros(6) for _ in range(3)]

            ok = recorder._save_single_episode(0, obs_list, action_list)
            assert ok

        # The depth sidecar video and metadata were written.
        ds_root = recorder.dataset_full_path
        assert (ds_root / "depth_videos" / "camera" / "episode_000000.mp4").exists()
        assert (ds_root / "depth_meta.json").exists()

        # LeRobot's add_frame never received a depth key (RGB-only pipeline).
        for call in mock_dataset_instance.add_frame.call_args_list:
            frame = call.args[0]
            assert not any(k.startswith("observation.depth.") for k in frame)


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
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

        recorder = LeRobotRecorder(cfg, env)

        # Create mock observation
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

        frame = recorder._convert_frame_to_lerobot(
            obs,
            action,
            "test_task",
            annotations={
                "episode_step": 4,
                "segment_id": 2,
                "segment_step": 1,
                "segment_start": False,
                "segment_end": True,
                "terminated": False,
                "truncated": False,
            },
            subtask_index=3,
        )

        assert "task" in frame
        assert frame["task"] == "test_task"
        assert frame["subtask_index"].tolist() == [3]
        assert LeRobotKey.OBS_STATE.value in frame
        assert LeRobotKey.ACTION.value in frame
        assert frame["annotation.segment_id"].tolist() == [2]
        assert frame["annotation.segment_end"].tolist() == [1]

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_convert_frame_with_depth_and_mask(self, mock_lerobot_dataset):
        """Test that camera auxiliary frames are added under numeric feature keys."""
        env = MockEnvForDataset(num_joints=6)
        env.single_observation_space["sensor"]["camera"].update(
            {
                "depth": Mock(dtype=np.dtype("float32"), shape=(480, 640)),
                "mask": Mock(dtype=np.dtype("int32"), shape=(480, 640)),
            }
        )

        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        cfg = MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "test task"},
                "extra": {"task_description": "test"},
                "use_videos": False,
            }
        )

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
            depth = torch.arange(480 * 640, dtype=torch.float32).reshape(480, 640)
            mask = torch.arange(480 * 640, dtype=torch.int32).reshape(480, 640)
            obs = TensorDict(
                {
                    "robot": {
                        "qpos": torch.zeros(6),
                        "qvel": torch.zeros(6),
                        "qf": torch.zeros(6),
                    },
                    "sensor": {
                        "camera": {
                            "color": torch.zeros(480, 640, 4, dtype=torch.uint8),
                            "depth": depth,
                            "mask": mask,
                        }
                    },
                },
                batch_size=[],
            )
            frame = recorder._convert_frame_to_lerobot(obs, torch.zeros(6), "test_task")

        assert torch.equal(frame["observation.depth.camera"], depth)
        assert torch.equal(frame["observation.mask.camera"], mask)
        assert frame["observation.depth.camera"].dtype == torch.float32
        assert frame["observation.mask.camera"].dtype == torch.int32


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_episode_metadata_sidecar_appends_json_lines(tmp_path) -> None:
    """The EmbodiChain sidecar is valid append-only JSONL."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder.dataset_full_path = tmp_path
    recorder._metadata_lock = threading.Lock()

    recorder._write_episode_metadata(
        {"episode_index": 1, "segments": [{"name": "pick"}]}
    )
    recorder._write_episode_metadata(
        {"episode_index": 2, "segments": [{"name": "place"}]}
    )

    metadata_path = tmp_path / "meta" / "embodichain_episodes.jsonl"
    records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert [record["episode_index"] for record in records] == [1, 2]
    assert [record["segments"][0]["name"] for record in records] == ["pick", "place"]


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_segment_fragment_payloads_are_independent_and_keep_provenance() -> None:
    """Only accepted natural segments are sliced by default."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    obs = TensorDict(
        {"state": torch.arange(5, dtype=torch.float32).unsqueeze(-1)},
        batch_size=[5],
    )
    actions = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
    annotations = {
        "valid": torch.ones(5, dtype=torch.bool),
        "episode_step": torch.arange(5),
        "segment_id": torch.tensor([0, 0, 1, 1, 1]),
        "segment_step": torch.tensor([0, 1, 0, 1, 2]),
        "segment_start": torch.tensor([True, False, True, False, False]),
        "segment_end": torch.tensor([False, True, False, False, True]),
        "segment_accepted": torch.tensor([True, True, False, False, False]),
        "segment_attempt_id": torch.full((5,), 2),
        "continuity_id": torch.zeros(5, dtype=torch.long),
        "terminated": torch.zeros(5, dtype=torch.bool),
        "truncated": torch.zeros(5, dtype=torch.bool),
    }
    metadata = {
        "output_mode": "segment_fragments",
        "save_failed_fragments": False,
        "episode_index": 9,
        "attempt_id": 2,
        "program_run_id": "9:2",
        "terminated": False,
        "truncated": True,
        "segments": [
            {
                "segment_id": 0,
                "start_step": 0,
                "end_step": 2,
                "success": True,
                "instruction": "Pick the cube",
                "attempt_id": 2,
                "continuity_id": 0,
                "outcome_kind": "succeeded",
                "metadata": {
                    "expert_program_id": "repeat_pick_place",
                    "program_segment_id": "pick_0",
                },
            },
            {
                "segment_id": 1,
                "start_step": 2,
                "end_step": 5,
                "success": False,
                "failure_reason": "segment_validation_failed",
                "outcome_kind": "validation_failed",
                "instruction": "Place the cube",
                "attempt_id": 2,
                "continuity_id": 0,
                "metadata": {},
            },
        ],
    }

    payloads = list(recorder._episode_payloads(0, obs, actions, annotations, metadata))

    assert len(payloads) == 1
    _, fragment_obs, fragment_actions, fragment_annotations, fragment_metadata = (
        payloads[0]
    )
    assert fragment_obs.batch_size == torch.Size([2])
    assert fragment_actions.shape == (2, 1)
    assert fragment_annotations["episode_step"].tolist() == [0, 1]
    assert fragment_annotations["segment_start"].tolist() == [True, False]
    assert fragment_annotations["segment_end"].tolist() == [False, True]
    assert fragment_annotations["segment_accepted"].all()
    assert fragment_metadata["fragment_origin"] == "natural_segment"
    assert fragment_metadata["source_program_id"] == "repeat_pick_place"
    assert fragment_metadata["program_segment_id"] == "pick_0"
    assert fragment_metadata["source_start_step"] == 0
    assert fragment_metadata["source_end_step"] == 2
    assert fragment_metadata["terminated"] is False
    assert fragment_metadata["truncated"] is False
    assert fragment_metadata["segments"][0]["start_step"] == 0
    assert fragment_metadata["segments"][0]["end_step"] == 2


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_failed_segment_fragment_requires_explicit_opt_in() -> None:
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    obs = TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2])
    actions = torch.zeros(2, 1)
    annotations = {
        "segment_id": torch.zeros(2, dtype=torch.long),
        "segment_accepted": torch.zeros(2, dtype=torch.bool),
    }
    metadata = {
        "output_mode": "segment_fragments",
        "save_failed_fragments": True,
        "segments": [
            {
                "segment_id": 0,
                "start_step": 0,
                "end_step": 2,
                "success": False,
                "outcome_kind": "runtime_failed",
                "metadata": {},
            }
        ],
    }

    payloads = list(recorder._episode_payloads(0, obs, actions, annotations, metadata))

    assert len(payloads) == 1
    assert not payloads[0][3]["segment_accepted"].any()
    assert payloads[0][4]["success"] is False


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_subtask_registry_writes_stable_deduplicated_indices(tmp_path) -> None:
    """Repeated descriptions retain one stable row in subtasks.parquet."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder.dataset_full_path = tmp_path
    recorder.dataset = MagicMock()
    recorder._metadata_lock = threading.Lock()
    recorder._subtask_to_index = {}

    first_indices = recorder._register_subtasks(
        ["pick cube", "place cube", "pick cube"]
    )
    second_indices = recorder._register_subtasks([" place cube ", "release cube"])

    assert first_indices == {"pick cube": 0, "place cube": 1}
    assert second_indices == {"place cube": 1, "release cube": 2}

    subtasks = pd.read_parquet(tmp_path / "meta" / "subtasks.parquet")
    assert subtasks.index.tolist() == ["pick cube", "place cube", "release cube"]
    assert subtasks["subtask_index"].tolist() == [0, 1, 2]


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_multisegment_episode_round_trips_task_and_subtasks(tmp_path) -> None:
    """LeRobot 0.4.4 reloads one overall task and per-frame subtasks."""
    env = MockEnvForDataset(num_joints=2, has_sensors=False)
    cfg = MockFunctorCfg(
        params={
            "save_path": str(tmp_path),
            "robot_meta": {"robot_type": "test_robot", "control_freq": 10},
            "instruction": {"lang": "Move the cube between two targets."},
            "extra": {"task_description": "multisegment_round_trip"},
            "use_videos": False,
        }
    )
    recorder = LeRobotRecorder(cfg, env)
    obs_list = [
        TensorDict(
            {
                "robot": {
                    "qpos": torch.full((2,), frame_index, dtype=torch.float32),
                    "qvel": torch.zeros(2),
                    "qf": torch.zeros(2),
                }
            },
            batch_size=[],
        )
        for frame_index in range(4)
    ]
    action_list = [torch.zeros(2) for _ in obs_list]
    annotations = {
        "segment_id": torch.tensor([0, 0, 1, 1]),
        "segment_step": torch.tensor([0, 1, 0, 1]),
        "segment_start": torch.tensor([1, 0, 1, 0]),
        "segment_end": torch.tensor([0, 1, 0, 1]),
    }
    episode_metadata = {
        "segments": [
            {
                "segment_id": 0,
                "start_step": 0,
                "end_step": 2,
                "instruction": "Pick up the cube.",
            },
            {
                "segment_id": 1,
                "start_step": 2,
                "end_step": 4,
                "instruction": "Place the cube at the next target.",
            },
        ]
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=("Conversion of an array with ndim > 0 to a scalar is deprecated"),
            category=DeprecationWarning,
        )
        assert recorder._save_single_episode(
            0,
            obs_list,
            action_list,
            annotations=annotations,
            episode_metadata=episode_metadata,
        )
    recorder.finalize()

    loaded = LeRobotDataset(
        repo_id=recorder.dataset_full_path.name,
        root=recorder.dataset_full_path,
    )
    samples = [loaded[index] for index in range(len(obs_list))]

    assert {sample["task"] for sample in samples} == {
        "Move the cube between two targets."
    }
    assert [sample["subtask"] for sample in samples] == [
        "Pick up the cube.",
        "Pick up the cube.",
        "Place the cube at the next target.",
        "Place the cube at the next target.",
    ]
    assert [sample["subtask_index"].item() for sample in samples] == [0, 0, 1, 1]
    assert [sample["annotation.segment_id"].item() for sample in samples] == [
        0,
        0,
        1,
        1,
    ]


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_post_commit_metadata_failure_does_not_reuse_episode_index() -> None:
    """A sidecar failure after LeRobot commit advances the global index first."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = MockEnvForDataset(has_sensors=False)
    recorder.instruction = None
    recorder.extra = {}
    recorder.total_time = 0.0
    recorder.curr_episode = 0
    recorder.dataset_full_path = Path("/tmp/test_dataset")
    recorder.dataset = MagicMock()
    recorder.dataset.meta.info = {"fps": 30}
    recorder._depth_manager = None
    recorder._register_subtasks = MagicMock(return_value={"unknown_task": 0})
    recorder._convert_frame_to_lerobot = MagicMock(return_value={})
    recorder._write_episode_metadata = MagicMock(side_effect=OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        recorder._save_single_episode(0, [object()], [object()])

    recorder.dataset.save_episode.assert_called_once_with()
    assert recorder.curr_episode == 1


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_missing_dense_annotations_fall_back_to_segment_sidecar_outcome() -> None:
    """Legacy buffers do not silently label a retained failed segment accepted."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = MockEnvForDataset(has_sensors=False)
    recorder.instruction = None
    recorder.extra = {}
    recorder.total_time = 0.0
    recorder.curr_episode = 0
    recorder.dataset_full_path = Path("/tmp/test_dataset")
    recorder.dataset = MagicMock()
    recorder.dataset.meta.info = {"fps": 30}
    recorder._depth_manager = None
    recorder._register_subtasks = MagicMock(return_value={"unknown_task": 0})
    recorder._convert_frame_to_lerobot = MagicMock(return_value={})
    recorder._write_episode_metadata = MagicMock()

    assert recorder._save_single_episode(
        0,
        [object()],
        [object()],
        episode_metadata={
            "attempt_id": 4,
            "continuity_id": 2,
            "segments": [
                {
                    "start_step": 0,
                    "end_step": 1,
                    "success": False,
                    "failure_reason": "segment_validation_failed",
                }
            ],
        },
    )

    frame_annotations = recorder._convert_frame_to_lerobot.call_args.kwargs[
        "annotations"
    ]
    assert frame_annotations["segment_accepted"] is False
    assert frame_annotations["segment_attempt_id"] == 4
    assert frame_annotations["continuity_id"] == 2


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_save_episodes_skips_empty_rollout() -> None:
    """An initial reset with no recorded frames is not a failed commit."""
    env = Mock()
    env.rollout_steps = torch.zeros(1, dtype=torch.long)
    env.rollout_buffer = MagicMock()

    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    recorder._save_single_episode = Mock()

    recorder._save_episodes(torch.tensor([0]))

    recorder._save_single_episode.assert_not_called()
    env.rollout_buffer.__getitem__.assert_not_called()


class TestDatasetFunctorCfg:
    """Tests for dataset functor configuration."""

    def test_functor_cfg_import(self):
        """Test that FunctorCfg can be imported."""
        from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

        # Should be able to instantiate
        cfg = DatasetFunctorCfg()
        assert cfg is not None
