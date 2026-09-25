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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensordict import TensorDict
from unittest.mock import MagicMock, Mock, patch

from embodichain.lab.gym.envs.expert_trajectory import build_expert_action_spec
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.action_types import (
    ActionDescriptor,
    ActionTermDescriptor,
)

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


def policy_descriptors(
    arm_representation: str = "eef_pose",
) -> tuple[ActionDescriptor, ActionDescriptor]:
    """Build one canonical arm-plus-parallel-gripper flat layout."""
    if arm_representation == "eef_pose":
        arm_names = ("x", "y", "z", "roll", "pitch", "yaw")
        arm_units = ("m", "m", "m", "rad", "rad", "rad")
    else:
        arm_names = tuple(f"joint_{index}" for index in range(7))
        arm_units = ("rad",) * 7
    arm = ActionDescriptor(
        "arm_action",
        0,
        len(arm_names),
        ActionTermDescriptor(
            arm_representation,
            len(arm_names),
            arm_names,
            arm_units,
            None,
            tuple(f"joint_{index}" for index in range(7)),
            {"frame": "arena"} if arm_representation == "eef_pose" else {},
        ),
    )
    gripper = ActionDescriptor(
        "gripper_action",
        len(arm_names),
        len(arm_names) + 1,
        ActionTermDescriptor(
            "parallel_gripper",
            1,
            ("gripper",),
            ("normalized",),
            "minus_one_to_one",
            ("finger_joint",),
            {},
        ),
    )
    return arm, gripper


def attach_eef_action_manager(env: MockEnvForDataset, scale: float = 1.0) -> None:
    """Attach the canonical descriptor-driven EEF action producer."""
    arm_term = SimpleNamespace(
        _scale=torch.tensor(scale),
        part_name="arm",
        controlled_joint_ids=tuple(range(6)),
    )
    gripper_term = SimpleNamespace(
        command_mode="continuous",
        controlled_joint_ids=(6,),
        lower_command=torch.tensor([0.0]),
        upper_command=torch.tensor([0.04]),
    )
    terms = {"arm_action": arm_term, "gripper_action": gripper_term}
    env.action_manager = SimpleNamespace(
        descriptors=policy_descriptors(),
        get_term=terms.__getitem__,
    )


def test_recorder_rejects_missing_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional dependencies fail explicitly before recorder setup."""
    from embodichain.lab.gym.envs.managers import datasets

    monkeypatch.setattr(datasets, "LEROBOT_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="LeRobot is not installed"):
        datasets.LeRobotRecorder(MockFunctorCfg(), MockEnvForDataset())


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
    def test_eef_gripper_contract_uses_descriptor_order(
        self, mock_lerobot_dataset
    ) -> None:
        """Policy feature width, names, and slices come from the manager."""
        env = MockEnvForDataset(num_joints=7, has_sensors=False)
        attach_eef_action_manager(env)
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "action_contract": {
                        "version": 1,
                        "representation": "eef_pose_parallel_gripper",
                    },
                }
            ),
            env,
        )

        feature = recorder._build_features()[LeRobotKey.ACTION.value]
        assert feature["shape"] == (7,)
        assert feature["names"] == [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper",
        ]
        assert feature["info"]["embodichain.action_terms"][1]["slice"] == [6, 7]

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_joint_gripper_contract_uses_descriptor_order(
        self, mock_lerobot_dataset
    ) -> None:
        """Joint-arm policy contracts use their manager-owned eight dimensions."""
        env = MockEnvForDataset(num_joints=8, has_sensors=False)
        env.action_manager = SimpleNamespace(
            descriptors=policy_descriptors("joint_position")
        )
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "action_contract": {
                        "version": 1,
                        "representation": "joint_position_parallel_gripper",
                    },
                }
            ),
            env,
        )

        feature = recorder._build_features()[LeRobotKey.ACTION.value]
        assert feature["shape"] == (8,)
        assert feature["names"] == [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    @pytest.mark.parametrize(
        ("descriptors", "message"),
        [
            (
                (
                    policy_descriptors()[0],
                    ActionDescriptor(
                        "gripper_action", 7, 8, policy_descriptors()[1].term
                    ),
                ),
                "contiguous",
            ),
            (policy_descriptors("joint_position"), "expected term sequence"),
        ],
    )
    def test_policy_contract_rejects_descriptor_mismatch(
        self, mock_lerobot_dataset, descriptors, message: str
    ) -> None:
        """Policy contracts reject descriptor gaps and wrong term sequences."""
        env = MockEnvForDataset(num_joints=7, has_sensors=False)
        env.action_manager = SimpleNamespace(
            descriptors=descriptors,
            get_term=lambda name: SimpleNamespace(_scale=torch.tensor(1.0)),
        )

        with pytest.raises(ValueError, match=message):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "eef_pose_parallel_gripper",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

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
        assert "info" not in features[LeRobotKey.ACTION.value]
        assert "action.controller_qpos" not in features
        assert "controller.qpos" not in features
        assert "observation.eef_pose" not in features
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
        assert set(features) == {
            LeRobotKey.OBS_STATE.value,
            LeRobotKey.OBS_QVEL.value,
            LeRobotKey.OBS_QF.value,
            LeRobotKey.ACTION.value,
            "subtask_index",
            "annotation.episode_step",
            "annotation.segment_id",
            "annotation.segment_step",
            "annotation.segment_start",
            "annotation.segment_end",
            "annotation.segment_accepted",
            "annotation.segment_attempt_id",
            "annotation.continuity_id",
            "annotation.terminated",
            "annotation.truncated",
        }

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_eef_action_uses_requested_target_and_declares_contract(
        self, mock_lerobot_dataset
    ):
        """EEF mode records the requested target instead of measured FK pose."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)
        attach_eef_action_manager(env)
        env.expert_action_spec = build_expert_action_spec(
            joint_names=[f"joint_{index}" for index in range(6)],
            joint_command_mode="position",
        )
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance
        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "action_contract": {
                        "version": 1,
                        "representation": "eef_pose_parallel_gripper",
                        "record_eef_observation": False,
                    },
                }
            ),
            env,
        )

        features = recorder._build_features()
        assert features[LeRobotKey.ACTION.value]["shape"] == (7,)
        contract = recorder._action_contract()
        assert contract["requested_target"] == "action"
        assert contract["measured_observation"] is None
        assert "executed_command" not in contract
        assert features[LeRobotKey.ACTION.value]["info"] == {
            "embodichain.action_contract": contract,
            "embodichain.action_terms": [
                descriptor.to_dict() for descriptor in policy_descriptors()
            ],
        }
        assert not any("controller_qpos" in key for key in features)
        assert "subtask_index" not in features

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
        requested = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
        frame = recorder._convert_frame_to_lerobot(
            obs, requested, "test_task", env_id=0
        )
        torch.testing.assert_close(frame[LeRobotKey.ACTION.value], requested)
        assert not any("controller_qpos" in key for key in frame)
        assert "subtask_index" not in frame

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_eef_action_contract_declares_explicit_action_and_observation(
        self, mock_lerobot_dataset
    ):
        """EEF mode exposes a 7D command and optional measured pose feature."""
        env = MockEnvForDataset(num_joints=6)
        attach_eef_action_manager(env)
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance

        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "robot_meta": {"robot_type": "test_robot"},
                    "instruction": {"lang": "test task"},
                    "action_contract": {
                        "version": 1,
                        "representation": "eef_pose_parallel_gripper",
                        "record_eef_observation": True,
                    },
                }
            ),
            env,
        )

        features = recorder._build_features()
        assert features[LeRobotKey.ACTION.value]["shape"] == (7,)
        assert "observation.eef_pose" in features

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_action_contract_rejects_unknown_version(self, mock_lerobot_dataset):
        """Only the supported action-contract version can change the schema."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)

        with pytest.raises(ValueError, match="action_contract version must be 1"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 2,
                            "representation": "joint_position",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    @pytest.mark.parametrize("legacy_key", ["action_mode", "record_eef_observation"])
    def test_action_contract_rejects_superseded_top_level_options(
        self, mock_lerobot_dataset, legacy_key: str
    ):
        """Removed PR-only options cannot silently fall back to the main schema."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)

        with pytest.raises(
            ValueError, match="must be configured inside action_contract"
        ):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={"save_path": "/tmp/test_dataset", legacy_key: True}
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_action_contract_rejects_unknown_representation(self, mock_lerobot_dataset):
        """Unknown action encodings fail before a dataset is created."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)

        with pytest.raises(ValueError, match="unsupported action representation"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "quaternion_pose",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    @pytest.mark.parametrize(
        "unknown_field", ["record_controller_pose", "record_controller_qpos"]
    )
    def test_action_contract_rejects_unknown_fields(
        self, mock_lerobot_dataset, unknown_field: str
    ):
        """Misspelled contract settings fail at the recorder boundary."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)

        with pytest.raises(ValueError, match="unknown action_contract fields"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "joint_position",
                            unknown_field: True,
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("version", True, "version must be 1"),
            (
                "record_eef_observation",
                "false",
                "record_eef_observation must be a bool",
            ),
        ],
    )
    def test_action_contract_rejects_wrong_field_types(
        self,
        mock_lerobot_dataset,
        field: str,
        value: object,
        message: str,
    ):
        """Contract fields do not coerce ambiguous serialized values."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)
        contract = {
            "version": 1,
            "representation": "joint_position",
            field: value,
        }

        with pytest.raises((TypeError, ValueError), match=message):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": contract,
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_eef_action_contract_rejects_incompatible_action_term(
        self, mock_lerobot_dataset
    ):
        """EEF recording fails before creation when the controller encoding differs."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)
        env.action_manager = SimpleNamespace(descriptors=())

        with pytest.raises(ValueError, match="requires ActionManager descriptors"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "eef_pose_parallel_gripper",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_eef_action_contract_rejects_missing_action_manager(
        self, mock_lerobot_dataset
    ):
        """EEF recording requires an explicit compatible policy controller."""
        env = MockEnvForDataset(num_joints=7, has_sensors=False)

        with pytest.raises(ValueError, match="requires a configured ActionManager"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "eef_pose_parallel_gripper",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_eef_action_contract_rejects_scaled_policy_commands(
        self, mock_lerobot_dataset
    ):
        """Recorded absolute EEF targets cannot differ from controller targets."""
        env = MockEnvForDataset(num_joints=6, has_sensors=False)
        attach_eef_action_manager(env, scale=0.5)

        with pytest.raises(ValueError, match="requires EEF action scale 1.0"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "eef_pose_parallel_gripper",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_position_velocity_action_feature_uses_canonical_layout(
        self, mock_lerobot_dataset
    ):
        env = MockEnvForDataset(num_joints=2, has_sensors=False)
        env.expert_action_spec = build_expert_action_spec(
            joint_names=["joint_0", "joint_1"],
            joint_command_mode="position_velocity",
        )
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance
        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "robot_meta": {"robot_type": "test_robot"},
                    "instruction": {"lang": "test task"},
                    "extra": {"task_description": "test"},
                    "use_videos": False,
                    "action_contract": {
                        "version": 1,
                        "representation": "joint_position_velocity",
                    },
                }
            ),
            env,
        )

        action_feature = recorder._build_features()[LeRobotKey.ACTION.value]

        assert action_feature["shape"] == (4,)
        assert action_feature["names"] == [
            "joint_0.position",
            "joint_1.position",
            "joint_0.velocity",
            "joint_1.velocity",
        ]
        contract = recorder._action_contract()
        assert contract["representation"] == "joint_position_velocity"
        assert contract["qpos_slice"] == [0, 2]
        assert contract["qvel_slice"] == [2, 4]

    @patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset")
    def test_joint_position_contract_rejects_position_velocity_buffer(
        self, mock_lerobot_dataset
    ):
        """Contract representation must match the expert rollout encoding."""
        env = MockEnvForDataset(num_joints=2, has_sensors=False)
        env.expert_action_spec = build_expert_action_spec(
            joint_names=["joint_0", "joint_1"],
            joint_command_mode="position_velocity",
        )

        with pytest.raises(ValueError, match="requires joint_command_mode='position'"):
            LeRobotRecorder(
                MockFunctorCfg(
                    params={
                        "save_path": "/tmp/test_dataset",
                        "action_contract": {
                            "version": 1,
                            "representation": "joint_position",
                        },
                    }
                ),
                env,
            )

        mock_lerobot_dataset.create.assert_not_called()

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
            env_id=0,
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
    def test_convert_frame_keeps_structured_position_velocity_action(
        self, mock_lerobot_dataset
    ):
        env = MockEnvForDataset(num_joints=2, has_sensors=False)
        env.expert_action_spec = build_expert_action_spec(
            joint_names=["joint_0", "joint_1"],
            joint_command_mode="position_velocity",
        )
        mock_dataset_instance = Mock()
        mock_dataset_instance.meta = Mock()
        mock_dataset_instance.meta.info = {"fps": 30}
        mock_lerobot_dataset.create.return_value = mock_dataset_instance
        recorder = LeRobotRecorder(
            MockFunctorCfg(
                params={
                    "save_path": "/tmp/test_dataset",
                    "robot_meta": {"robot_type": "test_robot"},
                    "instruction": {"lang": "test task"},
                    "extra": {"task_description": "test"},
                    "use_videos": False,
                }
            ),
            env,
        )
        obs = TensorDict(
            {
                "robot": {
                    "qpos": torch.zeros(2),
                    "qvel": torch.zeros(2),
                    "qf": torch.zeros(2),
                },
                "sensor": {},
            },
            batch_size=[],
        )
        action = TensorDict(
            {
                "qpos": torch.tensor([1.0, 2.0]),
                "qvel": torch.tensor([0.1, 0.2]),
            },
            batch_size=[],
        )

        frame = recorder._convert_frame_to_lerobot(obs, action, "test_task", env_id=0)

        torch.testing.assert_close(
            frame[LeRobotKey.ACTION.value],
            torch.tensor([1.0, 2.0, 0.1, 0.2]),
        )

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
            frame = recorder._convert_frame_to_lerobot(
                obs, torch.zeros(6), "test_task", env_id=0
            )

        assert torch.equal(frame["observation.depth.camera"], depth)
        assert torch.equal(frame["observation.mask.camera"], mask)
        assert frame["observation.depth.camera"].dtype == torch.float32
        assert frame["observation.mask.camera"].dtype == torch.int32


def test_raw_action_history_preserves_eef_command():
    """Raw policy actions remain available after action preprocessing."""
    env = SimpleNamespace(
        num_envs=1, _record_raw_actions=True, _raw_action_history=[[]]
    )
    raw_action = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]])

    EmbodiedEnv._record_raw_action(env, raw_action)
    recorded = EmbodiedEnv.get_raw_action_history(env, 0)

    torch.testing.assert_close(recorded, raw_action)


def test_raw_action_history_splits_mapping_rows() -> None:
    """Mapping-form policy actions retain one row per vector environment."""
    env = SimpleNamespace(
        num_envs=2, _record_raw_actions=True, _raw_action_history=[[], []]
    )
    raw_action = {
        "eef_pose": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.0],
            ]
        )
    }

    EmbodiedEnv._record_raw_action(env, raw_action)
    recorded = EmbodiedEnv.get_raw_action_history(env, 1)

    assert isinstance(recorded, TensorDict)
    torch.testing.assert_close(recorded["eef_pose"], raw_action["eef_pose"][1:2])


@pytest.mark.parametrize("width", [6, 8])
def test_policy_action_list_rejects_noncanonical_width(width: int) -> None:
    """Policy history width must match the descriptor-owned flat layout."""
    env = Mock()
    env.get_raw_action_history.return_value = torch.zeros(2, width)
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    recorder._policy_action_descriptors = policy_descriptors()

    with pytest.raises(RuntimeError, match=r"shape \(2, 7\)"):
        recorder._policy_action_list(0, 2)


def test_policy_action_list_returns_owned_requested_action() -> None:
    """Policy persistence clones the canonical flat manager input."""
    env = Mock()
    source = torch.zeros(2, 7)
    env.get_raw_action_history.return_value = source
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    recorder._policy_action_descriptors = policy_descriptors()

    result = recorder._policy_action_list(0, 2)
    source.fill_(1.0)

    torch.testing.assert_close(result, torch.zeros(2, 7))


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (torch.tensor([[float("nan"), 0, 0, 0, 0, 0, 0.0]]), "finite"),
        (torch.tensor([[0.0, 0, 0, 0, 0, 0, 1.1]]), r"within \[-1, 1\]"),
    ],
)
def test_policy_action_list_validates_term_values(
    action: torch.Tensor, message: str
) -> None:
    """Persistence rejects invalid values using each descriptor slice."""
    env = Mock()
    env.get_raw_action_history.return_value = action
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    recorder._policy_action_descriptors = policy_descriptors()

    with pytest.raises(ValueError, match=message):
        recorder._policy_action_list(0, 1)


def test_policy_action_validation_rejects_before_persistence() -> None:
    """The live policy validator uses the same descriptor value contract."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = SimpleNamespace(num_envs=1)
    recorder._policy_action_descriptors = policy_descriptors()

    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        recorder.validate_policy_action(torch.tensor([[0.0, 0, 0, 0, 0, 0, 1.1]]))


def test_eef_observation_uses_bound_parts_and_opposing_gripper_mapping() -> None:
    """Auxiliary EEF state inverts the configured independent-joint mapping."""
    arm_term = SimpleNamespace(
        part_name="manipulator",
        controlled_joint_ids=(0, 1),
    )
    gripper_term = SimpleNamespace(
        command_mode="continuous",
        controlled_joint_ids=(2, 3),
        lower_command=torch.tensor([0.0, 0.04]),
        upper_command=torch.tensor([0.04, 0.0]),
    )
    robot = Mock()
    robot.compute_fk.return_value = torch.eye(4).repeat(3, 1, 1)
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = SimpleNamespace(device=torch.device("cpu"), robot=robot)
    recorder._eef_observation_terms = (arm_term, gripper_term)
    qpos = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.04],
            [0.0, 0.0, 0.02, 0.02],
            [0.0, 0.0, 0.04, 0.0],
        ]
    )

    observation = recorder._to_eef_observation(qpos, env_ids=[2, 4, 6])

    torch.testing.assert_close(observation[:, -1], torch.tensor([-1.0, 0.0, 1.0]))
    assert robot.compute_fk.call_args.kwargs["name"] == "manipulator"
    assert robot.compute_fk.call_args.kwargs["env_ids"] == [2, 4, 6]


def test_joint_contract_action_list_keeps_primary_action_only() -> None:
    """Joint contracts do not add an auxiliary controller action."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._action_contract_cfg = {
        "version": 1,
        "representation": "joint_position",
    }
    stored_actions = torch.full((2, 3), 0.75)

    result = recorder._contract_action_list(0, 2, stored_actions)

    torch.testing.assert_close(result, stored_actions)


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
    actions = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    expected_actions = actions[:2].clone()
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
                    "task_program_id": "repeat_pick_place",
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
    actions.fill_(99.0)

    assert len(payloads) == 1
    _, fragment_obs, fragment_actions, fragment_annotations, fragment_metadata = (
        payloads[0]
    )
    assert fragment_obs.batch_size == torch.Size([2])
    assert fragment_actions.shape == (2, 7)
    torch.testing.assert_close(fragment_actions, expected_actions)
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
def test_fragment_retry_skips_an_earlier_independent_commit() -> None:
    """A later fragment failure cannot duplicate an earlier committed slice."""
    env = Mock()
    env.rollout_steps = torch.tensor([2])
    env.rollout_buffer = TensorDict(
        {
            "obs": torch.zeros(1, 2, 1),
            "actions": torch.zeros(1, 2, 1),
        },
        batch_size=[1, 2],
    )
    env.get_demo_episode_metadata.return_value = {"output_mode": "segment_fragments"}
    first_payload = (
        0,
        [object()],
        [object()],
        {},
        {"fragment": True, "fragment_id": "run:0:first"},
    )
    second_payload = (
        0,
        [object()],
        [object()],
        {},
        {"fragment": True, "fragment_id": "run:0:second"},
    )
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    recorder.curr_episode = 0
    recorder._episode_payloads = Mock(return_value=[first_payload, second_payload])
    recorder._save_single_episode = Mock(
        side_effect=[True, OSError("second fragment disk failure")]
    )

    with pytest.raises(RuntimeError, match="run:0:second") as error_info:
        recorder._save_episodes(torch.tensor([0]))

    assert "run:0:first" in str(error_info.value)
    assert recorder._committed_fragment_ids == {"run:0:first": 0}

    recorder._save_single_episode.reset_mock(side_effect=True)
    recorder._save_episodes(torch.tensor([0]))

    recorder._save_single_episode.assert_called_once()
    assert recorder._save_single_episode.call_args.kwargs["episode_metadata"] == {
        "fragment": True,
        "fragment_id": "run:0:second",
    }
    assert recorder._committed_fragment_ids == {
        "run:0:first": 0,
        "run:0:second": 0,
    }


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
def test_action_contract_uses_official_per_frame_tasks(tmp_path) -> None:
    """Contract datasets encode segment instructions through task_index."""
    env = MockEnvForDataset(num_joints=2, has_sensors=False)
    recorder = LeRobotRecorder(
        MockFunctorCfg(
            params={
                "save_path": str(tmp_path),
                "robot_meta": {"robot_type": "test_robot"},
                "instruction": {"lang": "Move the cube."},
                "extra": {"task_description": "official_tasks"},
                "action_contract": {
                    "version": 1,
                    "representation": "joint_position",
                },
            }
        ),
        env,
    )
    obs_list = [
        TensorDict(
            {
                "robot": {
                    "qpos": torch.zeros(2),
                    "qvel": torch.zeros(2),
                    "qf": torch.zeros(2),
                }
            },
            batch_size=[],
        )
        for _ in range(2)
    ]
    episode_metadata = {
        "segments": [
            {
                "segment_id": 0,
                "start_step": 0,
                "end_step": 1,
                "instruction": "Pick up the cube.",
            },
            {
                "segment_id": 1,
                "start_step": 1,
                "end_step": 2,
                "instruction": "Place the cube.",
            },
        ]
    }

    assert recorder._save_single_episode(
        0,
        obs_list,
        [torch.zeros(2), torch.zeros(2)],
        episode_metadata=episode_metadata,
    )
    recorder.finalize()

    loaded = LeRobotDataset(
        repo_id=recorder.dataset_full_path.name,
        root=recorder.dataset_full_path,
    )
    samples = [loaded[index] for index in range(2)]

    assert [sample["task"] for sample in samples] == [
        "Pick up the cube.",
        "Place the cube.",
    ]
    assert "subtask_index" not in samples[0]
    assert not (recorder.dataset_full_path / "meta" / "subtasks.parquet").exists()


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_legacy_episode_metadata_omits_action_contract() -> None:
    """Omitting action_contract preserves the main sidecar schema."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = MockEnvForDataset(has_sensors=False)
    recorder._action_contract_cfg = None
    recorder.use_official_task_index = False
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

    assert recorder._save_single_episode(0, [object()], [object()])

    metadata = recorder._write_episode_metadata.call_args.args[0]
    assert "action_contract" not in metadata


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
def test_policy_episode_sidecar_contains_action_terms() -> None:
    """Policy sidecars persist the same ordered descriptors as features."""
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = MockEnvForDataset(has_sensors=False)
    recorder._action_contract_cfg = {
        "version": 1,
        "representation": "eef_pose_parallel_gripper",
        "record_eef_observation": False,
    }
    recorder._policy_action_descriptors = policy_descriptors()
    recorder.record_eef_observation = False
    recorder.use_official_task_index = True
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

    assert recorder._save_single_episode(0, [object()], [torch.zeros(7)])

    assert recorder._convert_frame_to_lerobot.call_args.kwargs["env_id"] == 0
    metadata = recorder._write_episode_metadata.call_args.args[0]
    assert metadata["embodichain.action_terms"] == [
        descriptor.to_dict() for descriptor in policy_descriptors()
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
def test_post_commit_fragment_failure_is_sticky_and_not_duplicated() -> None:
    """A committed fragment with a failed sidecar cannot be written twice."""
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
    fragment_metadata = {
        "fragment": True,
        "fragment_id": "run:0:pick",
        "segments": [],
    }

    with pytest.raises(OSError, match="disk full"):
        recorder._persist_episode_payload(
            0,
            [object()],
            [object()],
            episode_metadata=fragment_metadata,
        )

    recorder._write_episode_metadata.side_effect = None
    with pytest.raises(RuntimeError, match="Refusing to write a duplicate"):
        recorder._persist_episode_payload(
            0,
            [object()],
            [object()],
            episode_metadata=fragment_metadata,
        )

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
