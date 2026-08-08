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

"""Tests for the asynchronous LeRobot recorder."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from tensordict import TensorDict

# Skip all tests if LeRobot is not available.
try:
    from embodichain.lab.gym.envs.managers.async_datasets import (
        AsyncLeRobotRecorder,
        LEROBOT_AVAILABLE,
    )

    from embodichain.data.enum import LeRobotKey

except ImportError:
    LEROBOT_AVAILABLE = False
    AsyncLeRobotRecorder = None  # type: ignore[assignment]
    LeRobotKey = None  # type: ignore[assignment]


class _MockRobot:
    def __init__(self, num_joints: int = 6):
        self.num_joints = num_joints
        self.joint_names = [f"joint_{i}" for i in range(num_joints)]


class _MockDataset:
    """Minimal LeRobotDataset stand-in that records save calls."""

    def __init__(self):
        self.image_writer = None
        self.meta = Mock()
        self.meta.info = {"fps": 30}
        self.add_frame_calls: list[dict] = []
        self.save_episode_calls: int = 0
        self.finalize_calls: int = 0

    def add_frame(self, frame):
        self.add_frame_calls.append(frame)

    def save_episode(self, *args, **kwargs):
        self.save_episode_calls += 1

    def stop_image_writer(self):
        pass

    def finalize(self):
        self.finalize_calls += 1


class _MockEnv:
    """Mock env with a rollout buffer for async recorder tests."""

    def __init__(self, num_envs: int = 2, num_joints: int = 6, steps: int = 5):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.active_joint_ids = list(range(num_joints))
        self.robot = _MockRobot(num_joints)
        self.has_sensors = False
        # single_observation_space: only robot (+ empty sensor) so _build_features
        # does not try to resolve camera sensors.
        self.single_observation_space = {
            "robot": {"qpos": Mock(), "qvel": Mock(), "qf": Mock()},
            "sensor": {},
        }
        self.observation_manager = Mock()
        self.observation_manager.active_functors = {"add": []}

        # Rollout buffer shaped [num_envs, steps] of per-frame obs/actions,
        # matching the real init_rollout_buffer_from_gym_space layout so that
        # rollout_buffer["obs"][env_id, :step] yields a [step] TensorDict.
        obs = TensorDict(
            {
                "robot": {
                    "qpos": torch.zeros(num_envs, steps, num_joints),
                    "qvel": torch.zeros(num_envs, steps, num_joints),
                    "qf": torch.zeros(num_envs, steps, num_joints),
                },
                "sensor": {},
            },
            batch_size=[num_envs, steps],
        )
        actions = torch.zeros(num_envs, steps, num_joints)
        episode_step = torch.arange(steps).repeat(num_envs, 1)
        self.rollout_buffer = TensorDict(
            {
                "obs": obs,
                "actions": actions,
                "valid": torch.ones(num_envs, steps, dtype=torch.bool),
                "episode_step": episode_step,
                "segment_id": torch.zeros(num_envs, steps, dtype=torch.long),
                "segment_step": episode_step.clone(),
                "segment_start": episode_step == 0,
                "segment_end": torch.zeros(num_envs, steps, dtype=torch.bool),
                "terminated": torch.zeros(num_envs, steps, dtype=torch.bool),
                "truncated": torch.zeros(num_envs, steps, dtype=torch.bool),
            },
            batch_size=[num_envs, steps],
        )
        self.current_rollout_step = steps
        self.episode_metadata = {
            "segments": [
                {
                    "start_step": 0,
                    "end_step": steps,
                    "instruction": "original segment task",
                }
            ]
        }

    def get_demo_episode_metadata(self, env_id: int):
        return self.episode_metadata


def _make_recorder(env: _MockEnv, mock_dataset: _MockDataset) -> AsyncLeRobotRecorder:
    """Build an AsyncLeRobotRecorder with the LeRobotDataset.create patched out."""
    from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg

    cfg = DatasetFunctorCfg(
        func=AsyncLeRobotRecorder,
        params={
            "save_path": "/tmp/test_async_recorder",
            "robot_meta": {"robot_type": "test", "control_freq": 30},
            "instruction": {"lang": "test"},
            "extra": {"scene_type": "s", "task_description": "t"},
            "use_videos": False,
            "image_writer_threads": 0,
        },
    )
    with patch("embodichain.lab.gym.envs.managers.datasets.LeRobotDataset") as mock_cls:
        mock_cls.create.return_value = mock_dataset
        return AsyncLeRobotRecorder(cfg, env)


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot not installed")
class TestAsyncLeRobotRecorder:
    """Tests for AsyncLeRobotRecorder enqueue/drain behavior."""

    def test_call_accepts_construction_only_kwargs(self):
        """``__call__`` must accept image_writer_threads/processes (regression).

        DatasetManager.apply passes ``**functor_cfg.params`` to ``__call__``;
        construction-only params must not raise TypeError.
        """
        env = _MockEnv(num_envs=1, steps=1)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)
        try:
            # Should not raise even though these params are unused at call time.
            recorder(
                env,
                env_ids=torch.tensor([0]),
                save_path="/tmp/x",
                robot_meta={},
                instruction=None,
                extra={},
                use_videos=False,
                image_writer_threads=4,
                image_writer_processes=0,
            )
        finally:
            recorder.finalize()

    def test_call_enqueues_without_blocking(self):
        """``__call__`` returns without saving; the worker saves later."""
        env = _MockEnv(num_envs=2, steps=4)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)

        recorder(env, env_ids=torch.tensor([0, 1]))
        # In the real flow env.reset() clears current_rollout_step after the
        # save; mirror that so super().finalize() does not re-save the buffer.
        env.current_rollout_step = 0

        recorder.finalize()

        # 2 envs x 4 steps = 8 frames, 2 episodes.
        assert mock_ds.save_episode_calls == 2
        assert len(mock_ds.add_frame_calls) == 8

    def test_worker_operates_on_clone_not_live_buffer(self):
        """Mutating the rollout buffer after __call__ must not corrupt the save.

        This is the core correctness guarantee: the slice is cloned in the
        caller thread so the worker is immune to buffer reuse on reset.
        """
        env = _MockEnv(num_envs=1, steps=3)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)

        recorder(env, env_ids=torch.tensor([0]))
        env.current_rollout_step = 0  # mirror post-save reset

        # Corrupt the live buffer after enqueue (simulates reset reuse).
        env.rollout_buffer["obs"][0, :3] = 999.0
        env.rollout_buffer["actions"][0, :3] = 999.0
        env.rollout_buffer["segment_id"][0, :3] = 999
        env.episode_metadata["segments"][0]["instruction"] = "corrupted"

        recorder.finalize()

        # Saved frames should be the originals (zeros), not 999.
        for frame in mock_ds.add_frame_calls:
            assert (frame[LeRobotKey.OBS_STATE.value] == 0).all()
            assert (frame[LeRobotKey.ACTION.value] == 0).all()
            assert frame["annotation.segment_id"].tolist() == [0]
            assert frame["task"] == "test"
            assert frame["subtask_index"].tolist() == [0]
        assert mock_ds.meta.subtasks.index.tolist() == ["original segment task"]
        assert mock_ds.add_frame_calls[-1]["annotation.segment_end"].tolist() == [1]

    def test_finalize_drains_and_finalizes_dataset(self):
        """finalize() must drain the worker then call dataset.finalize()."""
        env = _MockEnv(num_envs=1, steps=2)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)

        recorder(env, env_ids=torch.tensor([0]))
        env.current_rollout_step = 0  # mirror post-save reset

        first_path = recorder.finalize()
        second_path = recorder.close()

        assert mock_ds.save_episode_calls == 1
        assert mock_ds.finalize_calls == 1
        assert first_path == second_path == recorder.dataset_path

    def test_finalize_does_not_enqueue_uncommitted_live_rollout(self):
        """A rollout not explicitly passed to the recorder is discarded on close."""
        env = _MockEnv(num_envs=1, steps=2)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)

        recorder.finalize()

        assert mock_ds.save_episode_calls == 0
        assert mock_ds.finalize_calls == 1

    def test_call_skips_empty_rollout(self):
        """An initial reset must not enqueue a zero-frame episode."""
        env = _MockEnv(num_envs=1, steps=0)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)
        recorder._save_single_episode = Mock()

        recorder(env, env_ids=torch.tensor([0]))
        recorder.finalize()

        recorder._save_single_episode.assert_not_called()
        assert mock_ds.save_episode_calls == 0

    def test_finalize_aggregates_background_failures_after_draining(self):
        """All queued commits run and their failures surface at the barrier."""
        env = _MockEnv(num_envs=2, steps=2)
        mock_ds = _MockDataset()
        recorder = _make_recorder(env, mock_ds)

        def fail_save(env_id, *args, **kwargs):
            if env_id == 0:
                return False
            raise OSError("disk unavailable")

        recorder._save_single_episode = Mock(side_effect=fail_save)
        recorder(env, env_ids=torch.tensor([0, 1]))

        with pytest.raises(RuntimeError) as error_info:
            recorder.finalize()

        message = str(error_info.value)
        assert "2 committed episode(s)" in message
        assert "env 0" in message
        assert "env 1" in message
        assert "disk unavailable" in message
        assert recorder._save_single_episode.call_count == 2
        assert mock_ds.finalize_calls == 1

        with pytest.raises(RuntimeError, match="2 committed episode"):
            recorder.close()
        assert recorder._save_single_episode.call_count == 2
        assert mock_ds.finalize_calls == 1

    def test_call_after_finalize_is_rejected(self):
        """No commit may be queued behind the shutdown sentinel."""
        env = _MockEnv(num_envs=1, steps=1)
        recorder = _make_recorder(env, _MockDataset())
        recorder.finalize()

        with pytest.raises(RuntimeError, match="already finalized"):
            recorder(env, env_ids=torch.tensor([0]))
