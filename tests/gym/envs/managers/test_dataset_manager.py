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

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs.managers.dataset_manager import DatasetManager
from embodichain.lab.gym.envs.managers.record import (
    record_camera_data,
    record_camera_data_async,
)
from embodichain.lab.gym.utils.profiler import EnvProfiler


class DatasetManagerStub:
    """Minimal dataset manager used to exercise episode selection."""

    available_modes = ["save"]

    def __init__(self, save_failed_episodes: bool) -> None:
        self.save_failed_episodes = save_failed_episodes
        self.saved_env_ids: torch.Tensor | None = None

    def apply(self, mode: str, env_ids: torch.Tensor) -> None:
        assert mode == "save"
        self.saved_env_ids = env_ids.clone()


def make_env_for_episode_selection(
    *, save_failed_episodes: bool, successful_env_ids: list[int]
) -> tuple[SimpleNamespace, DatasetManagerStub]:
    num_envs = 3
    success_status = torch.zeros(num_envs, dtype=torch.bool)
    success_status[successful_env_ids] = True
    manager = DatasetManagerStub(save_failed_episodes)
    env = SimpleNamespace(
        num_envs=num_envs,
        dataset_manager=manager,
        episode_success_status=success_status,
        _task_success=torch.zeros(num_envs, dtype=torch.bool),
        # _initialize_episode opens profiler sections around each manager call;
        # a disabled EnvProfiler is a no-op so the stub can exercise the save
        # logic without a real BaseEnv (which would need a sim).
        _profiler=EnvProfiler(None, torch.device("cpu")),
        cfg=SimpleNamespace(
            events=None,
            observations=None,
            rewards=None,
            dataset=None,
        ),
        event_manager=None,
        observation_manager=None,
        reward_manager=None,
        rollout_buffer=None,
    )
    return env, manager


def test_save_failed_episodes_reads_typed_functor_config() -> None:
    def recorder(env, env_ids) -> None:
        pass

    manager = DatasetManager.__new__(DatasetManager)
    manager._mode_functor_cfgs = {
        "save": [
            DatasetFunctorCfg(
                func=recorder,
                save_failed_episodes=True,
            )
        ]
    }

    assert manager.save_failed_episodes is True


def test_save_failed_episodes_defaults_to_false() -> None:
    def recorder(env, env_ids) -> None:
        pass

    functor_cfg = DatasetFunctorCfg(func=recorder)

    assert functor_cfg.save_failed_episodes is False


def test_apply_forwards_only_functor_params() -> None:
    calls = []

    def recorder(env, env_ids, *, use_videos: bool) -> None:
        calls.append((env, env_ids, use_videos))

    env = object()
    env_ids = torch.tensor([0])
    manager = DatasetManager.__new__(DatasetManager)
    manager._env = env
    manager._mode_functor_names = {"save": ["recorder"]}
    manager._mode_functor_cfgs = {
        "save": [
            DatasetFunctorCfg(
                func=recorder,
                save_failed_episodes=True,
                params={"use_videos": True},
            )
        ]
    }

    manager.apply(mode="save", env_ids=env_ids)

    assert calls == [(env, env_ids, True)]


def make_manager_for_finalize(*named_functors) -> DatasetManager:
    """Create a manager lifecycle fixture without constructing a simulator."""
    manager = DatasetManager.__new__(DatasetManager)
    manager._mode_functor_names = {"save": [name for name, _ in named_functors]}
    manager._mode_functor_cfgs = {
        "save": [SimpleNamespace(func=functor) for _, functor in named_functors]
    }
    manager._finalize_lock = threading.Lock()
    manager._finalized = False
    manager._finalize_result = None
    manager._finalize_error = None
    return manager


def test_finalize_is_idempotent_after_success() -> None:
    """Repeated manager finalization returns the cached path without re-closing."""
    first = SimpleNamespace(finalize=MagicMock(return_value="/dataset/first"))
    second = SimpleNamespace(finalize=MagicMock(return_value="/dataset/second"))
    manager = make_manager_for_finalize(("first", first), ("second", second))

    assert manager.close() == "/dataset/first"
    assert manager.finalize() == "/dataset/first"
    first.finalize.assert_called_once_with()
    second.finalize.assert_called_once_with()


def test_finalize_aggregates_errors_and_attempts_every_functor() -> None:
    """One broken recorder cannot prevent the remaining recorders from closing."""
    first = SimpleNamespace(
        finalize=MagicMock(side_effect=OSError("first disk failure"))
    )
    successful = SimpleNamespace(finalize=MagicMock(return_value="/dataset/good"))
    third = SimpleNamespace(
        finalize=MagicMock(side_effect=RuntimeError("third writer failure"))
    )
    manager = make_manager_for_finalize(
        ("first", first), ("successful", successful), ("third", third)
    )

    with pytest.raises(RuntimeError) as error_info:
        manager.finalize()

    message = str(error_info.value)
    assert "2 dataset functor(s)" in message
    assert "first: first disk failure" in message
    assert "third: third writer failure" in message
    first.finalize.assert_called_once_with()
    successful.finalize.assert_called_once_with()
    third.finalize.assert_called_once_with()

    with pytest.raises(RuntimeError, match="2 dataset functor"):
        manager.finalize()
    first.finalize.assert_called_once_with()
    successful.finalize.assert_called_once_with()
    third.finalize.assert_called_once_with()


def test_initialize_episode_limits_successes_to_reset_envs() -> None:
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[0, 1],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[1, 2])

    assert torch.equal(manager.saved_env_ids, torch.tensor([1]))


def test_initialize_episode_saves_failed_reset_envs_when_enabled() -> None:
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=True,
        successful_env_ids=[1],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[1, 2])

    assert torch.equal(manager.saved_env_ids, torch.tensor([1, 2]))


def test_initialize_episode_saves_row_with_accepted_segment_fragment() -> None:
    """Fragment eligibility is independent of whole-episode success."""
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    env._demo_episode_metadata = [
        {"output_mode": "continuous", "segments": []},
        {
            "output_mode": "segment_fragments",
            "save_failed_fragments": False,
            "segments": [{"start_step": 0, "end_step": 2, "success": True}],
        },
        {"output_mode": "continuous", "segments": []},
    ]
    env._new_demo_episode_metadata = lambda env_id: {
        "output_mode": "continuous",
        "env_id": env_id,
        "segments": [],
    }

    EmbodiedEnv._initialize_episode(env, env_ids=[1, 2])

    assert torch.equal(manager.saved_env_ids, torch.tensor([1]))


def test_initialize_episode_commits_only_explicit_vector_rows() -> None:
    """An explicit commit subset persists only requested dataset rows."""
    env, manager = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._frames = [object()]
    recorder.save_and_clear = MagicMock()
    env.cfg.events = object()
    env.event_manager = SimpleNamespace(
        _mode_functor_cfgs={"interval": [SimpleNamespace(func=recorder)]},
        available_modes=[],
    )
    env._traj_buffer = object()
    env._traj_steps = torch.tensor([3, 2, 1])
    env.cfg.trajectory_auto_save = True
    env._save_trajectory_for_env = MagicMock()

    EmbodiedEnv._initialize_episode(
        env,
        env_ids=[0, 1, 2],
        save_data=False,
        commit_env_ids=[1],
    )

    assert torch.equal(manager.saved_env_ids, torch.tensor([1]))
    recorder.save_and_clear.assert_not_called()
    assert recorder._frames == []
    env._save_trajectory_for_env.assert_not_called()
    assert env._traj_steps.tolist() == [0, 0, 0]


def test_discard_reset_does_not_auto_save_trajectory() -> None:
    """save_data=False clears trajectory state without writing a file."""
    env, _ = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    env._traj_buffer = object()
    env._traj_steps = torch.tensor([3, 2, 1])
    env.cfg.trajectory_auto_save = True
    env._save_trajectory_for_env = MagicMock()

    EmbodiedEnv._initialize_episode(env, env_ids=[0], save_data=False)

    env._save_trajectory_for_env.assert_not_called()
    assert env._traj_steps.tolist() == [0, 2, 1]


def test_committed_trajectory_write_failure_preserves_buffer_and_raises() -> None:
    """A failed trajectory commit is visible and its live cursor is not cleared."""
    env, _ = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    env._traj_buffer = object()
    env._traj_steps = torch.tensor([3, 2, 1])
    env.cfg.trajectory_auto_save = True
    env._save_trajectory_for_env = MagicMock(side_effect=OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        EmbodiedEnv._initialize_episode(env, env_ids=[0], save_data=True)

    assert env._traj_steps.tolist() == [3, 2, 1]


def test_discard_reset_clears_camera_frames_without_saving() -> None:
    """save_data=False drops video frames through the recorder abort hook."""
    env, _ = make_env_for_episode_selection(
        save_failed_episodes=False,
        successful_env_ids=[],
    )
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._frames = [object()]
    recorder.save_and_clear = MagicMock()
    env.cfg.events = object()
    env.event_manager = SimpleNamespace(
        _mode_functor_cfgs={"interval": [SimpleNamespace(func=recorder)]},
        available_modes=[],
    )

    EmbodiedEnv._initialize_episode(env, env_ids=[0], save_data=False)

    recorder.save_and_clear.assert_not_called()
    assert recorder._frames == []


def test_camera_finalize_discards_uncommitted_frames_idempotently() -> None:
    """Closing a camera recorder cannot resurrect an explicitly discarded episode."""
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._name = "test"
    recorder._save_path = "/tmp/test-camera-recorder"
    recorder._current_episode = 0
    recorder._frames = [object()]
    recorder._finalize_lock = threading.Lock()
    recorder._finalized = False

    recorder.discard_and_clear()
    with patch(
        "embodichain.lab.gym.envs.managers.record.images_to_video"
    ) as save_video:
        recorder.finalize()
        recorder.close()

    save_video.assert_not_called()
    assert recorder._frames == []
    assert recorder._current_episode == 0
    with pytest.raises(RuntimeError, match="already finalized"):
        recorder.save_and_clear()


def test_camera_explicit_commit_is_not_repeated_by_finalize() -> None:
    """Only save_and_clear commits frames; finalize merely closes the empty buffer."""
    recorder = record_camera_data.__new__(record_camera_data)
    frame = object()
    recorder._name = "test"
    recorder._save_path = "/tmp/test-camera-recorder"
    recorder._current_episode = 0
    recorder._frames = [frame]
    recorder._finalize_lock = threading.Lock()
    recorder._finalized = False

    with patch(
        "embodichain.lab.gym.envs.managers.record.images_to_video"
    ) as save_video:
        recorder.save_and_clear()
        recorder.finalize()
        recorder.close()

    save_video.assert_called_once_with(
        [frame],
        "/tmp/test-camera-recorder",
        "episode_0_test",
        fps=20,
    )
    assert recorder._frames == []
    assert recorder._current_episode == 1


def _make_async_camera_recorder(num_envs: int = 2) -> record_camera_data_async:
    """Build an async camera recorder without constructing simulation sensors."""
    recorder = record_camera_data_async.__new__(record_camera_data_async)
    recorder._name = "test"
    recorder._save_path = "/tmp/test-async-camera-recorder"
    recorder._current_episode = 0
    recorder._frames = []
    recorder._finalize_lock = threading.Lock()
    recorder._finalized = False
    recorder._num_envs = num_envs
    recorder._frames_list = [[] for _ in range(num_envs)]
    recorder._ep_idx = [0 for _ in range(num_envs)]
    recorder._committed_env_episodes = [deque() for _ in range(num_envs)]
    recorder._async_camera_finalize_lock = threading.Lock()
    recorder._async_camera_finalized = False
    recorder._async_camera_finalize_error = None
    return recorder


def test_async_camera_commit_persists_without_a_later_step() -> None:
    """The final committed vector episode is durable before close/finalize."""
    recorder = _make_async_camera_recorder()
    recorder._frames_list = [
        [np.zeros((2, 2, 4), dtype=np.uint8)],
        [np.ones((2, 2, 4), dtype=np.uint8)],
    ]

    with patch(
        "embodichain.lab.gym.envs.managers.record.images_to_video"
    ) as save_video:
        recorder.save_and_clear(env_ids=torch.tensor([0, 1]))
        recorder.finalize()

    save_video.assert_called_once()
    _, save_path, video_name = save_video.call_args.args
    assert save_path == "/tmp/test-async-camera-recorder"
    assert video_name == "ep0_test_allenvs"
    assert save_video.call_args.kwargs == {"fps": 20}
    assert recorder._current_episode == 1
    assert all(not queue for queue in recorder._committed_env_episodes)


def test_async_camera_finalize_rejects_incomplete_committed_batch() -> None:
    """A partial vector commit is reported instead of silently discarded."""
    recorder = _make_async_camera_recorder()
    recorder._frames_list[0] = [np.zeros((2, 2, 4), dtype=np.uint8)]
    recorder.save_and_clear(env_ids=torch.tensor([0]))
    recorder.discard_and_clear(env_ids=torch.tensor([1]))

    with pytest.raises(RuntimeError, match="pending episodes per env: \\[1, 0\\]"):
        recorder.finalize()

    with pytest.raises(RuntimeError, match="incomplete committed environment batch"):
        recorder.close()


def test_pending_discard_attempts_every_camera_and_clears_trajectory() -> None:
    """One camera failure cannot skip later durability barriers or live cleanup."""
    first = record_camera_data.__new__(record_camera_data)
    first.discard_and_clear = MagicMock(side_effect=OSError("camera one failed"))
    first.finalize = MagicMock(side_effect=OSError("camera one flush failed"))
    second = record_camera_data.__new__(record_camera_data)
    second.discard_and_clear = MagicMock()
    second.finalize = MagicMock()
    env = SimpleNamespace(
        cfg=SimpleNamespace(events=True),
        event_manager=SimpleNamespace(
            _mode_functor_cfgs={
                "interval": [
                    SimpleNamespace(func=first),
                    SimpleNamespace(func=second),
                ]
            }
        ),
        _traj_steps=torch.tensor([3, 2]),
        rollout_buffer=None,
    )

    with pytest.raises(RuntimeError, match="camera one failed"):
        EmbodiedEnv._discard_pending_recordings(env)

    first.finalize.assert_called_once_with()
    second.discard_and_clear.assert_called_once_with()
    second.finalize.assert_called_once_with()
    assert env._traj_steps.tolist() == [0, 0]
