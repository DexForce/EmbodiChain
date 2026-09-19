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

"""Synchronous episode receipt boundaries without a simulator."""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.managers.dataset_manager import DatasetManager
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.lab.gym.envs.managers.async_datasets import AsyncLeRobotRecorder


class MemoryDataset:
    """Model LeRobot's append and independent episode counter."""

    def __init__(self):
        self.batch_encoding_size = 1
        self.episode_buffer = {"episode_index": 7, "size": 0}
        self.saved = []
        self.fail_save = False
        self.fail_add = False

    def add_frame(self, frame):
        if self.fail_add:
            raise OSError("frame failure")
        self.episode_buffer["size"] += 1

    def save_episode(self):
        if self.fail_save:
            raise OSError("primary failure")
        self.saved.append(dict(self.episode_buffer))
        self.episode_buffer = {
            "episode_index": self.episode_buffer["episode_index"] + 1,
            "size": 0,
        }

    def clear_episode_buffer(self):
        self.episode_buffer["size"] = 0


def make_recorder(tmp_path):
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    metadata = [
        {"augmentation": {"episode_id": f"episode-{row}", "commit_id": f"commit-{row}"}}
        for row in range(3)
    ]
    recorder._env = SimpleNamespace(
        num_envs=3,
        step_dt=0.1,
        rollout_steps=torch.tensor([2, 1, 2]),
        rollout_buffer={"obs": torch.zeros(3, 2, 1), "actions": torch.zeros(3, 2, 1)},
        get_demo_episode_metadata=lambda row: metadata[row],
    )
    recorder.instruction = None
    recorder.extra = {}
    recorder.curr_episode = 0  # Deliberately differs from LeRobot's real index.
    recorder.total_time = 0.0
    recorder.dataset_full_path = tmp_path
    recorder.dataset = MemoryDataset()
    recorder._depth_manager = None
    recorder._finalize_lock = threading.Lock()
    recorder._finalized = False
    recorder._metadata_lock = threading.Lock()
    recorder._register_subtasks = Mock(return_value={"unknown_task": 0})
    recorder._convert_frame_to_lerobot = Mock(return_value={})
    recorder._flush_episode_commit = Mock()
    return recorder, metadata


def make_manager(recorder):
    manager = DatasetManager.__new__(DatasetManager)
    manager._mode_functor_cfgs = {"save": [SimpleNamespace(func=recorder)]}
    return manager


def test_receipts_select_exact_rows_and_actual_dataset_indices(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    receipts = make_manager(recorder).commit_episode_rows(torch.tensor([2, 0]))
    assert [
        (r.env_id, r.episode_id, r.commit_id, r.dataset_episode_index) for r in receipts
    ] == [(2, "episode-2", "commit-2", 7), (0, "episode-0", "commit-0", 8)]
    assert all(r.confirmed_modalities == ("dataset", "metadata") for r in receipts)
    assert recorder.episode_commit_receipts == receipts
    with pytest.raises(FrozenInstanceError):
        receipts[0].env_id = 1
    import json

    rows = [
        json.loads(line)
        for line in (tmp_path / "meta/embodichain_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["lerobot_episode_index"] for r in rows] == [7, 8]
    assert [r["augmentation"]["commit_id"] for r in rows] == ["commit-2", "commit-0"]


def test_retry_returns_original_receipt_without_duplicate_append(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    manager = make_manager(recorder)
    first = manager.commit_episode_rows([1])
    assert manager.commit_episode_rows([1]) == first
    assert len(recorder.dataset.saved) == 1


def test_precommit_failure_clears_pending_frames_and_allows_retry(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    recorder.dataset.fail_add = True
    with pytest.raises(OSError, match="frame failure"):
        recorder.commit_episode_rows([0])
    assert recorder.episode_commit_receipts == ()
    assert recorder.total_time == 0
    recorder.dataset.fail_add = False
    receipts = recorder.commit_episode_rows([0])
    assert receipts[0].dataset_episode_index == 7
    assert recorder.dataset.saved == [{"episode_index": 7, "size": 2}]


@pytest.mark.parametrize("failure_location", ["metadata", "depth", "flush"])
def test_postcommit_failure_is_sticky_and_has_no_receipt(tmp_path, failure_location):
    recorder, _ = make_recorder(tmp_path)
    if failure_location == "metadata":
        recorder._write_episode_metadata = Mock(side_effect=OSError("disk full"))
    elif failure_location == "flush":
        recorder._flush_episode_commit = Mock(side_effect=OSError("disk full"))
    else:
        recorder._depth_sensor_specs = {"camera": (2, 2)}
        recorder._depth_manager = SimpleNamespace(
            start_episode=Mock(), end_episode=Mock(side_effect=OSError("disk full"))
        )
    with pytest.raises(OSError, match="disk full"):
        recorder.commit_episode_rows([0])
    assert recorder.episode_commit_receipts == ()
    with pytest.raises(RuntimeError, match="episode 7.*Refusing to write a duplicate"):
        recorder.commit_episode_rows([0])
    assert len(recorder.dataset.saved) == 1


def test_uncertain_primary_failure_blocks_retries_and_later_rows(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    original = recorder.dataset.save_episode

    def fail_second():
        if len(recorder.dataset.saved) == 1:
            # Model save_episode failing after its primary parquet append.
            original()
            raise OSError("metadata failed inside save_episode")
        original()

    recorder.dataset.save_episode = fail_second
    with pytest.raises(OSError, match="metadata failed"):
        recorder.commit_episode_rows([2, 0])
    assert [r.env_id for r in recorder.episode_commit_receipts] == [2]
    assert len(recorder.dataset.saved) == 2
    recorder.dataset.save_episode = original
    for rows in ([0], [1], [2, 0]):
        with pytest.raises(RuntimeError, match="Refusing"):
            recorder.commit_episode_rows(rows)
    assert len(recorder.dataset.saved) == 2
    with pytest.raises(RuntimeError, match="Refusing"):
        recorder._persist_episode_payload(1, torch.zeros(2, 1), torch.zeros(2, 1))
    assert len(recorder.dataset.saved) == 2


@pytest.mark.parametrize("change", ["empty", "missing", "duplicate", "fragment"])
def test_invalid_rows_fail_before_any_append(tmp_path, change):
    recorder, metadata = make_recorder(tmp_path)
    if change == "empty":
        recorder._env.rollout_steps[1] = 0
    elif change == "missing":
        metadata[1].clear()
    elif change == "duplicate":
        metadata[1]["augmentation"]["commit_id"] = "commit-0"
    else:
        metadata[1]["output_mode"] = "segment_fragments"
    with pytest.raises(ValueError):
        recorder.commit_episode_rows([0, 1])
    assert recorder.dataset.saved == []


def test_commit_id_cannot_be_reused_for_different_episode(tmp_path):
    recorder, metadata = make_recorder(tmp_path)
    recorder.commit_episode_rows([0])
    metadata[1]["augmentation"]["commit_id"] = "commit-0"
    with pytest.raises(ValueError, match="different"):
        recorder.commit_episode_rows([1])
    assert len(recorder.dataset.saved) == 1


@pytest.mark.parametrize(
    "kind", ["async", "unsupported", "multiple", "none", "deferred_video"]
)
def test_preflight_rejects_unsupported_recorders(tmp_path, kind):
    recorder, _ = make_recorder(tmp_path)
    manager = make_manager(recorder)
    if kind == "async":
        manager._mode_functor_cfgs["save"][0].func = AsyncLeRobotRecorder.__new__(
            AsyncLeRobotRecorder
        )
    elif kind == "unsupported":
        manager._mode_functor_cfgs["save"][0].func = object()
    elif kind == "multiple":
        manager._mode_functor_cfgs["save"] *= 2
    elif kind == "none":
        manager._mode_functor_cfgs.clear()
    else:
        recorder.dataset.batch_encoding_size = 2
    with pytest.raises((ValueError, RuntimeError), match="synchronous|encoding"):
        manager.validate_episode_commit_support()


def test_empty_selection_has_no_receipts_or_appends(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    assert recorder.commit_episode_rows([]) == ()
    assert recorder.dataset.saved == []


def test_completed_depth_is_confirmed(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    recorder._depth_sensor_specs = {"camera": (2, 2)}
    recorder._depth_manager = SimpleNamespace(start_episode=Mock(), end_episode=Mock())
    receipts = recorder.commit_episode_rows([1])
    assert receipts[0].confirmed_modalities == ("dataset", "metadata", "depth")


def test_pending_frame_cleanup_failure_blocks_entire_sink(tmp_path):
    recorder, _ = make_recorder(tmp_path)
    recorder.dataset.fail_add = True
    recorder.dataset.clear_episode_buffer = Mock(side_effect=OSError("cleanup failed"))
    with pytest.raises(OSError, match="frame failure"):
        recorder.commit_episode_rows([0])
    recorder.dataset.fail_add = False
    for rows in ([0], [1]):
        with pytest.raises(RuntimeError, match="clear pending.*Refusing"):
            recorder.commit_episode_rows(rows)
    assert recorder.episode_commit_receipts == ()
    assert recorder.dataset.saved == []
