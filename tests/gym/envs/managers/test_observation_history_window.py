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

"""Owned raw windows preserve row, episode, availability and timing identity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers import ObservationManager
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationFrame,
    ObservationSample,
)
from embodichain.lab.gym.utils.gym_utils import (
    _close_raw_rollout,
    _init_raw_rollout,
    _write_raw_rollout,
)


def _history():
    context = EpisodeContext(3)
    context._begin_episode([0, 1, 2])
    context._finish_initialization([0, 1, 2])
    manager = ObservationManager({}, SimpleNamespace(episode_context=context))
    storage = TensorDict({}, batch_size=[3])
    _init_raw_rollout(
        storage,
        context,
        {},
        torch.zeros(3, 3, 2),
    )
    manager.bind_rollout_storage(storage)
    pose_descriptor = ObservationDescriptor(
        "object", "pose", "arena", "xyz_xyzw", ("m", "unitless"), (7,), torch.float32
    )
    joint_descriptor = ObservationDescriptor(
        "robot",
        "qpos",
        "joint",
        "joint_order",
        ("joint_units",),
        (2,),
        torch.float64,
        joint_names=("a", "b"),
    )
    for slot in range(4):
        pose = torch.zeros(3, 7)
        pose[:, 0] = torch.arange(3) * 100 + slot
        pose[:, 6] = 1
        valid = torch.tensor([True, slot != 2, True])
        errors = (None, "source unavailable" if slot == 2 else None, None)
        sample = ObservationSample(pose_descriptor, pose, valid, errors)
        qpos = torch.stack(
            (
                torch.arange(3).double() * 10 + slot,
                torch.full((3,), slot, dtype=torch.float64),
            ),
            dim=1,
        )
        frame = ObservationFrame(
            None,
            {
                "pose": sample,
                "pose_alias": sample,
                "qpos": ObservationSample(
                    joint_descriptor, qpos, torch.ones(3, dtype=torch.bool), (None,) * 3
                ),
            },
            context.episode_ids,
            slot,
            timestamp=slot * 0.25,
            clock_domain="simulation_time",
            episode_owner=context,
        )
        manager._bind_schema(frame)
        kwargs = (
            {
                "field_ids": tuple(dict.fromkeys(manager.field_ids.values())),
                "env_ids": [0, 1, 2],
            }
            if slot == 0
            else {"action": torch.zeros(3, 2)}
        )
        _write_raw_rollout(storage, context, frame, **kwargs)
    return manager, context, storage


def test_owned_windows_preserve_ragged_native_rows_aliases_and_unavailable_samples():
    manager, _, storage = _history()
    storage["raw_timestamp"][2, 1] = float("nan")
    result = manager.read_history_window(
        [1, 0],
        [3, 2],
        fields=["pose_alias", "pose", "qpos"],
        episode_ids=[0, 0],
        env_ids=[2, 1],
    )
    assert result["env_ids"] == (2, 1)
    assert result["start_frames"] == (1, 0) and result["end_frames"] == (3, 2)
    assert result["revisions"][0].tolist() == [1, 2, 3]
    assert result["fields"]["pose"]["values"][0][:, 0].tolist() == [201, 202, 203]
    assert result["fields"]["pose_alias"]["valid"][1].tolist() == [True, True, False]
    assert result["fields"]["pose_alias"]["acquisition_errors"][1] == (
        None,
        None,
        "source unavailable",
    )
    assert result["fields"]["qpos"]["values"][0].dtype == torch.float64
    assert result["timestamp_valid"][0].tolist() == [False, True, True]
    assert result["clock_domain"] == "unavailable"
    result["fields"]["pose"]["values"][0].zero_()
    result["fields"]["pose"]["valid"][0].zero_()
    result["revisions"][0].zero_()
    again = manager.read_history_window(
        [2, 0], [3, 0], fields=["pose"], episode_ids=[0, 0], env_ids=[2, 1]
    )
    assert [len(v) for v in again["fields"]["pose"]["values"]] == [2, 1]
    assert again["fields"]["pose"]["values"][0][:, 0].tolist() == [202, 203]
    assert again["fields"]["pose"]["valid"][0].all()
    assert again["revisions"][0].tolist() == [2, 3]


@pytest.mark.parametrize(
    "damage", ["missing_slot", "inverted", "closed", "episode", "duplicate_env"]
)
def test_window_rejects_missing_or_unowned_boundaries(damage):
    manager, context, storage = _history()
    starts, ends, rows = [0, 1], [1, 3], [0, 2]
    if damage == "missing_slot":
        storage["raw_revision"][2, 2] = -1
    elif damage == "inverted":
        starts[0] = 2
    elif damage == "closed":
        _close_raw_rollout(storage)
    elif damage == "episode":
        context._begin_episode([2])
    else:
        rows = [0, 0]
    with pytest.raises((RuntimeError, ValueError)):
        manager.read_history_window(
            starts, ends, fields=["pose"], episode_ids=[0, 0], env_ids=rows
        )


def test_episode_replacement_during_copy_cannot_publish_a_window(monkeypatch):
    manager, context, _ = _history()
    original = context.snapshot
    count = 0

    def snapshot(env_ids=None):
        nonlocal count
        count += 1
        if count == 2:
            context._begin_episode([2])
        return original(env_ids)

    monkeypatch.setattr(context, "snapshot", snapshot)
    with pytest.raises(ValueError, match="episode"):
        manager.read_history_window(
            [0], [3], fields=["pose"], episode_ids=[0], env_ids=[2]
        )


def test_completed_boundaries_preserve_selected_rows_without_field_history_copy():
    manager, _, storage = _history()
    storage["raw_steps"][1] = 1
    assert manager.read_history_step_counts(episode_ids=[0, 0], env_ids=[2, 1]) == (
        3,
        1,
    )
    # Boundary reads do not need any field values or their validity masks.
    storage.del_("raw_obs")
    storage["raw_obs"] = TensorDict({}, batch_size=[3, 4])
    assert manager.read_history_step_counts(episode_ids=[0, 0], env_ids=[2, 1]) == (
        3,
        1,
    )
    storage["raw_revision"][2, 3] = -1
    with pytest.raises(ValueError, match="not fully recorded"):
        manager.read_history_step_counts(episode_ids=[0], env_ids=[2])


@pytest.mark.parametrize("replacement", ["close", "episode"])
def test_completed_boundary_read_rejects_owner_change_during_query(
    monkeypatch, replacement
):
    manager, context, storage = _history()
    original = context.snapshot
    count = 0

    def snapshot(env_ids=None):
        nonlocal count
        count += 1
        if count == 2:
            if replacement == "close":
                _close_raw_rollout(storage)
            else:
                context._begin_episode([2])
        return original(env_ids)

    monkeypatch.setattr(context, "snapshot", snapshot)
    with pytest.raises((ValueError, RuntimeError)):
        manager.read_history_step_counts(episode_ids=[0], env_ids=[2])
