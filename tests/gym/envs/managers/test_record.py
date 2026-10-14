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

from unittest.mock import Mock
import numpy as np
import pytest
import torch

import embodichain.lab.gym.envs.managers.record as record_module

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.sim.sensors import CameraCfg


class _CameraHandle:
    group_id = 7

    def __init__(self) -> None:
        self.rgb = torch.zeros(2, 4, 6, 4, dtype=torch.uint8)

    def update(self, **kwargs) -> None:
        pass

    def get_data(self) -> dict:
        return {"color": self.rgb}


class _Simulation:
    def __init__(self) -> None:
        self.camera = _CameraHandle()
        self.sensor_cfg: CameraCfg | None = None

    def add_sensor(self, sensor_cfg: CameraCfg) -> _CameraHandle:
        self.sensor_cfg = sensor_cfg
        return self.camera

    def render_camera_group(self, group_ids: list[int]) -> None:
        assert group_ids == [self.camera.group_id]


class _Environment:
    control_frequency = 24.0

    def __init__(self) -> None:
        self.sim = _Simulation()
        self.camera_group_ids: list[int] = []

    def add_camera_group_id(self, group_id: int) -> None:
        self.camera_group_ids.append(group_id)


def test_record_camera_is_marked_for_visualization() -> None:
    env = _Environment()
    cfg = FunctorCfg(func=record_camera_data, params={"name": "episode_camera"})

    record_camera_data(cfg, env)

    assert env.sim.sensor_cfg is not None
    assert env.sim.sensor_cfg.visualization_role == "record"


def _make_recorder(**params):
    env = _Environment()
    params = {"name": "views", **params}
    cfg = FunctorCfg(func=record_camera_data, params=params)
    return env, record_camera_data(cfg, env)


def test_per_environment_capture_copies_views_and_saves_at_control_frequency(
    monkeypatch,
):
    env, recorder = _make_recorder(per_env=True, fps=None)
    # Distinct pixels identify the arena view in each saved video.
    env.sim.camera.rgb[0, ..., :3] = 11
    env.sim.camera.rgb[1, ..., :3] = 71
    recorder(env, None, name="views", per_env=True, fps=None)
    env.sim.camera.rgb.fill_(0)  # Camera buffers are reused after acquisition.
    save_video = Mock()
    monkeypatch.setattr(record_module, "images_to_video", save_video)

    recorder.save_and_clear()
    recorder.finalize()
    assert save_video.call_count == 2
    for env_id, (expected, call) in enumerate(zip((11, 71), save_video.call_args_list)):
        assert call.args[2] == f"episode_0_views_env{env_id}"
        np.testing.assert_array_equal(call.args[0][0], np.full((4, 6, 3), expected))
        assert call.kwargs["fps"] == 24.0


def test_per_environment_reset_and_discard_preserve_unselected_recordings(monkeypatch):
    env, recorder = _make_recorder(per_env=True)
    save_video = Mock()
    monkeypatch.setattr(record_module, "images_to_video", save_video)
    recorder(env, None, name="views")
    recorder.save_and_clear(torch.tensor([0]))
    recorder(env, None, name="views")
    recorder.discard_and_clear(torch.tensor([0]))
    recorder.save_and_clear(torch.tensor([1]))
    assert [
        (call.args[2], len(call.args[0])) for call in save_video.call_args_list
    ] == [("episode_0_views_env0", 1), ("episode_0_views_env1", 2)]
    recorder(env, None, name="views")
    recorder.save_and_clear(torch.tensor([0]))
    assert save_video.call_args.args[2] == "episode_1_views_env0"
    recorder.finalize()
    assert not recorder._env_frames[1]


@pytest.mark.parametrize(
    ("env_ids", "max_env_num", "expected_env_ids"),
    [
        (None, 3, [0, 1, 2]),
        ([2, 0], 3, [2, 0]),
        ([2, 0], 2, [0]),
        ([], 3, []),
        ([2], 2, []),
    ],
)
def test_per_environment_capture_honors_event_rows_and_recording_limit(
    env_ids, max_env_num, expected_env_ids, monkeypatch
):
    env, recorder = _make_recorder(per_env=True)
    # Pixel values identify original arena IDs even for reordered selections.
    env.sim.camera.rgb = torch.stack(
        [torch.full((4, 6, 4), env_id, dtype=torch.uint8) for env_id in range(3)]
    )
    selection = None if env_ids is None else torch.tensor(env_ids, dtype=torch.long)
    recorder(env, selection, name="views", max_env_num=max_env_num)
    save_video = Mock()
    monkeypatch.setattr(record_module, "images_to_video", save_video)

    recorder.save_and_clear()

    assert [call.args[2] for call in save_video.call_args_list] == [
        f"episode_0_views_env{env_id}" for env_id in expected_env_ids
    ]
    for env_id, call in zip(expected_env_ids, save_video.call_args_list):
        np.testing.assert_array_equal(call.args[0][0], np.full((4, 6, 3), env_id))


def test_per_environment_capture_preserves_frames_for_rows_not_due(monkeypatch):
    env, recorder = _make_recorder(per_env=True)
    save_video = Mock()
    monkeypatch.setattr(record_module, "images_to_video", save_video)
    recorder(env, None, name="views")
    # After a partial reset, per-row event intervals can be out of phase.
    recorder(env, torch.tensor([1]), name="views")

    recorder.save_and_clear()

    assert [
        (call.args[2], len(call.args[0])) for call in save_video.call_args_list
    ] == [("episode_0_views_env0", 1), ("episode_0_views_env1", 2)]


def test_failed_per_environment_save_keeps_frames_for_retry(monkeypatch):
    env, recorder = _make_recorder(per_env=True)
    recorder(env, None, name="views")
    save_video = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr(record_module, "images_to_video", save_video)
    with pytest.raises(OSError, match="disk full"):
        recorder.save_and_clear(torch.tensor([1]))
    save_video.side_effect = None
    recorder.save_and_clear(torch.tensor([1]))
    assert save_video.call_args.args[2] == "episode_0_views_env1"
    assert len(save_video.call_args.args[0]) == 1


def test_camera_recording_honors_explicit_environment_limit(monkeypatch):
    env, recorder = _make_recorder(per_env=True)
    env.sim.camera.rgb = torch.zeros(17, 4, 6, 4, dtype=torch.uint8)
    recorder(env, None, name="views", max_env_num=17)
    save_video = Mock()
    monkeypatch.setattr(record_module, "images_to_video", save_video)
    recorder.save_and_clear()
    assert save_video.call_count == 17
    env, recorder = _make_recorder()
    recorder(env, None, name="views", max_env_num=1)
    assert recorder._frames[0].shape == (4, 6, 3)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_recording_fps_fails_before_camera_creation(fps):
    env = _Environment()
    with pytest.raises(ValueError, match="fps"):
        record_camera_data(FunctorCfg(params={"fps": fps}), env)
    assert env.sim.sensor_cfg is None
