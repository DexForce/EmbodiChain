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

"""Tests for camera-extrinsics randomization with mode-specific ranges."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization import visual
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_camera_extrinsics,
)


def _make_env(extrinsics: SimpleNamespace) -> SimpleNamespace:
    camera = MagicMock()
    camera.cfg.extrinsics = extrinsics
    sim = SimpleNamespace(get_sensor=lambda uid: camera)
    return SimpleNamespace(sim=sim, device="cpu", camera=camera)


def _look_at_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent=None, eye=[0.0, 0.0, 1.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]
    )


def _attach_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent="link", eye=None, pos=[0.0, 0.0, 0.1], quat=[1.0, 0.0, 0.0, 0.0]
    )


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(visual.logger, "log_warning", messages.append)
    monkeypatch.setattr(visual, "_WARNED_IGNORED_CAMERA_RANGES", set())
    return messages


@pytest.mark.no_sim
def test_look_at_camera_warns_once_about_pos_and_euler_ranges(warnings) -> None:
    env = _make_env(_look_at_extrinsics())
    env_ids = torch.arange(2)
    for _ in range(3):
        randomize_camera_extrinsics(
            env,
            env_ids,
            SceneEntityCfg(uid="cam_high"),
            pos_range=([-0.01] * 3, [0.01] * 3),
            euler_range=([-0.05] * 3, [0.05] * 3),
        )

    assert len(warnings) == 1
    assert "cam_high" in warnings[0]
    assert "pos_range, euler_range are ignored" in warnings[0]
    # The camera is still placed with its configured look_at pose on every reset.
    assert env.camera.look_at.call_count == 3
    env.camera.set_local_pose.assert_not_called()


@pytest.mark.no_sim
def test_look_at_camera_with_look_at_ranges_does_not_warn(warnings) -> None:
    env = _make_env(_look_at_extrinsics())
    randomize_camera_extrinsics(
        env,
        torch.arange(2),
        SceneEntityCfg(uid="cam_high"),
        eye_range=([-0.01] * 3, [0.01] * 3),
        target_range=([-0.01] * 3, [0.01] * 3),
    )

    assert warnings == []
    env.camera.look_at.assert_called_once()


@pytest.mark.no_sim
def test_attached_camera_warns_about_look_at_ranges(warnings) -> None:
    env = _make_env(_attach_extrinsics())
    randomize_camera_extrinsics(
        env,
        torch.arange(2),
        SceneEntityCfg(uid="cam_wrist"),
        pos_range=([-0.01] * 3, [0.01] * 3),
        eye_range=([-0.01] * 3, [0.01] * 3),
    )

    assert len(warnings) == 1
    assert "cam_wrist" in warnings[0]
    assert "eye_range is ignored" in warnings[0]
    env.camera.set_local_pose.assert_called_once()
