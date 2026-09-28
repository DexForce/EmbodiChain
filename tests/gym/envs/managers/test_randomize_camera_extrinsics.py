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

import math
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization import visual
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_camera_extrinsics,
)
from embodichain.utils.math import matrix_from_quat


def _make_env(extrinsics: SimpleNamespace) -> SimpleNamespace:
    camera = MagicMock()
    camera.cfg.extrinsics = extrinsics
    sim = SimpleNamespace(get_sensor=lambda uid: camera)
    return SimpleNamespace(sim=sim, device="cpu", num_envs=2, camera=camera)


def _look_at_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent=None, eye=[0.0, 0.0, 1.0], target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]
    )


def _attach_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent="link", eye=None, pos=[0.0, 0.0, 0.1], quat=[1.0, 0.0, 0.0, 0.0]
    )


def _arena_pose_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent=None,
        eye=None,
        pos=[0.0, 0.0, 0.1],
        # A non-identity starting orientation makes Euler perturbations observable.
        quat=[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    )


def _parented_look_at_extrinsics() -> SimpleNamespace:
    return SimpleNamespace(
        parent="link",
        eye=[0.0, 0.0, 1.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
    )


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(visual.logger, "log_warning", messages.append)
    monkeypatch.setattr(
        visual, "_WARNED_IGNORED_CAMERA_RANGES", weakref.WeakKeyDictionary()
    )
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
def test_envs_sharing_a_camera_uid_each_warn(warnings) -> None:
    envs = [_make_env(_look_at_extrinsics()) for _ in range(2)]
    for env in envs * 2:
        randomize_camera_extrinsics(
            env,
            torch.arange(2),
            SceneEntityCfg(uid="cam_high"),
            pos_range=([-0.01] * 3, [0.01] * 3),
        )

    # One warning per environment, not one per process.
    assert len(warnings) == 2
    assert all("cam_high" in message for message in warnings)


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


@pytest.mark.no_sim
def test_arena_pose_camera_uses_nonzero_pose_ranges(warnings, monkeypatch) -> None:
    def midpoint_sample(*, lower, upper, size, device):
        return ((lower + upper) / 2).expand(size).clone()

    monkeypatch.setattr(visual, "sample_uniform", midpoint_sample)
    env = _make_env(_arena_pose_extrinsics())
    randomize_camera_extrinsics(
        env,
        env_ids=None,
        entity_cfg=SceneEntityCfg(uid="cam_fixed"),
        pos_range=([0.02, -0.03, 0.01], [0.06, -0.01, 0.05]),
        euler_range=([0.02, -0.04, 0.03], [0.08, 0.02, 0.09]),
    )

    assert warnings == []
    env.camera.set_local_pose.assert_called_once()
    (pose,) = env.camera.set_local_pose.call_args.args
    assert pose.shape == (2, 4, 4)
    expected_position = torch.tensor([0.04, -0.02, 0.13]).repeat(2, 1)
    assert torch.allclose(pose[:, :3, 3], expected_position)

    # Arena-frame poses are converted to the OpenGL frame just like Camera.reset.
    baseline_rotation = matrix_from_quat(
        torch.tensor(_arena_pose_extrinsics().quat).unsqueeze(0)
    )[0]
    baseline_rotation[:, 1] *= -1
    baseline_rotation[:, 2] *= -1
    assert not torch.allclose(pose[0, :3, :3], baseline_rotation)


@pytest.mark.no_sim
def test_arena_pose_camera_warns_about_look_at_ranges(warnings) -> None:
    env = _make_env(_arena_pose_extrinsics())
    randomize_camera_extrinsics(
        env,
        torch.arange(2),
        SceneEntityCfg(uid="cam_fixed"),
        eye_range=([-0.01] * 3, [0.01] * 3),
    )

    assert len(warnings) == 1
    assert "cam_fixed" in warnings[0]
    assert "eye_range is ignored" in warnings[0]
    env.camera.set_local_pose.assert_called_once()


@pytest.mark.no_sim
def test_look_at_camera_defaults_missing_up_vector(warnings) -> None:
    extrinsics = _look_at_extrinsics()
    extrinsics.up = None
    env = _make_env(extrinsics)

    randomize_camera_extrinsics(
        env,
        torch.arange(2),
        SceneEntityCfg(uid="cam_high"),
        eye_range=([-0.01] * 3, [0.01] * 3),
    )

    assert warnings == []
    env.camera.look_at.assert_called_once()
    _, _, up = env.camera.look_at.call_args.args
    assert torch.allclose(up, torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1))


@pytest.mark.no_sim
def test_parented_look_at_camera_is_rejected(warnings) -> None:
    env = _make_env(_parented_look_at_extrinsics())

    with pytest.raises(ValueError, match="cannot combine parent-attached"):
        randomize_camera_extrinsics(
            env,
            torch.arange(2),
            SceneEntityCfg(uid="cam_invalid"),
            eye_range=([-0.01] * 3, [0.01] * 3),
        )

    assert warnings == []
    env.camera.look_at.assert_not_called()
    env.camera.set_local_pose.assert_not_called()
