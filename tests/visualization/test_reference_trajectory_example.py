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
"""Camera behavior for recorded reference trajectories."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from examples.sim.reference_trajectory import _camera_path, _root_link_index


def test_follow_camera_ignores_vertical_motion_and_articulation_extent():
    poses = np.zeros((4, 2, 3, 7))
    poses[..., 6] = 1
    poses[:, :, :, 0] = np.arange(4)[:, None, None]
    poses[:, :, :, 1] = np.arange(4)[:, None, None] * 0.5
    poses[:, :, :, 2] = np.array([0.2, 1.3, 0.4, 1.0])[:, None, None]
    poses[:, :, 2, 0] += np.array([0.1, 1.0, 0.2, 0.8])[:, None]
    offsets = np.array([[0, -0.5, 0], [0, 0.5, 0]])
    eye, target = _camera_path(poses, offsets, "follow_xy")
    np.testing.assert_allclose(np.diff(target[:, :2], axis=0), [[1, 0.5]] * 3)
    np.testing.assert_allclose(eye[:, 2], eye[0, 2])
    np.testing.assert_allclose(target[:, 2], target[0, 2])
    np.testing.assert_allclose(eye - target, np.tile(eye[0] - target[0], (4, 1)))


def test_fixed_camera_retains_one_view_over_translating_clip():
    poses = np.zeros((3, 1, 2, 7))
    poses[..., 6] = 1
    poses[:, :, :, :3] = np.arange(3)[:, None, None, None]
    eye, target = _camera_path(poses, np.zeros((1, 3)), "fixed")
    np.testing.assert_allclose(eye, np.tile(eye[0], (3, 1)))
    np.testing.assert_allclose(target, np.tile(target[0], (3, 1)))


@pytest.mark.parametrize("mode", ["follow_xy", "fixed"])
def test_camera_uses_named_root_when_recording_and_asset_link_orders_differ(
    tmp_path, mode
):
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot name="robot"><link name="tip"/><link name="root"/>'
        '<joint name="joint" type="fixed"><parent link="root"/><child link="tip"/></joint></robot>'
    )
    names = ("tip", "root")
    poses = np.zeros((3, 1, 2, 7))
    poses[..., 6] = 1
    poses[:, 0, 1, 0] = [0, 1, 2]
    poses[:, 0, 0, 0] = [10, 20, 40]
    poses[:, 0, 1, 3:] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    root_index = _root_link_index(str(asset), names)
    assert root_index == 1
    expected = _camera_path(poses[:, :, [1, 0]], np.zeros((1, 3)), mode)
    actual = _camera_path(poses, np.zeros((1, 3)), mode, root_index)
    np.testing.assert_allclose(actual, expected)


def test_fixed_view_accepts_visual_links_without_nonvisual_root(tmp_path):
    asset = tmp_path / "robot.urdf"
    asset.write_text(
        '<robot name="robot"><link name="root"/><link name="body">'
        '<visual><geometry><box size="1 1 1"/></geometry></visual></link>'
        '<joint name="joint" type="fixed"><parent link="root"/><child link="body"/></joint></robot>'
    )
    root_index = _root_link_index(str(asset), ("body",))
    assert root_index is None
    poses = np.zeros((3, 1, 1, 7))
    poses[..., 6] = 1
    poses[:, 0, 0, 0] = [0, 1, 2]
    eye, target = _camera_path(poses, np.zeros((1, 3)), "fixed", root_index)
    np.testing.assert_allclose(eye, np.tile(eye[0], (3, 1)))
    np.testing.assert_allclose(target, np.tile(target[0], (3, 1)))
    with pytest.raises(ValueError, match="--camera-mode fixed"):
        _camera_path(poses, np.zeros((1, 3)), "follow_xy", root_index)
