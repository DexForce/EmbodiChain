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

import json

import numpy as np
import pytest

from embodichain.lab.sim.utility.tcp_calibration import (
    TCPMarkerCalibrator,
    adjust_tcp_transform,
    relative_transform,
    save_solver_tcp_overrides,
    solver_tcp_overrides,
)


class _MarkerHandle:
    def __init__(self):
        self.pose = None

    def set_local_pose(self, pose):
        self.pose = np.asarray(pose)


class _Simulation:
    def __init__(self):
        self.drawn = []
        self.removed = []

    def draw_marker(self, cfg):
        marker = _MarkerHandle()
        marker.set_local_pose(cfg.axis_xpos)
        self.drawn.append(cfg)
        return [marker]

    def remove_marker(self, name):
        self.removed.append(name)


class _Robot:
    num_instances = 1
    link_names = ["left_ee"]

    def get_link_pose(self, link_name, env_ids, to_matrix):
        assert link_name == "left_ee"
        assert env_ids == [0]
        assert to_matrix
        pose = np.eye(4)
        pose[0, 3] = 0.5
        return np.expand_dims(pose, axis=0)


def test_adjust_tcp_transform_uses_ee_frame_translation_and_rotation():
    result = adjust_tcp_transform(
        np.eye(4),
        translation=(0.01, -0.02, 0.14),
        rotation_axis="z",
        rotation_degrees=90.0,
    )

    np.testing.assert_allclose(result[:3, 3], [0.01, -0.02, 0.14])
    np.testing.assert_allclose(
        result[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-7,
    )


def test_relative_transform_recovers_end_link_relative_tcp():
    ee_pose = np.eye(4)
    ee_pose[:3, 3] = [0.4, -0.2, 0.8]
    expected_tcp = np.eye(4)
    expected_tcp[:3, 3] = [0.01, 0.02, 0.14]
    tcp_pose = ee_pose @ expected_tcp

    np.testing.assert_allclose(relative_transform(ee_pose, tcp_pose), expected_tcp)


def test_solver_tcp_overrides_builds_robot_config_fragment():
    left_tcp = np.eye(4)
    left_tcp[2, 3] = 0.14

    result = solver_tcp_overrides({"left_arm": left_tcp})

    assert result["solver_cfg"]["left_arm"]["tcp"] == left_tcp.tolist()


def test_save_solver_tcp_overrides_writes_json(tmp_path):
    output_path = tmp_path / "calibration" / "marvin_tcp.json"
    right_tcp = np.eye(4)
    right_tcp[0, 3] = 0.03

    result = save_solver_tcp_overrides(output_path, {"right_arm": right_tcp})

    assert result == output_path
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["solver_cfg"]["right_arm"]["tcp"] == right_tcp.tolist()


@pytest.mark.parametrize(
    "invalid_transform",
    [np.eye(3), np.full((4, 4), np.nan)],
)
def test_adjust_tcp_transform_rejects_invalid_matrices(invalid_transform):
    with pytest.raises(ValueError):
        adjust_tcp_transform(invalid_transform)


def test_tcp_marker_calibrator_draws_updates_and_removes_markers():
    sim = _Simulation()
    initial_tcp = np.eye(4)
    initial_tcp[2, 3] = 0.14
    calibrator = TCPMarkerCalibrator(
        sim,
        _Robot(),
        control_part="left_arm",
        end_link_name="left_ee",
        tcp_transform=initial_tcp,
        marker_prefix="test_left",
    )

    calibrator.draw()
    calibrator.translate("z", 0.01)

    assert [cfg.name for cfg in sim.drawn] == ["test_left_ee", "test_left_tcp"]
    np.testing.assert_allclose(calibrator.tcp_transform[:3, 3], [0.0, 0.0, 0.15])
    np.testing.assert_allclose(calibrator.get_tcp_pose()[:3, 3], [0.5, 0.0, 0.15])

    calibrator.close()
    assert sim.removed == ["test_left_ee_0", "test_left_tcp_0"]
