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

"""Generic source mounting frames without robot presets or runtime construction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from embodichain.lab.sim.scene.source_robot_geometry import (
    source_arm_base_position,
    source_fixed_frame_pose,
)


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(f"""<robot name="generic">
      <link name="root"/><link name="mount"/><link name="arm_base"/><link name="upper"/>
      <joint name="mount_fixed" type="fixed"><parent link="root"/><child link="mount"/>
        <origin xyz="1 2 3" rpy="0 0 {math.pi / 2}"/></joint>
      <joint name="base_fixed" type="fixed"><parent link="mount"/><child link="arm_base"/>
        <origin xyz="0 2 0"/></joint>
      <joint name="shoulder" type="revolute"><parent link="arm_base"/><child link="upper"/>
        <origin xyz="1 0 0"/><axis xyz="0 1 0"/></joint>
    </robot>""")
    return path


def test_fixed_mount_and_declared_shoulder_keep_source_frame_order(tmp_path):
    path = _source(tmp_path)
    expected = np.array(
        [[0, -1, 0, -1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
    )
    np.testing.assert_allclose(
        source_fixed_frame_pose(path, "arm_base"), expected, rtol=0, atol=1e-14
    )
    position = source_arm_base_position(
        {"fpath": str(path), "control_parts": {"custom_motion": ["shoulder"]}},
        "custom_motion",
    )
    np.testing.assert_allclose(position, [-1, 3, 3], rtol=0, atol=1e-14)


def test_explicit_component_transform_precedes_single_file_source_lookup():
    pose = np.eye(4)
    pose[:3, 3] = [9, 8, 7]
    actual = source_arm_base_position(
        {
            "urdf_cfg": {
                "components": [
                    {"component_type": "custom_motion", "transform": pose.tolist()}
                ]
            }
        },
        "custom_motion",
    )
    np.testing.assert_array_equal(actual, [9, 8, 7])


@pytest.mark.parametrize("mode", ["moving", "cycle"])
def test_unobserved_moving_or_cyclic_attachment_cannot_supply_fixed_mount(
    tmp_path, mode
):
    if mode == "moving":
        path, frame, error = _source(tmp_path), "upper", "fixed"
    else:
        path = tmp_path / "cycle.urdf"
        path.write_text(
            '<robot name="cycle"><link name="a"/><link name="b"/><joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint></robot>'
        )
        frame, error = "a", "cycle"
    with pytest.raises(ValueError, match=error):
        source_fixed_frame_pose(path, frame)
