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
"""Configured Cartesian trajectories with scene binding and ordered IK."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.task_program.integrations._configured_track import (
    _decode_cartesian_track,
    _resolve_and_plan_track,
)

pytestmark = pytest.mark.no_sim


@pytest.fixture
def trajectory() -> dict:
    return {
        "control_hz": 4,
        "arm": "xyz",
        "gripper": "finger",
        "tool_link": "tip",
        "constraints": {
            "position_tolerance_m": 1e-5,
            "rotation_tolerance_rad": 1e-4,
        },
        "arcs": {
            "turn": {
                "axis": {"kind": "scene_entity", "entity_id": "post"},
                "radius_m": 0.03,
                "start_deg": 0,
                "sweep_deg": 450,
                "height_m": [0, 0.08],
                "tool_offset": [0, 0, 0],
                "tool_orientation_xyzw": [0, 0, 0, 1],
            }
        },
        "targets": {
            "entry": {"kind": "arc_endpoint", "arc": "turn", "end": False},
            "exit": {"kind": "arc_endpoint", "arc": "turn", "end": True},
        },
        "keyframes": [
            {"step": 0, "target": "home", "grip": 0.02},
            {"step": 4, "target": "entry", "grip": 0.02},
            {"step": 20, "target": "exit", "grip": 0.02, "interpolation": "arc"},
        ],
    }


@pytest.fixture
def robot(tmp_path: Path) -> SimpleNamespace:
    urdf = tmp_path / "cartesian.urdf"
    names = ("x", "y", "z", "finger")
    links = "".join(
        f'<link name="{name}"/>' for name in ("base", "lx", "ly", "tip", "jaw")
    )
    joints = []
    for name, parent, child, axis in (
        ("x", "base", "lx", "1 0 0"),
        ("y", "lx", "ly", "0 1 0"),
        ("z", "ly", "tip", "0 0 1"),
        ("finger", "tip", "jaw", "0 1 0"),
    ):
        joints.append(
            f'<joint name="{name}" type="prismatic"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="{axis}"/>'
            '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>'
        )
    urdf.write_text(f'<robot name="cartesian">{links}{"".join(joints)}</robot>')
    parts = {"xyz": [0, 1, 2], "finger": [3]}
    return SimpleNamespace(
        joint_names=names,
        control_parts=parts,
        device="cpu",
        cfg=SimpleNamespace(
            fpath=urdf,
            init_qpos=[0.0, 0.0, 0.0, 0.02],
            init_pos=[0.0, 0.0, 0.0],
            init_rot=[0.0, 0.0, 0.0],
        ),
        get_joint_ids=lambda name=None: list(range(4)) if name is None else parts[name],
        get_qpos_limits=lambda: torch.tensor([[[-1.0, 1.0]] * 4]),
        get_qvel_limits=lambda **_: torch.ones(1, 4),
    )


def _scene(post: tuple[float, float, float]) -> Mock:
    pose = torch.eye(4)[None]
    pose[0, :3, 3] = torch.tensor(post)
    registry = Mock()
    registry.lookup.return_value.state_provider.observe.return_value.pose = pose
    return registry


def test_arc_is_replanned_around_its_observed_axis(trajectory, robot) -> None:
    declaration = _decode_cartesian_track(trajectory, path="track")
    first = _resolve_and_plan_track(declaration, robot, _scene((0.0, 0.0, 0.1)))
    shifted = _scene((0.1, -0.05, 0.1))
    second = _resolve_and_plan_track(declaration, robot, shifted)

    # The middle of a 450-degree sweep is at 225 degrees.
    np.testing.assert_allclose(
        first.positions[12, :3],
        [-0.03 / np.sqrt(2), -0.03 / np.sqrt(2), 0.14],
        atol=1e-5,
    )
    np.testing.assert_allclose(first.positions[-1], [0, 0.03, 0.18, 0.02], atol=1e-5)
    np.testing.assert_allclose(
        second.positions[4:, :3] - first.positions[4:, :3],
        np.tile([0.1, -0.05, 0.0], (17, 1)),
        atol=1e-5,
    )
    shifted.lookup.assert_called_once_with("post")


def test_track_cannot_exceed_the_robot_speed_on_its_authored_grid(
    trajectory, robot
) -> None:
    robot.get_qvel_limits = lambda **_: torch.full((1, 4), 0.01)
    with pytest.raises(ValueError, match="Trajectory interval.*velocity limit"):
        declaration = _decode_cartesian_track(trajectory, path="track")
        _resolve_and_plan_track(declaration, robot, _scene((0.0, 0.0, 0.1)))


def test_track_rejects_a_goal_outside_the_robot_workspace(trajectory, robot) -> None:
    declaration = _decode_cartesian_track(trajectory, path="track")
    with pytest.raises(ValueError, match="Unreachable TCP target"):
        _resolve_and_plan_track(declaration, robot, _scene((2.0, 0.0, 0.1)))
