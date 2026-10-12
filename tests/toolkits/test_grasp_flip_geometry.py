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

"""Geometry preflight must preserve both feasible cup-inversion directions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.tools.grasp_assemble._geometry import GraspGeometry, interpolate_poses


def test_opposite_rotation_preserves_pose_and_both_spacing_bounds() -> None:
    start = np.eye(4)
    end = np.eye(4)
    end[:3, :3] = Rotation.from_euler("x", 120, degrees=True).as_matrix()
    end[0, 3] = 1.7
    translation_step, angular_step = 0.01, 5.0
    path = np.asarray(
        interpolate_poses(
            start, end, translation_step, angular_step, rotation_direction="opposite"
        )
    )
    np.testing.assert_array_equal(path[[0, -1]], np.stack([start, end]))
    assert np.linalg.norm(np.diff(path[:, :3, 3], axis=0), axis=1).max() <= (
        translation_step + 1e-12
    )
    increments = Rotation.from_matrix(
        path[:-1, :3, :3].transpose(0, 2, 1) @ path[1:, :3, :3]
    ).as_rotvec()
    assert np.linalg.norm(increments, axis=1).max() <= np.deg2rad(angular_step)
    # +120 and -240 degrees have the same endpoint, but traverse different arcs.
    assert increments[:, 0].sum() == pytest.approx(np.deg2rad(-240))


@pytest.fixture
def geometry() -> GraspGeometry:
    result = GraspGeometry.__new__(GraspGeometry)
    result.config = {"motion": {"pre_grasp_distance": 0.1, "retract_distance": 0.1}}
    result.initial = np.eye(4)
    result.initial[2, 3] = 0.1
    result.target = result.initial.copy()
    result.target[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    result.target[0, 3] = 0.5
    lifted, rotated, hover = (
        result.initial.copy(),
        result.target.copy(),
        result.target.copy(),
    )
    lifted[2, 3] = 0.6
    rotated[:3, 3] = lifted[:3, 3]
    hover[2, 3] = lifted[2, 3]
    result.waypoints = [lifted, rotated, hover, result.target.copy()]
    # Isolate route preflight from the already-tested finger-contact search.
    result._assemble_collision = lambda index, pose, q: index in (1, 2) and q >= 0.02
    result.assemble_checks = [SimpleNamespace(distance=lambda pose: 0.0)] * 3
    result.object_check = object()
    result._object_clearance = lambda pose: None
    result._initial_support = lambda: {"accepted": True}
    result.waypoint_names = ("grasp", "lift", "rotate", "hover", "place")
    result._overlap = lambda check, pose: False
    return result


def test_shortest_flip_collision_can_use_opposite_flip(geometry: GraspGeometry) -> None:
    visited = []

    def clearance(pose: np.ndarray, q: float) -> str | None:
        # A protruding gripper part sweeps opposite half-spaces in the two flips.
        visited.append(pose[2, 1])
        return "gripper hits rack" if pose[2, 1] > 0.8 else None

    geometry._world_clearance = clearance
    report = geometry.evaluate(np.eye(4))
    assert report["accepted"], report["reasons"]
    assert min(visited) < -0.8 and max(visited) > 0.8
    np.testing.assert_array_equal(report["T_world_assemble_target"], geometry.target)


def test_both_blocked_flips_are_rejected(geometry: GraspGeometry) -> None:
    geometry._world_clearance = lambda pose, q: (
        "gripper hits rack" if abs(pose[2, 1]) > 0.8 else None
    )
    report = geometry.evaluate(np.eye(4))
    assert not report["accepted"]
    assert "both directions" in report["reasons"][0]


def test_lift_collision_still_rejects_candidate(geometry: GraspGeometry) -> None:
    geometry._world_clearance = lambda pose, q: (
        "blocked lift" if pose[0, 3] == 0 and 0.2 < pose[2, 3] < 0.5 else None
    )
    report = geometry.evaluate(np.eye(4))
    assert not report["accepted"]
    assert report["reasons"] == ["blocked lift"]
