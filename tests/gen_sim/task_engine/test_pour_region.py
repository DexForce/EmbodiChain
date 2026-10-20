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
"""Pure receiver-region proposal tests, not physical pour evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon
import trimesh

from embodichain.gen_sim.task_engine._task_program import pour_region
from embodichain.gen_sim.task_engine._task_program.placement_proposals import (
    _projection,
)
from embodichain.gen_sim.task_engine._task_program.pour_geometry import (
    _top,
    staging_pose,
)


def geometry():
    source = np.asarray(trimesh.creation.box(extents=(0.1, 0.1, 0.2)).vertices)
    receiver = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    source_pose, receiver_pose = np.eye(4), np.eye(4)
    source_pose[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    receiver_pose[:3, 3] = [0.4, -0.3, 0.5]
    arm = np.array([3.0, -0.3, 0.2])
    return source, receiver, faces, source_pose, receiver_pose, arm


def call(values, budget=0.2):
    return pour_region.region_staging_pose(
        *values,
        clearance_m=0.025,
        gravity_world=(0, 0, -9.81),
        proxy_position_tolerance_m=budget,
    )


def anchor(values, pose):
    _, pivot = _top(values[0], pose[:3, :3])
    return pose[:3, :3] @ pivot + pose[:3, 3]


def test_region_moves_only_xy_and_uses_exact_inset_nearest_point():
    values = geometry()
    before = [value.copy() for value in values]
    result = call(values)
    old = staging_pose(
        values[0],
        values[1],
        *values[3:5],
        clearance_m=0.025,
        gravity_world=(0, 0, -9.81),
    )
    np.testing.assert_array_equal(result[:3, :3], old[:3, :3])
    assert result[2, 3] == old[2, 3]
    np.testing.assert_allclose(
        anchor(values, result)[:2], [1.2, -0.3], atol=1e-15, rtol=0
    )
    for actual, original in zip(values, before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_hole_is_not_filled_by_convex_hull_and_disconnected_regions_remain_separate():
    values = list(geometry())
    outer = np.array(
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
    )
    inner = outer * 0.25
    values[1] = np.concatenate((outer, inner))
    values[2] = np.array(
        [[i, (i + 1) % 4, 4 + (i + 1) % 4] for i in range(4)]
        + [[i, 4 + (i + 1) % 4, 4 + i] for i in range(4)]
    )
    values[4] = np.eye(4)
    values[5] = np.zeros(3)
    result = call(values)
    mesh = trimesh.Trimesh(vertices=values[1], faces=values[2], process=False)
    region = _projection(mesh).buffer(-0.2)
    xy = anchor(values, result)[:2]
    assert region.covers(Point(xy)) and np.linalg.norm(xy) > 0.5
    assert not _projection(mesh).covers(Point(0, 0))
    left = np.array(
        [[-3.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-3.0, 1.0, 0.0]]
    )
    values[1] = np.concatenate((left, left + [4.0, 0.0, 0.0]))
    values[2] = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    result = call(values)
    xy = anchor(values, result)[:2]
    assert abs(xy[0]) >= 1.2


@pytest.mark.parametrize("budget", [None, True, 0.0, -0.01, np.nan, np.inf, "0.02"])
def test_missing_nonpositive_nonfinite_or_coerced_budget_is_rejected(budget):
    with pytest.raises((ValueError, TypeError)):
        call(geometry(), budget)


@pytest.mark.parametrize(
    "damage",
    [
        "empty_faces",
        "float_faces",
        "bool_faces",
        "negative",
        "out_of_range",
        "wrong_shape",
        "nonfinite_vertices",
        "nonfinite_arm",
        "bad_source_R",
        "bad_receiver_R",
    ],
)
def test_invalid_geometry_and_frames_fail_without_fallback(damage):
    values = list(geometry())
    if damage == "empty_faces":
        values[2] = np.empty((0, 3), dtype=int)
    elif damage == "float_faces":
        values[2] = values[2].astype(float)
    elif damage == "bool_faces":
        values[2] = values[2].astype(bool)
    elif damage == "negative":
        values[2][0, 0] = -1
    elif damage == "out_of_range":
        values[2][0, 0] = len(values[1])
    elif damage == "wrong_shape":
        values[2] = np.array([0, 1, 2])
    elif damage == "nonfinite_vertices":
        values[1][0, 0] = np.nan
    elif damage == "nonfinite_arm":
        values[5][0] = np.inf
    else:
        values[3 if damage == "bad_source_R" else 4][0, 0] = 2
    with pytest.raises((ValueError, TypeError)):
        call(values)


def test_empty_or_invalid_region_is_unsupported_not_repaired(monkeypatch):
    with pytest.raises(ValueError):
        call(geometry(), 2.0)
    monkeypatch.setattr(
        pour_region,
        "_projection",
        lambda mesh: Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]),
    )
    with pytest.raises(ValueError):
        call(geometry())


def test_missing_budget_and_outside_nearest_point_have_no_fallback(monkeypatch):
    with pytest.raises(TypeError):
        pour_region.region_staging_pose(
            *geometry(), clearance_m=0.025, gravity_world=(0, 0, -9.81)
        )
    monkeypatch.setattr(
        pour_region, "nearest_points", lambda region, arm: (Point(99, 99), arm)
    )
    with pytest.raises(ValueError, match="membership"):
        call(geometry())


def test_gravity_preserving_world_se3_equivariance_and_tilted_gravity_rejection():
    values = list(geometry())
    result = call(values)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("z", 0.6).as_matrix()
    world[:3, 3] = [0.2, 0.4, 0.3]
    changed = [value.copy() for value in values]
    changed[3], changed[4] = world @ values[3], world @ values[4]
    changed[5] = world[:3, :3] @ values[5] + world[:3, 3]
    np.testing.assert_allclose(call(changed), world @ result, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="gravity"):
        pour_region.region_staging_pose(
            *values,
            clearance_m=0.025,
            gravity_world=(0, -1, -1),
            proxy_position_tolerance_m=0.2,
        )


@pytest.mark.parametrize("index", range(128))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nontrivial_source_receiver_and_arm_frames_do_not_create_numeric_rejection(
    index, dtype
):
    values = list(geometry())
    rng = np.random.default_rng(182)
    for _ in range(index + 1):
        source_pose, receiver_pose = np.eye(4), np.eye(4)
        source_pose[:3, :3] = Rotation.from_euler(
            "xyz", rng.uniform(-1, 1, 3)
        ).as_matrix()
        source_pose[:3, 3] = rng.uniform(-2, 2, 3)
        receiver_pose[:3, :3] = Rotation.from_euler("z", rng.uniform(-3, 3)).as_matrix()
        receiver_pose[:3, 3] = rng.uniform(-2, 2, 3)
        arm = rng.uniform(-4, 4, 3)
    values[3:] = [
        source_pose.astype(dtype),
        receiver_pose.astype(dtype),
        arm.astype(dtype),
    ]
    result = call(values)
    expected = staging_pose(
        values[0],
        values[1],
        *values[3:5],
        clearance_m=0.025,
        gravity_world=(0, 0, -9.81),
    )
    np.testing.assert_array_equal(result[:3, :3], expected[:3, :3])
    assert result[2, 3] == expected[2, 3]
    mesh = trimesh.Trimesh(
        vertices=values[1] @ values[4][:3, :3].T + values[4][:3, 3],
        faces=values[2],
        process=False,
    )
    assert _projection(mesh).covers(Point(anchor(values, result)[:2]))


def test_runtime_geometry_is_detached_readonly_and_calls_the_same_kernel():
    values = geometry()
    source, receiver, faces, source_pose, receiver_pose, arm = values
    runtime = pour_region.PourRegionGeometry(
        {"source": source, "receiver": receiver},
        {"receiver": faces},
        (0, 0, -9.81),
        0.2,
    )
    expected = call(values)
    np.testing.assert_array_equal(
        runtime.staging_pose(
            "source", "receiver", source_pose, receiver_pose, arm, clearance_m=0.025
        ),
        expected,
    )
    source[:] += 20
    receiver[:] += 20
    faces[:] = 0
    with pytest.raises(TypeError):
        runtime.vertices["other"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        runtime.faces["receiver"][0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        runtime.proxy_position_tolerance_m = 0
    np.testing.assert_array_equal(
        runtime.staging_pose(
            "source", "receiver", source_pose, receiver_pose, arm, clearance_m=0.025
        ),
        expected,
    )
