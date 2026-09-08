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

import math

import numpy as np
import pytest

from embodichain.lab.visualization.picker import ScenePicker


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [1, 5, 6],
            [1, 6, 2],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=np.int64,
    )
    return vertices, faces


IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_pick_hits_top_face_of_unit_cube() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        [("cubeA", "cube", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ)],
    )

    assert hit == "cubeA"


def test_pick_returns_none_when_ray_misses() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    hit = picker.pick(
        np.array([5.0, 5.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        [("cubeA", "cube", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ)],
    )

    assert hit is None


def test_pick_respects_translated_instance() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    # The cube is translated to x=3; a ray straight down at x=3 hits it.
    hit = picker.pick(
        np.array([3.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        [("cubeB", "cube", np.array([3.0, 0.0, 0.0], dtype=np.float32), IDENTITY_WXYZ)],
    )

    assert hit == "cubeB"


def test_pick_returns_closest_instance() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    instances = [
        ("near", "cube", np.array([0.0, 0.0, 0.0], dtype=np.float32), IDENTITY_WXYZ),
        ("far", "cube", np.array([0.0, 0.0, 2.0], dtype=np.float32), IDENTITY_WXYZ),
    ]
    # Ray from z=5 going -z hits "far" (top at z=2.5) before "near" (top at z=0.5).
    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        instances,
    )

    assert hit == "far"


def test_pick_respects_rotated_instance() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    # 90-degree rotation about y: the +z face now points along +x.
    angle = math.pi / 2.0
    wxyz = np.array([math.cos(angle), 0.0, math.sin(angle), 0.0], dtype=np.float32)
    hit = picker.pick(
        np.array([5.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        [("rotated", "cube", np.zeros(3, dtype=np.float32), wxyz)],
    )

    assert hit == "rotated"


def test_pick_skips_instances_with_unknown_geometry() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    instances = [
        ("unknown", "missing", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ),
        ("cubeA", "cube", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ),
    ]
    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        instances,
    )

    assert hit == "cubeA"


def test_pick_with_no_instances_returns_none() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        [],
    )

    assert hit is None


def test_pick_with_empty_geometry_is_skipped() -> None:
    picker = ScenePicker()
    picker.set_geometry(
        "empty", np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    )
    picker.set_geometry("cube", *_unit_cube())

    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        [("cubeA", "cube", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ)],
    )

    assert hit == "cubeA"


def test_set_geometry_rejects_bad_shapes() -> None:
    picker = ScenePicker()
    with pytest.raises(ValueError, match="vertices"):
        picker.set_geometry(
            "bad", np.zeros((4,), dtype=np.float32), np.zeros((1, 3), dtype=np.int64)
        )
    with pytest.raises(ValueError, match="faces"):
        picker.set_geometry(
            "bad", np.zeros((3, 3), dtype=np.float32), np.zeros((3,), dtype=np.int64)
        )


def test_pick_rejects_bad_ray_shapes() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())
    with pytest.raises(ValueError, match="ray_origin"):
        picker.pick(
            np.zeros((2,), dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            [],
        )


def test_pick_normalizes_direction_so_distance_is_in_world_units() -> None:
    picker = ScenePicker()
    picker.set_geometry("cube", *_unit_cube())

    # An unnormalized direction should produce the same hit as the normalized one.
    hit = picker.pick(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -2.0], dtype=np.float32),
        [("cubeA", "cube", np.zeros(3, dtype=np.float32), IDENTITY_WXYZ)],
    )

    assert hit == "cubeA"
