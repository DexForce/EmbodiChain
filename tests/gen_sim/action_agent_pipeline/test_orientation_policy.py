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

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.domain.orientation_policy import (
    principal_local_axis_order,
    resolve_target_rotation,
    rotated_local_z_min,
)
from embodichain.gen_sim.action_agent_pipeline.generation import relative_geometry
from embodichain.gen_sim.action_agent_pipeline.runtime import object_pose

_IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_CENTERED_CUBOID_BOUNDS = ((-3.0, -2.0, -1.0), (3.0, 2.0, 1.0))
_BOTTOM_ORIGIN_CUBOID_BOUNDS = ((-3.0, -2.0, 0.0), (3.0, 2.0, 2.0))


def test_principal_axis_order_uses_axis_index_to_break_extent_ties() -> None:
    bounds = ((-1.0, -2.0, -2.0), (1.0, 2.0, 2.0))

    assert principal_local_axis_order(bounds) == (1, 2, 0)


def test_lay_flat_uses_shortest_axis_as_world_z() -> None:
    rotation = resolve_target_rotation(
        orientation_goal="lay_flat",
        local_bounds=_CENTERED_CUBOID_BOUNDS,
        current_rotation=_IDENTITY_ROTATION,
        object_label="generic_block",
    )

    zmin = rotated_local_z_min(
        _cuboid_vertices(*_CENTERED_CUBOID_BOUNDS),
        rotation,
    )

    assert zmin == pytest.approx(-1.0)
    assert zmin != pytest.approx(-2.0)


def test_rotated_z_min_preserves_bottom_origin_instead_of_using_half_extent() -> None:
    rotation = resolve_target_rotation(
        orientation_goal="lay_flat",
        local_bounds=_BOTTOM_ORIGIN_CUBOID_BOUNDS,
        current_rotation=_IDENTITY_ROTATION,
        object_label="generic_block",
    )

    zmin = rotated_local_z_min(
        _cuboid_vertices(*_BOTTOM_ORIGIN_CUBOID_BOUNDS),
        rotation,
    )

    assert zmin == pytest.approx(0.0)


def test_normalized_label_makes_local_z_the_upright_axis() -> None:
    rotation = resolve_target_rotation(
        orientation_goal="upright",
        local_bounds=_CENTERED_CUBOID_BOUNDS,
        current_rotation=_IDENTITY_ROTATION,
        object_label="can_0",
    )

    assert _matrix_vector_mul(rotation, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )


def test_bottle_like_geometry_preserves_preview_secondary_direction() -> None:
    bounds = ((-3.0, -1.55, -1.5), (3.0, 1.55, 1.5))
    preview_rotation = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    rotation = resolve_target_rotation(
        orientation_goal="upright",
        local_bounds=bounds,
        current_rotation=preview_rotation,
        object_label="generic_object",
    )

    assert _matrix_vector_mul(rotation, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert _matrix_vector_mul(rotation, (0.0, 1.0, 0.0))[:2] == pytest.approx(
        (-1.0, 0.0)
    )


def test_upright_skips_a_near_vertical_preview_secondary_axis() -> None:
    preview_rotation = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    )

    rotation = resolve_target_rotation(
        orientation_goal="upright",
        local_bounds=_CENTERED_CUBOID_BOUNDS,
        current_rotation=preview_rotation,
        object_label="can_0",
    )

    assert _matrix_vector_mul(rotation, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )


def test_generation_applies_nonuniform_scale_before_resolving_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit_bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    monkeypatch.setattr(
        relative_geometry,
        "_load_mesh_vertices",
        lambda _: _cuboid_vertices(*unit_bounds),
    )
    obj_config = {
        "uid": "scaled_block",
        "shape": {"fpath": "unused.glb"},
        "body_scale": [3.0, 2.0, 1.0],
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [0.0, 0.0, 0.0],
    }

    zmin = relative_geometry._target_local_zmin_for_orientation(
        obj_config,
        "lay_flat",
    )

    assert zmin == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("orientation_goal", "expected_zmin"),
    [
        ("upright", -3.0),
        ("lay_flat", -1.0),
    ],
)
def test_generation_and_runtime_use_the_same_canonical_policy(
    monkeypatch: pytest.MonkeyPatch,
    orientation_goal: str,
    expected_zmin: float,
) -> None:
    vertices = _cuboid_vertices(*_CENTERED_CUBOID_BOUNDS)
    tensor_vertices = torch.tensor(vertices, dtype=torch.float32)
    monkeypatch.setattr(
        relative_geometry,
        "_load_mesh_vertices",
        lambda _: vertices,
    )
    obj_config = {
        "uid": "generic_block",
        # Description-only semantics are intentionally ignored because Runtime
        # can guarantee only the canonical object label.
        "description": "bottle",
        "shape": {"fpath": "unused.glb"},
        "body_scale": [1.0, 1.0, 1.0],
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [0.0, 0.0, 0.0],
    }
    monkeypatch.setattr(
        object_pose,
        "_held_object_mesh_vertices",
        lambda _state, _device: tensor_vertices,
    )
    state = SimpleNamespace(
        held_objects={
            "arm": SimpleNamespace(
                semantics=SimpleNamespace(label="generic_block"),
            )
        }
    )
    env = SimpleNamespace(robot=SimpleNamespace(device=torch.device("cpu")))
    current_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)

    generation_zmin = relative_geometry._target_local_zmin_for_orientation(
        obj_config,
        orientation_goal,
    )
    runtime_rotation = object_pose._resolve_object_orientation(
        env,
        {"orientation_goal": orientation_goal},
        current_pose,
        state,
    )
    runtime_zmin = object_pose._target_local_zmin_after_rotation(
        tensor_vertices,
        runtime_rotation,
    )

    assert generation_zmin == pytest.approx(expected_zmin)
    assert float(runtime_zmin) == pytest.approx(expected_zmin)


def _cuboid_vertices(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    return [
        (x, y, z)
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]


def _matrix_vector_mul(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )
