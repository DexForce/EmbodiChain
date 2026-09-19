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

"""Conservative acceleration of original-solid collision queries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation


@pytest.fixture
def collision_module():
    pytest.importorskip("fcl")
    pytest.importorskip("manifold3d")
    from scripts.tools.assemble import _collision

    return _collision


def test_separated_bounds_skip_both_exact_engines(
    collision_module, monkeypatch
) -> None:
    box = trimesh.creation.box([0.1, 0.1, 0.1])
    check = collision_module.ExactCollision(box, box)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
    pose[0, 3] = 1.0

    def unexpected(*args):
        pytest.fail("Disjoint bounds must not invoke the exact narrowphase")

    monkeypatch.setattr(collision_module.fcl, "collide", unexpected)
    check.assemble_solid = SimpleNamespace(transform=unexpected)
    assert not check.surface_collision(pose)
    assert check.intersection_volume(pose) == 0
    # Boundary distance retains its exact meaning for separated objects.
    assert check.distance(pose) > 0.8


def test_rotated_offset_mesh_uses_its_transformed_bounds(collision_module) -> None:
    base = trimesh.creation.box([0.1, 0.1, 0.1])
    moving = trimesh.creation.box([0.3, 0.02, 0.02])
    moving.apply_translation([0.2, -0.1, 0.3])
    check = collision_module.ExactCollision(base, moving)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    # Place the rotated long axis through the base; rotating only the extents
    # without also transforming the off-origin center would miss this overlap.
    pose[:3, 3] = -pose[:3, :3] @ moving.bounds.mean(axis=0)
    assert check.surface_collision(pose)
    assert check.intersection_volume(pose) == pytest.approx(0.1 * 0.02 * 0.02)
    pose[0, 3] += 0.2
    assert not check.surface_collision(pose)
    assert check.intersection_volume(pose) == 0


def test_touching_bounds_preserve_surface_contact(collision_module) -> None:
    box = trimesh.creation.box([0.1, 0.1, 0.1])
    check = collision_module.ExactCollision(box, box)
    pose = np.eye(4)
    pose[0, 3] = 0.1
    assert check.surface_collision(pose)
    assert check.intersection_volume(pose) == 0


def test_contained_solid_still_reaches_boolean_intersection(collision_module) -> None:
    outer = trimesh.creation.box([1, 1, 1])
    inner = trimesh.creation.box([0.1, 0.1, 0.1])
    check = collision_module.ExactCollision(outer, inner)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    assert not check.surface_collision(pose)
    assert check.intersection_volume(pose) == pytest.approx(inner.volume)


def test_subnanometer_gap_is_not_rejected_by_broadphase(collision_module) -> None:
    box = trimesh.creation.box([0.1, 0.1, 0.1])
    check = collision_module.ExactCollision(box, box)
    calls = []
    original = check.assemble_solid

    def transform(pose):
        calls.append(pose)
        return original.transform(pose)

    check.assemble_solid = SimpleNamespace(transform=transform)
    pose = np.eye(4)
    pose[0, 3] = 0.1 + 1e-10
    check.intersection_volume(pose)
    assert len(calls) == 1


def test_accelerated_queries_match_exact_engines_for_rotated_boxes(
    collision_module,
) -> None:
    rng = np.random.default_rng(4)
    base = trimesh.creation.box([0.2, 0.3, 0.1])
    moving = trimesh.creation.box([0.1, 0.15, 0.2])
    check = collision_module.ExactCollision(base, moving)
    for _ in range(30):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        pose[:3, 3] = rng.uniform(-0.25, 0.25, 3)
        expected_volume = (
            check.base_solid ^ check.assemble_solid.transform(pose[:3, :4])
        ).volume()
        check.assemble.setTransform(
            collision_module.fcl.Transform(pose[:3, :3], pose[:3, 3])
        )
        expected_surface = bool(
            collision_module.fcl.collide(
                check.base,
                check.assemble,
                collision_module.fcl.CollisionRequest(),
                collision_module.fcl.CollisionResult(),
            )
        )
        assert check.surface_collision(pose) == expected_surface
        assert check.intersection_volume(pose) == pytest.approx(expected_volume)
