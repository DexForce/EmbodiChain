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

import numpy as np
import pytest
from shapely.geometry import Polygon, box
import trimesh

from embodichain.gen_sim.task_engine._task_program.placement_proposals import (
    generate_on_targets,
)


def _box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _targets(mesh, reference, *, rotations=None, preferred=(0, 0)):
    return generate_on_targets(
        mesh,
        reference,
        [np.eye(3)] if rotations is None else rotations,
        preferred,
        clearance_m=0.01,
        contact_band_m=0.02,
    )


def test_offset_mesh_origin_rotates_real_bottom_without_mutating_inputs():
    mesh = _box((0.4, 0.4, 0.4), (0.2, 0.1, 0.3))
    reference = _box((2, 2, 0.2), (0, 0, 0.9))
    original, support = mesh.vertices.copy(), reference.vertices.copy()
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    proposals = _targets(mesh, reference, rotations=[rotation])

    assert proposals
    for result in proposals:
        pose = result["pose"]
        world = original @ pose[:3, :3].T + pose[:3, 3]
        assert world[:, 2].min() == pytest.approx(1.01)
        assert np.array_equal(pose[:3, :3], rotation)
        assert result["center_mass_source"] == "uniform_convex_hull_not_native"
        assert result["native_bearing_verified"] is False
    np.testing.assert_array_equal(mesh.vertices, original)
    np.testing.assert_array_equal(reference.vertices, support)


def test_batched_arm_preferences_retain_both_nearest_sides_without_mutation():
    mesh = _box((0.4, 0.4, 0.4), (0, 0, 0))
    reference = _box((2, 2, 0.2), (0, 0, 0.9))
    preferred = np.array([[-3.0, 0.0], [3.0, 0.0]])
    original = preferred.copy()

    proposals = _targets(mesh, reference, preferred=preferred)

    centers = [result["pose"][:2, 3] for result in proposals]
    assert any(np.allclose(center, (-0.8, 0)) for center in centers)
    assert any(np.allclose(center, (0.8, 0)) for center in centers)
    assert len({tuple(center) for center in centers}) == len(centers)
    np.testing.assert_array_equal(preferred, original)


def _ring():
    return trimesh.util.concatenate(
        [_box((2, 0.4, 0.2), (0, y, 0.9)) for y in (-0.8, 0.8)]
        + [_box((0.4, 1.2, 0.2), (x, 0, 0.9)) for x in (-0.8, 0.8)]
    )


def test_actual_projection_preserves_through_hole_in_final_footprint_check():
    reference = _ring()
    proposals = _targets(_box((0.2, 0.2, 0.2), (0, 0, 0)), reference)
    bearing_silhouette = box(-1, -1, 1, 1).difference(box(-0.6, -0.6, 0.6, 0.6))

    assert proposals
    assert all(
        bearing_silhouette.covers(Polygon(result["footprint"])) for result in proposals
    )
    assert _targets(_box((1.4, 1.4, 0.2), (0, 0, 0)), reference) == []


def test_disjoint_upper_patches_can_jointly_support_com_over_a_lower_groove():
    base = _box((2, 2, 0.1), (0, 0, 0.85))
    rails = [_box((0.4, 2, 0.1), (x, 0, 0.95)) for x in (-0.5, 0.5)]
    reference = trimesh.util.concatenate([base, *rails])

    proposals = _targets(_box((1.4, 1.0, 0.2), (0, 0, 0)), reference)

    assert proposals
    assert any(np.allclose(result["pose"][:2, 3], (0, 0)) for result in proposals)
    assert all(result["native_bearing_verified"] is False for result in proposals)


def test_contained_footprint_is_insufficient_when_contact_hull_misses_com():
    base = _box((2, 2, 0.1), (0, 0, 0.75))
    rail = _box((0.2, 2, 0.2), (-0.6, 0, 0.9))
    reference = trimesh.util.concatenate([base, rail])

    assert _targets(_box((1.4, 1.0, 0.2), (0, 0, 0)), reference) == []


def test_nonwatertight_source_uses_explicit_hull_prior_not_native_com():
    mesh = _box((0.4, 0.4, 0.4), (0, 0, 0))
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    assert not mesh.is_watertight

    proposals = _targets(mesh, _box((2, 2, 0.2), (0, 0, 0.9)))

    assert proposals
    assert all(
        result["center_mass_source"] == "uniform_convex_hull_not_native"
        and not result["native_bearing_verified"]
        for result in proposals
    )


@pytest.mark.parametrize("distance", [-0.01, float("nan"), float("inf")])
def test_invalid_contact_band_is_rejected(distance):
    mesh = _box((0.4, 0.4, 0.4), (0, 0, 0))
    with pytest.raises(ValueError, match="distances"):
        generate_on_targets(mesh, mesh, [np.eye(3)], (0, 0), 0, distance)


def test_invalid_rotation_is_rejected_without_projection_shortcut():
    mesh = _box((0.4, 0.4, 0.4), (0, 0, 0))
    with pytest.raises(ValueError, match="rotations"):
        _targets(mesh, _box((2, 2, 0.2), (0, 0, 0.9)), rotations=[np.ones((3, 3))])
