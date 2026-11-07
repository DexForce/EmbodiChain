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

"""Frozen source proposals retain candidate order, geometry and unavailable space."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon
import trimesh
from embodichain.compute.geometry.placement_proposals import generate_on_targets

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/placement_proposals_b0.json").read_text()
)


@pytest.mark.parametrize(
    "case",
    _REFERENCE["cases"],
    ids=lambda case: case["name"] + str(len(case["preferred_xy"])),
)
def test_source_proposal_records_and_input_ownership(case):
    subject = trimesh.Trimesh(**_REFERENCE["subject"], process=False)
    support = trimesh.Trimesh(**_REFERENCE["support"], process=False)
    vertices = subject.vertices.copy(), support.vertices.copy()
    results = generate_on_targets(
        subject,
        support,
        [np.asarray(x) for x in _REFERENCE["rotations"]],
        case["preferred_xy"],
        0.01,
        0.02,
        occupied_footprints=[Polygon(x) for x in case["occupied"]],
        separation_m=0.02,
    )
    actual = json.loads(json.dumps(results, default=np.ndarray.tolist))
    assert actual == [_REFERENCE["proposals"][index] for index in case["expected"]]
    np.testing.assert_array_equal(subject.vertices, vertices[0])
    np.testing.assert_array_equal(support.vertices, vertices[1])


def test_numeric_placement_extents_keep_alignment_and_articulation_root_edges():
    from embodichain.compute.geometry.placement_dimensions import (
        dominant_local_axis,
        vertical_placement_bounds,
        horizontal_placement_extent,
        horizontal_placement_distance,
    )

    vertices = trimesh.creation.box(extents=(0.1, 0.2, 0.6)).vertices
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert dominant_local_axis(vertices) == [0.0, 0.0, 1.0]
    for aligned, vertical, horizontal in (
        (False, (-0.05, 0.05), 0.3),
        (True, (-0.3, 0.3), 0.1),
    ):
        assert (
            vertical_placement_bounds(vertices, rotation, axis_aligned=aligned)
            == vertical
        )
        assert (
            horizontal_placement_extent(
                vertices, rotation, world_axis=0, axis_aligned=aligned
            )
            == horizontal
        )
    kwargs = dict(
        world_axis=1,
        object_axis_aligned=False,
        reference_axis_aligned=False,
        minimum=0.1,
        relation_clearance=0.02,
        axis_aligned_clearance=0.04,
    )
    for articulated, direction, expected in [
        (False, 1, 0.22),
        (True, 1, 0.62),
        (True, -1, 0.1),
    ]:
        value = horizontal_placement_distance(
            vertices,
            np.eye(3),
            vertices + [0, 0.4, 0],
            np.eye(3),
            reference_is_articulation=articulated,
            reference_direction=direction,
            **kwargs,
        )
        assert value == pytest.approx(expected)
    with pytest.raises(ValueError, match="unique dominant"):
        dominant_local_axis(trimesh.creation.box().vertices)


@pytest.mark.parametrize(
    "extent,vertical,expected",
    [
        ((0.02, 0.02, 0.06), 0.0, True),
        ((0.02, 0.02, 0.059), 0.0, False),
        ((0.02, 0.02, 0.06), 0.5, False),
        ((0.02, 0.02, 0.06), np.nextafter(0.5, 0.0), True),
        ((0.02, 0.02, 0.06), np.nextafter(0.5, 1.0), False),
        ((0.02, 0.02, 0.06), 0.499, True),
        ((0.0, 0.0, 0.06), 0.0, False),
    ],
)
def test_horizontal_full_mesh_selector_retains_source_boundary(
    extent, vertical, expected
):
    from embodichain.compute.geometry.placement_dimensions import (
        horizontal_slender_axis,
    )

    vertices = np.asarray([[0.0, 0.0, 0.0], extent])
    # Keep the tested component exact: cos(arccos(0.5)) can round below 0.5.
    horizontal = np.sqrt(1 - vertical**2)
    rotation = np.asarray(
        [
            [vertical, 0.0, horizontal],
            [0.0, 1.0, 0.0],
            [-horizontal, 0.0, vertical],
        ]
    )
    assert horizontal_slender_axis(vertices, rotation) is expected
