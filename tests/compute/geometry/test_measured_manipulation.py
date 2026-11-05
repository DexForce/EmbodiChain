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

"""Measured manipulation numeric proposals retain frozen B0 fitting/selection."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pytest
from shapely.geometry import Polygon
from embodichain.compute.geometry.measured_manipulation import (
    fit_projected_preshape,
    PadProjectionStroke,
    propose_occupied_floor_landing,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/measured_manipulation_b0.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["preshape"], ids=lambda c: c["name"])
def test_last_native_pad_fit_matches_source_projection(case):
    points = np.asarray(case["target"])
    xs = np.asarray(REFERENCE["pad_strokes"][case["stroke"]])
    pads = np.empty((*xs.shape, 3))
    pads[..., 0], pads[..., 1:] = xs, REFERENCE["pad_yz"]
    expected = case["expected"]

    def selected():
        return fit_projected_preshape(
            points,
            *pads,
            pad_rest_offsets=tuple(case["rests"]),
            target_rest_offset=case["target_rest"],
        )

    if "error" in expected:
        with pytest.raises(ValueError, match="No native preshape"):
            selected()
        return
    before = points.copy(), pads[0].copy(), pads[1].copy()
    result = selected()
    assert result.fraction == expected["fraction"]
    assert result.target_width_m == expected["target_width_m"]
    assert list(result.fitting_indices) == expected["fitting_indices"]
    for actual, original in zip((points, *pads), before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("case", REFERENCE["landing"], ids=lambda c: c["name"])
def test_occupied_floor_proposal_matches_source_reservations(case):
    case = REFERENCE["landing_defaults"] | case
    expected = case["expected"]

    def selected():
        return propose_occupied_floor_landing(
            Polygon(case["floor"]),
            case["floor_height"],
            np.asarray(case["target_vertices"], dtype=np.float32),
            np.asarray(case["target_position"], dtype=np.float32),
            tuple(
                (name, np.asarray(points, dtype=np.float32))
                for name, points in case["contents"]
            ),
        )

    if "error" in expected:
        with pytest.raises(ValueError, match="No measured inside landing"):
            selected()
        return
    result = selected()
    assert result.metadata() == expected["metadata"]
    if expected["displacement"] is None:
        assert result.displacement_parent is None
    else:
        # The owning Atomic path materializes the parent displacement in its
        # original float32 release tensor after this pure numeric calculation.
        np.testing.assert_array_equal(
            np.asarray(result.displacement_parent, dtype=np.float32),
            np.asarray(expected["displacement"], dtype=np.float32),
        )


def test_unavailable_pad_projection_never_becomes_a_fitting_proposal():
    points = np.asarray([[0.0, 0.0, 0.0], [0.02, 0.02, 0.02]])
    pads = np.ones((5, 2, 3))
    with pytest.raises(ValueError, match="axis"):
        fit_projected_preshape(
            points, pads, pads, pad_rest_offsets=(0.0, 0.0), target_rest_offset=0.0
        )


def test_mesh_pose_pad_projection_matches_dense_without_stroke_materialization():
    vertices = np.asarray(
        [[-0.002, -0.01, -0.01], [0.002, 0.01, 0.01]], dtype=np.float64
    )
    poses = []
    for sign in (-1, 1):
        transform = np.repeat(np.eye(4)[None], 65, axis=0)
        transform[:, 0, 3] = sign * np.linspace(0.07, 0.01, 65)
        poses.append(transform)
    strokes = tuple(PadProjectionStroke((vertices,), value) for value in poses)
    dense = tuple(
        np.stack([value.sample(i) for i in range(len(value))]) for value in strokes
    )
    target = np.asarray([[-0.02, -0.01, -0.01], [0.02, 0.01, 0.01]])
    common = dict(pad_rest_offsets=(0.001, 0.001), target_rest_offset=0.001)
    assert fit_projected_preshape(target, *strokes, **common) == fit_projected_preshape(
        target, *dense, **common
    )
    vertices[:] = 100
    poses[0][:] = 0
    np.testing.assert_array_equal(strokes[0].sample(0), dense[0][0])
    with pytest.raises(ValueError):
        strokes[0].tcp_poses.setflags(write=True)
