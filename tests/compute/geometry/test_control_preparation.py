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

"""Complete-interval support and bounded control-grasp dimensions preserve source rules."""

from __future__ import annotations
from pathlib import Path
import json
import math
from embodichain.compute.geometry.revolute_support import revolute_swept_min_z
import numpy as np
import pytest
from embodichain.compute.geometry.control_grasp_scale import select_control_grasp_scale
from embodichain.compute.geometry.control_resize import (
    control_bounds_overlap,
    control_clearance_translations,
)
from embodichain.compute.geometry.press_calibration import source_press_release_bias

CASES = json.loads((Path(__file__).parent / "data/control_scale_b0.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_bounded_control_grasp_scale_matches_original_source(case):
    kwargs = dict(
        source_scale=case["scale"],
        source_grip_depth=case["depth"],
        source_grip_width=case["width"],
        finger_thickness=0.01,
        minimum_opening_width=0.0,
        maximum_opening_width=0.13,
        opening_margin=0.003,
        calibration={
            "maximum_opening_width": 0.128,
            "minimum_knob_grip_depth": 0.0075,
            "pad_collision_extents": [0.027, 0.065, 0.0075],
            "basis": "canonical_pad_mesh_and_7p5mm_grip_depth_physical_qualification",
        },
        grip_depth_candidate=case["candidate"],
    )
    if "error" in case:
        with pytest.raises(ValueError, match="cannot satisfy"):
            select_control_grasp_scale(**kwargs)
    else:
        scale, audit = select_control_grasp_scale(**kwargs)
        audit.update(object_id="control", arm="left", source_sha256="source")
        assert scale == case["result"] and audit == case["audit"]


def test_control_relayout_keeps_stable_axis_order_and_overlap_threshold():
    bounds = np.array([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.1]])
    proposals = control_clearance_translations(bounds, bounds, 0.02)
    assert [name for _, name in proposals] == [
        "x_negative",
        "x_positive",
        "y_negative",
        "y_positive",
    ]
    np.testing.assert_array_equal(proposals[0][0], [-0.22, 0.0])
    assert control_bounds_overlap(bounds, bounds)
    moved = bounds.copy()
    moved[:, 0] += 0.2
    assert not control_bounds_overlap(bounds, moved)


def test_press_release_bias_retains_original_clamps_and_rejects_invalid_input():
    assert source_press_release_bias(1e-6, 0.001) == (500.0, 0.044721359549995794, 5.0)
    with pytest.raises(ValueError, match="positive mass and stroke"):
        source_press_release_bias(0.0, 0.001)
    with pytest.raises(ValueError, match="positive mass and stroke"):
        source_press_release_bias(0.1, float("nan"))


def test_revolute_support_considers_interior_angle_minimum_and_exact_limits():
    # Rotating a point around Y crosses z=-1 between the two equal z=0 endpoints.
    point = np.array([[1.0, 0.0, 0.0]])
    minimum = revolute_swept_min_z(
        point, np.eye(3), np.zeros(3), np.array([0.0, 1.0, 0.0]), (0.0, math.pi), 2.0
    )
    assert minimum == -2.0
    assert (
        revolute_swept_min_z(
            point, np.eye(3), np.zeros(3), np.array([0.0, 1.0, 0.0]), (0.0, 0.0), 2.0
        )
        == 0.0
    )
