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

"""Frozen source classifications, target poses and numeric placement declarations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodichain.compute.geometry.source_relations import (
    source_orientation_label,
    source_support_proposal,
)
from embodichain.compute.geometry.source_targets import (
    upright_source_quaternion,
    upright_origin_height,
)

CASES = json.loads(
    (Path(__file__).parent / "data/source_relations_b0.json").read_text()
)


@pytest.mark.parametrize("case", CASES["orientation"])
def test_source_orientation_classification(case):
    assert source_orientation_label(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["support"])
def test_source_support_candidate_and_tie_order(case):
    assert (
        source_support_proposal(
            case["uid"],
            np.asarray(case["child"]),
            case["objects"],
            tolerance=case["tolerance"],
        )
        == case["expected"]
    )


REFERENCE = json.loads(
    (Path(__file__).parent / "data/source_targets_b0.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_quaternion_target_matches_independent_complete_b0_reference(case):
    actual = np.asarray(
        upright_source_quaternion(case["euler"], case["axis"], world_yaw=case["yaw"])
    )
    expected = np.asarray(case["quaternion"])
    # q and -q encode the same pose; backend rounding can choose either near ties.
    if actual @ expected < 0:
        expected = -expected
    tolerance = 4 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert np.linalg.norm(actual) == pytest.approx(1, rel=0, abs=tolerance)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, 0.91),
        ({"staging_clearance": 0.2}, 1.11),
        ({"staging_radius": 0.3, "staging_margin": 0.02}, 1.12),
        ({"staging_radius": 0.05, "staging_margin": 0.02}, 0.91),
    ],
)
def test_mesh_origin_and_rotation_reserve_have_independent_declared_meaning(
    kwargs, expected
):
    assert upright_origin_height(-0.1, 0.8, 0.01, **kwargs) == expected


@pytest.mark.parametrize(
    "bottom,top,expected", [(-0.2, 0.2, 0.1), (-0.04, 0.04, 0.07), (-0.01, 0.01, 0.05)]
)
def test_carry_lift_consumes_explicit_numeric_span_without_a_scene(
    bottom, top, expected
):
    from embodichain.compute.geometry.source_targets import bounded_carry_lift

    assert (
        bounded_carry_lift(bottom, top, minimum=0.05, maximum=0.1, reserve=0.03)
        == expected
    )


PICK_REFERENCE = json.loads(
    (Path(__file__).parent / "data/constrained_pick_approach_b0.json").read_text()
)


@pytest.mark.parametrize(
    "row",
    PICK_REFERENCE["cases"],
    ids=lambda row: row["name"],
)
def test_numeric_pick_approach_matches_original_complete_source_factory(row):
    from embodichain.compute.geometry.source_targets import constrained_pick_approach

    assert constrained_pick_approach(row["world_axis_before_pick"]) == row["approach"]
