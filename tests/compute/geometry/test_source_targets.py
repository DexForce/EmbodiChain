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

"""Original source orientation arithmetic stays independent of task/runtime owners."""

from __future__ import annotations
from pathlib import Path
import json
import pytest
from embodichain.compute.geometry.source_targets import (
    upright_source_quaternion,
    upright_origin_height,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/source_targets_b0.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_quaternion_target_matches_independent_complete_b0_reference(case):
    assert (
        upright_source_quaternion(case["euler"], case["axis"], world_yaw=case["yaw"])
        == case["quaternion"]
    )


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
    [v for v in PICK_REFERENCE["cases"] if v["name"].startswith("e1_")],
    ids=lambda row: row["name"],
)
def test_numeric_pick_approach_matches_original_complete_source_factory(row):
    from embodichain.compute.geometry.source_targets import constrained_pick_approach

    assert constrained_pick_approach(row["world_axis_before_pick"]) == row["approach"]
