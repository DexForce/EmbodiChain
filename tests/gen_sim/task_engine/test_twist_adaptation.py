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

from copy import deepcopy
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program.twist_adaptation import (
    select_twist_scale,
)


def _profile(thickness=0.010, opening=0.128, model_id="parallel_jaw"):
    return {
        "skill_profile": {
            "runtime_services": {
                "grasp_pose_generators": {
                    "right_eef": {
                        "opening_margin": 0.003,
                        "model": {
                            "model_id": model_id,
                            "finger_thickness": thickness,
                            "min_opening_width": 0.002,
                            "max_opening_width": opening,
                        },
                    }
                }
            }
        }
    }


def _route(depth, width, scale=0.2):
    return SimpleNamespace(
        arm="right",
        binding=SimpleNamespace(
            object_id="any_calibrated_control",
            source_sha256="qualified_source_hash",
            scale=scale,
            grip_depth=depth,
            grip_width=width,
        ),
    )


def test_small_knob_uniform_scale_uses_measured_depth_and_selected_gripper():
    route = _route(0.003, 0.020)
    profile = _profile()
    before = deepcopy(profile)
    scale, audit = select_twist_scale(route, profile)
    assert scale == pytest.approx(0.2 * 10 / 3)
    assert audit["candidate_grip_depth"] == pytest.approx(0.010)
    assert audit["candidate_grip_width"] == pytest.approx(0.020 * 10 / 3)
    assert audit["reasons"] == ["insufficient_axial_grip_depth"]
    assert (
        audit["qualification"] == "dimensional_candidate_only_runtime_checks_required"
    )
    assert profile == before


@pytest.mark.parametrize("depth,width", [(0.010, 0.067), (0.015, 0.100)])
def test_normal_compatible_knob_is_not_rescaled_to_preferred_depth(depth, width):
    scale, audit = select_twist_scale(_route(depth, width), _profile())
    assert scale is None
    assert audit["reasons"] == []
    assert audit["candidate_body_scale"] == 0.2
    assert audit["candidate_grip_depth"] == depth


@pytest.mark.parametrize("depth", [0.0075, 0.0094, 0.010])
def test_canonical_hand_preserves_already_qualified_compatible_grip_depth(depth):
    scale, audit = select_twist_scale(
        _route(depth, 0.060), _profile(model_id="robotiq_arg2f_140")
    )
    assert scale is None
    assert audit["minimum_knob_grip_depth"] == 0.0075
    assert audit["finger_thickness"] == 0.010
    assert audit["pad_collision_extents"] == [0.027, 0.065, 0.0075]
    assert audit["depth_screening_basis"] == (
        "canonical_pad_mesh_and_7p5mm_grip_depth_physical_qualification"
    )


def test_canonical_hand_adapts_only_below_qualified_minimum():
    scale, audit = select_twist_scale(
        _route(0.005, 0.035), _profile(model_id="robotiq_arg2f_140")
    )
    assert scale == pytest.approx(0.4)
    assert audit["candidate_grip_depth"] == 0.010


def test_canonical_hand_opening_uses_measured_gap_not_larger_profile_estimate():
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_twist_scale(
            _route(0.010, 0.13),
            _profile(opening=0.15, model_id="robotiq_arg2f_140"),
        )


def test_wide_knob_can_shrink_only_when_depth_still_fits():
    scale, audit = select_twist_scale(_route(0.020, 0.200), _profile())
    assert scale == pytest.approx(0.1)
    assert audit["candidate_grip_width"] == pytest.approx(0.1)
    assert audit["reasons"] == ["outside_usable_gripper_opening"]


@pytest.mark.parametrize("depth,width", [(0.003, 0.1), (0.010, 0.2)])
def test_preferred_candidate_rejects_incompatible_width(depth, width):
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_twist_scale(_route(depth, width), _profile())


def test_selected_finger_thickness_is_not_assumed_from_task_or_robot_id():
    with pytest.raises(ValueError, match="cannot satisfy"):
        select_twist_scale(_route(0.005, 0.02), _profile(thickness=0.012))
    scale, audit = select_twist_scale(_route(0.008, 0.02), _profile(thickness=0.006))
    assert scale is None
    assert audit["finger_thickness"] == 0.006


def test_selected_arm_owns_its_dimensions():
    profile = _profile()
    generators = profile["skill_profile"]["runtime_services"]["grasp_pose_generators"]
    generators["left_eef"] = deepcopy(generators["right_eef"])
    generators["left_eef"]["model"]["finger_thickness"] = 0.005
    route = _route(0.008, 0.02)
    route.arm = "left"
    scale, audit = select_twist_scale(route, profile)
    assert scale is None
    assert audit["arm"] == "left"


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf")])
def test_unmeasured_or_invalid_dimensions_fail_closed(bad):
    with pytest.raises(ValueError, match="measured positive dimensions"):
        select_twist_scale(_route(bad, 0.02), _profile())


def test_opening_margin_must_leave_a_usable_interval():
    with pytest.raises(ValueError, match="usable opening interval"):
        select_twist_scale(_route(0.015, 0.02), _profile(opening=0.004))
