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

"""Experimental placement post-policies and validators share one position bound."""

from __future__ import annotations

from copy import deepcopy

from embodichain.gen_sim.task_engine.experiment_policy import apply_placement_tolerances


def test_experimental_relative_post_policy_matches_its_position_validator() -> None:
    placement = {
        "kind": "placement",
        "entity": "fruit",
        "reference": "dish",
        "displacement": [0.0, 0.0, 0.07],
        "position_tolerance": 0.05,
        "duration": 3.0,
        "timeout": 12.0,
    }
    supported = {"kind": "supported_placement", "position_tolerance": 0.05}
    hold = {"kind": "hold", "position_tolerance": 0.05}
    constraint = {
        "presets": {"release": placement, "supported": supported, "hold": hold}
    }
    validator = {"kind": "object_near_relative_target", "position_tolerance": 0.05}
    other_validator = {"kind": "joint_configuration", "position_tolerance": 0.05}
    program = {"program": {"items": [{"validators": [validator, other_validator]}]}}
    before, hold_before, other_before = (
        deepcopy(placement),
        deepcopy(hold),
        deepcopy(other_validator),
    )
    apply_placement_tolerances(constraint, program)
    assert placement == {**before, "position_tolerance": 0.10}
    assert validator["position_tolerance"] == placement["position_tolerance"]
    assert 0.063270025 <= placement["position_tolerance"]
    assert 0.100001 > placement["position_tolerance"]
    assert hold == hold_before and other_validator == other_before
    assert supported == {
        "kind": "supported_placement",
        "position_tolerance": 0.10,
        "support_tolerance": 0.03,
        "com_tolerance": 0.10,
    }
