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
"""Concrete source hand roles reuse canonical embodiment geometry declarations."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from embodichain_robots.robotiq.task_program import robotiq140_task_hand_calibrations


def _component():
    root = Path(__file__).parents[3]
    return yaml.safe_load(
        (
            root
            / "extensions/embodichain_tasks/configs/components/embodiments/dual_franka_robotiq_arg2f_140.yaml"
        ).read_text()
    )


@pytest.mark.parametrize(
    "identity",
    [
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    ],
)
def test_hand_roles_reuse_explicit_owner_calibration_and_do_not_change_commands(
    identity,
):
    value = _component()
    value["embodiment_id"] = identity
    before = deepcopy(value)
    expected = value["skill_profile"]["runtime_services"]["hand_geometries"]
    resolved = robotiq140_task_hand_calibrations(value)
    assert [item.side for item in resolved] == ["left", "right"]
    assert {
        item.grasp_target_id: item.geometry_declaration() for item in resolved
    } == expected
    assert resolved[0].finger_link_names == (
        "left_outer_finger",
        "left_inner_finger",
        "left_inner_finger_pad",
        "left_right_outer_finger",
        "left_right_inner_finger",
        "left_right_inner_finger_pad",
    )
    assert resolved[1].finger_link_names == (
        "right_left_outer_finger",
        "right_left_inner_finger",
        "right_left_inner_finger_pad",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_finger_pad",
    )
    assert resolved[0].nominal_envelope_declaration()["envelope"] == dict(
        finger_length=0.13,
        finger_width=0.03,
        finger_thickness=0.01,
        palm_depth=0.08,
        opening_margin=0.003,
    )
    assert value == before
    expected["left_eef"]["sample_count"] = 3
    assert resolved[0].geometry_declaration()["sample_count"] == 65


def test_frozen_snapshot_requires_explicit_owner_geometry_and_keeps_resource_names():
    value = _component()
    geometry = value["skill_profile"]["runtime_services"].pop("hand_geometries")
    value["skill_profile"]["resources"][0]["resource_id"] = "custom_resource"
    with pytest.raises(ValueError, match="explicit embodiment geometry"):
        robotiq140_task_hand_calibrations(value)
    resolved = robotiq140_task_hand_calibrations(value, hand_geometries=geometry)
    assert resolved[0].resource_id == "custom_resource"
    assert "hand_geometries" not in value["skill_profile"]["runtime_services"]
    value["skill_profile"]["runtime_services"]["grasp_pose_generators"]["left_eef"][
        "model"
    ]["model_id"] = "unknown"
    with pytest.raises(ValueError, match="explicitly declared Robotiq"):
        robotiq140_task_hand_calibrations(value, hand_geometries=geometry)
