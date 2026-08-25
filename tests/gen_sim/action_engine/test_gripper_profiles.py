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

import pytest

from embodichain.gen_sim.action_engine.gripper_profiles import (
    GripperModel,
    get_gripper_profile,
)


def test_gripper_models_are_strictly_validated() -> None:
    assert get_gripper_profile("pgi").model is GripperModel.PGI
    assert get_gripper_profile("robotiq").model is GripperModel.ROBOTIQ

    for invalid in ("", "PGI", "robotiq ", "unknown", None):
        with pytest.raises((TypeError, ValueError), match="pgi.*robotiq"):
            get_gripper_profile(invalid)  # type: ignore[arg-type]


def test_pgi_profile_owns_asset_control_mimic_tcp_and_grasp_geometry() -> None:
    profile = get_gripper_profile("pgi")

    assert profile.asset_path == "DH_PGI_140_80/DH_PGI_140_80.urdf"
    assert profile.control_joint_names("left") == ("left_gripper_finger1_joint_1",)
    assert profile.mimic_joint_names("left") == ("left_gripper_finger2_joint_1",)
    assert profile.simulated_joint_names("left") == (
        "left_gripper_finger1_joint_1",
        "left_gripper_finger2_joint_1",
    )
    assert profile.open_positions == (0.0,)
    assert profile.close_positions == (0.04,)
    assert profile.control_limits == ((0.0, 0.04),)
    assert profile.tcp_transform == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.121),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert profile.grasp_model.model_id == "dh_pgi_140_80"
    assert profile.grasp_model.max_opening_width == pytest.approx(0.100)
    assert profile.grasp_model.finger_length == pytest.approx(0.10)
    assert profile.grasp_model.opening_margin == pytest.approx(0.03)


def test_robotiq_profile_preserves_existing_rotation_and_joint_semantics() -> None:
    profile = get_gripper_profile("robotiq")

    assert profile.asset_path == ("Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf")
    assert profile.control_joint_names("left") == (
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    )
    assert profile.open_positions == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert profile.close_positions == (0.7, -0.7, 0.7, -0.7, -0.7, 0.7)
    assert profile.tcp_transform == (
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.2),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert profile.grasp_model.model_id == "robotiq_arg2f_140"
    assert profile.grasp_model.max_opening_width == pytest.approx(0.15)
    assert profile.grasp_model.finger_length == pytest.approx(0.13)
    assert profile.grasp_model.opening_margin == pytest.approx(0.01)


def test_profile_manifest_records_tcp_frame_and_transform_conventions() -> None:
    profile = get_gripper_profile("pgi")

    manifest = profile.runtime_manifest(
        tcp_parent_frames={"left": "left_ee_link", "right": "right_ee_link"}
    )

    assert manifest["model"] == "pgi"
    assert manifest["tcp"]["parent_frames"] == {
        "left": "left_ee_link",
        "right": "right_ee_link",
    }
    assert manifest["tcp"]["transform_direction"] == "parent_link_to_tcp"
    assert manifest["tcp"]["matrix_layout"] == "row_major_homogeneous_4x4"
    assert manifest["tcp"]["quaternion_order"] == "not_applicable"
    assert manifest["grasp_model"]["model_id"] == "dh_pgi_140_80"
