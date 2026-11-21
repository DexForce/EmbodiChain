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
"""Initial mounting repair preserves the scene and the dual-arm assembly."""

from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
import trimesh
from embodichain.data_pipeline.real2sim import RobotDemonstration
from embodichain.data_pipeline.real2sim.adaptation.inputs import (
    CameraObservation,
    RobotModel,
    ReconstructionInput,
    RegisteredScene,
)
from embodichain.data_pipeline.real2sim.adaptation._kinematics import Robot
from embodichain.data_pipeline.real2sim.adaptation.placement import place_robot_base


@pytest.mark.parametrize("moving_arm", [False, True])
def test_mounting_repair_moves_both_arms_without_moving_scene_or_joints(
    tmp_path, moving_arm
):
    nodes = [
        '<link name="base"><collision><geometry><box size=".2 .2 .2"/></geometry></collision></link>'
    ]
    names = []
    for side, y in [("left", 0.3), ("right", -0.3)]:
        for i in range(8):
            name = f"{side}_{i}"
            names.append(name)
            parent = "base" if i == 0 else f"{side}_link{i-1}"
            collision = (
                '<collision><origin xyz="0 .12 0"/><geometry><box size=".02 .02 .02"/></geometry></collision>'
                if moving_arm and side == "left" and i == 0
                else ""
            )
            nodes.append(
                f'<link name="{side}_link{i}">{collision}</link><joint name="{name}" type="{"revolute" if i < 6 else "prismatic"}"><parent link="{parent}"/><child link="{side}_link{i}"/><origin xyz="0 {y if i == 0 else 0} 0"/><axis xyz="0 0 1"/><limit lower="{ -3 if i < 6 else 0}" upper="{3 if i < 6 else .05}" effort="10" velocity="2"/></joint>'
            )
    urdf = tmp_path / "dual.urdf"
    urdf.write_text('<robot name="dual">' + "".join(nodes) + "</robot>")
    model = RobotModel(
        urdf,
        tuple(names),
        ("left", "right"),
        ("left_link5", "right_link5"),
        np.eye(4),
        (),
        0,
    )
    q = np.zeros((2, 16))
    if moving_arm:
        q[1, 0] = -np.pi / 2
    demo = RobotDemonstration(
        source_id="test",
        source_revision="1",
        unit_id="1",
        joint_names=tuple(names),
        joint_units=(("rad",) * 6 + ("m",) * 2) * 2,
        qpos=q,
        timestamps=np.array([0.0, 1 / 30]),
        time_unit="s",
        robot_geometry_sha256="a" * 64,
        recording_sha256="b" * 64,
    )
    camera = CameraObservation(tmp_path / "v", tmp_path / "i", np.arange(2))
    inputs = ReconstructionInput(demo, model, {"front": camera}, "front", "Pick")
    table = trimesh.creation.box(extents=[1, 1, 0.1])
    table.apply_translation([0.55, 0, 0])
    scene = RegisteredScene(
        {"objects": [{"id": "table", "transform": np.eye(4).tolist()}]},
        {"table": table.vertices.copy()},
    )
    placed = place_robot_base(inputs, scene, clearance_m=0.005)
    pose = np.asarray(placed.description["robot_base_pose_world"])
    # Table starts at x=.05, robot ends at x=.10. Back off 50mm + 5mm gap.
    # The rotating arm is initially clear, but reaches x=.13 in the second frame.
    expected = -0.0851 if moving_arm else -0.0551
    np.testing.assert_allclose(pose[:3, 3], [expected, 0, 0], atol=2e-7)
    repeated = place_robot_base(
        inputs, placed, clearance_m=0.005, replace_existing=True
    )
    np.testing.assert_allclose(repeated.description["robot_base_pose_world"], pose)
    before, after = Robot(model), Robot(replace(model, base_pose_world=pose))
    for arm in [0, 1]:
        np.testing.assert_array_equal(before.arm_limits[arm][0], np.full(6, -3.0))
        np.testing.assert_array_equal(before.arm_limits[arm][1], np.full(6, 3.0))
        np.testing.assert_allclose(after.tcp(q, arm), pose @ before.tcp(q, arm))
    relative_before = np.linalg.inv(before.tcp(q, 0)) @ before.tcp(q, 1)
    relative_after = np.linalg.inv(after.tcp(q, 0)) @ after.tcp(q, 1)
    np.testing.assert_allclose(relative_before, relative_after, atol=1e-12)
    assert placed.description["objects"] == scene.description["objects"]
    assert "robot_base_pose_world" not in scene.description
    np.testing.assert_array_equal(scene.vertices["table"], table.vertices)
    np.testing.assert_array_equal(inputs.demonstration.qpos, q)
    np.testing.assert_array_equal(model.base_pose_world, np.eye(4))
    with pytest.raises(ValueError, match="rigid"):
        replace(model, base_pose_world=np.zeros((4, 4)))
