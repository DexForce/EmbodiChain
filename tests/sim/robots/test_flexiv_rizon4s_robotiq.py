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
"""Public model conversion must preserve nonzero joint anchors and FK."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch


def test_compiled_robot_tree_preserves_nonzero_joint_frame(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    pk = pytest.importorskip("pytorch_kinematics")
    from embodichain_robots.flexiv.rizon4s_robotiq._menagerie import _append_model

    source = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base"><geom type="box" size=".1 .1 .1"/>
       <body name="arm" pos=".1 .2 .3" euler="15 20 5">
        <joint name="joint" axis="0 1 0" pos=".01 .02 .03" range="-90 90"/>
        <geom type="box" pos=".02 0 .1" size=".02 .03 .1" mass="1"/>
        <body name="tool" pos=".1 0 .2"><geom type="sphere" size=".01" mass=".01"/></body>
       </body>
      </body></worldbody></mujoco>""")
    urdf = ET.Element("robot", name="reference")
    _append_model(urdf, source, tmp_path)
    chain = pk.build_serial_chain_from_urdf(
        ET.tostring(urdf, encoding="unicode"),
        end_link_name="tool",
        root_link_name="base",
    ).to(dtype=torch.float64)
    data = mujoco.MjData(source)
    for angle in [-0.7, 0, 0.65]:
        data.qpos[0] = angle
        mujoco.mj_forward(source, data)
        actual = (
            chain.forward_kinematics(torch.tensor([[angle]], dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        tool_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "tool")
        expected = np.eye(4)
        expected[:3, 3] = data.xpos[tool_id]
        expected[:3, :3] = data.xmat[tool_id].reshape(3, 3)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_rizon_preset_qualified_model_gate_runs_before_simulation(
    tmp_path: Path,
) -> None:
    from embodichain_robots import FlexivRizon4SRobotiqCfg

    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="closed loops"):
        FlexivRizon4SRobotiqCfg.from_dict(
            {"source_urdf": str(urdf), "require_exact_dynamics": True}
        )
    cfg = FlexivRizon4SRobotiqCfg.from_dict({"source_urdf": str(urdf)})
    assert cfg.control_parts["arm"] == [f"joint{i}" for i in range(1, 8)]
    assert cfg.control_parts["eef"] == ["gripper_right_driver_joint"]
    restored = FlexivRizon4SRobotiqCfg.from_dict(cfg.to_dict())
    assert restored.control_parts == cfg.control_parts
    assert restored.source_urdf == str(urdf)


def test_default_asset_composition_keeps_materials_joint_names_and_arm_tcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    from embodichain_robots import FlexivRizon4SRobotiqCfg
    from embodichain_robots.flexiv.rizon4s_robotiq import _default_assets

    def base_link(name: str, color: str, mass: str) -> str:
        return f"""<link name="{name}"><inertial><mass value="{mass}"/>
          <inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>
          </inertial><visual><geometry><box size=".02 .02 .02"/></geometry>
          <material name=""><color rgba="{color}"/></material></visual>
          <collision><geometry><box size=".01 .01 .01"/></geometry></collision></link>"""

    arm = tmp_path / "arm.urdf"
    pieces = ['<robot name="arm">', base_link("base_link", ".97 .97 .97 1", "3")]
    for i in range(1, 8):
        parent = "base_link" if i == 1 else f"link{i-1}"
        pieces.append(
            f"""<link name="link{i}"/><joint name="joint{i}" type="revolute">
          <parent link="{parent}"/><child link="link{i}"/><origin xyz="0 0 .1"/>
          <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="2"/></joint>"""
        )
    pieces.append(
        '<link name="flange"/><joint name="flange_fixed" type="fixed"><parent link="link7"/><child link="flange"/></joint></robot>'
    )
    arm.write_text("".join(pieces))
    hand = tmp_path / "hand.urdf"
    pieces = [
        '<robot name="hand">',
        base_link("robotiq_85_base_link", ".1 .1 .1 1", ".4"),
    ]
    joints = [
        ("finger_joint", "left_outer_knuckle", None),
        ("left_inner_knuckle_joint", "left_inner_knuckle", 1),
        ("left_inner_finger_joint", "left_inner_finger", -1),
        ("right_inner_knuckle_joint", "right_inner_knuckle", -1),
        ("right_inner_finger_joint", "right_inner_finger", 1),
        ("right_outer_knuckle_joint", "right_outer_knuckle", -1),
    ]
    for name, child, multiplier in joints:
        mimic = (
            ""
            if multiplier is None
            else f'<mimic joint="finger_joint" multiplier="{multiplier}" offset="0"/>'
        )
        pieces.append(
            f'<link name="{child}"/><joint name="{name}" type="revolute"><parent link="robotiq_85_base_link"/><child link="{child}"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="2"/>{mimic}</joint>'
        )
    hand.write_text("".join(pieces) + "</robot>")
    assets = {
        "FlexivRizon4S/rizon4s.urdf": str(arm),
        "Robotiq2F85/Robotiq2F85.urdf": str(hand),
    }
    monkeypatch.setattr(_default_assets, "get_data_path", assets.__getitem__)
    monkeypatch.setattr(
        _default_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path / "cache")
    )
    arm_module = importlib.import_module("embodichain_robots.flexiv.rizon4s.cfg")
    monkeypatch.setattr(arm_module, "get_data_path", lambda path: str(arm))
    cfg = FlexivRizon4SRobotiqCfg.from_dict({"asset_source": "default"})
    root = ET.parse(cfg._pk_urdf_path).getroot()
    assert (
        root.find("link[@name='base_link']/visual/material/color").get("rgba")
        == ".97 .97 .97 1"
    )
    assert (
        root.find("link[@name='robotiq_85_base_link']/visual/material/color").get(
            "rgba"
        )
        == ".1 .1 .1 1"
    )
    assert len(root.findall("./link/collision")) == 2
    assert len(root.findall("./joint/mimic")) == 5
    assert cfg.control_parts["eef"] == ["finger_joint"]
    chain = cfg.build_pk_serial_chain()["arm"]
    assert chain.get_joint_parameter_names() == [f"joint{i}" for i in range(1, 8)]
    pose = chain.forward_kinematics(torch.zeros(1, 7)).get_matrix()[0]
    torch.testing.assert_close(pose[:3, 3], torch.tensor([0.0, 0.0, 0.845]))
    restored = FlexivRizon4SRobotiqCfg.from_dict(cfg.to_dict())
    assert (
        restored.asset_source == "default"
        and restored.control_parts == cfg.control_parts
    )
