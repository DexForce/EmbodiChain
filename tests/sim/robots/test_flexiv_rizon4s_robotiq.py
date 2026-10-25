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
