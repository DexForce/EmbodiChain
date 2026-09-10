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

"""CPU checks for consistent URDF frame semantics at both PK import entries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from embodichain.lab.sim.utility import solver_utils
from embodichain.lab.sim.utility.import_utils import lazy_import_pytorch_kinematics

_URDF = b"""<robot name="link-origin-regression">
  <link name="base"><origin xyz="0.7 0 0" rpy="0 0 0"/></link>
  <link name="arm"/>
  <link name="tip">
    <origin xyz="-0.01 0 0" rpy="0 0 0"/>
    <visual><origin xyz="0.4 0.5 0.6" rpy="0 0 0.2"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry></visual>
    <collision><origin xyz="0.7 0.8 0.9" rpy="0.1 0 0"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry></collision>
    <inertial><origin xyz="0.01 0.02 0.03" rpy="0 0.1 0"/>
      <mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="rotation" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0.2 0.3 0.4" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="tool" type="fixed">
    <parent link="arm"/><child link="tip"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""


@pytest.fixture
def urdf_path(tmp_path: Path) -> Path:
    """Use a temporary asset with both standard and nonstandard origins."""
    result = tmp_path / "robot.urdf"
    result.write_bytes(_URDF)
    return result


@pytest.mark.parametrize("serial", [False, True])
def test_pk_file_import_uses_joint_frames_without_extra_link_offset(
    urdf_path: Path, serial: bool
) -> None:
    if serial:
        chain = solver_utils.create_pk_serial_chain(
            str(urdf_path), torch.device("cpu"), "tip", "base"
        )
    else:
        chain = solver_utils.create_pk_chain(str(urdf_path), torch.device("cpu"))
    qpos = torch.tensor([[0.0], [torch.pi / 2]])
    fk = chain.forward_kinematics(qpos)
    pose = (fk if serial else fk["tip"]).get_matrix()
    # The fixed 0.1m tool rotates with the revolute joint's standard Rz(pi/2).
    expected_positions = torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.3, 0.4]])
    torch.testing.assert_close(pose[:, :3, 3], expected_positions, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        pose[0, :3, :3],
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
        rtol=0,
    )
    assert urdf_path.read_bytes() == _URDF


@pytest.mark.parametrize("serial", [False, True])
def test_pk_file_import_preserves_all_standard_origin_elements(
    urdf_path: Path, serial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[bytes] = []

    def build(data: bytes, **_: object) -> SimpleNamespace:
        received.append(data)
        return SimpleNamespace(to=lambda **kwargs: None)

    monkeypatch.setattr(
        solver_utils,
        "lazy_import_pytorch_kinematics",
        lambda: SimpleNamespace(
            build_chain_from_urdf=build, build_serial_chain_from_urdf=build
        ),
    )
    if serial:
        solver_utils.create_pk_serial_chain(str(urdf_path), torch.device("cpu"), "tip")
    else:
        solver_utils.create_pk_chain(str(urdf_path), torch.device("cpu"))
    original, normalized = ET.fromstring(_URDF), ET.fromstring(received[0])
    assert normalized.findall("link/origin") == []
    for xpath in (
        "joint/origin",
        "link/visual/origin",
        "link/collision/origin",
        "link/inertial/origin",
    ):
        assert [node.attrib for node in normalized.findall(xpath)] == [
            node.attrib for node in original.findall(xpath)
        ]
    assert urdf_path.read_bytes() == _URDF


def test_supplied_pk_chain_keeps_caller_defined_link_frames() -> None:
    pk = lazy_import_pytorch_kinematics()
    chain = pk.build_chain_from_urdf(_URDF)
    frames = chain.forward_kinematics(torch.zeros(1, 1))
    before = frames["tip"].get_matrix().clone()
    root_relative = torch.linalg.inv(frames["base"].get_matrix()) @ before
    serial = solver_utils.create_pk_serial_chain(
        chain=chain,
        device=torch.device("cpu"),
        end_link_name="tip",
        root_link_name="base",
    )
    torch.testing.assert_close(
        serial.forward_kinematics(torch.zeros(1, 1)).get_matrix(), root_relative
    )
    torch.testing.assert_close(
        chain.forward_kinematics(torch.zeros(1, 1))["tip"].get_matrix(), before
    )
