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

from pathlib import Path

import torch

from embodichain.lab.sim.utility.solver_utils import (
    create_pk_chain,
    create_pk_serial_chain,
)


SIMPLE_TWO_LINK_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _write_test_urdf(tmp_path: Path) -> Path:
    urdf_path = tmp_path / "two_link.urdf"
    urdf_path.write_text(SIMPLE_TWO_LINK_URDF)
    return urdf_path


def test_create_pk_serial_chain_from_existing_chain(tmp_path: Path) -> None:
    urdf_path = _write_test_urdf(tmp_path)
    chain = create_pk_chain(str(urdf_path), torch.device("cpu"))

    serial_chain = create_pk_serial_chain(
        chain=chain,
        end_link_name="link2",
        root_link_name="base_link",
        device=torch.device("cpu"),
    )

    qpos = torch.zeros(2, device=torch.device("cpu"))
    jacobian = serial_chain.jacobian(qpos)

    assert torch.device(serial_chain.device).type == "cpu"
    assert jacobian.device.type == "cpu"
    assert jacobian.shape == (1, 6, 2)
