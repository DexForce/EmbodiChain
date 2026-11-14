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

"""Read a known gravity reaction through the scheduled sensor contract."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, NewtonPhysicsCfg
from embodichain.lab.sim.sensors import SensorCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]

_URDF = """
<robot name="wrench">
  <link name="base"/>
  <link name="payload">
    <inertial><origin xyz="0.1 0 0"/><mass value="1"/>
      <inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><box size=".1 .1 .1"/></geometry></collision>
  </link>
  <link name="carrier">
    <inertial><mass value="1"/>
      <inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="load_cell" type="fixed">
    <parent link="carrier"/><child link="payload"/>
  </joint>
  <joint name="slider" type="prismatic">
    <parent link="base"/><child link="carrier"/>
    <origin xyz="0 0 0.5"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.1" effort="100" velocity="1"/>
  </joint>
</robot>
"""


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_joint_wrench_uses_child_frame_and_retains_unreset_rows(
    tmp_path: Path, use_cuda_graph: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA MuJoCo Warp is required")
    path = tmp_path / "wrench.urdf"
    path.write_text(_URDF)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            physics_cfg=NewtonPhysicsCfg(
                enable_joint_wrenches=True,
                num_substeps=3,
                use_cuda_graph=use_cuda_graph,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
        )
    )
    try:
        chain = sim.add_articulation(
            ArticulationCfg(uid="chain", fpath=str(path), build_pk_chain=False)
        )
        cfg = SensorCfg.from_dict(
            {
                "uid": "reaction",
                "sensor_type": "JointWrench",
                "asset_uid": "chain",
                "joint_name": "load_cell",
                "offset": {
                    "pos": [0, 0, 0],
                    "quat": [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
                },
            }
        )
        sensor = sim.add_sensor(cfg)
        chain.set_mass(
            torch.tensor([[1.0], [2.0]], device=sim.device), link_names="payload"
        )
        sensor.update()
        assert not sensor.get_data()["wrench_valid"].any()
        sim.update(step=3, render_final_step=False)
        if use_cuda_graph:
            assert sim.physics.cuda_graph_status == "captured"
        reading = sensor.get_data()
        assert reading["wrench_valid"].tolist() == [True, True]
        force = torch.tensor([[0, 9.81, 0], [0, 19.62, 0]], device=sim.device)
        torque = torch.tensor([[0, 0, 0.981], [0, 0, 1.962]], device=sim.device)
        torch.testing.assert_close(reading["force"], force, atol=1e-3, rtol=0)
        torch.testing.assert_close(reading["torque"], torque, atol=1e-3, rtol=0)
        sim.reset_objects_state(env_ids=[0])
        sensor.update()
        assert sensor.get_data()["wrench_valid"].tolist() == [False, True]
        assert sensor.get_sample_metadata()["valid"].tolist() == [False, True]
        torch.testing.assert_close(
            sensor.get_data()["force"][0], torch.zeros_like(force[0])
        )
        torch.testing.assert_close(
            sensor.get_data()["force"][1], force[1], atol=1e-3, rtol=0
        )
        sim.update(render_final_step=False)
        assert sensor.get_data()["wrench_valid"].tolist() == [True, True]
        torch.testing.assert_close(
            sensor.get_data()["torque"], torque, atol=1e-3, rtol=0
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
