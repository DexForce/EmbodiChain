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
"""Selective SDK scene reset preserves captured DC motor buffers."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)

pytestmark = [pytest.mark.gpu, pytest.mark.requires_sim]


def test_scene_reset_with_graph_motor_preserves_unselected_arena(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motor.urdf"
    link = """<link name="{name}">
      <inertial><mass value="1"/>
      <inertia ixx=".02" ixy="0" ixz="0" iyy=".03" iyz="0" izz=".04"/></inertial>
      <visual><geometry><box size=".1 .1 .1"/></geometry></visual>
      <collision><geometry><box size=".1 .1 .1"/></geometry></collision>
    </link>"""
    source.write_text(
        '<robot name="motor">'
        + link.format(name="base")
        + link.format(name="tip")
        + '<joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 .3"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="5"/></joint></robot>'
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            render_cfg=RenderCfg(renderer="no-render"),
            device="cuda:0",
            num_envs=2,
            physics_cfg=NewtonPhysicsCfg(
                use_cuda_graph=True, solver_cfg={"solver_type": "dexuni"}
            ),
        )
    )
    art = motor = None
    try:
        art = sim.add_articulation(
            ArticulationCfg(
                uid="motor",
                fpath=str(source),
                init_pos=(0, 0, 1),
                init_qpos=[0.0],
                build_pk_chain=False,
                asset_physics_mode="overlay",
                joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
            )
        )
        motor = sim._register_joint_dc_motor_control(
            "motor",
            dict(
                joint_names=["hinge"],
                stiffness=[4.0],
                damping=[0.1],
                effort_limit=[5.0],
                velocity_limit=[5.0],
                default_joint_position=[0.05],
            ),
        )
        sim.prepare()
        motor.targets.fill_(0.4)
        for _ in range(4):
            sim.update()
        assert sim.physics.cuda_graph_status == "captured"
        backend = get_newton_backend(sim._world)
        q_indices = wp.to_torch(motor.q_indices).long()
        before_q = wp.to_torch(backend._runtime.current_state.joint_q)[
            q_indices
        ].clone()
        buffers = (motor.targets, motor.effort, motor.velocity)
        before = [wp.to_torch(value).clone() for value in buffers]
        pointers = [value.ptr for value in buffers]
        scene = sim._spawn_scene.builder.result
        selected_arena = scene.get_articulation("arena_1/motor")._arena_index
        for _ in range(2):
            assert scene.reset_arenas([selected_arena]) == 1
            assert [value.ptr for value in buffers] == pointers
            for value, saved in zip(buffers, before):
                torch.testing.assert_close(
                    wp.to_torch(value)[0], saved[0], rtol=0, atol=0
                )
            assert wp.to_torch(motor.targets)[1, 0].item() == pytest.approx(0.05)
            assert wp.to_torch(motor.effort)[1, 0] == 0
            assert wp.to_torch(motor.velocity)[1, 0] == 0
            q = wp.to_torch(backend._runtime.current_state.joint_q)[q_indices]
            torch.testing.assert_close(q[0], before_q[0], rtol=0, atol=0)
            assert q[1, 0].item() == 0.0
        for _ in range(3):
            sim.update()
        assert sim.physics.cuda_graph_status == "captured"
        assert torch.isfinite(wp.to_torch(motor.effort)).all()
    finally:
        art = motor = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
