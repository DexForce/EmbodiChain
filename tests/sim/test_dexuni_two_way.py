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

"""Physical acceptance of DexUni contact feedback through EmbodiChain."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

pytestmark = [
    pytest.mark.requires_sim,
    pytest.mark.gpu,
    pytest.mark.usefixtures("requires_dexuni_two_way"),
]

FINGER_START = 0.08
FINGER_HALF = 0.01
BOX_HALF = 0.025
TARGET = FINGER_START - FINGER_HALF
CONTACT_TRAVEL = FINGER_START - FINGER_HALF - BOX_HALF


def _gripper_urdf(path: Path) -> None:
    """Write two light, effort-limited fingers without downloadable assets."""
    fingers = []
    for name, sign in (("left", -1.0), ("right", 1.0)):
        fingers.append(f"""
  <link name="{name}">
    <inertial><mass value="0.012"/><inertia ixx="0.000003" ixy="0" ixz="0"
        iyy="0.0000013" iyz="0" izz="0.000002"/></inertial>
    <collision><geometry><box size="0.02 0.04 0.03"/></geometry></collision>
  </link>
  <joint name="{name}" type="prismatic">
    <parent link="base"/><child link="{name}"/>
    <origin xyz="{sign * FINGER_START} 0 0.03"/><axis xyz="{-sign} 0 0"/>
    <limit lower="0" upper="0.1" effort="10" velocity="1"/>
  </joint>""")
    path.write_text(
        '<robot name="gripper"><link name="base"/>' + "".join(fingers) + "</robot>",
        encoding="utf-8",
    )


def _grip(path: Path, coupling: str, use_cuda_graph: bool) -> torch.Tensor:
    material = RigidBodyPhysicsCfg(
        collision_props=NewtonCollisionPropertiesCfg(margin=0.0, gap=0.0),
        material_props=NewtonRigidBodyMaterialCfg(
            ke=2.0e4, kd=100.0, dynamic_friction=1.0
        ),
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=1.0 / 60.0,
                num_substeps=10,
                use_cuda_graph=use_cuda_graph,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "joint_mode": "dynamic",
                    "coupling": coupling,
                    "vbd_options": {"iterations": 10},
                    "collision_options": {"broad_phase": "nxn"},
                },
            ),
        )
    )
    try:
        articulation = sim.add_articulation(
            ArticulationCfg(
                uid="gripper",
                fpath=str(path),
                asset_physics_mode="overlay",
                build_pk_chain=False,
                attrs=material,
                root_props=ArticulationRootPropertiesCfg(
                    newton_gravity_compensation=1.0
                ),
                joint_drive_props=NewtonJointDrivePropertiesCfg(
                    target_mode="position_velocity",
                    stiffness=1000.0,
                    damping=20.0,
                    passive_damping=50.0,
                    armature=0.3,
                    max_effort=10.0,
                ),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="floor",
                shape=CubeCfg(size=(1.0, 1.0, 0.1)),
                body_type="static",
                init_pos=(0.0, 0.0, -0.05),
                attrs=material,
            )
        )
        box_material = material.copy()
        box_material.mass_props = MassPropertiesCfg(mass=0.0625)
        box = sim.add_rigid_object(
            RigidObjectCfg(
                uid="box",
                shape=CubeCfg(size=(2 * BOX_HALF,) * 3),
                init_pos=(0.0, 0.0, BOX_HALF),
                attrs=box_material,
            )
        )
        sim.prepare()
        articulation.set_qpos(
            torch.full((1, 2), TARGET, device=sim.device), target=True
        )
        sim.update(step=120)
        qpos = articulation.get_qpos().detach().cpu()
        assert torch.isfinite(qpos).all()
        assert torch.isfinite(box.get_local_pose()).all()
        if use_cuda_graph:
            assert sim.physics.cuda_graph_status == "captured"

        sim.reset_objects_state()
        torch.testing.assert_close(
            articulation.get_qpos(), torch.zeros((1, 2), device=sim.device)
        )
        return qpos
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_dexuni_two_way_contact_blocks_dynamic_finger_targets(
    tmp_path: Path, use_cuda_graph: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DexUni two-way contract.")
    path = tmp_path / "gripper.urdf"
    _gripper_urdf(path)
    one_way = _grip(path, "one_way", use_cuda_graph)
    two_way = _grip(path, "two_way", use_cuda_graph)

    # Without feedback, the same effort-limited fingers reach the commanded
    # overlapping pose. Feedback stops them at the box's physical surface.
    assert torch.all(one_way > TARGET - 0.005), one_way
    assert torch.all(two_way < CONTACT_TRAVEL + 0.01), two_way
    assert torch.all(two_way > CONTACT_TRAVEL - 0.01), two_way
