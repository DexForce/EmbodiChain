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

"""Cable contact feedback through the public EmbodiChain CUDA interfaces."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CableObjectCfg,
    CablePhysicsCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.sim.sensors import ArticulationContactFilterCfg, ContactSensorCfg

pytestmark = [
    pytest.mark.subprocess_sim,
    pytest.mark.gpu,
    pytest.mark.usefixtures("requires_dexuni_two_way", "requires_dexuni_contact_query"),
]

# The finger moves along +X into the free middle segment of a clamped cable.
FINGER_START_X = -0.06
FINGER_HALF_X = 0.01
CABLE_RADIUS = 0.015
CABLE_HEIGHT = 0.3  # Clear the default ground and isolate cable contacts.
TARGET_TRAVEL = 0.09  # Move the finger completely through the cable in one-way mode.
CONTACT_TRAVEL = -FINGER_START_X - FINGER_HALF_X - CABLE_RADIUS
POSITION_TOLERANCE = CABLE_RADIUS / 2


def _write_finger(path: Path) -> None:
    """Author one dynamic slider, with no other collision geometry."""
    path.write_text(
        f"""<robot name="finger">
  <link name="base"/>
  <link name="moving">
    <inertial><mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <collision><geometry><box size="{2 * FINGER_HALF_X} 0.025 0.04"/></geometry></collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="moving"/>
    <origin xyz="{FINGER_START_X} 0 {CABLE_HEIGHT}"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.12" effort="10" velocity="1"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )


def _run_contact_case(path: Path, coupling: str) -> dict[str, float | int]:
    """Push a deformable cable with one-way or two-way articulation coupling."""
    envelope = NewtonCollisionPropertiesCfg(margin=0.0, gap=0.0)
    material = NewtonRigidBodyMaterialCfg(ke=2.0e4, kd=100.0, dynamic_friction=0.5)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            robot_ik_gizmo=None,
            enable_entity_gizmo=False,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                gravity=(0.0, 0.0, 0.0),
                physics_dt=1.0 / 60.0,
                num_substeps=10,
                use_cuda_graph=False,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "joint_mode": "dynamic",
                    "coupling": coupling,
                    "mujoco_articulation_paths": ("arena_0/finger",),
                    "vbd_options": {"iterations": 20},
                    "collision_options": {"broad_phase": "nxn"},
                },
            ),
        )
    )
    articulation = cable = sensor = None
    try:
        articulation = sim.add_articulation(
            ArticulationCfg(
                uid="finger",
                fpath=str(path),
                init_qpos=[0.0],
                asset_physics_mode="overlay",
                build_pk_chain=False,
                attrs=RigidBodyPhysicsCfg(
                    collision_props=envelope, material_props=material
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
        cable = sim.add_cable_object(
            CableObjectCfg(
                uid="wire",
                points=[(0.0, y, CABLE_HEIGHT) for y in (-0.09, -0.03, 0.03, 0.09)],
                radius=CABLE_RADIUS,
                fixed_segments=(0, 2),
                attrs=CablePhysicsCfg(
                    density=1000.0,
                    stretch_stiffness=1.0e5,
                    shear_stiffness=1.0e5,
                    bend_stiffness=10.0,
                    twist_stiffness=10.0,
                    collision_props=envelope,
                    material_props=material,
                ),
            )
        )
        sim.prepare()
        assert cable.data is not None
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="finger_contact",
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid="finger", link_name_list=["moving"]
                    )
                ],
                filter_need_both_actor=False,
                max_contacts_per_env=32,
            )
        )
        articulation.set_qpos(
            torch.tensor([[TARGET_TRAVEL]], device=sim.device), target=True
        )
        contact_rows = 0
        peak_cable_deflection = 0.0
        # Two seconds allow the damped slider to settle at the contact surface.
        for _ in range(120):
            sim.update(step=1, render_final_step=False)
            state = cable.data.segment_state_w
            assert torch.isfinite(state).all()
            peak_cable_deflection = max(peak_cable_deflection, float(state[0, 1, 0]))
            sensor.update()
            data = sensor.get_data()
            valid = data["is_valid"]
            if valid.any():
                # Cable shapes currently have no ContactActorInfo entry. With
                # match-any on the finger, the native query still emits their
                # rows. This isolated scene has no other possible counterpart.
                assert sensor.contact_capabilities.geometry
                assert not data["impulse_valid"][valid].any()
                assert not data["friction_valid"][valid].any()
                assert torch.count_nonzero(data["impulse"][valid]) == 0
                assert torch.count_nonzero(data["friction"][valid]) == 0
                assert torch.isfinite(data["position"][valid]).all()
                assert torch.isfinite(data["normal"][valid]).all()
                contact_rows += int(valid.sum())
        travel = float(articulation.get_qpos()[0, 0])
        return {
            "travel": travel,
            "peak_cable_deflection": peak_cable_deflection,
            "geometry_contact_rows": contact_rows,
        }
    finally:
        articulation = cable = sensor = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _assert_cable_feedback(path: Path) -> None:
    """Compare identical scenes; only the articulation reaction policy differs."""
    _write_finger(path)
    one_way = _run_contact_case(path, "one_way")
    two_way = _run_contact_case(path, "two_way")
    print(json.dumps({"one_way": one_way, "two_way": two_way}), flush=True)
    assert abs(one_way["travel"] - TARGET_TRAVEL) < POSITION_TOLERANCE, one_way
    assert abs(two_way["travel"] - CONTACT_TRAVEL) < POSITION_TOLERANCE, two_way
    assert one_way["travel"] - two_way["travel"] > CABLE_RADIUS, (one_way, two_way)
    assert two_way["geometry_contact_rows"] > 0, two_way
    # The contacted segment is free, so the reaction must load the actual cable.
    assert two_way["peak_cable_deflection"] > 1.0e-6, two_way


class TestCableTwoWayCUDA:
    """Physical feedback and contact validity in an isolated native process."""

    def test_cable_contact_blocks_dynamic_finger_with_two_way_coupling(
        self, tmp_path: Path
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for cable two-way contact feedback.")
        script = (
            "import runpy, sys; from pathlib import Path; "
            "module = runpy.run_path(sys.argv[1]); "
            "module['_assert_cable_feedback'](Path(sys.argv[2]))"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve()),
                str(tmp_path / "finger.urdf"),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
