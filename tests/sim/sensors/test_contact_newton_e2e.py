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
"""Newton end-to-end contract for the backend-neutral contact sensor."""

from __future__ import annotations

import math

import pytest
import torch
import warp as wp

from embodichain.lab.sim.cfg import (
    MassPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import (
    ArticulationContactFilterCfg,
    ContactSensorCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import URRobotCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]


def test_newton_contact_sensor_reports_each_arena() -> None:
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for the Newton contact E2E contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=1,
                use_cuda_graph=False,
                # Exercise a multi-point manifold so query-side reductions
                # cannot satisfy this regression contract accidentally.
                solver_cfg={
                    "solver_type": "mujoco_warp",
                    "enable_multiccd": True,
                },
            ),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[1.0, 1.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.2, 0.2, 0.2]),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 0.4),
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["ground", "cube"],
                max_contacts_per_env=16,
            )
        )

        sim.update(step=120)
        sensor.update()

        assert sensor.contact_capabilities.geometry
        assert sensor.contact_capabilities.impulse
        counts = sensor._num_contacts_per_env.cpu().tolist()
        assert counts == [4, 4]
        data = sensor.get_data()
        assert data["is_valid"].sum(dim=1).cpu().tolist() == counts
        assert (data["impulse"][data["is_valid"]] > 1.0e-7).all()
        actor_ids = data["user_ids"][data["is_valid"]]
        assert all(
            sensor.get_actor_info(actor_id).path.endswith(("/ground", "/cube"))
            for actor_id in set(actor_ids.flatten().cpu().tolist())
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()


def test_dexuni_pure_mujoco_contact_sensor_reports_impulse() -> None:
    """Pure DexUni/MuJoCo exposes geometry, normal and friction impulses."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for the DexUni contact contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=1,
                use_cuda_graph=False,
                solver_cfg={
                    "solver_type": "dexuni",
                    "joint_mode": "dynamic",
                    "contact_mode": "auto",
                },
            ),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[3.0, 3.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
                init_pos=(0.0, 0.0, -0.05),
            )
        )
        sim.add_robot(
            URRobotCfg.from_dict(
                {
                    "robot_type": "ur10",
                    "uid": "robot",
                    "init_pos": (0.0, 0.0, 0.0),
                }
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["ground"],
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(articulation_uid="robot")
                ],
                filter_need_both_actor=False,
                max_contacts_per_env=64,
            )
        )

        sim.update(step=10)
        sensor.update()

        assert sensor.contact_capabilities.geometry
        assert sensor.contact_capabilities.impulse
        assert sensor.contact_capabilities.friction
        data = sensor.get_data()
        valid = data["is_valid"]
        assert valid.any()
        assert torch.isfinite(data["position"][valid]).all()
        assert torch.isfinite(data["normal"][valid]).all()
        assert (data["impulse"][valid] > 1.0e-7).any()

        # Cross-check the adapter against DexUni's native force report.  The
        # sensor's normal impulse is the normal component of that force over
        # the solver substep, so this catches axis/sign/unit regressions that
        # finite-value assertions cannot detect.
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend
        from newton import Contacts

        backend = get_newton_backend(sim._world)
        assert backend.solver.features.backend == "pure_mujoco"
        report = Contacts(
            backend.mujoco_solver.get_max_contact_count(),
            0,
            device=backend.model.device,
            requested_attributes={"force"},
        )
        backend.mujoco_solver.update_contacts(report, backend.state_0)
        raw_count = int(report.rigid_contact_count.numpy()[0])
        raw_forces = report.force.numpy()[:raw_count, :3]
        raw_normals = report.rigid_contact_normal.numpy()[:raw_count]
        substep_dt = float(backend.cfg.dt) / int(backend.cfg.num_substeps)
        raw_normal_impulse = abs((raw_forces * raw_normals).sum(axis=1)) * substep_dt
        raw_normal_impulse = raw_normal_impulse[raw_normal_impulse > 1.0e-7]
        sensor_total = float(data["impulse"][valid].sum().item())
        raw_total = float(raw_normal_impulse.sum())
        assert raw_total > 0.0
        assert math.isclose(sensor_total, raw_total, rel_tol=2.0e-3, abs_tol=2.0e-4)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()


def test_dexuni_vbd_contact_sensor_exposes_geometry_only() -> None:
    """VBD-owned rigid contacts keep geometry but disable force fields."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for the DexUni contact contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                num_substeps=1,
                use_cuda_graph=False,
                solver_cfg={"solver_type": "dexuni"},
            ),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[1.0, 1.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.2, 0.2, 0.2]),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 0.4),
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["cube"],
                filter_need_both_actor=False,
                max_contacts_per_env=16,
            )
        )

        sim.update(step=120)
        sensor.update()

        assert sensor.contact_capabilities.geometry
        assert not sensor.contact_capabilities.impulse
        assert not sensor.contact_capabilities.friction
        data = sensor.get_data()
        valid = data["is_valid"]
        assert valid.any()
        assert torch.isfinite(data["position"][valid]).all()
        assert torch.isfinite(data["normal"][valid]).all()
        assert torch.count_nonzero(data["impulse"][valid]) == 0
        assert torch.count_nonzero(data["friction"][valid]) == 0
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()
