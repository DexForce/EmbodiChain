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
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    MassPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.lab.sim.sensors import (
    ArticulationContactFilterCfg,
    ContactSensorCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
from embodichain_robots import URRobotCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]


@pytest.mark.parametrize(
    "solver_type", ["mujoco_warp", pytest.param(None, id="default_dexuni")]
)
@pytest.mark.parametrize("num_substeps", [1, 10])
def test_newton_contact_sensor_reports_each_arena(
    solver_type: str | None, num_substeps: int
) -> None:
    from dexsim.scene import ContactBuffer

    if solver_type is None and not hasattr(ContactBuffer, "impulse_valid"):
        pytest.skip("DexUni contacts require DexSim's current dev API.")
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
                num_substeps=num_substeps,
                use_cuda_graph=False,
                # Exercise a multi-point manifold so query-side reductions
                # cannot satisfy this regression contract accidentally.
                solver_cfg=(
                    {"solver_type": solver_type, "enable_multiccd": True}
                    if solver_type is not None
                    else None
                ),
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
        # The default DexUni path assigns this all-rigid scene to MuJoCo.
        assert sensor.contact_capabilities.impulse
        assert sensor.contact_capabilities.friction
        counts = sensor._num_contacts_per_env.cpu().tolist()
        data = sensor.get_data()
        assert data["is_valid"].sum(dim=1).cpu().tolist() == counts
        if solver_type == "mujoco_warp":
            assert counts == [4, 4]
        else:
            assert all(count > 0 for count in counts)
        assert data["impulse_valid"][data["is_valid"]].all()
        assert data["friction_valid"][data["is_valid"]].all()
        measured = data["is_valid"] & data["impulse_valid"]
        assert (data["impulse"][measured] > 1.0e-7).all()
        actor_ids = data["user_ids"][data["is_valid"]]
        assert all(
            sensor.get_actor_info(actor_id).path.endswith(("/ground", "/cube"))
            for actor_id in set(actor_ids.flatten().cpu().tolist())
        )
        history = sensor.create_history(
            "support",
            sensor.get_actor_ids("cube"),
            counterpart_ids=sensor.get_actor_ids("ground"),
            force_threshold=1.0,
        )
        # Keep the physical load invariant while varying both the solver's
        # internal substeps and the actual World.update interval.
        for dt in (0.01, 0.02):
            sensor.begin_control_step()
            sim.update(dt, step=3, after_substep=sensor.update_physics_step)
            torch.testing.assert_close(
                history.force[..., 2],
                torch.full_like(history.contact, 9.81, dtype=torch.float32),
                rtol=0.01,
                atol=0.05,
            )
            assert history.found.all()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()


@pytest.fixture
def dexuni_contact_scene(tmp_path):
    """A native contact report with articulation dynamics invalidated by reset."""
    urdf = tmp_path / "slider.urdf"
    urdf.write_text("""<robot name="slider">
      <link name="base"/><link name="moving">
        <inertial><mass value="0.1"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
        <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
      </link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="moving"/>
        <origin xyz="0 0 0.5"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="0.1" effort="10" velocity="1"/>
      </joint>
    </robot>""")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=0.01,
                num_substeps=10,
                use_cuda_graph=False,
            ),
        )
    )
    try:
        sim.add_articulation(
            ArticulationCfg(uid="slider", fpath=str(urdf), build_pk_chain=False)
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[1.0, 1.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
                init_pos=(0.0, 0.0, 0.1),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.2, 0.2, 0.2]),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                init_pos=(0.0, 0.0, 0.4),
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["ground", "cube"],
                max_contacts_per_env=32,
            )
        )
        sim.update(step=150)
        sensor.update()
        assert sensor.get_data()["is_valid"].any()
        yield sim, sensor
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()


def test_dexuni_physics_reset_keeps_initial_sensor_observations_readable(
    dexuni_contact_scene,
):
    sim, sensor = dexuni_contact_scene
    env = SimpleNamespace(
        sim=sim,
        sensors={"contacts": sensor},
        device=sim.device,
        num_envs=sim.num_envs,
        _camera_group_ids=[],
        _profiler=SimpleNamespace(section=lambda *_: nullcontext()),
    )
    assert not sensor.requires_substep_update
    assert BaseEnv._get_sensor_obs(env)["contacts"]["is_valid"].any()
    # Sensor-only reset must retain the valid, same-step native report.
    sensor.reset([0])
    assert BaseEnv._get_sensor_obs(env)["contacts"]["is_valid"].any()
    # Reproduce Gym's physical reset ordering, including articulation clear.
    sim.reset_objects_state(env_ids=[0], excluded_sensor_uids=[sensor.uid])
    sensor.reset([0])
    for _ in range(2):
        data = BaseEnv._get_sensor_obs(env)["contacts"]
        assert not data["is_valid"].any()
        assert not data["impulse_valid"].any()
        assert not data["friction_valid"].any()
        assert sensor.total_current_contacts == 0
    sim.update(step=150)
    assert BaseEnv._get_sensor_obs(env)["contacts"]["is_valid"].any()


@pytest.mark.parametrize("history_sample", [False, True])
def test_dexuni_external_graph_consumes_eager_native_samples(
    dexuni_contact_scene, history_sample
):
    sim, sensor = dexuni_contact_scene
    if history_sample:
        sensor.create_history(
            "support", sensor.get_actor_ids("cube"), force_threshold=1.0
        )
        sim.update(step=3, after_substep=sensor.update_physics_step)
    stream = torch.cuda.Stream(device=sim.device)
    stream.wait_stream(torch.cuda.current_stream(sim.device))
    with torch.cuda.stream(stream):
        sensor.update()
    stream.synchronize()
    # This is the review's native same-step warmup/capture sequence. The
    # boundary error must occur before DexSim can wait on an uncaptured event.
    with pytest.raises(RuntimeError, match="outside external CUDA capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            if history_sample:
                sensor.update_physics_step(0.01)
            else:
                sensor.update()

    # Mixed DexUni reports may provide geometry-only rows with zero impulse.
    # Consume their validity so replay freshness does not assume force support.
    output = torch.empty_like(sensor.get_data()["is_valid"])
    stream.wait_stream(torch.cuda.current_stream(sim.device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        data = sensor.get_data()
        output.copy_(data["is_valid"])

    def replay_latest():
        stream.wait_stream(torch.cuda.current_stream(sim.device))
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        data = sensor.get_data()
        torch.testing.assert_close(output, data["is_valid"])

    replay_latest()
    assert output.any()
    sim.reset_objects_state(env_ids=[0], excluded_sensor_uids=[sensor.uid])
    sensor.reset([0])
    sensor.update()
    replay_latest()
    assert output.count_nonzero() == 0
    sim.update(
        step=150, after_substep=sensor.update_physics_step if history_sample else None
    )
    sensor.update()
    replay_latest()
    assert output.any()


@pytest.mark.parametrize("solver_path", ["pure_mujoco", "coupled"])
@pytest.mark.usefixtures("requires_dexuni_contact_query")
def test_dexuni_pure_mujoco_contact_sensor_reports_impulse(solver_path: str) -> None:
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
                    **({"coupling": "two_way"} if solver_path == "coupled" else {}),
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
        if solver_path == "coupled":
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid="box",
                    shape=CubeCfg(size=[0.1, 0.1, 0.1]),
                    attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                    init_pos=(0.3, 0.0, 0.2),
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

        sim.update(step=120 if solver_path == "coupled" else 10)
        sensor.update()

        assert sensor.contact_capabilities.geometry
        data = sensor.get_data()
        valid = data["is_valid"] & data["impulse_valid"] & data["friction_valid"]
        if solver_path == "pure_mujoco":
            assert sensor.contact_capabilities.impulse
            assert sensor.contact_capabilities.friction
            assert torch.equal(valid, data["is_valid"])
        else:
            assert not sensor.contact_capabilities.impulse
            assert (data["is_valid"] & ~data["impulse_valid"]).any()
            raw = sensor._query.fetch_solver_report()
            for field in ("impulse_valid", "friction_valid"):
                assert data[field].sum() == getattr(raw, field)[: raw.count].sum()
        assert valid.any()
        assert torch.isfinite(data["position"][valid]).all()
        assert torch.isfinite(data["normal"][valid]).all()
        assert torch.isfinite(data["friction"][valid]).all()
        assert torch.isfinite(data["impulse"][valid]).all()
        assert torch.isfinite(data["distance"][valid]).all()
        normal = data["normal"][valid]
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        assert torch.allclose(normal_norm, torch.ones_like(normal_norm), atol=1.0e-4)
        tangent_residual = (data["friction"][valid] * normal).sum(dim=-1).abs()
        assert tangent_residual.max() < 1.0e-4
        assert (data["user_ids"][valid] >= 0).all()
        assert (data["impulse"][valid] > 1.0e-7).any()

        # Cross-check the adapter against DexUni's native force report.  The
        # sensor's normal impulse is the normal component of that force over
        # the solver substep, so this catches axis/sign/unit regressions that
        # finite-value assertions cannot detect.
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend
        from newton import Contacts

        backend = get_newton_backend(sim._world)
        assert backend.solver.features.backend == solver_path
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


@pytest.mark.usefixtures("requires_dexuni_contact_query")
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
                # Explicit ownership keeps this a VBD geometry-only contract
                # even though all-rigid scenes now default to pure MuJoCo.
                solver_cfg={"solver_type": "dexuni", "mujoco_articulations": ()},
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
        assert torch.isfinite(data["distance"][valid]).all()
        assert not data["impulse_valid"][valid].any()
        assert not data["friction_valid"][valid].any()
        assert torch.count_nonzero(data["impulse"][valid]) == 0
        assert torch.count_nonzero(data["friction"][valid]) == 0
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
        from dexsim.engine.newton_physics import teardown_newton_physics

        teardown_newton_physics()
