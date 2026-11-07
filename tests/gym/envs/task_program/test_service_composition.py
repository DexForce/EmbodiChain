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

"""Closed ordinary service configuration and real factory assembly boundaries."""

from __future__ import annotations
from dataclasses import replace
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    EntityState,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.task_program.integrations.catalog import (
    IntegrationFingerprintMismatch,
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_runtime_services,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _merge_runtime_services,
)
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramFactory,
)
from .test_simulation_environment import (
    _Robot,
    _RigidObject,
    _Simulation,
    _profile_binding,
    _motion_generator,
)


def _declaration():
    return {
        "kind": "simulation_geometry",
        "free_motion_query": "discrete_mesh",
        "collision_roster": {
            "rigid_objects": [
                ["floor", "table"],
                ["payload", "cylinder"],
                ["rest", "pad"],
            ]
        },
        "place_scene_bindings": [
            {
                "invocation_id": "place:0",
                "floor_entity_id": "floor",
                "support_owner_id": "rest",
            }
        ],
        "observed_curobo": {
            "graph_planner_algorithm": "prm",
            "enable_timeopt": False,
            "fit_graph_seed_to_bspline": False,
        },
        "capture_root_pose": True,
        "capture_native_revision": True,
        "hold_qpos_source": "drive_target",
    }


def test_closed_service_composition_preserves_none_and_seals_active_configuration():
    base = SimulationTaskProgramRegistration(
        SimulationSceneBinding(registry_id="scene"), _profile_binding()
    )
    assert "planning_services" not in base.fingerprint_payload
    assert (
        replace(base, planning_services=None).fingerprint_payload
        == base.fingerprint_payload
    )
    raw = _declaration()
    merged = _merge_runtime_services({"planning_services": raw}, {})
    cfg = _decode_runtime_services(merged).planning_services
    active = replace(base, planning_services=cfg)
    assert active.fingerprint != base.fingerprint
    payload = active.fingerprint_payload["planning_services"]
    # This is the complete declared policy, not SDK or callable reconstruction.
    observed = payload["fields"]["observed_curobo"]["fields"]
    assert {
        name: observed[name]
        for name in (
            "max_voxel_count",
            "graph_planner_algorithm",
            "enable_timeopt",
            "fit_graph_seed_to_bspline",
            "use_cuda_graph",
            "warmup_iterations",
            "preserve_plan_samples",
        )
    } == {
        "max_voxel_count": 4_000_000,
        "graph_planner_algorithm": "prm",
        "enable_timeopt": False,
        "fit_graph_seed_to_bspline": False,
        "use_cuda_graph": False,
        "warmup_iterations": 0,
        "preserve_plan_samples": True,
    }
    cfg.observed_curobo.max_voxel_count = 1
    active.assert_unchanged()
    active.planning_services.observed_curobo.max_voxel_count = 2
    with pytest.raises(IntegrationFingerprintMismatch):
        active.assert_unchanged()
    with pytest.raises(ValueError, match="exactly one"):
        _merge_runtime_services({"planning_services": raw}, {"planning_services": raw})
    with pytest.raises(ValueError, match="native_revision_reader"):
        _decode_runtime_services(
            {"planning_services": {**raw, "native_revision_reader": "callable"}}
        )
    invalid = _declaration()
    invalid["observed_curobo"]["warmup_iterations"] = 1
    with pytest.raises(ValueError, match="zero warmup"):
        _decode_runtime_services({"planning_services": invalid})


def test_real_factory_forwards_owned_ports_and_rejects_world_drift(monkeypatch):
    import embodichain.lab.sim.motion.mesh_collision as mesh
    import embodichain.lab.sim.motion.planners.curobo.observed as observed
    import embodichain.lab.sim.scene.collision_sources as sources
    import embodichain.lab.sim.atomic_actions.curobo_queries as prefixes
    import embodichain.lab.sim.sensors.contact.sampling as sampling

    robot = _Robot()
    robot.qpos = robot.qpos[:1].clone()
    robot.target_qpos = robot.target_qpos[:1].clone()
    bodies = {uid: _RigidObject() for uid in ("table", "cylinder", "pad")}
    for body in bodies.values():
        body.pose = body.pose[:1].clone()
    sim = _Simulation(robot, bodies)
    sim.num_envs, sim.physics_backend, sim.instance_id, sim.simulation_time = (
        1,
        "default",
        17,
        0.75,
    )
    calls = {}

    class Free:
        def __init__(self, selected_robot, simulation):
            assert selected_robot is robot and simulation is sim
            self.robot = selected_robot

        def inspect(self, positions, *, control_part):
            raise AssertionError("This assembly test must not run collision queries.")

    class Shapes:
        def __init__(self, simulation, *, rigid_objects, articulation_links):
            assert simulation is sim and articulation_links == {}
            calls["rigid_roster"] = tuple(rigid_objects.items())

    class Queries:
        def __init__(self, selected_robot, cfg, shapes, *, native_revision_reader):
            self.robot = selected_robot
            calls["cfg"] = cfg
            calls["reader"] = native_revision_reader

    class Prefix:
        def __init__(self, queries, world):
            self.robot = queries.robot
            calls["world"] = world

        def plan(self, **kwargs):
            raise AssertionError("This assembly test must not solve a motion.")

    monkeypatch.setattr(mesh, "SimulationFreeMotionQuery", Free)
    monkeypatch.setattr(sources, "SimulationCollisionShapeReader", Shapes)
    monkeypatch.setattr(observed, "ObservedCuroboQueryFactory", Queries)
    monkeypatch.setattr(prefixes, "CuroboJointPrefixPlanner", Prefix)
    revision = [7]
    monkeypatch.setattr(
        sampling, "native_contact_revision", lambda simulation: revision[0]
    )
    cfg = _decode_runtime_services(
        {"planning_services": _declaration()}
    ).planning_services
    registration = SimulationTaskProgramRegistration(
        SimulationSceneBinding(registry_id="scene"),
        _profile_binding(),
        planning_services=cfg,
    )
    factory = SimulationTaskProgramFactory(
        sim,
        robot,
        registration,
        step_dt=0.04,
        motion_generator_factory=lambda: _motion_generator(robot),
    )
    engine = factory.create_atomic_action_engine(factory.create_robot_skill_profile())
    assert engine.planning_services.free_motion_query.robot is robot
    assert engine.planning_services.joint_prefix_planner.robot is robot
    assert engine.planning_services.place_scene_provider is not None
    assert calls["rigid_roster"] == (
        ("floor", "table"),
        ("payload", "cylinder"),
        ("rest", "pad"),
    )
    actual = calls["cfg"]
    assert (
        actual.world.max_voxel_count,
        actual.use_cuda_graph,
        actual.warmup_iterations,
        actual.preserve_plan_samples,
        actual.interpolation_dt,
        actual.graph_planner_algorithm,
        actual.enable_timeopt,
        actual.fit_graph_seed_to_bspline,
    ) == (4_000_000, False, 0, True, 0.04, "prm", False, False)
    stamp = factory.world_source_stamp()
    assert (
        stamp.source_kind == "default.contacts"
        and stamp.revision == 7
        and stamp.simulation_time == 0.75
    )
    assert calls["reader"]() == 7
    poses = {
        logical: EntityState(bodies[uid].get_local_pose(to_matrix=True))
        for logical, uid in calls["rigid_roster"]
    }
    context = PlanningContext(
        RobotObservation(0.5, robot.qpos, torch.zeros_like(robot.qpos)),
        TaskState(1, "cpu"),
        SceneSnapshot(0.5, 0, poses),
        torch.tensor([0]),
        control_dt=0.04,
        world_source_stamp=stamp,
    )
    world = calls["world"].capture(context)
    assert (
        tuple(world.poses) == ("floor", "payload", "rest")
        and world.native_revision == 7
    )
    revision[0] = 8
    with pytest.raises(ValueError, match="no longer matches"):
        calls["world"].capture(context)
    factory.close_runtime_resources()
    with pytest.raises(RuntimeError, match="closed"):
        calls["reader"]()
    with pytest.raises(RuntimeError, match="closed"):
        engine.planning_services.free_motion_query.inspect(
            robot.qpos, control_part="arm"
        )
