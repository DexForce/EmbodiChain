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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from dexsim.scene import Scene, SpawnedCable

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CableAttachmentCfg,
    CableObjectCfg,
    NewtonPhysicsCfg,
)
from embodichain.lab.sim.objects import CableObject, CableObjectData
from embodichain.lab.sim.physics import NewtonPhysicsBackend, DefaultPhysicsBackend
from embodichain.lab.sim.spawn.scene import SpawnScene

pytestmark = pytest.mark.no_sim


def _cfg() -> CableObjectCfg:
    return CableObjectCfg(
        uid="wire", points=[[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2]], radius=0.01
    )


class _CableBatch:
    def __init__(self) -> None:
        self.is_valid = True
        # Quaternion-then-translation: deliberately non-identity orientation
        # and different world origins distinguish both boundary mistakes.
        self.pose = torch.tensor([0.1, 0.2, 0.3, 0.9, 10, 20, 30]).repeat(2, 2, 1)
        self.pose[..., :4] /= torch.linalg.vector_norm(
            self.pose[..., :4], dim=-1, keepdim=True
        )
        self.pose[1, :, 4:] += 100
        self.velocity = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6)
        self.default_pose = self.pose.clone()
        self.default_velocity = self.velocity.clone()
        self.device = "cpu"
        self.selection = [0, 1]
        self.writes = 0
        self.step = 7
        self.reads = 0
        self.fail = False

    def fetch_state(self, pose, velocity) -> int:
        if not self.is_valid:
            raise RuntimeError("invalid Scene")
        if self.fail:
            self.fail = False
            raise RuntimeError("batch read failed")
        self.reads += 1
        pose.copy_(self.pose)
        velocity.copy_(self.velocity)
        return self.step

    def fetch_default_state(self, pose, velocity) -> int:
        if self.fail:
            self.fail = False
            raise RuntimeError("batch read failed")
        pose.copy_(self.default_pose)
        velocity.copy_(self.default_velocity)
        return 2

    def select(self, ids):
        self.selection = ids
        return self

    def apply_state(self, *, pose, velocity):
        self.writes += 1
        self.pose[self.selection] = pose
        self.velocity[self.selection] = velocity


def _handles() -> list[Mock]:
    handles = [Mock(spec=SpawnedCable), Mock(spec=SpawnedCable)]
    for handle in handles:
        handle.segment_count = 2
        handle.is_valid = True
    return handles


def _bound_cable() -> tuple[CableObject, _CableBatch]:
    batch = _CableBatch()
    scene = Mock(spec=Scene)
    scene.backend = "newton"
    scene.world = object()
    scene.create_cable_batch.return_value = batch
    cable = CableObject(_cfg())
    cable._initialize_spawn_declaration(2)
    cable.attach_spawn_handles(_handles())
    cable.bind_spawn(scene)
    return cable, batch


def test_state_layout_world_coordinates_and_coherent_snapshot() -> None:
    cable, batch = _bound_cable()
    data = cable.data
    before = batch.reads
    state = data.segment_state_w
    assert batch.reads == before + 1
    assert data.state_step == 7
    assert state.shape == (2, 2, 13)
    torch.testing.assert_close(state[..., :3], batch.pose[..., 4:])
    torch.testing.assert_close(state[..., 3:7], batch.pose[..., :4])
    torch.testing.assert_close(state[..., 7:], batch.velocity)


def test_returned_and_default_states_are_independent_snapshots() -> None:
    cable, batch = _bound_cable()
    default = cable.data.default_segment_state_w
    cable.data.segment_pose_w.zero_()
    cable.data.segment_vel_w.zero_()
    cable.data.default_segment_state_w.zero_()
    batch.pose[..., 4] += 5
    assert torch.count_nonzero(cable.data.segment_pose_w)
    torch.testing.assert_close(cable.data.default_segment_state_w, default)
    assert not torch.equal(cable.data.segment_state_w, default)


def test_failed_binding_can_be_retried_without_partial_bound_state() -> None:
    batch = _CableBatch()
    batch.fail = True
    scene = Mock(spec=Scene, backend="newton")
    scene.world = object()
    scene.create_cable_batch.return_value = batch
    cable = CableObject(_cfg())
    cable._initialize_spawn_declaration(2)
    cable.attach_spawn_handles(_handles())
    with pytest.raises(RuntimeError, match="batch read failed"):
        cable.bind_spawn(scene)
    assert cable.is_declared and cable.data is None
    cable.bind_spawn(scene)
    assert cable.is_spawn_bound
    with pytest.raises(RuntimeError, match="already Spawn-bound"):
        cable.bind_spawn(scene)


def test_data_rejects_mixed_topology() -> None:
    handles = _handles()
    handles[1].segment_count = 3
    with pytest.raises(ValueError, match="equal positive"):
        CableObjectData(handles, Mock(spec=Scene), torch.device("cpu"))


def test_selected_load_rows_follow_environment_order() -> None:
    cable, _ = _bound_cable()
    force = np.array([[1, 2, 3], [4, 5, 6]])
    cable.set_force(force, segment_index=1, env_ids=[1, 0])
    np.testing.assert_allclose(cable._entities[1].set_force.call_args.args[0], force[0])
    np.testing.assert_allclose(cable._entities[0].set_force.call_args.args[0], force[1])
    cable.set_torque([1, 0, 0], env_ids=[1])
    cable._entities[0].set_torque.assert_not_called()
    cable.clear_wrenches(env_ids=[0])
    cable._entities[1].clear_wrenches.assert_not_called()


@pytest.mark.parametrize(
    "env_ids, force, segment",
    [
        ([0, 0], [1, 2, 3], -1),
        ([2], [1, 2, 3], -1),
        ([True], [1, 2, 3], -1),
        ([0], [1, float("nan"), 3], -1),
        ([0], [[1, 2, 3], [4, 5, 6]], -1),
        ([0], [1, 2, 3], 2),
    ],
)
def test_invalid_load_does_not_mutate_any_environment(env_ids, force, segment) -> None:
    cable, _ = _bound_cable()
    with pytest.raises((ValueError, TypeError, IndexError)):
        cable.set_force(force, env_ids=env_ids, segment_index=segment)
    for handle in cable._entities:
        handle.set_force.assert_not_called()


def test_released_scene_blocks_reads_and_loads() -> None:
    cable, batch = _bound_cable()
    batch.is_valid = False
    with pytest.raises(RuntimeError, match="invalid Scene"):
        _ = cable.data.segment_state_w
    with pytest.raises(RuntimeError, match="released"):
        cable.set_force([1, 0, 0])


def test_invalid_handle_blocks_load_clearing_before_any_writes() -> None:
    cable, _ = _bound_cable()
    cable._entities[1].is_valid = False
    with pytest.raises(RuntimeError, match="invalid cable handle"):
        cable.clear_wrenches()
    for handle in cable._entities:
        handle.clear_wrenches.assert_not_called()


@pytest.mark.parametrize("operation", ["get_local_pose", "destroy"])
def test_missing_engine_operations_are_explicit(operation) -> None:
    cable, _ = _bound_cable()
    with pytest.raises(NotImplementedError):
        getattr(cable, operation)()


def _manager(solver="dexuni", device="cuda", requires_grad=False) -> SimulationManager:
    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(sim)
    sim.physics._configured_solver_type = solver
    sim.device = torch.device(device)
    sim.sim_config = SimpleNamespace(
        physics_cfg=NewtonPhysicsCfg(requires_grad=requires_grad)
    )
    sim._spawn_scene = object.__new__(SpawnScene)
    sim._spawn_scene._num_envs = 2
    sim._spawn_scene._assets = {}
    sim._spawn_scene.builder = SimpleNamespace(
        is_finalized=False,
        result=None,
        materials={},
        add_cable=lambda desc: desc,
    )
    sim._cable_objects = {}
    for name in (
        "_robots",
        "_articulations",
        "_rigid_objects",
        "_rigid_object_groups",
        "_deformable_objects",
        "_lights",
        "_sensors",
    ):
        setattr(sim, name, {})
    sim.notify_visualization_topology_changed = Mock()
    return sim


def test_manager_declaration_query_and_duplicate_uid() -> None:
    sim = _manager()
    cable = sim.add_cable_object(_cfg())
    assert cable.num_instances == 2 and cable.is_declared and cable.data is None
    assert sim.get_cable_object("wire") is cable
    assert sim.get_cable_object("missing") is None
    assert sim.get_cable_object_uid_list() == ["wire"]
    with pytest.raises(ValueError, match="already declared"):
        sim.add_cable_object(_cfg())


@pytest.mark.parametrize(
    "solver,device,grad",
    [
        ("auto", "cuda", False),
        ("vbd", "cuda", False),
        ("dexuni", "cpu", False),
        ("dexuni", "cuda", True),
    ],
)
def test_manager_rejects_unsupported_runtime_before_declaration(
    solver, device, grad
) -> None:
    sim = _manager(solver, device, grad)
    with pytest.raises(NotImplementedError):
        sim.add_cable_object(_cfg())
    assert sim._spawn_scene._assets == {} and sim._cable_objects == {}


def test_manager_reset_rejects_before_robot_state_mutation_even_if_cable_excluded() -> (
    None
):
    sim = _manager()
    sim.add_cable_object(_cfg())
    robot = Mock()
    sim._robots = {"arm": robot}
    with pytest.raises(NotImplementedError, match="no objects have been reset"):
        sim.reset_objects_state(env_ids=[0], excluded_uids=["wire"])
    robot.reset.assert_not_called()


def test_writes_convert_once_and_preserve_unselected_environment() -> None:
    cable, batch = _bound_cable()
    before = cable.data.segment_state_w
    state = before[[1]].clone()
    state[..., :3] += 0.25
    state[..., 7:] *= 0.5
    cable.write_segment_state_to_sim(state, env_ids=[1])
    torch.testing.assert_close(cable.data.segment_state_w[1], state[0])
    torch.testing.assert_close(cable.data.segment_state_w[0], before[0])
    assert batch.writes == 1
    cable.reset(env_ids=[1])
    torch.testing.assert_close(cable.data.segment_state_w, before)
    cable._entities[1].clear_wrenches.assert_called_once()
    cable._entities[0].clear_wrenches.assert_not_called()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "grad", "stride", "ids"])
def test_invalid_state_write_is_rejected_before_engine_mutation(invalid) -> None:
    cable, batch = _bound_cable()
    state = cable.data.segment_state_w
    ids = None
    if invalid == "shape":
        state = state[:, :1].contiguous()
    elif invalid == "dtype":
        state = state.double()
    elif invalid == "grad":
        state.requires_grad_()
    elif invalid == "stride":
        state = torch.zeros(2, 2, 26)[..., ::2]
    else:
        ids = [True, 0]
    with pytest.raises((ValueError, TypeError)):
        cable.write_segment_state_to_sim(state, ids)
    assert batch.writes == 0


def test_defaults_are_engine_creation_state_even_after_prior_motion() -> None:
    batch = _CableBatch()
    batch.pose[..., 4:] += 20
    scene = Mock(spec=Scene)
    scene.create_cable_batch.return_value = batch
    data = CableObjectData(_handles(), scene, torch.device("cpu"))
    torch.testing.assert_close(
        data.default_segment_state_w[..., :3], batch.default_pose[..., 4:]
    )
    assert not torch.equal(data.segment_state_w, data.default_segment_state_w)


def test_manager_cable_reset_resolves_arena_names_and_avoids_object_resets() -> None:
    sim = _manager()
    cable, _ = _bound_cable()
    sim._cable_objects = {"wire": cable}
    scene = Mock(spec=Scene)
    scene.arenas = SimpleNamespace(
        index=lambda name: {"arena_0": 2, "arena_1": 1}[name]
    )
    sim._spawn_scene.builder.result = scene
    sim._spawn_scene.builder.is_finalized = True
    sim._spawn_scene.builder.replicate_plan = SimpleNamespace(
        env_names=lambda: ("arena_0", "arena_1")
    )
    robot, sensor = Mock(), Mock()
    sim._robots = {"arm": robot}
    sim._sensors = {"camera": sensor}
    sim.reset_objects_state(env_ids=[1, 0], excluded_sensor_uids=["camera"])
    scene.reset_arenas.assert_called_once_with([1, 2])
    robot.reset.assert_not_called()
    sensor.reset.assert_not_called()


def test_finalized_cable_scene_rejects_topology_changes_before_builder_mutation() -> (
    None
):
    sim = _manager()
    sim.add_cable_object(_cfg())
    builder = sim._spawn_scene.builder
    builder.is_finalized = True
    builder.add_object = Mock()
    with pytest.raises(NotImplementedError, match="topology changes"):
        sim._spawn_scene.declare("rigid_object", "box", SimpleNamespace())
    builder.add_object.assert_not_called()
    with pytest.raises(NotImplementedError, match="pending removal"):
        sim._spawn_scene.remove("wire")


def test_default_backend_does_not_advertise_cables() -> None:
    sim = _manager()
    sim.physics = DefaultPhysicsBackend(SimpleNamespace())
    assert not sim.physics.supports_cables
    with pytest.raises(NotImplementedError, match="Newton backend"):
        sim.add_cable_object(_cfg())


@pytest.mark.parametrize("endpoint", ["start_attachment", "end_attachment"])
@pytest.mark.parametrize("qpos", [[0.025], torch.tensor([0.025])])
def test_nonzero_attachment_is_rejected_before_commit_or_control_registration(
    endpoint, qpos
):
    sim = _manager()
    cfg = _cfg()
    setattr(cfg, endpoint, CableAttachmentCfg(target="tool", link_name="tip"))
    sim.add_cable_object(cfg)
    sim._articulations["tool"] = SimpleNamespace(cfg=ArticulationCfg(init_qpos=qpos))
    sim._spawn_scene.commit = Mock()
    sim._spawn_scene.builder.add_runtime_control = Mock()
    with pytest.raises(NotImplementedError, match="before attachment construction"):
        sim.prepare()
    sim._spawn_scene.commit.assert_not_called()
    sim._spawn_scene.builder.add_runtime_control.assert_not_called()


def test_initial_joint_registration_is_retryable_after_preflight_and_idempotent():
    sim = _manager()
    cfg = _cfg()
    cfg.start_attachment = CableAttachmentCfg(target="tool", link_name="tip")
    sim.add_cable_object(cfg)
    sim._articulations["tool"] = SimpleNamespace(cfg=ArticulationCfg(init_qpos=[0.025]))
    builder = sim._spawn_scene.builder
    builder.add_runtime_control = Mock()
    builder.replicate_plan = SimpleNamespace(env_names=lambda: ("arena_0", "arena_1"))
    with pytest.raises(NotImplementedError):
        sim.physics._prepare_scene_build()
    sim._articulations["tool"].cfg.init_qpos = torch.tensor([0.0], requires_grad=True)
    sim.physics._prepare_scene_build()
    sim.physics._prepare_scene_build()
    controls = [call.args[0] for call in builder.add_runtime_control.call_args_list]
    assert [control.target for control in controls] == ["arena_0/tool", "arena_1/tool"]
    for control in controls:
        np.testing.assert_array_equal(control.positions, [[0.0]])


def test_unattached_nonzero_joint_state_is_seeded_without_rejecting_other_anchors():
    sim = _manager()
    cfg = _cfg()
    cfg.start_attachment = CableAttachmentCfg(target="table")
    sim.add_cable_object(cfg)
    sim._articulations["tool"] = SimpleNamespace(cfg=ArticulationCfg(init_qpos=[0.025]))
    builder = sim._spawn_scene.builder
    builder.add_runtime_control = Mock()
    builder.replicate_plan = SimpleNamespace(env_names=lambda: ("arena_0", "arena_1"))
    sim.physics._prepare_scene_build()
    assert builder.add_runtime_control.call_count == 2
    np.testing.assert_allclose(
        builder.add_runtime_control.call_args.args[0].positions, [[0.025]]
    )


def test_cable_scene_reset_resets_selected_lights_and_sensors_once():
    sim = _manager()
    cable, _ = _bound_cable()
    sim._cable_objects = {"wire": cable}
    scene = Mock(spec=Scene)
    scene.arenas = SimpleNamespace(
        index=lambda name: {"arena_0": 1, "arena_1": 2}[name]
    )
    sim._spawn_scene.builder.result = scene
    sim._spawn_scene.builder.is_finalized = True
    sim._spawn_scene.builder.replicate_plan = SimpleNamespace(
        env_names=lambda: ("arena_0", "arena_1")
    )
    light, sensor = Mock(), Mock()
    sim._lights = {"lamp": light}
    sim._sensors = {"camera": sensor}
    sim.reset_objects_state(env_ids=[1])
    scene.reset_arenas.assert_called_once_with([2])
    light.reset.assert_called_once_with([1])
    sensor.reset.assert_called_once_with([1])
    scene.reset_arenas.reset_mock()
    light.reset.reset_mock()
    sensor.reset.reset_mock()
    sim.reset_objects_state(env_ids=[])
    scene.reset_arenas.assert_not_called()
    light.reset.assert_not_called()
    sensor.reset.assert_not_called()


def test_preflight_validates_all_initial_states_before_registering_controls():
    sim = _manager()
    sim.add_cable_object(_cfg())
    sim._articulations = {
        "valid": SimpleNamespace(cfg=ArticulationCfg(init_qpos=[0.025])),
        "invalid": SimpleNamespace(cfg=ArticulationCfg(init_qpos=[float("nan")])),
    }
    builder = sim._spawn_scene.builder
    builder.add_runtime_control = Mock()
    with pytest.raises(ValueError, match="finite"):
        sim.physics._prepare_scene_build()
    builder.add_runtime_control.assert_not_called()
