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

from pathlib import Path
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
    sim._spawn_scene._resolve_materials = lambda materials: materials
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
        "_fluid_objects",
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
    from embodichain.lab.sim.material import VisualMaterialCfg

    scene = sim._spawn_scene
    del scene._resolve_materials
    scene._usd_texture_files = SimpleNamespace(resolve=lambda materials: materials)
    scene._resource_material_definitions = {}
    scene._material_resources = SimpleNamespace(renderer="no-render")
    duplicate = _cfg()
    duplicate.visual_material = VisualMaterialCfg(uid="shared", base_color=[1, 0, 0, 1])
    with pytest.raises(ValueError, match="already declared"):
        sim.add_cable_object(duplicate)

    other = _cfg()
    other.uid = "other"
    other.visual_material = VisualMaterialCfg(uid="shared", base_color=[0, 1, 0, 1])
    assert sim.add_cable_object(other) is sim.get_cable_object("other")
    assert sim.get_cable_object("wire") is cable
    assert cable.cfg.visual_material is None


@pytest.mark.parametrize("deformable_type", ["volume", "surface"])
@pytest.mark.parametrize("cable_first", [True, False])
def test_particle_and_cable_declarations_preserve_first_asset(
    deformable_type: str, cable_first: bool
) -> None:
    from embodichain.lab.sim.cfg import (
        SurfaceDeformableObjectCfg,
        VolumeDeformableObjectCfg,
    )

    sim = _manager()
    scene = sim._spawn_scene
    scene.builder.add_soft_object = scene.builder.add_cloth_object = lambda desc: desc
    cfg_type = (
        VolumeDeformableObjectCfg
        if deformable_type == "volume"
        else SurfaceDeformableObjectCfg
    )
    mesh = {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "triangles": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    }
    deformable_cfg = cfg_type(uid="particles", shape=mesh)
    add_first = sim.add_cable_object if cable_first else sim.add_deformable_object
    add_second = sim.add_deformable_object if cable_first else sim.add_cable_object
    first_cfg, second_cfg = (
        (_cfg(), deformable_cfg) if cable_first else (deformable_cfg, _cfg())
    )
    first = add_first(first_cfg)
    with pytest.raises(NotImplementedError, match="same arena"):
        add_second(second_cfg)
    assert sim.get_asset(first_cfg.uid) is first
    assert second_cfg.uid not in sim.asset_uids
    assert second_cfg.uid not in scene


def _usd_import_manager() -> SimulationManager:
    from dexsim.spawn import SceneBuilder, SceneDesc

    sim = _manager()
    scene = sim._spawn_scene
    # Keep the SDK's descriptor inventory and add_scene traversal without
    # allocating a native World or particle solver.
    builder = object.__new__(SceneBuilder)
    builder.backend = "newton"
    builder.scene_desc = SceneDesc(name="usd_particle_regression")
    builder.result = None
    builder._finalized = False
    builder._adapter = SimpleNamespace(backend="newton", add=Mock())
    builder.replicate_plan = SimpleNamespace(env_names=lambda: ("arena_0", "arena_1"))
    scene.builder = builder
    del scene._resolve_materials
    scene._usd_texture_files = SimpleNamespace(resolve=lambda materials: materials)
    scene._resource_material_definitions = {}
    scene._material_resources = SimpleNamespace(renderer="no-render")
    scene._resolve_materials = Mock(wraps=scene._resolve_materials)
    sim._visual_objects = {}
    return sim


def _write_cable_usd(tmp_path: Path, *, particle_source: Path | None = None) -> Path:
    source = (
        particle_source.read_text() if particle_source is not None else "#usda 1.0\n"
    )
    path = tmp_path / "cables.usda"
    path.write_text(source + """
def BasisCurves "Cable" (
    prepend apiSchemas = ["PhysicsCurvesDeformableSimAPI"]
)
{
    uniform token type = "linear"
    uniform token wrap = "nonperiodic"
    int[] curveVertexCounts = [2]
    point3f[] points = [(0, 0, 1), (0.1, 0, 1)]
}
""")
    return path


@pytest.mark.parametrize("cable_first", [True, False])
@pytest.mark.parametrize("cable_source", ["typed", "usd"])
def test_usd_particles_and_cables_reject_mix_before_scene_mutation(
    tmp_path: Path, volume_particle_usd: Path, cable_first: bool, cable_source: str
) -> None:
    from dexsim.spawn import SoftBodyDesc
    from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg
    from embodichain.lab.sim.material import VisualMaterialCfg

    sim = _usd_import_manager()
    scene = sim._spawn_scene
    builder = scene.builder
    recipe = UsdImportCfg(
        name="volume", source=UsdSourceCfg(file_path=str(volume_particle_usd))
    )
    cable_cfg = _cfg()
    cable_cfg.visual_material = VisualMaterialCfg(
        uid="cable_material", base_color=[1, 0, 0, 1]
    )

    def add_cable():
        if cable_source == "typed":
            return sim.add_cable_object(cable_cfg)
        imported = sim.add_usd(
            UsdImportCfg(
                name="cables",
                source=UsdSourceCfg(file_path=str(_write_cable_usd(tmp_path))),
            )
        )
        cable = imported["/Cable"]
        assert list(imported.resource_types.values()) == ["cable"]
        assert len(builder.cables) == 1
        return cable

    if cable_first:
        first = add_cable()
    else:
        imported = sim.add_usd(recipe)
        first = imported["/World/Soft"]
        assert len(builder.particle_sets) == 1
        assert isinstance(builder.particle_sets[0], SoftBodyDesc)
        assert sim.get_asset(first.uid) is first

    initial_entities = [id(item) for item in builder.scene_desc.iter_entities()]
    initial_materials = dict(builder.materials)
    initial_definitions = dict(scene._resource_material_definitions)
    initial_assets = dict(scene._assets)
    scene._resolve_materials.reset_mock()
    builder._adapter.add.reset_mock()
    with pytest.raises(NotImplementedError, match="same arena"):
        if cable_first:
            sim.add_usd(recipe)
        else:
            add_cable()

    assert [id(item) for item in builder.scene_desc.iter_entities()] == initial_entities
    assert builder.materials == initial_materials
    assert scene._resource_material_definitions == initial_definitions
    assert scene._assets == initial_assets
    if first is not None:
        assert sim.get_asset(first.uid) is first
    scene._resolve_materials.assert_not_called()
    builder._adapter.add.assert_not_called()


def test_one_usd_import_rejects_cables_and_particles_before_scene_mutation(
    tmp_path: Path,
    volume_particle_usd: Path,
) -> None:
    from dexsim.kit.usd import parse_usd
    from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg

    source = _write_cable_usd(tmp_path, particle_source=volume_particle_usd)
    parsed = parse_usd(str(source))
    assert len(parsed.cables) == len(parsed.particle_sets) == 1
    sim = _usd_import_manager()
    scene = sim._spawn_scene
    with pytest.raises(NotImplementedError, match="same arena"):
        sim.add_usd(
            UsdImportCfg(name="mixed", source=UsdSourceCfg(file_path=str(source)))
        )
    assert list(scene.builder.scene_desc.iter_entities()) == []
    assert not scene._assets and not sim._deformable_objects
    assert not scene.builder.materials and not scene._resource_material_definitions
    scene._resolve_materials.assert_not_called()
    scene.builder._adapter.add.assert_not_called()


@pytest.mark.parametrize("renderer", ["rt", "no-render"])
def test_generated_cable_material_uses_scene_resource_resolution(renderer) -> None:
    from embodichain.lab.sim._material_presets import _AssetMaterialDesc
    from embodichain.lab.sim.material import VisualMaterialCfg

    sim = _manager()
    scene = sim._spawn_scene
    del scene._resolve_materials
    scene._usd_texture_files = SimpleNamespace(resolve=lambda materials: materials)
    scene._resource_material_definitions = {}
    native_material = object()
    scene._material_resources = SimpleNamespace(
        renderer=renderer, resolve=Mock(return_value=(object(), native_material))
    )
    cfg = _cfg()
    cfg.visual_material = VisualMaterialCfg(parameters={"transmission": 1.0})

    sim.add_cable_object(cfg)

    descriptor = scene._assets["wire"].descriptor
    assert scene.builder.materials == {}
    if renderer == "no-render":
        scene._material_resources.resolve.assert_not_called()
        assert isinstance(descriptor.render_material, _AssetMaterialDesc)
    else:
        assert descriptor.render_material is native_material


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


def test_rejected_rigid_declaration_preserves_material_name_for_no_render_visual() -> (
    None
):
    from embodichain.lab.sim.cfg import RigidObjectCfg, VisualObjectCfg
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = _manager(device="cpu")
    sim._visual_objects = {}
    scene = sim._spawn_scene
    del scene._resolve_materials
    scene._usd_texture_files = SimpleNamespace(resolve=lambda materials: materials)
    scene._resource_material_definitions = {}
    scene._material_resources = SimpleNamespace(renderer="no-render")
    scene.track("cable_object", "wire", SimpleNamespace(name="wire"))
    scene.builder.is_finalized = True
    scene.builder.add_object = Mock()
    with pytest.raises(NotImplementedError, match="topology changes"):
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="blocked",
                shape=CubeCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=VisualMaterialCfg(
                        uid="reserved", base_color=[1, 0, 0, 1]
                    ),
                ),
            )
        )

    visual = sim.add_visual_object(
        VisualObjectCfg(
            uid="allowed",
            shape=CubeCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=VisualMaterialCfg(
                    uid="reserved", base_color=[0, 1, 0, 1]
                ),
            ),
        )
    )
    assert sim.get_visual_object("allowed") is visual
    assert list(visual.cfg.shape.visual_material.base_color) == [0, 1, 0, 1]
    assert "blocked" not in scene and "blocked" not in sim._rigid_objects
    assert scene._assets["allowed"].declaration_only
    scene.builder.add_object.assert_not_called()


@pytest.mark.parametrize(
    "fixed_kind", ["cable_object", "fluid_object", "cloth_object", "soft_object"]
)
@pytest.mark.parametrize("finalized", [False, True])
@pytest.mark.parametrize("mounted_camera", [False, True])
def test_robot_removal_preserves_mounted_resources_if_scene_is_immutable(
    fixed_kind: str, finalized: bool, mounted_camera: bool
) -> None:
    from embodichain.lab.sim.sensors import Camera, CameraCfg
    from embodichain.lab.visualization.markers import (
        MarkerGroup,
        MarkerGroupCfg,
        MarkerPrototypeCfg,
    )

    sim = _manager()
    scene = sim._spawn_scene
    camera = Mock(spec=Camera)
    camera.cfg = CameraCfg(
        uid="mounted", extrinsics=CameraCfg.ExtrinsicsCfg(parent="robot/tool")
    )
    scene.track(fixed_kind, "fixed", SimpleNamespace(name="fixed"))
    robot = Mock(
        cfg=SimpleNamespace(
            enable_sensor=mounted_camera,
            sensors=[camera.cfg] if mounted_camera else [],
        )
    )
    scene.track("articulation", "robot", SimpleNamespace(name="robot"), facade=robot)
    sim._robots["robot"] = robot
    if mounted_camera:
        sim._sensors["mounted"] = camera
    parent_poses = np.array([[1, 0, 0, 0, 0, 0, 1]] * 2, dtype=np.float64)
    group = MarkerGroup(
        MarkerGroupCfg(name="attached", prototypes={"box": MarkerPrototypeCfg()}),
        num_envs=2,
        pose_resolver=lambda parent, link, ids: parent_poses[ids],
    )
    group.update(translations=np.zeros((2, 1, 3)))
    group.attach("robot")
    sim._marker_groups = {group.name: group}
    scene.builder.is_finalized = finalized
    scene.builder.remove_articulation = Mock(
        return_value=scene._assets["robot"].descriptor
    )
    if finalized:
        with pytest.raises(NotImplementedError, match="topology changes"):
            sim.remove_asset("robot")
        assert sim.get_asset("robot") is robot
        if mounted_camera:
            assert sim.get_sensor("mounted") is camera
        camera.destroy.assert_not_called()
        scene.builder.remove_articulation.assert_not_called()
        parent_poses[:, 0] = 2
        group.update(colors=np.ones((2, 1, 4)))
        np.testing.assert_allclose(
            [marker.position for marker in group.snapshot()], [[2, 0, 0]] * 2
        )
    else:
        sim._gizmos = {}
        sim._disabled_robot_gizmos = set()
        assert sim.remove_asset("robot")
        assert "robot" not in scene and "mounted" not in sim.get_sensor_uid_list()
        assert camera.destroy.call_count == int(mounted_camera)
        parent_poses[:, 0] = 2
        group.update(colors=np.ones((2, 1, 4)))
        np.testing.assert_allclose(
            [marker.position for marker in group.snapshot()], [[1, 0, 0]] * 2
        )


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
    builder.add_runtime_control.assert_called_once()
    controls = builder.add_runtime_control.call_args.args[0].seeds
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
    builder.add_runtime_control.assert_called_once()
    for seed in builder.add_runtime_control.call_args.args[0].seeds:
        np.testing.assert_allclose(seed.positions, [[0.025]])


def test_cable_scene_reset_restores_selected_visuals_lights_and_sensors_once():
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
    visual = Mock(is_spawn_bound=True)
    excluded_visual = Mock(is_spawn_bound=True)
    unbound_visual = Mock(is_spawn_bound=False)
    sim._visual_objects = {
        "marker": visual,
        "retained": excluded_visual,
        "unbound": unbound_visual,
    }
    sim._lights = {"lamp": light}
    sim._sensors = {"camera": sensor}
    sim.reset_objects_state(env_ids=[1], excluded_uids=["retained"])
    scene.reset_arenas.assert_called_once_with([2])
    visual.reset.assert_called_once_with([1])
    excluded_visual.reset.assert_not_called()
    unbound_visual.reset.assert_not_called()
    light.reset.assert_called_once_with([1])
    sensor.reset.assert_called_once_with([1])
    scene.reset_arenas.reset_mock()
    light.reset.reset_mock()
    sensor.reset.reset_mock()
    visual.reset.reset_mock()
    sim.reset_objects_state(env_ids=[])
    scene.reset_arenas.assert_not_called()
    light.reset.assert_not_called()
    sensor.reset.assert_not_called()
    visual.reset.assert_not_called()


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
