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

"""Contract tests for the unified deformable-object API."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import Mock

from dexsim.scene import Scene

import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import (
    SurfaceDeformablePhysicsCfg,
    SurfaceDeformableObjectCfg,
    DeformableObjectCfg,
    NewtonPhysicsCfg,
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.objects import (
    SurfaceDeformableObject,
    DeformableObject,
    DeformableObjectData,
    VolumeDeformableObject,
)
from embodichain.lab.sim.physics import DefaultPhysicsBackend, NewtonPhysicsBackend
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.utils.configclass import update_class_from_dict

pytestmark = pytest.mark.no_sim


class _ParticleSet:
    def __init__(self, offset: float) -> None:
        self.positions = torch.tensor(
            [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        self.velocities = torch.zeros_like(self.positions)

    @property
    def particle_count(self) -> int:
        return len(self.positions)


class _ParticleBatch:
    def __init__(self, particle_sets: Sequence[_ParticleSet]) -> None:
        self.particle_sets = list(particle_sets)

    def fetch_particle_positions(self, out: torch.Tensor) -> int:
        out.copy_(torch.cat([item.positions for item in self.particle_sets]))
        return len(self.particle_sets)

    def fetch_particle_velocities(self, out: torch.Tensor) -> int:
        out.copy_(torch.cat([item.velocities for item in self.particle_sets]))
        return len(self.particle_sets)

    def apply_particle_positions(self, data: torch.Tensor) -> int:
        for index, particle_set in enumerate(self.particle_sets):
            start = index * particle_set.particle_count
            end = start + particle_set.particle_count
            particle_set.positions.copy_(data[start:end])
        return len(self.particle_sets)

    def apply_particle_velocities(self, data: torch.Tensor) -> int:
        for index, particle_set in enumerate(self.particle_sets):
            start = index * particle_set.particle_count
            end = start + particle_set.particle_count
            particle_set.velocities.copy_(data[start:end])
        return len(self.particle_sets)


class _ParticleScene:
    def create_particle_set_batch(
        self, particle_sets: Sequence[_ParticleSet]
    ) -> _ParticleBatch:
        return _ParticleBatch(particle_sets)


class _RenderParticleSet:
    def __init__(self, vertex_count: int) -> None:
        self.vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    def get_render_vertices(self) -> np.ndarray:
        return self.vertices

    def get_render_triangles(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.int32)


@pytest.mark.parametrize(
    "config_type", [VolumeDeformableObjectCfg, SurfaceDeformableObjectCfg]
)
def test_config_serialization_preserves_nested_values(config_type) -> None:
    cfg = config_type(uid="deformable", particle_radius=0.02)
    cfg.attrs.density = 12.0
    restored = config_type(uid="restored", particle_radius=0.01)
    update_class_from_dict(restored, cfg.to_dict())

    assert restored.to_dict() == cfg.to_dict()
    assert pickle.loads(pickle.dumps(cfg)).to_dict() == cfg.to_dict()
    copied = cfg.copy()
    copied.attrs.density = 24.0
    assert cfg.attrs.density == 12.0
    assert config_type().attrs.density != 12.0


def test_default_only_deformable_fields_are_not_accepted() -> None:
    with pytest.raises(TypeError, match="dynamic_friction"):
        VolumeDeformablePhysicsCfg(dynamic_friction=0.1)
    with pytest.raises(TypeError, match="thickness"):
        SurfaceDeformablePhysicsCfg(thickness=0.01)


def test_common_data_contract_combines_and_derives_nodal_state() -> None:
    particles = _ParticleSet(0.0)
    particles.positions[1] = torch.tensor([2.0, 4.0, 6.0])
    particles.velocities[:] = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    data = DeformableObjectData([particles], _ParticleScene(), torch.device("cpu"))

    assert data.nodal_state_w.shape == (1, 2, 6)
    torch.testing.assert_close(data.nodal_state_w[..., :3], data.nodal_pos_w)
    torch.testing.assert_close(data.nodal_state_w[..., 3:], data.nodal_vel_w)
    torch.testing.assert_close(data.root_pos_w, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(data.root_vel_w, torch.tensor([[2.0, 3.0, 4.0]]))


def test_backend_capabilities_are_newton_only() -> None:
    default = DefaultPhysicsBackend(SimpleNamespace())
    newton = NewtonPhysicsBackend(SimpleNamespace())

    assert not default.supports_volume_deformables
    assert not default.supports_surface_deformables
    assert newton.supports_volume_deformables
    assert newton.supports_surface_deformables


def test_manager_rejects_deformables_on_default_backend() -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = DefaultPhysicsBackend(SimpleNamespace())

    with pytest.raises(NotImplementedError, match="require the Newton backend"):
        sim.add_deformable_object(VolumeDeformableObjectCfg(uid="soft"))


def test_deformable_facade_rejects_default_spawn_scene() -> None:
    scene = Mock(spec=Scene)
    scene.backend = "dexsim"
    cloth = SurfaceDeformableObject(SurfaceDeformableObjectCfg(uid="cloth"))
    cloth._initialize_spawn_declaration(1)
    cloth.attach_spawn_handles([object()])

    with pytest.raises(NotImplementedError, match="Default backend"):
        cloth.bind_spawn(scene)

    assert cloth.is_declared
    assert not cloth.is_spawn_bound


@pytest.mark.parametrize("solver_type", ["mujoco_warp", "featherstone"])
def test_manager_rejects_non_particle_newton_solver(solver_type: str) -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(SimpleNamespace())
    sim.physics._configured_solver_type = solver_type
    sim.device = torch.device("cuda")

    with pytest.raises(NotImplementedError, match="does not support deformable"):
        sim.add_deformable_object(VolumeDeformableObjectCfg(uid="soft"))


@pytest.mark.parametrize("solver_type", ["auto", "dexuni"])
@pytest.mark.parametrize(
    "config_type", [SurfaceDeformableObjectCfg, VolumeDeformableObjectCfg]
)
def test_manager_declares_deformables_with_supported_solver(
    config_type,
    solver_type: str,
) -> None:
    from embodichain.lab.sim.shapes import MeshCfg
    from embodichain.lab.sim.spawn.scene import SpawnScene

    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(SimpleNamespace())
    sim.physics._configured_solver_type = solver_type
    sim.device = torch.device("cuda")
    sim.sim_config = SimpleNamespace(physics_cfg=NewtonPhysicsCfg())
    sim._deformable_objects = {}
    sim._visualization_topology_revision = 0
    sim._spawn_scene = object.__new__(SpawnScene)
    sim._spawn_scene._num_envs = 2
    sim._spawn_scene._assets = {}
    sim._spawn_scene.builder = SimpleNamespace(
        is_finalized=False,
        result=None,
        materials={},
        add_soft_object=lambda descriptor: descriptor,
        add_cloth_object=lambda descriptor: descriptor,
    )

    facade = sim.add_deformable_object(
        config_type(uid="deformable", shape=MeshCfg(fpath="mesh.obj"))
    )

    assert facade.is_declared
    assert facade.num_instances == 2
    assert sim.get_deformable_object("deformable") is facade


def test_manager_rejects_gradient_mode_deformable_mutation() -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(SimpleNamespace())
    sim.physics._configured_solver_type = "vbd"
    sim.device = torch.device("cuda")
    sim.sim_config = SimpleNamespace(physics_cfg=NewtonPhysicsCfg(requires_grad=True))

    with pytest.raises(NotImplementedError, match="requires_grad=True"):
        sim.add_deformable_object(VolumeDeformableObjectCfg(uid="soft"))


def test_particle_data_fetches_and_partially_applies_packed_state() -> None:
    particle_sets = [_ParticleSet(0.0), _ParticleSet(10.0)]
    data = DeformableObjectData(
        particle_sets,
        _ParticleScene(),
        torch.device("cpu"),
    )

    assert data.n_nodes == 2
    default_state = data.default_nodal_state_w
    torch.testing.assert_close(
        data.nodal_pos_w,
        torch.stack([item.positions for item in particle_sets]),
    )
    torch.testing.assert_close(
        data.default_nodal_state_w[..., 3:],
        torch.zeros((2, 2, 3)),
    )

    positions = torch.tensor(
        [[[20.0, 1.0, 2.0], [21.0, 3.0, 4.0]]], dtype=torch.float32
    )
    velocities = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32)
    data._apply_nodal_state(positions, velocities, env_ids=[1])

    torch.testing.assert_close(
        particle_sets[0].positions[:, 0], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(particle_sets[1].positions, positions[0])
    torch.testing.assert_close(particle_sets[1].velocities, velocities[0])
    torch.testing.assert_close(data.default_nodal_state_w, default_state)


def test_particle_data_requires_equal_topology() -> None:
    particle_sets: list[Any] = [_ParticleSet(0.0), _ParticleSet(1.0)]
    particle_sets[1].positions = torch.zeros((3, 3), dtype=torch.float32)
    particle_sets[1].velocities = torch.zeros((3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="same particle count"):
        DeformableObjectData(
            particle_sets,
            _ParticleScene(),
            torch.device("cpu"),
        )


def test_deformable_rejects_replicated_render_vertex_mismatch() -> None:
    deformable = object.__new__(SurfaceDeformableObject)
    deformable.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="render-clone topology mismatch"):
        deformable._initialize_topology([_RenderParticleSet(3), _RenderParticleSet(4)])


def test_manager_stores_both_topologies_in_one_registry() -> None:
    sim = object.__new__(SimulationManager)
    volume = object.__new__(VolumeDeformableObject)
    surface = object.__new__(SurfaceDeformableObject)
    sim._deformable_objects = {"volume": volume, "surface": surface}

    assert sim.get_deformable_object("volume") is volume
    assert sim.get_deformable_object_uid_list() == ["volume", "surface"]


@pytest.mark.parametrize(
    "property_name",
    [
        "nodal_pos_w",
        "nodal_vel_w",
        "nodal_state_w",
        "default_nodal_state_w",
        "root_pos_w",
        "root_vel_w",
    ],
)
def test_data_reads_return_independent_snapshots(property_name: str) -> None:
    particles = _ParticleSet(1.0)
    data = DeformableObjectData([particles], _ParticleScene(), torch.device("cpu"))
    expected = getattr(data, property_name).clone()
    snapshot = getattr(data, property_name)
    snapshot.fill_(100.0)
    torch.testing.assert_close(getattr(data, property_name), expected)

    snapshot = getattr(data, property_name)
    expected = snapshot.clone()
    particles.positions.add_(2.0)
    particles.velocities.add_(1.0)
    _ = getattr(data, property_name)
    torch.testing.assert_close(snapshot, expected)
