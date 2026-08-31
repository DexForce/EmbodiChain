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

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import (
    ClothPhysicalAttributesCfg,
    ClothObjectCfg,
    DeformableObjectCfg,
    NewtonPhysicsCfg,
    SoftObjectCfg,
    SoftbodyPhysicalAttributesCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.objects import (
    ClothBodyData,
    ClothObject,
    DeformableObject,
    DeformableObjectData,
    SoftBodyData,
    SoftObject,
    SurfaceDeformableData,
    SurfaceDeformableObject,
    VolumeDeformableData,
    VolumeDeformableObject,
)
from embodichain.lab.sim.physics import DefaultPhysicsBackend, NewtonPhysicsBackend
from embodichain.lab.sim.sim_manager import SimulationManager


class _Data(DeformableObjectData):
    def __init__(self) -> None:
        self._pos = torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]], dtype=torch.float32
        )
        self._vel = torch.tensor(
            [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=torch.float32
        )

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self._pos

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self._vel

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        return torch.cat((self._pos, torch.zeros_like(self._vel)), dim=-1)


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


def test_legacy_configs_specialize_common_deformable_config() -> None:
    assert issubclass(SoftObjectCfg, VolumeDeformableObjectCfg)
    assert issubclass(ClothObjectCfg, SurfaceDeformableObjectCfg)
    assert issubclass(VolumeDeformableObjectCfg, DeformableObjectCfg)
    assert issubclass(SurfaceDeformableObjectCfg, DeformableObjectCfg)
    assert SoftObjectCfg().deformable_type == "volume"
    assert ClothObjectCfg().deformable_type == "surface"


def test_default_only_deformable_fields_are_not_accepted() -> None:
    with pytest.raises(TypeError, match="dynamic_friction"):
        SoftbodyPhysicalAttributesCfg(dynamic_friction=0.1)
    with pytest.raises(TypeError, match="thickness"):
        ClothPhysicalAttributesCfg(thickness=0.01)


def test_legacy_objects_are_aliases_of_topology_specializations() -> None:
    assert SoftObject is VolumeDeformableObject
    assert ClothObject is SurfaceDeformableObject
    assert SoftBodyData is VolumeDeformableData
    assert ClothBodyData is SurfaceDeformableData
    assert issubclass(SoftObject, DeformableObject)
    assert issubclass(ClothObject, DeformableObject)


def test_common_data_contract_combines_and_derives_nodal_state() -> None:
    data = _Data()

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
    assert not default.supports_soft_bodies
    assert not default.supports_cloth
    assert newton.supports_volume_deformables
    assert newton.supports_surface_deformables
    assert newton.supports_soft_bodies
    assert newton.supports_cloth


def test_manager_rejects_deformables_on_default_backend() -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = DefaultPhysicsBackend(SimpleNamespace())

    with pytest.raises(NotImplementedError, match="require the Newton backend"):
        sim.add_deformable_object(SoftObjectCfg(uid="soft"))


def test_deformable_facade_rejects_default_spawn_scene() -> None:
    scene = SimpleNamespace(backend="dexsim")

    with pytest.raises(NotImplementedError, match="Default backend"):
        SurfaceDeformableObject(
            ClothObjectCfg(uid="cloth"),
            entities=[object()],
            device=torch.device("cpu"),
            spawn_result=scene,
        )


@pytest.mark.parametrize("solver_type", ["mujoco_warp", "featherstone"])
def test_manager_rejects_non_particle_newton_solver(solver_type: str) -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(SimpleNamespace())
    sim.physics.solver_type = solver_type
    sim.device = torch.device("cuda")

    with pytest.raises(NotImplementedError, match="does not support deformable"):
        sim.add_deformable_object(SoftObjectCfg(uid="soft"))


def test_manager_rejects_gradient_mode_deformable_mutation() -> None:
    sim = object.__new__(SimulationManager)
    sim.physics = NewtonPhysicsBackend(SimpleNamespace())
    sim.physics.solver_type = "vbd"
    sim.device = torch.device("cuda")
    sim.sim_config = SimpleNamespace(physics_cfg=NewtonPhysicsCfg(requires_grad=True))

    with pytest.raises(NotImplementedError, match="requires_grad=True"):
        sim.add_deformable_object(SoftObjectCfg(uid="soft"))


def test_particle_data_fetches_and_partially_applies_packed_state() -> None:
    particle_sets = [_ParticleSet(0.0), _ParticleSet(10.0)]
    data = SurfaceDeformableData(
        particle_sets,
        _ParticleScene(),
        torch.device("cpu"),
    )

    assert data.n_vertices == 2
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


def test_particle_data_requires_equal_topology() -> None:
    particle_sets: list[Any] = [_ParticleSet(0.0), _ParticleSet(1.0)]
    particle_sets[1].positions = torch.zeros((3, 3), dtype=torch.float32)
    particle_sets[1].velocities = torch.zeros((3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="same particle count"):
        SurfaceDeformableData(
            particle_sets,
            _ParticleScene(),
            torch.device("cpu"),
        )


def test_deformable_rejects_replicated_render_vertex_mismatch() -> None:
    deformable = object.__new__(SurfaceDeformableObject)
    deformable.device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="render-clone topology mismatch"):
        deformable._initialize_topology([_RenderParticleSet(3), _RenderParticleSet(4)])


def test_manager_generic_and_legacy_getters_share_one_registry() -> None:
    sim = object.__new__(SimulationManager)
    volume = object.__new__(VolumeDeformableObject)
    surface = object.__new__(SurfaceDeformableObject)
    sim._deformable_objects = {"volume": volume, "surface": surface}

    assert sim.get_deformable_object("volume") is volume
    assert sim.get_soft_object("volume") is volume
    assert sim.get_cloth_object("surface") is surface
    assert sim.get_deformable_object_uid_list() == ["volume", "surface"]
    assert sim.get_soft_object_uid_list() == ["volume"]
    assert sim.get_cloth_object_uid_list() == ["surface"]
