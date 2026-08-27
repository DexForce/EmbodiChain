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

import torch

from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
    DeformableObjectCfg,
    SoftObjectCfg,
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


def test_legacy_configs_specialize_common_deformable_config() -> None:
    assert issubclass(SoftObjectCfg, VolumeDeformableObjectCfg)
    assert issubclass(ClothObjectCfg, SurfaceDeformableObjectCfg)
    assert issubclass(VolumeDeformableObjectCfg, DeformableObjectCfg)
    assert issubclass(SurfaceDeformableObjectCfg, DeformableObjectCfg)
    assert SoftObjectCfg().deformable_type == "volume"
    assert ClothObjectCfg().deformable_type == "surface"


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


def test_backend_capabilities_keep_newton_deformable_entry_disabled() -> None:
    default = DefaultPhysicsBackend(SimpleNamespace())
    newton = NewtonPhysicsBackend(SimpleNamespace())

    assert default.supports_volume_deformables
    assert default.supports_surface_deformables
    assert default.supports_soft_bodies
    assert default.supports_cloth
    assert not newton.supports_volume_deformables
    assert not newton.supports_surface_deformables
    assert not newton.supports_soft_bodies
    assert not newton.supports_cloth


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
