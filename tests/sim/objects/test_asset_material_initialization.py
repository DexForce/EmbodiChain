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

from unittest.mock import MagicMock

from embodichain.lab.sim.material import VisualMaterialInst
from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.objects.cloth_object import ClothObject
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.soft_object import SoftObject

_ASSET_TYPES = (RigidObject, Articulation, SoftObject, ClothObject)
_LINK_NAME = "link"


def _make_asset(asset_type, material):
    render_body = MagicMock(name="render_body")
    render_body.get_mesh_count.return_value = 1
    render_body.get_material.return_value = material

    entity = MagicMock(name="entity")
    entity.get_render_body.return_value = render_body

    asset = asset_type.__new__(asset_type)
    asset._entities = [entity]
    asset.uid = asset_type.__name__
    if asset_type is Articulation:
        asset.link_names = [_LINK_NAME]
        asset._visual_material = [{}]
    else:
        asset._visual_material = [None]
    return asset


def _get_registered_material(asset):
    registered = asset.get_visual_material_inst()[0]
    return registered.get(_LINK_NAME) if isinstance(asset, Articulation) else registered


def test_asset_types_wrap_existing_material_during_initialization():
    material = MagicMock(name="material")
    material.get_name.return_value = "asset_material"
    material.get_template.return_value = MagicMock(name="template")

    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, material)
        asset._initialize_existing_visual_material()
        registered = _get_registered_material(asset)

        assert isinstance(registered, VisualMaterialInst)
        assert registered.mat is material


def test_asset_types_keep_empty_state_without_material():
    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, None)
        asset._initialize_existing_visual_material()

        assert _get_registered_material(asset) is None
