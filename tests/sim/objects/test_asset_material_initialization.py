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
from unittest.mock import MagicMock, call

import pytest
import torch

from embodichain.lab.sim.material import VisualMaterialInst
from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.objects.cloth_object import ClothObject
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.soft_object import SoftObject

_ASSET_TYPES = (RigidObject, Articulation, SoftObject, ClothObject)
_LINK_NAME = "link"


@pytest.fixture(params=_ASSET_TYPES, ids=lambda asset_type: asset_type.__name__)
def asset_type(request):
    return request.param


def _material(name: str):
    material = MagicMock(name=name)
    material.get_name.return_value = name
    template = MagicMock(name=f"{name}_template")
    template.get_name.return_value = f"{name}_template"
    material.get_template.return_value = template
    return material


def _make_asset(asset_type, materials):
    render_body = MagicMock(name="render_body")
    render_body.get_mesh_count.return_value = len(materials)
    render_body.get_material.side_effect = lambda mesh_id: materials[mesh_id]

    entity = MagicMock(name="entity")
    entity.get_render_body.return_value = render_body

    asset = asset_type.__new__(asset_type)
    asset._entities = [entity]
    asset._all_indices = [0]
    asset.is_shared_visual_material = False
    asset.uid = asset_type.__name__
    if asset_type is Articulation:
        asset.link_names = [_LINK_NAME]
        asset._visual_material = [{}]
    else:
        asset._visual_material = [None]
    return asset, render_body


def _registered_material(asset):
    material = asset.get_visual_material_inst()[0]
    return material.get(_LINK_NAME) if isinstance(asset, Articulation) else material


def _replacement_material():
    replacement = MagicMock(name="replacement")
    replacement.uid = "replacement"
    replacement.create_instance.return_value = MagicMock(name="replacement_inst")
    return replacement


def test_asset_wraps_existing_material_during_initialization(asset_type):
    original = _material("original")
    asset, _ = _make_asset(asset_type, [original])

    asset._initialize_existing_visual_material()

    registered = _registered_material(asset)
    assert isinstance(registered, VisualMaterialInst)
    assert registered.mat is original


def test_asset_keeps_empty_state_without_material(asset_type):
    asset, _ = _make_asset(asset_type, [None])

    asset._initialize_existing_visual_material()

    assert _registered_material(asset) is None


def test_asset_does_not_restore_before_initial_capture(asset_type):
    asset, render_body = _make_asset(asset_type, [_material("original")])

    asset.restore_visual_material()

    render_body.clean_material.assert_not_called()
    render_body.set_material.assert_not_called()


def test_asset_restores_original_material_after_replacement(asset_type):
    originals = [_material(f"original_{mesh_id}") for mesh_id in range(2)]
    current = list(originals)
    asset, render_body = _make_asset(asset_type, current)
    asset._initialize_existing_visual_material()

    replacement = _replacement_material()
    asset.set_visual_material(replacement)
    current[:] = [replacement.create_instance.return_value.mat] * len(originals)
    asset.restore_visual_material()

    render_body.clean_material.assert_not_called()
    assert render_body.set_material.call_args_list == [
        call(mesh_id, original) for mesh_id, original in enumerate(originals)
    ]
    assert _registered_material(asset).mat is originals[0]
    assert asset.is_shared_visual_material is False


def test_asset_restores_empty_original_assignment(asset_type):
    current = [None]
    asset, render_body = _make_asset(asset_type, current)
    asset._initialize_existing_visual_material()

    replacement = _replacement_material()
    asset.set_visual_material(replacement)
    current[0] = replacement.create_instance.return_value.mat
    asset.restore_visual_material()

    render_body.clean_material.assert_called_once_with()
    render_body.set_material.assert_not_called()
    assert _registered_material(asset) is None


def test_asset_skips_equivalent_material_handles(asset_type):
    originals = [_material(f"original_{mesh_id}") for mesh_id in range(2)]
    current = list(originals)
    asset, render_body = _make_asset(asset_type, current)
    asset._initialize_existing_visual_material()
    equivalent = _material("original_1")
    current[1] = equivalent

    asset.restore_visual_material()

    render_body.clean_material.assert_not_called()
    render_body.set_material.assert_not_called()


def test_asset_restores_only_changed_segments(asset_type):
    originals = [_material(f"original_{mesh_id}") for mesh_id in range(2)]
    current = list(originals)
    asset, render_body = _make_asset(asset_type, current)
    asset._initialize_existing_visual_material()
    current[1] = _material("replacement")

    asset.restore_visual_material()

    render_body.clean_material.assert_not_called()
    render_body.set_material.assert_called_once_with(1, originals[1])


def test_asset_reset_restores_selected_environment_material(asset_type):
    asset = asset_type.__new__(asset_type)
    asset._all_indices = [0]
    asset.device = torch.device("cpu")
    asset.cfg = SimpleNamespace(
        attrs=MagicMock(),
        init_pos=(0.0, 0.0, 0.0),
        init_rot=(0.0, 0.0, 0.0),
        init_qpos=(0.0,),
    )
    asset.restore_visual_material = MagicMock()
    asset.set_local_pose = MagicMock()

    if asset_type is RigidObject:
        asset.set_attrs = MagicMock()
        asset.clear_dynamics = MagicMock()
    elif asset_type is Articulation:
        asset.set_qpos = MagicMock()
        asset.clear_dynamics = MagicMock()
        asset._world = MagicMock()

    asset.reset(env_ids=[0])

    asset.restore_visual_material.assert_called_once_with(env_ids=[0])
