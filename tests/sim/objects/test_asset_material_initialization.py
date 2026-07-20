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

import torch

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
    asset._all_indices = [0]
    asset._visual_material_reset_generation = [0]
    asset._has_original_visual_material = False
    asset.is_shared_visual_material = False
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


def test_asset_types_do_not_restore_before_initial_material_capture():
    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, MagicMock(name="material"))
        render_body = asset._entities[0].get_render_body()

        asset.restore_visual_material()

        render_body.clean_material.assert_not_called()


def test_asset_types_restore_original_material_after_replacement():
    originals = [MagicMock(name=f"original_{mesh_id}") for mesh_id in range(2)]
    for mesh_id, original in enumerate(originals):
        original.get_name.return_value = f"asset_material_{mesh_id}"
        original.get_template.return_value = MagicMock(name=f"template_{mesh_id}")

    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, originals[0])
        render_body = asset._entities[0].get_render_body()
        render_body.get_mesh_count.return_value = len(originals)
        render_body.get_material.side_effect = originals
        asset._initialize_existing_visual_material()

        replacement = MagicMock(name="replacement")
        replacement.uid = "replacement"
        replacement_inst = MagicMock(name="replacement_inst")
        replacement.create_instance.return_value = replacement_inst

        asset.set_visual_material(replacement)
        render_body.get_material.side_effect = None
        render_body.get_material.return_value = replacement_inst.mat
        asset.restore_visual_material()

        render_body.clean_material.assert_not_called()
        assert render_body.set_material.call_args_list == [
            call(mesh_id, original) for mesh_id, original in enumerate(originals)
        ]
        assert _get_registered_material(asset).mat is originals[0]
        assert asset._visual_material_reset_generation == [1]
        assert asset.is_shared_visual_material is False


def test_asset_types_restore_default_for_originally_unmaterialed_segment():
    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, None)
        render_body = asset._entities[0].get_render_body()
        asset._initialize_existing_visual_material()

        replacement = MagicMock(name="replacement")
        replacement.uid = "replacement"
        replacement_inst = MagicMock(name="replacement_inst")
        replacement.create_instance.return_value = replacement_inst
        asset.set_visual_material(replacement)
        render_body.get_material.side_effect = [replacement_inst.mat, None]
        asset.restore_visual_material()

        render_body.clean_material.assert_called_once_with()
        render_body.set_material.assert_not_called()
        assert _get_registered_material(asset) is None


def test_asset_types_skip_setter_when_original_material_is_already_attached():
    originals = [MagicMock(name=f"original_{mesh_id}") for mesh_id in range(2)]
    for mesh_id, original in enumerate(originals):
        original.get_name.return_value = f"asset_material_{mesh_id}"
        template = MagicMock(name=f"template_{mesh_id}")
        template.get_name.return_value = f"template_{mesh_id}"
        original.get_template.return_value = template
    equivalent_handle = MagicMock(name="equivalent_original_1")
    equivalent_handle.get_name.return_value = "asset_material_1"
    equivalent_template = MagicMock(name="equivalent_template_1")
    equivalent_template.get_name.return_value = "template_1"
    equivalent_handle.get_template.return_value = equivalent_template
    current_materials = [originals[0], equivalent_handle]

    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, originals[0])
        render_body = asset._entities[0].get_render_body()
        render_body.get_mesh_count.return_value = len(originals)
        render_body.get_material.side_effect = originals
        asset._initialize_existing_visual_material()
        render_body.get_material.side_effect = lambda mesh_id: current_materials[
            mesh_id
        ]

        asset.restore_visual_material()

        render_body.clean_material.assert_not_called()
        render_body.set_material.assert_not_called()


def test_asset_types_only_restore_changed_material_segments():
    originals = [MagicMock(name=f"original_{mesh_id}") for mesh_id in range(2)]
    for mesh_id, original in enumerate(originals):
        original.get_name.return_value = f"asset_material_{mesh_id}"
        original.get_template.return_value = MagicMock(name=f"template_{mesh_id}")
    replacement = MagicMock(name="replacement")
    replacement.get_name.return_value = "replacement_material"

    for asset_type in _ASSET_TYPES:
        asset = _make_asset(asset_type, originals[0])
        render_body = asset._entities[0].get_render_body()
        render_body.get_mesh_count.return_value = len(originals)
        render_body.get_material.side_effect = originals
        asset._initialize_existing_visual_material()
        current_materials = [originals[0], replacement]
        render_body.get_material.side_effect = lambda mesh_id: current_materials[
            mesh_id
        ]

        asset.restore_visual_material()

        render_body.clean_material.assert_not_called()
        render_body.set_material.assert_called_once_with(1, originals[1])


def test_asset_reset_calls_material_restore_for_selected_environments():
    for asset_type in _ASSET_TYPES:
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
