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

import pytest
import torch

from embodichain.lab.sim.material import VisualMaterialInst
from embodichain.lab.gym.envs.managers.randomization.visual import (
    _normalize_env_ids,
    _select_texture_indices,
    randomize_visual_material,
)


class FakeMaterialInstance:
    def __init__(self):
        self.base_color_map = None

    def set_base_color_map(self, texture):
        self.base_color_map = texture


class FakeMaterial:
    def __init__(self):
        self.instances = {}

    def get_inst(self, uid):
        return self.instances.setdefault(uid, FakeMaterialInstance())


def test_set_base_color_texture_uses_texture_ref():
    material = FakeMaterial()
    instance = VisualMaterialInst("instance", material)
    texture_ref = object()

    instance.set_base_color_texture(texture_ref=texture_ref)

    assert material.get_inst("instance").base_color_map is texture_ref
    assert instance.base_color_texture is texture_ref


def test_normalize_env_ids_supports_all_input_forms():
    assert _normalize_env_ids(None, 3) == [0, 1, 2]
    assert _normalize_env_ids(torch.tensor([2, 0]), 3) == [2, 0]
    assert _normalize_env_ids([1], 3) == [1]
    assert _normalize_env_ids(slice(None), 3) == [0, 1, 2]


def test_texture_selection_modes():
    assert sorted(
        _select_texture_indices("without_replacement", [0, 1, 2], 3, None)
    ) == [0, 1, 2]
    assert _select_texture_indices("fixed", [3, 1], 4, {1: 2, 3: 0}) == [0, 2]
    with pytest.raises(ValueError, match="without_replacement"):
        _select_texture_indices("without_replacement", [0, 1], 1, None)


def test_partial_reset_targets_only_selected_environment_ids():
    assert _normalize_env_ids(torch.tensor([1, 3]), 4) == [1, 3]


def test_fixed_assignment_maps_global_environment_ids():
    assert _select_texture_indices("fixed", [1, 3], 4, {1: 2, 3: 0}) == [2, 0]


def test_per_instance_selection_is_reused_for_all_links():
    selected = _select_texture_indices("fixed", [2], 4, {2: 3})
    assert selected == [3]
    assert selected[0] == selected[0]


def test_generated_color_branch_sets_texture_without_unbound_error():
    functor = object.__new__(randomize_visual_material)
    functor.textures = []
    class Instance:
        texture = None

        def set_base_color_texture(self, texture_data=None, **kwargs):
            self.texture = texture_data

    instance = Instance()
    functor._randomize_mat_inst(instance, {}, random_texture_prob=1.0)
    assert instance.texture is not None
