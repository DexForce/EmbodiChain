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
import embodichain.lab.gym.envs.managers.randomization.visual as visual

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


def test_partial_reset_targets_only_selected_environment_ids(monkeypatch):
    class Obj:
        is_shared_visual_material = False

        def __init__(self):
            self.mats = [FakeMat() for _ in range(4)]

        def get_visual_material_inst(self, env_ids=None, **kwargs):
            return [self.mats[i] for i in env_ids]

    functor = object.__new__(randomize_visual_material)
    functor.entity_cfg = type("C", (), {"uid": "x", "link_names": None})()
    functor.entity = Obj()
    functor.textures = []
    monkeypatch.setattr(visual, "RigidObject", Obj)

    def mark(*, mat_inst, **kwargs):
        mat_inst.set_base_color([1, 1, 1, 1])

    functor._randomize_mat_inst = mark
    env = type("E", (), {"num_envs": 4})()
    functor.__call__(
        env, torch.tensor([1, 3]), functor.entity_cfg, random_texture_prob=0
    )
    assert functor.entity.mats[0].color is None and functor.entity.mats[2].color is None
    assert (
        functor.entity.mats[1].color is not None
        and functor.entity.mats[3].color is not None
    )


def test_fixed_assignment_maps_global_environment_ids(monkeypatch):
    class Obj:
        is_shared_visual_material = False

        def get_visual_material_inst(self, env_ids=None, **kwargs):
            return [{"l": FakeMat()} for _ in env_ids]

    f = object.__new__(randomize_visual_material)
    f.entity_cfg = type("C", (), {"uid": "x", "link_names": None})()
    f.entity = Obj()
    f.textures = ["a", "b", "c"]
    monkeypatch.setattr(visual, "RigidObject", Obj)
    seen = []

    def record(*, texture_idx, **kwargs):
        seen.append(texture_idx)

    f._randomize_mat_inst = record
    f.__call__(
        type("E", (), {"num_envs": 4})(),
        torch.tensor([1, 3]),
        f.entity_cfg,
        texture_sampling="fixed",
        texture_indices={1: 2, 3: 0},
    )
    assert seen == [2, 0]


def test_per_instance_selection_is_reused_for_all_links(monkeypatch):
    class Obj:
        is_shared_visual_material = False

        def get_visual_material_inst(self, env_ids=None, **kwargs):
            return [{"a": FakeMat(), "b": FakeMat()} for _ in env_ids]

    f = object.__new__(randomize_visual_material)
    f.entity_cfg = type("C", (), {"uid": "x", "link_names": None})()
    f.entity = Obj()
    f.textures = ["a", "b"]
    monkeypatch.setattr(visual, "RigidObject", type("Other", (), {}))
    monkeypatch.setattr(visual, "Articulation", Obj)
    seen = []

    def record(*, texture_idx, **kwargs):
        seen.append(texture_idx)

    f._randomize_mat_inst = record
    f.__call__(
        type("E", (), {"num_envs": 4})(),
        torch.tensor([1, 3]),
        f.entity_cfg,
        texture_sampling="fixed",
        texture_indices={1: 1, 3: 0},
        texture_scope="per_instance",
    )
    assert seen == [1, 1, 0, 0]


class FakeMat:
    def __init__(self):
        self.color = None

    def set_base_color(self, value):
        self.color = value


def test_generated_color_branch_sets_texture_without_unbound_error():
    functor = object.__new__(randomize_visual_material)
    texture = object()
    functor.textures = [texture]

    class Instance:
        texture = None

        def set_base_color_texture(self, texture_data=None, **kwargs):
            self.texture = texture_data

    instance = Instance()
    functor._randomize_mat_inst(instance, {}, random_texture_prob=1.0)
    assert instance.texture is texture


def test_texture_references_are_created_once(monkeypatch):
    functor = object.__new__(randomize_visual_material)
    functor.textures = [torch.zeros((2, 2, 4), dtype=torch.uint8)]
    functor.texture_refs = [object()]
    calls = []

    class Instance:
        def set_base_color_texture(self, *, texture_ref=None, **kwargs):
            calls.append(texture_ref)

    functor._randomize_mat_inst(Instance(), {}, random_texture_prob=1.0, texture_idx=0)
    functor._randomize_mat_inst(Instance(), {}, random_texture_prob=1.0, texture_idx=0)
    assert calls == [functor.texture_refs[0], functor.texture_refs[0]]


@pytest.mark.skip(reason="Requires renderer-backed four-environment fixture")
def test_four_environment_visual_texture_integration():
    pass
