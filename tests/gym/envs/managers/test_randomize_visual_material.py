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

"""Tests for visual-material randomization reuse and fallback behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    _select_texture_indices,
    randomize_visual_material,
)
from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst
from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.objects.rigid_object import RigidObject


@pytest.fixture(autouse=True)
def _patch_get_data_path(monkeypatch):
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.randomization.visual.get_data_path",
        lambda path: path,
    )


class _MockRigidObject(RigidObject):
    """Rigid object with only the state used by the functor."""

    def __init__(self, uid: str = "obj", num_envs: int = 1):
        self.uid = uid
        self._all_indices = list(range(num_envs))
        self._entities = [MagicMock(name=f"entity_{i}") for i in self._all_indices]
        self._visual_material = [None] * num_envs
        self.is_shared_visual_material = False
        self.set_visual_material = MagicMock()
        self.get_visual_material_inst = MagicMock(
            return_value=[MagicMock(name=f"inst_{i}") for i in self._all_indices]
        )
        self.get_existing_visual_material = MagicMock(return_value=[])
        self.apply_render_material_inst = MagicMock()


class _MockArticulation(Articulation):
    """Articulation with only the state used by the functor."""

    def __init__(self, link_names: list[str], num_envs: int = 1):
        self.uid = "art"
        self._entities = []
        self._all_indices = list(range(num_envs))
        self.is_shared_visual_material = False
        self.link_names = link_names
        self.apply_render_material_inst = MagicMock()


class _MockSim:
    def __init__(self, num_envs: int = 1):
        self.textures = {}
        self.created_visual_materials = []
        self.env = MagicMock(name="dexsim_env")
        self.env.create_color_texture.return_value = MagicMock(name="Texture")
        self.asset_uids = ["obj"]
        self._asset = _MockRigidObject(num_envs=num_envs)

    def get_texture_cache(self, key=None):
        return self.textures if key is None else self.textures.get(key)

    def set_texture_cache(self, key, value):
        self.textures[key] = value

    def create_visual_material(self, cfg):
        self.created_visual_materials.append(cfg.uid)
        return MagicMock(name="VisualMaterial")

    def get_visual_material(self, uid):
        material = MagicMock(name=f"{uid}_material")
        material.get_default_instance.return_value = MagicMock(name=f"{uid}_inst")
        return material

    def get_asset(self, uid):
        return self._asset

    def get_env(self):
        return self.env


class _MockEnv:
    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.sim = _MockSim(num_envs=num_envs)


def _make_cfg(params: dict | None = None, *, uid: str = "obj") -> FunctorCfg:
    cfg = FunctorCfg(func=randomize_visual_material)
    cfg.params = {"entity_cfg": SceneEntityCfg(uid=uid), **(params or {})}
    return cfg


def _segment(mesh_id: int = 0, original=None) -> ReuseSegmentState:
    working = MagicMock(spec=VisualMaterialInst, name=f"working_{mesh_id}")
    working.mat = MagicMock(name=f"working_mat_{mesh_id}")
    return ReuseSegmentState(
        mesh_id=mesh_id,
        original_inst=(
            original if original is not None else MagicMock(name=f"original_{mesh_id}")
        ),
        working_inst=working,
    )


def _make_rigid_functor(
    params: dict | None = None,
    *,
    num_envs: int = 1,
    states: list | None = None,
):
    env = _MockEnv(num_envs=num_envs)
    obj = env.sim.get_asset("obj")
    if states is None:
        states = [[_segment()] for _ in range(num_envs)]
    obj.get_existing_visual_material.return_value = states
    return env, obj, randomize_visual_material(_make_cfg(params), env)


def _make_articulation_functor(link_names: list[str]):
    env = _MockEnv()
    art = _MockArticulation(link_names)
    link_map = {link_name: [_segment()] for link_name in link_names}
    art.get_existing_visual_material = MagicMock(return_value=[link_map])
    env.sim.asset_uids = ["art"]
    env.sim._asset = art
    cfg = _make_cfg(uid="art")
    cfg.params["entity_cfg"].link_names = link_names
    return env, art, randomize_visual_material(cfg, env)


def _force_tier(functor, tier: int) -> None:
    probabilities = [0.0, 0.0, 0.0]
    probabilities[tier] = 1.0
    functor._p_original, functor._p_library, functor._p_solid = probabilities


def _run(functor, env, env_ids: torch.Tensor | None = None) -> None:
    env_ids = torch.arange(env.num_envs) if env_ids is None else env_ids
    functor(env, env_ids, entity_cfg=functor.entity_cfg)


def test_texture_selection_modes():
    assert _select_texture_indices("cycle", [3, 1, 2], 2, None) == [0, 1, 0]
    assert _select_texture_indices("fixed", [3, 1], 4, {"1": 2, "3": 0}) == [0, 2]
    assert sorted(
        _select_texture_indices("without_replacement", [0, 1, 2], 3, None)
    ) == [0, 1, 2]

    with pytest.raises(ValueError, match="at least one texture per target"):
        _select_texture_indices("without_replacement", [0, 1], 1, None)


def test_fixed_texture_selection_uses_global_environment_ids():
    states = [[_segment(env_id)] for env_id in range(4)]
    env, _, functor = _make_rigid_functor(num_envs=4, states=states)
    textures = [MagicMock(name=f"texture_{i}") for i in range(3)]
    functor._library_textures = textures
    _force_tier(functor, tier=1)

    functor(
        env,
        torch.tensor([1, 3]),
        entity_cfg=functor.entity_cfg,
        texture_sampling="fixed",
        texture_indices={1: 2, 3: 0},
    )

    assert (
        states[1][0].working_inst.set_base_color_texture.call_args.kwargs["texture_obj"]
        is textures[2]
    )
    assert (
        states[3][0].working_inst.set_base_color_texture.call_args.kwargs["texture_obj"]
        is textures[0]
    )


def test_fixed_texture_selection_is_reused_for_articulation_links():
    env, art, functor = _make_articulation_functor(["left", "right"])
    textures = [MagicMock(name=f"texture_{i}") for i in range(2)]
    functor._library_textures = textures
    _force_tier(functor, tier=1)

    functor(
        env,
        torch.tensor([0]),
        entity_cfg=functor.entity_cfg,
        texture_sampling="fixed",
        texture_indices={0: 1},
    )

    link_map = art.get_existing_visual_material.return_value[0]
    for segments in link_map.values():
        assert (
            segments[0].working_inst.set_base_color_texture.call_args.kwargs[
                "texture_obj"
            ]
            is textures[1]
        )


def test_fallback_to_new_preserves_legacy_path():
    env = _MockEnv(num_envs=2)
    obj = env.sim.get_asset("obj")
    functor = randomize_visual_material(_make_cfg({"fallback_to_new": True}), env)
    obj.set_visual_material.reset_mock()

    _run(functor, env)

    assert functor._new_mode is False
    assert env.sim.created_visual_materials == ["obj_random_mat"]
    obj.set_visual_material.assert_called_once()
    env.sim.env.clean_materials.assert_called_once()


def test_reuse_init_does_not_create_visual_material():
    env, _, functor = _make_rigid_functor()

    assert functor._new_mode is True
    assert env.sim.created_visual_materials == []


def test_reuse_init_degrades_to_legacy_on_failure():
    env = _MockEnv()
    env.sim.get_asset("obj").get_existing_visual_material.side_effect = ValueError(
        "no material"
    )

    functor = randomize_visual_material(_make_cfg(), env)

    assert functor._new_mode is False
    assert env.sim.created_visual_materials == ["obj_random_mat"]


def test_reuse_call_reattaches_without_cleaning():
    env, obj, functor = _make_rigid_functor()
    _force_tier(functor, tier=2)
    env.sim.env.clean_materials.reset_mock()

    _run(functor, env)
    _run(functor, env)

    assert obj.apply_render_material_inst.call_count == 2
    env.sim.env.clean_materials.assert_not_called()


def test_reuse_call_supports_partial_environment_selection():
    states = [[_segment(0)], [_segment(1)]]
    env, obj, functor = _make_rigid_functor(num_envs=2, states=states)
    _force_tier(functor, tier=2)

    _run(functor, env, torch.tensor([1]))

    call_args = obj.apply_render_material_inst.call_args.args
    assert (call_args[0], call_args[2]) == (1, 1)


def test_articulation_reuses_and_swaps_link_material():
    env, art, functor = _make_articulation_functor(["link"])
    _force_tier(functor, tier=2)

    _run(functor, env)

    assert functor._new_mode is True
    assert env.sim.created_visual_materials == []
    art.apply_render_material_inst.assert_called_once()


def test_articulation_samples_tier_per_link():
    env, art, functor = _make_articulation_functor(["library", "solid"])
    library_texture = MagicMock(name="library_texture")
    functor._library_textures = [library_texture]
    functor._sample_tiers = MagicMock(return_value=torch.tensor([1, 2]))

    _run(functor, env)

    link_map = art.get_existing_visual_material.return_value[0]
    library_call = link_map["library"][0].working_inst.set_base_color_texture.call_args
    solid_call = link_map["solid"][0].working_inst.set_base_color_texture.call_args
    assert library_call.kwargs["texture_obj"] is library_texture
    assert solid_call.kwargs["texture_obj"] in functor._solid_textures


def test_default_plane_randomizes_in_place_without_cleaning():
    env = _MockEnv()
    functor = randomize_visual_material(_make_cfg(uid="default_plane"), env)
    env.sim.env.clean_materials.reset_mock()

    _run(functor, env)

    assert functor._new_mode is False
    assert env.sim.created_visual_materials == []
    env.sim.env.clean_materials.assert_not_called()


@pytest.mark.parametrize(
    ("params", "has_library", "expected"),
    [
        ({"random_texture_prob": 0.3}, True, (0.0, 0.3, 0.7)),
        (
            {"p_original": 1.0, "p_library": 1.0, "p_solid": 2.0},
            True,
            (0.25, 0.25, 0.5),
        ),
        (
            {"p_original": 0.0, "p_library": 0.5, "p_solid": 0.5},
            False,
            (0.0, 0.0, 1.0),
        ),
    ],
    ids=["legacy-split", "normalized", "empty-library"],
)
def test_tier_probability_resolution(params, has_library, expected):
    _, _, functor = _make_rigid_functor(params)
    functor._library_textures = [MagicMock(name="Texture")] if has_library else []

    functor._resolve_tier_probs()

    actual = (functor._p_original, functor._p_library, functor._p_solid)
    assert actual == pytest.approx(expected)


def test_library_textures_are_cached_across_functors():
    env, _, functor = _make_rigid_functor()
    env.sim.env.create_color_texture.reset_mock()
    texture = torch.zeros((2, 2, 4), dtype=torch.uint8)
    functor.textures = [texture]
    functor._texture_key = "library"

    functor._build_library_textures(env)
    second = randomize_visual_material(_make_cfg(), env)
    second.textures = [texture]
    second._texture_key = "library"
    second._build_library_textures(env)

    env.sim.env.create_color_texture.assert_called_once()


def test_solid_randomization_reuses_bounded_palette_for_all_segments():
    segment_count = 3
    repeat_count = 3
    palette_size = 2
    segments = [_segment(mesh_id) for mesh_id in range(segment_count)]
    env, obj, functor = _make_rigid_functor(
        {"p_solid": 1.0, "solid_texture_count": palette_size}, states=[segments]
    )

    for _ in range(repeat_count):
        _run(functor, env)

    assert env.sim.env.create_color_texture.call_count == palette_size
    assert segments[0].working_inst.set_base_color.call_count == repeat_count
    assert all(
        segment.working_inst.set_base_color.call_count == 0 for segment in segments[1:]
    )
    mesh_ids = {call.args[2] for call in obj.apply_render_material_inst.call_args_list}
    assert mesh_ids == set(range(segment_count))


def test_solid_palette_stores_color_in_texture_pixels():
    fixed_color = [0.2, 0.4, 0.6]
    env, _, _ = _make_rigid_functor(
        {"base_color_range": [fixed_color, fixed_color], "solid_texture_count": 1}
    )

    texture_data = env.sim.env.create_color_texture.call_args.args[0]
    expected_rgba = torch.tensor([51, 102, 153, 255], dtype=torch.uint8).numpy()
    assert texture_data.flags.c_contiguous
    assert (texture_data == expected_rgba).all()


def test_library_tier_without_color_range_does_not_tint_texture():
    segment = _segment()
    env, _, functor = _make_rigid_functor(states=[[segment]])
    functor._library_textures = [MagicMock(name="library_texture")]
    _force_tier(functor, tier=1)

    _run(functor, env)

    segment.working_inst.set_base_color.assert_not_called()


def test_original_tier_restores_each_segment():
    segments = [_segment(mesh_id) for mesh_id in range(2)]
    env, obj, functor = _make_rigid_functor(states=[segments])
    _force_tier(functor, tier=2)
    _run(functor, env)
    obj.apply_render_material_inst.reset_mock()

    _force_tier(functor, tier=0)
    _run(functor, env)

    restored = {
        (call.args[1], call.args[2])
        for call in obj.apply_render_material_inst.call_args_list
    }
    assert restored == {
        (segment.original_inst, segment.mesh_id) for segment in segments
    }


def test_empty_reuse_state_is_a_noop():
    env, obj, functor = _make_rigid_functor(states=[])

    _run(functor, env)

    assert functor._new_mode is True
    obj.apply_render_material_inst.assert_not_called()
