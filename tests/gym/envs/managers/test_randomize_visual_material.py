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

"""Characterization tests for randomize_visual_material."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_visual_material,
)
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.sim.objects.rigid_object import RigidObject


@pytest.fixture(autouse=True)
def _patch_get_data_path(monkeypatch):
    """Make texture-less configs skip get_data_path(None) in these tests."""

    def _get_data_path(path):
        return None if path is None else path

    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.randomization.visual.get_data_path",
        _get_data_path,
    )


class _MockRigidObject(RigidObject):
    """RigidObject that skips the heavy __init__; methods mocked for tests."""

    def __init__(self, uid="obj", num_envs=2):
        self.uid = uid
        # NOTE: do not set self.num_instances - it is a read-only BatchEntity
        # property that returns len(self._entities).
        self._all_indices = list(range(num_envs))
        self._entities = [MagicMock(name=f"mesh{i}") for i in range(num_envs)]
        self._visual_material = [None] * num_envs
        self.is_shared_visual_material = False
        self.set_visual_material = MagicMock()
        self.get_visual_material_inst = MagicMock(
            return_value=[MagicMock(name=f"inst{i}") for i in range(num_envs)]
        )
        self.get_existing_visual_material = MagicMock(return_value=[])
        self.apply_render_material_inst = MagicMock()


class _MockSim:
    def __init__(self, num_envs=2):
        self.textures = {}
        self._visual_materials = {}
        self.created_visual_materials = []
        self.env = MagicMock(name="dexsim_env")
        self.env.create_color_texture = MagicMock(
            return_value=MagicMock(name="Texture")
        )
        self.env.clean_materials = MagicMock()
        self.asset_uids = ["obj"]
        self._asset = _MockRigidObject(num_envs=num_envs)

    def get_texture_cache(self, key=None):
        if key is None:
            return self.textures
        return self.textures.get(key)

    def set_texture_cache(self, key, value):
        self.textures[key] = value

    def create_visual_material(self, cfg):
        self.created_visual_materials.append(cfg.uid)
        mat = MagicMock(name="VisualMaterial")
        inst = MagicMock(name="VisualMaterialInst")
        mat.create_instance.return_value = inst
        inst.mat = MagicMock(name="MaterialInst")
        self._visual_materials[cfg.uid] = mat
        return mat

    def get_visual_material(self, uid):
        m = MagicMock(name="plane_VisualMaterial")
        m.get_default_instance.return_value = MagicMock(name="plane_inst")
        return m

    def get_asset(self, uid):
        return self._asset

    def get_env(self):
        return self.env


class _MockEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.sim = _MockSim(num_envs=num_envs)


def _make_cfg(params):
    cfg = FunctorCfg(func=randomize_visual_material)
    cfg.params = params
    return cfg


def test_legacy_init_creates_visual_material():
    env = _MockEnv()
    entity_cfg = SceneEntityCfg(uid="obj")
    cfg = _make_cfg({"entity_cfg": entity_cfg, "fallback_to_new": True})
    functor = randomize_visual_material(cfg, env)
    assert env.sim.created_visual_materials  # legacy path creates a material


def test_legacy_call_runs_clean_materials():
    env = _MockEnv()
    entity_cfg = SceneEntityCfg(uid="obj")
    cfg = _make_cfg({"entity_cfg": entity_cfg, "fallback_to_new": True})
    functor = randomize_visual_material(cfg, env)
    env.sim.env.clean_materials = MagicMock()
    functor(env, torch.arange(env.num_envs), entity_cfg=entity_cfg)
    env.sim.env.clean_materials.assert_called_once()


def _seg(mesh_id, orig, tmpl):
    """Build a ReuseSegmentState whose working_inst is a fully-mocked VisualMaterialInst."""
    from embodichain.lab.sim.material import ReuseSegmentState, VisualMaterialInst

    inst = VisualMaterialInst.__new__(VisualMaterialInst)
    inst.uid = f"w_{mesh_id}"
    tmpl.get_inst = MagicMock(return_value=MagicMock(name="working_mat_inst"))
    inst._mat = tmpl
    inst.base_color_texture = None
    inst.base_color = [1, 1, 1, 1]
    inst.metallic = 0.0
    inst.roughness = 0.7
    inst.ior = 1.5
    inst.emissive = [0, 0, 0]
    inst.set_base_color = MagicMock()
    inst.set_metallic = MagicMock()
    inst.set_roughness = MagicMock()
    inst.set_ior = MagicMock()
    inst.set_base_color_texture = MagicMock()
    return ReuseSegmentState(mesh_id=mesh_id, original_inst=orig, working_inst=inst)


def test_new_init_does_not_create_visual_material():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")  # cached _MockRigidObject
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[[seg]])
    cfg = _make_cfg(
        {"entity_cfg": SceneEntityCfg(uid="obj")}
    )  # fallback_to_new defaults False

    functor = randomize_visual_material(cfg, env)

    assert env.sim.created_visual_materials == []  # no new material
    assert functor._new_mode is True


def test_new_init_degrades_to_legacy_on_failure():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(side_effect=ValueError("no material"))
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})

    functor = randomize_visual_material(cfg, env)

    assert functor._new_mode is False
    assert env.sim.created_visual_materials  # degraded to legacy


def test_new_call_no_clean_and_swaps():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[[seg]])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)

    env.sim.env.clean_materials.reset_mock()
    # force original tier (p_original=1)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(env, torch.arange(env.num_envs), entity_cfg=SceneEntityCfg(uid="obj"))

    env.sim.env.clean_materials.assert_not_called()
    obj.apply_render_material_inst.assert_called()


def _art_asset(uid="art"):
    from embodichain.lab.sim.objects.articulation import Articulation

    class _A(Articulation):
        def __init__(self):
            self._entities = []
            self._all_indices = [0, 1]
            self.uid = uid
            # num_instances is a read-only property; not set here.
            self.is_shared_visual_material = False
            self.link_names = ["link0"]

    obj = _A()
    seg = _seg(0, MagicMock(name="orig"), MagicMock(name="tmpl"))
    obj.get_existing_visual_material = MagicMock(return_value=[{"link0": [seg]}])
    obj.apply_render_material_inst = MagicMock()
    return obj


def test_new_init_articulation_reuse():
    env = _MockEnv()
    env.sim.asset_uids = ["art"]
    env.sim.get_asset = lambda uid: _art_asset()
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="art", link_names=["link0"])})

    functor = randomize_visual_material(cfg, env)

    assert functor._new_mode is True
    assert env.sim.created_visual_materials == []


def test_new_call_articulation_swaps_per_link():
    env = _MockEnv()
    art = _art_asset()
    env.sim.get_asset = lambda uid: art
    env.sim.asset_uids = ["art"]
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="art", link_names=["link0"])})
    functor = randomize_visual_material(cfg, env)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(
        env,
        torch.arange(env.num_envs),
        entity_cfg=SceneEntityCfg(uid="art", link_names=["link0"]),
    )

    art.apply_render_material_inst.assert_called()


def test_plane_new_mode_uses_legacy_inplace_no_clean():
    env = _MockEnv()
    env.sim.asset_uids = ["default_plane"]
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="default_plane")})

    functor = randomize_visual_material(cfg, env)
    assert functor._new_mode is False  # plane never uses swap path

    env.sim.env.clean_materials.reset_mock()
    functor(
        env,
        torch.arange(env.num_envs),
        entity_cfg=SceneEntityCfg(uid="default_plane"),
    )
    env.sim.env.clean_materials.assert_not_called()


def test_tier_probs_backward_compat_derivation():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(
        return_value=[[_seg(0, MagicMock(), MagicMock())]]
    )
    cfg = _make_cfg(
        {"entity_cfg": SceneEntityCfg(uid="obj"), "random_texture_prob": 0.3}
    )
    functor = randomize_visual_material(cfg, env)

    # With a non-empty library, backward-compat derivation gives p_library=0.3, p_solid=0.7.
    functor._library_textures = [MagicMock(name="Texture")]
    functor._resolve_tier_probs()

    assert functor._p_original == 0.0
    assert pytest.approx(functor._p_library) == 0.3
    assert pytest.approx(functor._p_solid) == 0.7


def test_tier_probs_explicit_normalize():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(
        return_value=[[_seg(0, MagicMock(), MagicMock())]]
    )
    cfg = _make_cfg(
        {
            "entity_cfg": SceneEntityCfg(uid="obj"),
            "p_original": 1.0,
            "p_library": 1.0,
            "p_solid": 2.0,
        }
    )
    functor = randomize_visual_material(cfg, env)
    functor._library_textures = [MagicMock(name="Texture")]
    functor._resolve_tier_probs()

    assert pytest.approx(functor._p_original) == 0.25
    assert pytest.approx(functor._p_library) == 0.25
    assert pytest.approx(functor._p_solid) == 0.5


def test_empty_library_folds_into_solid():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(
        return_value=[[_seg(0, MagicMock(), MagicMock())]]
    )
    cfg = _make_cfg(
        {
            "entity_cfg": SceneEntityCfg(uid="obj"),
            "p_original": 0.0,
            "p_library": 0.5,
            "p_solid": 0.5,
        }
    )  # no texture_path -> empty library

    functor = randomize_visual_material(cfg, env)

    assert functor._p_library == 0.0
    assert pytest.approx(functor._p_solid) == 1.0


def test_library_textures_cached_across_functors():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(
        return_value=[[_seg(0, MagicMock(), MagicMock())]]
    )
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)

    # Simulate a non-empty texture library and run _build_library_textures directly.
    fake_tex = torch.zeros((2, 2, 4), dtype=torch.uint8)
    functor.textures = [fake_tex]
    functor._texture_key = "texA"
    functor._build_library_textures(env)
    assert env.sim.get_env().create_color_texture.call_count == 1

    # A second functor with the same key reuses the cached Textures (no new upload).
    functor2 = randomize_visual_material(cfg, env)
    functor2.textures = [fake_tex]
    functor2._texture_key = "texA"
    functor2._build_library_textures(env)
    assert env.sim.get_env().create_color_texture.call_count == 1


def test_fallback_to_new_preserves_legacy():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(
        return_value=[[_seg(0, MagicMock(), MagicMock())]]
    )
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj"), "fallback_to_new": True})

    functor = randomize_visual_material(cfg, env)
    assert functor._new_mode is False
    assert env.sim.created_visual_materials  # legacy created a material

    env.sim.env.clean_materials.reset_mock()
    functor(env, torch.arange(env.num_envs), entity_cfg=SceneEntityCfg(uid="obj"))
    env.sim.env.clean_materials.assert_called_once()  # legacy cleans


def test_multi_segment_all_swapped():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    segs = [_seg(0, MagicMock(), MagicMock()), _seg(1, MagicMock(), MagicMock())]
    obj.get_existing_visual_material = MagicMock(return_value=[segs])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})
    functor = randomize_visual_material(cfg, env)
    functor._p_original, functor._p_library, functor._p_solid = 1.0, 0.0, 0.0

    functor(env, torch.arange(env.num_envs), entity_cfg=SceneEntityCfg(uid="obj"))

    # two segments -> two apply calls; mesh_id is the 3rd positional arg
    mesh_ids = {call.args[2] for call in obj.apply_render_material_inst.call_args_list}
    assert mesh_ids == {0, 1}


def test_new_call_empty_reuse_state_no_crash():
    env = _MockEnv()
    obj = env.sim.get_asset("obj")
    obj.get_existing_visual_material = MagicMock(return_value=[])
    cfg = _make_cfg({"entity_cfg": SceneEntityCfg(uid="obj")})

    functor = randomize_visual_material(cfg, env)

    assert functor._new_mode is True
    # Non-shared mode with zero existing instances used to crash in _call_reuse
    # because torch.multinomial cannot sample zero samples.
    functor(env, torch.arange(env.num_envs), entity_cfg=SceneEntityCfg(uid="obj"))
