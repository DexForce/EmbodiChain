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
