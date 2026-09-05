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

pytest.importorskip("dexsim")
pytest.importorskip("open3d")

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.sim_utils import load_mesh_objects_from_cfg

_MESH_PATH = "mesh.obj"
_CONVEX_HULL_COUNT = 2


class _FakeMeshObject:
    def set_name(self, name: str) -> None:
        self.name = name


class _FakeArena:
    def __init__(self) -> None:
        self.acd_kwargs: dict[str, object] | None = None

    def load_actor_with_acd(self, _fpath: str, **kwargs: object) -> _FakeMeshObject:
        self.acd_kwargs = kwargs
        return _FakeMeshObject()


def _load_mesh_with_method(acd_method: str | None = None) -> _FakeArena:
    arena = _FakeArena()
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath=_MESH_PATH,
            max_convex_hull_num=_CONVEX_HULL_COUNT,
        ),
    )
    if acd_method is not None:
        cfg.acd_method = acd_method

    load_mesh_objects_from_cfg(cfg, [arena])
    return arena


def test_mesh_cfg_default_acd_method_is_forwarded_to_dexsim() -> None:
    arena = _load_mesh_with_method()

    assert arena.acd_kwargs is not None
    assert arena.acd_kwargs["method"] == "visacd"


def test_legacy_rigid_object_acd_method_overrides_mesh_default() -> None:
    arena = _load_mesh_with_method("coacd")

    assert arena.acd_kwargs is not None
    assert arena.acd_kwargs["method"] == "coacd"
