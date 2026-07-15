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

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.sim_utils import _load_rigid_mesh_prototype


class _FakeActor:
    def add_rigidbody(self, *args, **kwargs) -> None:
        pass


class _FakeArena:
    def __init__(self) -> None:
        self.load_actor_called = False
        self.acd_method: str | None = None

    def load_actor(self, *args, **kwargs) -> _FakeActor:
        self.load_actor_called = True
        return _FakeActor()

    def load_actor_with_acd(self, *args, method: str, **kwargs) -> _FakeActor:
        self.acd_method = method
        return _FakeActor()


def test_load_rigid_mesh_uses_shape_collision_defaults() -> None:
    arena = _FakeArena()
    cfg = RigidObjectCfg(uid="mesh", shape=MeshCfg(fpath="mesh.obj"))

    _load_rigid_mesh_prototype(
        arena,
        cfg,
        cache_dir=None,
        body_type=None,
        is_newton_backend=False,
    )

    assert arena.load_actor_called


def test_load_rigid_mesh_forwards_shape_acd_method() -> None:
    arena = _FakeArena()
    cfg = RigidObjectCfg(
        uid="mesh",
        shape=MeshCfg(
            fpath="mesh.obj",
            max_convex_hull_num=2,
            acd_method="vhacd",
        ),
    )

    _load_rigid_mesh_prototype(
        arena,
        cfg,
        cache_dir=None,
        body_type=None,
        is_newton_backend=False,
    )

    assert arena.acd_method == "vhacd"
