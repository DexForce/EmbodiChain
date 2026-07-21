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

from embodichain.lab.sim.objects.gizmo import Gizmo


class FakeGizmoNode:
    """Node stub that records callback and parent cleanup."""

    def __init__(self) -> None:
        self.callback = object()
        self.detached = False

    def set_flush_transform_callback(self, callback: object | None) -> None:
        self.callback = callback

    def detach_parent(self) -> None:
        self.detached = True


class FakeDexSimGizmo:
    """DexSim gizmo stub with a detachable scene node."""

    def __init__(self) -> None:
        self.node = FakeGizmoNode()


class FakeArena:
    """Arena stub that records explicit gizmo removal."""

    def __init__(self) -> None:
        self.removed_gizmo: object | None = None

    def remove_gizmo(self, gizmo: object) -> None:
        self.removed_gizmo = gizmo


def test_destroy_removes_gizmo_from_arena() -> None:
    """Destroy must release DexSim's arena-owned gizmo reference."""
    arena = FakeArena()
    dexsim_gizmo = FakeDexSimGizmo()
    gizmo = object.__new__(Gizmo)
    gizmo._env = arena
    gizmo._gizmo = dexsim_gizmo
    gizmo._target_type = "rigidobject"
    gizmo.target = object()

    gizmo.destroy()

    assert arena.removed_gizmo is dexsim_gizmo
    assert dexsim_gizmo.node.detached is True
    assert dexsim_gizmo.node.callback is None
    assert gizmo._gizmo is None
