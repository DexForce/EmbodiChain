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

from embodichain.lab.sim.sim_manager import SimulationManager


class _Resettable:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _NewtonManager:
    def __init__(self) -> None:
        self.lifecycle_state = SimpleNamespace(name="BUILDER")
        self.start_calls = 0

    def start_simulation(self) -> None:
        self.start_calls += 1
        self.lifecycle_state.name = "READY"


def _make_newton_sim() -> (
    tuple[SimulationManager, _NewtonManager, _Resettable, _Resettable]
):
    sim = object.__new__(SimulationManager)
    rigid_obj = _Resettable()
    rigid_obj_group = _Resettable()
    manager = _NewtonManager()

    sim._physics_backend = "newton"
    sim._newton_manager = manager
    sim._is_finalized_newton_physics = False
    sim._is_initialized_gpu_physics = False
    sim._has_reset_newton_entities_after_finalize = False
    sim._rigid_objects = {"rigid": rigid_obj}
    sim._rigid_object_groups = {"rigid_group": rigid_obj_group}

    return sim, manager, rigid_obj, rigid_obj_group


def test_finalize_newton_physics_resets_entities_after_ready() -> None:
    sim, manager, rigid_obj, rigid_obj_group = _make_newton_sim()

    sim.finalize_newton_physics()

    assert manager.start_calls == 1
    assert rigid_obj.reset_calls == 1
    assert rigid_obj_group.reset_calls == 0
    assert sim._is_finalized_newton_physics
    assert sim._is_initialized_gpu_physics


def test_finalize_newton_physics_does_not_repeat_deferred_reset() -> None:
    sim, manager, rigid_obj, rigid_obj_group = _make_newton_sim()

    sim.finalize_newton_physics()
    sim.finalize_newton_physics()

    assert manager.start_calls == 1
    assert rigid_obj.reset_calls == 1
    assert rigid_obj_group.reset_calls == 0


def test_newton_invalidation_allows_next_finalize_to_reset_again() -> None:
    sim, manager, rigid_obj, rigid_obj_group = _make_newton_sim()

    sim.finalize_newton_physics()
    sim._invalidate_newton_physics()
    sim.finalize_newton_physics()

    assert manager.start_calls == 1
    assert rigid_obj.reset_calls == 2
    assert rigid_obj_group.reset_calls == 0
