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
"""Unit tests for the Newton physics backend finalize/invalidate lifecycle.

These tests exercise :class:`NewtonPhysicsBackend` in isolation (no GPU and no
live dexsim world required) by injecting a fake Newton manager and patching the
``ensure_simulation_prepared_lazy`` rebuild entry point. They verify the
backend owns the dirty/finalize state machine that used to live inline in
:class:`SimulationManager`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from embodichain.lab.sim.physics import NewtonPhysicsBackend


class _Resettable:
    """Stand-in for a RigidObject/Articulation with a reset() call counter."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeNewtonManager:
    """Stand-in for dexsim's NewtonManager exposing only the lifecycle state."""

    def __init__(self) -> None:
        self.lifecycle_state = SimpleNamespace(name="BUILDER")


def _make_backend() -> tuple[
    NewtonPhysicsBackend,
    _FakeNewtonManager,
    _Resettable,
    _Resettable,
    _Resettable,
    _Resettable,
]:
    rigid_obj = _Resettable()
    rigid_group = _Resettable()  # groups must NOT be reset by the Newton backend.
    articulation = _Resettable()
    robot = _Resettable()  # a robot is an articulation and is reset like one.
    newton_mgr = _FakeNewtonManager()

    # Minimal owning-SimulationManager stand-in: only the attributes the backend
    # touches during finalize / reset are needed.
    manager = SimpleNamespace(
        _world=object(),
        _rigid_objects={"rigid": rigid_obj},
        _rigid_object_groups={"rigid_group": rigid_group},
        _articulations={"art": articulation},
        _robots={"robot": robot},
    )

    backend = NewtonPhysicsBackend(manager)
    # Inject the fake manager so finalize() does not call get_newton_manager.
    backend._newton_manager = newton_mgr
    return backend, newton_mgr, rigid_obj, rigid_group, articulation, robot


def _fake_ensure_prepared_lazy(mgr, world, *, rebuild_from_scene, warn):
    """Mimic the real rebuild: bring the Newton model to the READY state."""
    mgr.lifecycle_state.name = "READY"
    return True, None


@patch(
    "dexsim.engine.newton_physics.rebuild.ensure_simulation_prepared_lazy",
    new=_fake_ensure_prepared_lazy,
)
def test_finalize_resets_entities_after_ready() -> None:
    (
        backend,
        newton_mgr,
        rigid_obj,
        rigid_group,
        articulation,
        robot,
    ) = _make_backend()

    assert not backend.is_initialized
    backend.prepare()

    assert newton_mgr.lifecycle_state.name == "READY"
    assert backend.is_initialized
    assert rigid_obj.reset_calls == 1
    assert articulation.reset_calls == 1
    assert robot.reset_calls == 1
    # Rigid object groups are not supported on the Newton backend: not reset.
    assert rigid_group.reset_calls == 0


@patch(
    "dexsim.engine.newton_physics.rebuild.ensure_simulation_prepared_lazy",
    new=_fake_ensure_prepared_lazy,
)
def test_finalize_does_not_repeat_deferred_reset() -> None:
    (
        backend,
        _newton_mgr,
        rigid_obj,
        _rigid_group,
        articulation,
        robot,
    ) = _make_backend()

    backend.prepare()
    backend.prepare()

    assert rigid_obj.reset_calls == 1
    assert articulation.reset_calls == 1
    assert robot.reset_calls == 1


@patch(
    "dexsim.engine.newton_physics.rebuild.ensure_simulation_prepared_lazy",
    new=_fake_ensure_prepared_lazy,
)
def test_invalidation_allows_next_finalize_to_reset_again() -> None:
    (
        backend,
        _newton_mgr,
        rigid_obj,
        _rigid_group,
        articulation,
        robot,
    ) = _make_backend()

    backend.prepare()
    backend.invalidate()
    assert not backend.is_initialized
    backend.prepare()

    assert rigid_obj.reset_calls == 2
    assert articulation.reset_calls == 2
    assert robot.reset_calls == 2


@patch(
    "dexsim.engine.newton_physics.rebuild.ensure_simulation_prepared_lazy",
    new=_fake_ensure_prepared_lazy,
)
def test_finalize_raises_when_rebuild_unsafe() -> None:
    backend, _newton_mgr, rigid_obj, _rigid_group, _articulation, _robot = (
        _make_backend()
    )

    # An unsafe rebuild makes finalize() raise (logger.log_error raises by
    # default). It must not mark itself initialized nor reset entities.
    with patch(
        "dexsim.engine.newton_physics.rebuild.ensure_simulation_prepared_lazy",
        new=lambda mgr, world, *, rebuild_from_scene, warn: (False, None),
    ):
        try:
            backend.prepare()
        except RuntimeError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("finalize() should raise on an unsafe rebuild")

    assert not backend.is_initialized
    assert rigid_obj.reset_calls == 0


def test_invalidate_is_idempotent_and_only_clears_finalized_flag() -> None:
    (
        backend,
        _newton_mgr,
        _rigid_obj,
        _rigid_group,
        _articulation,
        _robot,
    ) = _make_backend()
    backend._is_finalized = True

    backend.invalidate()
    backend.invalidate()

    assert not backend.is_initialized
