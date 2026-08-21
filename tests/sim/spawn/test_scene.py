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

import pytest

from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.spawn.scene import SpawnScene

pytestmark = pytest.mark.no_sim


def _make_scene(handles: dict[str, object]) -> SpawnScene:
    scene = object.__new__(SpawnScene)
    scene.builder = SimpleNamespace(
        is_finalized=True,
        result=SimpleNamespace(handles=handles),
    )
    scene._assets = {}
    return scene


class _RetryableFacade:
    def __init__(self, *, fail_first: bool = False) -> None:
        self._entities: list[object] = []
        self.is_declared = True
        self.fail_first = fail_first
        self.bind_attempts = 0

    def attach_spawn_handles(self, entities: tuple[object, ...]) -> None:
        self._entities = list(entities)

    def bind_spawn(self, _result: object) -> None:
        self.bind_attempts += 1
        if self.fail_first and self.bind_attempts == 1:
            raise RuntimeError("bind failed")
        self.is_declared = False


def test_bind_retries_only_incomplete_declarations() -> None:
    first_handle = object()
    second_handle = object()
    scene = _make_scene({"first": first_handle, "second": second_handle})
    first = _RetryableFacade()
    second = _RetryableFacade(fail_first=True)

    scene.track(
        "rigid_object",
        "first",
        SimpleNamespace(name="first", per_env=False),
        facade=first,
    )
    scene.track(
        "rigid_object",
        "second",
        SimpleNamespace(name="second", per_env=False),
        facade=second,
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        scene.bind()
    scene.bind()
    scene.bind()

    assert first._entities == [first_handle]
    assert second._entities == [second_handle]
    assert first.bind_attempts == 1
    assert second.bind_attempts == 2


class _RetryableArticulation(Articulation):
    bind_attempts = 0
    reset_attempts = 0

    def __init__(
        self,
        cfg: object,
        entities: list[object] | None = None,
        device: object = "cpu",
        *,
        spawn_result: object | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.uid = cfg.uid
        self.device = device
        self._entities = [] if entities is None else entities
        self._spawn_result = spawn_result
        self._world = None if spawn_result is None else object()
        self._declared_num_instances = (
            len(entities) if entities is not None else int(declared_num_instances or 0)
        )

    def attach_spawn_handles(self, entities: list[object]) -> None:
        self._entities = list(entities)

    def _apply_spawn_config(self) -> None:
        type(self).bind_attempts += 1
        if type(self).bind_attempts == 1:
            raise RuntimeError("configuration failed")

    def reset(self, env_ids: object | None = None) -> None:
        del env_ids
        type(self).reset_attempts += 1


def test_articulation_binding_is_atomic_and_retryable() -> None:
    _RetryableArticulation.bind_attempts = 0
    _RetryableArticulation.reset_attempts = 0
    facade = _RetryableArticulation(
        SimpleNamespace(uid="robot"),
        declared_num_instances=1,
    )
    result = object()
    handles = [object()]
    facade.attach_spawn_handles(handles)

    with pytest.raises(RuntimeError, match="configuration failed"):
        facade.bind_spawn(result)

    assert facade.is_declared
    assert _RetryableArticulation.reset_attempts == 0
    facade.bind_spawn(result)
    assert facade.is_spawn_bound
    assert facade._entities == handles
    assert _RetryableArticulation.reset_attempts == 1
