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
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    refresh_articulation_contact_material,
)
from embodichain.lab.sim.objects.backends.articulation_physics import (
    _refresh_link_contact_material,
)


def _body(friction: float, fail: bool = False) -> SimpleNamespace:
    body = SimpleNamespace(friction=friction, writes=[])
    body.get_dynamic_friction = lambda: body.friction

    def write(value: float) -> None:
        body.friction = value
        body.writes.append(value)
        if fail and len(body.writes) == 1:
            raise RuntimeError("material update failed")

    body.set_dynamic_friction = write
    return body


@pytest.mark.parametrize("spawn_bound", [False, True])
def test_rebind_restores_original_friction(spawn_bound: bool) -> None:
    body = _body(0.73)
    native = SimpleNamespace(get_physical_body=Mock(return_value=body))
    entity = SimpleNamespace(_physics_binding=native) if spawn_bound else native

    _refresh_link_contact_material(entity, "handle", is_spawn_bound=spawn_bound)

    assert body.friction == 0.73
    assert body.writes == pytest.approx([0.731, 0.73])
    native.get_physical_body.assert_called_once_with("handle")


def test_rebind_restores_friction_after_failed_update() -> None:
    body = _body(0.5, fail=True)
    native = SimpleNamespace(get_physical_body=lambda name: body)
    with pytest.raises(RuntimeError, match="material update failed"):
        _refresh_link_contact_material(native, "handle", is_spawn_bound=False)
    assert body.friction == 0.5
    assert body.writes == pytest.approx([0.501, 0.5])


def test_rebind_rejects_missing_physical_body() -> None:
    native = SimpleNamespace(get_physical_body=lambda name: None)
    with pytest.raises(ValueError, match="no native physical body"):
        _refresh_link_contact_material(native, "handle", is_spawn_bound=False)


@pytest.mark.parametrize(
    "env_ids, selected",
    [(None, [0, 1, 2]), ([2], [2]), (torch.tensor([1, 0]), [1, 0]), ([], [])],
)
def test_event_refreshes_only_selected_links_and_rows(env_ids, selected) -> None:
    handles = [_body(0.5 + 0.1 * row) for row in range(3)]
    untouched = [_body(0.8) for _ in range(3)]
    entities = [
        SimpleNamespace(
            _physics_binding=SimpleNamespace(
                get_physical_body=lambda name, row=row: (
                    handles[row] if name == "handle" else untouched[row]
                )
            )
        )
        for row in range(3)
    ]
    articulation = SimpleNamespace(
        _entities=entities, is_spawn_bound=True, link_names=["handle", "cabinet"]
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cuda"),
        sim=SimpleNamespace(
            physics_backend="default", get_articulation=Mock(return_value=articulation)
        ),
    )

    refresh_articulation_contact_material(
        env, env_ids, SceneEntityCfg(uid="drawer", link_names=["handle"])
    )

    for row, body in enumerate(handles):
        assert body.friction == 0.5 + 0.1 * row
        assert len(body.writes) == (2 if row in selected else 0)
        assert untouched[row].writes == []


@pytest.mark.parametrize("backend, device", [("newton", "cuda"), ("default", "cpu")])
def test_event_skips_unaffected_backends(backend: str, device: str) -> None:
    sim = SimpleNamespace(physics_backend=backend, get_articulation=Mock())
    env = SimpleNamespace(sim=sim, device=torch.device(device))
    refresh_articulation_contact_material(
        env, None, SceneEntityCfg(uid="drawer", link_names=["handle"])
    )
    sim.get_articulation.assert_not_called()
