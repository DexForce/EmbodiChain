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

"""Focused contracts for the public atomic-action adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from embodichain.gen_sim.action_engine.runtime import actions
from embodichain.gen_sim.action_engine.runtime.actions import AtomicActionAdapter
from embodichain.lab.sim.atomic_actions import Affordance


class _MeshEntity:
    def get_vertices(self, *, env_ids: list[int], scale: bool) -> torch.Tensor:
        assert env_ids == [0]
        assert scale
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def get_triangles(self, *, env_ids: list[int]) -> torch.Tensor:
        assert env_ids == [0]
        return torch.tensor([[0, 1, 2]], dtype=torch.int64)


def test_semantics_prewarms_vhacd_cache_before_affordance(
    monkeypatch: Any,
) -> None:
    """The lazy shared checker must see V-HACD's pickle, never create CoACD."""
    events: list[str] = []
    observed: dict[str, Any] = {}
    entity = _MeshEntity()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: entity if uid == "cube" else None
        ),
        agent_grasp_runtime_defaults={"max_decomposition_hulls": 8},
    )

    def fake_prepare(**kwargs: Any) -> SimpleNamespace:
        events.append("cache")
        observed.update(kwargs)
        return SimpleNamespace(status="hit")

    def fake_affordance(**_kwargs: Any) -> Affordance:
        events.append("affordance")
        return Affordance()

    monkeypatch.setattr(
        actions,
        "ensure_vhacd_grasp_collision_cache",
        fake_prepare,
    )
    monkeypatch.setattr(actions, "AntipodalAffordance", fake_affordance)

    adapter = AtomicActionAdapter(env)
    first = adapter.semantics("cube")
    second = adapter.semantics("cube")

    assert first is second
    assert events == ["cache", "affordance"]
    assert observed["max_decomposition_hulls"] == 8
    assert observed["mesh_vertices"].dtype == torch.float32
    assert observed["mesh_triangles"].dtype == torch.int64
