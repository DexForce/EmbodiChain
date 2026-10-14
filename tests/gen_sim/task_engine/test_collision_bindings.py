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

from copy import deepcopy

import pytest

from embodichain.gen_sim.task_engine._task_program import collision_bindings as binding


def test_collision_owners_are_bound_without_task_labels_and_keep_existing_ids(
    monkeypatch,
):
    monkeypatch.setattr(
        binding,
        "collision_meshes",
        lambda _: {
            ("drawer", "/drawer/a"): object(),
            ("drawer", "/drawer/b"): object(),
            ("body", "/body"): object(),
        },
    )
    existing = [
        {
            "entity_id": "semantic_drawer",
            "articulation_id": "cabinet",
            "native_link_name": "drawer",
        }
    ]
    before = deepcopy(existing)
    expected = [
        {
            "entity_id": "gen_sim.collision_link.cabinet.body",
            "articulation_id": "cabinet",
            "native_link_name": "body",
        }
    ]
    assert (
        binding.additional_collision_links([{"uid": "cabinet"}], existing) == expected
    )
    assert existing == before
    assert (
        binding.additional_collision_links([{"uid": "cabinet"}], [*existing, *expected])
        == []
    )


def test_ambiguous_native_link_identity_is_not_guessed():
    links = [
        {
            "entity_id": identity,
            "articulation_id": "cabinet",
            "native_link_name": "body",
        }
        for identity in ("first", "second")
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        binding.additional_collision_links([], links)


def test_missing_physical_mesh_does_not_publish_empty_collision_coverage(monkeypatch):
    def unavailable(_):
        raise ValueError("missing mesh")

    monkeypatch.setattr(binding, "collision_meshes", unavailable)
    with pytest.raises(ValueError, match="missing mesh"):
        binding.additional_collision_links([{"uid": "cabinet"}], [])
