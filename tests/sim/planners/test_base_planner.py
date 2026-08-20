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

from dataclasses import FrozenInstanceError

import pytest

from embodichain.lab.sim.planners.base_planner import CollisionWorldInfo


def test_collision_world_info_represents_one_contract() -> None:
    info = CollisionWorldInfo(
        entity_ids=("cube", "table"),
        dynamic_entity_ids=("cube",),
        batch_mode="per_env",
        supports_updates=True,
    )

    assert info.entity_ids == ("cube", "table")
    assert info.dynamic_entity_ids == ("cube",)


def test_collision_world_info_is_immutable() -> None:
    info = CollisionWorldInfo()

    with pytest.raises(FrozenInstanceError):
        info.supports_updates = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("entity_ids", "error_type", "match"),
    [
        (("cube", "cube"), ValueError, "unique"),
        ((" cube",), TypeError, "outer whitespace"),
    ],
)
def test_collision_world_info_rejects_invalid_entity_ids(
    entity_ids: tuple[str, ...],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        CollisionWorldInfo(entity_ids=entity_ids)


def test_collision_world_info_requires_dynamic_ids_in_complete_world() -> None:
    with pytest.raises(ValueError, match="subset"):
        CollisionWorldInfo(
            entity_ids=("table",),
            dynamic_entity_ids=("cube",),
        )


def test_collision_world_info_rejects_invalid_batch_mode() -> None:
    with pytest.raises(ValueError, match="batch_mode"):
        CollisionWorldInfo(batch_mode="batched")  # type: ignore[arg-type]


def test_collision_world_info_requires_boolean_update_capability() -> None:
    with pytest.raises(TypeError, match="supports_updates"):
        CollisionWorldInfo(supports_updates=1)  # type: ignore[arg-type]
