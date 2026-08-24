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

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler import (
    GravitySettleBody,
    GravitySettler,
)

_TABLE_ID = "table"
_ASSET_ID = "cube_001"
_IDENTITY_LAYOUT = {
    "rot": [0.0, 0.0, 0.0],
    "pos": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
}


def _table_body() -> GravitySettleBody:
    return GravitySettleBody(
        scene_object=SceneObject(
            id=_TABLE_ID,
            kind="table",
            category="table",
            name="table",
            description="table",
        ),
        y_up_layout={"id": _TABLE_ID, **_IDENTITY_LAYOUT},
    )


def _asset_body() -> GravitySettleBody:
    return GravitySettleBody(
        scene_object=SceneObject(
            id=_ASSET_ID,
            kind="asset",
            category="cube",
            name="cube",
            description="cube",
        ),
        y_up_layout={"id": _ASSET_ID, **_IDENTITY_LAYOUT},
    )


def test_gravity_settler_returns_no_poses_without_dynamic_assets() -> None:
    settled_pose_by_id = GravitySettler(
        table_body=_table_body(),
        participant_bodies=[_asset_body()],
        dynamic_asset_ids=set(),
        static_asset_ids={_ASSET_ID},
    ).settle()

    assert settled_pose_by_id == {}


def test_gravity_settler_rejects_dynamic_assets_outside_participants() -> None:
    with pytest.raises(ValueError, match="exactly match participants"):
        GravitySettler(
            table_body=_table_body(),
            participant_bodies=[],
            dynamic_asset_ids={_ASSET_ID},
            static_asset_ids=set(),
        ).settle()
