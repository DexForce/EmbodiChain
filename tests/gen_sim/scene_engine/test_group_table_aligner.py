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

from pathlib import Path

import pytest
import trimesh

from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner import (
    AssetsGroupTableAligner,
    AssetsGroupTableAlignerConfig,
)


def _layout(object_id: str, y: float) -> dict[str, object]:
    return {
        "id": object_id,
        "pos": [0.0, y, 0.0],
        "rot": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def test_group_table_aligner_preserves_relative_vertical_offsets(
    tmp_path: Path,
) -> None:
    trimesh.creation.box(extents=(2.0, 1.0, 2.0)).export(tmp_path / "table.glb")
    trimesh.creation.box(extents=(0.5, 1.0, 0.5)).export(tmp_path / "first.glb")
    trimesh.creation.box(extents=(0.5, 1.0, 0.5)).export(tmp_path / "second.glb")
    assets_layout = [_layout("first", 0.0), _layout("second", 0.3)]

    _, aligned_assets = AssetsGroupTableAligner(
        table_layout=_layout("table", 0.0),
        assets_layout=assets_layout,
        geometry_root=tmp_path,
        config=AssetsGroupTableAlignerConfig(clearance_m=0.1),
    ).align()

    assert aligned_assets[0]["pos"][1] > assets_layout[0]["pos"][1]  # type: ignore[index]
    assert aligned_assets[1]["pos"][1] - aligned_assets[0]["pos"][1] == pytest.approx(  # type: ignore[index]
        0.3
    )


def test_group_table_aligner_returns_empty_assets_without_mesh_loading(
    tmp_path: Path,
) -> None:
    table_layout = _layout("table", 0.0)

    aligned_table, aligned_assets = AssetsGroupTableAligner(
        table_layout=table_layout,
        assets_layout=[],
        geometry_root=tmp_path,
    ).align()

    assert aligned_table is table_layout
    assert aligned_assets == []
