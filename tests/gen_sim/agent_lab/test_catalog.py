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

from embodichain.gen_sim.agent_lab.catalog import read_catalog


def test_catalog_keeps_task_identity_and_ignores_recipe_and_l4_appendix(
    tmp_path: Path,
) -> None:
    (tmp_path / "task100_revised.md").write_text(
        "| 001 | lift cup | E2 | extra hints | cup upright |\n"
        "| 095 | choose two | E1 | secret answer | sum is five |\n"
        "| 095 | appendix | action | exclusions |\n"
    )
    tasks = read_catalog(tmp_path)
    assert list(tasks) == ["task1101", "task1195"]
    assert tasks["task1195"].instruction == "choose two"
    assert "secret" not in str(tasks["task1195"].to_dict())


def test_missing_export_does_not_hide_intermediate_assets(tmp_path: Path) -> None:
    (tmp_path / "task100_revised.md").write_text("| 001 | move | E1 | notes | done |\n")
    geometry = tmp_path / "task1101/scene_generation/coarse_geometry"
    geometry.mkdir(parents=True)
    (geometry / "cup.glb").touch()
    assert read_catalog(tmp_path)["task1101"].asset_status == "intermediate_geometry"


def test_catalog_preserves_l3_and_l4_levels(tmp_path: Path) -> None:
    (tmp_path / "task100_revised.md").write_text(
        "## L3: 071-090\n| 078 | transport then handover | E5-E4 | hint | both stages |\n"
        "## L4: 091-100\n| 094 | mirror pattern | E1 | answer hint | symmetry |\n"
    )
    tasks = read_catalog(tmp_path)
    assert tasks["task1178"].level == "L3"
    assert tasks["task1194"].level == "L4"
    assert "answer hint" not in str(tasks["task1194"].to_dict())
