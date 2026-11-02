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

import json
from pathlib import Path
import pytest
from embodichain.gen_sim.task_engine.orchestration.source_scene import (
    resolve_source_scene,
)


@pytest.mark.parametrize("ids", [["other"], ["cup", "cup"], ["cup"]])
def test_scene_export_rejects_mixed_companion_ids(
    tmp_path: Path, ids: list[str]
) -> None:
    path = tmp_path / "scene_config.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "rigid_object": [{"uid": "cup"}],
            }
        )
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"objects": [{"id": uid} for uid in ids]})
    )
    if ids == ["cup"]:
        assert resolve_source_scene(path).path == path
    else:
        with pytest.raises(ValueError, match="companion"):
            resolve_source_scene(path)


@pytest.mark.parametrize("field", ["category", "name", "description", "is_articulated"])
def test_scene_export_rejects_companion_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "scene_config.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "rigid_object": [{"uid": "cup", field: "runtime"}],
            }
        )
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"objects": [{"id": "cup", field: "different"}]})
    )
    with pytest.raises(ValueError, match=f"companion {field} differs"):
        resolve_source_scene(path)
