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

from embodichain.gen_sim.scene_engine.cli import preview


class _PreviewSim:
    def __init__(self) -> None:
        self.rigid_objects: list[object] = []

    def add_rigid_object(self, cfg: object) -> None:
        self.rigid_objects.append(cfg)


def test_preview_add_objects_accepts_mesh_inside_scene_export(tmp_path: Path) -> None:
    config_dir = tmp_path / "scene_export"
    mesh_path = config_dir / "mesh_assets" / "table" / "table.glb"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(b"glTF")
    sim = _PreviewSim()

    preview._add_objects(
        sim=sim,
        entries=[
            {
                "uid": "table",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh_assets/table/table.glb",
                },
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
            }
        ],
        config_dir=config_dir,
        label="table",
    )

    assert len(sim.rigid_objects) == 1


@pytest.mark.parametrize("fpath", ["../outside.glb", "/tmp/outside.glb"])
def test_preview_add_objects_rejects_mesh_path_outside_scene_export(
    tmp_path: Path,
    fpath: str,
) -> None:
    config_dir = tmp_path / "scene_export"
    config_dir.mkdir()

    with pytest.raises(ValueError, match="must (be a relative path|stay within)"):
        preview._add_objects(
            sim=_PreviewSim(),
            entries=[
                {
                    "uid": "table",
                    "shape": {"shape_type": "Mesh", "fpath": fpath},
                    "init_pos": [0.0, 0.0, 0.0],
                    "init_rot": [0.0, 0.0, 0.0],
                }
            ],
            config_dir=config_dir,
            label="table",
        )
