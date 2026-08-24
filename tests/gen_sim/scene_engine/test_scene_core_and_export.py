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

import numpy as np
import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_exporter import SceneExporter
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    SceneExportImporter,
)


def _scene_object(
    *,
    object_id: str,
    kind: str,
    glb_path: Path | None = None,
    physics: ObjectPhysics | None = None,
) -> SceneObject:
    return SceneObject(
        id=object_id,
        kind=kind,  # type: ignore[arg-type]
        category=kind,
        name=object_id,
        description=f"{kind} object",
        simready_glb_path=str(glb_path) if glb_path is not None else None,
        rot=[0.0, 0.0, 0.0],
        pos=[1.0, 2.0, 3.0],
        scale=[1.0, 2.0, 3.0],
        physics=physics,
    )


def _physics(body_type: str) -> ObjectPhysics:
    return ObjectPhysics(
        body_type=body_type,  # type: ignore[arg-type]
        attrs={"mass": 1.0, "static_friction": 0.8},
        max_convex_hull_num=16,
    )


def _scene_graph(scene: Scene) -> SceneGraph:
    if scene.table is None:
        raise ValueError("Test scene must contain a table.")
    return SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            *[
                SceneGraphNode(
                    object_id=asset.id,
                    parent_id="table",
                    parent_relation="on",
                )
                for asset in scene.assets
            ],
        ]
    )


def test_scene_returns_one_table_and_ordered_assets() -> None:
    table = _scene_object(object_id="table", kind="table")
    asset = _scene_object(object_id="cup", kind="asset")
    scene = Scene(objects=[table, asset])

    assert scene.table is table
    assert scene.assets == [asset]
    assert scene.to_dict()["objects"][0]["id"] == "table"  # type: ignore[index]


def test_scene_rejects_multiple_tables() -> None:
    scene = Scene(
        objects=[
            _scene_object(object_id="table_001", kind="table"),
            _scene_object(object_id="table_002", kind="table"),
        ]
    )

    with pytest.raises(ValueError, match="only one table"):
        _ = scene.table


@pytest.mark.parametrize(
    ("body_type", "attrs", "hulls"),
    [
        ("static", {"mass": 1.0}, 1),
        ("dynamic", {}, 1),
        ("dynamic", {"mass": 1.0}, 0),
    ],
)
def test_object_physics_rejects_invalid_values(
    body_type: str,
    attrs: dict[str, float],
    hulls: int,
) -> None:
    with pytest.raises(ValueError):
        _physics = ObjectPhysics(  # noqa: F841
            body_type=body_type,  # type: ignore[arg-type]
            attrs=attrs,
            max_convex_hull_num=hulls,
        )


def test_scene_export_copies_meshes_and_converts_y_up_pose(tmp_path: Path) -> None:
    table_glb = tmp_path / "table.glb"
    asset_glb = tmp_path / "cup.glb"
    table_glb.write_bytes(b"glTF-table")
    asset_glb.write_bytes(b"glTF-cup")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_glb,
        physics=_physics("kinematic"),
    )
    asset = _scene_object(
        object_id="cup",
        kind="asset",
        glb_path=asset_glb,
        physics=_physics("dynamic"),
    )
    asset.center_xy = [0.25, -0.5]

    scene = Scene(objects=[table, asset])
    export_path = SceneExporter(
        scene=scene,
        scene_graph=_scene_graph(scene),
        output_root=tmp_path / "output",
    ).export()
    exported = json.loads(export_path.read_text(encoding="utf-8"))

    assert (
        export_path.parent / "mesh_assets/table/table.glb"
    ).read_bytes() == b"glTF-table"
    assert (export_path.parent / "mesh_assets/cup/cup.glb").read_bytes() == b"glTF-cup"
    entry = exported["rigid_object"][0]
    assert entry["uid"] == "cup"
    assert entry["category"] == "asset"
    assert entry["name"] == "cup"
    assert entry["body_type"] == "dynamic"
    assert entry["init_pos"] == [1.0, -3.0, 2.0]
    assert entry["body_scale"] == [1.0, 2.0, 3.0]
    assert entry["center_xy"] == [0.25, -0.5]
    assert np.allclose(entry["init_rot"], [0.0, 0.0, 0.0])
    assert json.loads((export_path.parent / "scene_graph.json").read_text()) == {
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
                "orientation_state": None,
            },
            {
                "object_id": "cup",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": None,
                "orientation_state": None,
            },
        ],
        "relations": [],
    }

    imported_scene, imported_graph = SceneExportImporter(
        output_root=tmp_path / "output"
    ).import_scene_and_graph()
    assert [asset.id for asset in imported_scene.assets] == ["cup"]
    assert imported_scene.assets[0].category == "asset"
    assert imported_scene.assets[0].name == "cup"
    assert imported_graph.to_dict() == _scene_graph(scene).to_dict()


def test_scene_graph_importer_restores_node_orientation_state() -> None:
    imported_graph = SceneExportImporter._scene_graph_from_data(
        {
            "nodes": [
                {
                    "object_id": "table",
                    "parent_id": None,
                    "parent_relation": None,
                    "table_region": None,
                    "orientation_state": None,
                },
                {
                    "object_id": "bottle_001",
                    "parent_id": "table",
                    "parent_relation": "on",
                    "table_region": None,
                    "orientation_state": "standing",
                },
            ],
            "relations": [],
        }
    )

    assert imported_graph.node_by_id()["bottle_001"].orientation_state == "standing"


def test_scene_export_overwrites_an_existing_scene_export(tmp_path: Path) -> None:
    table_glb = tmp_path / "table.glb"
    cup_glb = tmp_path / "cup.glb"
    banana_glb = tmp_path / "banana.glb"
    table_glb.write_bytes(b"glTF-table")
    cup_glb.write_bytes(b"glTF-cup")
    banana_glb.write_bytes(b"glTF-banana")
    output_root = tmp_path / "output"

    initial_table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=table_glb,
        physics=_physics("kinematic"),
    )
    initial_cup = _scene_object(
        object_id="cup",
        kind="asset",
        glb_path=cup_glb,
        physics=_physics("dynamic"),
    )
    initial_scene = Scene(objects=[initial_table, initial_cup])
    SceneExporter(
        scene=initial_scene,
        scene_graph=_scene_graph(initial_scene),
        output_root=output_root,
    ).export()

    # The imported table mesh already occupies its final export location.
    exported_table_glb = (
        output_root / "scene_export" / "mesh_assets" / "table" / "table.glb"
    )
    updated_table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=exported_table_glb,
        physics=_physics("kinematic"),
    )
    banana = _scene_object(
        object_id="banana",
        kind="asset",
        glb_path=banana_glb,
        physics=_physics("dynamic"),
    )
    updated_scene = Scene(objects=[updated_table, banana])
    SceneExporter(
        scene=updated_scene,
        scene_graph=_scene_graph(updated_scene),
        output_root=output_root,
    ).export()

    scene_export_root = output_root / "scene_export"
    assert exported_table_glb.read_bytes() == b"glTF-table"
    assert (
        scene_export_root / "mesh_assets" / "banana" / "banana.glb"
    ).read_bytes() == b"glTF-banana"
    assert not (scene_export_root / "mesh_assets" / "cup").exists()
    assert (
        json.loads((scene_export_root / "scene.json").read_text(encoding="utf-8"))[
            "objects"
        ][1]["id"]
        == "banana"
    )


def test_scene_export_requires_final_physics(tmp_path: Path) -> None:
    glb_path = tmp_path / "table.glb"
    glb_path.write_bytes(b"glTF")
    table = _scene_object(object_id="table", kind="table", glb_path=glb_path)

    with pytest.raises(ValueError, match="no SimReady physics"):
        SceneExporter(
            scene=Scene(objects=[table]),
            scene_graph=SceneGraph(
                nodes=[SceneGraphNode(object_id="table", parent_id=None)]
            ),
            output_root=tmp_path,
        ).export()


def test_scene_export_rejects_backslash_in_object_id(tmp_path: Path) -> None:
    glb_path = tmp_path / "table.glb"
    glb_path.write_bytes(b"glTF")
    table = _scene_object(
        object_id="table",
        kind="table",
        glb_path=glb_path,
        physics=_physics("kinematic"),
    )
    unsafe_asset = _scene_object(
        object_id=r"..\evil",
        kind="asset",
        glb_path=glb_path,
        physics=_physics("dynamic"),
    )

    with pytest.raises(ValueError, match="not safe for a GLB filename"):
        scene = Scene(objects=[table, unsafe_asset])
        SceneExporter(
            scene=scene,
            scene_graph=_scene_graph(scene),
            output_root=tmp_path / "output",
        ).export()
