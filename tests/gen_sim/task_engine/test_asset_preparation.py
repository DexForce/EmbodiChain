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

"""Semantic acquisition bindings delegate asset fit and normalization."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import pytest
import trimesh
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.asset_preparation import (
    fit_prepared_scene_assets,
    normalize_prepared_scene_assets,
)


def fixture(tmp_path):
    path = tmp_path / "object.glb"
    trimesh.creation.box(extents=[0.06, 0.08, 0.18]).export(path)
    source = tmp_path / "source.json"
    source.write_text("{}")
    table = dict(
        uid="table",
        shape=dict(shape_type="Cube", size=[1.2, 1.2, 0.1]),
        init_pos=[0.6, 0.0, 0.68],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
    )
    obj = dict(
        uid="object",
        shape=dict(shape_type="Mesh", fpath=str(path)),
        init_pos=[0.6, 0.0, 0.82],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
        attrs=dict(mass_props=dict(mass=0.005)),
    )
    planner = tuple(
        dict(value, runtime_uid=value["uid"], source_uid=value["uid"], role=role)
        for value, role in ((table, "background"), (obj, "rigid_object"))
    )
    scene = PreparedScene(
        source,
        tmp_path,
        planner,
        (table,),
        (obj,),
        (),
        {},
        0.73,
        0.0,
        "preserve",
        (1.0, 1.0, 1.0),
        {"object": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    graph = dict(
        nodes=[
            dict(
                task_type="E1",
                call=dict(kind="pick", object="object", resources=dict(primary="left")),
            )
        ]
    )
    return scene, graph


def test_optional_fit_precedes_normalization_and_keeps_both_views(tmp_path):
    scene, graph = fixture(tmp_path)
    fitted, report = fit_prepared_scene_assets(
        scene, graph, openings={"left": 0.05}, contact_clearances={"object": 0.003}
    )
    assert report["status"] == "scaled"
    assert report["records"][0]["task_types"] == ["E1"]
    assert (
        fitted.rigid_objects[0]["body_scale"] == fitted.planner_objects[1]["body_scale"]
    )
    assert fitted.rigid_objects[0]["init_pos"] == fitted.planner_objects[1]["init_pos"]
    assert scene.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    later = deepcopy(graph)
    later["nodes"].append(
        dict(
            task_type="E1",
            call=dict(kind="pick", object="object", resources=dict(primary="right")),
        )
    )
    with pytest.raises(ValueError, match="stricter"):
        fit_prepared_scene_assets(scene, later, openings={"left": 0.05, "right": 0.04})
