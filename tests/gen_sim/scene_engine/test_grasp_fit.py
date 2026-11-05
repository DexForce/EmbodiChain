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

"""Explicit asset-fit results preserve original bounded scale and support height."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import trimesh
from embodichain.gen_sim.scene_engine.grasp_fit import (
    SourceGraspFitRequest,
    fit_source_grasp_declarations,
)

CASES = json.loads((Path(__file__).parent / "data/grasp_fit_b0.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_source_fit_matches_independent_original_results(tmp_path, case):
    path = tmp_path / "mesh.glb"
    trimesh.creation.box(extents=[0.1, 0.18, 0.3]).export(path)
    source = deepcopy(case["object"])
    source["shape"]["fpath"] = str(path)
    before = deepcopy(source)
    fitted = fit_source_grasp_declarations(
        (source,),
        (
            SourceGraspFitRequest(
                "object", case["opening"], 0.003, case["mode"], "source"
            ),
        ),
    )
    actual = fitted.rigid_objects[0]
    record = dict(fitted.records[0])
    actual["shape"]["fpath"] = "mesh.glb"
    record["source_path"] = "mesh.glb"
    record.update(task_types=["E1"], resources=["left"])
    assert record == case["record"]
    assert actual == case["adapted"]
    assert source == before


def test_fit_rejects_an_impossible_bounded_scale_before_mutating_source(tmp_path):
    path = tmp_path / "large.glb"
    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(path)
    source = dict(
        uid="large",
        shape=dict(shape_type="Mesh", fpath=str(path)),
        body_scale=[1.0, 1.0, 1.0],
        init_pos=[0.0, 0.0, 0.5],
        init_rot=[0.0, 0.0, 0.0],
    )
    before = deepcopy(source)
    with pytest.raises(ValueError, match="below minimum"):
        fit_source_grasp_declarations(
            (source,), (SourceGraspFitRequest("large", 0.05),)
        )
    assert source == before


def test_fit_keeps_unselected_objects_source_order_and_sorted_owned_receipts(tmp_path):
    path = tmp_path / "mesh.glb"
    trimesh.creation.box(extents=[0.1, 0.18, 0.3]).export(path)

    def obj(uid):
        return dict(
            uid=uid,
            shape=dict(shape_type="Mesh", fpath=str(path)),
            body_scale=[1.0, 1.0, 1.0],
            init_pos=[0.0, 0.0, 0.5],
            init_rot=[0.0, 0.0, 0.0],
            attrs={"mass_props": {"mass": 0.1}},
        )

    objects = [
        obj("z"),
        {"uid": "untouched", "shape": {"shape_type": "Cube", "size": [1, 1, 1]}},
        obj("a"),
    ]
    before = deepcopy(objects)
    fitted = fit_source_grasp_declarations(
        objects, [SourceGraspFitRequest("z", 1.0), SourceGraspFitRequest("a", 1.0)]
    )
    assert [item["uid"] for item in fitted.rigid_objects] == ["z", "untouched", "a"]
    assert [record["object_id"] for record in fitted.records] == ["a", "z"]
    assert all(record["status"] == "unchanged" for record in fitted.records)
    assert list(fitted.rigid_objects) == objects and objects == before
    fitted.rigid_objects[0]["attrs"]["mass_props"]["mass"] = 5.0
    assert objects == before
    with pytest.raises(ValueError, match="unique typed"):
        fit_source_grasp_declarations(
            objects, [SourceGraspFitRequest("z", 1.0), SourceGraspFitRequest("z", 1.0)]
        )
    with pytest.raises(ValueError, match="one bound rigid"):
        fit_source_grasp_declarations(objects, [SourceGraspFitRequest("missing", 1.0)])
