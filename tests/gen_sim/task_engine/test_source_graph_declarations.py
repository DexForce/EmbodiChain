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

"""Unrefined inputs match original complete source declaration transformations."""

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import pytest
from embodichain.gen_sim.task_engine.source_graph_declarations import (
    refine_source_manipulation_declarations,
    refine_source_return_declarations,
)
from ._source_refinement_fixture import source_refinement_fixture

REFERENCE = json.loads(
    (Path(__file__).parent / "data/source_refinements_b0.json").read_text()
)


@pytest.mark.parametrize("row", REFERENCE["cases"], ids=lambda row: row["case"])
def test_unrefined_source_graph_matches_complete_original_geometry_pipeline(
    tmp_path, row
):
    graph, scene = source_refinement_fixture(tmp_path, row["case"])
    before = deepcopy(graph)
    expected = row["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match="identity collides"):
            refine_source_manipulation_declarations(graph, scene)
    else:
        result = refine_source_manipulation_declarations(graph, scene)
        result = refine_source_return_declarations(result, scene)
        assert (
            hashlib.sha256(
                json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            == expected["sha256"]
        )
    assert graph == before


PICK_REFERENCE = json.loads(
    (Path(__file__).parents[2] / "sim/scene/data/alignment_pick_b0.json").read_text()
)


@pytest.mark.parametrize("row", PICK_REFERENCE["cases"], ids=lambda row: row["name"])
def test_constrained_pick_declarations_match_complete_original_integration(
    tmp_path, row
):
    import trimesh
    from embodichain.gen_sim.task_engine.alignment_declarations import (
        resolve_alignment_pick_declarations,
    )

    mesh = trimesh.Trimesh(row["mesh_vertices"], row["mesh_faces"], process=False)
    asset = tmp_path / "source.glb"
    mesh.export(asset)
    sources = deepcopy(row["sources"])
    for source in sources:
        if source["uid"] != "table":
            source["shape"]["fpath"] = str(asset)
    kwargs = dict(
        table_top_z=row["table_top_z"],
        hands=(),
        default_pick_options={"hand_interp_steps": 12},
    )
    options, lowerers = resolve_alignment_pick_declarations(
        row["graph"], source_objects={v["uid"]: v for v in sources}, **kwargs
    )
    assert options == row["expected"]["options"]
    assert {v["call_id"]: v["routes"] for v in lowerers} == {
        v["call_id"]: v["routes"] for v in row["expected"]["lowerers"]
    }
    if row["name"] == "e2_vertical":
        # The original simple E2 public call still needs no source adaptation.
        ordinary_options, ordinary_lowerers = resolve_alignment_pick_declarations(
            row["graph"], **kwargs
        )
        assert ordinary_options == options
        assert ordinary_lowerers == lowerers
    else:
        with pytest.raises(ValueError, match="explicit geometry declaration"):
            resolve_alignment_pick_declarations(row["graph"], **kwargs)


@pytest.fixture(autouse=True)
def forbid_runtime_construction(monkeypatch):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("semantic graph declarations constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)
