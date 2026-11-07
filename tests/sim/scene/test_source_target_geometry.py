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

"""Direct Scene target inputs match original complete source refinements."""

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pytest
import trimesh
from embodichain.lab.sim.scene.source_target_geometry import (
    source_axis_vertical_component,
    source_vertical_bounds,
    source_upright_target,
    source_handover_staging_pose,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/source_targets_b0.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_authored_source_targets_match_complete_b0_graph_geometry(tmp_path, case):
    mesh = trimesh.Trimesh(
        vertices=REFERENCE["mesh_vertices"],
        faces=REFERENCE["mesh_faces"],
        process=False,
    )
    asset = tmp_path / "source.glb"
    mesh.export(asset)
    source = deepcopy(case["source"])
    source["shape"]["fpath"] = str(asset)
    expected = case["expected"]
    assert source_axis_vertical_component(source) == expected["axis_vertical"]
    for aligned, bounds in zip((False, True), expected["vertical_bounds"], strict=True):
        assert source_vertical_bounds(source, axis_aligned=aligned) == tuple(bounds)
    for full, name in ((False, "plain"), (True, "staging")):
        target = source_upright_target(
            source,
            support_height=0.8,
            world_yaw=3.141592653589793,
            ordinary_release_clearance=0.01,
            slender_release_clearance=0.01,
            full_rotation_staging=full,
            rotation_margin=0.02,
        )
        height, quaternion = target
        assert height == expected[name][0]
        actual_quaternion = np.asarray(quaternion)
        expected_quaternion = np.asarray(expected[name][1])
        assert actual_quaternion.shape == expected_quaternion.shape == (4,)
        # Quaternion signs can differ for the same pose. Retain the existing
        # float32 pose tolerance while keeping height and other geometry exact.
        if actual_quaternion @ expected_quaternion < 0:
            expected_quaternion = -expected_quaternion
        tolerance = float(4 * np.finfo(np.float32).eps)
        np.testing.assert_allclose(
            actual_quaternion, expected_quaternion, rtol=0, atol=tolerance
        )
        assert np.linalg.norm(actual_quaternion) == pytest.approx(
            1, rel=0, abs=tolerance
        )
    assert (
        source_handover_staging_pose(source, case["table"], lift_height=0.16)
        == expected["handover"]
    )


def test_missing_raw_source_bounds_remain_an_unavailable_target():
    assert (
        source_upright_target(
            {},
            support_height=0.8,
            world_yaw=0.0,
            ordinary_release_clearance=0.01,
            slender_release_clearance=0.01,
        )
        is None
    )


PICK_REFERENCE = json.loads(
    (Path(__file__).parent / "data/alignment_pick_b0.json").read_text()
)


@pytest.mark.parametrize("row", PICK_REFERENCE["cases"], ids=lambda row: row["name"])
def test_direct_constrained_pick_geometry_matches_original_integration(tmp_path, row):
    from embodichain.lab.sim.scene.source_target_geometry import (
        source_constrained_pick_geometry,
    )

    mesh = trimesh.Trimesh(row["mesh_vertices"], row["mesh_faces"], process=False)
    asset = tmp_path / "source.glb"
    mesh.export(asset)
    source = deepcopy(next(v for v in row["sources"] if v["uid"] != "table"))
    source["shape"]["fpath"] = str(asset)
    option = next(iter(row["expected"]["options"].values()))
    route = row["expected"]["lowerers"][0]["routes"][0]
    kwargs = dict(
        table_top_z=row["table_top_z"],
        release_origin_z=route["release_clearance_object_pose"]["position"][2],
        previously_aligned=row["name"] == "e1_after_align",
        force_vertical=row["name"] in {"e2_vertical", "supported_upright"},
        supported_upright=row["name"] == "supported_upright",
        ordinary_release_clearance=0.01,
        slender_release_clearance=0.01,
    )
    assert source_constrained_pick_geometry(source, **kwargs) == (
        option["approach_direction"],
        route["release_clearance_plane_z"],
    )
    if row["name"] == "e1_tilted":
        # This original source branch uses authored Euler angles. A matrix
        # override owns other scene queries, but cannot change this approach.
        source["init_local_pose"] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        assert (
            source_constrained_pick_geometry(source, **kwargs)[0]
            == option["approach_direction"]
        )


@pytest.fixture(autouse=True)
def forbid_live_runtime(monkeypatch):
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args, **kwargs):
        pytest.fail("authored target geometry constructed a live runtime")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(BaseSolver, "__init__", forbidden)
