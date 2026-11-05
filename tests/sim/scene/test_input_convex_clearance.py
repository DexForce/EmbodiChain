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

"""Input-solid clearance through actual FCL, including full containment."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
import trimesh
from embodichain.lab.sim.scene.convex_clearance import convex_input_clearance

_CASES = json.loads(
    (Path(__file__).parent / "fixtures/convex_clearance_reference.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", _CASES, ids=lambda item: item["name"])
def test_input_convex_clearance_matches_original_solids_and_frame_order(case):
    pytest.importorskip(
        "fcl", reason="Actual FCL qualification requires planning-geometry/python-fcl."
    )
    payload = torch.tensor(
        trimesh.creation.box((0.2, 0.2, 0.2)).vertices, dtype=torch.float64
    )[None]
    other = torch.tensor(
        trimesh.creation.box(case["extent"]).vertices, dtype=torch.float64
    )[None]
    poses = torch.eye(4, dtype=torch.float64).repeat(1, len(case["steps"]), 1, 1)
    poses[0, :, 0, 3] = torch.tensor(case["steps"], dtype=torch.float64)
    obstacle = torch.eye(4, dtype=torch.float64)[None]
    obstacle[:, 0, 3] = case["offset"]
    before = payload.clone(), other.clone(), poses.clone(), obstacle.clone()
    actual = convex_input_clearance(
        payload,
        poses,
        other,
        obstacle,
        torch.arange(len(case["steps"])),
        clearance_m=0.002,
        numerical_allowance_m=1e-6,
    )
    assert actual == case["result"]
    assert all(
        torch.equal(a, b) for a, b in zip((payload, other, poses, obstacle), before)
    )


def test_convex_input_clearance_rejects_affine_pose_before_claiming_clearance():
    pytest.importorskip("fcl")
    mesh = torch.tensor(trimesh.creation.box().vertices, dtype=torch.float64)[None]
    poses = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    poses[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid observed poses"):
        convex_input_clearance(
            mesh,
            poses,
            mesh,
            torch.eye(4, dtype=torch.float64)[None],
            torch.tensor([0]),
            clearance_m=0.002,
            numerical_allowance_m=1e-6,
        )
