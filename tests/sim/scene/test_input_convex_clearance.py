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

"""Declared input envelopes use native clearance without changing frame order."""

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
def test_input_convex_clearance_retains_declared_solids_and_frame_order(case):
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
    from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError

    def inspect():
        return convex_input_clearance(
            payload,
            poses,
            other,
            obstacle,
            torch.arange(len(case["steps"])),
            clearance_m=0.002,
            numerical_allowance_m=1e-6,
        )

    if case["name"] == "margin":
        # The geometric gap differs from the consumer threshold by only1um;
        # the native conversion interval cannot be forced to one endpoint.
        with pytest.raises(UnsupportedFeasibilityError):
            inspect()
        return
    actual = inspect()
    # Analytic box gaps: clear=0.4m minimum, touch=0m, containment overlaps,
    # margin=0.002m versus the explicitly requested0.002001m threshold.
    assert actual["valid"] is (case["name"] == "clear")
    assert actual["collision_backend"] == "newton.CollisionPipeline"
    assert actual["geometry_representation"] == "declared_input_convex_envelope"
    assert actual["checked_frames"] == (3 if case["name"] in {"clear", "touch"} else 1)
    if not actual["valid"]:
        assert actual["frame"] == (2 if case["name"] == "touch" else 0)
    assert all(
        torch.equal(a, b) for a, b in zip((payload, other, poses, obstacle), before)
    )


def test_convex_input_clearance_rejects_affine_pose_before_claiming_clearance():
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


def test_native_convex_envelope_retains_repeated_selected_frames():
    points = torch.tensor(
        trimesh.creation.box((0.25, 0.25, 0.25)).vertices, dtype=torch.float64
    )[None]
    poses = torch.eye(4, dtype=torch.float64).repeat(1, 3, 1, 1)
    other_pose = torch.eye(4, dtype=torch.float64)[None]
    other_pose[:, 0, 3] = 1.0
    result = convex_input_clearance(
        points,
        poses,
        points,
        other_pose,
        torch.tensor([2, 0, 2]),
        clearance_m=0.002,
        numerical_allowance_m=1e-6,
    )
    assert result["valid"] and result["checked_frames"] == 3
    assert result["required_clearance_m"] == 0.002
    assert result["numerical_allowance_m"] == 1e-6
