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

"""Immutable target filters retain source candidate order and failure padding."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope
from embodichain.toolkits.graspkit.pg_grasp.constraints import (
    GraspGeometryConstraint,
    ConstrainedGraspPoseGenerator,
    grasp_geometry_digest,
)


class Candidates(ParallelJawGraspPoseGenerator):
    def __init__(self):
        super().__init__(
            **{
                key: (
                    {
                        "model_id": "robotiq_arg2f_140",
                        "min_opening_width": 0.001,
                        "max_opening_width": 0.1,
                    }
                )[key]
                for key in ("model_id", "min_opening_width", "max_opening_width")
            }
        )

    def get_valid_grasp_poses(self, *, obj_poses, **kwargs):
        rows = []
        for obj in obj_poses:
            poses = obj[None].repeat(5, 1, 1)
            poses[:, :3, 3] += poses.new_tensor(
                [
                    [0, 0, -0.02],
                    [0, 0, 0.02],
                    [0, 0, 0.01],
                    [0.2, 0, 0.03],
                    [0, 0, 0.04],
                ]
            )
            rows.append((poses, torch.tensor([0.2, 0.4, 0.1, 0.0, torch.inf])))
        return rows

    def get_best_grasp_poses(self, **kwargs):
        raise AssertionError("unused")

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        raise AssertionError("unused")


def inputs(mode):
    vertices = torch.tensor([[-0.02, -0.02, -0.04], [0.02, 0.02, 0.04]])
    faces = torch.tensor([[0, 1, 0]])
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, 3] = torch.tensor([0.1, -0.2, 0.3])
    release = torch.eye(4)
    release[2, 3] = 0.6
    rule = GraspGeometryConstraint(
        grasp_geometry_digest(vertices, faces),
        (0.0, 0.0, 1.0),
        0.05 if mode == "empty" else 0.0,
        mode in {"region", "both", "empty"},
        release_pose=(
            None if mode in {"region", "empty"} else tuple(release.flatten().tolist())
        ),
        plane_z=None if mode in {"region", "empty"} else 0.54,
        margin=0.0,
        hand_envelope=(
            None
            if mode in {"region", "empty"}
            else NominalHandEnvelope(0.13, 0.03, 0.01, 0.08, 0.003)
        ),
        model_id=None if mode in {"region", "empty"} else "robotiq_arg2f_140",
    )
    return vertices, faces, poses, rule


@pytest.mark.parametrize("mode", ["region", "release", "both", "empty"])
def test_constraint_rows_match_frozen_source_filter(mode):
    expected = json.loads(
        (Path(__file__).parent / "data/grasp_constraint_b0.json").read_text()
    )[mode]
    vertices, faces, poses, rule = inputs(mode)
    original = vertices.clone()
    view = ConstrainedGraspPoseGenerator(Candidates(), rule)
    rows = view.get_valid_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=faces,
        obj_poses=poses,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    actual = [dict(poses=p.tolist(), costs=c.tolist()) for p, c in rows]
    assert actual == expected
    torch.testing.assert_close(vertices, original)
    vertices[0, 0] += 0.001
    with pytest.raises(ValueError, match="immutable geometry"):
        view.get_valid_grasp_poses(
            mesh_vertices=vertices,
            mesh_triangles=faces,
            obj_poses=poses,
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )


def test_constraint_rejects_foreign_nominal_hand_and_changed_topology():
    vertices, faces, poses, rule = inputs("release")
    view = ConstrainedGraspPoseGenerator(
        Candidates(), replace(rule, model_id="foreign")
    )
    with pytest.raises(ValueError, match="model differs"):
        view.accepts_grasp_poses(poses, poses, vertices, faces)
    with pytest.raises(ValueError, match="immutable geometry"):
        ConstrainedGraspPoseGenerator(Candidates(), rule).accepts_grasp_poses(
            poses, poses, vertices, faces[:, [0, 2, 1]]
        )
