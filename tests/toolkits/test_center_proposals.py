# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
import torch
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.proposals import (
    unqualified_center_proposals,
)


def inputs(*, ground=False):
    pairs = torch.tensor(
        [
            [[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]],
            [[-0.02, 0.01, 0.01], [0.02, 0.01, 0.01]],
            [[-0.025, -0.01, 0.015], [0.025, -0.01, 0.015]],
        ]
    )
    vertices = torch.tensor(
        [
            [-0.04, -0.02, -0.03],
            [0.04, 0.02, 0.03],
            [-0.04, 0.02, 0.03],
            [0.04, -0.02, -0.03],
        ]
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.5], [0.1, 0.2, 0.6]])
    generator = object.__new__(AntipodalGraspPoseGenerator)
    generator._algorithm_cfg = AntipodalGraspPoseGeneratorCfg(
        approach_direction_samples=3,
        approach_deviation_angle=0.35,
        max_candidates=2,
        center_mode="centroid",
    )
    generator._collision_cfg = ParallelJawGraspCollisionCfg(
        filter_ground_collision=ground
    )
    calls = []

    def prepared(**kwargs):
        calls.append(kwargs)
        return pairs

    generator.prepare_mesh = prepared
    args = {
        "mesh_vertices": vertices,
        "mesh_triangles": torch.tensor([[0, 1, 2], [0, 3, 1]]),
        "obj_poses": poses,
        "approach_direction": torch.tensor([0.0, 0.0, -1.0]),
    }
    return generator, args, calls


def test_unqualified_center_proposals_preserve_source_order_costs_and_rng():
    generator, args, calls = inputs()
    torch.manual_seed(55)
    result = unqualified_center_proposals(generator, **args)
    reference = json.loads(
        (Path(__file__).parent / "fixtures/center_proposals_reference.json").read_text()
    )
    assert [poses.tolist() for poses, _ in result] == reference["poses"]
    assert [costs.tolist() for _, costs in result] == reference["costs"]
    assert (
        hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        == reference["rng_sha256"]
    )
    assert len(calls) == 1
    assert len(result[0][0]) > generator.algorithm_cfg.max_candidates


def test_unqualified_proposals_cannot_bypass_required_ground_contract():
    generator, args, calls = inputs(ground=True)
    with pytest.raises(ValueError, match="cannot bypass required ground"):
        unqualified_center_proposals(generator, **args)
    assert not calls
