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

"""Task grasp filters preserve baseline candidate and failure contracts."""

from __future__ import annotations


import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
    GraspRule,
    TaskGraspPoseGenerator,
    accepted_candidates,
    geometry_key,
)
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)

__all__: list[str] = []

VERTICES = torch.tensor([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.20], [0.0, 0.01, 0.10]])
TRIANGLES = torch.tensor([[0, 1, 2]])


class CandidateGenerator(ParallelJawGraspPoseGenerator):
    def __init__(self, rows: tuple[tuple[float, ...], ...]) -> None:
        super().__init__(
            ParallelJawGripperModelCfg(
                model_id="test",
                finger_length=0.10,
                finger_width=0.02,
                finger_thickness=0.01,
            )
        )
        self.rows = rows

    def get_valid_grasp_poses(self, **kwargs):
        poses = kwargs["obj_poses"]
        results = []
        for row, heights in enumerate(self.rows):
            local = torch.eye(4).repeat(len(heights), 1, 1)
            local[:, 2, 3] = torch.tensor(heights)
            results.append(
                (poses[row] @ local, torch.arange(len(heights), dtype=torch.float32))
            )
        return results

    def get_best_grasp_poses(self, **kwargs):
        poses = kwargs["obj_poses"]
        return (
            torch.ones(len(poses), dtype=torch.bool),
            poses.clone(),
            torch.full((len(poses),), 0.05),
        )

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        return [None] * len(kwargs["obj_poses"])


def rule(*, upper_half=True, margin=0.02) -> GraspRule:
    return GraspRule(
        "can",
        "release",
        geometry_key(VERTICES, TRIANGLES),
        (0.0, 0.0, 1.0),
        0.10,
        upper_half,
        tuple(torch.eye(4).reshape(-1).tolist()),
        0.0,
        margin,
    )


def sample(provider, poses, **kwargs):
    return provider.get_valid_grasp_poses(
        mesh_vertices=VERTICES,
        mesh_triangles=TRIANGLES,
        obj_poses=poses,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        **kwargs,
    )


def test_filter_uses_object_local_region_independently_for_each_row() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    before = poses.clone()
    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((0.08, 0.15), (0.08,))), (rule(),)
    )
    rows = sample(provider, poses)
    assert rows[0][0].shape == (1, 4, 4)
    assert rows[0][0][0, 2, 3].item() == pytest.approx(0.15)
    assert torch.isfinite(rows[0][1]).all()
    assert rows[1][0].shape == (1, 4, 4)
    assert torch.isinf(rows[1][1]).all()
    torch.testing.assert_close(poses, before)


def test_release_filter_rejects_a_low_grasp_before_stock_ik() -> None:
    provider = TaskGraspPoseGenerator(
        CandidateGenerator(((0.06, 0.10),)), (rule(upper_half=False),)
    )
    rows = sample(provider, torch.eye(4).unsqueeze(0))
    assert rows[0][0][:, 2, 3].tolist() == pytest.approx([0.10])


def test_symmetric_roll_variants_have_identical_release_clearance() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, 2, 3] = torch.tensor([0.06, 0.15])
    mirrored = poses.clone()
    mirrored[:, :3, :2] *= -1
    model = CandidateGenerator(((),)).gripper_model
    torch.testing.assert_close(
        accepted_candidates(poses, torch.eye(4), rule(), model),
        accepted_candidates(mirrored, torch.eye(4), rule(), model),
    )


def test_no_candidate_is_encoded_as_ineligible_not_as_a_fake_success() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((), (0.05,))), (rule(),))
    rows = sample(provider, torch.eye(4).repeat(2, 1, 1))
    assert all(
        p.shape == (1, 4, 4) and torch.isfinite(p).all() and torch.isinf(c).all()
        for p, c in rows
    )


def test_end_specific_sampling_keeps_its_baseline_contract() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((0.06, 0.10),)), (rule(),))
    rows = sample(
        provider,
        torch.eye(4).unsqueeze(0),
        obj_longest_axis=torch.tensor([0.0, 0.0, 1.0]),
        is_positive_part=False,
    )
    assert rows[0][0].shape[0] == 2
    assert torch.isfinite(rows[0][1]).all()


def test_rule_identity_must_match_the_bound_object_and_geometry() -> None:
    provider = TaskGraspPoseGenerator(CandidateGenerator(((0.15,),)), (rule(),))
    provider.require_rule("can", "release", VERTICES, TRIANGLES)
    with pytest.raises(ValueError, match="matching"):
        provider.require_rule("other", "release", VERTICES, TRIANGLES)
    with pytest.raises(ValueError, match="matching"):
        provider.require_rule("can", "release", VERTICES * 2, TRIANGLES)
