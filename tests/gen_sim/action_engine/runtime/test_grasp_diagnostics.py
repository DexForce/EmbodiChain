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

import torch

from embodichain.gen_sim.action_engine.runtime.grasp_diagnostics import (
    _TracingAntipodalGraspPoseGenerator,
)
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg


class _CollisionChecker:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return torch.tensor([False, True, False]), torch.tensor([0.01, -0.002, 0.0])
        return torch.tensor([True, False]), torch.tensor([-0.004, 0.003])


class _Backend:
    device = torch.device("cpu")
    _max_deviation_angle = torch.pi / 6
    _approach_direction_samples = 4

    def __init__(self) -> None:
        self.antipodal_pairs = torch.tensor(
            [
                [[-0.08, -0.08, 0.0], [0.08, -0.08, 0.0]],
                [[-0.08, -0.06, 0.0], [0.08, -0.06, 0.0]],
                [[-0.08, 0.06, 0.0], [0.08, 0.06, 0.0]],
                [[-0.08, 0.08, 0.0], [0.08, 0.08, 0.0]],
            ],
            dtype=torch.float32,
        )
        self._collision_checker = _CollisionChecker()

    def get_dual_arm_valid_grasp_poses(self, **_kwargs):
        pose = torch.eye(4).repeat(3, 1, 1)
        left_colliding, _ = self._collision_checker.query(None, pose, torch.ones(3))
        right_colliding, _ = self._collision_checker.query(
            None, pose[:2], torch.ones(2)
        )
        return {
            "left": {
                "is_success": True,
                "grasp_poses": pose[~left_colliding],
                "open_lengths": torch.ones(2),
                "total_cost": torch.tensor([0.1, 0.2]),
            },
            "right": {
                "is_success": True,
                "grasp_poses": pose[:2][~right_colliding],
                "open_lengths": torch.ones(1),
                "total_cost": torch.tensor([0.3]),
            },
        }


def test_dual_grasp_trace_separates_generation_angle_nms_and_collision(
    monkeypatch,
) -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="trace_test")
    )
    backend = _Backend()
    monkeypatch.setattr(generator, "_backend", lambda *_args: backend)
    vertices = torch.tensor(
        [
            [-0.1, -0.1, -0.02],
            [0.1, -0.1, -0.02],
            [0.1, 0.1, 0.02],
            [-0.1, 0.1, 0.02],
        ]
    )

    result = generator.get_dual_arm_valid_grasp_poses(
        mesh_vertices=vertices,
        mesh_triangles=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        obj_poses=torch.eye(4).unsqueeze(0),
        left_to_right_arm_direction=torch.tensor([0.0, 1.0, 0.0]),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        middle_empty_ratio=0.4,
    )

    assert result[0] is not None
    trace = generator.last_dual_trace
    assert trace is not None
    assert trace["S1_grasp_pair_generation"]["antipodal_pair_count"] == 4
    assert trace["S2_approach_angle_filtering"] == {
        "left_partition_pair_count": 2,
        "right_partition_pair_count": 2,
        "left_angle_valid_pair_count": 2,
        "right_angle_valid_pair_count": 2,
    }
    assert trace["S3_nms"] == {
        "left_candidate_count": 3,
        "right_candidate_count": 2,
    }
    assert trace["S4_collision_filtering"]["left_candidate_count"] == 2
    assert trace["S4_collision_filtering"]["right_candidate_count"] == 1
    assert trace["S5_left_right_pairing"] == {
        "left_final_count": 2,
        "right_final_count": 1,
        "paired": True,
    }
    assert generator.last_dual_trace is not trace


def test_generator_context_selects_non_crossing_pair_and_canonical_half_turn() -> None:
    generator = _TracingAntipodalGraspPoseGenerator(
        ParallelJawGripperModelCfg(model_id="pair_test")
    )
    left_poses = torch.stack((_pose_with_y(-0.2), _pose_with_y(0.2)))
    left_poses[0, 0, 0] = -1.0
    left_poses[0, 1, 1] = -1.0
    right_poses = torch.stack((_pose_with_y(0.2), _pose_with_y(-0.2)))
    result = {
        "left": {
            "is_success": True,
            "grasp_poses": left_poses,
            "open_lengths": torch.ones(2),
            "total_cost": torch.tensor([0.1, 0.0]),
        },
        "right": {
            "is_success": True,
            "grasp_poses": right_poses,
            "open_lengths": torch.ones(2),
            "total_cost": torch.tensor([0.1, 0.0]),
        },
    }

    with generator.dual_arm_selection_context(
        left_eef=torch.eye(4).unsqueeze(0),
        right_eef=torch.eye(4).unsqueeze(0),
        left_base=_pose_with_y(-0.3).unsqueeze(0),
        right_base=_pose_with_y(0.3).unsqueeze(0),
        left_to_right_direction=torch.tensor([0.0, 1.0, 0.0]),
        pair_rank=0,
        minimum_separation=0.08,
        minimum_lateral_gap=0.05,
    ):
        selected, trace = generator._select_pair(result, row_index=0)

    assert selected is not None
    assert trace is not None
    assert trace["selected"] is True
    assert trace["selected_left_index"] == 0
    assert trace["selected_right_index"] == 0
    assert trace["selected_left_half_turn"] is True
    torch.testing.assert_close(
        selected["left"]["grasp_poses"][0, :3, :3],
        torch.eye(3),
    )


def _pose_with_y(y: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[1, 3] = y
    return pose
