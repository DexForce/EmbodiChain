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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.grasp_support import (
    _ActionAgentAntipodalAffordance,
)


@pytest.mark.parametrize("alignment_angle", [None, 0.35])
def test_action_agent_affordance_forwards_complete_grasp_contract(
    alignment_angle: float | None,
) -> None:
    affordance = _ActionAgentAntipodalAffordance()
    generator = Mock()
    generator.device = torch.device("cpu")
    generator.cfg = SimpleNamespace(max_deviation_angle=0.7)
    grasp_pose = torch.eye(4).unsqueeze(0)

    def get_valid_grasp_poses(
        object_pose, approach_direction, *, object_part, pose_cost_fn
    ):
        del object_pose, approach_direction
        assert object_part == "top"
        costs = pose_cost_fn(grasp_pose, torch.tensor([0.25]))
        return True, grasp_pose, torch.tensor([0.05]), costs

    generator.get_valid_grasp_poses.side_effect = get_valid_grasp_poses
    affordance._generator = generator
    affordance.set_custom_config(
        "action_agent_max_approach_alignment_angle",
        alignment_angle,
    )

    results = affordance.get_valid_grasp_poses(
        torch.eye(4).unsqueeze(0),
        object_part="top",
        grasp_cost_fn=lambda _object_pose, _grasp_poses, costs: costs + 1.0,
    )

    assert results[0][1].tolist() == [1.25]
    assert generator.cfg.max_deviation_angle == pytest.approx(0.7)
