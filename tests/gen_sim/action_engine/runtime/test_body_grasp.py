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

"""Tests for elongated-object axis analysis and body-grasp filtering."""

from __future__ import annotations

import math

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime.body_grasp import (
    AxisAlignBodyGraspAdapter,
    select_body_grasp_candidates,
)
from embodichain.gen_sim.action_engine.runtime.geometry_axes import (
    analyze_local_geometry_axes,
)
from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    AxisAlignGoal,
    ObjectSemantics,
)


def _box_vertices(extents: tuple[float, float, float]) -> torch.Tensor:
    half = torch.tensor(extents) * 0.5
    return torch.tensor(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )


def test_can_geometry_resolves_local_y_as_the_long_axis() -> None:
    axes = analyze_local_geometry_axes(_box_vertices((0.0611, 0.1143, 0.0632)))

    assert axes.long_axis_index == 1
    assert axes.short_axis_index == 0
    torch.testing.assert_close(axes.long_axis, torch.tensor([0.0, 1.0, 0.0]))
    assert axes.elongation_ratio == pytest.approx(0.1143 / 0.0632)


def test_axis_analysis_rejects_ambiguous_or_rotated_local_geometry() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        analyze_local_geometry_axes(_box_vertices((0.06, 0.06, 0.06)))

    vertices = _box_vertices((0.04, 0.12, 0.05))
    angle = math.radians(35.0)
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="not aligned"):
        analyze_local_geometry_axes(vertices @ rotation.T)


def test_body_grasp_rejects_caps_and_longitudinal_closing() -> None:
    axes = analyze_local_geometry_axes(_box_vertices((0.06, 0.12, 0.06)))
    candidates = torch.eye(4).repeat(1, 3, 1, 1)
    candidates[0, 0, 1, 3] = 0.055  # End-cap candidate with the best raw cost.
    candidates[0, 1, 1, 3] = 0.0  # Central radial body grasp.
    candidates[0, 2, 1, 3] = 0.0
    candidates[0, 2, :3, 0] = torch.tensor([0.0, 1.0, 0.0])
    candidates[0, 2, :3, 1] = torch.tensor([1.0, 0.0, 0.0])
    costs = torch.tensor([[0.0, 0.2, 0.1]])

    selected = select_body_grasp_candidates(
        candidates,
        costs,
        torch.eye(4).unsqueeze(0),
        axes,
    )

    assert selected.success.tolist() == [True]
    assert selected.candidate_indices.tolist() == [1]
    assert selected.body_candidate_counts.tolist() == [1]
    assert selected.ranked_candidate_indices.tolist() == [[1]]
    torch.testing.assert_close(selected.grasp_xpos[0], candidates[0, 1])


def test_body_grasp_chooses_a_reachable_body_candidate() -> None:
    axes = analyze_local_geometry_axes(_box_vertices((0.06, 0.12, 0.06)))
    candidates = torch.eye(4).repeat(1, 2, 1, 1)
    candidates[0, 1, 0, 3] = 0.01
    costs = torch.tensor([[0.0, 0.2]])

    selected = select_body_grasp_candidates(
        candidates,
        costs,
        torch.eye(4).unsqueeze(0),
        axes,
        feasible=torch.tensor([[False, True]]),
    )

    assert selected.candidate_indices.tolist() == [1]
    assert selected.reachable_candidate_counts.tolist() == [1]


def test_axis_align_adapter_injects_the_selected_body_grasp_unchanged() -> None:
    vertices = _box_vertices((0.06, 0.12, 0.06))
    triangles = torch.tensor([[0, 1, 2], [0, 2, 3]])
    goal = AxisAlignGoal(
        semantics=ObjectSemantics(
            label="can",
            geometry={},
            affordance=AxisAlignAffordance(
                mesh_vertices=vertices,
                mesh_triangles=triangles,
                internal_axis=torch.tensor([0.0, 1.0, 0.0]),
            ),
        )
    )
    candidate = torch.eye(4).unsqueeze(0)

    class Generator:
        def get_valid_grasp_poses(self, **_kwargs):
            return [(candidate, torch.tensor([0.1]))]

    adapted = AxisAlignBodyGraspAdapter().adapt(
        goal,
        object_pose=torch.eye(4).unsqueeze(0),
        grasp_generator=Generator(),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        target_axis=torch.tensor([0.0, 0.0, 1.0]),
        seed=7,
    )

    assert adapted.goal.grasp_xpos is not None
    assert len(adapted.alternative_goals) == 1
    assert adapted.alternative_rank_indices == (0,)
    explicit = adapted.goal.grasp_xpos
    torch.testing.assert_close(explicit, candidate)
