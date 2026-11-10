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


import numpy as np
import pytest

from embodichain.lab.sim.scene._layout_cpu import solve_aabb_layout


def _solve(*, seeds=None, bounds=None, fixed=None, **kwargs):
    seeds = np.zeros((2, 2)) if seeds is None else np.asarray(seeds, dtype=float)
    count = len(seeds)
    return solve_aabb_layout(
        seeds,
        np.full((count, 2), 0.1),
        (
            np.tile([[-1.0, -1.0], [1.0, 1.0]], (count, 1, 1))
            if bounds is None
            else bounds
        ),
        np.zeros(count, dtype=bool) if fixed is None else fixed,
        np.ones(count),
        **kwargs,
    )


def test_collision_repair_does_not_move_fixed_obstacle_or_input() -> None:
    seeds = np.zeros((2, 2))
    result = _solve(seeds=seeds, fixed=np.asarray([True, False]))
    assert result.status == "solved"
    np.testing.assert_array_equal(result.xy[0], seeds[0])
    np.testing.assert_array_equal(seeds, np.zeros((2, 2)))
    # Each box is 0.2 m wide, with a required 0.02 m extra gap.
    assert np.max(np.abs(result.xy[1])) >= 0.22 - 1e-7
    assert max(result.residuals.values()) <= 1e-7


def test_oversized_box_returns_no_solution_with_violation() -> None:
    result = _solve(
        seeds=[[0.0, 0.0]], bounds=np.asarray([[[-0.05, -0.05], [0.05, 0.05]]])
    )
    assert result.status == "no_solution"
    assert result.residuals["linear"] == pytest.approx(0.05)


def test_iteration_budget_has_distinct_result() -> None:
    result = _solve(max_slsqp_iterations=1)
    assert result.status == "budget_exhausted"


def test_success_on_last_collision_round_is_accepted() -> None:
    result = _solve(max_collision_rounds=1)
    assert result.status == "solved"


def test_fixed_overlap_requires_explicit_compatibility_exemption() -> None:
    fixed = np.ones(2, dtype=bool)
    assert _solve(fixed=fixed).status == "no_solution"
    assert _solve(fixed=fixed, skip_fixed_collisions=True).status == "solved"


def test_already_separated_boxes_still_get_required_clearance() -> None:
    result = _solve(seeds=[[0.0, 0.0], [0.21, 0.0]])
    assert result.status == "solved"
    assert result.xy[1, 0] - result.xy[0, 0] == pytest.approx(0.22)


def test_legacy_mode_retains_nonoverlapping_boxes() -> None:
    seeds = np.asarray([[0.0, 0.0], [0.21, 0.0]])
    result = _solve(seeds=seeds, enforce_all_clearances=False)
    np.testing.assert_array_equal(result.xy, seeds)


def test_contradictory_relations_do_not_claim_solved() -> None:
    result = _solve(relations=((0, 1, 0, 1.0), (1, 0, 0, 1.0)))
    assert result.status in {"no_solution", "budget_exhausted"}
    assert result.residuals["linear"] > 0.0


def test_fixed_outside_region_fails_before_numerical_search() -> None:
    result = _solve(seeds=[[2.0, 0.0]], fixed=np.ones(1, dtype=bool))
    assert result.status == "no_solution"
    assert result.residuals["linear"] > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"collision_margin_m": float("nan")},
        {"relation_clearance_m": -1.0},
        {"max_collision_rounds": 0},
        {"max_slsqp_iterations": True},
        {"skip_fixed_collisions": "false"},
        {"enforce_all_clearances": "false"},
    ],
)
def test_invalid_controls_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        _solve(**kwargs)


def test_empty_layout_is_solved() -> None:
    result = _solve(seeds=np.empty((0, 2)))
    assert result.status == "solved"
    assert result.xy.shape == (0, 2)


def test_collision_pair_budget_is_reported_separately_from_no_solution() -> None:
    result = _solve(
        seeds=np.zeros((3, 2)), max_collision_rounds=1, max_added_collision_pairs=1
    )
    assert result.status == "budget_exhausted"
    assert result.residuals["collision"] > 0.0


@pytest.mark.parametrize("fixed", [[0, 1], [False, float("nan")]])
def test_fixed_mask_cannot_silently_cast_invalid_values(fixed) -> None:
    with pytest.raises(ValueError, match="boolean"):
        _solve(fixed=fixed)


@pytest.mark.parametrize("relations", [((0.5, 1, 0, 1.0),), ((0, 1, 0.5, 1.0),)])
def test_relation_indices_reject_nonintegral_values(relations) -> None:
    with pytest.raises(ValueError):
        _solve(relations=relations)


@pytest.mark.parametrize("inner_half", [0.1, 0.0])
def test_contained_proxy_retains_separation_depth_not_intersection_width(inner_half):
    # A zero-width proxy still requires displacement to leave a enclosing box;
    # plain intersection width would incorrectly report zero collision here.
    result = solve_aabb_layout(
        np.zeros((2, 2)),
        np.array([[1.0, 1.0], [inner_half, inner_half]]),
        np.tile([[-2.0, -2.0], [2.0, 2.0]], (2, 1, 1)),
        np.ones(2, dtype=bool),
        np.ones(2),
        collision_margin_m=0.0,
    )
    assert result.status == "no_solution"
    assert result.residuals["collision"] == pytest.approx(1.0 + inner_half)


def test_tiny_contained_proxy_is_not_hidden_by_overlap_width_tolerance():
    tiny_half = 1e-9  # Smaller than the feasibility tolerance, but still enclosed.
    result = solve_aabb_layout(
        np.zeros((2, 2)),
        np.array([[1.0, 1.0], [tiny_half, tiny_half]]),
        np.tile([[-2.0, -2.0], [2.0, 2.0]], (2, 1, 1)),
        np.ones(2, dtype=bool),
        np.ones(2),
        collision_margin_m=0.0,
    )
    assert result.status == "no_solution"
    assert result.residuals["collision"] > 1.0
