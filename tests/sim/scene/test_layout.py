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

from dataclasses import replace

import numpy as np
import pytest

from embodichain.lab.sim.scene.layout import (
    LayoutCfg,
    LayoutProblem,
    PlanarRelation,
    solve_layout,
)


def _problem() -> LayoutProblem:
    return LayoutProblem(
        support_uid="tray",
        seed_xy_by_uid={"first": (-0.5, 0.0), "second": (0.5, 0.0)},
        half_extents_xy_by_uid={"first": (0.1, 0.1), "second": (0.1, 0.1)},
        bounds_xy_by_uid={
            uid: ((-1.0, -1.0), (1.0, 1.0)) for uid in ("first", "second")
        },
    )


def test_valid_seeds_have_zero_displacement_without_reserved_table_name() -> None:
    problem = _problem()
    solution = solve_layout(problem)
    assert solution.status == "solved"
    assert solution.xy_by_uid == problem.seed_xy_by_uid
    assert solution.geometry == "fixed_orientation_aabb"
    assert max(solution.residuals.values()) == 0.0


def test_axis_relation_moves_pair_minimally() -> None:
    problem = replace(
        _problem(),
        seed_xy_by_uid={"first": (0.0, 0.0), "second": (0.0, 0.0)},
        relations=(PlanarRelation("first", "left_of", "second"),),
    )
    result = solve_layout(problem)
    assert result.status == "solved"
    # Equal displacement weights split the 0.23 m required spacing evenly.
    assert result.xy_by_uid["first"][0] == pytest.approx(-0.115)
    assert result.xy_by_uid["second"][0] == pytest.approx(0.115)


def test_fixed_obstacle_can_be_unbounded_without_exempting_movable_bounds() -> None:
    problem = replace(
        _problem(),
        fixed_xy_by_uid={"first": (2.0, 0.0)},
        bounds_xy_by_uid={"second": ((-0.3, -0.3), (0.3, 0.3))},
    )
    solution = solve_layout(problem)
    assert solution.status == "solved"
    assert solution.xy_by_uid["first"] == (2.0, 0.0)
    assert solution.xy_by_uid["second"][0] == pytest.approx(0.2)


def test_fixed_position_outside_declared_region_is_rejected() -> None:
    problem = replace(_problem(), fixed_xy_by_uid={"first": (2.0, 0.0)})
    assert solve_layout(problem).status == "no_solution"


@pytest.mark.parametrize(
    "problem",
    [
        replace(_problem(), frame="world"),
        replace(
            _problem(),
            relations=(PlanarRelation("first", "left_of", "second", frame="camera"),),
        ),
        replace(_problem(), relations=(PlanarRelation("first", "inside", "second"),)),
    ],
)
def test_unsupported_frame_or_relation_returns_no_candidate(problem) -> None:
    solution = solve_layout(problem)
    assert solution.status == "unsupported"
    assert solution.xy_by_uid == {}


@pytest.mark.parametrize(
    "problem",
    [
        replace(_problem(), fixed_xy_by_uid={"missing": (0.0, 0.0)}),
        replace(_problem(), preferred_uids=("missing",)),
        replace(
            _problem(), relations=(PlanarRelation("missing", "left_of", "second"),)
        ),
        replace(_problem(), half_extents_xy_by_uid={"first": (0.1, 0.1)}),
        replace(
            _problem(),
            seed_xy_by_uid={"first": (float("nan"), 0.0), "second": (0.5, 0.0)},
        ),
    ],
)
def test_invalid_uid_or_geometry_inputs_are_rejected(problem) -> None:
    with pytest.raises(ValueError):
        solve_layout(problem)


def test_cfg_rejects_negative_seed_weight_at_public_boundary() -> None:
    with pytest.raises(ValueError, match="imported_seed_weight"):
        solve_layout(_problem(), LayoutCfg(imported_seed_weight=-1.0))


def test_custom_clearance_uses_metres() -> None:
    problem = replace(
        _problem(), seed_xy_by_uid={"first": (0.0, 0.0), "second": (0.0, 0.0)}
    )
    result = solve_layout(problem, LayoutCfg(collision_margin_m=0.2))
    assert result.status == "solved"
    delta = np.asarray(result.xy_by_uid["second"]) - result.xy_by_uid["first"]
    assert np.max(np.abs(delta)) == pytest.approx(0.4)
