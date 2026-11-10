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
from shapely.geometry import MultiPolygon, Polygon, box

from embodichain.lab.sim.scene._packing_cpu import AABBLayoutConfig, AABBLayoutOptimizer
from embodichain.lab.sim.scene._placement_cpu import (
    GroupSupportClampConfig,
    GroupSupportClamp,
    group_height_offset,
)


def _corners(x: float, y: float, size: float = 0.2) -> np.ndarray:
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]])


def test_group_clamp_preserves_hole_and_relative_offsets() -> None:
    support = Polygon(
        box(0, 0, 2, 2).exterior.coords, [box(0.7, 0.7, 1.3, 1.3).exterior.coords]
    )
    bounds = {"a": _corners(0.9, 0.9), "b": _corners(1.4, 0.9)}
    original = {uid: corners.copy() for uid, corners in bounds.items()}
    delta = GroupSupportClamp(
        config=GroupSupportClampConfig(grid_resolution_m=0.05)
    ).find_translation(support, bounds)
    assert delta is not None and not np.allclose(delta, 0)
    assert all(support.covers(Polygon(corners + delta)) for corners in bounds.values())
    for uid in bounds:
        np.testing.assert_array_equal(bounds[uid], original[uid])
    np.testing.assert_allclose(
        (bounds["b"] + delta) - (bounds["a"] + delta), bounds["b"] - bounds["a"]
    )


def test_group_can_move_to_disconnected_support_component() -> None:
    support = MultiPolygon([box(0, 0, 0.3, 0.3), box(1, 0, 2, 1)])
    bounds = {"part": _corners(0.5, 0.2, size=0.4)}
    delta = GroupSupportClamp(
        config=GroupSupportClampConfig(grid_resolution_m=0.05)
    ).find_translation(support, bounds)
    assert delta is not None
    assert support.covers(Polygon(bounds["part"] + delta))


def test_failed_group_search_does_not_claim_infeasibility() -> None:
    result = GroupSupportClamp().find_translation(
        box(0, 0, 0.1, 0.1), {"part": _corners(0, 0)}
    )
    assert result is None


def test_separation_accepts_solution_found_on_last_round() -> None:
    bounds = {"first": _corners(0.2, 0.2, 0.4), "second": _corners(0.4, 0.2, 0.4)}
    offsets = AABBLayoutOptimizer(
        config=AABBLayoutConfig(max_rounds=1, aabb_clearance_m=0.01)
    ).solve(box(0, 0, 2, 2), bounds)
    first = Polygon(bounds["first"] + offsets["first"])
    second = Polygon(bounds["second"] + offsets["second"])
    assert first.distance(second) >= 0.01 - 1e-9


def test_group_height_offset_preserves_stacked_heights() -> None:
    bounds = np.array([[[0, 0, -0.2], [1, 1, 0.2]], [[0, 0, 0.5], [1, 1, 0.7]]])
    original = bounds.copy()
    delta = group_height_offset(1.0, bounds, clearance_m=0.03)
    assert bounds[:, 0, 2].min() + delta == pytest.approx(1.03)
    np.testing.assert_array_equal(bounds, original)


@pytest.mark.parametrize(
    "config",
    [
        GroupSupportClampConfig(grid_resolution_m=float("nan")),
        GroupSupportClampConfig(margin_m=-1),
    ],
)
def test_invalid_search_controls_rejected(config: GroupSupportClampConfig) -> None:
    with pytest.raises(ValueError):
        GroupSupportClamp(config=config)


def test_degenerate_aabb_rejected_before_search() -> None:
    with pytest.raises(ValueError, match="positive XY extents"):
        GroupSupportClamp().find_translation(box(0, 0, 1, 1), {"bad": np.zeros((4, 2))})


def test_mutated_nan_clearance_cannot_accept_overlapping_boxes() -> None:
    cfg = AABBLayoutConfig()
    optimizer = AABBLayoutOptimizer(config=cfg)
    cfg.aabb_clearance_m = float("nan")
    with pytest.raises(ValueError, match="finite"):
        optimizer.solve(
            box(0, 0, 2, 2), {"a": _corners(0.2, 0.2), "b": _corners(0.2, 0.2)}
        )


def test_mutated_clamp_resolution_rejected_even_for_already_contained_box() -> None:
    cfg = GroupSupportClampConfig()
    clamp = GroupSupportClamp(config=cfg)
    cfg.grid_resolution_m = float("nan")
    with pytest.raises(ValueError, match="finite"):
        clamp.find_translation(box(0, 0, 2, 2), {"part": _corners(0.2, 0.2)})


def test_projection_cannot_accept_box_larger_than_support_by_roundoff_allowance() -> (
    None
):
    # A prior 1e-9 allowance bypassed exact support containment on the success path.
    with pytest.raises(ValueError, match="larger"):
        AABBLayoutOptimizer().solve(
            box(0, 0, 1, 1), {"part": _corners(0, 0, 1 + 5e-10)}
        )


def test_group_center_may_stay_in_gap_between_disconnected_supports() -> None:
    support = MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])
    # The two boxes fit on their respective supports after one common shift;
    # the group's center stays in the unsupported gap for every valid placement.
    bounds = {"left": _corners(-0.5, 0.4), "right": _corners(2.5, 0.4)}
    delta = GroupSupportClamp(
        config=GroupSupportClampConfig(grid_resolution_m=0.05)
    ).find_translation(support, bounds)
    assert delta is not None
    assert all(support.covers(Polygon(corners + delta)) for corners in bounds.values())


@pytest.mark.parametrize("object_count", [16, 256])
def test_candidate_pair_geometry_work_is_independent_of_scene_size(
    monkeypatch, object_count
):
    from embodichain.lab.sim.scene import _packing_cpu

    original = _packing_cpu.aabb_overlap_depth
    measured_pair_counts = []

    def counted_overlap(first, second):
        # Enforce the pair evaluator's workload budget, not a specific indexing
        # implementation. Candidate refinement must not compare unrelated bodies.
        measured_pair_counts.append(int(np.prod(first.shape[:-2])))
        assert first.numel() + second.numel() <= 8
        return original(first, second)

    monkeypatch.setattr(_packing_cpu, "aabb_overlap_depth", counted_overlap)
    footprints = np.stack([_corners(0.2, 0.2)] * object_count)
    result = AABBLayoutOptimizer()._overlap_details(
        footprints, np.zeros((object_count, 2)), (0, object_count - 1)
    )
    assert measured_pair_counts == [1]
    assert len(result) == 1
    assert result[0][:2] == pytest.approx((0.200001, 0.200001))
    assert result[0][2:] == (0, object_count - 1)
