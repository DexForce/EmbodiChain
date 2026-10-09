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

import pytest
from embodichain.data_analysis.dimensions import (
    DIMENSIONS,
    build_filters,
    dimension_summary,
    joint_cells,
    select_cell,
    match_dimension,
)


def row(i, **values):
    return {
        "episode_id": str(i),
        "dimensions": {
            k: {
                "value": v,
                "source": "unknown" if v is None else "measured",
                "unit": DIMENSIONS[k]["unit"],
                "frame": "world",
                "scope": "episode",
            }
            for k, v in values.items()
        },
    }


def test_definitions_cover_six_families_and_unknown_is_not_zero():
    assert {d["family"] for d in DIMENSIONS.values()} == {
        "asset",
        "pose",
        "affordance",
        "trajectory",
        "material",
        "light",
    }
    records = [row(0, pose_x=0.0), row(1, pose_x=None), row(2), row(3, pose_x="broken")]
    summary = next(x for x in dimension_summary(records) if x["key"] == "pose_x")
    assert (
        summary["known"],
        summary["unknown"],
        summary["missing"],
        summary["invalid"],
    ) == (1, 1, 1, 1)
    assert summary["completeness"] == 0.25
    assert summary["sources"] == {"measured": 2, "unknown": 1, "missing": 1}


def test_filters_are_typed_multi_select_and_validated():
    filters = build_filters(
        {"asset": ["cube", "cup"]},
        {"pose_x": (-0.5, 0.0), "light": (None, 7)},
        "affordance",
        "unknown",
    )
    assert match_dimension(row(1, asset="cube"), "asset", filters["asset"])
    assert match_dimension(row(1, pose_x=0.0), "pose_x", filters["pose_x"])
    assert not match_dimension(row(1, pose_x=None), "pose_x", filters["pose_x"])
    assert match_dimension(row(1, affordance=None), "affordance", filters["affordance"])
    assert not match_dimension(row(1), "affordance", filters["affordance"])
    for limits in [(2, 1), (float("nan"), 1)]:
        with pytest.raises(ValueError):
            build_filters({}, {"pose_x": limits})
    with pytest.raises(ValueError):
        build_filters({}, {"asset": (1, 2)})


def test_joint_cells_preserve_exact_boundaries_unknowns_and_ids():
    records = [
        row(0, asset="cube", pose_x=0.0),
        row(1, asset="cube", pose_x=1.0),
        row(2, asset="cube", pose_x=2.0),
        row(3, asset="cube", pose_x=None),
        row(4, asset="cube"),
    ]
    cells = joint_cells(records, "asset", "pose_x", y_bins=[0, 1, 2])
    assert [c["count"] for c in cells] == [1, 2, 1, 1]
    assert [r["episode_id"] for r in select_cell(records, cells[1])] == ["1", "2"]
    assert sum(c["count"] for c in cells) == len(records)
    assert cells[2]["y"] == "未知" and cells[3]["y"] == "缺失"
    assert len(joint_cells(records, "asset", "pose_x", y_bins=[0.5, 1.5])) == 5


def test_same_dimension_joint_is_rejected():
    with pytest.raises(ValueError):
        joint_cells([], "asset", "asset")


def test_numeric_joint_cells_merge_equivalent_int_and_float_values():
    cells = joint_cells(
        [row(1, material="wood", light=3), row(2, material="wood", light=3.0)],
        "material",
        "light",
    )
    assert len(cells) == 1 and cells[0]["count"] == 2
    assert cells[0]["episode_ids"] == ["1", "2"]
