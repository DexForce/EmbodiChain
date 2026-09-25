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

from embodichain.data_analysis import coverage, distribution, joint_distribution


def _record(episode_id: str, **dimensions: object) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "dimensions": {
            key: {
                "value": value,
                "source": "measured" if value is not None else "unknown",
                "missing_reason": "unavailable" if value is None else None,
            }
            for key, value in dimensions.items()
        },
    }


def test_distribution_counts_zero_unknown_and_missing_separately() -> None:
    records = [
        _record("zero", pose_x=0.0),
        _record("unknown", pose_x=None),
        _record("missing", material="wood"),
    ]

    assert distribution(records, "pose_x") == [
        {"value": 0.0, "count": 1},
        {"value": None, "count": 1},
        {"value": "__missing__", "count": 1},
    ]


def test_numeric_distribution_uses_explicit_bin_edges() -> None:
    records = [
        _record("a", duration_s=0.0),
        _record("b", duration_s=0.9),
        _record("c", duration_s=1.0),
    ]
    assert distribution(records, "duration_s", bins=[0.0, 1.0, 2.0]) == [
        {"bin_start": 0.0, "bin_end": 1.0, "count": 2},
        {"bin_start": 1.0, "bin_end": 2.0, "count": 1},
    ]


def test_joint_distribution_and_literal_three_of_four_coverage() -> None:
    records = [
        _record("a", material="wood", light="bright"),
        _record("b", material="wood", light="dim"),
        _record("c", material="metal", light="bright"),
        _record("unknown", material=None, light="dim"),
    ]
    rows = joint_distribution(records, "material", "light")
    assert rows == [
        {"material": "metal", "light": "bright", "count": 1},
        {"material": "wood", "light": "bright", "count": 1},
        {"material": "wood", "light": "dim", "count": 1},
        {"material": None, "light": "dim", "count": 1},
    ]
    assert coverage(
        rows,
        valid_cells=[
            ("wood", "bright"),
            ("wood", "dim"),
            ("metal", "bright"),
            ("metal", "dim"),
        ],
    ) == {
        "covered": 3,
        "total": 4,
        "coverage": 0.75,
    }


def test_coverage_without_target_definition_has_no_percentage() -> None:
    assert coverage([], valid_cells=None) == {
        "covered": 0,
        "total": None,
        "coverage": None,
    }
