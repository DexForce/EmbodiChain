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

"""Pure Polars-backed distribution and coverage queries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import polars as pl

__all__ = ["coverage", "distribution", "joint_distribution"]

_MISSING = "__missing__"


def _value(record: Mapping[str, Any], dimension: str) -> Any:
    dimensions = record.get("dimensions", {})
    if dimension not in dimensions:
        return _MISSING
    measurement = dimensions[dimension]
    return measurement.get("value") if isinstance(measurement, Mapping) else _MISSING


def _sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (1, "")
    if value == _MISSING:
        return (2, "")
    return (0, f"{type(value).__name__}:{value}")


def distribution(
    records: Iterable[Mapping[str, Any]],
    dimension: str,
    *,
    bins: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Count one dimension, optionally using explicit numeric bin edges."""
    values = [_value(record, dimension) for record in records]
    if bins is not None:
        edges = [float(edge) for edge in bins]
        if len(edges) < 2 or any(
            right <= left for left, right in zip(edges, edges[1:])
        ):
            raise ValueError(
                "Bins must contain at least two strictly increasing edges."
            )
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        rows = []
        for index, (left, right) in enumerate(zip(edges, edges[1:])):
            count = sum(
                left <= value < right or (index == len(edges) - 2 and value == right)
                for value in numeric
            )
            rows.append({"bin_start": left, "bin_end": right, "count": count})
        return rows
    if not values:
        return []
    encoded = [json.dumps(value, sort_keys=True) for value in values]
    frame = pl.DataFrame({"encoded": encoded})
    rows = [
        {"value": json.loads(row["encoded"]), "count": row["count"]}
        for row in frame.group_by("encoded").len(name="count").to_dicts()
    ]
    return sorted(rows, key=lambda row: _sort_key(row["value"]))


def joint_distribution(
    records: Iterable[Mapping[str, Any]], x: str, y: str
) -> list[dict[str, Any]]:
    """Count observed pairs for two dimensions, preserving unknown and missing cells."""
    values = [{x: _value(record, x), y: _value(record, y)} for record in records]
    if not values:
        return []
    encoded = [
        {x: json.dumps(row[x], sort_keys=True), y: json.dumps(row[y], sort_keys=True)}
        for row in values
    ]
    frame = pl.DataFrame(encoded)
    rows = [
        {x: json.loads(row[x]), y: json.loads(row[y]), "count": row["count"]}
        for row in frame.group_by([x, y]).len(name="count").to_dicts()
    ]
    return sorted(rows, key=lambda row: (_sort_key(row[x]), _sort_key(row[y])))


def coverage(
    joint_rows: Iterable[Mapping[str, Any]],
    *,
    valid_cells: Sequence[tuple[Any, Any]] | None,
) -> dict[str, int | float | None]:
    """Calculate target-cell coverage, or report an undefined denominator."""
    if valid_cells is None:
        return {"covered": 0, "total": None, "coverage": None}
    valid = set(valid_cells)
    observed = {
        tuple(value for key, value in row.items() if key != "count")
        for row in joint_rows
        if row.get("count", 0) > 0
    }
    covered = len(valid & observed)
    total = len(valid)
    return {
        "covered": covered,
        "total": total,
        "coverage": covered / total if total else None,
    }
