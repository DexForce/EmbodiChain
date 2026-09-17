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

"""Versioned feature definitions and exact, offline slice membership."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "DIMENSIONS",
    "build_filters",
    "dimension_summary",
    "joint_cells",
    "select_cell",
    "match_dimension",
]

#: Versioned feature definitions with display labels, families, value kinds and units.
DIMENSIONS = {
    "asset": {"label": "物体资产", "family": "asset", "kind": "category", "unit": ""},
    "pose_x": {"label": "初始 X", "family": "pose", "kind": "number", "unit": "m"},
    "pose_y": {"label": "初始 Y", "family": "pose", "kind": "number", "unit": "m"},
    "pose_z": {"label": "初始 Z", "family": "pose", "kind": "number", "unit": "m"},
    "yaw": {"label": "初始偏航角", "family": "pose", "kind": "number", "unit": "rad"},
    "affordance": {
        "label": "Affordance",
        "family": "affordance",
        "kind": "category",
        "unit": "",
    },
    "approach": {
        "label": "接近方式 / 方向标注",
        "family": "affordance",
        "kind": "category",
        "unit": "",
    },
    "trajectory_family": {
        "label": "轨迹几何族",
        "family": "trajectory",
        "kind": "category",
        "unit": "",
    },
    "duration_s": {
        "label": "轨迹时长",
        "family": "trajectory",
        "kind": "number",
        "unit": "s",
    },
    "material": {"label": "材质", "family": "material", "kind": "category", "unit": ""},
    "light": {
        "label": "光照强度",
        "family": "light",
        "kind": "number",
        "unit": "renderer_intensity",
    },
}
for _definition in DIMENSIONS.values():
    _definition["version"] = 1


def _state(record: Mapping[str, Any], key: str) -> str:
    measurement = record.get("dimensions", {}).get(key)
    if measurement is None:
        return "missing"
    value = measurement.get("value")
    if value is None:
        return "unknown"
    spec = DIMENSIONS[key]
    if measurement.get("source") == "unknown":
        return "invalid"
    if spec["kind"] == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return "invalid"
        if measurement.get("unit") != spec["unit"]:
            return "invalid"
    elif not isinstance(value, str) or not value:
        return "invalid"
    return "known"


def dimension_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report completeness and sources separately for every defined feature.

    Known means a non-null value with compatible type and, for numeric values,
    expected units. Unknown, absent and incompatible facts never count as zero.
    """
    rows = []
    for key, spec in DIMENSIONS.items():
        counts = Counter(_state(r, key) for r in records)
        sources = Counter(
            r.get("dimensions", {}).get(key, {}).get("source", "missing")
            for r in records
        )
        rows.append(
            dict(
                key=key,
                **spec,
                **{s: counts[s] for s in ("known", "unknown", "missing", "invalid")},
                completeness=counts["known"] / len(records) if records else None,
                sources=dict(sources),
            )
        )
    return rows


def build_filters(
    categories: Mapping[str, Sequence[str]],
    bounds: Mapping[str, tuple[float | None, float | None]],
    availability_dimension: str | None = None,
    availability: str = "all",
) -> dict[str, Any]:
    """Build validated inclusive range and multi-value filters for the catalog.

    Empty selections are unrestricted. Availability modes distinguish missing
    dimensions, explicit unknowns, incompatible facts and known values.
    """
    result = {}
    for key, values in categories.items():
        if key not in DIMENSIONS or DIMENSIONS[key]["kind"] != "category":
            raise ValueError(f"Unsupported categorical dimension: {key}")
        if isinstance(values, str) or any(not isinstance(v, str) for v in values or []):
            raise ValueError(f"{key}: expected a list of categorical values.")
        if values:
            result[key] = {"values": list(values), "availability": "known"}
    for key, (lower, upper) in bounds.items():
        if key not in DIMENSIONS or DIMENSIONS[key]["kind"] != "number":
            raise ValueError(f"Unsupported numeric dimension: {key}")
        if any(
            v is not None
            and (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            )
            for v in (lower, upper)
        ):
            raise ValueError(f"{key}: bounds must be finite numbers.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{key}: minimum exceeds maximum.")
        if lower is not None or upper is not None:
            result[key] = {"min": lower, "max": upper, "availability": "known"}
    if availability not in ("all", "known", "unknown", "missing", "invalid"):
        raise ValueError("Unsupported availability mode.")
    if availability_dimension:
        if availability_dimension not in DIMENSIONS:
            raise ValueError("Unknown availability dimension.")
        if availability != "all":
            if availability_dimension in result and availability != "known":
                raise ValueError(
                    "Clear value/range filters for a dimension before selecting unavailable values."
                )
            result.setdefault(availability_dimension, {})["availability"] = availability
    elif availability != "all":
        raise ValueError("Select a dimension for availability filtering.")
    return result


def match_dimension(
    record: Mapping[str, Any], key: str, condition: Mapping[str, Any]
) -> bool:
    """Evaluate the structured filter contract produced by :func:`build_filters`."""
    if key not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {key}")
    state = _state(record, key)
    availability = condition.get("availability", "all")
    if availability != "all" and state != availability:
        return False
    value = record.get("dimensions", {}).get(key, {}).get("value")
    if "values" in condition and value not in condition["values"]:
        return False
    if any(condition.get(k) is not None for k in ("min", "max")):
        if state != "known" or DIMENSIONS[key]["kind"] != "number":
            return False
        if condition.get("min") is not None and value < condition["min"]:
            return False
        if condition.get("max") is not None and value > condition["max"]:
            return False
    return True


def joint_cells(
    records: Sequence[Mapping[str, Any]],
    x: str,
    y: str,
    *,
    y_bins: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Count joint cells and retain exact member IDs, including unavailable facts.

    Numeric bins are left-closed/right-open except the final bin, which includes
    its right edge. Underflow and overflow remain explicit, preserving totals.
    """
    if x == y or x not in DIMENSIONS or y not in DIMENSIONS:
        raise ValueError("Select two distinct registered dimensions.")
    edges = list(y_bins) if y_bins is not None else None
    if edges is not None and (
        DIMENSIONS[y]["kind"] != "number"
        or len(edges) < 2
        or any(not math.isfinite(v) for v in edges)
        or any(a >= b for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError("Bins require finite, strictly increasing numeric edges.")

    def cell(
        r: Mapping[str, Any], key: str, bins: list[float] | None
    ) -> tuple[str, str]:
        state = _state(r, key)
        if state != "known":
            return (
                state,
                {"unknown": "未知", "missing": "缺失", "invalid": "不兼容"}[state],
            )
        value = r["dimensions"][key]["value"]
        if DIMENSIONS[key]["kind"] == "number":
            value = float(value)
            if value == 0.0:
                value = 0.0  # Canonicalize negative zero as well.
        if bins is None:
            return json.dumps(["known", value], ensure_ascii=False), str(value)
        if value < bins[0]:
            return "underflow", f"< {bins[0]:.4g}"
        if value > bins[-1]:
            return "overflow", f"> {bins[-1]:.4g}"
        for i, (a, b) in enumerate(zip(bins, bins[1:])):
            if a <= value < b or (i == len(bins) - 2 and value == b):
                return f"bin:{i}", f'[{a:.4g}, {b:.4g}{"]" if i==len(bins)-2 else ")"}'
        raise ValueError("Value cannot be assigned to bins.")

    groups = {}
    for r in records:
        a, la = cell(r, x, None)
        b, lb = cell(r, y, edges)
        token = (a, b)
        if token not in groups:
            groups[token] = dict(
                x=la,
                y=lb,
                x_key=x,
                y_key=y,
                x_token=a,
                y_token=b,
                y_bins=edges,
                definition_version=1,
                episode_ids=[],
                count=0,
            )
        groups[token]["episode_ids"].append(r["episode_id"])
        groups[token]["count"] += 1
    return list(groups.values())


def select_cell(
    records: Sequence[Mapping[str, Any]], cell: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Select exact cell members from the already-applied population."""
    ids = set(cell["episode_ids"])
    return [r for r in records if r["episode_id"] in ids]
