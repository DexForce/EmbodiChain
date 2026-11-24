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
"""Reduce time-scoped physical measurements into terminal scalar observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__: list[str] = []

_OPERATIONS = frozenset(
    {"first", "last", "min", "max", "mean", "range", "displacement", "count"}
)
_UNITS = frozenset({"1", "m", "rad", "deg", "s"})


def _reduce_measurements(
    frames: Sequence[Mapping[str, Any]], reductions: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    results = {}
    for name, spec in reductions.items():
        first, last = spec.get("window", (0, len(frames) - 1))
        rows = frames[first : last + 1]
        operation = spec["operation"]
        if operation == "count":
            results[name] = float(len(rows))
            continue
        if spec.get("samples"):
            rows = [sample for row in rows for sample in row[spec["samples"]]]
        if spec.get("where"):
            rows = [
                row
                for row in rows
                if all(
                    lo <= row[field] <= hi for field, (lo, hi) in spec["where"].items()
                )
            ]
        values = [row[spec["field"]] for row in rows]
        if not values or any(value is None for value in values):
            continue
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            continue
        if spec.get("absolute"):
            array = np.abs(array)
        if operation == "first":
            value = array[0]
        elif operation == "last":
            value = array[-1]
        elif operation == "min":
            value = array.min()
        elif operation == "max":
            value = array.max()
        elif operation == "mean":
            value = array.mean()
        elif operation == "range":
            value = np.ptp(array)
        elif operation == "displacement":
            value = np.linalg.norm(array - array[0], axis=1).max()
        else:
            raise ValueError(f"Unknown measurement operation {operation!r}.")
        results[name] = float(value)
    return results


def _evaluate_scalar_checks(
    graph: Mapping[str, Any], measurements: Mapping[str, float]
) -> dict[str, bool]:
    """Evaluate reduced replay observations with the standard rule graph."""
    import torch

    from . import EvaluationSpec, EvaluationStatus, builtin_predicates

    compiled = builtin_predicates().compile(EvaluationSpec.from_dict(graph))
    observations = {
        f"measurements/{name}": torch.tensor(
            [[value]], dtype=torch.float64, device="cpu"
        )
        for name, value in measurements.items()
    }
    return {
        name: rule.status.item() == EvaluationStatus.SATISFIED
        for name, rule in compiled.evaluate(observations, batch_size=1).items()
    }
