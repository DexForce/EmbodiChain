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

"""Auditable fixed-clock resampling with exact source event anchors."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import numpy as np

__all__ = ["AnchoredTimeMap", "resample_with_anchors"]


def _owned(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class AnchoredTimeMap:
    """Bidirectional timing evidence and source interpolation brackets.

    ``mapped_source_times`` maps each original sample onto execution time;
    ``source_query_times`` maps each execution sample back onto original time.
    All times are relative seconds. Anchors refer to samples, not commands.

    Args:
        source_times: Original zero-based clock.
        mapped_source_times: Execution time assigned to each source sample.
        execution_times: Zero-based destination control clock.
        source_query_times: Original time queried by each destination sample.
        left_indices: Left source interpolation bracket per destination sample.
        right_indices: Right source interpolation bracket per destination sample.
        weights: Interpolation fractions within the brackets.
        anchor_source_indices: Ordered source samples that must be preserved.
        anchor_execution_indices: Corresponding exact destination sample indices.
    """

    source_times: np.ndarray
    mapped_source_times: np.ndarray
    execution_times: np.ndarray
    source_query_times: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray
    weights: np.ndarray
    anchor_source_indices: np.ndarray
    anchor_execution_indices: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            object.__setattr__(self, name, _owned(value))
        n, m = len(self.source_times), len(self.execution_times)
        if n < 2 or m < 2 or len(self.mapped_source_times) != n:
            raise ValueError("Time map needs nonempty source and execution clocks")
        if any(
            len(getattr(self, k)) != m
            for k in ("source_query_times", "left_indices", "right_indices", "weights")
        ):
            raise ValueError("Execution mapping arrays must have matching lengths")
        if (
            np.any(np.diff(self.source_times) <= 0)
            or np.any(np.diff(self.execution_times) <= 0)
            or np.any(np.diff(self.mapped_source_times) <= 0)
        ):
            raise ValueError("Source and mapped clocks must strictly increase")
        for indices, bound in [
            (self.left_indices, n),
            (self.right_indices, n),
            (self.anchor_source_indices, n),
            (self.anchor_execution_indices, m),
        ]:
            if (
                indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices >= bound)
            ):
                raise ValueError("Time map indices must be in-range integers")
        if (
            len(self.anchor_source_indices) != len(self.anchor_execution_indices)
            or np.any(self.left_indices > self.right_indices)
            or np.any((self.weights < 0) | (self.weights > 1))
        ):
            raise ValueError("Invalid interpolation or anchor mapping")
        a, b = self.anchor_source_indices, self.anchor_execution_indices
        if (
            len(a) < 2
            or a[0] != 0
            or a[-1] != n - 1
            or b[0] != 0
            or b[-1] != m - 1
            or np.any(a[1:] <= a[:-1])
            or np.any(b[1:] <= b[:-1])
        ):
            raise ValueError(
                "Anchors must be strictly ordered and include both endpoints"
            )
        if (
            self.source_times[0] != 0
            or self.execution_times[0] != 0
            or np.any(np.diff(self.source_query_times) <= 0)
        ):
            raise ValueError(
                "Relative source and execution clocks must start at zero and advance"
            )
        reconstructed = self.source_times[self.left_indices] + self.weights * (
            self.source_times[self.right_indices] - self.source_times[self.left_indices]
        )
        if (
            not np.allclose(
                reconstructed, self.source_query_times, atol=1e-12, rtol=1e-10
            )
            or not np.allclose(
                self.source_query_times[b], self.source_times[a], atol=1e-12, rtol=1e-10
            )
            or not np.allclose(
                self.mapped_source_times[a],
                self.execution_times[b],
                atol=1e-12,
                rtol=1e-10,
            )
        ):
            raise ValueError(
                "Time mappings disagree with interpolation brackets or hard anchors"
            )


def resample_with_anchors(
    positions: np.ndarray,
    source_times: np.ndarray,
    control_dt: float,
    anchor_indices: Sequence[int],
    *,
    max_anchor_delay: float | None = None,
    max_samples: int = 1_000_000,
) -> tuple[np.ndarray, AnchoredTimeMap]:
    """Resample a path while preserving ordered hard event samples exactly.

    Anchors are rounded forward to control ticks. Between anchors a monotone
    piecewise-affine time map retains source path timing. The first/last sample
    are always anchors. If two events cannot fit within the permitted timing
    error, fail instead of deleting a phase or silently extending its delay.

    Args:
        positions: Finite floating ``(N,D)`` source positions.
        source_times: Strictly increasing ``(N,)`` seconds, starting at zero.
        control_dt: Positive destination control period; physics is untouched.
        anchor_indices: Explicit hard source sample indices.
        max_anchor_delay: Maximum delay per anchor; defaults to one control tick.
        max_samples: Maximum output allocation, checked before creating the grid.

    Returns:
        Resampled positions and an owned, bidirectional time/index mapping.
    """
    q = np.asarray(positions)
    t = np.asarray(source_times, dtype=np.float64)
    if (
        q.ndim != 2
        or q.dtype.kind != "f"
        or min(q.shape) < 1
        or len(q) < 2
        or not np.isfinite(q).all()
        or t.shape != (len(q),)
        or not np.isfinite(t).all()
        or t[0] != 0
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Expected finite (N>=2,D) positions and a strictly increasing zero-based clock"
        )
    if (
        type(control_dt) not in (float, int)
        or not math.isfinite(control_dt)
        or control_dt <= 0
    ):
        raise ValueError("control_dt must be positive and finite")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be an integer >= 2")
    delay = float(control_dt) if max_anchor_delay is None else max_anchor_delay
    if type(delay) not in (float, int) or not math.isfinite(delay) or delay < 0:
        raise ValueError("max_anchor_delay must be finite and nonnegative")
    raw_anchors = tuple(anchor_indices)
    if any(type(i) is not int or not 0 <= i < len(q) for i in raw_anchors):
        raise ValueError("Anchors must be in-range integer sample indices")
    anchors = np.array(sorted({0, len(q) - 1, *raw_anchors}), dtype=np.int64)
    ticks = []
    for source_index in anchors:
        ratio = float(t[source_index] / control_dt)
        if not math.isfinite(ratio) or ratio >= max_samples:
            raise ValueError("Execution grid exceeds max_samples")
        nearest = round(ratio)
        tick = (
            nearest
            if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-8)
            else math.ceil(ratio)
        )
        tick = max(ticks[-1] + 1, tick) if ticks else 0
        if tick >= max_samples:
            raise ValueError("Execution grid exceeds max_samples")
        if tick * control_dt - t[source_index] > delay + 1e-10:
            raise ValueError(
                "Required anchors cannot fit the control clock within max_anchor_delay"
            )
        ticks.append(tick)
    target_anchors = np.array(ticks, dtype=np.int64)
    anchor_times = target_anchors * control_dt
    grid = np.arange(ticks[-1] + 1, dtype=np.float64) * control_dt
    mapped = np.interp(t, t[anchors], anchor_times)
    query = np.interp(grid, anchor_times, t[anchors])
    upper = np.searchsorted(t, query, side="right").clip(1, len(t) - 1)
    lower = upper - 1
    weights = ((query - t[lower]) / (t[upper] - t[lower])).clip(0.0, 1.0)
    result = q[lower] + weights[:, None] * (q[upper] - q[lower])
    result = result.astype(q.dtype)
    # Restore anchors by assignment, not floating interpolation.
    result[target_anchors] = q[anchors]
    query[target_anchors] = t[anchors]
    lower[target_anchors] = anchors
    upper[target_anchors] = anchors
    weights[target_anchors] = 0.0
    return result, AnchoredTimeMap(
        t, mapped, grid, query, lower, upper, weights, anchors, target_anchors
    )
