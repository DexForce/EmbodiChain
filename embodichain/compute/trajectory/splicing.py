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

"""Fixed-clock joint replacement with unchanged boundary derivative stencils."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["splice_joint_segment", "joint_path_diagnostics"]


def _path(qpos: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, t = np.asarray(qpos), np.asarray(timestamps, dtype=np.float64)
    if (
        q.ndim != 2
        or q.dtype.kind != "f"
        or len(q) < 3
        or t.shape != (len(q),)
        or not np.isfinite(q).all()
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Expected finite floating (T,D) joints and increasing (T,) times."
        )
    return q, t


def splice_joint_segment(
    original: np.ndarray,
    replacement: np.ndarray,
    timestamps: np.ndarray,
    *,
    start: int,
    stop: int,
    joint_indices: tuple[int, ...],
    blend_samples: int = 12,
    anchors: tuple[int, ...] = (),
) -> NDArray[np.floating]:
    """Replace a half-open segment on its original clock.

    A quintic correction envelope preserves three original samples at both
    ends. Thus the three-point position/velocity/acceleration stencils crossing
    the outer joins are identical to the original. This is a discrete C2
    boundary contract, not a claim that noisy observations are a smooth curve.
    Anchor rows use the native replacement exactly. Callers must independently
    validate TCP/contact constraints and limits after blending.

    Args:
        original: Full source joints, shaped (T,D).
        replacement: Native reconstructed segment, shaped (stop-start,D).
        timestamps: Source sample times; never retimed.
        start: First included source sample.
        stop: First excluded source sample.
        joint_indices: Explicit editable columns.
        blend_samples: Width of each quintic boundary connection.
        anchors: Absolute source indices that must retain the native solution.

    Returns:
        An owned array with source dtype and unchanged values outside the mask.
    """
    q, _ = _path(original, timestamps)
    count = stop - start
    candidate = np.asarray(replacement)
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= len(q)
        or count < 12
        or candidate.shape != (count, q.shape[1])
        or not np.isfinite(candidate).all()
        or type(blend_samples) is not int
        or blend_samples < 4
        or not joint_indices
        or len(set(joint_indices)) != len(joint_indices)
        or any(type(i) is not int or not 0 <= i < q.shape[1] for i in joint_indices)
    ):
        raise ValueError("Invalid replacement, editable columns, or connection width.")
    width = min(blend_samples, (count - 6) // 2)
    interior_anchors = [a - start for a in anchors if start + 3 <= a < stop - 3]
    if interior_anchors:
        width = min(width, min(interior_anchors) - 3, count - 4 - max(interior_anchors))
    if width < 3:
        raise ValueError("Boundary connection overlaps a protected skill anchor.")
    u = np.linspace(0.0, 1.0, width + 1)
    ramp = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
    weight = np.ones(count)
    weight[:3] = weight[-3:] = 0.0
    weight[3 : 3 + width + 1] = ramp
    weight[-3 - width - 1 : -3] = ramp[::-1]
    ids = list(joint_indices)
    for anchor in anchors:
        if type(anchor) is not int or not start <= anchor < stop:
            raise ValueError("Anchor is outside the selected segment.")
        index = anchor - start
        if index < 3 or index >= count - 3:
            if not np.allclose(
                candidate[index, ids], q[anchor, ids], atol=1e-6, rtol=0
            ):
                raise ValueError(
                    "Native anchor conflicts with the immutable boundary stencil."
                )
        else:
            # A hard anchor cannot introduce an isolated blend-weight jump.
            if weight[index] < 1.0 - 1e-12:
                raise ValueError(
                    "Boundary connection overlaps a protected skill anchor."
                )
    result = q.copy()
    source = q[start:stop, ids]
    result[start:stop, ids] = source + weight[:, None] * (
        candidate[:, ids].astype(np.float64) - source
    )
    # Preserve these bytes, including floating signed zero.
    result[start : start + 3, ids] = q[start : start + 3, ids]
    result[stop - 3 : stop, ids] = q[stop - 3 : stop, ids]
    return result


def joint_path_diagnostics(
    qpos: np.ndarray,
    timestamps: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
) -> dict[str, object]:
    """Measure finite-grid limits using physical timestamps.

    Args:
        qpos: Full joint reference, shaped (T,D).
        timestamps: Strictly increasing physical sample times.
        lower: Per-joint position lower bounds.
        upper: Per-joint position upper bounds.
        velocity_limits: Positive per-joint speed caps.
        acceleration_limits: Positive per-joint acceleration caps.

    Returns:
        Limit acceptance and worst per-joint derivative magnitudes. Collision
        and physical effect verification are intentionally absent.
    """
    q, t = _path(qpos, timestamps)
    arrays = [
        np.asarray(v, dtype=np.float64)
        for v in (lower, upper, velocity_limits, acceleration_limits)
    ]
    if any(v.shape != (q.shape[1],) or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Limits must be finite per-joint vectors.")
    lo, hi, speed, accel = arrays
    if np.any(lo >= hi) or np.any(speed <= 0) or np.any(accel <= 0):
        raise ValueError("Position intervals and derivative caps must be positive.")
    # Edge velocities cannot hide a single-step jump through central averaging.
    edge_v = np.diff(q.astype(np.float64), axis=0) / np.diff(t)[:, None]
    edge_a = np.diff(edge_v, axis=0) / (
        (np.diff(t)[1:] + np.diff(t)[:-1])[:, None] * 0.5
    )
    peak_v = np.max(np.abs(edge_v), axis=0)
    peak_a = np.max(np.abs(edge_a), axis=0)
    return {
        "accepted": bool(
            np.all(q >= lo - 1e-6)
            and np.all(q <= hi + 1e-6)
            and np.all(peak_v <= speed * (1 + 1e-5))
            and np.all(peak_a <= accel * (1 + 1e-5))
        ),
        "position_violations": int(np.count_nonzero((q < lo - 1e-6) | (q > hi + 1e-6))),
        "peak_velocity": peak_v.tolist(),
        "peak_acceleration": peak_a.tolist(),
        "velocity_ratio": float(np.max(peak_v / speed)),
        "acceleration_ratio": float(np.max(peak_a / accel)),
    }
