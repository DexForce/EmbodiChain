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

"""Pure XY/yaw grid validation and symmetric pose quantization."""

from __future__ import annotations
import math
from collections.abc import Mapping
import numpy as np

__all__: list[str] = []


def _measurements(value: Mapping, grids: Mapping) -> dict:
    from copy import deepcopy

    if not isinstance(value, Mapping) or not value or set(value) != set(grids):
        raise ValueError("Named coverage measurements require matching grids")
    for name, declaration in value.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(declaration, Mapping)
            or set(declaration) - {"entity", "boundary", "segment", "fixed"}
            or not {"entity", "boundary"} <= declaration.keys()
            or not isinstance(declaration["entity"], str)
            or not declaration["entity"]
            or declaration["boundary"] not in ("initial", "terminal", "segment_end")
        ):
            raise ValueError("Invalid named coverage measurement")
        if (declaration["boundary"] == "segment_end") != ("segment" in declaration) or (
            "segment" in declaration
            and (
                not isinstance(declaration["segment"], str)
                or not declaration["segment"]
            )
        ):
            raise ValueError("Segment coverage requires an explicit segment name")
        if "fixed" in declaration and type(declaration["fixed"]) is not bool:
            raise ValueError("Coverage fixed flag must be boolean")
    return deepcopy(dict(value))


def _grid(value: Mapping) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "lower",
        "upper",
        "cell_size",
        "yaw_period",
    }:
        raise ValueError("Pose grids require lower, upper, cell_size and yaw_period")
    lower, upper, size = (
        np.asarray(value[k], dtype=float) for k in ("lower", "upper", "cell_size")
    )
    period = value["yaw_period"]
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (lower, upper, size)):
        raise ValueError("Pose grid coordinates must be three finite numbers")
    if type(period) not in (int, float) or not math.isfinite(period) or period <= 0:
        raise ValueError("yaw_period must be finite and positive")
    if period > 2 * math.pi or not math.isclose(
        2 * math.pi / period, round(2 * math.pi / period), rel_tol=0, abs_tol=1e-8
    ):
        raise ValueError("yaw_period must divide a complete turn")
    if (
        np.any(upper <= lower)
        or np.any(size <= 0)
        or not math.isclose(upper[2] - lower[2], period)
    ):
        raise ValueError("Pose grids need positive extents and one complete yaw period")
    bins = (upper - lower) / size
    if not np.allclose(bins, np.round(bins), rtol=0, atol=1e-8):
        raise ValueError("Pose grid extents must be divisible by cell_size")
    return {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "cell_size": size.tolist(),
        "yaw_period": float(period),
    }


def _pose_cell(pose, grid: Mapping) -> tuple[int, int, int] | None:
    value = np.asarray(pose, dtype=float)
    if value.shape == (4, 4):
        x, y = value[:2, 3]
        yaw = math.atan2(value[1, 0], value[0, 0])
    elif value.ndim == 1 and value.shape[0] >= 7:
        x, y = value[:2]
        q = value[3:7]
        norm = np.linalg.norm(q)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Pose quaternion must be finite and nonzero")
        qx, qy, qz, qw = q / norm
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    else:
        raise ValueError("Expected a pose matrix or xyz/xyzw state")
    if not np.isfinite(value).all():
        raise ValueError("Pose must be finite")
    lo, hi, size = (np.asarray(grid[k]) for k in ("lower", "upper", "cell_size"))
    point = np.array([x, y, (yaw - lo[2]) % grid["yaw_period"] + lo[2]])
    if np.any(point[:2] < lo[:2] - 1e-7) or np.any(point[:2] > hi[:2] + 1e-7):
        return None
    bins = np.rint((hi - lo) / size).astype(int)
    return tuple(
        np.minimum(
            bins - 1, np.maximum(0, np.floor((point - lo) / size).astype(int))
        ).tolist()
    )
