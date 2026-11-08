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

"""Shared numerical comparisons and stability accumulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__: list[str] = []


def _within_target(value: Any, target: Any, tolerance: float) -> Any:
    return abs(value - target) <= tolerance


def _temporal_window_decision(
    mode: str, *, matched: Any, violated: Any, closing: Any, missing: Any
) -> tuple[Any, Any]:
    """Return satisfaction and failure masks for a scalar or batched control slot.

    Match and violation inputs refer to the current qualified slot. For a hold,
    the match additionally requires the caller's consecutive count. Missing
    evidence prevents a decision by exhaustion, while a witness for eventual
    success or an observed violation of always is conclusive immediately.
    """
    complete = closing & (missing ^ True)
    if mode == "always":
        return matched & complete, violated
    return matched, complete & (matched ^ True)


def _compare(observed: Any, expected: Any, operator: str) -> bool:
    if operator == "eq":
        return observed == expected
    if operator == "ne":
        return observed != expected
    if operator == "gt":
        return observed > expected
    if operator == "gte":
        return observed >= expected
    if operator == "lt":
        return observed < expected
    if operator == "lte":
        return observed <= expected
    if operator == "in":
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise TypeError("operator='in' expects a sequence as expected value")
        return observed in expected
    if operator == "contains":
        return expected in observed
    if operator == "approx":
        if not isinstance(observed, (int, float)) or isinstance(observed, bool):
            raise TypeError("operator='approx' requires a numeric observation")
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            raise TypeError("operator='approx' requires a numeric expected value")
        return math.isclose(
            float(observed), float(expected), rel_tol=1e-6, abs_tol=1e-6
        )
    raise ValueError(f"unsupported predicate operator {operator!r}")


def _measure_lift(pose, baseline, minimum):
    import torch

    lift = pose[:, 2, 3] - baseline[:, 2, 3]
    valid = torch.isfinite(pose).all(dim=(1, 2)) & torch.isfinite(baseline).all(
        dim=(1, 2)
    )
    return lift >= minimum, valid, lift


def _measure_relative_position(pose, reference, offset, tolerances, *, parent_axes):
    import torch

    displacement = pose[:, :3, 3] - reference[:, :3, 3]
    if parent_axes:
        displacement = torch.bmm(
            reference[:, :3, :3].transpose(1, 2), displacement.unsqueeze(-1)
        ).squeeze(-1)
    error = torch.abs(displacement - offset)
    valid = torch.isfinite(pose).all(dim=(1, 2)) & torch.isfinite(reference).all(
        dim=(1, 2)
    )
    passed = valid.clone()
    for axis, tolerance in enumerate(tolerances):
        if tolerance is not None:
            passed &= error[:, axis] < tolerance
    return passed, valid, error


def _measure_upright(pose, maximum_tilt):
    import torch

    tilt = torch.rad2deg(torch.acos(pose[:, 2, 2].clamp(-1.0, 1.0)))
    return tilt < maximum_tilt, torch.isfinite(pose).all(dim=(1, 2)), tilt


def _speeds_within(linear, angular, linear_limit, angular_limit):
    return (linear <= linear_limit) & (angular <= angular_limit)


def _advance_hold(previous, stable, *, increment=1, maximum=None):
    import torch

    count = previous + increment
    if maximum is not None:
        count = count.clamp(max=maximum)
    return torch.where(stable, count, 0)


def _inside_region(position, lower, upper):
    return ((position >= lower) & (position <= upper)).all(dim=-1)


def _evaluation_pose_matrix(pose):
    from embodichain.utils.math import matrix_from_quat

    matrix = pose.new_zeros((len(pose), 4, 4))
    matrix[:, :3, :3] = matrix_from_quat(pose[:, 3:])
    matrix[:, :3, 3] = pose[:, :3]
    matrix[:, 3, 3] = 1
    return matrix


def _corners_in_frame(pose, frame, corners):
    import torch
    from embodichain.utils.math import matrix_from_quat

    world = (
        torch.einsum("bij,kj->bki", matrix_from_quat(pose[:, 3:]), corners)
        + pose[:, None, :3]
    )
    return torch.einsum(
        "bji,bkj->bki",
        matrix_from_quat(frame[:, 3:]),
        world - frame[:, None, :3],
    )
