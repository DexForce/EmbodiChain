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

"""Validated, explicit spatial draws for reproducible boundary experiments."""

from __future__ import annotations
from collections.abc import Mapping
import math

__all__: list[str] = []


def validate_samples(
    joints: object, knots: object, via_count: int
) -> tuple[dict[str, float], tuple[float, ...]]:
    if not isinstance(joints, Mapping) or any(
        not isinstance(k, str) or not k for k in joints
    ):
        raise ValueError("joint_samples must map joint names to signed fractions")
    if not isinstance(knots, (tuple, list)):
        raise ValueError("knot_samples must be a sequence")
    values = tuple(joints.values()) + tuple(knots)
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1
        for v in values
    ):
        raise ValueError("explicit spatial samples must be finite and in [-1,1]")
    if knots and len(knots) != via_count:
        raise ValueError("knot_samples must match via_count")
    return {k: float(v) for k, v in joints.items()}, tuple(float(v) for v in knots)
