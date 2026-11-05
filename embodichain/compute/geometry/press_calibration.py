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

"""Source release-bias drive arithmetic with explicit physical operands."""

from __future__ import annotations
import math

__all__ = ["source_press_release_bias"]


def source_press_release_bias(
    runtime_mass: float, source_stroke: float
) -> tuple[float, float, float]:
    """Retain the original spring/damping/effort calibration for source buttons.

    Args:
        runtime_mass: Positive source-compatible runtime moving-link mass.
        source_stroke: Positive original full travel in metres.

    Returns:
        Stiffness, critical damping and effort reserve. The values describe an
        explicitly generated release bias, not source mechanism identification.
    """
    if (
        not math.isfinite(runtime_mass)
        or runtime_mass <= 0
        or not math.isfinite(source_stroke)
        or source_stroke <= 0
    ):
        raise ValueError("Source Press calibration requires positive mass and stroke.")
    stiffness = max(500.0, runtime_mass * 9.81 / (0.04 * source_stroke))
    damping = 2.0 * math.sqrt(stiffness * runtime_mass)
    effort = max(5.0, 2.0 * stiffness * source_stroke)
    return stiffness, damping, effort
