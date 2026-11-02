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

"""Robotiq_V2 2F-85 linkage coordinates and measured-aperture conversion."""

from __future__ import annotations

import numpy as np

__all__ = ["robotiq85_joint_positions"]

_SIGNS = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])


def robotiq85_joint_positions(closed_fraction: np.ndarray) -> np.ndarray:
    """Convert measured normalized aperture to the selected 2F-85 joint layout.

    DROID uses ``width = 0.085 * (1 - closed_fraction)``. The standard Robotiq
    linkage has ``width = 0.010 + 0.1143*sin(theta_zero - theta)``; calibration
    uses the open endpoint and clips only to the URDF's 0.8 rad closed stop.
    Returned columns follow the preset's explicit ``eef`` control part. This
    converts observed aperture independently of any closing command.
    The layout belongs to the shared ``Robotiq`` archive; the separately
    registered ``Robotiq2F85`` asset requires independent qualification.

    Args:
        closed_fraction: Finite normalized closed fractions in [0, 1].

    Returns:
        Float64 joint positions with shape ``(*closed_fraction.shape, 6)``.
    """
    fraction = np.asarray(closed_fraction, dtype=np.float64)
    if not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("Robotiq closed fraction must lie in [0, 1]")
    zero = np.arcsin((0.085 - 0.010) / 0.1143)
    angle = np.minimum(zero - np.arcsin((0.085 * (1 - fraction) - 0.010) / 0.1143), 0.8)
    return angle[..., None] * _SIGNS
