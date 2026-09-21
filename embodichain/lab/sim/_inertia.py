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
"""Mass-property conversion at the DexSim descriptor boundary."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


def _principal_inertia_matrix(
    inertia: np.ndarray, quaternion_xyzw: np.ndarray
) -> np.ndarray:
    """Express principal moments in body axes about the same COM."""
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return np.asarray((rotation * inertia) @ rotation.T, dtype=np.float32)
