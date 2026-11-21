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

"""Explicit nominal hand dimensions separate from physical collision geometry."""

from __future__ import annotations
from dataclasses import dataclass
import math

__all__ = ["NominalHandEnvelope"]


@dataclass(frozen=True, slots=True)
class NominalHandEnvelope:
    """Caller-owned nominal dimensions in metres; no robot/model defaults.

    These operands preserve existing release/grasp clearance heuristics. Actual
    URDF/native collision surfaces remain independently qualified by their owners.
    """

    finger_length: float
    finger_width: float
    finger_thickness: float
    palm_depth: float
    opening_margin: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive length.")
