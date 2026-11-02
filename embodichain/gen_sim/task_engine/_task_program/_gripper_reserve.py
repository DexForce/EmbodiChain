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

"""GenSim's nominal hand reserves, separate from physical collision geometry."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class NominalHandReserve:
    """Historical task heuristics in metres; never a mesh collision certificate."""

    finger_length: float
    finger_width: float
    finger_thickness: float
    palm_depth: float
    opening_margin: float


def nominal_hand_reserve(model_id: str) -> NominalHandReserve:
    """Retain the generated Robotiq task's explicitly owned nominal reserves.

    GraspKit checks the complete fixed-opening URDF collision surfaces. These
    additional task-specific clearance and scale heuristics do not replace that
    check and are intentionally absent from its public model/configuration.
    """
    if model_id != "robotiq_arg2f_140":
        raise ValueError(
            f"GenSim has no nominal hand-reserve calibration for {model_id!r}."
        )
    return NominalHandReserve(0.13, 0.03, 0.01, 0.08, 0.003)
