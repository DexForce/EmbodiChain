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

"""Action Engine-specific adapters for mainline atomic actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from embodichain.lab.sim.atomic_actions import (
    MoveHeldObject,
    MoveHeldObjectOptions,
)

__all__ = ["ExactTargetMoveHeldObject", "ExactTargetMoveHeldObjectOptions"]


@dataclass(frozen=True, slots=True, eq=False)
class ExactTargetMoveHeldObjectOptions(MoveHeldObjectOptions):
    """Action Engine transport options for a grounded object target."""


class ExactTargetMoveHeldObject(MoveHeldObject):
    """Preserve a selected semantic orientation when explicitly requested."""

    OptionsType = ExactTargetMoveHeldObjectOptions
    binding_contract = MoveHeldObject.binding_contract

    def _apply_automatic_transport_rotation(
        self,
        move_eef_xpos: torch.Tensor,
        end_arm_xpos: torch.Tensor,
    ) -> None:
        """Keep target shaping in GenSim grounding and free-yaw search."""
        del move_eef_xpos, end_arm_xpos
