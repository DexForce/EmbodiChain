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

"""Explicit terminal and phase-qualified semantic effect declarations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.task_program.compiler.lowering import RegisteredSemanticEffect

__all__ = ["PlacementEffectContract", "TransportEffectContract"]


class PlacementEffectContract(str, Enum):
    """Choose an existing release qualification contract, without changing motion."""

    PHASED_RELEASE = "phased_release"
    TERMINAL_RELEASE = "terminal_release"


class TransportEffectContract(str, Enum):
    """Choose transport-phase retention or the original terminal attachment proof."""

    GUARDED_ATTACHMENT = "guarded_attachment"
    TERMINAL_ATTACHMENT = "terminal_attachment"


def _release_effect(
    object_id: str, contract: PlacementEffectContract
) -> RegisteredSemanticEffect:
    from embodichain.lab.task_program.compiler.lowering import (
        RegisteredSemanticEffect,
        RegisteredHeldObjectEffect,
    )
    from embodichain.lab.task_program.semantics import (
        HeldObjectRelation,
        SemanticEffectKind,
    )

    if not isinstance(contract, PlacementEffectContract):
        raise TypeError("Release contract must be a typed PlacementEffectContract.")
    phased = contract is PlacementEffectContract.PHASED_RELEASE
    return RegisteredSemanticEffect(
        effect_kind=SemanticEffectKind.RELEASE,
        held_objects=(
            RegisteredHeldObjectEffect(
                expectation_id="primary",
                relation=HeldObjectRelation.DETACHED,
                object_id=object_id,
                slot_id="primary",
                release_before_segment="retract" if phased else None,
                retain_during_segments=("approach",) if phased else (),
            ),
        ),
    )


def _attachment_effect(
    object_id: str, contract: TransportEffectContract
) -> RegisteredSemanticEffect:
    from embodichain.lab.task_program.compiler.lowering import (
        RegisteredSemanticEffect,
        RegisteredHeldObjectEffect,
    )
    from embodichain.lab.task_program.semantics import (
        HeldObjectRelation,
        SemanticEffectKind,
    )

    if not isinstance(contract, TransportEffectContract):
        raise TypeError("Transport contract must be a typed TransportEffectContract.")
    return RegisteredSemanticEffect(
        effect_kind=SemanticEffectKind.ATTACH,
        held_objects=(
            RegisteredHeldObjectEffect(
                expectation_id="primary",
                relation=HeldObjectRelation.ATTACHED,
                object_id=object_id,
                slot_id="primary",
                retain_during_segments=(
                    ("transport",)
                    if contract is TransportEffectContract.GUARDED_ATTACHMENT
                    else ()
                ),
            ),
        ),
    )
