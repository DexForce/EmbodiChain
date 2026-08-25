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

"""Shared binding-contract declarations for built-in manipulation skills."""

from __future__ import annotations

from collections.abc import Mapping

from ..control import ControlCommand
from ..requirements import (
    DisjointSlotEndpoints,
    GRASP_CAPABILITY,
    SkillEndpointRequirement,
    SkillResourceSlot,
)


def make_motion_slot(
    role: str,
    *,
    capabilities: frozenset[str],
) -> SkillResourceSlot:
    """Build one motion-endpoint resource slot."""
    return SkillResourceSlot(
        slot_id=role,
        endpoints=(
            SkillEndpointRequirement(
                endpoint_id="motion",
                capabilities=capabilities,
            ),
        ),
    )


def make_manipulation_slot(
    role: str,
    *,
    motion_capabilities: frozenset[str],
    grasp_commands: Mapping[str, type[ControlCommand]],
) -> SkillResourceSlot:
    """Build one disjoint motion-and-grasp participant slot."""
    return SkillResourceSlot(
        slot_id=role,
        endpoints=(
            SkillEndpointRequirement(
                endpoint_id="motion",
                capabilities=motion_capabilities,
            ),
            SkillEndpointRequirement(
                endpoint_id="grasp",
                capabilities=frozenset({GRASP_CAPABILITY}),
                required_commands=grasp_commands,
            ),
        ),
        constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
    )


__all__ = ["make_manipulation_slot", "make_motion_slot"]
