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

"""Typed, composable motion-policy references persisted in symbolic graphs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

__all__ = [
    "MOTION_MODIFIER_MODES",
    "MOTION_POLICY_VERSION",
    "motion_policy",
    "validate_motion_policy",
]

MOTION_POLICY_VERSION: Final = "action_engine_motion_policy_v3"
MOTION_MODIFIER_MODES: Final = {
    "orientation": frozenset({"upright"}),
    "handover_role": frozenset({"transfer"}),
}

_POLICY_KEYS = frozenset({"modifiers"})
_MODIFIER_KEYS = frozenset({"type", "mode"})


def motion_policy(*modifiers: tuple[str, str]) -> dict[str, Any]:
    """Build one canonical policy reference from typed modifier pairs."""
    return validate_motion_policy(
        {
            "modifiers": [
                {"type": modifier_type, "mode": mode}
                for modifier_type, mode in modifiers
            ]
        }
    )


def validate_motion_policy(
    value: Any,
    context: str = "motion_policy",
) -> dict[str, Any]:
    """Validate and detach one symbolic motion-policy reference."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{context} must be a mapping with typed modifiers; named string "
            "policies are no longer supported. Regenerate the graph."
        )
    if set(value) != _POLICY_KEYS:
        raise ValueError(f"{context} fields must be {sorted(_POLICY_KEYS)}.")
    raw_modifiers = value.get("modifiers")
    if not isinstance(raw_modifiers, (list, tuple)):
        raise ValueError(f"{context}.modifiers must be a sequence.")

    modifiers: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_types: set[str] = set()
    for index, raw_modifier in enumerate(raw_modifiers):
        modifier_context = f"{context}.modifiers[{index}]"
        if not isinstance(raw_modifier, Mapping):
            raise ValueError(f"{modifier_context} must be a mapping.")
        if set(raw_modifier) != _MODIFIER_KEYS:
            raise ValueError(
                f"{modifier_context} fields must be {sorted(_MODIFIER_KEYS)}."
            )
        modifier_type = raw_modifier.get("type")
        mode = raw_modifier.get("mode")
        if not isinstance(modifier_type, str) or not modifier_type:
            raise ValueError(f"{modifier_context}.type must be a non-empty string.")
        if modifier_type not in MOTION_MODIFIER_MODES:
            raise ValueError(
                f"{modifier_context}.type {modifier_type!r} is unsupported; "
                f"expected one of {sorted(MOTION_MODIFIER_MODES)}."
            )
        if (
            not isinstance(mode, str)
            or mode not in MOTION_MODIFIER_MODES[modifier_type]
        ):
            raise ValueError(
                f"{modifier_context}.mode {mode!r} is unsupported for "
                f"{modifier_type!r}; expected one of "
                f"{sorted(MOTION_MODIFIER_MODES[modifier_type])}."
            )
        key = (modifier_type, mode)
        if key in seen:
            raise ValueError(f"{modifier_context} duplicates modifier {key!r}.")
        if modifier_type in seen_types:
            raise ValueError(
                f"{context} may select only one mode for modifier type "
                f"{modifier_type!r}."
            )
        seen.add(key)
        seen_types.add(modifier_type)
        modifiers.append({"type": modifier_type, "mode": mode})

    return {"modifiers": deepcopy(modifiers)}
