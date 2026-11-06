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

"""Pure declarative recipes; physical policy qualification belongs to Lab."""

from __future__ import annotations

from typing import Any

__all__: list[str] = []

SIDE_RELATIONS = frozenset(
    {
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "front_right_of",
        "back_left_of",
        "back_right_of",
    }
)
PARK_CALL = "gen_sim.articulation_park"
SLIDE_CALL = "gen_sim.articulation_slide"
WITHDRAW_CALL = "gen_sim.articulation_withdraw"
TWIST_PREPARE_CALL = "gen_sim.twist_prepare"
TWIST_CALL = "gen_sim.twist"
PRESS_PREPARE_CALL = "gen_sim.press_prepare"
PRESS_CALL = "gen_sim.press"


def slide_recipe(
    object_id: str, state: str, arm: str, part_id: str | None = None
) -> list[dict[str, Any]]:
    if state not in {"open", "closed"} or arm not in {"left", "right"}:
        raise ValueError("GenSim E6 requires an open/closed target and one arm.")
    arguments = {"object": object_id, "state": state}
    if part_id is not None:
        arguments["part"] = part_id
    return [
        {
            "kind": "registered",
            "call_id": call_id,
            "arguments": arguments,
            "resources": {"primary": arm},
        }
        for call_id in (SLIDE_CALL, WITHDRAW_CALL)
    ] + [
        {
            "kind": "registered",
            "call_id": PARK_CALL,
            "arguments": {},
            "resources": {"primary": arm},
        }
    ]


def press_recipe(
    object_id: str, arm: str, *, operation_id: str = ""
) -> list[dict[str, Any]]:
    if arm not in {"left", "right"}:
        raise ValueError("E9 requires one explicit arm.")
    return [
        {
            "kind": "registered",
            "call_id": f"{call}.{operation_id}" if operation_id else call,
            "arguments": {"object": object_id},
            "resources": {"primary": arm},
        }
        for call in (PRESS_PREPARE_CALL, PRESS_CALL)
    ] + [
        {
            "kind": "registered",
            "call_id": PARK_CALL,
            "arguments": {},
            "resources": {"primary": arm},
        }
    ]


def twist_recipe(
    object_id: str,
    arm: str,
    setting: int,
    *,
    chunk_count: int,
    operation_id: str = "",
) -> list[dict]:
    return (
        [
            {
                "kind": "registered",
                "call_id": f"{call}.{operation_id}" if operation_id else call,
                "arguments": {"object": object_id, "setting": setting},
                "resources": {"primary": arm},
            }
            for call in (TWIST_PREPARE_CALL,)
        ]
        + [
            {
                "kind": "registered",
                "call_id": (
                    f"{TWIST_CALL}.{operation_id}" if operation_id else TWIST_CALL
                ),
                "arguments": {"object": object_id, "setting": setting},
                "resources": {"primary": arm},
            }
            for _ in range(chunk_count)
        ]
        + [
            {
                "kind": "registered",
                "call_id": PARK_CALL,
                "arguments": {},
                "resources": {"primary": arm},
            }
        ]
    )


def required_chunk_count(target_qpos: float) -> int:
    """Read occurrence coverage from the explicit migrated physical policy."""
    from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
        TwistPlanningRecovery,
        required_twist_occurrences,
    )

    return required_twist_occurrences(target_qpos, TwistPlanningRecovery())
