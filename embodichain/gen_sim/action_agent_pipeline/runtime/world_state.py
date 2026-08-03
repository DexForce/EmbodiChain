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

"""World-state helpers for arm-local Action Agent runtime slots."""

from __future__ import annotations

from embodichain.lab.sim.atomic_actions import HeldObjectState, WorldState

__all__ = ["get_arm_local_held_object"]


def get_arm_local_held_object(state: WorldState) -> HeldObjectState | None:
    """Return the held object from an arm-local runtime state.

    The Action Agent keeps a separate ``WorldState`` per semantic arm. A state
    with multiple physical-arm entries violates that runtime invariant and is
    rejected instead of selecting an arbitrary object.
    """
    if not state.held_objects:
        return None
    if len(state.held_objects) != 1:
        raise ValueError(
            "An arm-local WorldState must contain at most one held object, "
            f"got control parts {sorted(state.held_objects)}."
        )
    return next(iter(state.held_objects.values()))
