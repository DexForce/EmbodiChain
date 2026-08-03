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

"""Resolve semantic Action Engine arms to physical robot control parts."""

from __future__ import annotations

from typing import Any

__all__ = ["arm_control_part"]


def arm_control_part(env: Any, arm: str) -> str:
    """Return the physical arm control part for a semantic arm name."""
    if arm not in {"left_arm", "right_arm"}:
        raise ValueError(f"Expected a semantic arm, got {arm!r}.")
    if hasattr(env, "get_agent_arm_control_part"):
        part = env.get_agent_arm_control_part(arm == "left_arm")
        if part:
            return str(part)
    return arm
