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

"""Annotation choices without importing simulation or optional fitting dependencies."""

from __future__ import annotations

__all__ = []

# Annotation categories mirror BUILTIN_SKILL_TYPES. Listing them must also work
# in installations containing only the image annotation dependencies.
SKILL_CHOICES: tuple[tuple[str, str], ...] = (
    ("pick_up", "Pickup"),
    ("twist", "Twist"),
    ("slide", "Slide"),
    ("hand_over", "HandOver"),
    ("execute_demo_skill", "DemoSkill"),
    ("place", "Place"),
    ("pour", "Pour"),
    ("move_end_effector", "MoveEndEffector"),
    ("move_joints", "MoveJoints"),
    ("move_held_object", "MoveHeldObject"),
    ("axis_align", "AxisAlign"),
    ("push_object", "PushObject"),
    ("press", "Press"),
    ("open_door", "OpenDoor"),
    ("coordinated_pickment", "CoordinatedPickment"),
    ("coordinated_placement", "CoordinatedPlacement"),
)
RECONSTRUCTION_SKILLS = frozenset({"pick_up", "slide", "twist", "hand_over"})


def _canonical_skill_id(label: str) -> str | None:
    """Recognize built-in names while leaving historical free-text labels intact."""
    key = label.casefold().replace("_", "").replace("-", "").replace(" ", "")
    for identifier, name in SKILL_CHOICES:
        if key in {identifier.replace("_", ""), name.casefold()}:
            return identifier
    return None
