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

"""Shared target contracts for object-centric atomic actions."""

from __future__ import annotations

from dataclasses import dataclass

from .core import ActionTarget, ObjectSemantics


@dataclass(frozen=True, slots=True, eq=False)
class ObjectActionTarget(ActionTarget):
    """Base target for atomic actions operating on a semantic object.

    Concrete actions add only the pose roles and constraints they actually
    consume. This shared contract deliberately does not define a generic pose:
    an object pose, a single-arm grasp pose, and a dual-arm grasp pair have
    different meanings and shapes.
    """

    semantics: ObjectSemantics
    """Semantic description of the object on which the action operates."""

    def __post_init__(self) -> None:
        if not isinstance(self.semantics, ObjectSemantics):
            raise TypeError(
                "semantics must be an ObjectSemantics, "
                f"got {type(self.semantics).__name__}."
            )


__all__ = ["ObjectActionTarget"]
