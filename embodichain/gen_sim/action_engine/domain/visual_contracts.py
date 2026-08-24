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

"""Canonical visual-fact contracts shared by planning and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

__all__ = [
    "OCCLUSION_RELATION",
    "VISUAL_RELATION_PARTICIPANTS",
    "requested_visual_task_predicates",
]


OCCLUSION_RELATION = "occludes"

# Participant order is semantic. For ``occludes`` it is
# ``[occluder_uid, occluded_uid]``.
VISUAL_RELATION_PARTICIPANTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {OCCLUSION_RELATION: ("occluder", "occluded")}
)


def requested_visual_task_predicates(task_spec: Mapping[str, Any]) -> frozenset[str]:
    """Return task-level visual predicates explicitly requested by a TaskSpec."""
    result: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("type") == "visual_relation":
                relation = value.get("relation")
                if isinstance(relation, str) and relation:
                    result.add(relation)
            for child in value.values():
                collect(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                collect(child)

    collect(task_spec.get("success", {}))
    return frozenset(result)
