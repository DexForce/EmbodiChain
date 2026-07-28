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

"""Render serialized spatial relations as canonical prompt language."""

from __future__ import annotations

__all__ = ["RELATIVE_RELATION_PHRASES", "relative_relation_phrase"]

RELATIVE_RELATION_PHRASES = {
    "inside": "inside",
    "on": "on top of",
    "left_of": "to the left of",
    "right_of": "to the right of",
    "front_of": "in front of",
    "behind": "behind",
    "front_left_of": "to the front-left of",
    "back_left_of": "to the back-left of",
    "front_right_of": "to the front-right of",
    "back_right_of": "to the back-right of",
}


def relative_relation_phrase(relation: str) -> str:
    """Return the canonical English phrase used in prompts and datasets."""
    try:
        return RELATIVE_RELATION_PHRASES[relation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported relative placement relation: {relation!r}."
        ) from exc
