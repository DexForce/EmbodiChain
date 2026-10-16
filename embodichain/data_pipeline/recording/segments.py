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

"""Linear-time expansion of validated, half-open episode segment ranges."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["segments_for_frames"]


def segments_for_frames(
    episode_metadata: Mapping[str, Any] | None, length: int
) -> list[Mapping[str, Any] | None]:
    """Resolve each frame's segment once, rejecting ambiguous ranges.

    Args:
        episode_metadata: Episode sidecar containing half-open ``segments``.
        length: Number of observation/action pairs in the saved episode.

    Returns:
        One segment reference or ``None`` per frame. Gaps have no owner and
        use the recorder's legacy annotation/overall-instruction fallback.

    Raises:
        ValueError: If a segment is out of bounds or overlaps another segment.
        TypeError: If a segment is not a mapping or has non-integer boundaries.
    """
    if type(length) is not int or length < 0:
        raise ValueError("Episode length must be a non-negative integer.")
    result: list[Mapping[str, Any] | None] = [None] * length
    for segment in (episode_metadata or {}).get("segments", []):
        if not isinstance(segment, Mapping):
            raise TypeError("Segment sidecar metadata must be a mapping.")
        start, end = segment.get("start_step", 0), segment.get("end_step", 0)
        if type(start) is not int or type(end) is not int:
            raise TypeError("Segment boundaries must be integers.")
        if start < 0 or end < start or end > length:
            raise ValueError(
                f"Segment span [{start}, {end}) is outside episode length {length}."
            )
        for index in range(start, end):
            if result[index] is not None:
                raise ValueError(f"Segment spans overlap at frame {index}.")
            result[index] = segment
    return result
