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

"""Frozen camera descriptors for exporting observations without a simulator."""

from __future__ import annotations
from dataclasses import dataclass

__all__ = ["RecordedCamera"]


@dataclass(frozen=True)
class RecordedCamera:
    """Image dimensions retained with recorded camera frames.

    Args:
        width: Recorded image width in pixels.
        height: Recorded image height in pixels.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0 for value in (self.width, self.height)
        ):
            raise ValueError("recorded camera dimensions must be positive integers")

    @property
    def cfg(self) -> RecordedCamera:
        """Expose the dimension contract shared with native camera configuration."""
        return self
