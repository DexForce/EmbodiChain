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

"""Approximate convex decomposition toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .urdf_modifider import generate_urdf_collision_convexes


def __getattr__(name: str) -> Any:
    """Lazily import APIs with optional geometry dependencies."""
    if name == "generate_urdf_collision_convexes":
        from .urdf_modifider import generate_urdf_collision_convexes

        return generate_urdf_collision_convexes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["generate_urdf_collision_convexes"]
