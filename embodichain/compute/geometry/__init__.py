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

"""Lazy geometry facade; existing Warp consumers need no scene dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SupportSurfaceConfig",
    "DetectedSupportSurface",
    "SupportSurfaceDetector",
    "SupportGeometry",
    "GroupSupportClampConfig",
    "GroupSupportClamp",
    "group_height_offset",
    "AABBLayoutConfig",
    "AABBLayoutOptimizer",
]

_EXPORT_MODULES = {
    "SupportSurfaceConfig": "support",
    "DetectedSupportSurface": "support",
    "SupportSurfaceDetector": "support",
    "SupportGeometry": "placement",
    "GroupSupportClampConfig": "placement",
    "GroupSupportClamp": "placement",
    "group_height_offset": "placement",
    "AABBLayoutConfig": "aabb",
    "AABBLayoutOptimizer": "aabb",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
