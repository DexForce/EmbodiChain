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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RenderObjectScenesRequest", "RenderObjectScenesResult"]


@dataclass(frozen=True)
class RenderObjectScenesRequest:
    """Request to render internal Z-up object scenes with Blender."""

    object_scenes: list[tuple[str, Any]]
    output_path: Path
    timeout_seconds: int = 180


@dataclass(frozen=True)
class RenderObjectScenesResult:
    """Result of rendering object scenes."""

    output_path: Path
