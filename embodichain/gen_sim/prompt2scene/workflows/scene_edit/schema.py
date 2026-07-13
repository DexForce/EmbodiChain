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

__all__ = ["SceneEditRequest", "SceneEditResult"]


@dataclass(frozen=True)
class SceneEditRequest:
    """Input for editing an existing generated scene."""

    output_root: Path
    prompt: str
    cleanup_scene_edit_dir: bool = False
    optimize_new_objects_only: bool = True
    gravity_settle_mode: str = "geometry"
    z_axis_align_assets: bool = True


@dataclass(frozen=True)
class SceneEditResult:
    """Structured result for the scene edit workflow skeleton."""

    status: str
    prompt: str
    scene_state_path: Path
    reason: str
    steps: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prompt": self.prompt,
            "scene_state_path": str(self.scene_state_path),
            "reason": self.reason,
            "steps": self.steps,
        }
