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

from embodichain.gen_sim.prompt2scene.workflows.artifact_writer import (
    IMAGE_SEGMENTS_STEP,
    STEP_RESULT_FILENAME,
    UNIFIED_SCENE_GEN_STEP,
    UNIFIED_SCENE_STEP,
)

__all__ = ["UnifiedScenePaths", "resolve_generated_path"]


def resolve_generated_path(value: Any, output_root: Path) -> Path:
    """Resolve an absolute or output-root-relative generated artifact path."""
    if not value:
        return Path()
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root.expanduser().resolve() / path).resolve()


@dataclass(frozen=True)
class UnifiedScenePaths:
    """High-level paths owned by the unified-scene generation workflow."""

    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_root",
            self.output_root.expanduser().resolve(),
        )

    @property
    def workflow_root(self) -> Path:
        return self.output_root / UNIFIED_SCENE_GEN_STEP

    @property
    def image_gen_dir(self) -> Path:
        return self.workflow_root / "image_gen"

    @property
    def glb_gen_dir(self) -> Path:
        return self.workflow_root / "glb_gen"

    @property
    def debug_dir(self) -> Path:
        return self.workflow_root / "debug"

    @property
    def text_clutter_dir(self) -> Path:
        return self.glb_gen_dir / "text_clutter_settled"

    @property
    def table_fit_dir(self) -> Path:
        return self.glb_gen_dir / "table_fit_to_clutter"

    @property
    def image_segments_result(self) -> Path:
        return self.output_root / IMAGE_SEGMENTS_STEP / STEP_RESULT_FILENAME

    def prepare_generation_dirs(self) -> tuple[Path, Path, Path]:
        """Create and return the workflow's high-level generation directories."""
        directories = (self.image_gen_dir, self.glb_gen_dir, self.debug_dir)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def resolve_scene_result(self, explicit_path: Path | None) -> Path:
        """Resolve the unified-scene result produced by the preceding workflow."""
        if explicit_path is not None:
            return explicit_path.expanduser().resolve()

        scene_dir = self.output_root / UNIFIED_SCENE_STEP
        result_path = scene_dir / STEP_RESULT_FILENAME
        if result_path.is_file():
            return result_path

        legacy_path = scene_dir / "results.json"
        return legacy_path if legacy_path.is_file() else result_path
