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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEBUG_DIRNAME",
    "IMAGE_SEGMENTS_STEP",
    "IMAGE_SPATIAL_RELATIONS_STEP",
    "RAW_MODEL_OUTPUT_FILENAME",
    "SCENE_INTAKE_STEP",
    "SCENE_EDIT_STEP",
    "SCENE_PROMPT_ROUTE_STEP",
    "SCENE_RANDOMIZATION_STEP",
    "STEP_RESULT_FILENAME",
    "UNIFIED_SCENE_GEN_STEP",
    "UNIFIED_SCENE_STEP",
    "PipelinePaths",
    "debug_dir_path",
    "debug_round_dir_path",
    "next_debug_round_dir_path",
    "next_debug_round_name",
    "resolve_generated_path",
    "step_dir_path",
    "step_result_path",
]

STEP_RESULT_FILENAME = "result.json"
DEBUG_DIRNAME = "debug"
RAW_MODEL_OUTPUT_FILENAME = "raw_model_output.json"

SCENE_INTAKE_STEP = "scene_intake"
SCENE_EDIT_STEP = "scene_edit"
SCENE_PROMPT_ROUTE_STEP = "scene_prompt_route"
SCENE_RANDOMIZATION_STEP = "scene_randomization"
IMAGE_SEGMENTS_STEP = "image_segments"
IMAGE_SPATIAL_RELATIONS_STEP = "image_spatial_relations"
UNIFIED_SCENE_STEP = "unified_scene"
UNIFIED_SCENE_GEN_STEP = "unified_scene_gen"

_DEBUG_ROUND_PATTERN = re.compile(r"^round_(\d+)(?:_|$)")


def step_dir_path(output_root: Path, step_name: str) -> Path:
    return output_root / step_name


def step_result_path(output_root: Path, step_name: str) -> Path:
    return step_dir_path(output_root, step_name) / STEP_RESULT_FILENAME


def debug_dir_path(output_root: Path, step_name: str) -> Path:
    return step_dir_path(output_root, step_name) / DEBUG_DIRNAME


def debug_round_dir_path(
    output_root: Path,
    step_name: str,
    round_name: str,
) -> Path:
    return debug_dir_path(output_root, step_name) / round_name


def next_debug_round_name(
    output_root: Path,
    step_name: str,
    label: str | None = None,
) -> str:
    debug_dir = debug_dir_path(output_root, step_name)
    max_index = 0
    if debug_dir.is_dir():
        for entry in debug_dir.iterdir():
            if not entry.is_dir():
                continue
            match = _DEBUG_ROUND_PATTERN.match(entry.name)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
    name = f"round_{max_index + 1:03d}"
    if label:
        name = f"{name}_{_path_token(label)}"
    return name


def next_debug_round_dir_path(
    output_root: Path,
    step_name: str,
    label: str | None = None,
) -> Path:
    return debug_round_dir_path(
        output_root,
        step_name,
        next_debug_round_name(output_root, step_name, label),
    )


def resolve_generated_path(value: Any, output_root: Path) -> Path:
    if not value:
        return Path()
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root.expanduser().resolve() / path).resolve()


def _path_token(value: str) -> str:
    token = "".join(c if c.isalnum() else "_" for c in value)
    return token.strip("_")[:80] or "round"


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", self.output_root.expanduser().resolve())

    @property
    def scene_intake_dir(self) -> Path:
        return self.output_root / SCENE_INTAKE_STEP

    @property
    def scene_edit_dir(self) -> Path:
        return self.output_root / SCENE_EDIT_STEP

    @property
    def image_segments_dir(self) -> Path:
        return self.output_root / IMAGE_SEGMENTS_STEP

    @property
    def image_spatial_relations_dir(self) -> Path:
        return self.output_root / IMAGE_SPATIAL_RELATIONS_STEP

    @property
    def unified_scene_dir(self) -> Path:
        return self.output_root / UNIFIED_SCENE_STEP

    @property
    def unified_scene_gen_dir(self) -> Path:
        return self.output_root / UNIFIED_SCENE_GEN_STEP

    def step_result(self, step: str) -> Path:
        return step_result_path(self.output_root, step)

    @property
    def scene_intake_result(self) -> Path:
        return self.step_result(SCENE_INTAKE_STEP)

    @property
    def image_segments_result(self) -> Path:
        return self.step_result(IMAGE_SEGMENTS_STEP)

    @property
    def unified_scene_result(self) -> Path:
        return self.step_result(UNIFIED_SCENE_STEP)

    def resolve_scene_result(self, explicit_path: Path | None) -> Path:
        if explicit_path is not None:
            return explicit_path.expanduser().resolve()
        result = self.unified_scene_result
        if result.is_file():
            return result
        legacy = self.unified_scene_dir / "results.json"
        return legacy if legacy.is_file() else result

    @property
    def gen_image_dir(self) -> Path:
        return self.unified_scene_gen_dir / "image_gen"

    @property
    def gen_glb_dir(self) -> Path:
        return self.unified_scene_gen_dir / "glb_gen"

    @property
    def gen_debug_dir(self) -> Path:
        return self.unified_scene_gen_dir / "debug"

    @property
    def table_fit_dir(self) -> Path:
        return self.gen_glb_dir / "table_fit_to_clutter"

    @property
    def simready_to_aligned_manifest(self) -> Path:
        return self.gen_glb_dir / "simready_to_aligned_manifest.json"

    @property
    def table_fit_manifest(self) -> Path:
        return self.table_fit_dir / "table_fit_to_clutter_manifest.json"

    @property
    def gym_export_dir(self) -> Path:
        return self.output_root / "gym_export"

    @property
    def gym_config(self) -> Path:
        return self.gym_export_dir / "gym_config.json"

    def prepare_generation_dirs(self) -> tuple[Path, Path, Path]:
        dirs = (self.gen_image_dir, self.gen_glb_dir, self.gen_debug_dir)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return dirs
