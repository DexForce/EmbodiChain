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

from pathlib import Path
import re
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.io import write_json

__all__ = [
    "DEBUG_DIRNAME",
    "IMAGE_SEGMENTS_STEP",
    "IMAGE_SPATIAL_RELATIONS_STEP",
    "RAW_MODEL_OUTPUT_FILENAME",
    "SCENE_INTAKE_STEP",
    "STEP_RESULT_FILENAME",
    "TEXT_RELATIONS_STEP",
    "UNIFIED_SCENE_GEN_STEP",
    "UNIFIED_SCENE_STEP",
    "WorkflowArtifactWriter",
    "debug_dir_path",
    "debug_round_dir_path",
    "next_debug_round_dir_path",
    "next_debug_round_name",
    "step_dir_path",
    "step_result_path",
    "write_debug_json",
    "write_debug_round_json",
    "write_next_raw_model_output",
    "write_raw_model_output",
    "write_step_result",
]

STEP_RESULT_FILENAME = "result.json"
DEBUG_DIRNAME = "debug"
RAW_MODEL_OUTPUT_FILENAME = "raw_model_output.json"

SCENE_INTAKE_STEP = "scene_intake"
IMAGE_SEGMENTS_STEP = "image_segments"
IMAGE_SPATIAL_RELATIONS_STEP = "image_spatial_relations"
TEXT_RELATIONS_STEP = "text_relations"
UNIFIED_SCENE_STEP = "unified_scene"
UNIFIED_SCENE_GEN_STEP = "unified_scene_gen"

DEBUG_ROUND_PATTERN = re.compile(r"^round_(\d+)(?:_|$)")


def step_dir_path(output_root: Path, step_name: str) -> Path:
    """Return the directory path for a pipeline step."""
    return output_root / step_name


def step_result_path(output_root: Path, step_name: str) -> Path:
    """Return the final result JSON path for a pipeline step."""
    return step_dir_path(output_root, step_name) / STEP_RESULT_FILENAME


def debug_dir_path(output_root: Path, step_name: str) -> Path:
    """Return the debug directory path for a pipeline step."""
    return step_dir_path(output_root, step_name) / DEBUG_DIRNAME


def debug_round_dir_path(
    output_root: Path,
    step_name: str,
    round_name: str,
) -> Path:
    """Return a debug subdirectory path for one model/tool round."""
    return debug_dir_path(output_root, step_name) / round_name


def next_debug_round_name(
    output_root: Path,
    step_name: str,
    label: str | None = None,
) -> str:
    """Return the next step-local debug round name."""
    debug_dir = debug_dir_path(output_root, step_name)
    max_index = 0
    if debug_dir.is_dir():
        for path in debug_dir.iterdir():
            if not path.is_dir():
                continue
            match = DEBUG_ROUND_PATTERN.match(path.name)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
    round_name = f"round_{max_index + 1:03d}"
    if label:
        round_name = f"{round_name}_{_path_token(label)}"
    return round_name


def next_debug_round_dir_path(
    output_root: Path,
    step_name: str,
    label: str | None = None,
) -> Path:
    """Return the next step-local debug round directory path."""
    return debug_round_dir_path(
        output_root,
        step_name,
        next_debug_round_name(output_root, step_name, label),
    )


def write_step_result(
    output_root: Path,
    step_name: str,
    payload: dict[str, Any],
) -> Path:
    """Write a step's final result JSON and return its path."""
    path = step_result_path(output_root, step_name)
    write_json(path, payload)
    return path


def write_debug_json(
    output_root: Path,
    step_name: str,
    round_name: str,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    """Write a debug JSON file under one step debug round."""
    path = debug_round_dir_path(output_root, step_name, round_name) / filename
    write_json(path, payload)
    return path


def write_debug_round_json(
    debug_round_dir: Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    """Write a debug JSON file under an already selected debug round directory."""
    path = debug_round_dir / filename
    write_json(path, payload)
    return path


def write_raw_model_output(
    output_root: Path,
    step_name: str,
    round_name: str,
    payload: dict[str, Any],
) -> Path:
    """Write one raw structured model output under a step debug round."""
    return write_debug_json(
        output_root,
        step_name,
        round_name,
        RAW_MODEL_OUTPUT_FILENAME,
        payload,
    )


def write_next_raw_model_output(
    output_root: Path,
    step_name: str,
    payload: dict[str, Any],
    label: str | None = None,
) -> Path:
    """Write raw model output under the next step-local debug round."""
    round_name = next_debug_round_name(output_root, step_name, label)
    return write_raw_model_output(output_root, step_name, round_name, payload)


class WorkflowArtifactWriter:
    """Write workflow artifacts under a fixed step directory."""

    def __init__(self, output_root: Path, step_name: str) -> None:
        self._output_root = output_root
        self._step_name = step_name

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def step_name(self) -> str:
        return self._step_name

    @property
    def step_dir(self) -> Path:
        return step_dir_path(self._output_root, self._step_name)

    @property
    def debug_dir(self) -> Path:
        return debug_dir_path(self._output_root, self._step_name)

    @property
    def result_path(self) -> Path:
        return step_result_path(self._output_root, self._step_name)

    def next_debug_round_name(self, label: str | None = None) -> str:
        """Return the next debug round name for this step."""
        return next_debug_round_name(self._output_root, self._step_name, label)

    def next_debug_round_dir(self, label: str | None = None) -> Path:
        """Return the next debug round directory for this step."""
        return next_debug_round_dir_path(self._output_root, self._step_name, label)

    def debug_round_dir(self, round_name: str) -> Path:
        """Return one debug round directory under this step."""
        return debug_round_dir_path(self._output_root, self._step_name, round_name)

    def write_step_result(self, payload: dict[str, Any]) -> Path:
        """Write the step's final result JSON."""
        return write_step_result(self._output_root, self._step_name, payload)

    def write_debug_round_json(
        self,
        *,
        round_name: str,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write a JSON artifact inside one named debug round."""
        return write_debug_round_json(
            self.debug_round_dir(round_name),
            filename=filename,
            payload=payload,
        )

    def write_raw_model_output(
        self,
        *,
        round_name: str,
        payload: dict[str, Any],
    ) -> Path:
        """Write a raw model output into one named debug round."""
        return write_raw_model_output(
            self._output_root,
            self._step_name,
            round_name,
            payload,
        )

    def write_next_raw_model_output(
        self,
        *,
        payload: dict[str, Any],
        label: str | None = None,
    ) -> Path:
        """Write a raw model output into the next available debug round."""
        return write_next_raw_model_output(
            self._output_root,
            self._step_name,
            payload,
            label=label,
        )


def _path_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value)
    return token.strip("_")[:80] or "round"
