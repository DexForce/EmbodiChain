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
from typing import Any

from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.workflows.paths import (
    DEBUG_DIRNAME,
    IMAGE_SEGMENTS_STEP,
    IMAGE_SPATIAL_RELATIONS_STEP,
    RAW_MODEL_OUTPUT_FILENAME,
    SCENE_EDIT_STEP,
    SCENE_INTAKE_STEP,
    STEP_RESULT_FILENAME,
    UNIFIED_SCENE_GEN_STEP,
    UNIFIED_SCENE_STEP,
    debug_dir_path,
    debug_round_dir_path,
    next_debug_round_dir_path,
    next_debug_round_name,
    step_dir_path,
    step_result_path,
)

__all__ = [
    "DEBUG_DIRNAME",
    "IMAGE_SEGMENTS_STEP",
    "IMAGE_SPATIAL_RELATIONS_STEP",
    "RAW_MODEL_OUTPUT_FILENAME",
    "SCENE_EDIT_STEP",
    "SCENE_INTAKE_STEP",
    "STEP_RESULT_FILENAME",
    "UNIFIED_SCENE_GEN_STEP",
    "UNIFIED_SCENE_STEP",
    "WorkflowArtifactWriter",
    "write_debug_json",
    "write_debug_round_json",
    "write_next_raw_model_output",
    "write_raw_model_output",
    "write_step_result",
]


def write_step_result(
    output_root: Path,
    step_name: str,
    payload: dict[str, Any],
) -> Path:
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
    path = debug_round_dir_path(output_root, step_name, round_name) / filename
    write_json(path, payload)
    return path


def write_debug_round_json(
    debug_round_dir: Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    path = debug_round_dir / filename
    write_json(path, payload)
    return path


def write_raw_model_output(
    output_root: Path,
    step_name: str,
    round_name: str,
    payload: dict[str, Any],
) -> Path:
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
    round_name = next_debug_round_name(output_root, step_name, label)
    return write_raw_model_output(output_root, step_name, round_name, payload)


class WorkflowArtifactWriter:
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
        return next_debug_round_name(self._output_root, self._step_name, label)

    def next_debug_round_dir(self, label: str | None = None) -> Path:
        return next_debug_round_dir_path(self._output_root, self._step_name, label)

    def debug_round_dir(self, round_name: str) -> Path:
        return debug_round_dir_path(self._output_root, self._step_name, round_name)

    def write_step_result(self, payload: dict[str, Any]) -> Path:
        return write_step_result(self._output_root, self._step_name, payload)

    def write_debug_round_json(
        self,
        *,
        round_name: str,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
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
        return write_next_raw_model_output(
            self._output_root,
            self._step_name,
            payload,
            label=label,
        )
