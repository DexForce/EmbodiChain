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

"""Publish canonical generation artifacts without intermediate copies."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from embodichain.gen_sim.action_engine.protocol import (
    AGENT_CONFIG_FILENAME,
    EXECUTION_PROGRAM_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
    SCENE_REQUIREMENTS_FILENAME,
    SEED_TASK_GRAPH_PNG_FILENAME,
    TASK_SPEC_FILENAME,
)

from .models import GeneratedConfigPaths

__all__ = ["artifact_paths", "write_generation_artifacts"]


def artifact_paths(
    output_dir: str | Path,
    *,
    planning_mode: str = "offline",
) -> GeneratedConfigPaths:
    """Return canonical resolved paths for one output directory."""
    directory = Path(output_dir).expanduser().resolve()
    _validate_planning_mode(planning_mode)
    graph_directory = directory if planning_mode == "offline" else directory / "offline"
    return GeneratedConfigPaths(
        gym_config=directory / FAST_GYM_CONFIG_FILENAME,
        agent_config=directory / AGENT_CONFIG_FILENAME,
        task_spec=directory / TASK_SPEC_FILENAME,
        scene_requirements=directory / SCENE_REQUIREMENTS_FILENAME,
        seed_task_graph=graph_directory / EXECUTION_PROGRAM_FILENAME,
        seed_task_graph_png=graph_directory / SEED_TASK_GRAPH_PNG_FILENAME,
        planning_mode=planning_mode,
    )


def write_generation_artifacts(
    output_dir: str | Path,
    *,
    gym_config: Mapping[str, Any],
    agent_config: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    scene_requirements: Mapping[str, Any],
    seed_task_graph: Mapping[str, Any],
    seed_task_graph_png: bytes,
    overwrite: bool,
    planning_mode: str = "offline",
) -> GeneratedConfigPaths:
    """Serialize validated artifacts and replace their destinations atomically."""
    paths = artifact_paths(output_dir, planning_mode=planning_mode)
    if not isinstance(seed_task_graph_png, (bytes, bytearray)):
        raise TypeError("seed_task_graph_png must be bytes.")
    payloads = {
        paths.gym_config: _serialize_json(gym_config),
        paths.agent_config: _serialize_json(agent_config),
        paths.task_spec: _serialize_json(task_spec),
        paths.scene_requirements: _serialize_json(scene_requirements),
        paths.seed_task_graph: _serialize_json(seed_task_graph),
        paths.seed_task_graph_png: bytes(seed_task_graph_png),
    }
    existing = sorted(path for path in payloads if path.exists())
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Generated artifacts already exist in {paths.gym_config.parent}: "
            f"{names}. Pass --overwrite to replace them."
        )

    paths.gym_config.parent.mkdir(parents=True, exist_ok=True)
    temporary: dict[Path, Path] = {}
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary[destination] = _write_temporary(destination.parent, payload)
        for destination, temporary_path in temporary.items():
            os.replace(temporary_path, destination)
    finally:
        for temporary_path in temporary.values():
            temporary_path.unlink(missing_ok=True)
    return paths


def _serialize_json(value: Mapping[str, Any]) -> str:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Generated artifact is not strict JSON data.") from exc


def _write_temporary(directory: Path, payload: str | bytes) -> Path:
    data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=directory,
        prefix=".action_engine_",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def _validate_planning_mode(value: Any) -> None:
    if value not in {"offline", "ab"}:
        raise ValueError("planning_mode must be 'offline' or 'ab'.")
