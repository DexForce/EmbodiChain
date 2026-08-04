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

"""Orchestrate source-scene preparation, planning, compilation, and publication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.config import (
    generation_defaults,
    resolve_agent_runtime_policy,
)
from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    EXECUTION_PROGRAM_FILENAME,
    TASK_AGENT_FILENAME,
)

from .artifacts import artifact_paths, write_generation_artifacts
from .assets import normalize_scene_assets
from .config_builder import build_agent_config, build_fast_gym_config
from .models import GeneratedConfigPaths
from .source_scene import prepare_scene

__all__ = ["generate_action_engine_config"]

_GENERATION_DEFAULTS = generation_defaults()
_TASK_DEFAULTS = _GENERATION_DEFAULTS["task"]
_SCENE_DEFAULTS = _GENERATION_DEFAULTS["scene"]
_DEFAULT_BODY_SCALE = tuple(float(value) for value in _SCENE_DEFAULTS["body_scale"])


def generate_action_engine_config(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str,
    task_description: str | None = None,
    task_agent: Mapping[str, Any] | str | Path | None = None,
    robot_profile: str = str(_TASK_DEFAULTS["default_robot_profile"]),
    llm_model: str | None = None,
    source_scene_z_rotation_degrees: float | None = None,
    body_scale_policy: str = str(_SCENE_DEFAULTS["body_scale_policy"]),
    body_scale: Sequence[float] = _DEFAULT_BODY_SCALE,
    overwrite: bool = False,
    max_episodes: int = int(_TASK_DEFAULTS["max_episodes"]),
    max_episode_steps: int = int(_TASK_DEFAULTS["max_episode_steps"]),
    randomize_scene: bool = False,
    randomize_table_material: bool = False,
) -> GeneratedConfigPaths:
    """Generate the complete Action Engine input bundle.

    The LLM produces only a route-free semantic Task Agent. The deterministic
    compiler then lowers it directly into the coordinate-free Execution Program
    consumed by runtime; there is no compiled JSON copy or second graph format.
    """
    task_name = str(task_name).strip()
    task_description = "" if task_description is None else str(task_description).strip()
    if not task_name:
        raise ValueError("task_name must be a non-empty string.")
    if not task_description and task_agent is None:
        raise ValueError(
            "task_description is required when task_agent is not supplied."
        )
    _raise_if_outputs_exist(output_dir, overwrite=overwrite)

    scene = prepare_scene(
        gym_project,
        z_rotation_degrees=source_scene_z_rotation_degrees,
        body_scale_policy=body_scale_policy,
        body_scale=body_scale,
    )
    scene = normalize_scene_assets(scene, output_dir)

    # Delayed imports keep scene/config tooling lightweight and avoid importing
    # an LLM client when callers only inspect exported projects.
    from embodichain.gen_sim.action_engine.compiler import compile_task_agent
    from embodichain.gen_sim.action_engine.domain import (
        execution_program_hash,
        validate_execution_program,
        validate_task_agent,
    )
    from embodichain.gen_sim.action_engine.planning import plan_task

    known_objects = [str(item["runtime_uid"]) for item in scene.planner_objects]
    if task_agent is None:
        planned = plan_task(
            task_name=task_name,
            task_description=task_description,
            scene_objects=[deepcopy(obj) for obj in scene.planner_objects],
            model=llm_model,
        )
    else:
        planned = _read_task_agent(task_agent)
        if not task_description:
            task_description = str(planned.get("goal", "")).strip()
    task_agent = _validated_mapping(
        planned,
        validator=lambda value: validate_task_agent(
            value,
            known_objects=known_objects,
        ),
        label="Task Agent",
    )
    _require_matching_task(task_agent, task_name, label="Task Agent")

    compiled = compile_task_agent(
        task_agent,
        known_objects=known_objects,
    )
    execution_program = _validated_mapping(
        compiled,
        validator=validate_execution_program,
        label="Execution Program",
    )
    _require_matching_task(execution_program, task_name, label="Execution Program")
    program_hash = str(execution_program_hash(execution_program))
    if not program_hash:
        raise ValueError("Execution Program hash must be non-empty.")

    # Rendering consumes the exact validated in-memory program that runtime
    # consumes. The PNG is review-only and never appears in agent input fields.
    from embodichain.gen_sim.action_engine.graph_visualization import (
        render_seed_task_graph_png,
    )

    seed_task_graph_png = render_seed_task_graph_png(execution_program)
    if not isinstance(seed_task_graph_png, bytes):
        raise TypeError("render_seed_task_graph_png must return bytes.")

    agent_config = build_agent_config(
        task_name=task_name,
        robot_profile=robot_profile,
        execution_program_hash=program_hash,
        source_config_path=scene.source_config_path,
        uid_map=scene.uid_map,
    )
    gym_config = build_fast_gym_config(
        scene,
        task_name=task_name,
        task_description=task_description,
        robot_profile=robot_profile,
        execution_program_hash=program_hash,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        randomize_scene=randomize_scene,
        randomize_table_material=randomize_table_material,
    )
    _validate_agent_config(agent_config)
    return write_generation_artifacts(
        output_dir,
        gym_config=gym_config,
        agent_config=agent_config,
        task_agent=task_agent,
        execution_program=execution_program,
        seed_task_graph_png=seed_task_graph_png,
        overwrite=overwrite,
    )


def _read_task_agent(
    source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source))
    path = Path(source).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read Task Agent at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task Agent at {path} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Task Agent JSON must contain an object.")
    return deepcopy(dict(value))


def _validated_mapping(
    value: Any,
    *,
    validator: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{label} producer returned {type(value).__name__}, not a mapping."
        )
    candidate = deepcopy(dict(value))
    validated = validator(candidate)
    if validated is None:
        # Validators may either return a normalized mapping or validate in place.
        validated = candidate
    if not isinstance(validated, Mapping):
        raise TypeError(f"{label} validator must return a mapping or None.")
    return deepcopy(dict(validated))


def _require_matching_task(
    program: Mapping[str, Any],
    task_name: str,
    *,
    label: str,
) -> None:
    if program.get("task") != task_name:
        raise ValueError(
            f"{label} task {program.get('task')!r} does not match "
            f"requested task_name {task_name!r}."
        )


def _validate_agent_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != ACTION_ENGINE_CONFIG_SCHEMA:
        raise ValueError("Agent config has an unexpected schema_version.")
    if config.get("task_agent") != TASK_AGENT_FILENAME:
        raise ValueError("Agent config must point to the canonical Task Agent.")
    if config.get("execution_program") != EXECUTION_PROGRAM_FILENAME:
        raise ValueError("Agent config must point to the canonical Execution Program.")
    resolve_agent_runtime_policy(config)


def _raise_if_outputs_exist(output_dir: str | Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    paths = artifact_paths(output_dir)
    existing = [
        path
        for path in (
            paths.gym_config,
            paths.agent_config,
            paths.task_agent,
            paths.execution_program,
            paths.seed_task_graph_png,
        )
        if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Generated artifacts already exist in {paths.gym_config.parent}: "
            f"{names}. Pass --overwrite to replace them."
        )
