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

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    PIPELINE_HISTORY_SCHEMA_VERSION,
    REPO_ROOT,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_records import (
    find_history_entry_by_index,
    history_entry_has_source,
    history_entry_index,
    path_from_history_entry,
    pipeline_history_path,
    read_pipeline_history,
)
from embodichain.gen_sim.action_agent_pipeline.cli.prompt2scene_stage import (
    run_prompt2scene_stage,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    is_prompt2scene_gym_export,
)

__all__ = [
    "PROMPT2SCENE_PROJECT_MODES",
    "ProjectResolution",
    "is_prompt2scene_gym_export",
    "resolve_gym_project",
    "resolve_task_description_for_generation",
]

_PROMPT2SCENE_EXISTING_PROJECT_MODE = "prompt2scene_existing_gym_project"
PROMPT2SCENE_PROJECT_MODES = frozenset(
    {"prompt2scene", _PROMPT2SCENE_EXISTING_PROJECT_MODE}
)


@dataclass(frozen=True)
class ProjectResolution:
    path: Path
    mode: str
    base_history: dict[str, Any] | None = None


def resolve_task_description_for_generation(args: argparse.Namespace) -> str:
    """Return the task goal the pipeline hands to config generation.

    Config generation has no default task template, so the goal is mandatory
    here too. Failing in the pipeline gives a clearer message than letting the
    generator reject an empty description after scene resolution has run.
    """
    task_description = str(args.task_description or "").strip()
    if not task_description:
        raise ValueError(
            "--task_description is required. Provide the natural-language task "
            "goal so the task router can select a supported route."
        )
    return task_description


def resolve_gym_project(args: argparse.Namespace) -> ProjectResolution:
    use_history = args.base_task_name is not None or args.base_history_index is not None
    selected_modes = [
        args.use_prompt2scene,
        args.use_existing_gym_project,
        use_history,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError(
            "Use only one of --use-prompt2scene, --use-existing-gym-project, "
            "or --base-task-name/--base-history-index."
        )

    if args.use_existing_gym_project:
        project_path = Path(args.gym_project).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"gym project not found: {project_path}")
        prompt2scene_prompt = str(
            getattr(args, "prompt2scene_prompt", "") or ""
        ).strip()
        if prompt2scene_prompt:
            raise ValueError(
                "--prompt2scene-prompt cannot be used with "
                "--use-existing-gym-project. Use --use-prompt2scene with "
                "--prompt2scene-output-root for prompt2scene edit/randomization."
            )
        mode = (
            _PROMPT2SCENE_EXISTING_PROJECT_MODE
            if is_prompt2scene_gym_export(project_path)
            else "existing_gym_project"
        )
        print(f"Using existing gym project: {project_path}", flush=True)
        if mode == _PROMPT2SCENE_EXISTING_PROJECT_MODE:
            print(
                "Detected prompt2scene gym_export; applying prompt2scene "
                "action-agent alignment.",
                flush=True,
            )
        return ProjectResolution(path=project_path, mode=mode)

    if use_history:
        history_entry = _resolve_base_history_entry(args)
        project_path = path_from_history_entry(history_entry, repo_root=REPO_ROOT)
        print(
            "Using base history "
            f"#{history_entry.get('index')} ({history_entry.get('task_name')}): "
            f"{project_path}",
            flush=True,
        )
        return ProjectResolution(
            path=project_path,
            mode="history",
            base_history=history_entry,
        )

    return ProjectResolution(
        path=run_prompt2scene_stage(args),
        mode="prompt2scene",
    )


def _resolve_base_history_entry(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_history_index is not None and args.base_history_index <= 0:
        raise ValueError("--base-history-index must be a positive integer.")

    history_path = pipeline_history_path(args)
    history = read_pipeline_history(
        history_path,
        schema_version=PIPELINE_HISTORY_SCHEMA_VERSION,
    )
    runs = history["runs"]

    if args.base_history_index is not None:
        entry = find_history_entry_by_index(runs, args.base_history_index)
        if entry is None:
            raise ValueError(
                f"Pipeline history index not found: {args.base_history_index}"
            )
        if args.base_task_name and entry.get("task_name") != args.base_task_name:
            raise ValueError(
                "Pipeline history entry "
                f"#{args.base_history_index} has task_name={entry.get('task_name')!r}, "
                f"expected {args.base_task_name!r}."
            )
        return dict(entry)

    if not args.base_task_name:
        raise ValueError("--base-task-name is required without --base-history-index.")

    candidates = [
        entry
        for entry in runs
        if entry.get("task_name") == args.base_task_name
        and history_entry_has_source(entry)
    ]
    if not candidates:
        raise ValueError(
            "No pipeline history entry found for task_name="
            f"{args.base_task_name!r} in {history_path}"
        )
    return dict(max(candidates, key=history_entry_index))
