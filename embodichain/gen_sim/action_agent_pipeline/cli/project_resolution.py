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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.cli.image2scene_stage import (
    resolve_image2tabletop_server,
    run_image2scene_pipeline,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    DEFAULT_IMAGE,
    DEFAULT_TASK_TEMPLATE_NAMES,
    IMAGE_SUFFIXES,
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

__all__ = [
    "ProjectResolution",
    "resolve_gym_project",
    "resolve_task_description_for_generation",
]

_DEFAULT_IMAGE_DIR = DEFAULT_IMAGE.parent


@dataclass(frozen=True)
class ProjectResolution:
    path: Path
    mode: str
    base_history: dict[str, Any] | None = None


def resolve_task_description_for_generation(args: argparse.Namespace) -> str | None:
    task_description = str(args.task_description or "").strip()
    if args.task_name in DEFAULT_TASK_TEMPLATE_NAMES:
        if task_description:
            print(
                f"Ignoring --task_description for {args.task_name}; "
                "using the default basket task template.",
                flush=True,
            )
        return None
    return task_description or None


def resolve_gym_project(args: argparse.Namespace) -> ProjectResolution:
    use_history = args.base_task_name is not None or args.base_history_index is not None
    selected_modes = [
        args.use_image2scene,
        args.use_prompt2scene,
        args.use_existing_gym_project,
        use_history,
    ]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise ValueError(
            "Use only one of --use-image2scene, --use-prompt2scene, "
            "--use-existing-gym-project, or --base-task-name/--base-history-index."
        )

    if args.use_existing_gym_project:
        project_path = Path(args.gym_project).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"gym project not found: {project_path}")
        print(f"Using existing gym project: {project_path}", flush=True)
        return ProjectResolution(path=project_path, mode="existing_gym_project")

    if args.use_image2scene:
        return ProjectResolution(
            path=run_image2scene_pipeline(args),
            mode="image2scene",
        )

    if args.use_prompt2scene:
        return ProjectResolution(
            path=run_prompt2scene_stage(args),
            mode="prompt2scene",
        )

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

    from embodichain.gen_sim.action_agent_pipeline.gym_project_api.image2tabletop_client import (
        check_health,
        collect_image_paths,
        process_image,
    )

    image_input = _resolve_image_input(args)
    image_path = _resolve_single_image(str(image_input), collect_image_paths)
    server = resolve_image2tabletop_server(args)
    if not args.skip_health_check:
        check_health(server)

    return ProjectResolution(
        path=process_image(
            server=server,
            image_path=image_path,
            output_root=Path(args.gym_project_root),
            poll_interval=args.poll_interval,
            overwrite=args.overwrite_gym_project,
            job_timeout_s=args.job_timeout_s,
        ),
        mode="image2tabletop",
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


def _resolve_single_image(
    image_input: str,
    collect_image_paths: Callable[[Path], list[Path]],
) -> Path:
    image_paths = collect_image_paths(Path(image_input))
    if len(image_paths) != 1:
        paths = ", ".join(path.as_posix() for path in image_paths)
        raise ValueError(
            "This pipeline expects exactly one image, but got "
            f"{len(image_paths)}: {paths}"
        )
    return image_paths[0]


def _resolve_image_input(args: argparse.Namespace) -> Path:
    if args.image_name:
        return _resolve_image_name(args.image_name)
    if args.image:
        return Path(args.image)
    return DEFAULT_IMAGE


def _resolve_image_name(image_name: str) -> Path:
    image_path = Path(image_name)
    if image_path.parent != Path("."):
        raise ValueError(
            "--image-name only accepts a file name under "
            f"{_DEFAULT_IMAGE_DIR.as_posix()}. Use --image for a full path."
        )
    if image_path.suffix:
        return _DEFAULT_IMAGE_DIR / image_path

    matches = [
        _DEFAULT_IMAGE_DIR / f"{image_name}{suffix}" for suffix in IMAGE_SUFFIXES
    ]
    existing = [path for path in matches if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        candidates = ", ".join(path.name for path in matches)
        raise FileNotFoundError(
            f"Image name {image_name!r} was not found. Tried: {candidates}"
        )

    matched = ", ".join(path.name for path in existing)
    raise ValueError(
        f"Image name {image_name!r} is ambiguous. Use --image-name with a suffix: "
        f"{matched}"
    )
