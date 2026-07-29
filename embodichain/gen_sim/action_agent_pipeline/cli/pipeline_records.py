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

"""Pipeline history and manifest record helpers."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
)

__all__ = [
    "append_pipeline_history",
    "build_pipeline_record",
    "find_history_entry_by_index",
    "history_entry_has_source",
    "history_entry_index",
    "path_from_history_entry",
    "pipeline_history_path",
    "read_pipeline_history",
    "resolve_record_path",
    "resolve_source_gym_config",
    "write_pipeline_manifests",
]

_PROMPT2SCENE_EXISTING_PROJECT_MODE = "prompt2scene_existing_gym_project"


def pipeline_history_path(args: argparse.Namespace) -> Path:
    return Path(args.pipeline_history_path).expanduser().resolve()


def read_pipeline_history(
    history_path: Path,
    *,
    schema_version: int,
) -> dict[str, Any]:
    if not history_path.exists():
        return {"schema_version": schema_version, "runs": []}

    data = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline history must be a JSON object: {history_path}")
    runs = data.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"Pipeline history must contain a runs list: {history_path}")
    return {
        "schema_version": data.get("schema_version", schema_version),
        "runs": runs,
    }


def find_history_entry_by_index(
    runs: list[Any], history_index: int
) -> dict[str, Any] | None:
    for entry in runs:
        if isinstance(entry, dict) and history_entry_index(entry) == history_index:
            return entry
    return None


def history_entry_index(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("index", 0))
    except (TypeError, ValueError):
        return 0


def history_entry_has_source(entry: dict[str, Any]) -> bool:
    return bool(entry.get("source_gym_config") or entry.get("source_gym_project_dir"))


def path_from_history_entry(entry: dict[str, Any], *, repo_root: Path) -> Path:
    source = entry.get("source_gym_config") or entry.get("source_gym_project_dir")
    if not source:
        raise ValueError(
            f"Pipeline history entry #{entry.get('index')} has no source gym path."
        )
    path = resolve_record_path(str(source), repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline history source path does not exist: {path}")
    return path


def resolve_record_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def write_pipeline_manifests(
    *,
    args: argparse.Namespace,
    resolution: Any,
    generated_paths: Any,
    repo_root: Path,
    schema_version: int,
    manifest_filename: str,
) -> dict[str, Any]:
    history_path = pipeline_history_path(args)
    record = build_pipeline_record(
        args=args,
        resolution=resolution,
        generated_paths=generated_paths,
        history_path=history_path,
        repo_root=repo_root,
        schema_version=schema_version,
    )
    record = append_pipeline_history(
        history_path,
        record,
        schema_version=schema_version,
    )

    manifest_path = Path(generated_paths.output_dir) / manifest_filename
    manifest_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"Updated pipeline history: {history_path}", flush=True)
    print(f"Wrote pipeline manifest: {manifest_path}", flush=True)
    return record


def build_pipeline_record(
    *,
    args: argparse.Namespace,
    resolution: Any,
    generated_paths: Any,
    history_path: Path,
    repo_root: Path,
    schema_version: int,
) -> dict[str, Any]:
    source_gym_config = resolve_source_gym_config(
        Path(resolution.path),
        gym_config_preference=("gym_config_merged.json", "gym_config.json"),
    )
    source_gym_project_dir = source_gym_config.parent
    source_sha256 = _file_sha256(source_gym_config)
    record: dict[str, Any] = {
        "schema_version": schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "task_name": args.task_name,
        "source_mode": resolution.mode,
        "source_id": f"gym_config_sha256:{source_sha256}",
        "source_gym_config_sha256": source_sha256,
        "path_base": "repo_root",
        "source_gym_project_dir": _record_path(source_gym_project_dir, repo_root),
        "source_gym_config": _record_path(source_gym_config, repo_root),
        "input_path": _record_path(Path(resolution.path), repo_root),
        "config_output_dir": _record_path(Path(generated_paths.output_dir), repo_root),
        "generated_gym_config": _record_path(
            Path(generated_paths.gym_config),
            repo_root,
        ),
        "generated_agent_config": _record_path(
            Path(generated_paths.agent_config),
            repo_root,
        ),
        "generated_task_prompt": _record_path(
            Path(generated_paths.task_prompt),
            repo_root,
        ),
        "generated_seed_task_graph": (
            _record_path(Path(generated_paths.seed_task_graph), repo_root)
            if getattr(generated_paths, "seed_task_graph", None) is not None
            else None
        ),
        "generated_seed_task_graph_png": (
            _record_path(Path(generated_paths.seed_task_graph_png), repo_root)
            if getattr(generated_paths, "seed_task_graph_png", None) is not None
            else None
        ),
        "generated_task_graph": (
            _record_path(Path(generated_paths.task_graph), repo_root)
            if getattr(generated_paths, "task_graph", None) is not None
            else None
        ),
        "generated_task_graph_png": (
            _record_path(Path(generated_paths.task_graph_png), repo_root)
            if getattr(generated_paths, "task_graph_png", None) is not None
            else None
        ),
        "generated_basic_background": _record_path(
            Path(generated_paths.basic_background),
            repo_root,
        ),
        "generated_atom_actions": _record_path(
            Path(generated_paths.atom_actions),
            repo_root,
        ),
        "pipeline_history_path": _record_path(history_path, repo_root),
        "robot_profile": getattr(args, "robot_profile", None),
        "target_body_scale": args.target_body_scale,
        "target_body_scale_mode": getattr(args, "target_body_scale_mode", None),
        "load_template_material": getattr(args, "load_template_material", False),
        "inside_container_slot_distance_scale": (
            args.inside_container_slot_distance_scale
        ),
        "surface_release_clearance": getattr(
            args,
            "surface_release_clearance",
            DEFAULT_SURFACE_RELEASE_CLEARANCE,
        ),
        "acd_method": args.acd_method,
        "overwrite_config": args.overwrite_config,
        "regenerate": args.regenerate,
        "skip_run_agent": args.skip_run_agent,
        "headless": getattr(args, "headless", False),
        "generation_summary": generated_paths.summary,
    }
    if args.task_description:
        record["task_description"] = args.task_description
    record.update(_source_request_record(args, resolution, repo_root=repo_root))
    return record


def resolve_source_gym_config(
    input_path: Path,
    *,
    gym_config_preference: Sequence[str],
) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.name not in gym_config_preference:
            expected = ", ".join(gym_config_preference)
            raise ValueError(f"Expected one of {expected}, got: {input_path}")
        return input_path

    for filename in gym_config_preference:
        path = input_path / filename
        if path.is_file():
            return path.resolve()

    matches = []
    for filename in gym_config_preference:
        matches.extend(sorted(input_path.rglob(filename)))
    unique_matches = sorted({path.resolve() for path in matches})
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        expected = " or ".join(gym_config_preference)
        raise FileNotFoundError(f"{expected} not found under: {input_path}")
    match_text = ", ".join(path.as_posix() for path in unique_matches)
    raise ValueError(
        f"Multiple gym config files found under {input_path}: {match_text}"
    )


def append_pipeline_history(
    history_path: Path,
    record: dict[str, Any],
    *,
    schema_version: int,
) -> dict[str, Any]:
    history = read_pipeline_history(history_path, schema_version=schema_version)
    runs = history["runs"]
    next_index = (
        max(
            (history_entry_index(entry) for entry in runs if isinstance(entry, dict)),
            default=0,
        )
        + 1
    )
    record = dict(record)
    record["index"] = next_index

    runs.append(record)
    history["schema_version"] = schema_version
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return record


def _source_request_record(
    args: argparse.Namespace,
    resolution: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    if args.image_name:
        record["image_name"] = args.image_name
    if args.image:
        record["image"] = _record_path(Path(args.image).expanduser(), repo_root)
    if resolution.mode == "prompt2scene":
        record.update(
            {
                "prompt2scene_output_root": _record_path(
                    Path(args.prompt2scene_output_root).expanduser(),
                    repo_root,
                ),
                "prompt2scene_llm_config": _record_path(
                    Path(args.prompt2scene_llm_config).expanduser(),
                    repo_root,
                ),
            }
        )
        prompt2scene_text = getattr(args, "prompt2scene_text", None)
        if prompt2scene_text:
            record["prompt2scene_text"] = prompt2scene_text
        prompt2scene_prompt = getattr(args, "prompt2scene_prompt", None)
        if prompt2scene_prompt:
            record["prompt2scene_prompt"] = prompt2scene_prompt
        record["prompt2scene_gravity_settle_mode"] = getattr(
            args,
            "prompt2scene_gravity_settle_mode",
            "geometry",
        )
        record["prompt2scene_scene_z_rotation_degrees"] = (
            args.prompt2scene_scene_z_rotation_degrees
        )
    elif resolution.mode == _PROMPT2SCENE_EXISTING_PROJECT_MODE:
        record["gym_project"] = _record_path(
            Path(args.gym_project).expanduser(),
            repo_root,
        )
        record["prompt2scene_scene_z_rotation_degrees"] = (
            args.prompt2scene_scene_z_rotation_degrees
        )
    elif resolution.mode == "existing_gym_project":
        record["gym_project"] = _record_path(
            Path(args.gym_project).expanduser(),
            repo_root,
        )
    elif resolution.mode == "history" and resolution.base_history is not None:
        base_source_path = path_from_history_entry(
            resolution.base_history,
            repo_root=repo_root,
        )
        record.update(
            {
                "base_task_name": args.base_task_name,
                "base_history_index": resolution.base_history.get("index"),
                "base_history_task_name": resolution.base_history.get("task_name"),
                "base_history_source_id": resolution.base_history.get("source_id"),
                "base_history_source_gym_config": _record_path(
                    base_source_path,
                    repo_root,
                ),
            }
        )
    return record


def _record_path(path: Path, repo_root: Path) -> str:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    repo_root = repo_root.expanduser().resolve()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
