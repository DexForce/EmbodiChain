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
    target_replacements: Sequence[object],
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
        target_replacements=target_replacements,
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
    target_replacements: Sequence[object],
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
        "generated_basic_background": _record_path(
            Path(generated_paths.basic_background),
            repo_root,
        ),
        "generated_atom_actions": _record_path(
            Path(generated_paths.atom_actions),
            repo_root,
        ),
        "pipeline_history_path": _record_path(history_path, repo_root),
        "target_body_scale": args.target_body_scale,
        "target_replacements": _target_replacement_records(
            args,
            target_replacements,
        ),
        "sync_replacement_names": args.sync_replacement_names,
        "reuse_target_replacements": args.reuse_target_replacements,
        "prewarm_coacd_cache": args.prewarm_coacd_cache,
        "overwrite_config": args.overwrite_config,
        "regenerate": args.regenerate,
        "skip_run_agent": args.skip_run_agent,
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
    if args.use_image2scene:
        record.update(
            {
                "server": args.server,
                "background": args.background,
                "image2scene_root": _record_path(
                    Path(args.image2scene_root).expanduser(),
                    repo_root,
                ),
                "image2scene_download_dir": str(args.image2scene_download_dir),
                "image2scene_output_root": str(args.image2scene_output_root),
                "image2scene_gen_config": str(args.image2scene_gen_config),
                "image2scene_client_url": args.image2scene_client_url or args.server,
                "image2scene_llm_config": str(args.image2scene_llm_config),
            }
        )
        if args.image2scene_extract_dir is not None:
            record["image2scene_extract_dir"] = str(args.image2scene_extract_dir)
        if args.image2scene_merged_output is not None:
            record["image2scene_merged_output"] = str(args.image2scene_merged_output)
    elif resolution.mode == "prompt2scene":
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
        if args.prompt2scene_text:
            record["prompt2scene_text"] = args.prompt2scene_text
    elif resolution.mode == "image2tabletop":
        record.update(
            {
                "server": args.server,
                "gym_project_root": _record_path(
                    Path(args.gym_project_root).expanduser(),
                    repo_root,
                ),
                "overwrite_gym_project": args.overwrite_gym_project,
            }
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


def _target_replacement_records(
    args: argparse.Namespace,
    target_replacements: Sequence[object],
) -> list[dict[str, str]]:
    requested_by_output_dir = {
        output_dir_name: replacement[0]
        for output_dir_name, replacement in (
            ("new1", args.target_replacement1),
            ("new2", args.target_replacement2),
        )
        if replacement and len(replacement) == 2
    }
    records = []
    for replacement in target_replacements:
        output_dir_name = str(getattr(replacement, "output_dir_name"))
        source_uid = str(getattr(replacement, "source_uid"))
        record = {
            "source_uid": source_uid,
            "prompt": str(getattr(replacement, "prompt")),
            "output_dir_name": output_dir_name,
        }
        requested_source_uid = requested_by_output_dir.get(output_dir_name)
        if requested_source_uid and requested_source_uid != source_uid:
            record["requested_source_uid"] = requested_source_uid
        records.append(record)
    return records


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
