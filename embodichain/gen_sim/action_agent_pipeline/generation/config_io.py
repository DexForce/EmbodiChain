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

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    AGENT_CONFIG_FILENAME,
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
    TASK_GRAPH_FILENAME,
    TASK_PROMPT_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    GeneratedActionAgentConfigPaths,
)

__all__ = [
    "read_json",
    "raise_if_generated_files_exist",
    "write_config_bundle",
    "write_json",
    "write_text",
]


def write_config_bundle(
    *,
    output_dir: Path,
    bundle: Mapping[str, Any],
    overwrite: bool,
) -> GeneratedActionAgentConfigPaths:
    """Write runtime inputs and review diagnostics as one portable bundle.

    The JSON task graph is the executable source. The task prompt, background,
    and atomic-action text files describe the same generated plan for auditing
    and backward-compatible tooling; runtime execution does not parse them.
    """
    paths = GeneratedActionAgentConfigPaths(
        output_dir=output_dir,
        gym_config=output_dir / FAST_GYM_CONFIG_FILENAME,
        agent_config=output_dir / AGENT_CONFIG_FILENAME,
        task_prompt=output_dir / TASK_PROMPT_FILENAME,
        task_graph=output_dir / TASK_GRAPH_FILENAME,
        basic_background=output_dir / BASIC_BACKGROUND_FILENAME,
        atom_actions=output_dir / ATOM_ACTIONS_FILENAME,
        summary=dict(bundle.get("summary", {})),
    )
    raise_if_generated_files_exist(output_dir, overwrite)

    serialized_files = [
        (paths.gym_config, _serialize_json(bundle["gym_config"])),
        (paths.agent_config, _serialize_json(bundle["agent_config"])),
        (paths.task_prompt, _serialize_text(bundle["task_prompt"])),
        (paths.task_graph, _serialize_json(bundle["task_graph"])),
        (paths.basic_background, _serialize_text(bundle["basic_background"])),
        (paths.atom_actions, _serialize_text(bundle["atom_actions"])),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_file_transaction(serialized_files)
    return paths


def raise_if_generated_files_exist(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    output_files = [
        output_dir / FAST_GYM_CONFIG_FILENAME,
        output_dir / AGENT_CONFIG_FILENAME,
        output_dir / TASK_PROMPT_FILENAME,
        output_dir / TASK_GRAPH_FILENAME,
        output_dir / BASIC_BACKGROUND_FILENAME,
        output_dir / ATOM_ACTIONS_FILENAME,
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        existing_text = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            f"Generated file(s) already exist: {existing_text}. "
            "Pass overwrite=True or --overwrite to replace them."
        )


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_transaction([(path, _serialize_json(data))])


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_transaction([(path, _serialize_text(content))])


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _serialize_json(data: Mapping[str, Any]) -> str:
    """Serialize JSON before touching any destination file."""
    return json.dumps(data, ensure_ascii=False, indent=4) + "\n"


def _serialize_text(content: str) -> str:
    """Normalize one generated text artifact before staging it."""
    return content.rstrip() + "\n"


def _write_file_transaction(files: list[tuple[Path, str]]) -> None:
    """Replace a group of files and restore the old group on ordinary failure.

    Every new file is fully written and fsynced in its destination directory
    before any public path changes. Each ``os.replace`` is atomic on a single
    filesystem; backups provide rollback if a later replacement raises.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in files:
            staged.append((destination, _stage_text_file(destination, content)))
        _commit_staged_files(staged)
    finally:
        for _, staged_path in staged:
            staged_path.unlink(missing_ok=True)


def _stage_text_file(destination: Path, content: str) -> Path:
    """Write and fsync one hidden temporary file beside its destination."""
    descriptor, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temp_path.chmod(0o644)
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _commit_staged_files(staged: list[tuple[Path, Path]]) -> None:
    """Publish staged files, rolling back destinations if publication fails."""
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, staged_path in staged:
            if destination.exists():
                backup_path = _reserve_backup_path(destination)
                os.replace(destination, backup_path)
                backups[destination] = backup_path
            os.replace(staged_path, destination)
            installed.append(destination)
    except BaseException:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        for destination, backup_path in reversed(list(backups.items())):
            if backup_path.exists():
                destination.unlink(missing_ok=True)
                os.replace(backup_path, destination)
        raise
    else:
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)


def _reserve_backup_path(destination: Path) -> Path:
    """Reserve a unique sibling name without leaving an empty backup file."""
    descriptor, backup_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".bak",
    )
    os.close(descriptor)
    backup_path = Path(backup_name)
    backup_path.unlink()
    return backup_path
