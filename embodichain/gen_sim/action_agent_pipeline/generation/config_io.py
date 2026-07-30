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
import re
import tempfile
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    AGENT_CONFIG_FILENAME,
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    COMPILED_GRAPH_FILENAME,
    FAST_GYM_CONFIG_FILENAME,
    SEED_TASK_GRAPH_FILENAME,
    SEED_TASK_GRAPH_PNG_FILENAME,
    TASK_GRAPH_FILENAME,
    TASK_GRAPH_PNG_FILENAME,
    TASK_PROMPT_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    GeneratedActionAgentConfigPaths,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.graph_visualization import (
    render_seed_task_graph_png,
)

__all__ = [
    "read_json",
    "raise_if_generated_files_exist",
    "write_config_bundle",
    "write_json",
    "write_text",
]

_SAFE_TASK_DIR_RE = re.compile(r"[^0-9A-Za-z._-]+")


def write_config_bundle(
    *,
    output_dir: Path,
    bundle: Mapping[str, Any],
    overwrite: bool,
    graph_output_root: Path | None = None,
) -> GeneratedActionAgentConfigPaths:
    """Write Seed v5 runtime inputs and review diagnostics as one bundle."""
    task_name = _bundle_task_name(bundle, fallback=output_dir.name)
    graph_output_dir = _resolve_graph_output_root(graph_output_root) / _safe_task_dir(
        task_name
    )
    paths = GeneratedActionAgentConfigPaths(
        output_dir=output_dir,
        graph_output_dir=graph_output_dir,
        gym_config=output_dir / FAST_GYM_CONFIG_FILENAME,
        agent_config=output_dir / AGENT_CONFIG_FILENAME,
        task_prompt=output_dir / TASK_PROMPT_FILENAME,
        seed_task_graph=output_dir / SEED_TASK_GRAPH_FILENAME,
        seed_task_graph_png=graph_output_dir / SEED_TASK_GRAPH_PNG_FILENAME,
        basic_background=output_dir / BASIC_BACKGROUND_FILENAME,
        atom_actions=output_dir / ATOM_ACTIONS_FILENAME,
        summary=dict(bundle.get("summary", {})),
    )
    raise_if_generated_files_exist(
        output_dir,
        overwrite,
        task_name=task_name,
        graph_output_root=graph_output_root,
    )
    _validate_seed_bundle(bundle)

    serialized_files: list[tuple[Path, str | bytes]] = [
        (paths.gym_config, _serialize_json(bundle["gym_config"])),
        (paths.agent_config, _serialize_json(bundle["agent_config"])),
        (paths.task_prompt, _serialize_text(bundle["task_prompt"])),
    ]
    serialized_files.append(
        (paths.seed_task_graph, _serialize_json(bundle["seed_task_graph"]))
    )
    try:
        seed_graph_png = render_seed_task_graph_png(bundle["seed_task_graph"])
    except Exception as error:
        raise RuntimeError("Failed to render seed_task_graph.png.") from error
    serialized_files.append((paths.seed_task_graph_png, seed_graph_png))
    serialized_files.extend(
        [
            (paths.basic_background, _serialize_text(bundle["basic_background"])),
            (paths.atom_actions, _serialize_text(bundle["atom_actions"])),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(path.parent == graph_output_dir for path, _ in serialized_files):
        graph_output_dir.mkdir(parents=True, exist_ok=True)
    _write_file_transaction(serialized_files)
    if overwrite:
        # These legacy artifacts are not valid runtime inputs for Seed v5. Remove
        # them only after the complete replacement bundle has been published.
        for obsolete in (
            output_dir / SEED_TASK_GRAPH_PNG_FILENAME,
            output_dir / TASK_GRAPH_FILENAME,
            output_dir / TASK_GRAPH_PNG_FILENAME,
            output_dir / COMPILED_GRAPH_FILENAME,
            graph_output_dir / TASK_GRAPH_PNG_FILENAME,
        ):
            obsolete.unlink(missing_ok=True)
    return paths


def raise_if_generated_files_exist(
    output_dir: Path,
    overwrite: bool,
    task_name: str | None = None,
    *,
    graph_output_root: Path | None = None,
) -> None:
    if overwrite:
        return
    output_files = [
        output_dir / FAST_GYM_CONFIG_FILENAME,
        output_dir / AGENT_CONFIG_FILENAME,
        output_dir / TASK_PROMPT_FILENAME,
        output_dir / SEED_TASK_GRAPH_FILENAME,
        output_dir / SEED_TASK_GRAPH_PNG_FILENAME,
        output_dir / TASK_GRAPH_FILENAME,
        output_dir / TASK_GRAPH_PNG_FILENAME,
        output_dir / COMPILED_GRAPH_FILENAME,
        output_dir / BASIC_BACKGROUND_FILENAME,
        output_dir / ATOM_ACTIONS_FILENAME,
    ]
    if task_name:
        graph_output_dir = _resolve_graph_output_root(
            graph_output_root
        ) / _safe_task_dir(task_name)
        output_files.extend(
            [
                graph_output_dir / SEED_TASK_GRAPH_PNG_FILENAME,
                graph_output_dir / TASK_GRAPH_PNG_FILENAME,
            ]
        )
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


def _validate_seed_bundle(bundle: Mapping[str, Any]) -> None:
    """Require the executable Seed v5 that is the sole config-stage graph."""
    seed_graph = bundle.get("seed_task_graph")
    if not isinstance(seed_graph, Mapping):
        raise TypeError("seed_task_graph bundle entry must be a mapping.")
    validate_seed_task_graph(seed_graph)
    if "task_graph" in bundle:
        raise ValueError(
            "Config generation must not publish task_graph.json. Runtime creates "
            "one grounded Task graph per environment and episode."
        )


def _serialize_json(data: Mapping[str, Any]) -> str:
    """Serialize JSON before touching any destination file."""
    return json.dumps(data, ensure_ascii=False, indent=4) + "\n"


def _serialize_text(content: str) -> str:
    """Normalize one generated text artifact before staging it."""
    return content.rstrip() + "\n"


def _bundle_task_name(bundle: Mapping[str, Any], *, fallback: str) -> str:
    for graph_key in ("seed_task_graph",):
        graph = bundle.get(graph_key)
        if isinstance(graph, Mapping):
            task_name = graph.get("task")
            if isinstance(task_name, str) and task_name.strip():
                return task_name.strip()
    if fallback.strip():
        return fallback.strip()
    raise ValueError("Config bundle requires a task name for graph visualization.")


def _safe_task_dir(task_name: str) -> str:
    safe_name = _SAFE_TASK_DIR_RE.sub("_", task_name).strip("._")
    if not safe_name:
        raise ValueError(
            "Task name does not contain a safe graph output directory name."
        )
    return safe_name


def _resolve_graph_output_root(graph_output_root: Path | None) -> Path:
    if graph_output_root is not None:
        return Path(graph_output_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent / "outputs" / "graph"
    raise RuntimeError("Unable to resolve the repository outputs/graph directory.")


def _write_file_transaction(files: list[tuple[Path, str | bytes]]) -> None:
    """Replace a group of files and restore the old group on ordinary failure.

    Every new file is fully written and fsynced in its destination directory
    before any public path changes. Each ``os.replace`` is atomic on a single
    filesystem; backups provide rollback if a later replacement raises.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in files:
            staged.append((destination, _stage_file(destination, content)))
        _commit_staged_files(staged)
    finally:
        for _, staged_path in staged:
            staged_path.unlink(missing_ok=True)


def _stage_file(destination: Path, content: str | bytes) -> Path:
    """Write and fsync one text or binary temporary file beside its destination."""
    is_binary = isinstance(content, bytes)
    descriptor, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=not is_binary,
    )
    temp_path = Path(temp_name)
    try:
        mode = "wb" if is_binary else "w"
        with os.fdopen(
            descriptor,
            mode,
            encoding=None if is_binary else "utf-8",
        ) as file:
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
