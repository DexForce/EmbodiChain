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

"""Transactional publication for Task Engine artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from embodichain.gen_sim.action_engine.runtime import (
    EXECUTION_REPORT_FILENAME,
    write_execution_report as _write_execution_report,
)

__all__ = [
    "BINDING_REPORT_FILENAME",
    "CONSERVATIVE_SCENE_GRAPH_FILENAME",
    "EXECUTION_REPORT_FILENAME",
    "GROUNDED_TASK_PLAN_FILENAME",
    "FEASIBILITY_REPORT_FILENAME",
    "FINAL_SCENE_INSPECTION_FILENAME",
    "PREPARATION_FAILURE_FILENAME",
    "ROLE_BINDINGS_FILENAME",
    "SCENE_MANIFEST_FILENAME",
    "STATIC_SCENE_MANIFEST_FILENAME",
    "SUCCESS_SPEC_FILENAME",
    "TASK_CANDIDATE_SET_FILENAME",
    "TASK_DRAFT_FILENAME",
    "SCENE_REQUEST_FILENAME",
    "ArtifactTransaction",
    "TaskEngineArtifactPaths",
    "task_engine_artifact_paths",
    "write_task_engine_artifacts",
    "write_execution_report",
    "write_preparation_failure",
]


TASK_CANDIDATE_SET_FILENAME = "task_candidate_set.json"
TASK_DRAFT_FILENAME = "task_draft.json"
SCENE_REQUEST_FILENAME = "scene_request.json"
SUCCESS_SPEC_FILENAME = "success_spec.json"
SCENE_MANIFEST_FILENAME = "scene_manifest.json"
STATIC_SCENE_MANIFEST_FILENAME = "static_scene_manifest.json"
CONSERVATIVE_SCENE_GRAPH_FILENAME = "conservative_scene_graph.json"
ROLE_BINDINGS_FILENAME = "role_bindings.json"
BINDING_REPORT_FILENAME = "binding_report.json"
FEASIBILITY_REPORT_FILENAME = "feasibility_report.json"
FINAL_SCENE_INSPECTION_FILENAME = "final_scene_inspection.json"
GROUNDED_TASK_PLAN_FILENAME = "grounded_task_plan.json"
PREPARATION_FAILURE_FILENAME = "preparation_failure.json"


@dataclass(frozen=True)
class TaskEngineArtifactPaths:
    """Canonical Task Engine paths rooted at one published bundle."""

    root: Path
    task_candidate_set: Path
    task_draft: Path
    scene_request: Path
    success_spec: Path
    scene_manifest: Path
    static_scene_manifest: Path
    conservative_scene_graph: Path
    role_bindings: Path
    binding_report: Path
    feasibility_report: Path
    final_scene_inspection: Path
    grounded_task_plan: Path
    preparation_failure: Path
    execution_report: Path


def task_engine_artifact_paths(
    output_dir: str | Path,
) -> TaskEngineArtifactPaths:
    """Return all Task Engine paths without creating the directory."""
    root = Path(output_dir).expanduser().resolve()
    return TaskEngineArtifactPaths(
        root=root,
        task_candidate_set=root / TASK_CANDIDATE_SET_FILENAME,
        task_draft=root / TASK_DRAFT_FILENAME,
        scene_request=root / SCENE_REQUEST_FILENAME,
        success_spec=root / SUCCESS_SPEC_FILENAME,
        scene_manifest=root / SCENE_MANIFEST_FILENAME,
        static_scene_manifest=root / STATIC_SCENE_MANIFEST_FILENAME,
        conservative_scene_graph=root / CONSERVATIVE_SCENE_GRAPH_FILENAME,
        role_bindings=root / ROLE_BINDINGS_FILENAME,
        binding_report=root / BINDING_REPORT_FILENAME,
        feasibility_report=root / FEASIBILITY_REPORT_FILENAME,
        final_scene_inspection=root / FINAL_SCENE_INSPECTION_FILENAME,
        grounded_task_plan=root / GROUNDED_TASK_PLAN_FILENAME,
        preparation_failure=root / PREPARATION_FAILURE_FILENAME,
        execution_report=root / EXECUTION_REPORT_FILENAME,
    )


class ArtifactTransaction:
    """Build a complete bundle beside its destination and publish it by rename."""

    def __init__(self, output_dir: str | Path, *, overwrite: bool = False) -> None:
        raw = Path(output_dir).expanduser()
        self.output_dir = (
            (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw.resolve()
        )
        self.overwrite = bool(overwrite)
        self.staging_dir: Path | None = None
        self._committed = False

    def __enter__(self) -> "ArtifactTransaction":
        destination = self.output_dir
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {destination}. "
                "Pass overwrite=True to replace it."
            )
        self.staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.staging-",
                dir=destination.parent,
            )
        )
        return self

    def commit(self) -> Path:
        """Rewrite staging-local absolute paths, then atomically publish."""
        if self.staging_dir is None:
            raise RuntimeError("ArtifactTransaction has not been entered.")
        if self._committed:
            raise RuntimeError("ArtifactTransaction has already been committed.")
        staging = self.staging_dir
        destination = self.output_dir
        _relocate_json_paths(staging, destination)

        backup: Path | None = None
        if destination.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {destination}."
                )
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup-",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        else:
            self._committed = True
            self.staging_dir = None
            if backup is not None:
                _remove_path(backup)
            _fsync_directory(destination.parent)
        return destination

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.staging_dir is not None and self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        return False


def write_task_engine_artifacts(
    output_dir: str | Path,
    *,
    candidate_set: Mapping[str, Any],
    scene_manifest: Mapping[str, Any] | None,
    role_bindings: Mapping[str, Any] | None,
    binding_report: Mapping[str, Any],
    grounded_task_plan: Mapping[str, Any] | None = None,
    static_scene_manifest: Mapping[str, Any] | None = None,
    conservative_scene_graph: Mapping[str, Any] | None = None,
    feasibility_report: Mapping[str, Any] | None = None,
    final_scene_inspection: Mapping[str, Any] | None = None,
) -> TaskEngineArtifactPaths:
    """Write Task Engine protocols into an unpublished staging directory.

    An unsuccessful adaptation can omit SceneManifest and RoleBindings rather
    than publishing protocol filenames whose payloads do not satisfy their
    schemas.
    """
    paths = task_engine_artifact_paths(output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_json(paths.task_candidate_set, candidate_set)
    if scene_manifest is not None:
        _write_json(paths.scene_manifest, scene_manifest)
    if static_scene_manifest is not None:
        _write_json(paths.static_scene_manifest, static_scene_manifest)
    if conservative_scene_graph is not None:
        _write_json(paths.conservative_scene_graph, conservative_scene_graph)
    if role_bindings is not None:
        _write_json(paths.role_bindings, role_bindings)
    _write_json(paths.binding_report, binding_report)
    if feasibility_report is not None:
        _write_json(paths.feasibility_report, feasibility_report)
    if final_scene_inspection is not None:
        _write_json(paths.final_scene_inspection, final_scene_inspection)

    if grounded_task_plan is not None:
        _write_json(paths.grounded_task_plan, grounded_task_plan)
        _write_json(paths.task_draft, grounded_task_plan["task_draft"])
        candidate_id = grounded_task_plan["selected_candidate_id"]
        selected = next(
            candidate
            for candidate in candidate_set["candidates"]
            if candidate["candidate_id"] == candidate_id
        )
        _write_json(paths.scene_request, selected["scene_request"])
        _write_json(paths.success_spec, grounded_task_plan["success_spec"])
    return paths


def write_execution_report(output_dir: str | Path, value: Any) -> Path:
    """Publish through the Action Engine-owned report boundary."""
    return _write_execution_report(output_dir, value)


def write_preparation_failure(output_dir: str | Path, value: Any) -> Path:
    """Write a strict-JSON audit for a failed candidate planning transaction."""
    path = task_engine_artifact_paths(output_dir).preparation_failure
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)
    return path


def _write_json(path: Path, value: Any) -> None:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Artifact {path.name} is not strict JSON data.") from exc
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _relocate_json_paths(staging: Path, destination: Path) -> None:
    """Replace staging-root absolute paths embedded by the legacy generator."""
    source_prefix = staging.resolve().as_posix()
    destination_prefix = destination.resolve().as_posix()

    def relocate(value: Any) -> Any:
        if isinstance(value, str):
            if value == source_prefix:
                return destination_prefix
            if value.startswith(source_prefix + "/"):
                return destination_prefix + value[len(source_prefix) :]
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {
                relocate(key) if isinstance(key, str) else key: relocate(item)
                for key, item in value.items()
            }
        return value

    for path in staging.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generated artifact is invalid JSON: {path}") from exc
        relocated = relocate(value)
        if relocated != value:
            _write_json(path, relocated)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
