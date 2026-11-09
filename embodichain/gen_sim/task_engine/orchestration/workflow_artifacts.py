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

"""Stable run-path ownership, separate from sealed declaration publication."""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

__all__ = ["validate_run_output_directory", "write_run_json"]


def validate_run_output_directory(path: str | Path) -> Path:
    """Validate a run-output directory without creating or modifying it.

    Args:
        path: Caller-owned execution or workflow output directory.

    Returns:
        Canonical absolute path after checking direct and resolved ancestors.
        A directory inside a sealed bundle, including through a symbolic link,
        rejects. Callers additionally enforce fresh-attempt and non-overwrite
        rules for their execution receipts and reports.
    """
    selected = Path(path).expanduser().absolute()
    _assert_unsealed_path(selected / ".run-output")
    if selected.exists() and not selected.is_dir():
        raise NotADirectoryError(selected)
    return selected.resolve()


class WorkflowRunArtifacts:
    """Keep bundle paths stable while publishing the final run manifest last.

    Bundles are individually atomically published by their owner. A run itself
    contains mutable attempt/report files, so moving it after sealing would
    invalidate external input paths. Existing sealed run histories never overwrite.
    """

    def __init__(self, output_dir: str | Path, *, overwrite: bool = False) -> None:
        self.output_dir = validate_run_output_directory(output_dir)
        self.overwrite = overwrite
        self.staging_dir: Path | None = None
        self._backup: Path | None = None
        self._committed = False

    def __enter__(self) -> WorkflowRunArtifacts:
        destination = self.output_dir
        _assert_unsealed_path(destination / "run_manifest.json")
        if destination.is_symlink():
            raise ValueError("A workflow output directory cannot be a symbolic link.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {destination}."
                )
            if not destination.is_dir():
                raise NotADirectoryError(destination)
            if any(destination.rglob("integration_fingerprint.json")):
                raise ValueError(
                    "A workflow containing sealed bundles cannot be overwritten; select a new run directory."
                )
            self._backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.previous-", dir=destination.parent
                )
            )
            self._backup.rmdir()
            os.replace(destination, self._backup)
        try:
            destination.mkdir(exist_ok=False)
        except BaseException:
            if self._backup is not None and not destination.exists():
                os.replace(self._backup, destination)
            raise
        self.staging_dir = destination
        return self

    def commit(self) -> Path:
        """Complete run publication without relocating any declaration bytes."""
        if self.staging_dir is None or self._committed:
            raise RuntimeError("Workflow run artifacts are not an active publication.")
        if not (self.output_dir / "run_manifest.json").is_file():
            raise ValueError("Workflow completion requires its final run manifest.")
        self._committed = True
        if self._backup is not None:
            shutil.rmtree(self._backup)
        return self.output_dir

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> bool:
        if self._committed:
            return False
        destination = self.output_dir
        if destination.exists() and any(
            destination.rglob("integration_fingerprint.json")
        ):
            # A cancelled run is retained rather than deleting/moving sealed inputs.
            record = dict(
                schema_version="task_engine.workflow_interruption/v1",
                status="interrupted",
                error_type=None if kind is None else kind.__name__,
                message=None if error is None else str(error),
                previous_output=None if self._backup is None else str(self._backup),
            )
            write_run_json(destination / "workflow_interruption.json", record)
            return False
        if destination.exists():
            shutil.rmtree(destination)
        if self._backup is not None and self._backup.exists():
            os.replace(self._backup, destination)
        return False


def write_run_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    """Atomically refresh one run-owned audit, never a sealed declaration file.

    Args:
        path: Explicit run-owned JSON destination, outside all sealed bundles.
        value: JSON-serializable audit. Non-finite values reject before writing.
        overwrite: Whether an existing run-owned audit may be refreshed. False
            publishes a fresh receipt/report atomically and rejects an existing file.

    Existing run-owned audits may be refreshed. Execution hosts retain their
    fresh-run and receipt/report non-overwrite checks.
    """
    path = Path(path).expanduser().absolute()
    _assert_unsealed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = None
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
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _assert_unsealed_path(path: Path) -> None:
    path = path.expanduser()
    for candidate in (path.absolute(), path.resolve()):
        for parent in candidate.parents:
            if (parent / "integration_fingerprint.json").is_file():
                raise ValueError(
                    "A run audit cannot replace a sealed bundle declaration."
                )
