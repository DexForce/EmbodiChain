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

"""Directory ownership shared by supervised training entry points."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

__all__: list[str] = []


class _RunArtifacts:
    """Bind training artifacts to a durable run identity, independent of paths.

    This is an admission check, not a multi-writer lock. A caller owns one
    training writer; concurrent distributed writers must coordinate externally.
    Metadata validation never deserializes an existing destination checkpoint.
    """

    def __init__(self, output: Path) -> None:
        self.output = output
        self.run_id = uuid.uuid4().hex
        self._marker = output / "run.json"
        self._claimed = False

    def _occupied(self) -> bool:
        return (
            any(
                path.exists()
                for path in (
                    self.output / "metrics.jsonl",
                    self.output / "last.pt",
                    self.output / "best.pt",
                )
            )
            or next(self.output.glob("step_*.pt"), None) is not None
        )

    def _read_identity(self) -> str:
        try:
            record = json.loads(self._marker.read_text())
            identity = record["run_id"]
            if (
                record.get("version") != 1
                or not isinstance(identity, str)
                or not identity
            ):
                raise ValueError("Invalid training run identity")
            return identity
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                "Invalid training run identity; use a fresh output directory"
            ) from error

    def claim(self) -> None:
        """Admit one run without replacing another run's artifacts."""
        if self._claimed:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        if self._marker.exists():
            if self._read_identity() != self.run_id:
                raise FileExistsError(
                    "Training output belongs to another run; use a fresh output directory"
                )
            self._claimed = True
            return
        if self._occupied():
            raise FileExistsError(
                "Training output contains another run; use a fresh output directory"
            )
        try:
            with self._marker.open("x") as stream:
                json.dump({"version": 1, "run_id": self.run_id}, stream)
                stream.write("\n")
        except FileExistsError:
            if self._read_identity() != self.run_id:
                raise FileExistsError(
                    "Training output belongs to another run; use a fresh output directory"
                ) from None
        self._claimed = True

    def validate_resume(self, saved_run_id: str | None) -> str:
        """Check destination identity without mutating files or runtime state.

        Checkpoints without an identity can be recovered into an artifact-free
        destination and acquire an identity with their next save. Existing
        directories require an explicit matching run marker and checkpoint ID.
        """
        if saved_run_id is not None and (
            not isinstance(saved_run_id, str) or not saved_run_id
        ):
            raise ValueError("Invalid checkpoint run identity")
        occupied = self._marker.exists() or self._occupied()
        if occupied and (
            saved_run_id is None
            or not self._marker.exists()
            or self._read_identity() != saved_run_id
        ):
            raise FileExistsError(
                "Resume destination belongs to another run; use a fresh output directory"
            )
        return saved_run_id or self.run_id

    def adopt_resume(self, run_id: str) -> None:
        """Adopt an identity only after all checkpoint checks have succeeded."""
        self.run_id = run_id
        self._claimed = False

    def validate_history(self, committed_step: int, *, key: str) -> None:
        """Keep existing history intact when admitting a checkpoint boundary."""
        if type(committed_step) is not int or committed_step < 0:
            raise ValueError("Invalid checkpoint update cursor")
        path = self.output / "metrics.jsonl"
        if not path.exists():
            return
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    step = json.loads(line)[key]
                    if type(step) is not int or step < 0:
                        raise ValueError("Invalid history update cursor")
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        "Training output history is incomplete or invalid; use a fresh output directory"
                    ) from error
                if step > committed_step:
                    raise ValueError(
                        "Training output history is ahead of this checkpoint; use a fresh output directory"
                    )
