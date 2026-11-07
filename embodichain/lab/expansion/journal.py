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

"""Durable logical recipes, execution attempts and independently retryable stages."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import json
import time
from uuid import uuid4
from embodichain.data_pipeline.recording.media import _atomic_json as _atomic_write
from .recipes import ExpansionRecipe
from .contracts import CommitReceipt

__all__ = ["RecipeAttempt", "ExpansionJournal"]


def _atomic_json(path, value):
    _atomic_write(path, value)


@dataclass(frozen=True)
class RecipeAttempt:
    """A new physical execution of a stable logical recipe."""

    recipe: ExpansionRecipe
    attempt_id: int
    sequence: int
    directory: Path


class ExpansionJournal:
    """Single-writer job ledger; the caller must hold the job's process lock.

    The immutable job manifest owns semantic identity. Each invocation records
    its resource limits separately. After a hard process loss, cumulative time
    is a recorded lower bound and is explicitly marked uncertain.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.run_id = None
        self.job_instance_id = None
        self.operation = "all"
        self._started = 0.0
        self._last_checkpoint = 0.0

    def attempts(self) -> list[dict]:
        """Read ordered durable attempt records without inferring success."""
        return [
            json.loads(p.read_text())
            for p in sorted((self.root / "attempts").glob("*/attempt.json"))
        ]

    def open_run(
        self,
        report: Mapping[str, object],
        config: Mapping[str, object],
        *,
        resume: bool,
        operation: str = "all",
    ) -> None:
        """Pin semantic inputs while allowing new execution capacity and budgets."""
        if operation not in {"all", "export"}:
            raise ValueError("invalid run operation")
        self.operation = operation
        manifest = self.root / "job.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("job output already exists; use resume")
            previous = json.loads(manifest.read_text())
            if previous.get("schema_version") != 2 and operation != "export":
                raise ValueError(
                    "legacy job requires the legacy configuration exactly; migrate into a new v2 job before changing execution settings"
                )
            if previous.get("semantic_id", previous["job_id"]) != report["semantic_id"]:
                raise ValueError("job source or physical recipe semantics changed")
        elif resume:
            raise FileNotFoundError("resume requires an existing job manifest")
        else:
            _atomic_json(
                manifest,
                {
                    **report,
                    "schema_version": 2,
                    "job_instance_id": uuid4().hex,
                    "config": dict(config),
                },
            )
        saved = json.loads(manifest.read_text())
        import hashlib

        self.job_instance_id = saved.get(
            "job_instance_id", hashlib.sha256(manifest.read_bytes()).hexdigest()
        )
        for path in (self.root / "runs").glob("*/status.json"):
            record = json.loads(path.read_text())
            if record.get("state") == "running":
                record.update(state="interrupted", time_uncertain=True)
                _atomic_json(path, record)
        self.run_id = uuid4().hex
        _atomic_json(
            self.root / "runs" / self.run_id / "configuration.json",
            {
                "config": dict(config),
                "execution_id": report["execution_id"],
                "operation": operation,
                "semantic_id": report["semantic_id"],
                "render_profile_snapshots": report.get("render_profile_snapshots"),
            },
        )
        _atomic_json(
            self.root / "execution.json",
            {
                "config": dict(config),
                "run_id": self.run_id,
                "render_profile_snapshots": report.get("render_profile_snapshots"),
                "execution_id": report["execution_id"],
            },
        )
        if report.get("render_profile_snapshots") is not None:
            _atomic_json(
                self.root / "runs" / self.run_id / "render_profiles.json",
                {
                    "profiles": {
                        name: value["profile"]
                        for name, value in report["render_profile_snapshots"].items()
                    }
                },
            )
        self._started = time.monotonic()
        self.checkpoint(force=True)

    def time_usage(self) -> tuple[float, bool]:
        """Return accumulated recorded seconds and whether a hard loss occurred."""
        total = 0.0
        uncertain = False
        for path in (self.root / "runs").glob("*/status.json"):
            record = json.loads(path.read_text())
            total += float(record.get("elapsed_seconds", 0))
            uncertain |= record.get("time_uncertain", False)
        if self.run_id:
            record = json.loads(
                (self.root / "runs" / self.run_id / "status.json").read_text()
            )
            if record["state"] == "running":
                total += max(
                    0.0, time.monotonic() - self._started - record["elapsed_seconds"]
                )
        return total, uncertain

    def checkpoint(self, *, force: bool = False, state: str = "running") -> None:
        """Persist bounded-rate runtime usage; normal exceptions close the timer."""
        if self.run_id is None:
            return
        now = time.monotonic()
        if not force and now - self._last_checkpoint < 1.0:
            return
        _atomic_json(
            self.root / "runs" / self.run_id / "status.json",
            {
                "state": state,
                "elapsed_seconds": now - self._started,
                "time_uncertain": False,
            },
        )
        self._last_checkpoint = now

    def update(self, attempt: RecipeAttempt | Path, **values: object) -> dict:
        """Atomically update one existing attempt, retaining its recipe identity."""
        directory = attempt.directory if isinstance(attempt, RecipeAttempt) else attempt
        path = directory / "attempt.json"
        record = json.loads(path.read_text())
        record.update(values)
        _atomic_json(path, record)
        return record

    def reconcile(self, sink: object) -> None:
        """Repair frozen writes before admitting any physical retry."""
        confirmed = {r.commit_id: r for r in sink.inspect()}
        for path in sorted((self.root / "attempts").glob("*/attempt.json")):
            record = json.loads(path.read_text())
            commit = record.get("commit_id")
            receipt = confirmed.get(commit) if commit else None
            if receipt is None and commit:
                receipt = sink.recover(commit)
            if receipt is not None:
                if (
                    not isinstance(receipt, CommitReceipt)
                    or not receipt.confirmed
                    or receipt.commit_id != commit
                    or (
                        record.get("episode_id") is not None
                        and receipt.episode_id != record["episode_id"]
                    )
                ):
                    raise ValueError(
                        "recovery did not return a matching confirmed receipt"
                    )
                record.update(status="committed", storage_id=receipt.storage_id)
            elif record["status"] == "pending_write":
                raise RuntimeError("pending write has lost its frozen artifact")
            elif record["status"] in {"proposed", "prepared", "running", "accepted"}:
                record.update(
                    status="interrupted",
                    executed_steps_lower_bound=(
                        1 if record["status"] == "running" else 0
                    ),
                )
            _atomic_json(path, record)

    def next_batch(
        self,
        factory: Callable[[int], ExpansionRecipe],
        *,
        capacity: int,
        target: int,
        max_attempts: int,
        retry_interrupted: bool = True,
    ) -> tuple[RecipeAttempt, ...]:
        """Allocate bounded attempts, retrying interrupted recipes before new ones."""
        if self.operation != "all":
            raise RuntimeError("export-only recovery cannot allocate physical attempts")
        records = self.attempts()
        if any(
            record["status"]
            in {"proposed", "prepared", "running", "accepted", "pending_write"}
            for record in records
        ):
            raise RuntimeError(
                "reconcile in-flight attempts before allocating another cohort"
            )
        committed = sum(record["status"] == "committed" for record in records)
        count = min(
            capacity, max(0, target - committed), max(0, max_attempts - len(records))
        )
        latest = {record["recipe_id"]: record for record in records}
        retries = (
            [record for record in latest.values() if record["status"] == "interrupted"]
            if retry_interrupted
            else []
        )
        ordinal = max((r["ordinal"] for r in records), default=-1) + 1
        result = []
        for index in range(count):
            if retries:
                prior = retries.pop(0)
                recipe = ExpansionRecipe.from_dict(prior["recipe_spec"])
                if (
                    recipe.recipe_id != prior["recipe_id"]
                    or recipe.recipe_id != factory(prior["ordinal"]).recipe_id
                ):
                    raise ValueError("stored recipe differs from its pinned source")
                number = prior["attempt_id"] + 1
            else:
                recipe = factory(ordinal)
                ordinal += 1
                number = 0
            sequence = len(records) + index
            directory = self.root / "attempts" / f"{sequence:08d}"
            if directory.exists():
                raise FileExistsError("attempt sequence already exists")
            _atomic_json(
                directory / "attempt.json",
                {
                    "sequence": sequence,
                    "ordinal": recipe.context.ordinal,
                    "recipe_id": recipe.recipe_id,
                    "recipe_spec": recipe.to_dict(),
                    "attempt_id": number,
                    "run_id": self.run_id,
                    "status": "proposed",
                    "steps": 0,
                },
            )
            result.append(RecipeAttempt(recipe, number, sequence, directory))
        return tuple(result)

    def stage(
        self, name: str, state: str, *, key: str = "job", **detail: object
    ) -> None:
        """Record a stage independently of whether collection reached its target."""
        import hashlib

        path = (
            self.root
            / "stages"
            / name
            / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        )
        record = {
            "stage": name,
            "key": key,
            "state": state,
            "run_id": self.run_id,
            **detail,
        }
        _atomic_json(path, record)
        if self.run_id:
            _atomic_json(
                self.root
                / "runs"
                / self.run_id
                / "stage_history"
                / (f"{name}-{time.time_ns()}.json"),
                record,
            )
