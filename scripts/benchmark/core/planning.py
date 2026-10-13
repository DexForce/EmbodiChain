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

"""Deterministic parameter expansion, run plans and budget accounting."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
import time

from .artifacts import stable_hash
from .contracts import Budget, ExperimentDefinition
from .records import RunSpec

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "MatrixCase",
    "RunPlan",
    "build_run_plan",
    "expand_parameter_matrix",
]


@dataclass(frozen=True)
class MatrixCase:
    """One deterministic point in an experiment parameter matrix."""

    case_id: str
    parameters: Mapping[str, object]
    parameter_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable case identity."""
        return {
            "case_id": self.case_id,
            "parameters": dict(self.parameters),
            "parameter_sha256": self.parameter_sha256,
        }


def expand_parameter_matrix(
    parameter_matrix: Mapping[str, Sequence[object]], *, prefix: str = "case"
) -> tuple[MatrixCase, ...]:
    """Expand a mapping into a stable Cartesian product.

    Keys are sorted before expansion. An empty matrix yields one empty case so
    a domain can express a scalar experiment without a special code path.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a nonempty string")
    keys = tuple(sorted(parameter_matrix))
    values: list[tuple[object, ...]] = []
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("parameter keys must be nonempty strings")
        options = tuple(parameter_matrix[key])
        if not options:
            raise ValueError(f"parameter matrix {key!r} must not be empty")
        values.append(options)
    products = itertools.product(*values) if keys else [()]
    cases = []
    for index, combination in enumerate(products):
        parameters = {key: value for key, value in zip(keys, combination, strict=True)}
        cases.append(
            MatrixCase(
                case_id=f"{prefix}-{index:04d}",
                parameters=parameters,
                parameter_sha256=stable_hash(parameters),
            )
        )
    return tuple(cases)


class BudgetExceeded(RuntimeError):
    """Raised when a run or attempt would exceed a fixed experiment budget."""


@dataclass
class BudgetLedger:
    """Mutable accounting ledger that makes retries consume the same budget."""

    budget: Budget
    runs_started: int = 0
    attempts_started: int = 0
    completed: int = 0
    failed: int = 0
    _started_at: float = field(default_factory=time.monotonic, init=False, repr=False)

    def _check_wall_time(self) -> None:
        if self.budget.wall_time_s is not None and (
            time.monotonic() - self._started_at >= self.budget.wall_time_s
        ):
            raise BudgetExceeded("wall-time budget exhausted")

    def start_run(self) -> None:
        """Reserve one run slot before a worker starts."""
        self._check_wall_time()
        if (
            self.budget.max_runs is not None
            and self.runs_started >= self.budget.max_runs
        ):
            raise BudgetExceeded("run budget exhausted")
        self.runs_started += 1

    def reserve_run(self) -> None:
        """Atomically reserve one run and one attempt for a worker."""
        self._check_wall_time()
        if (
            self.budget.max_runs is not None
            and self.runs_started >= self.budget.max_runs
        ):
            raise BudgetExceeded("run budget exhausted")
        if (
            self.budget.max_attempts is not None
            and self.attempts_started >= self.budget.max_attempts
        ):
            raise BudgetExceeded("attempt budget exhausted")
        self.runs_started += 1
        self.attempts_started += 1

    def start_attempt(self) -> None:
        """Reserve one attempt slot, including retries."""
        self._check_wall_time()
        if (
            self.budget.max_attempts is not None
            and self.attempts_started >= self.budget.max_attempts
        ):
            raise BudgetExceeded("attempt budget exhausted")
        self.attempts_started += 1

    def finish_attempt(self, status: str) -> None:
        """Record an attempt outcome without changing its denominator."""
        if status == "completed":
            self.completed += 1
        else:
            self.failed += 1

    def to_dict(self) -> dict[str, object]:
        """Return conservation counters and configured limits."""
        return {
            "budget": self.budget.to_dict(),
            "runs_started": self.runs_started,
            "attempts_started": self.attempts_started,
            "completed": self.completed,
            "failed": self.failed,
            "remaining_runs": (
                None
                if self.budget.max_runs is None
                else max(0, self.budget.max_runs - self.runs_started)
            ),
            "remaining_attempts": (
                None
                if self.budget.max_attempts is None
                else max(0, self.budget.max_attempts - self.attempts_started)
            ),
        }


@dataclass(frozen=True)
class RunPlan:
    """Frozen matrix-to-worker expansion consumed by the execution layer."""

    experiment_id: str
    cases: tuple[MatrixCase, ...]
    runs: tuple[RunSpec, ...]
    budget: Budget

    def to_dict(self) -> dict[str, object]:
        """Return the plan without environment values."""
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "cases": [case.to_dict() for case in self.cases],
            "budget": self.budget.to_dict(),
            "runs": [
                {
                    "backend": run.backend,
                    "case_id": run.case_id,
                    "repeat": run.repeat,
                    "attempt": run.attempt,
                    "output": str(run.output),
                    "command": list(run.command),
                    "timeout_s": run.timeout_s,
                }
                for run in self.runs
            ],
        }


def build_run_plan(
    definition: ExperimentDefinition,
    *,
    backends: Sequence[str],
    repeats: int,
    output_root: Path,
    command_factory: Callable[[str, MatrixCase, int, Path], Sequence[str]],
    timeout_s: float = 300.0,
) -> RunPlan:
    """Expand an experiment definition into immutable worker invocations."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    backend_tuple = tuple(backends)
    if not backend_tuple or len(set(backend_tuple)) != len(backend_tuple):
        raise ValueError("backends must be nonempty and unique")
    if any(
        not isinstance(backend, str) or not backend.strip() for backend in backend_tuple
    ):
        raise ValueError("backends must contain nonempty strings")
    cases = expand_parameter_matrix(definition.parameter_matrix)
    expected_runs = len(cases) * len(backend_tuple) * repeats
    if (
        definition.budget.max_runs is not None
        and expected_runs > definition.budget.max_runs
    ):
        raise ValueError(
            f"max_runs budget {definition.budget.max_runs} is smaller than {expected_runs} planned runs"
        )
    if (
        definition.budget.max_attempts is not None
        and expected_runs > definition.budget.max_attempts
    ):
        raise ValueError(
            f"max_attempts budget {definition.budget.max_attempts} is smaller than {expected_runs} planned attempts"
        )
    root = Path(output_root).absolute()
    runs: list[RunSpec] = []
    for repeat in range(repeats):
        ordered_backends = (
            backend_tuple if repeat % 2 == 0 else tuple(reversed(backend_tuple))
        )
        for backend in ordered_backends:
            for case in cases:
                run_dir = root / f"r{repeat:02d}_{backend}_{case.case_id}"
                command = tuple(command_factory(backend, case, repeat, run_dir))
                runs.append(
                    RunSpec(
                        backend=backend,
                        repeat=repeat,
                        command=command,
                        output=run_dir,
                        case_id=case.case_id,
                        timeout_s=timeout_s,
                        attempt=0,
                    )
                )
    return RunPlan(definition.experiment_id, cases, tuple(runs), definition.budget)
