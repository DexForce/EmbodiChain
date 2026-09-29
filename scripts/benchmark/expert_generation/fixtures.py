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

"""Deterministic G-03 fixture executor for protocol and report validation."""

from __future__ import annotations

from .contracts import GenerationAttempt, GenerationCase, PersistenceReceipt

__all__ = ["fixture_executor"]


def fixture_executor(case: GenerationCase, attempt_id: int) -> GenerationAttempt:
    """Return a fixed sequence of accepted and rejected generation outcomes.

    This fixture exercises execution, measured validation, task acceptance and
    persistence confirmation without constructing a simulator or a production
    candidate coordinator.
    """
    if attempt_id == 1:
        return GenerationAttempt(
            case,
            attempt_id,
            "completed",
            "failed",
            "not_evaluated",
            elapsed_s=0.2,
            failure_reason="measured_validation_failed",
        )
    if attempt_id == 2:
        receipt = PersistenceReceipt(
            episode_id=f"{case.case_id}-episode-{attempt_id}",
            commit_id=f"{case.case_id}-commit-{attempt_id}",
            submission_id=0,
            storage_id="fixture-storage",
            confirmed=False,
            error="fixture_storage_unavailable",
        )
        return GenerationAttempt(
            case,
            attempt_id,
            "completed",
            "passed",
            "passed",
            receipt=receipt,
            elapsed_s=0.3,
            failure_reason="fixture_storage_unavailable",
        )
    receipt = PersistenceReceipt(
        episode_id=f"{case.case_id}-episode-{attempt_id}",
        commit_id=f"{case.case_id}-commit-{attempt_id}",
        submission_id=0,
        storage_id="fixture-storage",
        confirmed=True,
    )
    return GenerationAttempt(
        case,
        attempt_id,
        "completed",
        "passed",
        "passed",
        receipt=receipt,
        elapsed_s=0.1,
        metrics={"trajectory_steps": 16, "task_completion_time_s": 0.1},
    )
