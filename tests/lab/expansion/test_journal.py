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

"""Recovery separates logical recipe retries, resource revisions and frozen writes."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import pytest
from embodichain.lab.expansion.journal import ExpansionJournal
from embodichain.lab.expansion.contracts import CommitReceipt
from embodichain.lab.expansion.recipes import (
    ExpansionRecipe,
    ExpansionContext,
    BatchCompatibility,
    RecipeOperation,
    OperatorCapability,
)


def recipe(ordinal):
    return ExpansionRecipe(
        ExpansionContext("task_program", "pinned", 7, ordinal),
        BatchCompatibility(
            "arm",
            ("joint",),
            "topology",
            "program",
            "default",
            0.01,
            "schema",
            "lights",
        ),
        (
            RecipeOperation(
                "mass",
                "1",
                OperatorCapability("physics", "prepare", "environment"),
                '{"cube":1.2}',
            ),
        ),
    )


def test_interrupted_recipe_retries_with_new_attempt_after_capacity_change(tmp_path):
    report = {"job_id": "sem", "semantic_id": "sem", "execution_id": "small"}
    first = ExpansionJournal(tmp_path)
    first.open_run(report, {"batch_size": 1}, resume=False)
    attempts = first.next_batch(recipe, capacity=2, target=3, max_attempts=5)
    first.update(attempts[0], status="running")
    first.update(attempts[1], status="committed", commit_id="confirmed")
    next_run = ExpansionJournal(tmp_path)
    next_run.open_run(
        {**report, "execution_id": "large"}, {"batch_size": 8}, resume=True
    )
    sink = SimpleNamespace(
        inspect=lambda: (
            CommitReceipt(
                episode_id="episode",
                candidate_id="candidate",
                attempt_id=0,
                commit_id="confirmed",
                scene_case_id="scene",
                submission_id=0,
                storage_id="frozen",
            ),
        ),
        recover=lambda _: None,
    )
    next_run.reconcile(sink)
    remaining = next_run.next_batch(recipe, capacity=8, target=3, max_attempts=5)
    assert remaining[0].recipe.recipe_id == attempts[0].recipe.recipe_id
    assert remaining[0].attempt_id == 1 and remaining[0].sequence == 2
    assert remaining[1].recipe.context.ordinal == 2 and remaining[1].attempt_id == 0
    assert next_run.time_usage()[1]
    with pytest.raises(RuntimeError, match="in-flight"):
        next_run.next_batch(recipe, capacity=1, target=3, max_attempts=5)
    next_run.checkpoint(force=True, state="closed")


def test_changed_semantics_cannot_reuse_job(tmp_path):
    ledger = ExpansionJournal(tmp_path)
    ledger.open_run(
        {"job_id": "a", "semantic_id": "a", "execution_id": "r"}, {}, resume=False
    )
    with pytest.raises(ValueError, match="semantics"):
        ledger.open_run(
            {"job_id": "b", "semantic_id": "b", "execution_id": "r"}, {}, resume=True
        )


def test_missing_frozen_pending_artifact_blocks_new_attempts(tmp_path):
    ledger = ExpansionJournal(tmp_path)
    ledger.open_run(
        {"job_id": "a", "semantic_id": "a", "execution_id": "r"}, {}, resume=False
    )
    attempt = ledger.next_batch(recipe, capacity=1, target=1, max_attempts=2)[0]
    ledger.update(attempt, status="pending_write", commit_id="lost")
    with pytest.raises(RuntimeError, match="lost its frozen"):
        ledger.reconcile(SimpleNamespace(inspect=lambda: (), recover=lambda _: None))


def test_recipe_roundtrip_and_scope_rejection_are_independent_of_slot():
    original = recipe(13)
    decoded = ExpansionRecipe.from_dict(original.to_dict())
    assert decoded == original and decoded.recipe_id == original.recipe_id
    with pytest.raises(ValueError, match="scope"):
        decoded.validate_host(
            {"mass": OperatorCapability("physics", "prepare", "world")},
            decoded.compatibility,
        )
    with pytest.raises(ValueError, match="World"):
        decoded.validate_host(
            {"mass": decoded.operations[0].capability},
            replace(decoded.compatibility, world_profile="different"),
        )
