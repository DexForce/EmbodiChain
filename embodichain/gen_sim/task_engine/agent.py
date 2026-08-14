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

"""Scene-independent Task Agent for the first collaboration workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .contracts import (
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    TaskCandidate,
    TaskCandidateSet,
    canonical_hash,
    validate_task_candidate,
    validate_task_candidate_set,
)
from .interpretation import (
    InstructionCaller,
    InstructionDraftResult,
    interpret_instruction_draft,
    validate_instruction_intent,
)
from .ontology import TASK_CONTRACTS, task_success_type

__all__ = [
    "TaskAgent",
    "TaskGenerationError",
    "derive_scene_request",
    "derive_success_spec",
]

DraftInterpreter = Callable[..., InstructionDraftResult]


class TaskGenerationError(ValueError):
    """Raised when every independently generated candidate fails validation."""


@dataclass(frozen=True)
class _CandidateAttempt:
    index: int
    result: InstructionDraftResult | None = None
    error: str = ""


class TaskAgent:
    """Generate, validate, normalize, and vote on independent task drafts."""

    def __init__(
        self,
        *,
        caller: InstructionCaller | None = None,
        interpreter: DraftInterpreter = interpret_instruction_draft,
    ) -> None:
        self._caller = caller
        self._interpreter = interpreter

    def generate(
        self,
        task_id: str,
        instruction: str,
        model: str | None = None,
        candidate_count: int = 3,
    ) -> TaskCandidateSet:
        """Generate candidates concurrently and retain votes after deduplication."""
        normalized_task_id = str(task_id).strip()
        normalized_instruction = str(instruction).strip()
        if not normalized_task_id or not normalized_instruction:
            raise ValueError("task_id and instruction must be non-empty.")
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 1
        ):
            raise ValueError("candidate_count must be a positive integer.")

        attempts: list[_CandidateAttempt] = []
        with ThreadPoolExecutor(
            max_workers=candidate_count,
            thread_name_prefix="task-agent",
        ) as executor:
            futures = {
                executor.submit(
                    self._interpreter,
                    normalized_instruction,
                    model=model,
                    caller=self._caller,
                ): index
                for index in range(candidate_count)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    attempts.append(
                        _CandidateAttempt(index=index, result=future.result())
                    )
                except Exception as error:  # Each candidate is an isolated vote.
                    attempts.append(
                        _CandidateAttempt(
                            index=index,
                            error=f"candidate_{index + 1:02d}: {type(error).__name__}: {error}",
                        )
                    )
        attempts.sort(key=lambda item: item.index)
        errors = [item.error for item in attempts if item.result is None]
        unique: dict[str, TaskCandidate] = {}
        valid_response_count = 0
        for attempt in attempts:
            if attempt.result is None:
                continue
            assert attempt.result is not None
            try:
                canonical_intent = _canonicalize_intent(attempt.result.intent)
                draft = {
                    "schema_version": TASK_DRAFT_SCHEMA,
                    "task_id": normalized_task_id,
                    "instruction": normalized_instruction,
                    "steps": canonical_intent["steps"],
                }
                semantic_hash = canonical_hash(draft["steps"])
                candidate_id = f"candidate_{len(unique) + 1:02d}"
                candidate = validate_task_candidate(
                    {
                        "candidate_id": candidate_id,
                        "draft": draft,
                        "scene_request": derive_scene_request(draft),
                        "success_spec": derive_success_spec(draft),
                        "semantic_hash": semantic_hash,
                        "vote_count": 1,
                        "attempts": attempt.result.attempts,
                        "normalizations": deepcopy(list(attempt.result.normalizations)),
                    }
                )
                existing = unique.get(semantic_hash)
                if existing is not None:
                    existing["vote_count"] += 1
                    existing["attempts"] = max(
                        existing["attempts"], attempt.result.attempts
                    )
                    existing["normalizations"].extend(candidate["normalizations"])
                else:
                    unique[semantic_hash] = candidate
                valid_response_count += 1
            except Exception as error:  # Post-processing failures stay candidate-local.
                errors.append(
                    f"candidate_{attempt.index + 1:02d}: "
                    f"{type(error).__name__}: {error}"
                )

        if not unique:
            raise TaskGenerationError(
                "All Task Agent candidates failed validation: " + "; ".join(errors)
            )

        return validate_task_candidate_set(
            {
                "schema_version": TASK_CANDIDATE_SET_SCHEMA,
                "task_id": normalized_task_id,
                "instruction": normalized_instruction,
                "candidates": list(unique.values()),
                "requested_candidate_count": candidate_count,
                "valid_response_count": valid_response_count,
                "errors": errors,
            }
        )


def derive_scene_request(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Derive structural scene constraints without classifying reference text."""
    from .contracts import validate_scene_request, validate_task_draft

    normalized = validate_task_draft(draft)
    references: list[dict[str, Any]] = []
    for step in normalized["steps"]:
        task_type = str(step["task_type"])
        contract = TASK_CONTRACTS[task_type]
        for role in ("object", "target"):
            selector = step[role]
            if selector["kind"] != "scene_ref":
                continue
            if role == "object":
                structure = contract.source_structure
                affordances = sorted(contract.scene_affordances)
                initial_state = {"orientation": "fallen"} if task_type == "E2" else {}
                attributes: dict[str, Any] = {}
            else:
                structure = _target_structure(task_type, str(step["relation"]))
                affordances = _target_affordances(task_type, str(step["relation"]))
                initial_state = {}
                attributes = {}
            references.append(
                {
                    "reference_id": f"{step['id']}.{role}",
                    "step_id": step["id"],
                    "role": role,
                    "reference": selector["reference"],
                    "quantifier": selector["quantifier"],
                    "count": selector["count"],
                    "source_structure": structure,
                    "affordances": affordances,
                    "initial_state": initial_state,
                    "attributes": attributes,
                }
            )
    if not references:
        raise ValueError("A TaskDraft must contain at least one scene_ref selector.")
    return validate_scene_request(
        {
            "schema_version": SCENE_REQUEST_SCHEMA,
            "task_id": normalized["task_id"],
            "references": references,
        }
    )


def derive_success_spec(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Derive every success term exclusively from the E-task ontology."""
    from .contracts import validate_success_spec, validate_task_draft

    normalized = validate_task_draft(draft)
    return validate_success_spec(
        {
            "schema_version": SUCCESS_SPEC_SCHEMA,
            "task_id": normalized["task_id"],
            "op": "all",
            "terms": [
                {
                    "step_id": step["id"],
                    "type": task_success_type(step["task_type"], step),
                }
                for step in normalized["steps"]
            ],
        },
        draft=normalized,
    )


def _canonicalize_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Remove arbitrary model step IDs while preserving the explicit DAG order."""
    normalized = validate_instruction_intent(intent)
    id_map = {
        step["id"]: f"step_{index + 1:02d}"
        for index, step in enumerate(normalized["steps"])
    }
    steps = deepcopy(normalized["steps"])
    for step in steps:
        old_id = step["id"]
        step["id"] = id_map[old_id]
        step["depends_on"] = [id_map[item] for item in step["depends_on"]]
        for selector_name in ("object", "target"):
            selector = step[selector_name]
            if selector["kind"] == "step_result":
                selector["step_id"] = id_map[selector["step_id"]]
    return validate_instruction_intent({"steps": steps})


def _target_affordances(task_type: str, relation: str) -> list[str]:
    if task_type == "E3" or (task_type == "E1" and relation == "inside"):
        return ["container"]
    return []


def _target_structure(task_type: str, relation: str) -> str:
    if task_type == "E1" and relation == "on":
        return "physical_entity"
    if task_type == "E3" or (task_type == "E1" and relation == "inside"):
        return "rigid_object"
    return "scene_entity"
