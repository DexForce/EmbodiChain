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

from pathlib import Path

import pytest

from embodichain.gen_sim.task_spec import (
    EvaluationStatus,
    Predicate,
    PredicateEvaluation,
    RoleSpec,
    TaskSpecCache,
    TaskSpecValidationError,
    TaskTemplate,
    TemporalCondition,
    ValidationCertificate,
    canonicalize,
    evaluate_task_template,
    validate_task_template,
)


def _selector(
    kind: str = "none", *, reference: str = "", quantifier: str = "one"
) -> dict[str, object]:
    return {
        "kind": kind,
        "step_id": "",
        "reference": reference,
        "quantifier": quantifier,
        "count": 0,
    }


def _candidate_set() -> dict[str, object]:
    from embodichain.gen_sim.task_engine.agent import TaskAgent
    from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult

    step = {
        "id": "orient",
        "task_type": "E2",
        "object": _selector("scene_ref", reference="purple can"),
        "target": _selector(),
        "relation": "none",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }

    def interpreter(*_args: object, **_kwargs: object) -> InstructionDraftResult:
        return InstructionDraftResult(
            intent={"steps": [step]},
            model="test",
            attempts=1,
            latency_seconds=0.0,
            normalizations=(),
        )

    return TaskAgent(interpreter=interpreter).generate(
        "upright_can", "make the can upright", candidate_count=1
    )


def test_task_template_hash_excludes_generation_metadata() -> None:
    base = TaskTemplate(
        task_id="demo",
        instruction="place the object",
        roles=(RoleSpec("object:cup"),),
        goal=(Predicate("semantic_goal", observation="goal.done"),),
        metadata={"candidate_id": "one"},
    )
    rerun = TaskTemplate(
        task_id="demo",
        instruction="place the object",
        roles=(RoleSpec("object:cup"),),
        goal=(Predicate("semantic_goal", observation="goal.done"),),
        metadata={"candidate_id": "two", "model": "other"},
    )
    assert base.semantic_hash == rerun.semantic_hash
    assert TaskTemplate.from_dict(base.to_dict()) == base


def test_task_template_predicate_order_is_canonical_but_explicit_ids_survive() -> None:
    first = TaskTemplate(
        task_id="demo",
        instruction="check",
        roles=(RoleSpec("object"),),
        goal=(
            Predicate("state", predicate_id="closed", observation="closed"),
            Predicate("exists", observation="exists"),
        ),
    )
    second = TaskTemplate(
        task_id="demo",
        instruction="check",
        roles=(RoleSpec("object"),),
        goal=(
            Predicate("exists", observation="exists"),
            Predicate("state", predicate_id="closed", observation="closed"),
        ),
    )
    assert first.semantic_hash == second.semantic_hash
    assert any(item.predicate_id == "closed" for item in first.goal)


def test_unknown_values_and_callable_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown TaskSpec predicate"):
        Predicate("user_supplied_python")
    with pytest.raises(TypeError, match="callables"):
        canonicalize({"value": lambda: None})


def test_missing_observation_is_unavailable_and_success_is_certifiable() -> None:
    template = TaskTemplate(
        task_id="demo",
        instruction="press the button",
        roles=(RoleSpec("button", kind="button"),),
        goal=(Predicate("pressed", observation="goal.pressed"),),
        temporal=(
            TemporalCondition(
                Predicate("contact", observation="contact"), within_steps=2
            ),
        ),
    )
    unavailable = evaluate_task_template(template, {})
    assert unavailable.status is EvaluationStatus.UNAVAILABLE
    report = evaluate_task_template(
        template,
        {
            "goal": {"goal.pressed": True},
            "trajectory": [{"contact": False}, {"contact": True}],
        },
    )
    assert report.status is EvaluationStatus.SATISFIED
    assert report.certificate_ready


def test_task_engine_generation_is_content_addressed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from embodichain.gen_sim.task_engine.task_spec import generate_task_spec

    cache = TaskSpecCache(tmp_path)
    first = generate_task_spec(_candidate_set(), cache=cache)
    second = generate_task_spec(_candidate_set(), cache=cache)
    assert first.semantic_hash == second.semantic_hash
    assert [milestone.id for milestone in first.milestones] == ["step_01"]
    assert [predicate.name for predicate in first.goal] == ["object_upright"]
    assert cache.get(first.semantic_hash) == first
    assert cache.hashes() == (first.semantic_hash,)
    assert validate_task_template(first)["semantic_hash"] == first.semantic_hash
    alternate = TaskTemplate(
        task_id=first.task_id,
        instruction=first.instruction,
        roles=first.roles,
        init=first.init,
        goal=first.goal,
        invariants=first.invariants,
        temporal=first.temporal,
        milestones=first.milestones,
        requirements=first.requirements,
        metadata={"candidate_id": "another-run"},
    )
    assert cache.put(alternate) == cache.path_for(first.semantic_hash)


def test_require_observations_is_an_explicit_validation_mode() -> None:
    template = TaskTemplate(
        task_id="demo",
        instruction="do it",
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists"),),
    )
    with pytest.raises(TaskSpecValidationError, match="observation keys"):
        validate_task_template(template, require_observations=True)


def test_compact_task_spec_schema_round_trips_without_runtime_artifacts() -> None:
    value = {
        "schema_version": "task_spec/v0.1",
        "task_id": "upright_can",
        "instruction": "make the can upright",
        "roles": [
            {
                "id": "role_001",
                "kind": "object",
                "cardinality": "one",
                "affordances": ["graspable", "orientable"],
                "properties": {},
            }
        ],
        "init": [
            {
                "predicate": "state",
                "subject": "role_001",
                "key": "orientation",
                "expected": "fallen",
            }
        ],
        "goal": [
            {
                "predicate": "object_upright",
                "subject": "role_001",
                "expected": True,
            }
        ],
        "invariants": [],
        "temporal": [],
        "milestones": [
            {
                "id": "upright_once",
                "achieve": [
                    {
                        "predicate": "object_upright",
                        "subject": "role_001",
                        "expected": True,
                    }
                ],
            }
        ],
        "requirements": [
            {
                "consumer": "asset",
                "key": "affordances",
                "value": ["graspable", "orientable"],
            }
        ],
    }
    spec = TaskTemplate.from_dict(value)
    encoded = spec.to_dict()
    assert encoded["schema_version"] == "task_spec/v0.1"
    assert encoded["roles"][0]["id"] == "role_001"
    assert encoded["init"][0]["subject"] == "role_001"
    assert encoded["milestones"][0]["id"] == "upright_once"
    assert TaskTemplate.from_dict(encoded) == spec


def test_task_spec_hash_ignores_logical_id_and_instruction() -> None:
    common = {
        "roles": (RoleSpec("role_001"),),
        "goal": (Predicate("exists", arguments={"subject": "role_001"}),),
    }
    first = TaskTemplate(task_id="a", instruction="first wording", **common)
    second = TaskTemplate(task_id="b", instruction="different wording", **common)
    assert first.semantic_hash == second.semantic_hash


def test_task_spec_rejects_cyclic_milestones() -> None:
    predicate = {"predicate": "exists", "expected": True}
    with pytest.raises(ValueError, match="acyclic"):
        TaskTemplate(
            roles=(RoleSpec("role_001"),),
            goal=(Predicate("exists"),),
            milestones=(
                {"id": "a", "achieve": (predicate,), "after": ("b",)},
                {"id": "b", "achieve": (predicate,), "after": ("a",)},
            ),
        )


def test_certificate_status_cannot_hide_failed_predicate() -> None:
    digest = "0" * 64
    with pytest.raises(ValueError, match="aggregate predicate status"):
        ValidationCertificate(
            certificate_id="certificate",
            task_template_hash=digest,
            scene_instance_hash=digest,
            witness_id=None,
            checker="test",
            status="satisfied",
            predicate_results=(
                PredicateEvaluation(
                    predicate_id="goal_001",
                    predicate="exists",
                    status=EvaluationStatus.FAILED,
                    observed=False,
                    expected=True,
                    reason="missing",
                ),
            ),
        )
