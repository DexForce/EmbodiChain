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

from dataclasses import replace
from pathlib import Path

import pytest

from embodichain.gen_sim.task_spec import (
    EvaluationStatus,
    Predicate,
    RoleSpec,
    TaskSpecCache,
    TaskSpecValidationError,
    TaskSpec,
    TemporalCondition,
    canonicalize,
    evaluate_task_spec,
    validate_task_spec,
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


@pytest.mark.parametrize(
    "changes",
    [
        {"metadata": {"candidate_id": "two", "model": "other"}},
        {"task_id": "other"},
        {"instruction": "different wording"},
    ],
)
def test_task_spec_hash_excludes_generation_identity(changes) -> None:
    base = TaskSpec(
        task_id="demo",
        instruction="place the object",
        roles=(RoleSpec("object:cup"),),
        goal=(Predicate("semantic_goal", observation="goal.done"),),
        metadata={"candidate_id": "one"},
    )
    rerun = replace(base, **changes)
    assert base.semantic_hash == rerun.semantic_hash
    assert TaskSpec.from_dict(base.to_dict()) == base


def test_task_spec_predicate_order_is_canonical_but_explicit_ids_survive() -> None:
    first = TaskSpec(
        task_id="demo",
        instruction="check",
        roles=(RoleSpec("object"),),
        goal=(
            Predicate("state", predicate_id="closed", observation="closed"),
            Predicate("exists", observation="exists"),
        ),
    )
    second = replace(first, goal=tuple(reversed(first.goal)))
    assert first.semantic_hash == second.semantic_hash
    assert any(item.predicate_id == "closed" for item in first.goal)


def test_unknown_values_and_callable_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown TaskSpec predicate"):
        Predicate("user_supplied_python")
    with pytest.raises(TypeError, match="callables"):
        canonicalize({"value": lambda: None})


def test_missing_observation_is_unavailable_and_success_is_certifiable() -> None:
    template = TaskSpec(
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
    unavailable = evaluate_task_spec(template, {})
    assert unavailable.status is EvaluationStatus.UNAVAILABLE
    report = evaluate_task_spec(
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
    assert validate_task_spec(first)["semantic_hash"] == first.semantic_hash
    alternate = replace(first, metadata={"candidate_id": "another-run"})
    assert cache.put(alternate) == cache.path_for(first.semantic_hash)


def test_require_observations_is_an_explicit_validation_mode() -> None:
    template = TaskSpec(
        task_id="demo",
        instruction="do it",
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists"),),
    )
    with pytest.raises(TaskSpecValidationError, match="observation keys"):
        validate_task_spec(template, require_observations=True)


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
    spec = TaskSpec.from_dict(value)
    encoded = spec.to_dict()
    assert encoded["schema_version"] == "task_spec/v0.1"
    assert encoded["roles"][0]["id"] == "role_001"
    assert encoded["init"][0]["subject"] == "role_001"
    assert encoded["milestones"][0]["id"] == "upright_once"
    assert TaskSpec.from_dict(encoded) == spec


def test_task_spec_rejects_cyclic_milestones() -> None:
    predicate = {"predicate": "exists", "expected": True}
    with pytest.raises(ValueError, match="acyclic"):
        TaskSpec(
            roles=(RoleSpec("role_001"),),
            goal=(Predicate("exists"),),
            milestones=(
                {"id": "a", "achieve": (predicate,), "after": ("b",)},
                {"id": "b", "achieve": (predicate,), "after": ("a",)},
            ),
        )


def test_compat_certificate_cannot_hide_failed_predicate() -> None:
    from embodichain.gen_sim.task_spec import PredicateEvaluation
    from embodichain.gen_sim.task_spec.compat import ValidationCertificate

    failed = PredicateEvaluation(
        "goal_001", "exists", EvaluationStatus.FAILED, False, True, "missing"
    )
    with pytest.raises(ValueError, match="aggregate predicate status"):
        ValidationCertificate(
            certificate_id="certificate",
            task_template_hash="0" * 64,
            scene_instance_hash="0" * 64,
            witness_id=None,
            checker="test",
            status="satisfied",
            predicate_results=(failed,),
        )


@pytest.mark.parametrize("observation", [False, None])
def test_required_milestone_never_certifies_failed_or_missing_evidence(
    observation,
) -> None:
    from embodichain.gen_sim.task_spec.compat import build_validation_certificate

    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists", observation="final"),),
        milestones=(
            {
                "id": "press",
                "achieve": ({"predicate": "pressed", "observation": "pressed"},),
            },
        ),
    )
    trajectory = None if observation is None else [{"pressed": observation}]
    report = evaluate_task_spec(spec, {"final": True}, trajectory=trajectory)
    assert not report.certificate_ready
    certificate = build_validation_certificate(
        report,
        certificate_id="test",
        task_template_hash=spec.semantic_hash,
        scene_instance_hash="0" * 64,
    )
    assert certificate.status != "satisfied"
    assert any(
        item.predicate_id == "milestone:press" for item in certificate.predicate_results
    )


@pytest.mark.parametrize(
    ("trajectory", "accepted"),
    [
        ([{"opened": True, "placed": False}, {"opened": False, "placed": True}], True),
        ([{"opened": False, "placed": True}, {"opened": True, "placed": False}], False),
        ([{"opened": True, "placed": True}], False),
    ],
)
def test_milestones_require_achievement_strictly_after_predecessors(
    trajectory, accepted
) -> None:
    # IDs deliberately sort opposite to dependency order.
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists", observation="final"),),
        milestones=(
            {
                "id": "z_open",
                "achieve": ({"predicate": "state", "observation": "opened"},),
            },
            {
                "id": "a_place",
                "after": ("z_open",),
                "achieve": ({"predicate": "semantic_goal", "observation": "placed"},),
            },
        ),
    )
    report = evaluate_task_spec(spec, {"final": True}, trajectory=trajectory)
    assert report.certificate_ready is accepted
    assert len(report.milestones) == 2


def test_milestone_requires_joint_achievement_and_declared_observations() -> None:
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists", observation="final"),),
        milestones=(
            {
                "id": "hold",
                "achieve": (
                    {"predicate": "contact", "observation": "contact"},
                    {"predicate": "state", "observation": "stable"},
                ),
            },
        ),
    )
    report = evaluate_task_spec(
        spec,
        {"final": True},
        trajectory=[
            {"contact": True, "stable": False},
            {"contact": False, "stable": True},
        ],
    )
    assert not report.certificate_ready
    missing = replace(
        spec, milestones=({"id": "hold", "achieve": (Predicate("contact"),)},)
    )
    with pytest.raises(TaskSpecValidationError, match="milestone.hold.contact"):
        validate_task_spec(missing, require_observations=True)


@pytest.mark.parametrize(
    ("mode", "value", "accepted"),
    [
        ("always", True, False),
        ("always", False, False),
        ("eventually", False, False),
        ("eventually", True, True),
    ],
)
def test_partial_temporal_window_requires_conclusive_evidence(
    mode, value, accepted
) -> None:
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("exists", observation="final"),),
        temporal=(
            TemporalCondition(
                Predicate("contact", observation="contact"), within_steps=10, mode=mode
            ),
        ),
    )
    report = evaluate_task_spec(spec, {"final": True}, trajectory=[{"contact": value}])
    assert report.certificate_ready is accepted
    if mode == "always" and not value:
        assert report.temporal[0].status is EvaluationStatus.FAILED
    elif not accepted:
        assert report.temporal[0].status is EvaluationStatus.UNAVAILABLE


def test_spec_identity_ignores_evidence_ids_and_observation_keys() -> None:
    arguments = {"subject": "object"}
    first = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(
            Predicate("exists", arguments, predicate_id="a", observation="first"),
            Predicate(
                "object_upright", arguments, predicate_id="b", observation="second"
            ),
        ),
    )
    second = replace(
        first,
        goal=(
            Predicate("exists", arguments, predicate_id="b", observation="renamed_two"),
            Predicate(
                "object_upright", arguments, predicate_id="a", observation="renamed_one"
            ),
        ),
    )
    assert first.semantic_hash == second.semantic_hash


def test_step_result_keeps_the_selected_subject_in_spec_identity() -> None:
    from copy import deepcopy
    from embodichain.gen_sim.task_engine.agent import TaskAgent
    from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult
    from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate

    template = _candidate_set()["candidates"][0]["draft"]["steps"][0]
    first = deepcopy(template)
    first.update(id="first", object=_selector("scene_ref", reference="red can"))
    second = deepcopy(template)
    second.update(id="second", object=_selector("scene_ref", reference="blue can"))
    specs = []
    for reference in ("first", "second"):
        final = deepcopy(template)
        result_selector = _selector("step_result")
        result_selector["step_id"] = reference
        final.update(
            id="final",
            task_type="E1",
            object=result_selector,
            target=_selector("scene_ref", reference="table"),
            relation="on",
            orientation_goal="preserve",
            depends_on=["first", "second"],
        )
        draft = InstructionDraftResult(
            intent={"steps": [first, second, final]},
            model="test",
            attempts=1,
            latency_seconds=0.0,
            normalizations=(),
        )
        candidate = TaskAgent(
            interpreter=lambda *a, _draft=draft, **k: _draft
        ).generate(
            "task",
            "Stand both cans upright and place the selected one on the table.",
            candidate_count=1,
        )[
            "candidates"
        ][
            0
        ]
        specs.append(task_spec_from_candidate(candidate))
    assert all(
        "subject" in predicate.arguments for spec in specs for predicate in spec.goal
    )
    assert specs[0].semantic_hash != specs[1].semantic_hash
