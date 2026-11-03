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


def test_task_spec_cache_is_content_addressed(tmp_path: Path) -> None:
    cache = TaskSpecCache(tmp_path)
    first = TaskSpec(
        roles=(RoleSpec("button", kind="button"),),
        goal=(Predicate("pressed", observation="pressed"),),
        metadata={"candidate_id": "first"},
    )
    second = replace(first, metadata={"candidate_id": "another-run"})
    assert cache.put(first) == cache.put(second)
    assert cache.get(first.semantic_hash) == first
    assert cache.hashes() == (first.semantic_hash,)
    assert validate_task_spec(first)["semantic_hash"] == first.semantic_hash


@pytest.mark.parametrize("consumer", ["decode", "validate", "evaluate", "cache"])
def test_nested_semantic_mutation_is_rejected_at_public_boundaries(
    consumer: str, tmp_path: Path
) -> None:
    spec = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("state", expected={"mode": ["closed"]}, observation="state"),),
    )
    spec.goal[0].expected["mode"][0] = "open"
    with pytest.raises(ValueError, match="semantic_hash"):
        if consumer == "decode":
            TaskSpec.from_dict(spec)
        elif consumer == "validate":
            validate_task_spec(spec)
        elif consumer == "evaluate":
            evaluate_task_spec(spec, {"state": {"mode": ["open"]}})
        else:
            TaskSpecCache(tmp_path).put(spec)


def test_instance_decode_preserves_identity_and_owns_nested_values() -> None:
    source = TaskSpec(
        roles=(RoleSpec("object"),),
        goal=(Predicate("state", expected={"mode": ["closed"]}, observation="state"),),
    )
    snapshot = TaskSpec.from_dict(source)
    assert snapshot.to_dict() == source.to_dict()
    assert snapshot.semantic_hash == source.semantic_hash
    snapshot.goal[0].expected["mode"][0] = "open"
    assert source.goal[0].expected == {"mode": ["closed"]}
    assert evaluate_task_spec(source, {"state": {"mode": ["closed"]}}).certificate_ready


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


@pytest.mark.parametrize("observation", [False, None])
def test_required_milestone_never_certifies_failed_or_missing_evidence(
    observation,
) -> None:

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
    assert report.status != EvaluationStatus.SATISFIED
    assert any(item.predicate_id == "milestone:press" for item in report.milestones)


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
