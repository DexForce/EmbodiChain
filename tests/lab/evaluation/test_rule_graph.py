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

"""Regression coverage for compiled evaluation graphs and episode isolation."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.evaluation import (
    BooleanRule,
    EvaluationSpec,
    EvaluationStatus as Status,
    MilestoneRule,
    PredicateDefinition,
    PredicateMeasurement,
    PredicateSpec,
    RuleSession,
    TemporalRule,
    builtin_predicates,
    evaluate_recorded_rules,
)


def _graph():
    registry = builtin_predicates()

    def measure(values, params):
        value = values["force"]
        return PredicateMeasurement(
            value >= params["minimum"], torch.isfinite(value), {"force": value}
        )

    registry.register(
        "measured_force", PredicateDefinition(1, ("force",), dict, measure)
    )
    return registry.compile(
        EvaluationSpec(
            {
                "touching": PredicateSpec(
                    "measured_force", {"force": "contact"}, {"minimum": 1.0}
                ),
                "closed": PredicateSpec(
                    "joint_range", {"position": "joint"}, {"lower": 0.0, "upper": 0.1}
                ),
                "not_closed": BooleanRule("not", ("closed",)),
                "hold": TemporalRule("hold_for", "touching", 2, steps=2),
                "in_range": TemporalRule("always", "closed", 2),
                "touched": TemporalRule("eventually", "touching", 2),
                "first": MilestoneRule(("touching",)),
                "second": MilestoneRule(("touching", "closed"), after=("first",)),
                "goal": BooleanRule("all", ("hold", "second")),
                "progress": BooleanRule("any", ("first", "closed")),
            },
            ("goal", "progress"),
        ),
        available_fields={"contact", "joint"},
    )


def test_registered_predicate_boolean_and_missing_evidence():
    graph = _graph()
    assert graph.required_fields == ("contact", "joint")
    result = graph.evaluate(
        {"contact": torch.tensor([1.0, float("nan")])}, batch_size=2
    )
    assert result["touching"].status.tolist() == [Status.SATISFIED, Status.UNAVAILABLE]
    assert result["not_closed"].status.tolist() == [Status.UNAVAILABLE] * 2
    assert result["hold"].status.tolist() == [Status.NOT_RUN] * 2
    assert result["progress"].status.tolist() == [Status.UNAVAILABLE] * 2


def test_shared_segment_sessions_preserve_selection_qualification_and_names():
    from embodichain.lab.evaluation.observations import _combine_selections

    registry = builtin_predicates()
    position = "robot/arm/gripper/qpos"
    command = "robot/arm/gripper/command"
    compiled = registry.compile(
        EvaluationSpec(
            {
                "preset": PredicateSpec(
                    "command_preset_match",
                    {"position": position},
                    {"preset": "closed", "positions": [0.0], "tolerance": 0.1},
                ),
                "tracking": PredicateSpec(
                    "command_position_match",
                    {"position": position, "command": command},
                    {"tolerance": 0.1},
                ),
            },
            ("preset", "tracking"),
        )
    )
    preset, tracking = compiled.select("preset"), compiled.select("tracking")
    # Foreign command order invalidates the pair, but cannot invalidate a
    # separately selected preset that never used the command channel.
    assert _combine_selections({"preset": preset, "tracking": tracking}) == (
        (("preset",), preset),
        (("tracking",), tracking),
    )
    independent = registry.compile(
        EvaluationSpec(
            {
                "preset": PredicateSpec(
                    "command_preset_match",
                    {"position": position},
                    {"preset": "open", "positions": [1.0], "tolerance": 0.1},
                ),
            },
            ("preset",),
        )
    )
    assert _combine_selections({"first": preset, "second": independent}) == (
        (("first",), preset),
        (("second",), independent),
    )


def test_temporal_witnesses_and_partial_reset_are_row_local():
    graph = _graph()
    session = RuleSession(graph, env_ids=(4, 1), episode_ids=(9, 8), scope="pick/0")
    good = {"contact": torch.ones(2), "joint": torch.zeros(2)}
    session.advance(0, good, revisions=torch.tensor([20, 30]), episode_ids=(9, 8))
    before = session.snapshot()
    assert before.rules["second"].status.tolist() == [Status.UNKNOWN] * 2
    session.evaluate(good)
    assert session.snapshot().step == 0

    session.advance(1, good, revisions=torch.tensor([21, 31]), episode_ids=(9, 8))
    completed = session.snapshot()
    assert completed.rules["goal"].status.tolist() == [Status.SATISFIED] * 2
    assert completed.rules["hold"].witness_revisions.tolist() == [[20, 21], [30, 31]]
    assert completed.rules["second"].witness_steps.tolist() == [[1, 1], [1, 1]]
    session.reset((4,), (10,))
    assert session.snapshot().rules["hold"].status.tolist() == [
        Status.NOT_RUN,
        Status.SATISFIED,
    ]
    session.advance(2, good, revisions=torch.tensor([1, 32]), episode_ids=(9, 8))
    assert session.snapshot().rules["in_range"].status.tolist() == [
        Status.UNKNOWN,
        Status.SATISFIED,
    ]
    session.advance(4, good, revisions=torch.tensor([2, 33]), episode_ids=(10, 8))
    after = session.snapshot(closed=True)
    assert after.rules["hold"].status.tolist() == [Status.UNAVAILABLE, Status.SATISFIED]
    assert after.rules["in_range"].status.tolist() == [
        Status.UNAVAILABLE,
        Status.SATISFIED,
    ]
    assert after.rules["touched"].witness_revisions.tolist() == [[2, 2], [30, 30]]
    assert completed.rules["hold"].witness_revisions.tolist() == [[20, 21], [30, 31]]


@pytest.mark.parametrize(
    ("mode", "samples", "status", "witness", "violation"),
    [
        ("eventually", (None, 1.0, 0.0), Status.SATISFIED, (2, 2), None),
        ("eventually", (0.0, 0.0, 0.0), Status.FAILED, None, None),
        ("eventually", (0.0, None, 0.0), Status.UNAVAILABLE, None, None),
        ("always", (1.0, 1.0, 1.0), Status.SATISFIED, (1, 3), None),
        ("always", (None, 0.0, 1.0), Status.FAILED, None, (2, 2)),
        ("always", (1.0, 1.0), Status.UNAVAILABLE, None, None),
    ],
)
def test_taskspec_and_tensor_windows_share_conclusions_and_evidence(
    mode, samples, status, witness, violation
):
    from embodichain.gen_sim.task_spec.expressions import (
        Predicate,
        TemporalCondition,
        evaluate_temporal_condition,
    )

    condition = TemporalCondition(
        Predicate("contact", observation="force", operator="gte", expected=1.0),
        start_step=1,
        within_steps=2,
        mode=mode,
    )
    observations = [
        {"force": 0.0},
        *({"force": value} if value is not None else {} for value in samples),
    ]
    scalar = evaluate_temporal_condition(condition, observations)
    graph = builtin_predicates().compile(
        EvaluationSpec(
            {
                "contact": PredicateSpec(
                    "joint_range", {"position": "force"}, {"lower": 1.0}
                ),
                "window": TemporalRule(mode, "contact", start=1, end=3),
            },
            ("window",),
        )
    )
    session = RuleSession(graph, env_ids=(7,), episode_ids=(2,), scope="contact")
    for step, observation in enumerate(observations):
        values = {name: torch.tensor([value]) for name, value in observation.items()}
        session.advance(
            step, values, revisions=torch.tensor([10 + step]), episode_ids=(2,)
        )
    batched = session.snapshot(closed=True).rules["window"]
    values = torch.tensor(
        [[observation.get("force", float("nan")) for observation in observations]]
    )
    recorded = evaluate_recorded_rules(
        graph,
        {
            "raw_observations": {
                "schema_version": 1,
                "env_ids": (7,),
                "episode_ids": (2,),
                "revisions": torch.arange(10, 10 + len(observations))[None, :],
                "lengths": torch.tensor([len(observations) - 1]),
                "fields": {
                    "force": {
                        "values": values,
                        "valid": torch.isfinite(values),
                        "descriptor": {
                            "quantity": "scalar",
                            "shape": (),
                            "dtype": "torch.float32",
                        },
                    }
                },
            }
        },
        env_ids=(7,),
        episode_ids=(2,),
        scope="contact",
        chunk_steps=2,
    ).rules["window"]
    assert batched.status.item() == status
    assert scalar.status.value == status.name.lower()
    assert scalar.witness_steps == witness
    assert scalar.violation_steps == violation
    assert tuple(batched.witness_steps[0].tolist()) == (witness or (-1, -1))
    assert tuple(batched.violation_steps[0].tolist()) == (violation or (-1, -1))
    for key in (
        "status",
        "witness_steps",
        "witness_revisions",
        "violation_steps",
        "violation_revisions",
    ):
        torch.testing.assert_close(getattr(recorded, key), getattr(batched, key))
    assert scalar.to_dict()["observed"] == list(samples)
    assert "witness_steps" not in scalar.to_dict()


def test_always_keeps_first_qualified_violation_and_closes_missing_window():
    graph = _graph()
    session = RuleSession(graph, env_ids=(0, 1), episode_ids=(2, 2), scope="place")
    session.advance(
        0,
        {"joint": torch.tensor([0.3, 0.0])},
        revisions=torch.tensor([3, 3]),
        episode_ids=(2, 2),
    )
    snap = session.snapshot(closed=True).rules["in_range"]
    assert snap.status.tolist() == [Status.FAILED, Status.UNAVAILABLE]
    assert snap.violation_revisions.tolist() == [[3, 3], [-1, -1]]


def test_closed_milestone_distinguishes_failure_and_missing_evidence():
    session = RuleSession(_graph(), env_ids=(0, 1), episode_ids=(2, 2), scope="place")
    session.advance(
        0,
        {"contact": torch.tensor([0.0, float("nan")])},
        revisions=torch.tensor([3, 3]),
        episode_ids=(2, 2),
    )
    assert session.snapshot(closed=True).rules["first"].status.tolist() == [
        Status.FAILED,
        Status.UNAVAILABLE,
    ]


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ({"root": BooleanRule("any", ())}, "references"),
        ({"root": BooleanRule("not", ("missing",))}, "Unknown evaluation reference"),
        (
            {"root": MilestoneRule(("child",)), "child": MilestoneRule(("root",))},
            "cycle",
        ),
        (
            {
                "root": PredicateSpec(
                    "joint_range", {"position": "wrong"}, {"lower": 0, "upper": 1}
                )
            },
            "Unresolved",
        ),
        (
            {"root": PredicateSpec("unknown", {"position": "joint"}, {})},
            "Unknown predicate",
        ),
    ],
)
def test_compiler_rejects_invalid_graphs(rules, message):
    with pytest.raises(ValueError, match=message):
        builtin_predicates().compile(
            EvaluationSpec(rules, ("root",)), available_fields={"joint"}
        )


def test_graph_rejects_mapping_in_reference_lists():
    with pytest.raises(ValueError, match="references"):
        EvaluationSpec.from_dict(
            {
                "rules": {
                    "root": {
                        "kind": "below_height",
                        "bindings": {"position": "pose"},
                        "params": {"height": 1},
                    }
                },
                "roots": {"root": "silently ignored"},
            }
        )


def test_legacy_geometric_measurements_are_named_predicates():
    rules = {
        "lift": PredicateSpec(
            "lift", {"pose": "a", "baseline": "b"}, {"minimum_lift": 0.05}
        ),
        "upright": PredicateSpec(
            "upright", {"pose": "a"}, {"maximum_tilt_degrees": 45}
        ),
        "relative": PredicateSpec(
            "relative_position",
            {"pose": "a", "reference": "b"},
            {"offset": (0, 0, 0.1), "tolerances": (0.01, 0.01, 0.01)},
        ),
        "near_relative": PredicateSpec(
            "near_relative_target",
            {"position": "position", "reference": "reference"},
            {"displacement": (0, 0, 0.1), "tolerance": 0.01},
        ),
    }
    graph = builtin_predicates().compile(EvaluationSpec(rules, tuple(rules)))
    reference = torch.tensor([[0, 0, 0, 0, 0, 0, 1.0]]).expand(2, -1)
    pose = reference.clone()
    pose[0, 2] = 0.1
    pose[1, 3] = 1
    pose[1, 6] = 0
    result = graph.evaluate(
        {
            "a": pose,
            "b": reference,
            "position": pose[:, :3],
            "reference": reference[:, :3],
        },
        batch_size=2,
        device="cpu",
    )
    for name in rules:
        assert result[name].status.tolist() == [Status.SATISFIED, Status.FAILED]


def test_file_replay_uses_original_segment_and_qualifies_each_row(tmp_path):
    graph = _graph().select("goal")
    fields = {
        name: {
            "values": values,
            "valid": torch.ones_like(values, dtype=torch.bool),
            "descriptor": {"quantity": "scalar", "shape": (), "dtype": "torch.float32"},
        }
        for name, values in {
            "contact": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "joint": torch.zeros(2, 4),
        }.items()
    }
    file = tmp_path / "recording.pt"
    torch.save(
        {
            "raw_observations": {
                "schema_version": 1,
                "env_ids": (4, 1),
                "episode_ids": (9, 8),
                "revisions": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
                "lengths": torch.tensor([3, 3]),
                "fields": fields,
            },
            "meta": {
                "demo_episodes": [
                    {
                        "env_id": row,
                        "episode_context": {"episode": episode},
                        "segments": [
                            {"segment_id": 7, "start_step": 1, "end_step": 3},
                        ],
                    }
                    for row, episode in [(4, 9), (1, 8)]
                ]
            },
        },
        file,
    )
    for chunk in (1, 2, 64):
        result = evaluate_recorded_rules(
            graph,
            file,
            env_ids=(1, 4),
            episode_ids=(8, 10),
            scope="pick/7",
            segment_id=7,
            chunk_steps=chunk,
        )
        assert result.step == 3
        assert result.rules["goal"].status.tolist() == [
            Status.SATISFIED,
            Status.UNAVAILABLE,
        ]
        assert result.rules["hold"].witness_steps.tolist() == [[2, 3], [-1, -1]]
        assert result.rules["hold"].witness_revisions.tolist() == [[22, 23], [-1, -1]]


def test_recorded_joint_uses_native_order_and_rejects_foreign_field():
    field = "articulations/drawer/qpos/rail"
    graph = builtin_predicates().compile(
        EvaluationSpec(
            {
                "closed": PredicateSpec(
                    "joint_range", {"position": field}, {"upper": 0.01}
                ),
            },
            ("closed",),
        )
    )
    descriptor = {
        "quantity": "qpos",
        "entity_uid": "drawer",
        "coordinate_frame": "joint",
        "representation": "joint_order",
        "units": ("joint_units",),
        "shape": (2,),
        "dtype": "torch.float32",
        "joint_names": ("latch", "rail"),
    }
    payload = {
        "raw_observations": {
            "schema_version": 1,
            "env_ids": (1, 4),
            "episode_ids": (3, 3),
            "lengths": torch.tensor([0, 0]),
            "revisions": torch.tensor([[11], [15]]),
            "fields": {
                field: {
                    "values": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                    "valid": torch.ones(2, 1, dtype=torch.bool),
                    "descriptor": descriptor,
                }
            },
        }
    }

    def replay():
        return (
            evaluate_recorded_rules(
                graph, payload, env_ids=(1, 4), episode_ids=(3, 3), scope="final"
            )
            .rules["closed"]
            .status.tolist()
        )

    assert replay() == [Status.SATISFIED, Status.FAILED]
    descriptor["units"] = ("pixels",)
    assert replay() == [Status.UNAVAILABLE, Status.UNAVAILABLE]
    descriptor["units"] = ("joint_units",)
    descriptor["entity_uid"] = "foreign"
    assert replay() == [Status.UNAVAILABLE, Status.UNAVAILABLE]


def test_replay_preserves_row_ends_and_rejects_missing_endpoint():
    graph = builtin_predicates().compile(
        EvaluationSpec(
            {
                "closed": PredicateSpec(
                    "joint_range", {"position": "joint"}, {"upper": 1}
                )
            },
            ("closed",),
        )
    )
    payload = {
        "raw_observations": {
            "schema_version": 1,
            "env_ids": (5, 9),
            "episode_ids": (7, 2),
            "revisions": torch.tensor([[10, 11, -1], [20, 21, 22]]),
            "lengths": torch.tensor([1, 2]),
            "fields": {
                "joint": {
                    "values": torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.4, 0.3]]),
                    "valid": torch.ones(2, 3, dtype=torch.bool),
                    "descriptor": {
                        "quantity": "scalar",
                        "shape": (),
                        "dtype": "torch.float32",
                    },
                },
            },
        }
    }
    for chunk in (1, 2, 64):
        result = evaluate_recorded_rules(
            graph,
            payload,
            env_ids=(5, 12, 9),
            episode_ids=(7, 0, 2),
            scope="rows",
            chunk_steps=chunk,
        )
        assert result.revisions.tolist() == [11, -1, 22]
        assert result.steps.tolist() == [1, -1, 2]
        assert result.rules["closed"].status.tolist() == [
            Status.SATISFIED,
            Status.UNAVAILABLE,
            Status.SATISFIED,
        ]
        assert result.rules["closed"].metrics["position"][
            [0, 2]
        ].tolist() == pytest.approx([0.2, 0.3])
        assert torch.isnan(result.rules["closed"].metrics["position"][1])
    beyond = evaluate_recorded_rules(
        graph,
        payload,
        env_ids=(5, 9),
        episode_ids=(7, 2),
        scope="rows",
        end_step=10,
    )
    assert beyond.step == 10
    assert beyond.revisions.tolist() == [-1, -1]
    assert beyond.rules["closed"].status.tolist() == [Status.UNAVAILABLE] * 2
