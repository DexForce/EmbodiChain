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

"""Invocation policy ownership, independent of recipe suffixes and call types."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
    GenSimActionEngine,
    bind_cartesian_calls,
    bind_motion_samples,
    cartesian_call_declarations,
    motion_sample_declarations,
    bind_pour_receivers,
    pour_receiver_declarations,
)
from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy


def node(family: str, name: str, *, on: bool = False) -> dict:
    call = {"kind": "pick", "object": "payload"}
    if family == "E5":
        call = {
            "kind": "registered",
            "call_id": "simulation.coordinated_transport",
            "arguments": {"object": "payload", "target": "destination"},
        }
        if on:
            call["arguments"].update(reference="support", relation="on")
    return {"id": name, "task_type": family, "task_instance_id": name, "call": call}


@pytest.mark.parametrize("first", ["E1", "E2", "E3", "E4", "E5", "E6"])
@pytest.mark.parametrize("later", ["E1", "E2", "E3", "E4", "E5", "E6"])
def test_appending_any_recipe_preserves_prefix_motion_budget(first, later):
    initial = node(first, "first", on=True)
    alone = motion_sample_declarations({"nodes": [initial]})
    combined = motion_sample_declarations(
        {"nodes": [initial, node(later, "later", on=True)]}
    )
    assert alone["first"] == combined["first"]


def test_only_current_recipe_drawer_and_on_requirements_raise_the_budget():
    graph = {
        "nodes": [
            node("E1", "ordinary"),
            node("E1", "drawer"),
            node("E5", "directional"),
            node("E5", "on", on=True),
        ]
    }
    assert motion_sample_declarations(graph, drawer_groups=frozenset({"drawer"})) == {
        "ordinary": 140,
        "drawer": 260,
        "directional": 140,
        "on": 260,
    }


@pytest.mark.parametrize(
    "marker", ["hand_over", "gen_sim.clear_released", "gen_sim.articulation_withdraw"]
)
def test_later_recipe_does_not_change_ordinary_cartesian_mode(marker):
    ordinary = node("E1", "ordinary")
    later = node("E4", "later")
    later["call"] = (
        {"kind": "hand_over"}
        if marker == "hand_over"
        else {"kind": "registered", "call_id": marker, "arguments": {}}
    )
    assert cartesian_call_declarations({"nodes": [ordinary]}) == []
    assert cartesian_call_declarations({"nodes": [ordinary, later]}) == ["later"]


def test_cartesian_scope_binding_rejects_stale_or_duplicate_names():
    segments = [
        SimpleNamespace(
            name="one",
            segment_id="segment-0",
            calls=(SimpleNamespace(segment_call_index=0),),
        ),
        SimpleNamespace(
            name="two",
            segment_id="segment-1",
            calls=(SimpleNamespace(segment_call_index=0),),
        ),
    ]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    assert bind_cartesian_calls(["two"], program) == ("test/segment-1:0",)
    for value in (["absent"], ["two", "two"], [True], {"two": True}):
        with pytest.raises(ValueError):
            bind_cartesian_calls(value, program)


def test_cartesian_planning_scope_restores_on_nested_failure(monkeypatch):
    from dataclasses import dataclass, field
    from embodichain.gen_sim.task_engine._task_program import motion

    @dataclass
    class Diagnostics:
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Plan:
        diagnostics: Diagnostics = field(default_factory=Diagnostics)

    observed = []

    def plan(self, request, context=None, *, plan_transform=None):
        observed.append(motion._CARTESIAN_APPROACH.get())
        if request.invocation_id == "fails":
            raise RuntimeError("injected")
        return Plan()

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    engine = object.__new__(GenSimActionEngine)
    engine._cartesian_calls = frozenset({"special", "fails"})
    engine._articulation_calls = {}

    def request(name):
        return SimpleNamespace(
            invocation_id=name, motion_policy=MotionPolicy(sample_count=140)
        )

    with motion.cartesian_approach_scope(True):
        result = engine._plan_request(request("ordinary"))
        assert motion._CARTESIAN_APPROACH.get() is True
        with pytest.raises(RuntimeError, match="injected"):
            engine._plan_request(request("fails"))
        assert motion._CARTESIAN_APPROACH.get() is True
    engine._plan_request(request("special"))
    assert observed == [False, True, True]
    assert motion._CARTESIAN_APPROACH.get() is False
    assert (
        result.diagnostics.metadata["gen_sim_invocation_policy"]["cartesian_approach"]
        is False
    )


def test_compiled_call_binding_is_exact_and_does_not_use_object_identity():
    def segment(name, index):
        return SimpleNamespace(
            name=name,
            segment_id=f"segment-{index}",
            calls=(SimpleNamespace(segment_call_index=0),),
        )

    segments = [segment("first", 0), segment("second", 1)]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    assert bind_motion_samples({"first": 140, "second": 260}, program) == (
        ("test/segment-0:0", 140),
        ("test/segment-1:0", 260),
    )
    for bad in (
        {"first": 140},
        {"first": 140, "second": 260, "absent": 140},
        {"first": True, "second": 260},
    ):
        with pytest.raises(ValueError):
            bind_motion_samples(bad, program)


def test_resolution_owns_policies_and_cannot_leak_between_same_skill_calls(monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Invocation:
        invocation_id: str
        motion_policy: MotionPolicy

    monkeypatch.setattr(AtomicActionEngine, "_resolve", lambda self, value: value)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"first": 140, "second": 260}
    base = MotionPolicy(sample_count=140)
    first = Invocation("first", base)
    second = Invocation("second", base)
    assert engine._resolve(second).motion_policy.sample_count == 260
    assert engine._resolve(first).motion_policy.sample_count == 140
    assert base.sample_count == 140
    assert second.motion_policy is base
    larger = Invocation("second", MotionPolicy(sample_count=400))
    assert engine._resolve(larger).motion_policy.sample_count == 400
    with pytest.raises(ValueError, match="invocation"):
        engine._resolve(Invocation("unknown", base))


def test_resolution_failure_does_not_mutate_another_calls_policy(monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Invocation:
        invocation_id: str
        motion_policy: MotionPolicy

    def resolve(self, value):
        if value.invocation_id == "failed":
            raise RuntimeError("injected resolution failure")
        return value

    monkeypatch.setattr(AtomicActionEngine, "_resolve", resolve)
    engine = object.__new__(GenSimActionEngine)
    engine._motion_samples = {"failed": 260, "next": 140}
    policy = MotionPolicy(sample_count=140)
    with pytest.raises(RuntimeError, match="injected"):
        engine._resolve(Invocation("failed", policy))
    assert engine._resolve(Invocation("next", policy)).motion_policy.sample_count == 140
    assert policy.sample_count == 140


def test_two_pours_of_the_same_object_keep_distinct_receiver_bindings():
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPour

    def recipe(name, receiver):
        return [
            {
                "id": name + "_stage",
                "task_instance_id": name,
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.move_held_object",
                    "arguments": {"object": "cup", "reference": receiver},
                },
            },
            {
                "id": name + "_pour",
                "task_instance_id": name,
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.pour",
                    "arguments": {"object": "cup"},
                },
            },
        ]

    first = recipe("first", "receiver_a")
    graph = {"nodes": first + recipe("second", "receiver_b")}
    assert pour_receiver_declarations({"nodes": first}) == {"first_pour": "receiver_a"}
    declarations = pour_receiver_declarations(graph)
    assert declarations == {"first_pour": "receiver_a", "second_pour": "receiver_b"}
    segments = [
        SimpleNamespace(
            name=n["id"],
            segment_id=f"s{i}",
            calls=(
                SimpleNamespace(
                    segment_call_index=0,
                    call=SimpleNamespace(call_id=n["call"]["call_id"]),
                ),
            ),
        )
        for i, n in enumerate(graph["nodes"])
    ]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    bindings = dict(bind_pour_receivers(declarations, program))
    action = GenSimPour(bindings)
    assert action._receiver_id("test/s1:0") == "receiver_a"
    assert action._receiver_id("test/s3:0") == "receiver_b"
    bindings["test/s1:0"] = "mutated"
    assert action._receiver_id("test/s1:0") == "receiver_a"
    with pytest.raises(ValueError, match="Unbound"):
        action._receiver_id("unknown")
    with pytest.raises(ValueError, match="exact receiver"):
        bind_pour_receivers({"first_pour": "receiver_a"}, program)
