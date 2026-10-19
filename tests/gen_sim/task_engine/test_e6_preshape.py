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

"""E6 aperture-only recovery keeps hand commands and geometric screening aligned."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.e6_clearance import HandClearance
from embodichain.gen_sim.task_engine._task_program.e6_preshape import (
    defer_initial_capture,
    filter_candidates,
    plan_with_preshape,
)
from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
    E6ApproachGraspPoseGenerator,
)
from embodichain.lab.sim.atomic_actions import SlideOptions


@dataclass
class Diagnostics:
    metadata: dict = field(default_factory=dict)


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


@dataclass
class Request:
    skill_id: str = "slide"
    invocation_id: str = "bound"
    skill_options: object = field(
        default_factory=lambda: SlideOptions(preshape_fraction=0.85)
    )
    binding: object = field(
        default_factory=lambda: NS(
            endpoint=lambda *_: NS(require_target=lambda _: NS(target_id="hand"))
        )
    )
    motion_policy: object = field(default_factory=lambda: NS(strategy="ik_interp"))


@pytest.fixture
def case():
    provider = object.__new__(E6ApproachGraspPoseGenerator)
    provider._clearance = HandClearance(torch.zeros(1, 3), 0.01678, 0.028, 0.0)
    measured = []

    def measure(fraction):
        measured.append(fraction)
        return replace(
            provider._clearance, aperture=0.01678 + (0.85 - fraction) * 0.1391
        )

    provider._clearance_factory = measure
    engine = NS(
        _articulation_calls={"bound": "slide"},
        grasp_pose_generators={"hand": provider},
        robot=NS(device=torch.device("cpu")),
    )
    return (
        engine,
        provider,
        measured,
        NS(batch_size=1, robot=NS(qpos=torch.zeros(1, 1))),
    )


def screen(provider, width=0.0193, z=1.0):
    poses = torch.eye(4)[None]
    poses[:, 2, 3] = z
    return filter_candidates(
        provider, provider._clearance, poses, torch.tensor([width]), torch.zeros(1)
    )[2]


def test_nominal_success_is_identical_and_never_measures_wider_hand(case):
    engine, provider, measured, context = case
    original = Plan(torch.tensor([True]))
    calls = []

    def build(request):
        calls.append(request)
        assert torch.isfinite(screen(provider, width=0.01)).all()
        return original

    result = plan_with_preshape(engine, Request(), context, build)
    assert result is original and len(calls) == 1 and measured == []


def test_aperture_failure_selects_smallest_measured_opening_and_matches_request(case):
    engine, provider, measured, context = case
    fractions = []

    def build(request):
        fractions.append(request.skill_options.preshape_fraction)
        valid = torch.isfinite(screen(provider)).all().reshape(1)
        assert bool(valid) == (request.skill_options.preshape_fraction == 0.83)
        return Plan(valid)

    request = Request()
    result = plan_with_preshape(engine, request, context, build)
    assert result.plan_success.tolist() == [True]
    assert fractions == [0.85, 0.83] and measured == [0.84, 0.83]
    assert request.skill_options.preshape_fraction == 0.85
    assert provider._clearance.aperture == 0.01678
    assert torch.isinf(screen(provider)).all()
    assert (
        result.diagnostics.metadata["gen_sim_e6_preshape"]["selected_fraction"] == 0.83
    )


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_table_or_later_motion_failure_does_not_trigger_opening_recovery(case, z):
    engine, provider, measured, context = case
    original = Plan(torch.tensor([False]))

    def build(_):
        screen(provider, width=0.01, z=z)
        return original

    assert plan_with_preshape(engine, Request(), context, build) is original
    assert measured == []


def test_oversized_handle_remains_rejected_without_a_wider_motion_attempt(case):
    engine, provider, measured, context = case
    calls = []

    def build(request):
        calls.append(request)
        return Plan(torch.isfinite(screen(provider, width=0.10)).all().reshape(1))

    result = plan_with_preshape(engine, Request(), context, build)
    assert result.plan_success.tolist() == [False] and len(calls) == 1
    assert measured == [0.84, 0.83, 0.82, 0.81, 0.8]


@pytest.mark.parametrize(
    "change", ["unbound", "batch", "custom", "other_skill", "planner"]
)
def test_other_scopes_keep_existing_behavior(case, change):
    engine, provider, measured, context = case
    request = Request()
    if change == "unbound":
        request.invocation_id = "other"
    elif change == "batch":
        context.batch_size = 2
    elif change == "custom":
        request.skill_options = SlideOptions(preshape_fraction=0.80)
    elif change == "other_skill":
        request.skill_id = "pick_up"
    else:
        request.motion_policy = NS(strategy="motion_gen")
    calls = []

    def build(selected):
        calls.append(selected)
        return Plan(torch.isfinite(screen(provider)).all().reshape(1))

    plan_with_preshape(engine, request, context, build)
    assert calls == [request] and measured == []


@pytest.mark.parametrize("raises", [False, True])
def test_failed_recovery_restores_rng_and_call_local_clearance(case, raises):
    engine, provider, _, context = case
    before = torch.random.get_rng_state()
    torch.rand(1)
    nominal_after = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    values = []

    def build(request):
        values.append(torch.rand(1))
        screen(provider)
        if request.skill_options.preshape_fraction != 0.85 and raises:
            raise RuntimeError("injected")
        return Plan(torch.tensor([False]))

    try:
        if raises:
            with pytest.raises(RuntimeError, match="injected"):
                plan_with_preshape(engine, Request(), context, build)
        else:
            plan_with_preshape(engine, Request(), context, build)
        assert len(values) == 2 and torch.equal(values[0], values[1])
        assert torch.equal(torch.random.get_rng_state(), nominal_after)
        assert torch.isinf(screen(provider)).all()
    finally:
        torch.random.set_rng_state(before)


def test_initial_capture_publishes_only_selected_plan(case, monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import planning_probe

    engine, provider, _, context = case
    published = []

    def accept(request, plan, robot):
        published.append(
            (request.skill_options.preshape_fraction, plan.plan_success.tolist())
        )
        return plan

    monkeypatch.setattr(
        planning_probe, "initial_plan_capture", lambda _: NS(accept=accept)
    )

    def build(request):
        assert defer_initial_capture(request.invocation_id)
        assert not defer_initial_capture("another_call")
        return Plan(torch.isfinite(screen(provider)).all().reshape(1))

    plan_with_preshape(engine, Request(), context, build)
    assert published == [(0.83, [True])]
    assert not defer_initial_capture("bound")


def test_wider_hand_does_not_bypass_table_rejection(case):
    engine, provider, measured, context = case
    original = Plan(torch.tensor([False]))

    def build(request):
        assert torch.isinf(screen(provider, z=0.0)).all()
        return original

    result = plan_with_preshape(engine, Request(), context, build)
    assert result.plan_success.tolist() == [False]
    assert measured == [0.84, 0.83]
    assert result.diagnostics.metadata["gen_sim_e6_preshape"]["accepted_plan"] is False


@pytest.mark.parametrize("explicit", [False, True])
def test_engine_uses_one_context_for_implicit_and_explicit_e6_calls(case, explicit):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    source, provider, _, context = case
    engine = object.__new__(GenSimActionEngine)
    engine._actions = {}
    engine._articulation_calls = source._articulation_calls
    engine._planning_services = NS(
        robot=source.robot, grasp_pose_generators=source.grasp_pose_generators
    )
    captures, observed = [], []

    def initial():
        captures.append(True)
        return context

    def build(request, current, **kwargs):
        assert current is context
        observed.append(request.skill_options.preshape_fraction)
        return Plan(torch.isfinite(screen(provider)).all().reshape(1))

    engine.initial_context = initial
    engine._plan_request_impl = build
    result = engine._plan_request(Request(), context if explicit else None)
    assert result.plan_success.tolist() == [True]
    assert observed == [0.85, 0.83]
    assert len(captures) == (0 if explicit else 1)
