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

"""E6 continuation qualification preserves the separate execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS
import random

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    articulation_recovery as recovery,
)
from embodichain.gen_sim.task_engine._task_program import (
    articulation_slide,
    invocation_policy,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy


@dataclass
class Diagnostics:
    metadata: dict = field(default_factory=dict)
    failure: object = None


@dataclass
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    segments: tuple = (NS(name="pull"),)


def segment(index, kind, *, object_id="cabinet", resource="right"):
    return NS(
        name=f"s{index}",
        segment_id=f"segment-{index}",
        calls=(
            NS(
                segment_call_index=0,
                call=NS(
                    call_id=f"gen_sim.articulation_{kind}",
                    arguments={"object": object_id, "state": "open", "part": "part_a"},
                    resources={"primary": resource},
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "mismatch", [None, "object", "part", "resource", "intervening", "state"]
)
def test_only_adjacent_exact_same_drawer_and_resource_bind(mismatch):
    first, second = segment(0, "slide"), segment(1, "withdraw")
    if mismatch == "object":
        second.calls[0].call.arguments["object"] = "another_cabinet"
    elif mismatch == "part":
        second.calls[0].call.arguments["part"] = "part_b"
    elif mismatch == "resource":
        second.calls[0].call.resources["primary"] = "left"
    elif mismatch == "state":
        second.calls[0].call.arguments["state"] = "closed"
    segments = [first, second]
    if mismatch == "intervening":
        segments.insert(1, segment(2, "park"))
    program = NS(program_id="p", iter_segments=lambda: iter(segments))
    assert invocation_policy.bind_articulation_followups(program) == (
        (("p/segment-0:0", "p/segment-1:0"),) if mismatch is None else ()
    )


def test_followup_requires_declared_withdrawal_budget():
    with pytest.raises(ValueError, match="declared withdrawal budget"):
        invocation_policy.GenSimActionEngine(
            object(),
            articulation_calls=(("slide", "slide"), ("withdraw", "withdraw")),
            articulation_followups=(("slide", "withdraw"),),
            motion_samples=(("slide", 260),),
        )


def test_engine_passes_followup_budget_not_slide_budget(monkeypatch):
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    generator = object()
    engine = object.__new__(invocation_policy.GenSimActionEngine)
    engine._planning_services = NS(motion_generator=generator)
    engine._articulation_calls = {"slide": "slide", "withdraw": "withdraw"}
    engine._articulation_followups = {"slide": "withdraw"}
    engine._articulation_sessions = {}
    engine._motion_samples = {"slide": 260, "withdraw": 320}
    engine._cartesian_calls = {"slide", "withdraw"}
    seen = []

    def original(self, request, context=None, **kwargs):
        seen.append(recovery.motion_scope(generator).withdrawal_samples)
        return Plan(torch.tensor([True]), None)

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", original)
    request = NS(
        invocation_id="slide",
        skill_id="move_end_effector",
        goal=None,
        motion_policy=NS(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
    )
    result = engine._plan_request(request, NS(batch_size=1))
    assert seen == [320]
    assert result.plan_success.tolist() == [True]


@pytest.mark.parametrize("changed", [False, True])
def test_actual_gate_requires_terminal_reselection_not_just_pairing(
    monkeypatch, changed
):
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    generator = object()
    engine = object.__new__(invocation_policy.GenSimActionEngine)
    engine._planning_services = NS(motion_generator=generator)
    engine._articulation_calls = {"slide": "slide", "withdraw": "withdraw"}
    engine._articulation_followups = {"slide": "withdraw"}
    attempt = NS(
        request=NS(invocation_id="slide"),
        planned_mask=torch.tensor([True]),
        plan=NS(
            diagnostics=Diagnostics(
                metadata={
                    "gen_sim_articulation_recovery": {
                        "withdrawal_selection_changed": changed
                    }
                }
            )
        ),
    )
    engine._articulation_sessions = {
        "withdraw": ("slide", NS(plan_attempts=(attempt,)))
    }
    engine._motion_samples = {"slide": 260, "withdraw": 320}
    engine._cartesian_calls = {"slide", "withdraw"}
    seen, screened = [], []

    def original(self, request, context=None, **kwargs):
        seen.append(recovery.motion_scope(generator).withdrawal_samples)
        return Plan(torch.tensor([True]), None)

    def screen(scope, plan):
        screened.append(True)
        return plan

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", original)
    monkeypatch.setattr(recovery, "_screen_withdrawal", screen)
    request = NS(
        invocation_id="withdraw",
        skill_id="move_end_effector",
        goal=None,
        motion_policy=NS(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=320
        ),
    )
    engine._plan_request(request, NS(batch_size=1))
    assert seen == [320 if changed else None]
    assert screened == ([True] if changed else [])


def test_installed_session_evidence_survives_speculative_failure_and_later_nominal_replan(
    monkeypatch,
):
    from embodichain.lab.sim.atomic_actions import AtomicActionEngine

    def attempt(changed, identity="slide", eligible=True):
        return NS(
            request=NS(invocation_id=identity),
            planned_mask=torch.tensor([eligible]),
            plan=NS(
                diagnostics=Diagnostics(
                    metadata={
                        "gen_sim_articulation_recovery": {
                            "withdrawal_selection_changed": changed
                        }
                    }
                )
            ),
        )

    engine = object.__new__(invocation_policy.GenSimActionEngine)
    engine._planning_services = NS(motion_generator=object())
    engine._twist_session_controller = None
    engine._twist_session_controllers = {}
    engine._articulation_calls = {"slide": "slide", "withdraw": "withdraw"}
    engine._articulation_followups = {"slide": "withdraw"}
    engine._articulation_sessions = {}
    engine._motion_samples = {"slide": 260, "withdraw": 260}
    engine._cartesian_calls = {"slide", "withdraw"}
    session = NS(plan_attempts=(attempt(True),))
    monkeypatch.setattr(AtomicActionEngine, "start", lambda *a, **k: session)
    invocation = NS(skill_id="slide", invocation_id="slide")
    assert engine.start((invocation,), NS()) is session
    assert engine._withdrawal_was_reselected("withdraw")

    def failed(*args, **kwargs):
        raise RuntimeError("replacement planning failed before transactional install")

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", failed)
    request = NS(
        invocation_id="slide",
        skill_id="slide",
        goal=None,
        skill_options=NS(),
        motion_policy=NS(
            strategy="ik_interp", dynamic_collision_mode="off", sample_count=260
        ),
    )
    with pytest.raises(RuntimeError, match="transactional install"):
        engine._plan_request(request, NS(batch_size=1))
    assert engine._withdrawal_was_reselected("withdraw")
    session.plan_attempts += (attempt(False),)
    assert engine._withdrawal_was_reselected("withdraw")

    monkeypatch.setattr(AtomicActionEngine, "start", failed)
    with pytest.raises(RuntimeError, match="transactional install"):
        engine.start((invocation,), NS())
    assert engine._withdrawal_was_reselected("withdraw")

    next_session = NS(
        plan_attempts=(
            attempt(False),
            attempt(True, identity="foreign"),
            attempt(True, eligible=False),
        )
    )
    monkeypatch.setattr(AtomicActionEngine, "start", lambda *a, **k: next_session)
    engine.start((invocation,), NS())
    assert not engine._withdrawal_was_reselected("withdraw")


def test_preview_and_runtime_share_unchanged_withdrawal_targets():
    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([0.2, 0.3, 0.8])
    outward = torch.tensor([[0.0, -1.0, 0.0]])
    original = current.clone()
    result = articulation_slide._withdrawal_targets(current, outward)
    torch.testing.assert_close(result[:, 0], original)
    torch.testing.assert_close(result[0, 1, :3, 3], torch.tensor([0.2, 0.3, 0.92]))
    torch.testing.assert_close(result[0, 2, :3, 3], torch.tensor([0.2, 0.2, 0.92]))
    torch.testing.assert_close(current, original)
    torch.testing.assert_close(result[0, :, :3, :3], current[0, :3, :3].expand(3, 3, 3))


@pytest.mark.parametrize("accepted_index", [0, 1, None])
def test_failed_withdraw_preview_rejects_terminal_and_tries_next(
    monkeypatch, accepted_index
):
    solver = NS(device=torch.device("cpu"), _seed_sampler=None)
    monkeypatch.setattr(recovery, "PytorchSolver", type(solver))
    robot = NS(get_solver=lambda _: solver, get_qpos=lambda: torch.zeros(1, 2))
    generator = NS(robot=robot, planner=NS(cfg=NS(sim_instance_id=7)))
    scope = recovery.PlanningScope(generator, "slide", withdrawal_samples=260)
    options = NS(control_part="arm", start_qpos=torch.zeros(1, 2))
    scope.calls = [recovery.MotionCall([], options) for _ in range(4)]
    request = NS(
        skill_options=NS(release_retreat_distance=0.04),
        motion_policy=MotionPolicy(sample_count=260),
    )
    original = Plan(
        torch.tensor([False]), None, Diagnostics(failure="original_failure")
    )
    monkeypatch.setattr(recovery, "_Search", lambda *a: NS())
    monkeypatch.setattr(recovery, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    monkeypatch.setattr(recovery.SimulationManager, "get_instance", lambda _: object())
    monkeypatch.setattr(recovery, "check_free_motion", lambda *a: {"valid": True})

    def tracks(*args):
        for index in range(2):
            yield request, (torch.full((3, 2), index * 0.01),) * 4, {
                "endpoint_seed_index": index
            }

    monkeypatch.setattr(recovery, "_tracks", tracks)
    checked = []

    def preview(scope, request, context, search, trajectory):
        index = int(round(float(trajectory.positions[0, -1, 0]) * 100))
        checked.append(index)
        return {"valid": index == accepted_index, "scope": "kinematic_preview"}

    monkeypatch.setattr(recovery, "_preview_withdrawal", preview)

    def build(selected):
        scope.cursor = 4
        return Plan(
            torch.tensor([True]),
            NS(positions=scope.tracks[0][None], dt=torch.tensor([[0.0, 0.04, 0.04]])),
        )

    result = recovery.recover(
        scope,
        request,
        NS(batch_size=1, require_control_dt=lambda: 0.04),
        original,
        build,
    )
    assert checked == ([0] if accepted_index == 0 else [0, 1])
    assert result.plan_success.tolist() == [accepted_index is not None]
    if accepted_index is not None:
        assert (
            result.diagnostics.metadata["gen_sim_articulation_recovery"][
                "endpoint_seed_index"
            ]
            == accepted_index
        )
        assert result.diagnostics.metadata["gen_sim_articulation_recovery"][
            "withdrawal_selection_changed"
        ] is (accepted_index == 1)
    else:
        assert result.diagnostics.failure == "original_failure"
    assert scope.tracks is None and scope.search is None


@pytest.mark.parametrize("paired,clear", [(False, False), (True, False), (True, True)])
def test_actual_withdrawal_uses_fresh_scene_before_dispatch(monkeypatch, paired, clear):
    robot = NS(get_solver=lambda _: object(), get_qpos=lambda: torch.zeros(1, 2))
    generator = NS(robot=robot, planner=NS(cfg=NS(sim_instance_id=7)))
    scope = recovery.PlanningScope(
        generator, "withdraw", withdrawal_samples=260 if paired else None
    )
    scope.calls = [
        recovery.MotionCall(
            [], NS(control_part="arm", start_qpos=torch.zeros(1, 2), sample_count=260)
        )
    ]
    trajectory = NS(
        positions=torch.zeros(1, 3, 2), dt=torch.tensor([[0.0, 0.04, 0.04]])
    )
    original = Plan(torch.tensor([True]), trajectory)
    simulation = object()
    monkeypatch.setattr(
        recovery.SimulationManager, "get_instance", lambda _: simulation
    )
    monkeypatch.setattr(recovery, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    seen = []

    def check(r, sim, positions, part):
        assert r is robot and sim is simulation and part == "arm"
        assert torch.equal(positions, trajectory.positions[0])
        seen.append(True)
        return {"valid": clear, "scope": "discrete_free_motion_meshes"}

    monkeypatch.setattr(recovery, "check_free_motion", check)
    result = recovery.recover(
        scope,
        NS(),
        NS(batch_size=1),
        original,
        lambda _: pytest.fail("unsupported fake solver must not replan"),
    )
    assert seen == ([True] if paired else [])
    assert result.plan_success.tolist() == [not paired or clear]
    if paired and not clear:
        assert result.diagnostics.failure.code == "gen_sim_withdrawal_clearance_failed"
    else:
        assert result.joint_trajectory is trajectory


@pytest.mark.parametrize("failure", [None, "ik", "velocity", "robot_collision"])
@pytest.mark.parametrize("axis_sign", [-1, 1])
def test_preview_preserves_frames_passive_joints_rng_and_rejects_invalid_paths(
    monkeypatch, failure, axis_sign
):
    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([0.2, 0.3, 0.8])
    link = torch.eye(4)[None]
    link[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    inverse = torch.eye(4)
    inverse[:2, :2] = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    inverse[:3, 3] = torch.tensor([0.1, -0.2, 0.05])
    terminal = torch.tensor([[0.2, 0.1, 0.3, 0.4]])
    robot = NS(get_joint_ids=lambda **kw: [1, 3], compute_fk=lambda **kw: current)
    seen = []

    def path(poses, arm):
        torch.testing.assert_close(arm, terminal[:, [1, 3]])
        expected = inverse @ torch.tensor([0.2 + 0.1 * axis_sign, 0.3, 0.92, 1.0])
        torch.testing.assert_close(poses[-1, :3, 3], expected[:3])
        torch.rand(2)
        random.random()
        np.random.rand(2)
        if failure == "ik":
            return None
        result = arm.expand(6, -1).clone()
        if failure == "velocity":
            result[-1, 0] += 0.5
        return result

    search = NS(
        robot=robot, part="arm", dt=0.04, to_root=lambda p: inverse @ p, path=path
    )
    scope = recovery.PlanningScope(NS(robot=robot), "slide", withdrawal_samples=6)
    request = NS(
        goal=NS(
            target_pose=link,
            semantics=NS(
                affordance=NS(
                    translation_axis=torch.tensor([0.0, float(axis_sign), 0.0])
                )
            ),
        )
    )
    monkeypatch.setattr(recovery, "_joint_velocity_limits", lambda *a: torch.ones(1, 4))

    def check(r, full, part):
        assert r is robot and part == "arm"
        torch.testing.assert_close(full[:, [0, 2]], terminal[:, [0, 2]].expand(6, 2))
        seen.append(True)
        return {"valid": failure != "robot_collision"}

    monkeypatch.setattr(recovery, "check_robot_motion", check)
    torch_rng, py_rng, np_rng = (
        torch.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )
    result = recovery._preview_withdrawal(
        scope, request, NS(), search, NS(positions=terminal[:, None])
    )
    assert result["valid"] is (failure is None)
    assert seen == ([] if failure in {"ik", "velocity"} else [True])
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert np_rng[2:] == np.random.get_state()[2:]
