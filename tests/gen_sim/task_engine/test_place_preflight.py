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

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import place_recovery
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions


def fixture():
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 0.8
    arm = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        get_held_object=lambda _: SimpleNamespace(semantics=SimpleNamespace()),
    )
    request = SimpleNamespace(
        goal=PlaceGoal(pose),
        binding=SimpleNamespace(
            endpoint=lambda *_: SimpleNamespace(
                task_state_key="held", require_target=lambda _: arm
            )
        ),
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
        skill_options=PlaceOptions(
            preserve_current_object_orientation=True,
            cartesian_waypoint_count=2,
            lift_height=0.05,
        ),
    )
    captured = {}

    def ik(*, pose, joint_seed, name):
        torch.rand(1)
        assert name == "arm"
        assert joint_seed.shape == (1, 6, 2)
        captured["poses"] = pose.clone()
        return torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 6, 2)

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        compute_fk=lambda **_: torch.eye(4)[None],
        compute_batch_ik=ik,
        compute_batch_fk=lambda **_: captured["poses"].clone(),
        get_qpos_limits=lambda **_: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
        get_solver=lambda _: SimpleNamespace(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
    )
    action = SimpleNamespace(
        robot=robot,
        device=robot.device,
        _resolve_place_xpos=lambda *_, **__: (pose.clone(), None),
    )
    return action, request, context, captured


def test_required_keyframe_preflight_preserves_rng_goal_and_all_keyframes():
    action, request, context, captured = fixture()
    rng = torch.get_rng_state().clone()
    before = request.goal.xpos.clone()
    result = place_recovery.preflight(action, request, context)
    assert result["valid"] is True
    assert result["required_keyframes"] == 6
    assert result["full_plans"] == 0
    assert result["physical_evidence"] is False
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(before, request.goal.xpos)
    assert torch.equal(captured["poses"][:, 3], before)
    assert captured["poses"][0, 1, 2, 3] == pytest.approx(0.85)


@pytest.mark.parametrize("failure", ["ik", "fk", "limits"])
def test_one_bad_required_keyframe_cannot_be_hidden_by_other_successes(failure):
    action, request, context, captured = fixture()
    original = action.robot.compute_batch_ik

    def rejected(**kwargs):
        success, qpos = original(**kwargs)
        if failure == "ik":
            success[0, 1] = False
        if failure == "limits":
            qpos[0, 1, 0] = 2.0
        return success, qpos

    action.robot.compute_batch_ik = rejected
    if failure == "fk":

        def wrong_fk(**_):
            actual = captured["poses"].clone()
            actual[:, 1, 0, 3] += 0.01
            return actual

        action.robot.compute_batch_fk = wrong_fk
    result = place_recovery.preflight(action, request, context)
    assert result["valid"] is False
    assert result["scope"] == "finite_required_waypoint_ik_not_global_unreachability"


def test_explicit_clearance_and_nonordinary_requests_keep_existing_path():
    from dataclasses import replace

    action, request, context, _ = fixture()
    request.skill_options = replace(request.skill_options, max_approach_retract_z=0.9)
    action.robot.compute_batch_ik = lambda **_: pytest.fail("Protected call was probed")
    assert place_recovery.preflight(action, request, context) is None


def test_unqualified_keyframe_stops_before_any_full_plan(monkeypatch):
    from dataclasses import dataclass

    from embodichain.gen_sim.task_engine._task_program import actions
    from embodichain.lab.sim.atomic_actions import PlannerDiagnostics
    from embodichain.lab.sim.atomic_actions.primitives.place import Place

    @dataclass(frozen=True)
    class Plan:
        diagnostics: PlannerDiagnostics

    source, request, context, _ = fixture()
    action = actions.GenSimPlace()
    action._planning_services = SimpleNamespace(
        robot=source.robot, motion_generator=SimpleNamespace()
    )
    action.recovery_scene_vertices = {"measured": object()}
    monkeypatch.setattr(
        actions.GenSimStateHandoff,
        "from_request",
        lambda *_, **__: SimpleNamespace(resource="held"),
    )
    monkeypatch.setattr(place_recovery, "preflight", lambda *_: {"valid": False})
    monkeypatch.setattr(Place, "_plan", lambda *_: pytest.fail("Full planning ran"))
    failure = {}

    def failed(*_, **kwargs):
        failure.update(kwargs)
        return Plan(PlannerDiagnostics(backend="test"))

    monkeypatch.setattr(action, "failed_plan", failed)
    result = action._plan(request, context)
    assert failure["failure_code"] == "gen_sim_place_keyframes_unqualified"
    assert failure["retryable"] is False
    assert result.diagnostics.metadata["gen_sim_place_keyframe_preflight"] == {
        "valid": False
    }
