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
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import assembly, motion
from embodichain.gen_sim.task_engine._task_program.actions import GenSimPlace
from embodichain.lab.sim.atomic_actions.primitives.place import Place


@pytest.mark.parametrize("kind", ["place", "transport"])
def test_explicit_drawer_scope_preserves_its_existing_primitive(kind, monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimMoveHeldObject,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
        MoveHeldObject,
    )

    action_type, primitive, flag = (
        (GenSimPlace, Place, "direct_place_planning")
        if kind == "place"
        else (GenSimMoveHeldObject, MoveHeldObject, "direct_transport_planning")
    )
    action = action_type()
    action._planning_services = SimpleNamespace(
        motion_generator=SimpleNamespace(**{flag: True})
    )
    request, context, result = object(), object(), object()

    def delegated(self, actual_request, actual_context):
        assert actual_request is request and actual_context is context
        assert motion._PLACE_IK_RECOVERY.get() is None
        return result

    monkeypatch.setattr(primitive, "_plan", delegated)
    assert action._plan(request, context) is result


@pytest.mark.parametrize("cartesian_approaches", [False, True])
def test_factory_place_recovery_is_independent_of_other_cartesian_calls(
    monkeypatch: pytest.MonkeyPatch, cartesian_approaches: bool
) -> None:
    """The adapter's legacy whole-program flag must not gate Place recovery."""
    step_dt, sample_count = 0.04, 5
    start = torch.eye(4).unsqueeze(0)
    target = start.clone()
    target[:, 2, 3] = 0.1
    robot = SimpleNamespace(uid="robot", compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(
        motion.MotionGenerator,
        "__init__",
        lambda self, cfg: setattr(self, "robot", robot),
    )
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    monkeypatch.setattr(assembly, "install_grasp_filters", lambda *a: {})
    captured = {}

    def factory(*args, **kwargs):
        captured["generator"] = kwargs["motion_generator_factory"]()
        return SimpleNamespace(create_adapter=lambda: object())

    monkeypatch.setattr(assembly, "_TaskFactory", factory)
    registration = SimpleNamespace(
        assert_unchanged=lambda: None,
        robot_profile_binding=SimpleNamespace(
            presets=[
                SimpleNamespace(motion_policy=SimpleNamespace(strategy="ik_interp"))
            ]
        ),
    )
    adapter = assembly.TaskAdapterFactory(
        registration,
        "test",
        (),
        (),
        cartesian_calls=("other",) if cartesian_approaches else (),
    )
    adapter.create_adapter(SimpleNamespace(robot=robot, sim=object(), step_dt=step_dt))
    generator = captured["generator"]
    assert type(generator) is motion.CheckedMotionGenerator

    failed = motion.PlanResult(
        success=torch.tensor([False]),
        positions=torch.zeros(1, sample_count, 1),
        dt=torch.full((1, sample_count), step_dt),
    )
    recovered = motion.PlanResult(
        success=torch.tensor([True]), positions=failed.positions, dt=failed.dt
    )
    attempts = []

    def plan_native(self, targets, options=None):
        attempts.append((targets, options))
        return failed

    monkeypatch.setattr(motion.MotionGenerator, "generate", plan_native)
    local = Mock(return_value=recovered)
    monkeypatch.setattr(motion, "_local_place_ik", local)

    def plan_place(self, request, context):
        result = generator.generate(
            [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=target)],
            motion.MotionGenOptions(
                strategy="ik_interp",
                control_part="arm",
                start_qpos=torch.zeros(1, 1),
                sample_count=sample_count,
                interpolation_dt=step_dt,
            ),
        )
        return SimpleNamespace(
            plan_success=result.success,
            joint_trajectory=SimpleNamespace(positions=result.positions, dt=result.dt),
        )

    monkeypatch.setattr(Place, "_plan", plan_place)
    action = GenSimPlace()
    action._planning_services = SimpleNamespace(robot=robot, motion_generator=generator)
    result = action._plan(object(), object())

    assert result.plan_success.all()
    local.assert_called_once()
    assert [options.sample_count for _, options in attempts] == [sample_count, 260, 320]
    for targets, options in attempts:
        assert len(targets) == options.sample_count - 1
        assert options.preserve_cartesian_samples and options.is_linear
        assert options.interpolation_dt == step_dt
        torch.testing.assert_close(targets[-1].xpos, target)
    assert motion._PLACE_IK_RECOVERY.get() is None
