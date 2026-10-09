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
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TrajectorySegment,
)
from embodichain.lab.sim.atomic_actions.policies import RecoveryPolicy
from embodichain.lab.sim.atomic_actions.primitives.place import Place
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    TimedCommandSequence,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy
from embodichain.lab.sim.motion.motion_generator import MotionGenCfg


def _place_plan(
    *,
    dependencies: tuple[str, ...] = (),
    monitor_until: dict[str, int] | None = None,
    approach: bool = True,
) -> ActionPlan:
    env_ids = torch.tensor([0])
    frame = RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget("arm", (0,)),
                payload=JointPositionPayload(torch.zeros(1, 1)),
            ),
        ),
        active_mask=torch.ones(1, dtype=torch.bool),
        env_ids=env_ids,
        hold_duration=torch.tensor([0.04]),
    )
    return ActionPlan(
        skill_id="place",
        plan_success=torch.ones(1, dtype=torch.bool),
        commands=TimedCommandSequence(frames=(frame,) * 6, env_ids=env_ids),
        recovery_policy=RecoveryPolicy(),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=0,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics(backend="test"),
        segments=(
            (
                TrajectorySegment("approach", 0, 2),
                TrajectorySegment("release", 2, 4),
                TrajectorySegment("retract", 4, 6),
            )
            if approach
            else ()
        ),
        scene_dependencies=dependencies,
        scene_dependency_monitor_until=monitor_until or {},
    )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("existing_cutoff", [None, 1])
def test_gensim_place_restores_v38_contact_boundary_without_changing_shared_plan(
    monkeypatch: pytest.MonkeyPatch, direct: bool, existing_cutoff: int | None
) -> None:
    original = _place_plan(
        dependencies=("support", "target"),
        monitor_until=({} if existing_cutoff is None else {"support": existing_cutoff}),
    )
    monkeypatch.setattr(Place, "_plan", lambda *_a: original)
    action = GenSimPlace()
    action._planning_services = SimpleNamespace(
        motion_generator=SimpleNamespace(direct_place_planning=direct)
    )

    result = action._plan(object(), object())

    assert result.scene_dependency_monitor_until == {
        "support": 2 if existing_cutoff is None else existing_cutoff,
        "target": 2,
    }
    torch.testing.assert_close(result.plan_success, original.plan_success)
    torch.testing.assert_close(result.commands.env_ids, original.commands.env_ids)
    assert len(result.commands.frames) == len(original.commands.frames)
    for actual, expected in zip(result.commands.frames, original.commands.frames):
        torch.testing.assert_close(actual.active_mask, expected.active_mask)
        torch.testing.assert_close(actual.hold_duration, expected.hold_duration)
        assert len(actual.commands) == len(expected.commands)
        for actual_command, expected_command in zip(actual.commands, expected.commands):
            assert actual_command.target == expected_command.target
            torch.testing.assert_close(
                actual_command.payload.positions, expected_command.payload.positions
            )
    assert result.expected_effects is original.expected_effects
    assert result.effect_verification is original.effect_verification
    assert result.diagnostics.backend == original.diagnostics.backend
    assert result.diagnostics.metadata == original.diagnostics.metadata
    assert result.segments == original.segments
    assert result.scene_dependencies == original.scene_dependencies
    shared = Place()._plan(object(), object())
    assert shared is original
    assert shared.scene_dependency_monitor_until == (
        {} if existing_cutoff is None else {"support": existing_cutoff}
    )


@pytest.mark.parametrize("approach", [False, True])
def test_gensim_place_leaves_unrelated_plans_unchanged(
    monkeypatch: pytest.MonkeyPatch, approach: bool
) -> None:
    original = _place_plan(
        dependencies=() if approach else ("support",), approach=approach
    )
    monkeypatch.setattr(Place, "_plan", lambda *_a: original)
    action = GenSimPlace()
    action._planning_services = SimpleNamespace(
        motion_generator=SimpleNamespace(direct_place_planning=True)
    )
    assert action._plan(object(), object()) is original


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
    request, context, result = object(), object(), _place_plan()

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
    captured = {}

    def initialize_generator(self: motion.MotionGenerator, cfg: MotionGenCfg) -> None:
        self.robot = robot
        captured["planner_cfg"] = cfg.planner_cfg

    monkeypatch.setattr(
        motion.MotionGenerator,
        "__init__",
        initialize_generator,
    )
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    monkeypatch.setattr(assembly, "install_grasp_filters", lambda *a: {})

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
    simulation = SimpleNamespace(instance_id="place-policy-composition-sim")
    adapter.create_adapter(
        SimpleNamespace(robot=robot, sim=simulation, step_dt=step_dt)
    )
    generator = captured["generator"]
    assert type(generator) is motion.CheckedMotionGenerator
    assert captured["planner_cfg"].sim_instance_id == simulation.instance_id

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
            segments=(),
            scene_dependencies=(),
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
