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

"""Tests for the isolated Task Program bundle subprocess boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from typing import ClassVar

import pytest
import json
import torch

from embodichain.gen_sim.task_engine import _bundle_runner
from embodichain.gen_sim.task_engine._bundle_runner import _exception_metadata


@pytest.mark.parametrize(
    "targets, expected",
    [
        ([[0.01, 0.01], [0.01, 0.2]], [True, False]),
        ([[0.2, 0.01], [0.01, 0.2]], [False, False]),
        ([[0.01, 0.01], [0.01, 0.01]], [True, True]),
    ],
)
def test_initial_velocity_gate_masks_real_session_before_dispatch(
    monkeypatch, targets, expected
):
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        ActionOptions,
        AtomicAction,
        JointPositionGoal,
        SkillBindingContract,
        SkillResourceSlot,
        SkillEndpointRequirement,
        JOINT_POSITION_CAPABILITY,
        TimedTrajectory,
    )
    from embodichain.gen_sim.task_engine._task_program import planning_probe
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    class FixtureAction(AtomicAction[JointPositionGoal, ActionOptions]):
        skill_id: ClassVar[str] = "fixture"
        GoalType: ClassVar[type] = JointPositionGoal
        binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
            slots=(
                SkillResourceSlot(
                    slot_id="primary",
                    endpoints=(
                        SkillEndpointRequirement(
                            endpoint_id="motion",
                            capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
                        ),
                    ),
                ),
            )
        )

        def _plan(self, request, context):
            torch.rand(4)
            return self.build_plan(
                request,
                context,
                success=torch.ones(2, dtype=torch.bool),
                trajectory=TimedTrajectory.from_uniform_step(
                    torch.stack((context.robot.qpos, request.goal.target), dim=1),
                    env_ids=context.env_ids,
                    step_dt=context.require_control_dt(),
                ),
            )

    robot = Mock(device=torch.device("cpu"), dof=2, control_parts={"all": object()})
    robot.get_qpos.return_value = torch.zeros(2, 2)
    robot.get_qvel.return_value = torch.zeros(2, 2)
    robot.get_joint_ids.return_value = [0, 1]
    generator = Mock(robot=robot, device=torch.device("cpu"))
    generator.planner.cfg.planner_type = "fixture"
    engine = GenSimActionEngine(generator, load_builtins=False)
    engine.register(FixtureAction())
    request_id = "test/segment-0:0"
    invocation = ActionInvocation(
        skill_id="fixture",
        invocation_id=request_id,
        goal=JointPositionGoal(torch.tensor(targets)),
        binding=engine.bind_control_parts("fixture", {"primary": {"motion": "all"}}),
    )
    compiled = SimpleNamespace(
        program_id="test",
        preflight_analyses=lambda: (
            SimpleNamespace(kind="sequential", calls=(object(),)),
        ),
        iter_segments=lambda: iter(
            (
                SimpleNamespace(
                    segment_id="segment-0",
                    calls=(SimpleNamespace(segment_call_index=0),),
                ),
            )
        ),
    )
    monkeypatch.setattr(
        planning_probe, "_joint_velocity_limits", lambda *a: torch.ones(2, 2)
    )
    evidence = []
    context = engine.initial_context(control_dt=0.04)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        torch.rand(4)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(17)
        with planning_probe.capture_initial_plan(
            compiled, evidence.append, _exception_metadata
        ):
            session = engine.start((invocation,), context)
            tick = session.tick(context)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert evidence[0]["plan_success"] == expected
    assert len(session.plan_attempts) == 1
    if any(expected):
        assert tick.command.active_mask.tolist() == expected
    else:
        assert tick.command is None
        assert not session.plan_attempts[0].plan.diagnostics.failure.retryable
    assert not session.task_state.held_objects
    assert planning_probe.initial_plan_capture(request_id) is None


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_initial_capture_restores_outer_scope_on_failure_or_cancel(error):
    from embodichain.gen_sim.task_engine._task_program.planning_probe import (
        capture_initial_plan,
        initial_plan_capture,
    )

    def program(name):
        return SimpleNamespace(
            program_id=name,
            preflight_analyses=lambda: (
                SimpleNamespace(kind="sequential", calls=(object(),)),
            ),
            iter_segments=lambda: iter(
                (
                    SimpleNamespace(
                        segment_id="segment",
                        calls=(SimpleNamespace(segment_call_index=0),),
                    ),
                )
            ),
        )

    with capture_initial_plan(program("outer"), Mock(), _exception_metadata) as outer:
        with pytest.raises(error):
            with capture_initial_plan(
                program("inner"), Mock(), _exception_metadata
            ) as inner:
                assert initial_plan_capture(outer.invocation_id) is None
                assert initial_plan_capture(inner.invocation_id) is inner
                raise error()
        assert initial_plan_capture(outer.invocation_id) is outer
    assert initial_plan_capture(outer.invocation_id) is None


def test_initial_capture_counts_sequential_workflows_not_segments():
    from embodichain.gen_sim.task_engine._task_program.planning_probe import (
        InitialPlanCapture,
    )

    segments = tuple(
        SimpleNamespace(
            segment_id=f"segment-{i}", calls=(SimpleNamespace(segment_call_index=0),)
        )
        for i in range(3)
    )
    compiled = SimpleNamespace(
        program_id="test",
        iter_segments=lambda: iter(segments),
        preflight_analyses=lambda: (
            SimpleNamespace(kind="sequential", calls=tuple(object() for _ in range(3))),
        ),
    )
    records = []
    capture = InitialPlanCapture(compiled, records.append, _exception_metadata)
    capture.failed(ValueError("fixture"))
    assert records[0]["analysis_call_count"] == 3
    assert records[0]["unplanned_call_indices"] == [1, 2]
    assert records[0]["remaining_analysis_count"] == 0


@pytest.mark.parametrize(
    "probe_only, outcome",
    [
        (False, "success"),
        (True, "success"),
        (False, "planning_failure"),
        (False, "velocity_failure"),
        (False, "exception"),
    ],
)
def test_bundle_plans_initial_call_once(tmp_path, monkeypatch, probe_only, outcome):
    import gymnasium
    from dataclasses import dataclass
    from embodichain.lab.gym.envs import demo
    from embodichain.lab.gym.utils import gym_utils, registration
    from embodichain.lab.task_program import language
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        MotionPolicy,
        PlannerDiagnostics,
        PlanningFailure,
    )
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.gen_sim.task_engine._task_program import assembly, cargo
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    @dataclass
    class Plan:
        plan_success: torch.Tensor
        joint_trajectory: object
        diagnostics: PlannerDiagnostics

    root = tmp_path / "bundle"
    (root / "task_program").mkdir(parents=True)
    for name in (
        "task_program_deployment.yaml",
        "task_program/program.yaml",
        "semantic_task_graph.json",
        "integration_fingerprint.json",
    ):
        (root / name).touch()
    graph = {
        "task_id": "test",
        "integration_fingerprint": "0" * 64,
        "nodes": [{"id": "first"}],
        "task_groups": [{"id": "step", "node_ids": ["first"]}],
    }
    segment = SimpleNamespace(
        segment_id="segment-0", calls=(SimpleNamespace(segment_call_index=0),)
    )
    compiled = SimpleNamespace(
        program_id="test",
        iter_segments=lambda: iter((segment,)),
        preflight_analyses=lambda: (
            SimpleNamespace(kind="sequential", calls=(object(),)),
        ),
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                catalog=SimpleNamespace(preflight=lambda p: compiled)
            )
        ),
        selection=None,
    )
    robot_file = tmp_path / "robot.urdf"
    robot_file.write_text(
        '<robot><joint name="arm"><limit velocity="1"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(robot_file)),
        joint_names=["arm"],
        get_joint_ids=lambda **kw: [0],
        get_qvel_limits=lambda **kw: torch.ones(1, 1),
    )
    engine = object.__new__(GenSimActionEngine)
    engine._cartesian_calls = frozenset()
    engine._articulation_calls = {}
    engine._planning_services = SimpleNamespace(robot=robot)
    request = SimpleNamespace(
        invocation_id="test/segment-0:0",
        motion_policy=MotionPolicy(),
        skill_options=SimpleNamespace(),
    )
    positions = torch.zeros(1, 2, 1)
    if outcome == "velocity_failure":
        positions[0, 1, 0] = 0.2
    planning = Mock(
        return_value=Plan(
            torch.tensor([outcome != "planning_failure"]),
            SimpleNamespace(positions=positions, dt=torch.tensor([[0.0, 0.04]])),
            PlannerDiagnostics(
                backend="test",
                failure=(
                    PlanningFailure("fixture_unreachable")
                    if outcome == "planning_failure"
                    else None
                ),
            ),
        )
    )
    if outcome == "exception":
        planning.side_effect = ValueError("fixture planning exception")
    monkeypatch.setattr(AtomicActionEngine, "_plan_request", planning)
    env = SimpleNamespace(reset=Mock(), close=Mock())
    monkeypatch.setattr(gymnasium, "make", lambda **kw: env)
    monkeypatch.setattr(registration, "discover_task_packages", lambda: None)
    monkeypatch.setattr(registration, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(assembly, "register_deployment", lambda *a, **kw: None)
    monkeypatch.setattr(language, "load_task_program", lambda *a, **kw: object())
    monkeypatch.setattr(
        gym_utils,
        "build_env_cfg_from_args",
        lambda *a, **kw: (SimpleNamespace(), {"id": "fixture"}, {}),
    )
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", lambda: None)
    rejected = outcome in {"planning_failure", "velocity_failure"}
    monkeypatch.setattr(
        cargo, "capture_cargo", lambda *a: [object()] if rejected else []
    )
    cargo_check = Mock(return_value={"accepted_mask": [True], "contents": []})
    monkeypatch.setattr(cargo, "check_cargo", cargo_check)
    monkeypatch.setattr(_bundle_runner, "_verify_source", lambda *a: None)
    monkeypatch.setattr(_bundle_runner, "validate_semantic_task_graph", lambda d: graph)
    monkeypatch.setattr(_bundle_runner, "_read_json", lambda p: {})
    monkeypatch.setattr(
        _bundle_runner, "_verify_integration_fingerprint", lambda *a: deployment
    )
    monkeypatch.setattr(_bundle_runner, "_verify_program_projection", lambda *a: None)
    monkeypatch.setattr(
        _bundle_runner,
        "load_config",
        lambda *a: {"id": "fixture", "max_episode_steps": 100},
    )
    monkeypatch.setattr(_bundle_runner, "_write_terminal_robot_state", lambda *a: None)
    monkeypatch.setattr(
        _bundle_runner, "_preserve_failed_execution_recording", lambda *a, **kw: None
    )
    monkeypatch.setattr(_bundle_runner, "_print_json", lambda *a: None)

    def probe(*args):
        plan = engine._plan_request(request)
        return {"plan_success": plan.plan_success.tolist()}

    probe_call = Mock(side_effect=probe)
    monkeypatch.setattr(_bundle_runner, "_isolated_initial_probe", probe_call)

    dispatched = []

    def execute(*args, **kwargs):
        plan = engine._plan_request(request)
        if plan.plan_success.any():
            dispatched.append(plan)
        return SimpleNamespace(
            success=plan.plan_success.tolist(),
            terminal_reasons=["success"],
            completed=True,
            to_metadata=lambda: {
                "segments": [
                    {
                        "name": "first",
                        "active": [True],
                        "successes": plan.plan_success.tolist(),
                    }
                ]
            },
        )

    execute_call = Mock(side_effect=execute)
    monkeypatch.setattr(demo, "execute_demo_episode", execute_call)
    assert _bundle_runner.execute_bundle(
        root,
        ["--plan-probe-only"] if probe_only else (),
        execution_output=tmp_path / "execution",
    ) == (0 if outcome == "success" else 2)
    assert planning.call_count == 1
    assert probe_call.call_count == int(probe_only)
    assert execute_call.call_count == int(not probe_only)
    report = json.loads((tmp_path / "execution/planning_probe.json").read_text())
    assert report["plan_success"] == (
        [] if outcome == "exception" else [outcome == "success"]
    )
    assert len(dispatched) == int(not probe_only and outcome == "success")
    if rejected:
        cargo_check.assert_not_called()
    if not probe_only:
        assert report["planning_source"] == "execution"
        from embodichain.gen_sim.task_engine._task_program.planning_probe import (
            initial_plan_capture,
        )

        assert initial_plan_capture(request.invocation_id) is None
        execution_report = json.loads(
            (tmp_path / "execution/execution_report.json").read_text()
        )
        assert execution_report["status"] == (
            "succeeded"
            if outcome == "success"
            else "failed" if outcome == "exception" else "rejected"
        )
    if outcome == "exception":
        assert report["failure"]["message"] == "fixture planning exception"


@pytest.mark.parametrize("fails", [False, True])
def test_initial_probe_restores_random_streams(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    import random
    import numpy as np

    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()

    def probe(*args):
        random.random()
        np.random.random(20)
        torch.rand(20)
        if fails:
            raise ValueError("probe failed")
        return {"plan_success": [True]}

    monkeypatch.setattr(_bundle_runner, "_probe_initial_plan", probe)
    if fails:
        with pytest.raises(ValueError, match="probe failed"):
            _bundle_runner._isolated_initial_probe(None, None, None)
    else:
        assert _bundle_runner._isolated_initial_probe(None, None, None)[
            "plan_success"
        ] == [True]
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
    assert torch.equal(torch.random.get_rng_state(), before_torch)


@pytest.mark.parametrize("success", [True, False])
def test_initial_probe_plans_without_dispatch_or_effect_commit(
    success: bool, monkeypatch
) -> None:
    from embodichain.gen_sim.task_engine._task_program import (
        planning_probe as assembly_module,
    )

    monkeypatch.setattr(
        assembly_module, "_joint_velocity_limits", lambda *a: torch.ones(1, 2)
    )
    context = object()
    invocation = object()
    compiler = Mock()
    compiler.analyze.return_value = SimpleNamespace(
        calls=(
            SimpleNamespace(downstream_object_targets=(object(),)),
            SimpleNamespace(downstream_object_targets=()),
        )
    )
    compiler.ground.return_value = SimpleNamespace(invocation=invocation)
    engine = Mock()
    engine.plan.return_value = SimpleNamespace(
        plan_success=torch.tensor([success]),
        joint_trajectory=SimpleNamespace(
            positions=torch.zeros(1, 2, 2), dt=torch.tensor([[0.0, 0.04]])
        ),
    )
    observer = Mock()
    observer.observe.return_value = context
    assembly = SimpleNamespace(
        compiler=compiler, engine=engine, observation_provider=observer
    )
    compiled = SimpleNamespace(
        preflight_analyses=lambda: [SimpleNamespace(kind="sequential", calls=())],
        program_id="test",
        iter_segments=lambda: iter([SimpleNamespace(segment_id="first")]),
    )
    adapter = Mock()
    adapter.compile.return_value = compiled
    adapter.assemble_runtime.return_value = assembly
    factory = Mock()
    factory.create_adapter.return_value = adapter
    deployment = SimpleNamespace(
        integration=SimpleNamespace(adapter_factory=factory), selection=object()
    )
    env = SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda: torch.zeros(1, 2)))
    result = _bundle_runner._probe_initial_plan(env, deployment, object())
    compiler.analyze.assert_called_once_with((), workflow_id="test/first")
    assert result["plan_success"] == [success]
    assert result["task_success"] is None
    assert result["analysis_call_count"] == 2
    assert result["initial_downstream_target_count"] == 1
    assert result["unplanned_call_indices"] == [1]
    assert result["remaining_analysis_count"] == 0
    engine.plan.assert_called_once_with(invocation, context)
    assert len(engine.mock_calls) == 1
    state = observer.observe.call_args.args[0]
    assert not state.held_objects
    assert not state.coordinated_held_objects


def test_terminal_snapshot_preserves_pre_reset_measurement(tmp_path: Path) -> None:
    qpos = torch.tensor([[0.1, 0.2]])
    robot = SimpleNamespace(
        cfg=SimpleNamespace(control_parts={"hand": [0, 1]}),
        joint_names=["finger", "knuckle"],
        get_joint_ids=lambda **kwargs: [0, 1],
        get_qpos=lambda **kwargs: qpos,
    )
    env = SimpleNamespace(robot=robot)
    _bundle_runner._write_terminal_robot_state(env, tmp_path)
    qpos.zero_()
    _bundle_runner._write_terminal_robot_state(env, tmp_path)
    data = json.loads((tmp_path / "terminal_robot_state.json").read_text())
    assert data["control_parts"]["hand"]["joint_names"] == ["finger", "knuckle"]
    assert data["control_parts"]["hand"]["measured_qpos"][0] == pytest.approx(
        [0.1, 0.2]
    )


@pytest.mark.parametrize("argv, expected_seed", [([], 0), (["--seed", "7"], 7)])
def test_runner_parser_preserves_seed_with_shared_launcher(
    argv: list[str], expected_seed: int
) -> None:
    """Compose the real launcher without registering its seed argument twice."""
    parser = _bundle_runner._runner_parser()

    assert parser.parse_args(argv).seed == expected_seed


def test_exception_metadata_preserves_explicit_causal_chain() -> None:
    """The report retains the physical planner error hidden by demo cleanup."""
    try:
        try:
            raise ValueError("invalid coordinated trajectory")
        except ValueError as planner_error:
            raise RuntimeError("demo safe-stop completed") from planner_error
    except RuntimeError as runtime_error:
        metadata = _exception_metadata(runtime_error)

    assert metadata == {
        "type": "RuntimeError",
        "message": "demo safe-stop completed",
        "causes": [
            {
                "type": "ValueError",
                "message": "invalid coordinated trajectory",
            }
        ],
    }


def test_exception_metadata_rejects_non_exception_values() -> None:
    """The private serializer fails closed on an invalid diagnostic value."""
    with pytest.raises(TypeError, match="BaseException"):
        _exception_metadata("failure")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fingerprint",
    [
        {},
        {"schema_version": "semantic_integration_fingerprint/v1"},
        {
            "schema_version": "semantic_integration_fingerprint/v2",
            "adapter_contract": "gen_sim.task_program/2620929c/v2",
        },
        {
            "schema_version": "semantic_integration_fingerprint/v2",
            "adapter_contract": "unknown",
        },
    ],
)
def test_old_bundle_is_rejected_before_component_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fingerprint: dict
) -> None:
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("Old contracts must fail before component loading.")

    monkeypatch.setattr(_bundle_runner, "load_config", forbidden_load)
    with pytest.raises(ValueError, match="regenerate"):
        _bundle_runner._verify_integration_fingerprint(
            tmp_path, tmp_path / "deployment.yaml", {}, fingerprint
        )


def test_execution_report_preserves_partial_row_success() -> None:
    """A normal partial result remains eligible for Task Engine any/at-least."""
    graph = {
        "task_id": "partial",
        "integration_fingerprint": "0" * 64,
        "nodes": [{"id": "step_01"}],
        "task_groups": [{"id": "group_01", "node_ids": ["step_01"]}],
    }
    runtime_result = {
        "segments": [
            {
                "name": "step_01",
                "active": [True, True],
                "successes": [True, False],
            }
        ]
    }

    report = _bundle_runner._build_execution_report(
        graph,
        runtime_result,
        row_success=[True, False],
        terminal_reasons=["success", "task_incomplete"],
        failure=None,
        trajectory_root=Path("trajectory"),
    )

    assert report["status"] == "failed"
    assert [row["success"] for row in report["environments"]] == [True, False]
    assert report["failure"] is None


def test_cargo_failure_masks_only_affected_semantic_groups() -> None:
    graph = {
        "task_id": "cargo",
        "integration_fingerprint": "0" * 64,
        "nodes": [
            {"id": "move", "task_type": "E1", "call": {"object": "cup"}},
            {
                "id": "carry",
                "task_type": "E5",
                "call": {"arguments": {"object": "tray"}},
            },
        ],
        "task_groups": [
            {"id": "e1", "node_ids": ["move"]},
            {"id": "e5", "node_ids": ["carry"]},
        ],
    }
    runtime = {
        "segments": [
            {"name": name, "active": [True, True], "successes": [True, True]}
            for name in ("move", "carry")
        ]
    }
    report = _bundle_runner._build_execution_report(
        graph,
        runtime,
        row_success=[True, True],
        terminal_reasons=["success", "success"],
        failure=None,
        trajectory_root=Path("trajectory"),
        cargo_report={
            "accepted_mask": [True, False],
            "contents": [{"carrier": "tray", "accepted_mask": [True, False]}],
        },
    )
    assert report["status"] == "failed"
    assert report["environments"][0]["success"] is True
    assert report["environments"][1]["success"] is False
    assert report["environments"][1]["semantic_success"] == {"e1": True, "e5": False}
    assert report["environments"][1]["terminal_reason"] == "cargo_envelope_failed"


def test_execution_report_masks_rows_after_global_failure() -> None:
    """An infrastructure failure invalidates every row in the attempt."""
    graph = {
        "task_id": "failed",
        "integration_fingerprint": "0" * 64,
        "nodes": [],
        "task_groups": [],
    }

    report = _bundle_runner._build_execution_report(
        graph,
        None,
        row_success=[True, False],
        terminal_reasons=["success", "runtime_failed"],
        failure={"type": "RuntimeError", "message": "transport failed"},
        trajectory_root=Path("trajectory"),
    )

    assert [row["success"] for row in report["environments"]] == [False, False]


@pytest.mark.parametrize("capture_fails", [False, True])
def test_failed_attempt_captures_a_terminal_frame_before_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_fails: bool
) -> None:
    from embodichain.lab.gym.envs.managers import record

    calls = []

    class Recorder:
        def __call__(self, env, env_ids, **params):
            calls.append(("capture", env, env_ids, params))
            if capture_fails:
                raise RuntimeError("camera fetch failed")

        def save_and_clear(self):
            calls.append(("flush",))

    monkeypatch.setattr(record, "record_camera_data", Recorder)
    params = {"name": "audience", "resolution": [640, 360]}
    env = SimpleNamespace(
        event_manager=SimpleNamespace(
            _mode_functor_cfgs={
                "interval": [SimpleNamespace(func=Recorder(), params=params)]
            }
        )
    )
    _bundle_runner._preserve_failed_execution_recording(env, tmp_path, num_envs=1)

    assert calls == [("capture", env, None, params), ("flush",)]


def test_module_entrypoint_flushes_protocol_before_fast_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulator worker skips native interpreter-order destruction."""
    flushes: list[str] = []
    exit_codes: list[int] = []

    monkeypatch.setattr(_bundle_runner, "main", lambda: 7)
    monkeypatch.setattr(
        _bundle_runner.sys,
        "stdout",
        SimpleNamespace(flush=lambda: flushes.append("stdout")),
    )
    monkeypatch.setattr(
        _bundle_runner.sys,
        "stderr",
        SimpleNamespace(flush=lambda: flushes.append("stderr")),
    )

    def fake_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        raise SystemExit(exit_code)

    monkeypatch.setattr(_bundle_runner.os, "_exit", fake_exit)

    with pytest.raises(SystemExit, match="7"):
        _bundle_runner._module_entrypoint()

    assert flushes == ["stdout", "stderr"]
    assert exit_codes == [7]


__all__: list[str] = []
