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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from typing import ClassVar
import pytest
import torch
from embodichain.gen_sim.task_engine import _bundle_runner
from embodichain.gen_sim.task_engine._bundle_runner import _exception_metadata


def test_environment_modifier_keeps_shared_full_episode_recording(tmp_path):
    state_event = {"func": "record_joint_state"}
    config = {"task_program": {}, "env": {"events": {"record_state": state_event}}}
    _bundle_runner._configure_environment(config, tmp_path)
    assert "task_program" not in config
    events = config["env"]["events"]
    assert set(events) == {"record_state", "record_camera"}
    assert events["record_state"] is state_event
    assert events["record_camera"]["func"] == "record_camera_data"
    camera = events["record_camera"]["params"]
    assert camera["name"] == "task_program_audience_view"
    assert camera["resolution"] == [640, 360]
    assert camera["save_path"] == str(tmp_path / "videos")


@pytest.mark.parametrize("raises", [False, True])
def test_single_cpu_runtime_thread_scope_restores_caller(monkeypatch, raises):
    from embodichain.gen_sim.task_engine._task_program.motion import _CPU_BATCH_THREADS

    state = {"threads": 8}
    calls = []

    def set_threads(value):
        calls.append(value)
        state["threads"] = value

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu", num_envs=1))
    try:
        with _bundle_runner._cpu_runtime_threads(env):
            assert state["threads"] == 1
            assert _CPU_BATCH_THREADS.get() == 8
            if raises:
                raise RuntimeError("episode failure")
    except RuntimeError:
        assert raises
    assert calls == [1, 8]
    assert state["threads"] == 8
    assert _CPU_BATCH_THREADS.get() is None


@pytest.mark.parametrize("device, num_envs", [("cuda:0", 1), ("cpu", 2)])
def test_runtime_threads_leave_gpu_and_batch_policy_unchanged(
    monkeypatch, device, num_envs
):
    setter = Mock()
    monkeypatch.setattr(torch, "set_num_threads", setter)
    env = SimpleNamespace(unwrapped=SimpleNamespace(device=device, num_envs=num_envs))
    with _bundle_runner._cpu_runtime_threads(env):
        pass
    setter.assert_not_called()


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
            "adapter_contract": "gen_sim.task_program/2620929c/v13",
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
