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

import pytest
import json
import torch

from embodichain.gen_sim.task_engine import _bundle_runner
from embodichain.gen_sim.task_engine._bundle_runner import _exception_metadata


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
        assembly as assembly_module,
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
        preflight_analyses=lambda: [SimpleNamespace(kind="sequential", calls=())]
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
