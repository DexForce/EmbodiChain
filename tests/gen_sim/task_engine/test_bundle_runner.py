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

import pytest

from embodichain.gen_sim.task_engine import _bundle_runner
from embodichain.gen_sim.task_engine._bundle_runner import _exception_metadata


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
