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

import os
import threading
import time
from unittest.mock import Mock

import pytest
import torch

from examples.data_pipeline import _ods_pick_place_training as training
from examples.data_pipeline._ods_pick_place_benchmark import resource_snapshot


def _snapshot(*args, **kwargs) -> dict:
    return {
        "processes": {},
        "missing_pids": [],
        "host_pss_bytes": None,
        "torch_cuda_this_process": {},
        "monotonic_seconds": time.monotonic(),
    }


def test_deferred_scalars_use_checkpoint_policy_step() -> None:
    writer = Mock()
    report = {
        "policy_step": "5000",
        "success_rate": 0.9,
        "lift_rate": 1.0,
        "goal_error": 0.01,
        "episode_length": 320.0,
    }
    training._write_evaluation_scalars(writer, report)
    assert [call.args[2] for call in writer.add_scalar.call_args_list] == [5000] * 4
    assert [call.args[0] for call in writer.add_scalar.call_args_list] == [
        "eval/success_rate",
        "eval/lift_rate",
        "eval/goal_error",
        "eval/episode_length",
    ]


def test_invalid_policy_step_cannot_publish_mislabeled_scalars() -> None:
    writer = Mock()
    with pytest.raises(ValueError, match="nonnegative"):
        training._write_evaluation_scalars(writer, {"policy_step": -1})
    writer.add_scalar.assert_not_called()


def test_short_training_run_records_boundaries_without_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(training, "resource_snapshot", _snapshot)
    sampler = training._ResourceSampler((), interval_seconds=5.0)
    sampler.start()
    sampler.capture("training_start")
    sampler.capture("training_end")
    records = sampler.stop()
    assert [row["boundary"] for row in records] == [
        "before_worker_startup",
        "training_start",
        "training_end",
    ]
    assert sampler.report()["stopped"]
    assert sampler.report()["samples"] == 3
    assert sampler.report()["collection_wall_seconds"] >= 0
    sampler.capture("too_late")
    assert len(sampler.stop()) == 3


def test_wall_cadence_runs_while_owner_waits_and_stop_joins(monkeypatch) -> None:
    periodic_entered = threading.Event()
    release = threading.Event()
    calls = 0

    def snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            periodic_entered.set()
            assert release.wait(2)
        return _snapshot()

    monkeypatch.setattr(training, "resource_snapshot", snapshot)
    sampler = training._ResourceSampler((), interval_seconds=0.01)
    sampler.start()
    assert periodic_entered.wait(2)
    stopped = threading.Event()
    stopper = threading.Thread(target=lambda: (sampler.stop(), stopped.set()))
    stopper.start()
    assert sampler._stop.wait(2)
    assert not stopped.is_set()  # The outstanding sample must finish first.
    release.set()
    assert stopped.wait(2)
    stopper.join(2)
    assert not sampler._thread.is_alive()
    assert sampler.report()["samples"] >= 2


def test_missing_process_measurements_stay_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        training,
        "resource_snapshot",
        lambda *a, **k: {**_snapshot(), "missing_pids": [999], "host_pss_bytes": None},
    )
    sampler = training._ResourceSampler(())
    sampler.start()
    records = sampler.stop()
    assert records[0]["host_pss_bytes"] is None
    assert sampler.report()["missing_pid_observations"] == 1


def test_collection_failure_is_reported_and_cleanup_is_idempotent(monkeypatch) -> None:
    def denied(*args, **kwargs):
        raise PermissionError("unavailable process metrics")

    monkeypatch.setattr(training, "resource_snapshot", denied)
    sampler = training._ResourceSampler(())
    sampler.start()
    assert sampler.stop() == []
    assert sampler.stop() == []
    report = sampler.report()
    assert report["samples"] == 0
    assert "PermissionError" in report["errors"][0]["error"]
    assert report["stopped"]


def test_resource_sampling_does_not_initialize_cuda_or_consume_rng(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Resource sampling must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "memory_allocated", forbidden)
    monkeypatch.setattr(torch.cuda, "memory_reserved", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", forbidden)
    before = torch.get_rng_state().clone()
    snapshot = resource_snapshot([os.getpid()], cuda_devices=(0,))
    assert snapshot["torch_cuda_this_process"]["cuda:0"] is None
    assert torch.equal(before, torch.get_rng_state())


def test_worker_startup_failure_stops_sampler_and_preserves_original_error(
    monkeypatch, tmp_path
) -> None:
    import json
    import sys
    from types import SimpleNamespace
    from examples.data_pipeline.ods_pick_place_showcase import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "showcase",
            "--output",
            str(tmp_path / "run"),
            "--train-device",
            "cpu",
            "--sim-device",
            "cpu",
            "--no-viewer",
            "--no-video",
        ],
    )
    args = _parse_args()
    monkeypatch.setattr(training, "load_task_config", lambda *a, **k: {})
    monkeypatch.setattr(training, "resource_snapshot", _snapshot)

    class FailedProcess:
        pid = None
        exitcode = None

        def start(self):
            raise RuntimeError("deliberate worker startup failure")

    monkeypatch.setattr(
        training.mp,
        "get_context",
        lambda *a: SimpleNamespace(
            Queue=lambda: Mock(), Process=lambda *a, **k: FailedProcess()
        ),
    )
    with pytest.raises(RuntimeError, match="deliberate worker startup failure"):
        training.run(args)
    resources = json.loads((args.output / "resources.json").read_text())
    assert resources["sampling"]["started"]
    assert resources["sampling"]["stopped"]
    assert not any(
        thread.name == "ods-resource-sampler" for thread in threading.enumerate()
    )
    failure = json.loads((args.output / "failure.json").read_text())
    assert "deliberate worker startup failure" in failure["error"]
    cleanup = json.loads((args.output / "cleanup.json").read_text())
    assert cleanup["resource_sampler_cleanup_error"] is None
