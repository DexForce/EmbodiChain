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
"""Protect unfiltered M06 latency statistics and episode/memory-call boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.tech_reports.physics.experiments.dexuni import (
    m06_measurement as m,
)
from scripts.benchmark.tech_reports.reporting.dexuni.build_m06_report import _stat


def _complete() -> tuple[dict, dict, np.ndarray]:
    """Supply valid timing inputs with deliberately implausible final geometry."""
    steps = 32
    manifest = {
        "status": "completed",
        "outer_steps": steps,
        "config": {"solver": {"dt": 1 / 60}},
        "requested_iterations": 8,
        "camera_count": 0,
        "effective": {
            "cuda_graph_status": "captured",
            "num_substeps": 10,
            "vbd_iterations": 8,
            "outer_steps": steps,
            "simulated_elapsed_s": steps / 60,
        },
    }
    states = {
        "time": np.arange(steps + 1) / 60,
        "particles": np.full((steps + 1, 1, 3), 1e6),
        "qpos": np.zeros((steps + 1, 9)),
        "target_q": np.zeros((steps + 1, 9)),
        "tcp": np.zeros((steps + 1, 3)),
    }
    return manifest, states, np.full(steps, 0.01)


def test_final_geometry_does_not_exclude_latency_sample() -> None:
    manifest, states, seconds = _complete()
    assert all(m.performance_checks(manifest, states, seconds).values())


@pytest.mark.parametrize(
    "problem",
    ["actual_time", "extra_steps", "budget", "tracking", "nonfinite", "partial_timing"],
)
def test_invalid_executed_protocol_is_detected(problem: str) -> None:
    manifest, states, seconds = _complete()
    if problem == "actual_time":
        states["time"] *= 10
    elif problem == "extra_steps":
        manifest["effective"]["outer_steps"] *= 10
    elif problem == "budget":
        manifest["effective"]["vbd_iterations"] = 15
    elif problem == "tracking":
        states["qpos"][-1, 0] = 1.0
    elif problem == "nonfinite":
        states["particles"][1, 0, 0] = np.nan
    else:
        seconds = seconds[:-1]
    assert not all(m.performance_checks(manifest, states, seconds).values())


def test_statistics_treat_episodes_as_independent_samples() -> None:
    result = _stat([1, 2, 3, 4, 5])
    assert result["mean"] == 3
    assert result["sample_std"] == pytest.approx(np.sqrt(2.5))


def test_failed_fold_is_saved_and_memory_step_is_not_an_episode_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the recorder with a failed goal and one separate memory operation."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib import (
        gpu_memory,
    )

    class Tracker:
        def start(self):
            return True

        def sample(self, _label):
            return 1.0

        def metrics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(gpu_memory, "GpuProcessMemoryTracker", Tracker)
    monkeypatch.setattr(m, "TASK", tmp_path)
    monkeypatch.setattr(m, "STEPS", 32)
    monkeypatch.setattr(m, "_environment", lambda: {})
    (tmp_path / "measurement").mkdir()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "measurement/protocol.json").write_text("{}")
    output = tmp_path / "result"
    output.mkdir()
    frame = [0]

    def advance():
        frame[0] += 1

    def timer(operation, _synchronize):
        operation()
        return 0.01

    def readings():
        return {
            "time": np.asarray(frame[0] / 60),
            "particles": np.zeros((1, 3)),
            "qpos": np.zeros(9),
            "tcp": np.zeros(3),
        }

    def diagnostics():
        return {
            "cuda_graph_status": "captured",
            "outer_steps": frame[0],
            "num_substeps": 10,
            "vbd_iterations": 15,
            "simulated_elapsed_s": frame[0] / 60,
        }

    def memory_call(operation, read, _synchronize, _tracker):
        operation()
        read()
        return {"scope": "independent memory step"}

    monkeypatch.setattr(m, "completed_step_seconds", timer)
    monkeypatch.setattr(m, "_memory_call", memory_call)
    trajectory = {
        "time": np.arange(33) / 60,
        "joint_targets": np.zeros((33, 9)),
        "desired_tcp": np.zeros((33, 3)),
    }
    m.record_episode(
        "isaaclab",
        output,
        {"solver": {"dt": 1 / 60}},
        {"left_ids": np.array([0]), "right_ids": np.array([0])},
        trajectory,
        advance,
        readings,
        lambda: None,
        diagnostics,
        0.1,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert frame[0] == 33
    assert manifest["status"] == "completed" and not manifest["task_success"]
    assert all(manifest["performance_checks"].values())
    assert manifest["effective"]["outer_steps"] == 32
    assert manifest["summary"]["measured_steps"] == 2
    with np.load(output / "states.npz") as states:
        assert len(states["time"]) == 33
        assert states["time"][-1] == pytest.approx(32 / 60)
    assert len((output / "timing.csv").read_text().splitlines()) == 33
