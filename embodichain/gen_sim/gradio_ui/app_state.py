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

"""Shared, thread-safe runtime state and timing helpers."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app_config import (
    DEFAULT_ROBOT_PROFILE,
    LANGUAGE_EN,
    PHASE_DEFINITIONS,
    SCENE_MODE_INITIAL,
    TIMING_PHASE_LABELS,
    TIMING_PHASE_ORDER,
)


@dataclass(frozen=True)
class Phase:
    progress: int
    label: str


PHASES = {key: Phase(*value) for key, value in PHASE_DEFINITIONS.items()}


@dataclass
class RuntimeState:
    is_busy: bool = False
    run_token: str = field(default_factory=lambda: uuid.uuid4().hex)
    auto_loop_active: bool = False
    auto_loop_token: str | None = None
    auto_round: int = 0
    auto_scene_mode: str = SCENE_MODE_INITIAL
    auto_parallel_env: bool = False
    auto_robot_profile: str = DEFAULT_ROBOT_PROFILE
    language: str = LANGUAGE_EN
    process: subprocess.Popen[str] | None = None
    sim_process: subprocess.Popen[str] | None = None
    scene_engine_process: subprocess.Popen[str] | None = None
    scene_preview_process: subprocess.Popen[str] | None = None
    scene_engine_is_running: bool = False
    sim_started: bool = False
    sim_finished: bool = False
    sim_returncode: int | None = None
    phase_key: str = "idle"
    status: str = "Idle."
    task_text: str = ""
    input_task_text: str = ""
    input_scene_text: str = ""
    image_path: Path | None = None
    video_path: Path | None = None
    last_sent_video_signature: tuple[str, int] | None = None
    lerobot_video_path: Path | None = None
    lerobot_dataset_path: Path | None = None
    submitted_input_revision: int = 0
    object_model_path: Path | None = None
    scene_model_path: Path | None = None
    edited_scene_model_path: Path | None = None
    last_error: str | None = None
    log_lines: deque[str] = field(default_factory=deque)
    timing_started_ns: int | None = None
    current_timing_phase_key: str | None = None
    current_timing_phase_started_ns: int | None = None
    phase_durations_ns: dict[str, int] = field(default_factory=dict)
    simulation_started_monotonic_ns: int | None = None
    simulation_duration_ns: int | None = None


runtime = RuntimeState()
runtime_lock = threading.Lock()


def clear_run_timing_locked() -> None:
    runtime.timing_started_ns = None
    runtime.current_timing_phase_key = None
    runtime.current_timing_phase_started_ns = None
    runtime.phase_durations_ns.clear()
    runtime.simulation_started_monotonic_ns = None
    runtime.simulation_duration_ns = None


def start_run_timing_locked(phase_key: str) -> None:
    now_ns = time.monotonic_ns()
    runtime.timing_started_ns = now_ns
    runtime.current_timing_phase_key = phase_key
    runtime.current_timing_phase_started_ns = now_ns
    runtime.phase_durations_ns.clear()
    runtime.simulation_started_monotonic_ns = None
    runtime.simulation_duration_ns = None


def record_phase_transition_locked(new_phase_key: str) -> None:
    current_key = runtime.current_timing_phase_key
    current_started_ns = runtime.current_timing_phase_started_ns
    now_ns = time.monotonic_ns()
    if current_key is None or current_started_ns is None:
        runtime.timing_started_ns = runtime.timing_started_ns or now_ns
        runtime.current_timing_phase_key = new_phase_key
        runtime.current_timing_phase_started_ns = now_ns
        return
    if new_phase_key == current_key:
        return
    runtime.phase_durations_ns[current_key] = runtime.phase_durations_ns.get(
        current_key, 0
    ) + max(0, now_ns - current_started_ns)
    runtime.current_timing_phase_key = new_phase_key
    runtime.current_timing_phase_started_ns = now_ns


def set_runtime_phase_locked(new_phase_key: str) -> None:
    record_phase_transition_locked(new_phase_key)
    runtime.phase_key = new_phase_key


def record_simulation_started_locked() -> None:
    runtime.simulation_started_monotonic_ns = time.monotonic_ns()
    runtime.simulation_duration_ns = None


def record_simulation_finished_locked() -> None:
    started_ns = runtime.simulation_started_monotonic_ns
    if started_ns is not None:
        runtime.simulation_duration_ns = max(0, time.monotonic_ns() - started_ns)
        runtime.simulation_started_monotonic_ns = None


def snapshot_timing_locked() -> tuple[dict[str, int], int | None]:
    durations = dict(runtime.phase_durations_ns)
    current_key = runtime.current_timing_phase_key
    current_started_ns = runtime.current_timing_phase_started_ns
    if (
        current_key
        and current_started_ns is not None
        and current_key not in {"complete", "failed", "idle"}
    ):
        durations[current_key] = durations.get(current_key, 0) + max(
            0, time.monotonic_ns() - current_started_ns
        )
    simulation_duration_ns = runtime.simulation_duration_ns
    if (
        simulation_duration_ns is None
        and runtime.simulation_started_monotonic_ns is not None
    ):
        simulation_duration_ns = max(
            0, time.monotonic_ns() - runtime.simulation_started_monotonic_ns
        )
    return durations, simulation_duration_ns


def format_duration_ns(duration_ns: int) -> str:
    seconds = duration_ns / 1_000_000_000
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds - minutes * 60:05.2f}s"


def format_timing_lines(
    phase_durations_ns: dict[str, int], simulation_duration_ns: int | None
) -> list[str]:
    timing_values = dict(phase_durations_ns)
    if simulation_duration_ns is not None:
        timing_values["action_graph_execution"] = simulation_duration_ns
    return [
        f"- {TIMING_PHASE_LABELS[key]}: {format_duration_ns(value) if (value := timing_values.get(key)) is not None else 'skipped'}"
        for key in TIMING_PHASE_ORDER
    ]
