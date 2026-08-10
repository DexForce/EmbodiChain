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

import sys
from pathlib import Path
from typing import Any

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_processes  # noqa: E402
import app_state  # noqa: E402
import app_workflows  # noqa: E402

FIRST_SESSION = "first-session"
SECOND_SESSION = "second-session"
TEST_VISER_PORT = 18_082


class FakeProcess:
    """Minimal subprocess stand-in for workflow ownership tests."""


class FakeRequest:
    """Minimal Gradio request carrying a stable session hash."""

    def __init__(self, session_hash: str) -> None:
        self.session_hash = session_hash


@pytest.fixture
def isolated_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    app_state.SessionRuntimeRegistry,
    app_processes.SessionProcessRegistry,
    app_processes.SessionProcessRegistry,
    app_processes.SessionProcessRegistry,
]:
    """Give each workflow test fresh session and process registries."""
    runtime_registry = app_state.SessionRuntimeRegistry()
    scene_runs = app_processes.SessionProcessRegistry()
    action_runs = app_processes.SessionProcessRegistry()
    action_preview_runs = app_processes.SessionProcessRegistry()
    monkeypatch.setattr(app_workflows, "runtime_registry", runtime_registry)
    monkeypatch.setattr(app_workflows, "_scene_runs", scene_runs)
    monkeypatch.setattr(app_workflows, "_action_runs", action_runs)
    monkeypatch.setattr(app_workflows, "_action_preview_runs", action_preview_runs)
    return runtime_registry, scene_runs, action_runs, action_preview_runs


def _attach_process(
    registry: app_processes.SessionProcessRegistry,
    session_id: str,
    process: FakeProcess,
) -> str:
    """Register one fake process and return its run token."""
    token = registry.begin(session_id)
    assert registry.attach(session_id, token, process)
    return token


def test_runtime_registry_returns_distinct_state_per_session() -> None:
    registry = app_state.SessionRuntimeRegistry()

    first_state = registry.get(FIRST_SESSION)
    second_state = registry.get(SECOND_SESSION)

    assert registry.get(FIRST_SESSION) is first_state
    assert second_state is not first_state


def test_stop_action_engine_terminates_only_requesting_session_action_processes(
    isolated_registries: tuple[
        app_state.SessionRuntimeRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_registry, scene_runs, action_runs, action_preview_runs = isolated_registries
    first_action = FakeProcess()
    first_action_preview = FakeProcess()
    first_scene = FakeProcess()
    second_action = FakeProcess()
    first_action_token = _attach_process(action_runs, FIRST_SESSION, first_action)
    first_preview_token = _attach_process(
        action_preview_runs, FIRST_SESSION, first_action_preview
    )
    first_scene_token = _attach_process(scene_runs, FIRST_SESSION, first_scene)
    second_action_token = _attach_process(action_runs, SECOND_SESSION, second_action)
    first_state = runtime_registry.get(FIRST_SESSION)
    first_state.is_busy = True
    stopped: list[Any] = []
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    updates = app_workflows.stop_action_engine(FakeRequest(FIRST_SESSION))

    assert stopped == [first_action, first_action_preview]
    assert not action_runs.is_active(FIRST_SESSION, first_action_token, first_action)
    assert not action_preview_runs.is_active(
        FIRST_SESSION, first_preview_token, first_action_preview
    )
    assert scene_runs.is_active(FIRST_SESSION, first_scene_token, first_scene)
    assert action_runs.is_active(SECOND_SESSION, second_action_token, second_action)
    assert updates[2:5] == (None, "", 0)


def test_reset_scene_engine_terminates_only_requesting_session_scene_process(
    isolated_registries: tuple[
        app_state.SessionRuntimeRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_registry, scene_runs, action_runs, _action_preview_runs = (
        isolated_registries
    )
    first_scene = FakeProcess()
    second_scene = FakeProcess()
    first_action = FakeProcess()
    first_scene_token = _attach_process(scene_runs, FIRST_SESSION, first_scene)
    second_scene_token = _attach_process(scene_runs, SECOND_SESSION, second_scene)
    first_action_token = _attach_process(action_runs, FIRST_SESSION, first_action)
    first_state = runtime_registry.get(FIRST_SESSION)
    first_state.is_busy = True
    first_state.scene_engine_is_running = True
    second_state = runtime_registry.get(SECOND_SESSION)
    second_state.is_busy = True
    second_state.scene_engine_is_running = True
    stopped: list[Any] = []
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    app_workflows.reset_scene_engine(FakeRequest(FIRST_SESSION))

    assert stopped == [first_scene]
    assert not scene_runs.is_active(FIRST_SESSION, first_scene_token, first_scene)
    assert scene_runs.is_active(SECOND_SESSION, second_scene_token, second_scene)
    assert action_runs.is_active(FIRST_SESSION, first_action_token, first_action)
    assert not first_state.is_busy
    assert not first_state.scene_engine_is_running
    assert second_state.is_busy
    assert second_state.scene_engine_is_running


def test_action_preview_replaces_only_same_session_preview(
    isolated_registries: tuple[
        app_state.SessionRuntimeRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
    ],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _runtime_registry, scene_runs, _action_runs, action_preview_runs = (
        isolated_registries
    )
    previous_preview = FakeProcess()
    other_preview = FakeProcess()
    scene_preview = FakeProcess()
    new_preview = FakeProcess()
    _attach_process(action_preview_runs, FIRST_SESSION, previous_preview)
    other_token = _attach_process(action_preview_runs, SECOND_SESSION, other_preview)
    scene_token = _attach_process(scene_runs, FIRST_SESSION, scene_preview)
    commands: list[list[str]] = []
    stopped: list[Any] = []
    monkeypatch.setattr(app_workflows, "_saved_scene_root", lambda _name: tmp_path)
    monkeypatch.setattr(
        app_workflows, "_select_available_port", lambda _port: TEST_VISER_PORT
    )
    monkeypatch.setattr(app_workflows, "_wait_for_viser", lambda _port, _process: True)
    monkeypatch.setattr(
        app_workflows,
        "start_pipeline",
        lambda command: commands.append(command) or new_preview,
    )
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    app_workflows.preview_saved_scene(
        "saved-scene",
        FakeRequest(FIRST_SESSION),
    )

    assert stopped == [previous_preview]
    assert scene_runs.is_active(FIRST_SESSION, scene_token, scene_preview)
    assert action_preview_runs.is_active(SECOND_SESSION, other_token, other_preview)
    assert commands[0][-1] == str(TEST_VISER_PORT)


def test_session_cleanup_preserves_other_sessions_processes_and_state(
    isolated_registries: tuple[
        app_state.SessionRuntimeRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
        app_processes.SessionProcessRegistry,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_registry, scene_runs, action_runs, action_preview_runs = isolated_registries
    first_processes = [FakeProcess(), FakeProcess(), FakeProcess()]
    second_processes = [FakeProcess(), FakeProcess(), FakeProcess()]
    first_registrations = [
        (registry, _attach_process(registry, FIRST_SESSION, process), process)
        for registry, process in zip(
            (scene_runs, action_runs, action_preview_runs), first_processes
        )
    ]
    second_registrations = [
        (registry, _attach_process(registry, SECOND_SESSION, process), process)
        for registry, process in zip(
            (scene_runs, action_runs, action_preview_runs), second_processes
        )
    ]
    first_state = runtime_registry.get(FIRST_SESSION)
    second_state = runtime_registry.get(SECOND_SESSION)
    stopped: list[Any] = []
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    app_workflows.cleanup_workflow_session(FakeRequest(FIRST_SESSION))

    assert stopped == first_processes
    assert all(
        not registry.is_active(FIRST_SESSION, token, process)
        for registry, token, process in first_registrations
    )
    assert all(
        registry.is_active(SECOND_SESSION, token, process)
        for registry, token, process in second_registrations
    )
    assert runtime_registry.get(FIRST_SESSION) is not first_state
    assert runtime_registry.get(SECOND_SESSION) is second_state
