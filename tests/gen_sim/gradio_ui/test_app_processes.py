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


class FakeProcess:
    """Minimal subprocess stand-in for ownership tests."""


def test_reset_stops_only_the_requesting_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = app_processes.SessionProcessRegistry()
    first_process = FakeProcess()
    second_process = FakeProcess()
    stopped: list[Any] = []
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    first_token = registry.begin("first-session")
    second_token = registry.begin("second-session")
    assert registry.attach("first-session", first_token, first_process)
    assert registry.attach("second-session", second_token, second_process)

    registry.reset("second-session")

    assert stopped == [second_process]
    assert registry.is_active("first-session", first_token, first_process)
    assert not registry.is_active("second-session", second_token, second_process)


def test_new_run_replaces_only_the_same_sessions_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = app_processes.SessionProcessRegistry()
    previous_process = FakeProcess()
    stopped: list[Any] = []
    monkeypatch.setattr(app_processes, "terminate_process_group", stopped.append)

    previous_token = registry.begin("session")
    assert registry.attach("session", previous_token, previous_process)

    replacement_token = registry.begin("session")

    assert stopped == [previous_process]
    assert not registry.is_active("session", previous_token)
    assert registry.is_active("session", replacement_token)


def test_codex_environment_excludes_service_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp/codex-user")
    monkeypatch.setenv("OPENAI_API_KEY", "server-api-key")
    monkeypatch.setenv("SIMREADY_OPENAI_API_KEY", "simready-api-key")
    monkeypatch.setenv("SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL", "https://service")

    child_env = app_processes.build_codex_env()

    assert child_env["PATH"] == "/usr/bin"
    assert child_env["HOME"] == "/tmp/codex-user"
    assert "OPENAI_API_KEY" not in child_env
    assert "SIMREADY_OPENAI_API_KEY" not in child_env
    assert "SCENE_ENGINE_IMAGE_SEGMENTATION_BASE_URL" not in child_env


def test_sensitive_output_values_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_TOKEN", "sensitive-token-value")

    assert (
        app_processes.redact_sensitive_text("credential=sensitive-token-value")
        == "credential=[REDACTED]"
    )
