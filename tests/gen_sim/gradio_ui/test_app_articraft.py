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
from types import SimpleNamespace
from typing import Any

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_articraft  # noqa: E402

VIEWER_PORT = 54_321
REQUEST_ID = "0123456789abcdef"


class _FakeServerClient:
    """Provide deterministic task states for Gradio callback tests."""

    def __init__(
        self,
        statuses: list[dict[str, Any]],
        *,
        request_id: str = REQUEST_ID,
        submit_error: Exception | None = None,
        status_error: Exception | None = None,
        cancel_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.statuses = statuses
        self.request_id = request_id
        self.submit_error = submit_error
        self.status_error = status_error
        self.cancel_error = cancel_error
        self.download_error = download_error
        self.cancelled: list[str] = []
        self.downloaded: list[tuple[str, str]] = []
        self.submitted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ready"}

    def submit(self, prompt: str, *, image: Path | None = None) -> dict[str, Any]:
        self.submitted.append(prompt)
        if self.submit_error is not None:
            raise self.submit_error
        return {"request_id": self.request_id}

    def status(self, request_id: str) -> dict[str, Any]:
        if self.status_error is not None:
            raise self.status_error
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def cancel(self, request_id: str) -> dict[str, Any]:
        self.cancelled.append(request_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        return {"request_id": request_id, "status": "cancelled"}

    def download(self, request_id: str, artifact: str, destination: Path) -> Path:
        if self.download_error is not None:
            raise self.download_error
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"usdc")
        self.downloaded.append((request_id, artifact))
        return destination.resolve()


def _request(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(session_hash=session_id)


@pytest.mark.parametrize("hostname", ["127.0.0.1", "localhost"])
def test_articraft_viewer_port_accepts_local_http_url(hostname: str) -> None:
    viewer_output = f"Viewer URL: http://{hostname}:{VIEWER_PORT}/"

    assert app_articraft._articraft_viewer_port(viewer_output) == VIEWER_PORT


@pytest.mark.parametrize(
    "viewer_output",
    [
        f"Viewer URL: https://localhost:{VIEWER_PORT}",
        f"Viewer URL: http://example.com:{VIEWER_PORT}",
        "Viewer URL: http://localhost:not-a-port",
        "Articraft viewer is starting",
    ],
)
def test_articraft_viewer_port_rejects_untrusted_or_invalid_output(
    viewer_output: str,
) -> None:
    assert app_articraft._articraft_viewer_port(viewer_output) is None


def test_codex_checkout_cannot_be_the_embodichain_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_articraft,
        "ARTICRAFT_ROOT",
        app_articraft.EMBODICHAIN_ROOT,
    )

    assert "dedicated nested or external Git checkout" in (
        app_articraft._articraft_isolation_error() or ""
    )


def test_default_provider_routes_to_remote_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ("server.usdc", "server-run", "done", "log", "preview")

    def generate_server(*_args: object):
        yield expected

    monkeypatch.setattr(
        app_articraft, "_generate_server_articulation_asset", generate_server
    )

    results = list(
        app_articraft._generate_selected_articulation_asset(
            app_articraft._DEFAULT_ARTICULATION_PROVIDER,
            "prompt",
            None,
            _request("default-provider"),
        )
    )

    assert app_articraft._DEFAULT_ARTICULATION_PROVIDER == "Remote server"
    assert results == [expected]


def test_local_codex_provider_uses_existing_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ("local.usdc", "local-run", "done", "log", "preview")

    def generate_local(*_args: object):
        yield expected

    monkeypatch.setattr(app_articraft, "generate_articraft_asset", generate_local)

    results = list(
        app_articraft._generate_selected_articulation_asset(
            "Local Codex", "prompt", None, _request("local-provider")
        )
    )

    assert results == [expected]


def test_local_codex_provider_uses_existing_environment_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_articraft,
        "configure_articraft_environment",
        lambda: "local environment ready",
    )
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_TIMEOUT_S", "30s")
    monkeypatch.setattr(
        app_articraft, "ARTICULATION_SERVER_TASK_TIMEOUT_S", "eventually"
    )
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_POLL_INTERVAL_S", "often")

    result = app_articraft._configure_selected_articulation_provider("Local Codex")

    assert result == "local environment ready"


def test_remote_server_environment_check_uses_health_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient([{"status": "running"}])
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)

    result = app_articraft._configure_selected_articulation_provider("Remote server")

    assert "server is ready" in result
    assert (tmp_path / "server").is_dir()


def test_remote_server_environment_check_requires_explicit_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_BASE_URL", "")

    result = app_articraft._configure_selected_articulation_provider("Remote server")

    assert "server is not ready" in result
    assert "ARTICULATION_SERVER_BASE_URL" in result


def test_local_codex_command_preserves_existing_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        app_articraft, "_articraft_cli_command", lambda *args: list(args)
    )

    command = app_articraft._articraft_generation_command("prompt", None)

    assert command[:3] == ["generate", "--provider", "codex-cli"]
    assert command[-1] == "prompt"


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("ARTICULATION_SERVER_TASK_TIMEOUT_S", float("nan")),
        ("ARTICULATION_SERVER_TASK_TIMEOUT_S", float("inf")),
        ("ARTICULATION_SERVER_POLL_INTERVAL_S", float("nan")),
        ("ARTICULATION_SERVER_POLL_INTERVAL_S", float("inf")),
    ],
)
def test_remote_server_rejects_non_finite_polling_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setting: str,
    value: float,
) -> None:
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, setting, value)

    with pytest.raises(ValueError, match=f"{setting}.*finite positive number"):
        app_articraft._server_output_root()


@pytest.mark.parametrize(
    "setting",
    [
        "ARTICULATION_SERVER_TIMEOUT_S",
        "ARTICULATION_SERVER_TASK_TIMEOUT_S",
        "ARTICULATION_SERVER_POLL_INTERVAL_S",
    ],
)
def test_remote_server_reports_malformed_timing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setting: str,
) -> None:
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, setting, "30s")

    result = app_articraft._configure_selected_articulation_provider("Remote server")

    assert "server is not ready" in result
    assert setting in result
    assert "finite positive number" in result


def test_remote_server_log_is_bounded() -> None:
    log_lines: list[str] = []

    for index in range(app_articraft._SERVER_LOG_LIMIT + 5):
        app_articraft._append_server_log(log_lines, f"line-{index}")

    assert len(log_lines) == app_articraft._SERVER_LOG_LIMIT
    assert log_lines[0] == "line-5"


def test_remote_server_success_polls_and_downloads_usdc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient(
        [{"status": "running", "stage": "model"}, {"status": "succeeded"}]
    )
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)
    monkeypatch.setattr(app_articraft.time, "sleep", lambda _seconds: None)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-success")
        )
    )

    artifact = tmp_path / "server" / REQUEST_ID / "model.usdc"
    assert results[-1][0] == artifact.resolve().as_posix()
    assert results[-1][1] == artifact.parent.as_posix()
    assert "generation completed" in results[-1][2]
    assert client.downloaded == [(REQUEST_ID, "usdc")]
    assert artifact.read_bytes() == b"usdc"


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
def test_remote_server_terminal_failure_is_reported_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_status: str,
) -> None:
    client = _FakeServerClient(
        [{"status": terminal_status, "error": "generation stopped"}]
    )
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request(f"remote-{terminal_status}")
        )
    )

    assert terminal_status in results[-1][2]
    assert "not retried with Local Codex" in results[-1][2]
    assert client.downloaded == []


def test_remote_server_timeout_requests_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient([{"status": "running"}])
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_TASK_TIMEOUT_S", 1.0)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)
    monkeypatch.setattr(app_articraft.time, "monotonic", lambda: next(clock))

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-timeout")
        )
    )

    assert "timed out" in results[-1][2]
    assert client.cancelled == [REQUEST_ID]


def test_remote_server_poll_sleep_does_not_exceed_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient([{"status": "running"}])
    clock = iter([0.0, 0.0, 0.0, 5.0])
    sleeps: list[float] = []
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_TASK_TIMEOUT_S", 5.0)
    monkeypatch.setattr(app_articraft, "ARTICULATION_SERVER_POLL_INTERVAL_S", 60.0)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)
    monkeypatch.setattr(app_articraft.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(app_articraft.time, "sleep", sleeps.append)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-bounded-sleep")
        )
    )

    assert sleeps == [5.0]
    assert "timed out" in results[-1][2]
    assert client.cancelled == [REQUEST_ID]


def test_remote_server_submit_error_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient(
        [{"status": "running"}],
        submit_error=app_articraft.ArticulationServerError("connection refused"),
    )
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-error")
        )
    )

    assert "connection refused" in results[-1][2]
    assert "not retried with Local Codex" in results[-1][2]


def test_remote_server_status_error_requests_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient(
        [{"status": "running"}],
        status_error=app_articraft.ArticulationServerError("status unavailable"),
    )
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-status-error")
        )
    )

    assert "status check failed" in results[-1][2]
    assert client.cancelled == [REQUEST_ID]


def test_remote_server_download_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient(
        [{"status": "succeeded"}],
        download_error=app_articraft.ArticulationServerError("artifact unavailable"),
    )
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "a service bell", None, _request("remote-download-error")
        )
    )

    assert "could not be downloaded" in results[-1][2]
    assert "artifact unavailable" in results[-1][2]


def test_reset_cancels_active_remote_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient([{"status": "running"}])
    session_id = "remote-reset"
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)
    generator = app_articraft._generate_server_articulation_asset(
        "a service bell", None, _request(session_id)
    )
    next(generator)

    app_articraft.cleanup_articraft_session(session_id)

    assert client.cancelled == [REQUEST_ID]
    assert list(generator) == []


def test_reset_reports_remote_cancellation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeServerClient(
        [{"status": "running"}],
        cancel_error=app_articraft.ArticulationServerError("cancel unavailable"),
    )
    session_id = "remote-reset-error"
    request = _request(session_id)
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(app_articraft, "_articulation_server_client", lambda: client)
    generator = app_articraft._generate_server_articulation_asset(
        "a service bell", None, request
    )
    next(generator)

    reset_values = app_articraft.reset_articraft_asset(request)

    assert "could not be cancelled" in reset_values[5]
    client.cancel_error = None
    app_articraft.cleanup_articraft_session(session_id)
    assert list(generator) == []


def test_invalid_remote_request_id_does_not_remain_registered() -> None:
    client = _FakeServerClient(
        [{"status": "running"}],
        cancel_error=ValueError("invalid request id"),
    )
    session_id = "remote-invalid-request-id"
    token = app_articraft._articraft_runs.begin(session_id)
    assert app_articraft._register_server_task(
        session_id, token, client, "not/a/request-id"
    )

    cancellation_error = app_articraft._cancel_server_task(session_id)

    assert cancellation_error is None
    assert session_id not in app_articraft._server_tasks
    app_articraft._articraft_runs.reset(session_id)


def test_new_request_cancels_replaced_remote_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _FakeServerClient([{"status": "running"}], request_id="a" * 16)
    second = _FakeServerClient([{"status": "running"}], request_id="b" * 16)
    clients = iter([first, second])
    session_id = "remote-replacement"
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        app_articraft, "_articulation_server_client", lambda: next(clients)
    )
    first_generator = app_articraft._generate_server_articulation_asset(
        "first", None, _request(session_id)
    )
    second_generator = app_articraft._generate_server_articulation_asset(
        "second", None, _request(session_id)
    )
    next(first_generator)

    next(second_generator)

    assert first.cancelled == ["a" * 16]
    assert list(first_generator) == []
    app_articraft.cleanup_articraft_session(session_id)
    assert second.cancelled == ["b" * 16]


def test_replacement_is_blocked_when_previous_task_cannot_be_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _FakeServerClient(
        [{"status": "running"}],
        request_id="a" * 16,
        cancel_error=app_articraft.ArticulationServerError("cancel unavailable"),
    )
    second = _FakeServerClient([{"status": "running"}], request_id="b" * 16)
    clients = iter([first, second])
    session_id = "remote-replacement-blocked"
    monkeypatch.setattr(app_articraft, "ARTICRAFT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        app_articraft, "_articulation_server_client", lambda: next(clients)
    )
    first_generator = app_articraft._generate_server_articulation_asset(
        "first", None, _request(session_id)
    )
    next(first_generator)

    results = list(
        app_articraft._generate_server_articulation_asset(
            "second", None, _request(session_id)
        )
    )

    assert "No replacement task was submitted" in results[-1][2]
    assert second.submitted == []
    first.cancel_error = None
    app_articraft.cleanup_articraft_session(session_id)
