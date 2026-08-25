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

"""Remote and Codex-backed Articulation generation for the Asset engine.

Remote generation uses the configured articulation-server by default. The
existing local integration continues to run the configured Articraft fork with
its ``codex-cli`` provider and native USDZ viewer. All downloaded and generated
artifacts are kept under ``ARTICRAFT_OUTPUT_ROOT``.
"""

from __future__ import annotations

import html
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import gradio as gr

from _articulation_server_client import (
    ArticulationServerClient,
    ArticulationServerError,
)
from app_env import (
    ARTICULATION_SERVER_BASE_URL,
    ARTICULATION_SERVER_POLL_INTERVAL_S,
    ARTICULATION_SERVER_TASK_TIMEOUT_S,
    ARTICULATION_SERVER_TIMEOUT_S,
    ARTICRAFT_CONDA_ENV,
    ARTICRAFT_OUTPUT_ROOT,
    ARTICRAFT_REPOSITORY_URL,
    ARTICRAFT_ROOT,
    EMBODICHAIN_ROOT,
    validate_gradio_artifact_root,
)
from app_processes import (
    SessionProcessRegistry,
    build_codex_env,
    get_request_session_id,
    read_process_output,
    redact_sensitive_text,
    register_managed_process,
    terminate_process_group,
)
from embodichain.gen_sim.env import find_gen_sim_env_file

__all__ = [
    "build_articraft_panel",
    "cleanup_articraft_session",
    "configure_articraft_environment",
    "generate_articraft_asset",
    "reset_articraft_asset",
]

_ARTICRAFT_ENVIRONMENT_SETUP_TIMEOUT_SECONDS = 1_200
_articraft_environment_lock = threading.Lock()
_articraft_runs = SessionProcessRegistry()
_articraft_viewers = SessionProcessRegistry()
_REMOTE_SERVER_PROVIDER = "Remote server"
_LOCAL_CODEX_PROVIDER = "Local Codex"
_DEFAULT_ARTICULATION_PROVIDER = _REMOTE_SERVER_PROVIDER
_SERVER_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_SERVER_LOG_LIMIT = 300
_server_tasks_lock = threading.Lock()
_server_tasks: dict[str, tuple[str, ArticulationServerClient, str]] = {}
_ARTICRAFT_IDLE_PREVIEW = (
    "<div style='padding: 1rem; color: #6b7280;'>"
    "The Articulation result preview will appear here after generation."
    "</div>"
)


def reset_articraft_asset(request: gr.Request) -> tuple[Any, ...]:
    """Clear Articraft state and stop the requesting session's generation.

    Args:
        request: Gradio request for the browser session initiating Reset.

    Returns:
        Reset values for all Articraft panel widgets.
    """
    cancellation_error = _cleanup_articraft_session(get_request_session_id(request))
    status = "**Status:** waiting for a description."
    if cancellation_error:
        status = (
            "**Reset completed, but the remote task could not be cancelled.**\n\n"
            f"- {cancellation_error}\n"
            "- Retry Reset before starting another generation."
        )
    return (
        "**Environment:** not checked.",
        "",
        None,
        None,
        "",
        status,
        "",
        _ARTICRAFT_IDLE_PREVIEW,
    )


def cleanup_articraft_session(session_id: str) -> None:
    """Stop Articraft generation for one browser session.

    Args:
        session_id: Stable Gradio session identifier.
    """
    _cleanup_articraft_session(session_id)


def _cleanup_articraft_session(session_id: str) -> str | None:
    """Stop local work and request cancellation of one remote task."""
    _articraft_runs.reset(session_id, force=True)
    cancellation_error = _cancel_server_task(session_id)
    _articraft_viewers.reset(session_id, force=True)
    return cancellation_error


def _cancel_server_task(session_id: str) -> str | None:
    """Cancel and forget one session's active remote task."""
    with _server_tasks_lock:
        active = _server_tasks.pop(session_id, None)
    if active is None:
        return None
    _token, client, request_id = active
    try:
        client.cancel(request_id)
    except ValueError:
        return None
    except (ArticulationServerError, OSError) as exc:
        with _server_tasks_lock:
            _server_tasks.setdefault(session_id, active)
        return redact_sensitive_text(str(exc))
    return None


def _register_server_task(
    session_id: str,
    token: str,
    client: ArticulationServerClient,
    request_id: str,
) -> bool:
    """Attach a submitted server task to its active logical run."""
    if not _articraft_runs.is_active(session_id, token):
        try:
            client.cancel(request_id)
        except (ArticulationServerError, OSError, ValueError):
            pass
        return False
    with _server_tasks_lock:
        _server_tasks[session_id] = (token, client, request_id)
    if _articraft_runs.is_active(session_id, token):
        return True
    with _server_tasks_lock:
        active = _server_tasks.get(session_id)
        if active is not None and active[0] == token:
            _server_tasks.pop(session_id, None)
    try:
        client.cancel(request_id)
    except (ArticulationServerError, OSError, ValueError):
        pass
    return False


def _finish_server_task(session_id: str, token: str, request_id: str) -> None:
    """Forget a terminal server task without cancelling it."""
    with _server_tasks_lock:
        active = _server_tasks.get(session_id)
        if active is not None and active[0] == token and active[2] == request_id:
            _server_tasks.pop(session_id, None)


def _command_path(name: str) -> str | None:
    """Resolve a command even when Gradio has a non-interactive PATH."""
    configured = os.environ.get(f"{name.upper()}_EXE")
    return configured or shutil.which(name)


def _conda_path() -> str | None:
    """Return the Conda executable used to activate Articraft."""
    configured = os.environ.get("CONDA_EXE")
    if configured and Path(configured).is_file():
        return configured
    return _command_path("conda")


def _articraft_conda_prefix() -> Path | None:
    """Return the filesystem prefix for the configured Conda environment."""
    conda = _conda_path()
    if not conda:
        return None
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        environments = json.loads(result.stdout or "{}").get("envs", [])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError):
        return None
    for value in environments:
        prefix = Path(str(value)).expanduser()
        if prefix.name == ARTICRAFT_CONDA_ENV:
            return prefix.resolve()
    return None


def _conda_run_command(*args: str) -> list[str]:
    """Run a command inside the configured Articraft Conda environment."""
    conda = _conda_path()
    if not conda:
        raise RuntimeError("Conda was not found. Set CONDA_EXE before starting Gradio.")
    prefix = _articraft_conda_prefix()
    if prefix is None:
        raise RuntimeError(f"Conda environment {ARTICRAFT_CONDA_ENV!r} was not found.")

    library_paths = [str(prefix / "lib")]
    if inherited_library_path := os.environ.get("LD_LIBRARY_PATH"):
        library_paths.append(inherited_library_path)
    return [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        ARTICRAFT_CONDA_ENV,
        "env",
        f"LD_LIBRARY_PATH={os.pathsep.join(library_paths)}",
        f"PYTHONPATH={ARTICRAFT_ROOT / 'src'}",
        *args,
    ]


def _articraft_cli_command(*args: str) -> list[str]:
    """Run the new fork's source inside the existing Articraft Conda env."""
    return _conda_run_command("python", "-m", "articraft.app", *args)


def _short_output(
    result: subprocess.CompletedProcess[str], *, limit: int = 1_800
) -> str:
    """Return a bounded diagnostic suffix from a completed command."""
    output = (result.stdout or "").strip()
    return output[-limit:] if len(output) > limit else (output or "(no output)")


def _normalized_repository_url(value: str) -> str:
    """Normalize equivalent HTTPS and SSH GitHub repository URLs."""
    normalized = value.strip().rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    elif normalized.startswith("ssh://git@github.com/"):
        normalized = "https://github.com/" + normalized.removeprefix(
            "ssh://git@github.com/"
        )
    return normalized.removesuffix(".git").lower()


def _articraft_checkout_remote() -> str | None:
    """Return the checkout's origin URL when it is a readable Git repository."""
    git = _command_path("git")
    if not git or not (ARTICRAFT_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [git, "remote", "get-url", "origin"],
            cwd=ARTICRAFT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or "").strip() if result.returncode == 0 else None


def _articraft_checkout_matches_repository() -> bool:
    """Return whether the checkout was cloned from the configured fork."""
    remote = _articraft_checkout_remote()
    return remote is not None and _normalized_repository_url(
        remote
    ) == _normalized_repository_url(ARTICRAFT_REPOSITORY_URL)


def _articraft_isolation_error() -> str | None:
    """Return an error when Articraft paths could expose deployment secrets."""
    checkout = ARTICRAFT_ROOT.expanduser().resolve()
    repository = EMBODICHAIN_ROOT.resolve()
    if checkout == repository or repository.is_relative_to(checkout):
        return (
            "ARTICRAFT_ROOT must be a dedicated nested or external Git checkout, "
            "not the EmbodiChain repository or one of its parents."
        )

    env_path = find_gen_sim_env_file()
    if env_path is not None and env_path.resolve().is_relative_to(checkout):
        return "ARTICRAFT_ROOT must not contain the shared GenSim dotenv file."
    try:
        output_root = validate_gradio_artifact_root(ARTICRAFT_OUTPUT_ROOT)
    except ValueError as exc:
        return str(exc)
    if env_path is not None and env_path.resolve().is_relative_to(output_root):
        return "ARTICRAFT_OUTPUT_ROOT must not contain the shared GenSim dotenv file."
    return None


def _prepare_articraft_checkout() -> tuple[bool, str]:
    """Clone the configured fork when absent, without overwriting a checkout."""
    if isolation_error := _articraft_isolation_error():
        return False, isolation_error
    if ARTICRAFT_ROOT.exists():
        if (ARTICRAFT_ROOT / ".git").exists() and (
            ARTICRAFT_ROOT / "pyproject.toml"
        ).is_file():
            remote = _articraft_checkout_remote()
            if _articraft_checkout_matches_repository():
                return True, f".articraft checkout: {ARTICRAFT_ROOT} ({remote})"
            return (
                False,
                f"{ARTICRAFT_ROOT} was not cloned from {ARTICRAFT_REPOSITORY_URL}; "
                f"its origin is {remote or '(unavailable)'}. It was left untouched.",
            )
        return (
            False,
            f"{ARTICRAFT_ROOT} exists but is not an Articraft Git checkout; "
            "it was left untouched.",
        )

    git = _command_path("git")
    if not git:
        return False, "Git is not on PATH, so .articraft cannot be cloned."
    try:
        ARTICRAFT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        clone = subprocess.run(
            [
                git,
                "clone",
                "--branch",
                "main",
                "--single-branch",
                ARTICRAFT_REPOSITORY_URL,
                str(ARTICRAFT_ROOT),
            ],
            cwd=ARTICRAFT_ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Unable to clone Articraft: {exc}"
    if clone.returncode:
        return False, f"Articraft clone failed: {_short_output(clone, limit=3_000)}"
    return True, f"Cloned .articraft from {ARTICRAFT_REPOSITORY_URL}"


def _ensure_articraft_environment() -> tuple[bool, str]:
    """Verify the new Articraft CLI inside the existing Conda environment."""
    if not _conda_path():
        return False, "Conda is not on PATH. Set CONDA_EXE to the conda executable."
    prefix = _articraft_conda_prefix()
    if prefix is None:
        return False, f"Conda environment {ARTICRAFT_CONDA_ENV!r} was not found."
    with _articraft_environment_lock:
        try:
            result = subprocess.run(
                _articraft_cli_command("--help"),
                cwd=ARTICRAFT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_ARTICRAFT_ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
                check=False,
                env=build_codex_env(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Unable to verify the Articraft Conda environment: {exc}"
    if result.returncode:
        return False, (
            "Unable to run the new Articraft CLI in its Conda environment: "
            f"{_short_output(result, limit=3_000)}"
        )
    return True, f"Articraft Conda environment: {ARTICRAFT_CONDA_ENV} ({prefix})"


def _run_check(
    command: list[str], *, timeout: int = 45, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a bounded environment check from the Articraft checkout."""
    return subprocess.run(
        command,
        cwd=ARTICRAFT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _check_requirements() -> tuple[list[str], list[str], str | None]:
    """Return Articraft diagnostics without creating an asset."""
    errors: list[str] = []
    details: list[str] = []
    isolation_error = _articraft_isolation_error()
    if isolation_error:
        errors.append(isolation_error)
    elif not (
        ARTICRAFT_ROOT.is_dir()
        and (ARTICRAFT_ROOT / ".git").exists()
        and (ARTICRAFT_ROOT / "pyproject.toml").is_file()
    ):
        errors.append(f".articraft checkout is not ready: {ARTICRAFT_ROOT}")
    elif not _articraft_checkout_matches_repository():
        errors.append(
            ".articraft checkout origin does not match "
            f"{ARTICRAFT_REPOSITORY_URL}: "
            f"{_articraft_checkout_remote() or '(unavailable)'}"
        )
    if not _conda_path():
        errors.append("Conda is not on PATH. Set CONDA_EXE to the conda executable.")
    elif _articraft_conda_prefix() is None:
        errors.append(f"Conda environment {ARTICRAFT_CONDA_ENV!r} was not found.")

    codex = _command_path("codex")
    if not codex:
        errors.append("Codex CLI is not on PATH. Install it or set CODEX_EXE.")
    elif not errors:
        try:
            result = _run_check(
                _conda_run_command(codex, "--version"), env=build_codex_env()
            )
            if result.returncode:
                errors.append(f"Codex CLI check failed: {_short_output(result)}")
            else:
                details.append(f"Codex: {_short_output(result, limit=120)}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Codex CLI check failed: {exc}")

    return details, errors, codex


def configure_articraft_environment() -> str:
    """Clone the fork and verify it inside the existing Conda environment."""
    checkout_ready, checkout_message = _prepare_articraft_checkout()
    if not checkout_ready:
        return "**Articulation is not ready.**\n\n- " + checkout_message
    environment_ready, environment_message = _ensure_articraft_environment()
    if not environment_ready:
        return "**Articulation is not ready.**\n\n- " + environment_message
    try:
        (ARTICRAFT_OUTPUT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"**Unable to prepare the Articulation output folder:** `{exc}`"

    details, errors, _ = _check_requirements()
    if errors:
        return "**Articulation is not ready.**\n\n" + "\n".join(
            f"- {error}" for error in errors
        )
    details[0:0] = [checkout_message, environment_message]
    details.extend(
        (
            f"Shared output: `{ARTICRAFT_OUTPUT_ROOT}`",
            "Generation activates the Articraft Conda environment and uses the "
            "fork's native USDZ output.",
        )
    )
    return "**Articulation is ready.**\n\n" + "\n".join(
        f"- {detail}" for detail in details
    )


def _validated_reference_image(value: Any) -> Path | None:
    """Return one supported reference image supplied by Gradio."""
    if not value:
        return None
    image_path = Path(str(value)).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(
            "The reference image is no longer available; please upload it again."
        )
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Reference image must be PNG, JPG, JPEG, or WEBP.")
    return image_path


def _articraft_generation_command(
    prompt: str, reference_image: Path | None
) -> list[str]:
    """Build the Articraft CLI command backed by the Codex provider."""
    command = [
        "generate",
        "--provider",
        "codex-cli",
        "--output-dir",
        str((ARTICRAFT_OUTPUT_ROOT / "runs").resolve()),
        "--no-tui",
    ]
    if reference_image is not None:
        command.extend(["--image", str(reference_image)])
    command.append(prompt)
    return _articraft_cli_command(*command)


def _generation_run_path(log_lines: list[str]) -> Path:
    """Read the run directory printed by ``articraft generate --no-tui``."""
    for line in reversed(log_lines):
        if line.startswith("run:"):
            raw_path = line.partition(":")[2].strip()
            if raw_path:
                path = Path(raw_path).expanduser()
                return (
                    path.resolve()
                    if path.is_absolute()
                    else (ARTICRAFT_ROOT / path).resolve()
                )
    raise FileNotFoundError("Articraft completed without reporting its run directory.")


def _generation_result(log_lines: list[str]) -> tuple[Path, Path, Path]:
    """Return the run, native USDZ, and EmbodiChain-compatible USDC."""
    run_dir = _generation_run_path(log_lines)
    output_runs = (ARTICRAFT_OUTPUT_ROOT / "runs").resolve()
    try:
        run_dir.relative_to(output_runs)
    except ValueError as exc:
        raise ValueError(
            f"Articraft reported a run outside {output_runs}: {run_dir}"
        ) from exc

    record_path = run_dir / "record.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Articraft run record is unavailable: {record_path}") from exc
    if not isinstance(record, dict) or record.get("status") != "success":
        raise ValueError(f"Articraft run did not finish successfully: {record_path}")

    result_value = str(record.get("result") or "").strip()
    if not result_value:
        raise ValueError("Successful Articraft run has no result artifact.")
    native_artifact = (run_dir / result_value).resolve()
    try:
        native_artifact.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("Articraft result points outside its run directory.") from exc
    if not native_artifact.is_file() or native_artifact.suffix.lower() != ".usdz":
        raise FileNotFoundError(
            f"Articraft USDZ result is unavailable: {native_artifact}"
        )

    manifest_path = run_dir / "result" / "model.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        compatible_value = str(files["embodichain_usdc"]).strip()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Articraft EmbodiChain artifact is unavailable: {manifest_path}"
        ) from exc
    compatible_artifact = (manifest_path.parent / compatible_value).resolve()
    try:
        compatible_artifact.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            "Articraft EmbodiChain result points outside its run directory."
        ) from exc
    if (
        not compatible_artifact.is_file()
        or compatible_artifact.suffix.lower() != ".usdc"
    ):
        raise FileNotFoundError(
            "Articraft EmbodiChain USDC result is unavailable: "
            f"{compatible_artifact}"
        )
    return run_dir, native_artifact, compatible_artifact


def _articraft_result_preview(run_dir: Path, artifact: Path) -> str:
    """Render a compact summary for the EmbodiChain-compatible artifact."""
    return (
        "<div style='padding:1rem; border:1px solid #d1d5db; border-radius:8px;'>"
        "<strong>EmbodiChain-compatible USDC generated successfully.</strong><br>"
        f"Artifact: <code>{html.escape(artifact.name)}</code><br>"
        f"Run: <code>{html.escape(run_dir.name)}</code>"
        "</div>"
    )


def _articraft_viewer_port(line: str) -> int | None:
    """Return the local port announced by Articraft's native viewer."""
    prefix = "Viewer URL: "
    if not line.startswith(prefix):
        return None
    try:
        parsed = urlsplit(line.removeprefix(prefix).strip())
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        return None
    return port


def _articraft_viewer_iframe(run_id: str, port: int) -> str:
    """Embed Articraft's native viewer through the Gradio page hostname."""
    srcdoc = (
        "<script>window.location.replace(window.top.location.protocol + '//' + "
        f"window.top.location.hostname + ':{port}');</script>"
    )
    escaped_run_id = html.escape(run_id)
    return (
        "<div style='margin-top:0.5rem'><strong>Articraft preview: "
        f"{escaped_run_id}</strong>"
        f"<iframe title='Articraft preview {escaped_run_id}' "
        f'srcdoc="{html.escape(srcdoc, quote=True)}" '
        "style='width:100%; height:680px; border:1px solid #d1d5db; "
        "border-radius:8px; margin-top:0.5rem;'></iframe></div>"
    )


def _start_articraft_viewer(session_id: str, run_dir: Path) -> str:
    """Start Articraft's native USDZ viewer for one Gradio session."""
    token = _articraft_viewers.begin(session_id)
    environment = build_codex_env()
    environment.update({"BROWSER": "true", "PYTHONUNBUFFERED": "1"})
    process = register_managed_process(
        subprocess.Popen(
            _articraft_cli_command("view", str(run_dir)),
            cwd=ARTICRAFT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=environment,
        )
    )
    if not _articraft_viewers.attach(session_id, token, process):
        terminate_process_group(process)
        raise RuntimeError("Articraft viewer request was superseded.")

    if process.stdout is not None:
        for line in process.stdout:
            if port := _articraft_viewer_port(line):
                return _articraft_viewer_iframe(run_dir.name, port)

    terminate_process_group(process)
    _articraft_viewers.finish(session_id, token, process)
    raise RuntimeError("Articraft viewer did not start.")


def _articulation_server_client() -> ArticulationServerClient:
    """Build a client from the shared Gradio deployment settings."""
    return ArticulationServerClient(
        ARTICULATION_SERVER_BASE_URL,
        timeout_seconds=_server_positive_float(
            "ARTICULATION_SERVER_TIMEOUT_S", ARTICULATION_SERVER_TIMEOUT_S
        ),
    )


def _server_positive_float(name: str, value: object) -> float:
    """Parse one optional remote setting only when that backend is used."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _server_polling_settings() -> tuple[float, float]:
    """Return validated task timeout and polling interval values."""
    return (
        _server_positive_float(
            "ARTICULATION_SERVER_TASK_TIMEOUT_S",
            ARTICULATION_SERVER_TASK_TIMEOUT_S,
        ),
        _server_positive_float(
            "ARTICULATION_SERVER_POLL_INTERVAL_S",
            ARTICULATION_SERVER_POLL_INTERVAL_S,
        ),
    )


def _server_output_root() -> Path:
    """Validate remote task settings and return their local artifact root."""
    _server_polling_settings()
    return validate_gradio_artifact_root(ARTICRAFT_OUTPUT_ROOT) / "server"


def _configure_selected_articulation_provider(provider: str) -> str:
    """Check the selected remote or local generation backend."""
    if provider == _LOCAL_CODEX_PROVIDER:
        return configure_articraft_environment()
    if provider != _REMOTE_SERVER_PROVIDER:
        return f"**Unknown Articulation backend:** `{provider}`"

    try:
        output_root = _server_output_root()
        health = _articulation_server_client().health()
        output_root.mkdir(parents=True, exist_ok=True)
    except (ArticulationServerError, OSError, ValueError) as exc:
        detail = redact_sensitive_text(str(exc))
        return (
            "**Remote Articulation server is not ready.**\n\n"
            f"- {detail}\n"
            "- Check `ARTICULATION_SERVER_BASE_URL` and server connectivity."
        )
    if health.get("status") != "ready":
        state = redact_sensitive_text(str(health.get("status") or "unknown"))
        return (
            "**Remote Articulation server is not ready.**\n\n"
            f"- Health status: `{state}`"
        )
    return (
        "**Remote Articulation server is ready.**\n\n"
        f"- Endpoint: `{ARTICULATION_SERVER_BASE_URL}`\n"
        f"- Local output: `{output_root}`"
    )


def _server_task_line(task: dict[str, Any]) -> str:
    """Format one bounded status line for the remote generation log."""
    status = str(task.get("status") or "unknown")
    stage = str(task.get("stage") or "").strip()
    return f"status: {status}" + (f"; stage: {stage}" if stage else "")


def _append_server_log(log_lines: list[str], line: str) -> None:
    """Append one remote log line while bounding per-session memory."""
    log_lines.append(line)
    if len(log_lines) > _SERVER_LOG_LIMIT:
        del log_lines[:-_SERVER_LOG_LIMIT]


def _server_task_error(task: dict[str, Any]) -> str:
    """Return a redacted remote failure description."""
    detail = task.get("error")
    if not detail and isinstance(task.get("details"), dict):
        detail = task["details"].get("message")
    return redact_sensitive_text(str(detail or "The server reported no error detail."))


def _generate_server_articulation_asset(
    prompt_value: str,
    image_value: Any,
    request: gr.Request,
) -> Iterator[tuple[Any, ...]]:
    """Generate an articulation through the remote HTTP service."""
    session_id = get_request_session_id(request)
    token = _articraft_runs.begin(session_id)
    replacement_error = _cancel_server_task(session_id)
    if replacement_error:
        yield (
            None,
            "",
            "**The previous remote Articulation task could not be cancelled.**\n\n"
            f"- {replacement_error}\n"
            "- No replacement task was submitted; retry Reset before continuing.",
            "",
            "",
        )
        return
    prompt = (prompt_value or "").strip()
    if not prompt:
        yield None, "", "**Input error:** enter an articulated-object description.", "", ""
        return

    try:
        output_root = _server_output_root()
        output_root.mkdir(parents=True, exist_ok=True)
        reference_image = _validated_reference_image(image_value)
        client = _articulation_server_client()
        submitted = client.submit(prompt, image=reference_image)
        request_id_value = submitted.get("request_id")
        if not isinstance(request_id_value, str) or not request_id_value.strip():
            raise ArticulationServerError("server response has no request_id")
        request_id = request_id_value.strip()
    except (ArticulationServerError, OSError, ValueError) as exc:
        detail = redact_sensitive_text(str(exc))
        yield (
            None,
            "",
            "**Remote Articulation request could not start.**\n\n"
            f"- {detail}\n"
            "- The request was not retried with Local Codex.",
            "",
            "",
        )
        return

    if not _register_server_task(session_id, token, client, request_id):
        return
    log_lines = [
        f"Server: {ARTICULATION_SERVER_BASE_URL}",
        f"Request: {request_id}",
    ]
    yield (
        None,
        "",
        "**Remote server accepted the Articulation request.**",
        "\n".join(log_lines),
        "",
    )

    task_timeout_seconds, poll_interval_seconds = _server_polling_settings()
    deadline = time.monotonic() + task_timeout_seconds
    previous_line = ""
    while True:
        if not _articraft_runs.is_active(session_id, token):
            return
        if time.monotonic() >= deadline:
            cancellation_error = _cancel_server_task(session_id)
            if cancellation_error:
                _append_server_log(
                    log_lines, f"Cancellation warning: {cancellation_error}"
                )
            yield (
                None,
                "",
                "**Remote Articulation request timed out and cancellation was requested.**\n\n"
                f"- Request: `{request_id}`\n"
                "- The request was not retried with Local Codex.",
                "\n".join(log_lines[-300:]),
                "",
            )
            return
        try:
            task = client.status(request_id)
        except (ArticulationServerError, OSError, ValueError) as exc:
            cancellation_error = _cancel_server_task(session_id)
            detail = redact_sensitive_text(str(exc))
            if cancellation_error:
                _append_server_log(
                    log_lines, f"Cancellation warning: {cancellation_error}"
                )
            yield (
                None,
                "",
                "**Remote Articulation status check failed.**\n\n"
                f"- {detail}\n"
                "- The request was not retried with Local Codex.",
                "\n".join(log_lines[-300:]),
                "",
            )
            return

        status_line = _server_task_line(task)
        if status_line != previous_line:
            _append_server_log(log_lines, status_line)
            previous_line = status_line
        status = task.get("status")
        if status in _SERVER_TERMINAL_STATUSES:
            _finish_server_task(session_id, token, request_id)
            break
        yield (
            None,
            "",
            f"**Remote server is generating the Articulation.**\n\n- {status_line}",
            "\n".join(log_lines[-300:]),
            "",
        )
        remaining_seconds = max(0.0, deadline - time.monotonic())
        time.sleep(min(poll_interval_seconds, remaining_seconds))

    if not _articraft_runs.is_active(session_id, token):
        return
    if status != "succeeded":
        detail = _server_task_error(task)
        yield (
            None,
            "",
            f"**Remote Articulation request {status}.**\n\n- {detail}\n"
            "- The request was not retried with Local Codex.",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    output_dir = output_root / request_id
    try:
        artifact = client.download(request_id, "usdc", output_dir / "model.usdc")
    except (ArticulationServerError, OSError, ValueError) as exc:
        detail = redact_sensitive_text(str(exc))
        yield (
            None,
            output_dir.as_posix(),
            "**Remote Articulation completed, but its USDC could not be downloaded.**\n\n"
            f"- {detail}",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    if not _articraft_runs.is_active(session_id, token):
        return
    yield (
        artifact.as_posix(),
        output_dir.as_posix(),
        "**Remote Articulation generation completed.**\n\n"
        f"- Request: `{request_id}`\n"
        f"- EmbodiChain USDC: `{artifact}`",
        "\n".join(log_lines[-300:]),
        _articraft_result_preview(output_dir, artifact),
    )


def generate_articraft_asset(
    prompt_value: str,
    image_value: Any,
    request: gr.Request,
) -> Iterator[tuple[Any, ...]]:
    """Generate native USDZ and EmbodiChain-compatible USDC artifacts."""
    session_id = get_request_session_id(request)
    token = _articraft_runs.begin(session_id)
    cancellation_error = _cancel_server_task(session_id)
    if cancellation_error:
        yield (
            None,
            "",
            "**The previous remote Articulation task could not be cancelled.**\n\n"
            f"- {cancellation_error}\n"
            "- Local Codex was not started; retry Reset before continuing.",
            "",
            "",
        )
        return
    prompt = (prompt_value or "").strip()
    if not prompt:
        yield None, "", "**Input error:** enter an articulated-object description.", "", ""
        return

    details, errors, codex = _check_requirements()
    if errors or not codex:
        message = "\n".join(f"- {error}" for error in errors)
        yield None, "", f"**Articulation is not ready.**\n\n{message}", "", ""
        return

    try:
        reference_image = _validated_reference_image(image_value)
        command = _articraft_generation_command(prompt, reference_image)
    except (OSError, ValueError, RuntimeError) as exc:
        yield None, "", f"**Input error:** {exc}", "", ""
        return

    log_lines = [*details, f"Shared output: {ARTICRAFT_OUTPUT_ROOT}"]
    log_lines.append("$ " + " ".join([*command[:-1], "<prompt>"]))
    yield (
        None,
        "",
        "**Articraft and Codex are generating the articulated asset…**",
        "\n".join(log_lines),
        "",
    )

    environment = build_codex_env()
    environment.update(
        {
            "ARTICRAFT_CODEX_CLI_BIN": codex,
            "PYTHONUNBUFFERED": "1",
        }
    )
    try:
        process = register_managed_process(
            subprocess.Popen(
                command,
                cwd=ARTICRAFT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=environment,
            )
        )
        if not _articraft_runs.attach(session_id, token, process):
            terminate_process_group(process)
            return
    except OSError as exc:
        if _articraft_runs.is_active(session_id, token):
            yield None, "", f"**Articraft could not start:** {exc}", "\n".join(
                log_lines
            ), ""
        return

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue),
        kwargs={"redact_sensitive": True},
        daemon=True,
    )
    reader.start()
    while process.poll() is None:
        if not _articraft_runs.is_active(session_id, token, process):
            return
        try:
            while True:
                log_lines.append(output_queue.get_nowait())
        except queue.Empty:
            pass
        yield (
            None,
            "",
            "**Articraft and Codex are generating the articulated asset…**",
            "\n".join(log_lines[-300:]),
            "",
        )
        time.sleep(0.75)

    try:
        reader.join(timeout=2)
        try:
            while True:
                log_lines.append(output_queue.get_nowait())
        except queue.Empty:
            pass
    finally:
        _articraft_runs.finish(session_id, token, process)

    if not _articraft_runs.is_active(session_id, token):
        return
    if process.returncode:
        yield (
            None,
            "",
            f"**Articraft generation failed** (exit code {process.returncode}).",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    try:
        run_dir, native_artifact, artifact = _generation_result(log_lines)
    except (OSError, ValueError) as exc:
        yield (
            None,
            "",
            f"**Articraft finished, but its result could not be loaded:** {exc}",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    status = (
        "**Articraft generation completed.**\n\n"
        f"- Run: `{run_dir}`\n"
        f"- Native USDZ: `{native_artifact}`\n"
        f"- EmbodiChain USDC: `{artifact}`"
    )
    try:
        preview_html = _start_articraft_viewer(session_id, run_dir)
        status += "\n- Interactive Articraft preview: ready"
    except (OSError, RuntimeError) as exc:
        preview_html = _articraft_result_preview(run_dir, artifact)
        status += f"\n- Interactive preview could not start: `{exc}`"
    if not _articraft_runs.is_active(session_id, token):
        return
    yield (
        artifact.as_posix(),
        run_dir.as_posix(),
        status,
        "\n".join(log_lines[-300:]),
        preview_html,
    )


def _generate_selected_articulation_asset(
    provider: str,
    prompt_value: str,
    image_value: Any,
    request: gr.Request,
) -> Iterator[tuple[Any, ...]]:
    """Route one request to the explicitly selected generation backend."""
    if provider == _REMOTE_SERVER_PROVIDER:
        yield from _generate_server_articulation_asset(
            prompt_value, image_value, request
        )
        return
    if provider == _LOCAL_CODEX_PROVIDER:
        yield from generate_articraft_asset(prompt_value, image_value, request)
        return
    yield None, "", f"**Unknown Articulation backend:** `{provider}`", "", ""


def build_articraft_panel() -> None:
    """Render the Articraft tab inside the Asset engine."""
    gr.Markdown(
        "### Articulation\n"
        "Generate an EmbodiChain-compatible articulated USDC from text and an optional "
        "reference image; only submit trusted requests."
    )
    provider = gr.Radio(
        choices=[_REMOTE_SERVER_PROVIDER, _LOCAL_CODEX_PROVIDER],
        value=_DEFAULT_ARTICULATION_PROVIDER,
        label="Generation backend",
    )
    with gr.Row():
        configure_button = gr.Button("Check Articulation backend")
        generate_button = gr.Button("Generate articulation", variant="primary")
        reset_button = gr.Button("Reset Articulation", variant="stop")
    environment_status = gr.Markdown("**Environment:** not checked.")
    with gr.Row():
        prompt = gr.Textbox(
            label="Articulated object description",
            lines=5,
            placeholder=(
                "e.g. A countertop toaster oven with a hinged door and rotating "
                "temperature knob."
            ),
        )
        image = gr.Image(
            label="Optional reference image",
            type="filepath",
            image_mode="RGB",
            sources=["upload"],
        )
    with gr.Row():
        output_file = gr.File(
            label="EmbodiChain Articulation result (.usdc)", interactive=False
        )
        record_folder = gr.Textbox(label="Articulation run folder", interactive=False)
    articulation_preview = gr.HTML(_ARTICRAFT_IDLE_PREVIEW)
    generation_status = gr.Markdown("**Status:** waiting for a description.")
    generation_log = gr.Textbox(
        label="Articulation generation log", lines=14, interactive=False
    )

    configure_button.click(
        _configure_selected_articulation_provider,
        inputs=[provider],
        outputs=[environment_status],
        queue=False,
    )
    generate_button.click(
        _generate_selected_articulation_asset,
        inputs=[provider, prompt, image],
        outputs=[
            output_file,
            record_folder,
            generation_status,
            generation_log,
            articulation_preview,
        ],
    )
    reset_button.click(
        reset_articraft_asset,
        outputs=[
            environment_status,
            prompt,
            image,
            output_file,
            record_folder,
            generation_status,
            generation_log,
            articulation_preview,
        ],
        queue=False,
    )
