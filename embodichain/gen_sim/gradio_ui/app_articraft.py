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

"""Codex-backed Articraft generation for the Asset engine.

The integration uses Articraft's external-agent workflow: Articraft owns
record creation and validation while Codex authors the generated model. All
mutable run data is kept under ``ARTICRAFT_OUTPUT_ROOT``.
"""

from __future__ import annotations

import atexit
import os
import queue
import json
import shutil
import html
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from app_config import (
    ARTICRAFT_CONDA_ENV,
    ARTICRAFT_OUTPUT_ROOT,
    ARTICRAFT_REPOSITORY_URL,
    ARTICRAFT_ROOT,
    ARTICRAFT_VISER_PORT,
)
from app_processes import read_process_output, start_pipeline, terminate_process_group

__all__ = [
    "build_articraft_panel",
    "configure_articraft_environment",
    "generate_articraft_asset",
    "stop_articraft_viser_preview",
]

_VISER_START_TIMEOUT_SECONDS = 15.0
_VISER_STOP_TIMEOUT_SECONDS = 5.0
_ARTICRAFT_PYTHON_VERSION = "3.12"
_CONDA_ENVIRONMENT_SETUP_TIMEOUT_SECONDS = 1_200
_articraft_environment_lock = threading.Lock()


def _command_path(name: str) -> str | None:
    """Resolve commands even when Gradio did not inherit an interactive PATH."""
    configured = os.environ.get(f"{name.upper()}_EXE")
    return configured or shutil.which(name)


def _conda_path() -> str | None:
    configured = os.environ.get("CONDA_EXE")
    if configured and Path(configured).is_file():
        return configured
    return _command_path("conda")


def _conda_command(*args: str) -> list[str]:
    conda = _conda_path()
    if not conda:
        raise RuntimeError("Conda was not found. Set CONDA_EXE before starting Gradio.")
    return [conda, "run", "--no-capture-output", "-n", ARTICRAFT_CONDA_ENV, *args]


def _articraft_cli_command(*args: str) -> list[str]:
    """Run the CLI from the checked-out source without installing it with pip."""
    return _conda_command("python", "-m", "cli.main", *args)


def _articraft_conda_environment_exists() -> bool:
    """Check only for the named Conda environment, not package installation."""
    conda = _conda_path()
    if not conda:
        return False
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            return False
        environments = json.loads(result.stdout or "{}").get("envs", [])
        return any(Path(path).name == ARTICRAFT_CONDA_ENV for path in environments)
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _ensure_articraft_conda_environment() -> tuple[bool, str]:
    """Create and populate the Articraft Conda environment when it is absent.

    Articraft currently supports Python 3.11 and 3.12, while the Gradio process
    can use a different interpreter. The setup therefore creates an isolated
    Python 3.12 environment and installs the checked-out project's runtime
    dependencies into it.

    Returns:
        Whether the environment is ready and a status message suitable for the
        Gradio configuration panel.
    """
    conda = _conda_path()
    if not conda:
        return False, "Conda is not on PATH. Set CONDA_EXE to the conda executable."

    with _articraft_environment_lock:
        if _articraft_conda_environment_exists():
            return True, f"Conda environment: {ARTICRAFT_CONDA_ENV} (already exists)"

        create_command = [
            conda,
            "create",
            "--yes",
            "--name",
            ARTICRAFT_CONDA_ENV,
            f"python={_ARTICRAFT_PYTHON_VERSION}",
            "pip",
        ]
        try:
            created = subprocess.run(
                create_command,
                cwd=ARTICRAFT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=_CONDA_ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Unable to create Conda environment: {exc}"
        if created.returncode and not _articraft_conda_environment_exists():
            return False, (
                "Conda environment creation failed: "
                f"{_short_output(created, limit=3000)}"
            )

        for install_args, description in (
            (
                ["python", "-m", "pip", "install", "--upgrade", "pip"],
                "upgrade pip",
            ),
            (["python", "-m", "pip", "install", "."], "install Articraft dependencies"),
        ):
            install_command = [
                conda,
                "run",
                "--no-capture-output",
                "--name",
                ARTICRAFT_CONDA_ENV,
                *install_args,
            ]
            try:
                installed = subprocess.run(
                    install_command,
                    cwd=ARTICRAFT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=_CONDA_ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return False, f"Unable to {description}: {exc}"
            if installed.returncode:
                return (
                    False,
                    f"Unable to {description}: {_short_output(installed, limit=3000)}",
                )

    return True, (
        f"Created Conda environment: {ARTICRAFT_CONDA_ENV} "
        f"(Python {_ARTICRAFT_PYTHON_VERSION})"
    )


def _run_check(
    command: list[str], *, timeout: int = 45
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ARTICRAFT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def _short_output(
    result: subprocess.CompletedProcess[str], *, limit: int = 1800
) -> str:
    output = (result.stdout or "").strip()
    return output[-limit:] if len(output) > limit else (output or "(no output)")


def _check_requirements() -> tuple[list[str], list[str], str | None]:
    """Return diagnostics and the Codex executable, without creating an asset."""
    errors: list[str] = []
    details: list[str] = []
    if not (
        ARTICRAFT_ROOT.is_dir()
        and (ARTICRAFT_ROOT / ".git").exists()
        and (ARTICRAFT_ROOT / "pyproject.toml").is_file()
    ):
        errors.append(f".articraft checkout is not ready: {ARTICRAFT_ROOT}")
    if not _conda_path():
        errors.append("Conda is not on PATH. Set CONDA_EXE to the conda executable.")
    elif not _articraft_conda_environment_exists():
        errors.append(f"Conda environment not found: {ARTICRAFT_CONDA_ENV}")
    else:
        details.append(f"Conda environment: {ARTICRAFT_CONDA_ENV}")

    codex = _command_path("codex")
    if not codex:
        errors.append("Codex CLI is not on PATH. Install it or set CODEX_EXE.")
    elif not errors:
        try:
            result = _run_check([codex, "--version"])
            if result.returncode:
                errors.append(f"Codex CLI check failed: {_short_output(result)}")
            else:
                details.append(f"Codex: {_short_output(result, limit=120)}")
        except Exception as exc:
            errors.append(f"Codex CLI check failed: {exc}")

    if not errors:
        details.append(f".articraft checkout: {ARTICRAFT_ROOT}")
    return details, errors, codex


def _prepare_articraft_checkout() -> tuple[bool, str]:
    """Clone the configured checkout when absent, without overwriting a directory."""
    if ARTICRAFT_ROOT.exists():
        if (ARTICRAFT_ROOT / ".git").exists() and (
            ARTICRAFT_ROOT / "pyproject.toml"
        ).is_file():
            return True, f".articraft checkout: {ARTICRAFT_ROOT}"
        return (
            False,
            f"{ARTICRAFT_ROOT} exists but is not an Articraft Git checkout; it was left untouched.",
        )

    git = _command_path("git")
    if not git:
        return False, "Git is not on PATH, so .articraft cannot be cloned."
    try:
        ARTICRAFT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        clone = subprocess.run(
            [git, "clone", ARTICRAFT_REPOSITORY_URL, str(ARTICRAFT_ROOT)],
            cwd=ARTICRAFT_ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        return False, f"Unable to clone Articraft: {exc}"
    if clone.returncode:
        return False, f"Articraft clone failed: {_short_output(clone, limit=3000)}"
    return True, f"Cloned .articraft from {ARTICRAFT_REPOSITORY_URL}"


def configure_articraft_environment() -> str:
    """Clone the checkout, prepare its Conda environment, and verify Codex."""
    checkout_ready, checkout_message = _prepare_articraft_checkout()
    if not checkout_ready:
        return "**Articulation is not ready.**\n\n- " + checkout_message
    environment_ready, environment_message = _ensure_articraft_conda_environment()
    if not environment_ready:
        return "**Articulation is not ready.**\n\n- " + environment_message
    try:
        for directory in (
            ARTICRAFT_OUTPUT_ROOT,
            ARTICRAFT_OUTPUT_ROOT / "runs",
            ARTICRAFT_OUTPUT_ROOT / "exports",
        ):
            directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return f"**Unable to prepare the shared Articulation output folder:** `{exc}`"
    details, errors, _ = _check_requirements()
    if errors:
        return "**Articulation is not ready.**\n\n" + "\n".join(
            f"- {error}" for error in errors
        )
    details.insert(0, checkout_message)
    details.insert(1, environment_message)
    details.extend(
        (
            f"Shared output: `{ARTICRAFT_OUTPUT_ROOT}`",
            "Generation runs the `.articraft` checkout directly with `conda run`; no `pip install -e .` is required.",
        )
    )
    return "**Articulation is ready.**\n\n" + "\n".join(
        f"- {detail}" for detail in details
    )


def _record_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Articraft validates external IDs against the required ``rec_`` prefix.
    return f"rec_ui_articraft_{timestamp}_{uuid.uuid4().hex[:8]}"


def _copy_reference_image(value: Any, run_root: Path) -> Path | None:
    if not value:
        return None
    source = Path(str(value))
    if not source.is_file():
        raise ValueError(
            "The reference image is no longer available; please upload it again."
        )
    suffix = source.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Reference image must be PNG, JPG, JPEG, or WEBP.")
    target = run_root / f"reference{suffix}"
    shutil.copy2(source, target)
    return target


def _active_model_path(record_dir: Path) -> Path:
    candidates = sorted(record_dir.glob("revisions/*/model.py"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one active model.py in {record_dir}, found {len(candidates)}."
        )
    return candidates[0]


def _make_result_bundle(record_id: str) -> tuple[Path, Path]:
    materialized = (
        ARTICRAFT_OUTPUT_ROOT / "data" / "cache" / "record_materialization" / record_id
    )
    if not (materialized / "model.urdf").is_file():
        raise FileNotFoundError(
            "Articraft completed without a compiled model.urdf output."
        )
    exports_root = ARTICRAFT_OUTPUT_ROOT / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    archive = Path(
        shutil.make_archive(
            (exports_root / record_id).as_posix(),
            "zip",
            root_dir=materialized,
        )
    )
    return materialized, archive


def _articraft_viser_iframe(record_id: str) -> str:
    """Embed the Articulation Viser service through the Gradio page hostname."""
    srcdoc = (
        "<script>window.location.replace(window.top.location.protocol + '//' + "
        f"window.top.location.hostname + ':{ARTICRAFT_VISER_PORT}');</script>"
    )
    escaped_record_id = html.escape(record_id)
    return (
        "<div style='margin-top:0.5rem'><strong>Viser articulation preview: "
        f"{escaped_record_id}</strong>"
        f"<iframe title='Viser articulation preview {escaped_record_id}' "
        f'srcdoc="{html.escape(srcdoc, quote=True)}" '
        "style='width:100%; height:680px; border:1px solid #d1d5db; border-radius:8px; margin-top:0.5rem;'></iframe>"
        "</div>"
    )


class _ArticraftViserPreview:
    """Own the single Articraft Viser process and its dedicated TCP port."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def start(self, urdf_path: Path, record_id: str) -> str:
        """Replace the active preview with a verified preview of one URDF."""
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Compiled URDF is missing: {urdf_path}")

        with self._lock:
            self._stop_managed_process()
            self._clear_stale_listener()
            process = start_pipeline(self._command(urdf_path))
            if not self._wait_until_owned(process):
                terminate_process_group(process)
                raise RuntimeError("New Articraft Viser preview did not bind its port.")
            self._process = process
        return _articraft_viser_iframe(record_id)

    def stop(self) -> None:
        """Stop the preview process, if this panel started one."""
        with self._lock:
            self._stop_managed_process()

    def _command(self, urdf_path: Path) -> list[str]:
        return [
            sys.executable,
            str(Path(__file__).with_name("app_media.py")),
            "--asset_path",
            str(urdf_path),
            "--asset_type",
            "articulation",
            "--headless",
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            str(self._port),
        ]

    def _stop_managed_process(self) -> None:
        if self._process is not None:
            terminate_process_group(self._process)
            self._process = None

    def _clear_stale_listener(self) -> None:
        if self._port_is_available():
            return
        listener_pids = self._listener_pids()
        if listener_pids is None:
            raise RuntimeError(
                "Cannot identify the process using the Articraft Viser port."
            )
        if not listener_pids:
            raise RuntimeError(
                f"Port {self._port} is unavailable without a visible listener."
            )

        self._signal_listeners(listener_pids, signal.SIGTERM, "stop")
        if self._wait_for_port_release():
            return

        remaining_pids = self._listener_pids()
        if remaining_pids is None:
            raise RuntimeError(
                f"Cannot identify the stale Viser service on port {self._port}."
            )
        self._signal_listeners(remaining_pids, signal.SIGKILL, "force-stop")
        if not self._wait_for_port_release():
            raise RuntimeError(
                f"The stale Viser service is still listening on port {self._port}."
            )

    def _signal_listeners(
        self, listener_pids: set[int], signal_value: int, action: str
    ) -> None:
        for pid in listener_pids:
            try:
                os.kill(pid, signal_value)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise RuntimeError(
                    f"Cannot {action} Viser process {pid} using port {self._port}."
                ) from exc

    def _wait_for_port_release(self) -> bool:
        deadline = time.monotonic() + _VISER_STOP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._port_is_available():
                return True
            time.sleep(0.1)
        return self._port_is_available()

    def _wait_until_owned(self, process: subprocess.Popen[str]) -> bool:
        deadline = time.monotonic() + _VISER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=0.2):
                    listener_pids = self._listener_pids()
                    if listener_pids is not None and process.pid in listener_pids:
                        return True
            except OSError:
                pass
            time.sleep(0.25)
        return False

    def _port_is_available(self) -> bool:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", self._port))
        except OSError:
            return False
        finally:
            probe.close()
        return True

    def _listener_pids(self) -> set[int] | None:
        for command in self._listener_commands():
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except OSError:
                continue
            if result.returncode in (0, 1):
                return {int(pid) for pid in result.stdout.split() if pid.isdecimal()}
        return None

    def _listener_commands(self) -> list[list[str]]:
        commands: list[list[str]] = []
        if lsof := _command_path("lsof"):
            commands.append([lsof, "-nP", f"-iTCP:{self._port}", "-sTCP:LISTEN", "-t"])
        if fuser := _command_path("fuser"):
            commands.append([fuser, "-n", "tcp", str(self._port)])
        return commands


_articraft_viser_preview = _ArticraftViserPreview(ARTICRAFT_VISER_PORT)


def stop_articraft_viser_preview() -> None:
    """Stop the Viser subprocess currently owned by the Articraft panel.

    The preview runs independently from Gradio so it can be embedded through an
    iframe.  Expose its cleanup explicitly so application shutdown can release
    the dedicated port instead of leaving an orphaned Viser server behind.
    """
    _articraft_viser_preview.stop()


atexit.register(stop_articraft_viser_preview)


def _start_articraft_viser_preview(materialized: Path, record_id: str) -> str:
    """Load the compiled URDF as an articulation and expose it through Viser."""
    return _articraft_viser_preview.start(materialized / "model.urdf", record_id)


def _external_check_is_unsupported(result: subprocess.CompletedProcess[str]) -> bool:
    """Recognize the older Articraft CLI, which has no ``external check``."""
    output = (result.stdout or "").lower()
    return "invalid choice: 'check'" in output and "external" in output


def _compile_report_failures(record_id: str) -> list[str]:
    """Read blocking QC/test signals from the older CLI's compile report."""
    report_path = (
        ARTICRAFT_OUTPUT_ROOT
        / "data"
        / "cache"
        / "record_materialization"
        / record_id
        / "compile_report.json"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"Compile report is unavailable: {report_path}"]
    bundle = report.get("signal_bundle") if isinstance(report, dict) else None
    signals = bundle.get("signals") if isinstance(bundle, dict) else None
    if not isinstance(signals, list):
        return ["Compile report contains no validation signals."]
    failures: list[str] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        if signal.get("severity") == "failure" or signal.get("blocking") is True:
            failures.append(
                str(
                    signal.get("summary")
                    or signal.get("code")
                    or "Unnamed validation failure"
                )
            )
    return failures


def _build_codex_prompt(
    *,
    prompt: str,
    record_id: str,
    record_dir: Path,
    model_path: Path,
    reference_image: Path | None,
) -> str:
    image_note = (
        f"A reference image is attached and also copied at {reference_image}. Use it as visual reference."
        if reference_image
        else "No reference image was supplied."
    )
    return f"""You are the Codex external author for one Articraft articulated 3D asset.

User request:
{prompt}

{image_note}

The Articraft source repository is {ARTICRAFT_ROOT}. The shared UI output/storage root is
{ARTICRAFT_OUTPUT_ROOT}. Articraft has already created this external workbench record:
record_id={record_id}
record_dir={record_dir}
active_model={model_path}

Codex itself is launched from the Gradio environment, not the Articraft Conda environment.
For every Articraft CLI invocation, use this command prefix:

{_conda_path()} run --no-capture-output -n {ARTICRAFT_CONDA_ENV} python -m cli.main

Follow EXTERNAL_AGENT_DATA.md exactly. Read the design and link-naming guidance it references,
then use relevant SDK docs/examples. Edit only the active model.py for this record. Do not create
record folders or metadata manually, edit unrelated records, commit/push, or promote this
workbench record to the dataset.

Create a realistic mechanically meaningful articulated object matching the request. Use semantic
parts, visible plausible joints, appropriate materials, and prompt-specific run_tests(). Iterate
until this succeeds:

{_conda_path()} run --no-capture-output -n {ARTICRAFT_CONDA_ENV} python -m cli.main external --repo-root {ARTICRAFT_OUTPUT_ROOT} check {record_id}

Then run:

{_conda_path()} run --no-capture-output -n {ARTICRAFT_CONDA_ENV} python -m cli.main external --repo-root {ARTICRAFT_OUTPUT_ROOT} finalize {record_id}

The Gradio app packages the compiled URDF and meshes after you finish. In your final response,
briefly state the articulation mechanisms and validation result."""


def generate_articraft_asset(prompt_value: str, image_value: Any):
    """Initialize a record, let Codex author it, and expose one result bundle."""
    prompt = (prompt_value or "").strip()
    if not prompt:
        yield None, "", "**Input error:** enter a description of the articulated object.", "", ""
        return

    details, errors, codex = _check_requirements()
    if errors or not codex:
        message = (
            "\n".join(f"- {error}" for error in errors) or "Codex CLI is unavailable."
        )
        yield None, "", f"**Articulation is not ready.**\n\n{message}", "", ""
        return

    record_id = _record_id()
    run_root = ARTICRAFT_OUTPUT_ROOT / "runs" / record_id
    record_dir = ARTICRAFT_OUTPUT_ROOT / "data" / "records" / record_id
    log_lines = [*details, f"Shared output: {ARTICRAFT_OUTPUT_ROOT}"]
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        reference_image = _copy_reference_image(image_value, run_root)
        init_command = _articraft_cli_command(
            "external",
            "--repo-root",
            str(ARTICRAFT_OUTPUT_ROOT),
            "init",
            "--agent",
            "codex",
            "--record-id",
            record_id,
            prompt,
        )
        log_lines.append("$ " + " ".join(init_command[:-1]) + " <prompt>")
        initialized = _run_check(init_command, timeout=90)
        log_lines.append(_short_output(initialized, limit=4000))
        if initialized.returncode:
            yield None, "", "**Articraft record initialization failed.**", "\n".join(
                log_lines
            ), ""
            return
        model_path = _active_model_path(record_dir)
    except Exception as exc:
        yield None, "", f"**Setup failed:** {exc}", "\n".join(log_lines), ""
        return

    final_message = run_root / "codex_final_message.txt"
    codex_command = [
        codex,
        "exec",
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "-C",
        str(ARTICRAFT_ROOT),
        "--add-dir",
        str(ARTICRAFT_OUTPUT_ROOT),
        "--output-last-message",
        str(final_message),
    ]
    if reference_image:
        codex_command.extend(["--image", str(reference_image)])
    codex_command.append(
        _build_codex_prompt(
            prompt=prompt,
            record_id=record_id,
            record_dir=record_dir,
            model_path=model_path,
            reference_image=reference_image,
        )
    )
    log_lines.append("$ codex exec --sandbox workspace-write …")
    yield None, record_dir.as_posix(), "**Codex is generating and validating the Articraft model…**", "\n".join(
        log_lines
    ), ""

    try:
        process = subprocess.Popen(
            codex_command,
            cwd=ARTICRAFT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=os.environ.copy(),
        )
    except Exception as exc:
        yield None, record_dir.as_posix(), f"**Codex could not start:** {exc}", "\n".join(
            log_lines
        ), ""
        return

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output, args=(process, output_queue), daemon=True
    )
    reader.start()
    while process.poll() is None:
        try:
            while True:
                log_lines.append(output_queue.get_nowait())
        except queue.Empty:
            pass
        yield None, record_dir.as_posix(), "**Codex is generating and validating the Articraft model…**", "\n".join(
            log_lines[-240:]
        ), ""
        time.sleep(0.75)
    reader.join(timeout=2)
    try:
        while True:
            log_lines.append(output_queue.get_nowait())
    except queue.Empty:
        pass

    if final_message.is_file():
        final_text = final_message.read_text(encoding="utf-8", errors="replace").strip()
        if final_text:
            log_lines.append("\nCodex final response:\n" + final_text)
    if process.returncode:
        yield None, record_dir.as_posix(), f"**Codex generation failed** (exit code {process.returncode}).", "\n".join(
            log_lines[-300:]
        ), ""
        return

    # Do not rely solely on Codex's final message: independently run the
    # external validation and finalize gates before exposing an output bundle.
    check_command = _articraft_cli_command(
        "external",
        "--repo-root",
        str(ARTICRAFT_OUTPUT_ROOT),
        "check",
        record_id,
    )
    log_lines.append("$ " + " ".join(check_command))
    yield (
        None,
        record_dir.as_posix(),
        "**Codex finished. Articraft is running the final validation gate…**",
        "\n".join(log_lines[-300:]),
        "",
    )
    try:
        checked = _run_check(check_command, timeout=300)
        log_lines.append(_short_output(checked, limit=5000))
    except Exception as exc:
        yield (
            None,
            record_dir.as_posix(),
            f"**Final Articraft validation could not run:** {exc}",
            "\n".join(log_lines[-300:]),
            "",
        )
        return
    if checked.returncode:
        if not _external_check_is_unsupported(checked):
            yield (
                None,
                record_dir.as_posix(),
                "**Articraft validation failed; no output bundle was published.**",
                "\n".join(log_lines[-300:]),
                "",
            )
            return
        # The older CLI reports external init/finalize/categories only. Its
        # equivalent strict model validation is the top-level compile command.
        compile_command = _articraft_cli_command(
            "compile",
            "--repo-root",
            str(ARTICRAFT_OUTPUT_ROOT),
            "--target",
            "full",
            "--validate",
            "--strict-geom-qc",
            record_id,
        )
        log_lines.append(
            "external check is unavailable; falling back to compile --validate."
        )
        log_lines.append("$ " + " ".join(compile_command))
        yield (
            None,
            record_dir.as_posix(),
            "**Using this Articraft version's compile validation gate…**",
            "\n".join(log_lines[-300:]),
            "",
        )
        try:
            compiled = _run_check(compile_command, timeout=300)
            log_lines.append(_short_output(compiled, limit=5000))
        except Exception as exc:
            yield (
                None,
                record_dir.as_posix(),
                f"**Fallback Articraft validation could not run:** {exc}",
                "\n".join(log_lines[-300:]),
                "",
            )
            return
        if compiled.returncode:
            yield (
                None,
                record_dir.as_posix(),
                "**Articraft validation failed; no output bundle was published.**",
                "\n".join(log_lines[-300:]),
                "",
            )
            return
        failures = _compile_report_failures(record_id)
        if failures:
            log_lines.append("Blocking compile-report failures: " + "; ".join(failures))
            yield (
                None,
                record_dir.as_posix(),
                "**Articraft validation found blocking model defects; no output bundle was published.**",
                "\n".join(log_lines[-300:]),
                "",
            )
            return

    finalize_command = _articraft_cli_command(
        "external",
        "--repo-root",
        str(ARTICRAFT_OUTPUT_ROOT),
        "finalize",
        record_id,
    )
    log_lines.append("$ " + " ".join(finalize_command))
    try:
        finalized = _run_check(finalize_command, timeout=300)
        log_lines.append(_short_output(finalized, limit=5000))
    except Exception as exc:
        yield (
            None,
            record_dir.as_posix(),
            f"**Articraft finalization could not run:** {exc}",
            "\n".join(log_lines[-300:]),
            "",
        )
        return
    if finalized.returncode:
        yield (
            None,
            record_dir.as_posix(),
            "**Articraft finalization failed; no output bundle was published.**",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    try:
        materialized, archive = _make_result_bundle(record_id)
        status = (
            "**Articraft generation completed and passed the Codex validation workflow.**\n\n"
            f"- Record: `{record_dir}`\n- Compiled output: `{materialized}`\n- Downloadable bundle: `{archive}`"
        )
        try:
            preview_html = _start_articraft_viser_preview(materialized, record_id)
            status += "\n- Interactive Viser preview: ready"
        except Exception as exc:
            preview_html = ""
            status += f"\n- Interactive Viser preview could not start: `{exc}`"
            log_lines.append(f"Viser preview failed: {exc}")
        yield archive.as_posix(), record_dir.as_posix(), status, "\n".join(
            log_lines[-300:]
        ), preview_html
    except Exception as exc:
        yield None, record_dir.as_posix(), f"**Codex finished, but result packaging failed:** {exc}", "\n".join(
            log_lines[-300:]
        ), ""


def build_articraft_panel() -> None:
    """Render the Articraft tab inside the Asset engine."""
    gr.Markdown(
        "### Articulation\n"
        "Generate an articulated object from text and an optional reference image. Codex writes and validates the Articraft model; only submit trusted requests."
    )
    with gr.Row():
        configure_button = gr.Button("Configure Articulation & check Codex")
        generate_button = gr.Button("Generate articulation", variant="primary")
    environment_status = gr.Markdown("**Environment:** not checked.")
    with gr.Row():
        prompt = gr.Textbox(
            label="Articulated object description",
            lines=5,
            placeholder="e.g. A countertop toaster oven with a hinged door and rotating temperature knob.",
        )
        image = gr.Image(
            label="Optional reference image",
            type="filepath",
            image_mode="RGB",
            sources=["upload"],
        )
    with gr.Row():
        output_file = gr.File(
            label="Compiled Articulation result bundle (.zip)", interactive=False
        )
        record_folder = gr.Textbox(
            label="Articulation record folder", interactive=False
        )
    articulation_preview = gr.HTML(
        "<div style='padding: 1rem; color: #6b7280;'>"
        "The interactive Viser articulation preview will appear here after generation."
        "</div>"
    )
    generation_status = gr.Markdown("**Status:** waiting for a description.")
    generation_log = gr.Textbox(
        label="Codex / Articraft log", lines=14, interactive=False
    )

    configure_button.click(
        configure_articraft_environment, outputs=[environment_status], queue=False
    )
    generate_button.click(
        generate_articraft_asset,
        inputs=[prompt, image],
        outputs=[
            output_file,
            record_folder,
            generation_status,
            generation_log,
            articulation_preview,
        ],
    )
