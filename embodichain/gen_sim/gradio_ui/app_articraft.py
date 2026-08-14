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
import html
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from app_env import (
    ARTICRAFT_CONDA_ENV,
    ARTICRAFT_OUTPUT_ROOT,
    ARTICRAFT_REPOSITORY_URL,
    ARTICRAFT_ROOT,
    ARTICRAFT_VISER_PORT,
    EMBODICHAIN_ROOT,
    validate_gradio_artifact_root,
)
from app_processes import (
    SessionProcessRegistry,
    build_codex_env,
    build_pipeline_env,
    get_request_session_id,
    kill_process_group,
    read_process_output,
    redact_sensitive_text,
    register_managed_process,
    start_pipeline,
    terminate_process_group,
)
from embodichain.gen_sim.env import find_gen_sim_env_file

__all__ = [
    "build_articraft_panel",
    "cleanup_articraft_session",
    "configure_articraft_environment",
    "generate_articraft_asset",
    "reset_articraft_asset",
    "stop_articraft_viser_preview",
]

_VISER_START_TIMEOUT_SECONDS = 15.0
_ARTICRAFT_PYTHON_VERSION = "3.12"
_CONDA_ENVIRONMENT_SETUP_TIMEOUT_SECONDS = 1_200
_INTERACTION_ANNOTATION_TIMEOUT_SECONDS = 600
_ARTICRAFT_PART_MASS_KG = 0.1
_ROTATE_JOINT_TYPES = frozenset({"revolute", "continuous"})
_TRANSLATE_JOINT_TYPES = frozenset({"prismatic"})
_INTERACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["interactions"],
    "properties": {
        "interactions": {
            "type": "array",
            "maxItems": 256,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["link", "visual"],
                "properties": {
                    "link": {"type": "string", "minLength": 1},
                    "visual": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}
_articraft_environment_lock = threading.Lock()
_articraft_runs = SessionProcessRegistry()
_ARTICRAFT_IDLE_PREVIEW = (
    "<div style='padding: 1rem; color: #6b7280;'>"
    "The interactive Viser articulation preview will appear here after generation."
    "</div>"
)


def _run_articraft_generation_check(
    command: list[str], *, session_id: str, token: str, timeout: int
) -> subprocess.CompletedProcess[str] | None:
    """Run one Articraft CLI gate so Reset can stop its whole process group."""
    process = register_managed_process(
        subprocess.Popen(
            command,
            cwd=ARTICRAFT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=build_pipeline_env(),
        )
    )
    if not _articraft_runs.attach(session_id, token, process):
        terminate_process_group(process)
        return None
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        raise
    finally:
        _articraft_runs.finish(session_id, token, process)
    if not _articraft_runs.is_active(session_id, token):
        return None
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        redact_sensitive_text(stdout or ""),
    )


def reset_articraft_asset(request: gr.Request) -> tuple[Any, ...]:
    """Clear Articraft state and stop only the requesting session's processes.

    Args:
        request: Gradio request for the browser session initiating Reset.

    Returns:
        Reset values for all Articraft panel widgets.
    """
    session_id = get_request_session_id(request)
    cleanup_articraft_session(session_id)
    return (
        "**Environment:** not checked.",
        "",
        None,
        None,
        "",
        "**Status:** waiting for a description.",
        "",
        _ARTICRAFT_IDLE_PREVIEW,
    )


def cleanup_articraft_session(session_id: str) -> None:
    """Stop Articraft generation and preview processes for one session.

    Args:
        session_id: Stable Gradio session identifier.
    """
    _articraft_runs.reset(session_id, force=True)
    stop_articraft_viser_preview(session_id, force=True)


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
    isolation_error = _articraft_isolation_error()
    if isolation_error:
        errors.append(isolation_error)
    elif not (
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
    if isolation_error := _articraft_isolation_error():
        return False, isolation_error
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


def _articraft_isolation_error() -> str | None:
    """Return an error when Codex roots could contain deployment secrets."""
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


def _materialized_record_dir(record_id: str) -> Path:
    """Return one record's immutable Articraft materialization directory."""
    return (
        ARTICRAFT_OUTPUT_ROOT / "data" / "cache" / "record_materialization" / record_id
    )


def _validate_link_inertials(urdf_path: Path, *, expected_mass: float) -> None:
    """Require every generated URDF link to have finite inertial properties."""
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Could not parse generated URDF: {urdf_path}") from exc

    failures: list[str] = []
    for link in root.findall("link"):
        link_name = (link.get("name") or "<unnamed>").strip()
        inertial = link.find("inertial")
        mass_element = None if inertial is None else inertial.find("mass")
        inertia_element = None if inertial is None else inertial.find("inertia")
        if mass_element is None or inertia_element is None:
            failures.append(f"{link_name}: missing <inertial>, <mass>, or <inertia>")
            continue
        try:
            mass = float(mass_element.get("value", ""))
            inertia_values = [
                float(inertia_element.get(name, ""))
                for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            ]
        except ValueError:
            failures.append(f"{link_name}: mass or inertia tensor is not numeric")
            continue
        if not math.isfinite(mass) or not math.isclose(
            mass,
            expected_mass,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            failures.append(f"{link_name}: expected mass {expected_mass}, got {mass}")
        if not all(math.isfinite(value) for value in inertia_values):
            failures.append(f"{link_name}: inertia tensor contains a non-finite value")

    if failures:
        raise ValueError(
            "Generated URDF does not provide the required inertial properties for every link: "
            + "; ".join(failures)
        )


def _prepare_result_bundle(record_id: str) -> Path:
    """Copy a materialized record into a mutable, record-scoped export directory."""
    materialized = _materialized_record_dir(record_id)
    model_path = materialized / "model.urdf"
    if not model_path.is_file():
        raise FileNotFoundError(
            "Articraft completed without a compiled model.urdf output."
        )
    _validate_link_inertials(model_path, expected_mass=_ARTICRAFT_PART_MASS_KG)
    exports_root = ARTICRAFT_OUTPUT_ROOT / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    result_dir = exports_root / record_id
    if result_dir.exists():
        raise FileExistsError(f"Articraft export already exists: {result_dir}")
    shutil.copytree(materialized, result_dir)
    shutil.copy2(result_dir / "model.urdf", result_dir / "model.raw.urdf")
    return result_dir


def _archive_result_bundle(record_id: str, result_dir: Path) -> Path:
    """Archive one fully post-processed Articraft export directory."""
    archive = Path(
        shutil.make_archive(
            (result_dir.parent / record_id).as_posix(),
            "zip",
            root_dir=result_dir,
        )
    )
    return archive


def _articraft_viser_iframe(record_id: str, port: int) -> str:
    """Embed the Articulation Viser service through the Gradio page hostname."""
    srcdoc = (
        "<script>window.location.replace(window.top.location.protocol + '//' + "
        f"window.top.location.hostname + ':{port}');</script>"
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
    """Own an isolated Articraft Viser process for each Gradio session."""

    def __init__(self, preferred_port: int) -> None:
        self._preferred_port = preferred_port
        self._lock = threading.Lock()
        self._processes: dict[str, tuple[subprocess.Popen[str], int]] = {}

    def start(self, session_id: str, urdf_path: Path, record_id: str) -> str:
        """Replace one session's preview with a verified preview of one URDF."""
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Compiled URDF is missing: {urdf_path}")

        with self._lock:
            previous = self._processes.pop(session_id, None)
            if previous is not None:
                terminate_process_group(previous[0])
            port = self._select_available_port()
            process = start_pipeline(self._command(urdf_path, port))
            if not self._wait_until_owned(process, port):
                terminate_process_group(process)
                raise RuntimeError("New Articraft Viser preview did not bind its port.")
            self._processes[session_id] = (process, port)
        return _articraft_viser_iframe(record_id, port)

    def stop(self, session_id: str | None = None, *, force: bool = False) -> None:
        """Stop one session's preview, or every preview during shutdown."""
        with self._lock:
            if session_id is None:
                processes = tuple(
                    process for process, _port in self._processes.values()
                )
                self._processes.clear()
            else:
                current = self._processes.pop(session_id, None)
                processes = () if current is None else (current[0],)
        stop_process = kill_process_group if force else terminate_process_group
        for process in processes:
            stop_process(process)

    def _command(self, urdf_path: Path, port: int) -> list[str]:
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
            str(port),
        ]

    def _wait_until_owned(self, process: subprocess.Popen[str], port: int) -> bool:
        deadline = time.monotonic() + _VISER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return True
            except OSError:
                pass
            time.sleep(0.25)
        return False

    def _select_available_port(self) -> int:
        if self._port_is_available(self._preferred_port):
            return self._preferred_port
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", 0))
            return int(probe.getsockname()[1])
        finally:
            probe.close()

    @staticmethod
    def _port_is_available(port: int) -> bool:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            return False
        finally:
            probe.close()
        return True


_articraft_viser_preview = _ArticraftViserPreview(ARTICRAFT_VISER_PORT)


def stop_articraft_viser_preview(
    session_id: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Stop the Viser subprocess currently owned by the Articraft panel.

    The preview runs independently from Gradio so it can be embedded through an
    iframe. Expose its cleanup explicitly so application shutdown can release
    the dedicated port instead of leaving an orphaned Viser server behind.

    Args:
        session_id: Optional owning Gradio session. ``None`` stops all previews.
        force: Whether to immediately send ``SIGKILL`` for interactive cleanup.
    """
    _articraft_viser_preview.stop(session_id, force=force)


atexit.register(stop_articraft_viser_preview)


def _start_articraft_viser_preview(
    session_id: str, materialized: Path, record_id: str
) -> str:
    """Load the compiled URDF as an articulation and expose it through Viser."""
    return _articraft_viser_preview.start(
        session_id,
        materialized / "model.urdf",
        record_id,
    )


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


def _build_interaction_annotation_prompt(*, prompt: str) -> str:
    """Build the constrained Codex prompt used for interaction post-processing."""
    return f"""You are performing semantic post-processing on one completed articulated URDF.

Original user request:
{prompt}

Read model.urdf in the current directory and inspect its full link, visual, collision, and joint
structure. You may inspect referenced local mesh files when names and primitive geometry are not
enough. Do not modify any file.

Identify the named visual geometry that a robot should physically contact to operate each
user-facing articulated mechanism. Select the actual handle, grip, knob cap, button cap, lever, or
other contact surface. Do not select frames, windows, decorative panels, hinge barrels, mounting
blocks, hidden shafts, plungers, or an entire link merely because that link moves.

Return only strict JSON in exactly this shape:
{{"interactions":[{{"link":"child_link_name","visual":"visual_name"}}]}}

Use names exactly as they appear in model.urdf. Include multiple entries only when there are
multiple legitimate contact surfaces. Do not return a motion type: the application derives
rotate or translate deterministically from the selected link's parent joint. Do not include
Markdown fences, commentary, confidence values, or any additional keys."""


def _parse_interaction_targets(response: str) -> list[tuple[str, str]]:
    """Parse and validate Codex's strict interaction-target response."""
    raw = response.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex interaction response is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"interactions"}:
        raise ValueError(
            "Codex interaction response must contain only an 'interactions' array."
        )
    interactions = payload["interactions"]
    if not isinstance(interactions, list):
        raise ValueError("Codex interaction response 'interactions' must be an array.")
    if len(interactions) > 256:
        raise ValueError("Codex interaction response contains too many targets.")

    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, interaction in enumerate(interactions):
        if not isinstance(interaction, dict) or set(interaction) != {"link", "visual"}:
            raise ValueError(
                f"Interaction target {index} must contain only 'link' and 'visual'."
            )
        link = interaction["link"]
        visual = interaction["visual"]
        if not isinstance(link, str) or not link.strip():
            raise ValueError(f"Interaction target {index} has an invalid link name.")
        if not isinstance(visual, str) or not visual.strip():
            raise ValueError(f"Interaction target {index} has an invalid visual name.")
        target = (link.strip(), visual.strip())
        if target in seen:
            raise ValueError(
                f"Codex interaction response repeats target {target[0]!r}/{target[1]!r}."
            )
        seen.add(target)
        targets.append(target)
    return targets


def _interaction_type_for_joint(joint_type: str) -> str:
    """Map one URDF joint type to the supported interaction motion vocabulary."""
    normalized = joint_type.strip().lower()
    if normalized in _ROTATE_JOINT_TYPES:
        return "rotate"
    if normalized in _TRANSLATE_JOINT_TYPES:
        return "translate"
    raise ValueError(
        f"Joint type {joint_type!r} cannot drive a rotate/translate interaction."
    )


def _inject_interaction_annotations(
    urdf_path: Path,
    *,
    targets: list[tuple[str, str]],
    metadata_path: Path,
) -> list[dict[str, str]]:
    """Validate targets, inject visual tags, and persist normalized metadata."""
    try:
        tree = ET.parse(urdf_path)
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Could not parse generated URDF: {urdf_path}") from exc
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("Generated URDF root element must be <robot>.")

    links: dict[str, ET.Element] = {}
    for link_element in root.findall("link"):
        link_name = (link_element.get("name") or "").strip()
        if not link_name:
            raise ValueError("Generated URDF contains a link without a name.")
        if link_name in links:
            raise ValueError(f"Generated URDF repeats link name {link_name!r}.")
        links[link_name] = link_element

    parent_joints: dict[str, tuple[str, str]] = {}
    for joint_element in root.findall("joint"):
        joint_name = (joint_element.get("name") or "").strip()
        joint_type = (joint_element.get("type") or "").strip()
        child_element = joint_element.find("child")
        child_name = (
            (child_element.get("link") or "").strip()
            if child_element is not None
            else ""
        )
        if not joint_name or not joint_type or not child_name:
            raise ValueError("Generated URDF contains an incomplete joint declaration.")
        if child_name in parent_joints:
            raise ValueError(
                f"Generated URDF gives link {child_name!r} multiple parent joints."
            )
        parent_joints[child_name] = (joint_name, joint_type)

    for link_element in links.values():
        for visual_element in link_element.findall("visual"):
            for existing in visual_element.findall("interact"):
                visual_element.remove(existing)

    normalized: list[dict[str, str]] = []
    for link_name, visual_name in targets:
        link_element = links.get(link_name)
        if link_element is None:
            raise ValueError(
                f"Interaction target references unknown link {link_name!r}."
            )
        matches = [
            visual_element
            for visual_element in link_element.findall("visual")
            if (visual_element.get("name") or "").strip() == visual_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Interaction target {link_name!r}/{visual_name!r} must match exactly one visual."
            )

        collision_matches = [
            collision_element
            for collision_element in link_element.findall("collision")
            if (collision_element.get("name") or "").strip() == visual_name
        ]
        if len(collision_matches) != 1:
            raise ValueError(
                f"Interaction target {link_name!r}/{visual_name!r} must have one same-named collision."
            )

        parent_joint = parent_joints.get(link_name)
        if parent_joint is None:
            raise ValueError(
                f"Interaction target link {link_name!r} has no parent articulation."
            )
        joint_name, joint_type = parent_joint
        interaction_type = _interaction_type_for_joint(joint_type)
        ET.SubElement(matches[0], "interact", {"type": interaction_type})
        normalized.append(
            {
                "link": link_name,
                "visual": visual_name,
                "joint": joint_name,
                "type": interaction_type,
            }
        )

    ET.indent(tree, space="  ")
    temporary_path = urdf_path.with_name(f".{urdf_path.name}.interaction.tmp")
    tree.write(temporary_path, encoding="unicode", xml_declaration=False)
    os.replace(temporary_path, urdf_path)
    metadata = {"schema_version": 1, "interactions": normalized}
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return normalized


def _run_interaction_annotation_codex(
    *,
    codex: str,
    prompt: str,
    result_dir: Path,
    run_root: Path,
    reference_image: Path | None,
    session_id: str,
    token: str,
) -> subprocess.CompletedProcess[str] | None:
    """Run the isolated Codex semantic pass with reset-aware process ownership."""
    response_path = run_root / "interaction_codex_response.json"
    schema_path = run_root / "interaction_response_schema.json"
    schema_path.write_text(
        json.dumps(_INTERACTION_RESPONSE_SCHEMA, indent=2) + "\n",
        encoding="utf-8",
    )
    command = [
        codex,
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        'web_search="disabled"',
        "--color",
        "never",
        "-C",
        str(result_dir),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
    ]
    if reference_image:
        command.extend(["--image", str(reference_image)])
    command.append(_build_interaction_annotation_prompt(prompt=prompt))

    process = register_managed_process(
        subprocess.Popen(
            command,
            cwd=result_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=build_codex_env(),
        )
    )
    if not _articraft_runs.attach(session_id, token, process):
        terminate_process_group(process)
        return None
    try:
        stdout, _ = process.communicate(timeout=_INTERACTION_ANNOTATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        raise
    finally:
        _articraft_runs.finish(session_id, token, process)
    if not _articraft_runs.is_active(session_id, token):
        return None
    if process.returncode == 0 and not response_path.is_file():
        raise RuntimeError("Codex completed without an interaction JSON response.")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        redact_sensitive_text(stdout or ""),
    )


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

Do not spend generation effort hand-authoring `part.inertial` values or inertial-specific tests.
Articraft deterministically derives missing link mass, center of mass, and inertia from the complete
link geometry during URDF compilation.

{_conda_path()} run --no-capture-output -n {ARTICRAFT_CONDA_ENV} python -m cli.main external --repo-root {ARTICRAFT_OUTPUT_ROOT} check {record_id}

Then run:

{_conda_path()} run --no-capture-output -n {ARTICRAFT_CONDA_ENV} python -m cli.main external --repo-root {ARTICRAFT_OUTPUT_ROOT} finalize {record_id}

The Gradio app packages the compiled URDF and meshes after you finish. In your final response,
briefly state the articulation mechanisms and validation result."""


def generate_articraft_asset(
    prompt_value: str,
    image_value: Any,
    request: gr.Request,
) -> Iterator[tuple[Any, ...]]:
    """Initialize a record, let Codex author it, and expose one result bundle.

    Args:
        prompt_value: Requested articulated-object description.
        image_value: Optional Gradio reference-image value.
        request: Gradio request identifying the owning browser session.

    Yields:
        Updated artifact, status, log, and Viser preview values for the panel.
    """
    session_id = get_request_session_id(request)
    token = _articraft_runs.begin(session_id)
    prompt = (prompt_value or "").strip()
    if not prompt:
        if _articraft_runs.is_active(session_id, token):
            yield None, "", "**Input error:** enter a description of the articulated object.", "", ""
        return

    details, errors, codex = _check_requirements()
    if errors or not codex:
        message = (
            "\n".join(f"- {error}" for error in errors) or "Codex CLI is unavailable."
        )
        if _articraft_runs.is_active(session_id, token):
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
        initialized = _run_articraft_generation_check(
            init_command,
            session_id=session_id,
            token=token,
            timeout=90,
        )
        if initialized is None:
            return
        log_lines.append(_short_output(initialized, limit=4000))
        if initialized.returncode:
            yield None, "", "**Articraft record initialization failed.**", "\n".join(
                log_lines
            ), ""
            return
        model_path = _active_model_path(record_dir)
    except Exception as exc:
        if _articraft_runs.is_active(session_id, token):
            yield None, "", f"**Setup failed:** {exc}", "\n".join(log_lines), ""
        return

    if not _articraft_runs.is_active(session_id, token):
        return

    final_message = run_root / "codex_final_message.txt"
    codex_command = [
        codex,
        "exec",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        'web_search="disabled"',
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
        process = register_managed_process(
            subprocess.Popen(
                codex_command,
                cwd=ARTICRAFT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=build_codex_env(),
            )
        )
        if not _articraft_runs.attach(session_id, token, process):
            terminate_process_group(process)
            return
    except Exception as exc:
        if _articraft_runs.is_active(session_id, token):
            yield None, record_dir.as_posix(), f"**Codex could not start:** {exc}", "\n".join(
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
        yield None, record_dir.as_posix(), "**Codex is generating and validating the Articraft model…**", "\n".join(
            log_lines[-240:]
        ), ""
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

    if final_message.is_file():
        final_text = final_message.read_text(encoding="utf-8", errors="replace").strip()
        if final_text:
            log_lines.append(
                "\nCodex final response:\n" + redact_sensitive_text(final_text)
            )
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
        checked = _run_articraft_generation_check(
            check_command,
            session_id=session_id,
            token=token,
            timeout=300,
        )
        if checked is None:
            return
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
            compiled = _run_articraft_generation_check(
                compile_command,
                session_id=session_id,
                token=token,
                timeout=300,
            )
            if compiled is None:
                return
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
        finalized = _run_articraft_generation_check(
            finalize_command,
            session_id=session_id,
            token=token,
            timeout=300,
        )
        if finalized is None:
            return
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

    if not _articraft_runs.is_active(session_id, token):
        return
    try:
        result_dir = _prepare_result_bundle(record_id)
    except Exception as exc:
        yield None, record_dir.as_posix(), f"**Codex finished, but result staging failed:** {exc}", "\n".join(
            log_lines[-300:]
        ), ""
        return

    log_lines.append("$ codex exec … <interaction annotation prompt>")
    yield (
        None,
        record_dir.as_posix(),
        "**Articraft is complete. Codex is identifying robot interaction surfaces…**",
        "\n".join(log_lines[-300:]),
        "",
    )
    try:
        annotated = _run_interaction_annotation_codex(
            codex=codex,
            prompt=prompt,
            result_dir=result_dir,
            run_root=run_root,
            reference_image=reference_image,
            session_id=session_id,
            token=token,
        )
        if annotated is None:
            return
        log_lines.append(_short_output(annotated, limit=5000))
    except Exception as exc:
        yield None, record_dir.as_posix(), f"**Interaction annotation could not run:** {exc}", "\n".join(
            log_lines[-300:]
        ), ""
        return
    if annotated.returncode:
        yield (
            None,
            record_dir.as_posix(),
            "**Codex interaction analysis failed; no output bundle was published.**",
            "\n".join(log_lines[-300:]),
            "",
        )
        return

    try:
        interaction_response_path = run_root / "interaction_codex_response.json"
        targets = _parse_interaction_targets(
            interaction_response_path.read_text(encoding="utf-8")
        )
        interactions = _inject_interaction_annotations(
            result_dir / "model.urdf",
            targets=targets,
            metadata_path=result_dir / "interactions.json",
        )
        log_lines.append(
            f"Validated and injected {len(interactions)} interaction annotation(s)."
        )
        archive = _archive_result_bundle(record_id, result_dir)
        status = (
            "**Articraft generation and interaction annotation completed.**\n\n"
            f"- Record: `{record_dir}`\n- Annotated output: `{result_dir}`"
            f"\n- Interaction targets: `{len(interactions)}`\n- Downloadable bundle: `{archive}`"
        )
        try:
            preview_html = _start_articraft_viser_preview(
                session_id,
                result_dir,
                record_id,
            )
            status += "\n- Interactive Viser preview: ready"
        except Exception as exc:
            preview_html = ""
            status += f"\n- Interactive Viser preview could not start: `{exc}`"
            log_lines.append(f"Viser preview failed: {exc}")
        yield archive.as_posix(), record_dir.as_posix(), status, "\n".join(
            log_lines[-300:]
        ), preview_html
    except Exception as exc:
        yield None, record_dir.as_posix(), f"**Interaction annotation validation or packaging failed:** {exc}", "\n".join(
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
        reset_button = gr.Button("Reset Articulation", variant="stop")
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
    articulation_preview = gr.HTML(_ARTICRAFT_IDLE_PREVIEW)
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
