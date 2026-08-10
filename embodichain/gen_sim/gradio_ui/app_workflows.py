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

"""Scene- and Action-engine workflows for the Gradio workspace."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import io
import json
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from app_config import (
    AGENT_CONFIG,
    COMMANDS,
    GEN_SIM_ROOT,
    GEN_SIM_SCENE_ROOT,
    FAST_GYM_CONFIG,
)
from app_env import (
    ACTION_ENGINE_VISER_PORT,
    SCENE_ENGINE_VISER_PORT,
    configure_direct_network_env,
)
from app_media import latest_audience_output_video
from app_processes import (
    SessionProcessRegistry,
    build_run_agent_command,
    get_request_session_id,
    read_process_output,
    start_pipeline,
    terminate_process_group,
)
from app_state import (
    PHASES,
    Phase,
    RuntimeState,
    runtime_lock,
    runtime_registry,
    set_runtime_phase_locked,
)

__all__ = [
    "cleanup_workflow_session",
    "format_status",
    "preview_saved_scene",
    "refresh_saved_scenes",
    "reset_scene_engine",
    "run_action_engine_from_current",
    "run_scene_engine",
    "stop_action_engine",
    "ui_snapshot",
]

configure_direct_network_env()

_scene_runs = SessionProcessRegistry()
_action_runs = SessionProcessRegistry()
_action_preview_runs = SessionProcessRegistry()
_preview_start_lock = threading.Lock()

_ACTION_IDLE_PREVIEW = (
    "<div style='padding: 1rem; color: #6b7280;'>"
    "Select a generated scene to preview it."
    "</div>"
)


def _drain_output_queue(output_queue: queue.Queue[str]) -> list[str]:
    lines: list[str] = []
    while True:
        try:
            lines.append(output_queue.get_nowait())
        except queue.Empty:
            return lines


def _scene_engine_phase_from_log(line: str, current_key: str) -> str:
    """Map Scene Engine stage names to the shared progress UI."""
    text = line.lower()
    mapping = (
        ("scene understanding", "scene_intake"),
        ("scene segmentation", "relations"),
        ("coarse layout", "asset_generation"),
        ("scene export", "gym_export"),
    )
    current_progress = PHASES.get(current_key, PHASES["idle"]).progress
    for needle, phase_key in mapping:
        if needle in text and PHASES[phase_key].progress > current_progress:
            return phase_key
    return current_key


def _scene_engine_updates(
    runtime: RuntimeState,
    output_root: Path | None = None,
    preview_html: str | None = None,
) -> tuple[int, str, str | None, str]:
    with runtime_lock:
        phase = PHASES.get(runtime.phase_key, PHASES["idle"])
        status = format_status(
            runtime.status,
            phase=phase,
            busy=runtime.is_busy,
            last_error=runtime.last_error,
        )
    return (
        phase.progress,
        status,
        output_root.as_posix() if output_root is not None else None,
        preview_html or "",
    )


def _prepare_scene_engine_input(
    image_value: str | np.ndarray | Image.Image,
) -> tuple[str, Path, Path]:
    """Normalize an uploaded image and store it under a stable content hash."""
    if image_value is None:
        raise ValueError("Please upload an image first.")
    if isinstance(image_value, str):
        image = Image.open(image_value)
    elif isinstance(image_value, np.ndarray):
        image = Image.fromarray(image_value)
    elif isinstance(image_value, Image.Image):
        image = image_value
    else:
        raise TypeError(f"Unsupported image input type: {type(image_value)!r}")

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    image_bytes = io.BytesIO()
    normalized.save(image_bytes, format="PNG")
    scene_hash = hashlib.sha256(image_bytes.getvalue()).hexdigest()[:16]
    output_root = GEN_SIM_SCENE_ROOT / scene_hash
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / "input.png"
    image_path.write_bytes(image_bytes.getvalue())
    return scene_hash, output_root, image_path


def _wait_for_viser(port: int, process: subprocess.Popen[str]) -> bool:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _select_available_port(preferred_port: int) -> int:
    """Return the preferred Viser port, or an ephemeral port when occupied."""
    for port in (preferred_port, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(listener.getsockname()[1])
    raise RuntimeError("Could not allocate a local Viser port.")


def _viser_iframe(port: int, scene_hash: str) -> str:
    srcdoc = (
        "<script>window.location.replace(window.top.location.protocol + '//' + "
        f"window.top.location.hostname + ':{port}');</script>"
    )
    return (
        "<div style='margin-top:0.5rem'><strong>Viser preview: "
        f"{html.escape(scene_hash)}</strong>"
        f"<iframe title='Viser scene preview {html.escape(scene_hash)}' "
        f'srcdoc="{html.escape(srcdoc, quote=True)}" '
        "style='width:100%; height:680px; border:1px solid #d1d5db; "
        "border-radius:8px; margin-top:0.5rem;'></iframe></div>"
    )


def _saved_scene_root(scene_name: str) -> Path:
    """Resolve a scene-list value without allowing paths outside the store."""
    if not scene_name or Path(scene_name).name != scene_name:
        raise ValueError("Select a valid generated scene.")
    scene_store = GEN_SIM_SCENE_ROOT.resolve()
    scene_root = (scene_store / scene_name).resolve()
    if scene_root.parent != scene_store:
        raise ValueError("Selected scene must stay within the generated scene store.")
    config_path = scene_root / "scene_export" / "scene_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Scene export is incomplete: {config_path}")
    try:
        scene_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scene config is invalid: {config_path}") from exc
    if not isinstance(scene_config, dict) or scene_config.get("format") != (
        "embodichain.scene-export/v1"
    ):
        raise ValueError(f"Unsupported scene export: {config_path}")
    return scene_root


def saved_scene_choices() -> list[tuple[str, str]]:
    """List complete Scene Engine exports, newest first."""
    if not GEN_SIM_SCENE_ROOT.is_dir():
        return []
    choices: list[tuple[int, str, str]] = []
    for scene_root in GEN_SIM_SCENE_ROOT.iterdir():
        if not scene_root.is_dir():
            continue
        config_path = scene_root / "scene_export" / "scene_config.json"
        if not config_path.is_file():
            continue
        try:
            scene_config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(scene_config, dict) or scene_config.get("format") != (
                "embodichain.scene-export/v1"
            ):
                continue
            scene_id = scene_config.get("scene_id")
            label = (
                f"{scene_root.name} · {scene_id}"
                if isinstance(scene_id, str) and scene_id
                else scene_root.name
            )
            modified_ns = config_path.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            continue
        choices.append((modified_ns, label, scene_root.name))
    choices.sort(reverse=True)
    return [(label, value) for _modified_ns, label, value in choices]


def refresh_saved_scenes(selected_scene: str | None = None):
    """Refresh the Action-engine scene list without selecting a scene implicitly."""
    choices = saved_scene_choices()
    values = {value for _label, value in choices}
    value = selected_scene if selected_scene in values else None
    status = (
        f"**Scene list:** {len(choices)} generated scene(s) available."
        if choices
        else "**Scene list:** no complete generated scenes found."
    )
    return gr.update(choices=choices, value=value), status


def preview_saved_scene(
    scene_name: str | None,
    request: gr.Request,
) -> tuple[str, str]:
    """Start a session-owned Viser preview for one saved scene.

    Args:
        scene_name: Hash-named generated scene selected in the Action panel.
        request: Gradio request carrying the owning session hash.

    Returns:
        Preview iframe HTML and a human-readable preview status.
    """
    session_id = get_request_session_id(request)
    if not scene_name:
        return _ACTION_IDLE_PREVIEW, "**Scene preview:** no scene selected."

    try:
        scene_root = _saved_scene_root(scene_name)
    except (ValueError, FileNotFoundError) as exc:
        return _ACTION_IDLE_PREVIEW, f"**Scene preview error:** {exc}"

    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        if runtime.scene_engine_is_running:
            return (
                _ACTION_IDLE_PREVIEW,
                "**Scene preview:** Scene Engine is still running.",
            )
        token = _action_preview_runs.begin(session_id)

    with _preview_start_lock:
        port = _select_available_port(ACTION_ENGINE_VISER_PORT)
        preview_command = [
            sys.executable,
            COMMANDS["scene_engine"]["preview_script"],
            "--output_root",
            str(scene_root),
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            str(port),
        ]
        try:
            preview_process = start_pipeline(preview_command)
        except Exception as exc:
            return _ACTION_IDLE_PREVIEW, f"**Scene preview error:** {exc}"

        if not _action_preview_runs.attach(session_id, token, preview_process):
            terminate_process_group(preview_process)
            return _ACTION_IDLE_PREVIEW, "**Scene preview:** request was superseded."
        if not _wait_for_viser(port, preview_process):
            terminate_process_group(preview_process)
            _action_preview_runs.finish(session_id, token, preview_process)
            return _ACTION_IDLE_PREVIEW, "**Scene preview error:** Viser did not start."

    if not _action_preview_runs.is_active(session_id, token, preview_process):
        terminate_process_group(preview_process)
        return _ACTION_IDLE_PREVIEW, "**Scene preview:** request was superseded."

    return (
        _viser_iframe(port, scene_name),
        f"**Scene preview:** `{scene_name}` is ready.",
    )


def reset_scene_engine(
    request: gr.Request,
) -> tuple[None, int, str, str, str]:
    """Reset only the requesting session's Scene Engine state and processes.

    Args:
        request: Gradio request carrying the owning session hash.

    Returns:
        Reset values for the Scene Engine input, progress, status, output, and
        preview widgets.
    """
    session_id = get_request_session_id(request)
    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        owns_runtime = runtime.scene_engine_is_running
        action_running = runtime.is_busy and not owns_runtime
        if owns_runtime:
            runtime.is_busy = False
        if not action_running:
            set_runtime_phase_locked(runtime, "idle")
            runtime.status = "Scene Engine reset."
            runtime.last_error = None
            runtime.log_lines.clear()
        runtime.image_path = None
        runtime.scene_engine_is_running = False
        _scene_runs.reset(session_id, force=True)

    message = (
        "Scene Engine reset."
        if not action_running
        else "Scene Engine preview reset; Action Engine is still running."
    )
    return (
        None,
        PHASES["idle"].progress,
        format_status(message),
        "",
        "<div style='padding: 1rem; color: #6b7280;'>"
        "The Viser preview will appear here after generation."
        "</div>",
    )


def run_scene_engine(
    image_value: str | np.ndarray | Image.Image,
    request: gr.Request,
) -> Iterator[tuple[int, str, str | None, str]]:
    """Generate one scene for the requesting Gradio session.

    Args:
        image_value: Uploaded image path, array, or PIL image.
        request: Gradio request carrying the owning session hash.

    Yields:
        Progress, status, output directory, and Viser preview updates.
    """
    session_id = get_request_session_id(request)
    output_root: Path | None = None
    preview_html = ""
    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        if runtime.is_busy:
            runtime.status = "Another engine is already running in this session."
            runtime.last_error = runtime.status
            busy_message = runtime.status
        else:
            token = _scene_runs.begin(session_id)
            runtime.is_busy = True
            runtime.scene_engine_is_running = True
            set_runtime_phase_locked(runtime, "received")
            runtime.status = "Preparing Scene Engine input."
            runtime.last_error = None
            runtime.log_lines.clear()
            busy_message = None

    if busy_message is not None:
        yield _scene_engine_updates(runtime, output_root, preview_html)
        return

    try:
        scene_hash, output_root, image_path = _prepare_scene_engine_input(image_value)
    except Exception as exc:
        with runtime_lock:
            if not _scene_runs.is_active(session_id, token):
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked(runtime, "failed")
            runtime.status = f"Input error: {exc}"
            runtime.last_error = str(exc)
        yield _scene_engine_updates(runtime, output_root, preview_html)
        return

    with runtime_lock:
        if not _scene_runs.is_active(session_id, token):
            return
        runtime.status = f"Image saved. Generating Scene Engine output {scene_hash}."
        runtime.image_path = image_path

    command = [
        sys.executable,
        "-m",
        COMMANDS["scene_engine"]["module"],
        *COMMANDS["scene_engine"]["base_args"],
        "--image",
        str(image_path),
        "--output_root",
        str(output_root),
    ]
    scene_engine_log = output_root / "scene_engine.log"
    scene_engine_log.write_text("$ " + " ".join(command) + "\n", encoding="utf-8")
    with runtime_lock:
        runtime.log_lines.append("$ " + " ".join(command))
    yield _scene_engine_updates(runtime, output_root, preview_html)

    try:
        process = start_pipeline(command)
    except Exception as exc:
        with runtime_lock:
            if not _scene_runs.is_active(session_id, token):
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked(runtime, "failed")
            runtime.status = f"Scene Engine start failed: {exc}"
            runtime.last_error = str(exc)
        yield _scene_engine_updates(runtime, output_root, preview_html)
        return

    if not _scene_runs.attach(session_id, token, process):
        terminate_process_group(process)
        return
    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue, scene_engine_log),
        daemon=True,
    )
    with runtime_lock:
        if not _scene_runs.is_active(session_id, token, process):
            terminate_process_group(process)
            return
        set_runtime_phase_locked(runtime, "started")
        runtime.status = "Scene Engine generation started."
    reader.start()

    while process.poll() is None:
        drained = _drain_output_queue(output_queue)
        with runtime_lock:
            if not _scene_runs.is_active(session_id, token, process):
                return
            for line in drained:
                runtime.log_lines.append(line)
                set_runtime_phase_locked(
                    runtime,
                    _scene_engine_phase_from_log(line, runtime.phase_key),
                )
            if (output_root / "scene_export" / "scene_config.json").is_file():
                set_runtime_phase_locked(runtime, "gym_export")
            runtime.status = PHASES[runtime.phase_key].label + "."
        yield _scene_engine_updates(runtime, output_root, preview_html)
        time.sleep(0.5)

    reader.join(timeout=1.0)
    with runtime_lock:
        if not _scene_runs.is_active(session_id, token, process):
            return
        for line in _drain_output_queue(output_queue):
            runtime.log_lines.append(line)
            set_runtime_phase_locked(
                runtime,
                _scene_engine_phase_from_log(line, runtime.phase_key),
            )
    _scene_runs.finish(session_id, token, process)

    scene_export = output_root / "scene_export" / "scene_config.json"
    if process.returncode != 0 or not scene_export.is_file():
        detail = (
            f"Scene Engine exited with code {process.returncode}."
            if process.returncode != 0
            else f"Scene Engine did not create {scene_export}."
        )
        with runtime_lock:
            if not _scene_runs.is_active(session_id, token):
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked(runtime, "failed")
            runtime.status = detail
            runtime.last_error = detail
        yield _scene_engine_updates(runtime, output_root, preview_html)
        return

    preview_error: str | None = None
    with _preview_start_lock:
        port = _select_available_port(SCENE_ENGINE_VISER_PORT)
        preview_command = [
            sys.executable,
            COMMANDS["scene_engine"]["preview_script"],
            "--output_root",
            str(output_root),
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            str(port),
        ]
        try:
            preview_process = start_pipeline(preview_command)
        except Exception as exc:
            preview_error = f"Viser preview start failed: {exc}"
        else:
            if not _scene_runs.attach(session_id, token, preview_process):
                terminate_process_group(preview_process)
                return
            with runtime_lock:
                runtime.log_lines.append("$ " + " ".join(preview_command))
                set_runtime_phase_locked(runtime, "preview")
                runtime.status = "Starting Viser preview..."

            if not _wait_for_viser(port, preview_process):
                terminate_process_group(preview_process)
                _scene_runs.finish(session_id, token, preview_process)
                preview_error = "Viser preview did not start."

    if preview_error is not None:
        with runtime_lock:
            if not _scene_runs.is_active(session_id, token):
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked(runtime, "failed")
            runtime.status = preview_error
            runtime.last_error = preview_error
        yield _scene_engine_updates(runtime, output_root, preview_html)
        return

    preview_html = _viser_iframe(port, scene_hash)
    with runtime_lock:
        if not _scene_runs.is_active(session_id, token, preview_process):
            terminate_process_group(preview_process)
            return
        runtime.is_busy = False
        runtime.scene_engine_is_running = False
        set_runtime_phase_locked(runtime, "complete")
        runtime.status = "Scene generated successfully. Viser preview is ready."
        runtime.last_error = None
    yield _scene_engine_updates(runtime, output_root, preview_html)


def _action_scene_is_available() -> bool:
    return FAST_GYM_CONFIG.is_file() and AGENT_CONFIG.is_file()


def _action_agent_cli_is_available() -> bool:
    try:
        return importlib.util.find_spec(COMMANDS["agent"]["module"]) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def run_action_engine_from_current(
    task_text: str,
    robot_profile: str | None,
    request: gr.Request,
) -> tuple[object, ...]:
    """Launch DexSim for the requesting Gradio session.

    Args:
        task_text: Natural-language task for the action agent.
        robot_profile: Optional robot selection exposed by the UI.
        request: Gradio request carrying the owning session hash.

    Returns:
        Current Action Engine widget values for the session.
    """
    session_id = get_request_session_id(request)
    task_text = (task_text or "").strip()
    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        if not task_text:
            failure = "Enter a task description first."
        elif not _action_scene_is_available():
            failure = "Current Gym scene/config is unavailable."
        elif runtime.is_busy:
            failure = "Another engine is already running in this session."
        elif not _action_agent_cli_is_available():
            failure = "Action-agent CLI is unavailable in this environment."
        else:
            failure = None
            token = _action_runs.begin(session_id)
            runtime.is_busy = True
            runtime.task_text = task_text
            runtime.status = "Starting DexSim action simulation..."
            runtime.last_error = None
            runtime.log_lines.clear()
            set_runtime_phase_locked(runtime, "started")

        if failure is not None:
            runtime.status = failure
            runtime.last_error = failure

    if failure is None:
        error = _launch_current_simulation(
            session_id,
            runtime,
            token,
            robot_profile=robot_profile,
        )
        if error:
            with runtime_lock:
                if _action_runs.is_active(session_id, token):
                    runtime.is_busy = False
                    set_runtime_phase_locked(runtime, "failed")
                    runtime.status = error
                    runtime.last_error = error
    return ui_snapshot(session_id)


def stop_action_engine(request: gr.Request) -> tuple[object, ...]:
    """Stop only the requesting session's Action processes and reset its UI.

    The DexSim process group includes any child processes it launches. The
    separately managed Action scene preview is also stopped. Other sessions and
    the requesting session's Scene and Asset processes remain untouched.

    Args:
        request: Gradio request for the browser session initiating Stop.

    Returns:
        Reset values for the Action preview, status, video, task, and progress
        widgets.
    """
    session_id = get_request_session_id(request)
    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        _action_runs.reset(session_id, force=True)
        _action_preview_runs.reset(session_id, force=True)
        if not runtime.scene_engine_is_running:
            runtime.is_busy = False
            set_runtime_phase_locked(runtime, "idle")
            runtime.status = "Action Engine stopped."
            runtime.last_error = None
            runtime.log_lines.clear()
        runtime.task_text = ""
        runtime.video_path = None
        runtime.last_sent_video_signature = None

    return (
        _ACTION_IDLE_PREVIEW,
        "**Scene preview:** Action Engine stopped.",
        None,
        "",
        PHASES["idle"].progress,
        format_status("Action Engine stopped."),
    )


def cleanup_workflow_session(request: gr.Request) -> None:
    """Stop Scene and Action processes owned by a disconnected session.

    Args:
        request: Gradio unload request carrying the disconnected session hash.
    """
    session_id = get_request_session_id(request)
    with runtime_lock:
        _scene_runs.reset(session_id, force=True)
        _action_runs.reset(session_id, force=True)
        _action_preview_runs.reset(session_id, force=True)
        runtime_registry.reset(session_id)


def _launch_current_simulation(
    session_id: str,
    runtime: RuntimeState,
    token: str,
    *,
    robot_profile: str | None = None,
) -> str | None:
    command = build_run_agent_command(robot_profile=robot_profile)
    started_at_ns = time.time_ns()
    try:
        process = start_pipeline(command)
    except Exception as exc:
        return f"DexSim launch failed: {exc}"

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue),
        daemon=True,
    )
    monitor = threading.Thread(
        target=_monitor_simulation,
        args=(session_id, runtime, token, process, output_queue, reader, started_at_ns),
        daemon=True,
    )

    if not _action_runs.attach(session_id, token, process):
        terminate_process_group(process)
        return None
    with runtime_lock:
        if not _action_runs.is_active(session_id, token, process):
            terminate_process_group(process)
            return None
        runtime.log_lines.append("$ " + " ".join(command))

    reader.start()
    monitor.start()
    return None


def _monitor_simulation(
    session_id: str,
    runtime: RuntimeState,
    token: str,
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
    reader: threading.Thread,
    started_at_ns: int,
) -> None:
    while process.poll() is None:
        _append_simulation_logs(
            session_id,
            runtime,
            token,
            process,
            _drain_output_queue(output_queue),
        )
        if not _action_runs.is_active(session_id, token, process):
            return
        time.sleep(0.5)

    reader.join(timeout=1.0)
    _append_simulation_logs(
        session_id,
        runtime,
        token,
        process,
        _drain_output_queue(output_queue),
    )
    if not _action_runs.is_active(session_id, token, process):
        return
    source_video = latest_audience_output_video(min_mtime_ns=started_at_ns)
    display_video: Path | None = None
    if source_video is not None:
        destination = GEN_SIM_ROOT / "action_videos" / token / source_video.name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, destination)
            display_video = destination
        except OSError as exc:
            _append_simulation_logs(
                session_id,
                runtime,
                token,
                process,
                [f"Could not copy the simulation preview into the workspace: {exc}"],
            )

    with runtime_lock:
        if not _action_runs.is_active(session_id, token, process):
            return
        runtime.is_busy = False
        runtime.video_path = display_video
        if process.returncode == 0:
            set_runtime_phase_locked(runtime, "complete")
            runtime.status = "DexSim simulation finished successfully."
            runtime.last_error = None
            if display_video is None:
                runtime.log_lines.append("No simulation preview video was found.")
        else:
            set_runtime_phase_locked(runtime, "failed")
            runtime.status = f"DexSim exited with return code {process.returncode}."
            runtime.last_error = runtime.status
    _action_runs.finish(session_id, token, process)


def _append_simulation_logs(
    session_id: str,
    runtime: RuntimeState,
    token: str,
    process: subprocess.Popen[str],
    lines: list[str],
) -> None:
    if not lines:
        return
    with runtime_lock:
        if _action_runs.is_active(session_id, token, process):
            runtime.log_lines.extend(lines)


def ui_snapshot(
    session_id: str,
    extra_status: str | None = None,
) -> tuple[object, ...]:
    """Return Action-engine widget values for one Gradio session.

    Args:
        session_id: Stable Gradio session identifier.
        extra_status: Optional text appended to the stored status.

    Returns:
        Video, task, progress, status, and compatibility placeholder values.
    """
    with runtime_lock:
        runtime = runtime_registry.get(session_id)
        phase = PHASES.get(runtime.phase_key, PHASES["idle"])
        video_value = None
        video_signature = None
        if runtime.video_path and runtime.video_path.is_file():
            video_value = runtime.video_path.as_posix()
            video_signature = (video_value, runtime.video_path.stat().st_mtime_ns)
        if video_signature != runtime.last_sent_video_signature:
            runtime.last_sent_video_signature = video_signature
            video_update = video_value
        else:
            video_update = gr.update()
        task_text = runtime.task_text
        status_text = runtime.status
        if extra_status:
            status_text = f"{status_text}\n{extra_status}"
        busy = runtime.is_busy
        last_error = runtime.last_error

    return (
        video_update,
        task_text,
        phase.progress,
        format_status(
            status_text,
            phase=phase,
            busy=busy,
            last_error=last_error,
        ),
        None,
        None,
        None,
    )


def format_status(
    status_text: str,
    *,
    phase: Phase | None = None,
    busy: bool = False,
    last_error: str | None = None,
) -> str:
    """Format an engine status for display in Gradio."""
    if phase is None:
        phase = PHASES["idle"]
    state = "running" if busy else "ready"
    parts = [
        f"**State:** {state}",
        f"**Phase:** {phase.progress}% - {phase.label}",
        f"**Status:** {status_text}",
    ]
    if last_error:
        escaped_error = last_error.replace("`", "'")
        if "\n" in escaped_error:
            parts.append(f"**Last error:**\n```text\n{escaped_error}\n```")
        else:
            parts.append(f"**Last error:** `{escaped_error}`")
    return "\n\n".join(parts)
