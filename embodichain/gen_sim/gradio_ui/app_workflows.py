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
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from app_config import (
    AGENT_CONFIG,
    COMMANDS,
    GEN_SIM_SCENE_ROOT,
    FAST_GYM_CONFIG,
)
from app_env import SCENE_ENGINE_VISER_PORT, configure_direct_network_env
from app_media import latest_audience_output_video
from app_processes import (
    build_run_agent_command,
    read_process_output,
    start_pipeline,
    terminate_process_group,
)
from app_state import PHASES, Phase, runtime, runtime_lock, set_runtime_phase_locked

__all__ = [
    "format_status",
    "preview_saved_scene",
    "refresh_saved_scenes",
    "reset_scene_engine",
    "run_action_engine_from_current",
    "run_scene_engine",
    "ui_snapshot",
]

configure_direct_network_env()


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


def preview_saved_scene(scene_name: str | None):
    """Start a Viser preview for the explicitly selected generated scene."""
    idle_preview = (
        "<div style='padding: 1rem; color: #6b7280;'>"
        "Select a generated scene to preview it."
        "</div>"
    )
    if not scene_name:
        return idle_preview, "**Scene preview:** no scene selected."

    try:
        scene_root = _saved_scene_root(scene_name)
    except (ValueError, FileNotFoundError) as exc:
        return idle_preview, f"**Scene preview error:** {exc}"

    with runtime_lock:
        if runtime.scene_engine_is_running:
            return idle_preview, "**Scene preview:** Scene Engine is still running."
        old_preview = runtime.scene_preview_process
        runtime.scene_preview_process = None

    if old_preview is not None:
        terminate_process_group(old_preview)

    port = SCENE_ENGINE_VISER_PORT
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
        return idle_preview, f"**Scene preview error:** {exc}"

    with runtime_lock:
        runtime.scene_preview_process = preview_process
    if not _wait_for_viser(port, preview_process):
        terminate_process_group(preview_process)
        with runtime_lock:
            if runtime.scene_preview_process is preview_process:
                runtime.scene_preview_process = None
        return idle_preview, "**Scene preview error:** Viser did not start."

    return (
        _viser_iframe(port, scene_name),
        f"**Scene preview:** `{scene_name}` is ready.",
    )


def reset_scene_engine():
    """Clear Scene Engine widgets and stop its owned process groups."""
    with runtime_lock:
        generator_process = runtime.scene_engine_process
        preview_process = runtime.scene_preview_process
        owns_runtime = runtime.scene_engine_is_running
        action_running = runtime.sim_process is not None
        if owns_runtime:
            runtime.run_token = uuid.uuid4().hex
            runtime.is_busy = False
        if not action_running:
            set_runtime_phase_locked("idle")
            runtime.status = "Scene Engine reset."
            runtime.last_error = None
            runtime.log_lines.clear()
        runtime.image_path = None
        runtime.scene_engine_process = None
        runtime.scene_preview_process = None
        runtime.scene_engine_is_running = False

    for process in {generator_process, preview_process}:
        if process is not None:
            terminate_process_group(process)

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


def run_scene_engine(image_value: str | np.ndarray | Image.Image):
    """Generate one image-conditioned scene and expose its Viser preview."""
    output_root: Path | None = None
    preview_html = ""
    token = uuid.uuid4().hex
    with runtime_lock:
        if runtime.is_busy:
            runtime.status = "Another engine is already running."
            runtime.last_error = runtime.status
            busy_message = runtime.status
        else:
            runtime.run_token = token
            runtime.is_busy = True
            runtime.scene_engine_is_running = True
            set_runtime_phase_locked("received")
            runtime.status = "Preparing Scene Engine input."
            runtime.last_error = None
            runtime.log_lines.clear()
            busy_message = None

    if busy_message is not None:
        yield _scene_engine_updates(output_root, preview_html)
        return

    try:
        scene_hash, output_root, image_path = _prepare_scene_engine_input(image_value)
    except Exception as exc:
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked("failed")
            runtime.status = f"Input error: {exc}"
            runtime.last_error = str(exc)
        yield _scene_engine_updates(output_root, preview_html)
        return

    with runtime_lock:
        if runtime.run_token != token:
            return
        old_preview = runtime.scene_preview_process
        runtime.scene_preview_process = None
        runtime.status = f"Image saved. Generating Scene Engine output {scene_hash}."
        runtime.image_path = image_path

    if old_preview is not None:
        terminate_process_group(old_preview)

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
    yield _scene_engine_updates(output_root, preview_html)

    try:
        process = start_pipeline(command)
    except Exception as exc:
        with runtime_lock:
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked("failed")
            runtime.status = f"Scene Engine start failed: {exc}"
            runtime.last_error = str(exc)
        yield _scene_engine_updates(output_root, preview_html)
        return

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue, scene_engine_log),
        daemon=True,
    )
    with runtime_lock:
        if runtime.run_token != token:
            terminate_process_group(process)
            return
        runtime.scene_engine_process = process
        set_runtime_phase_locked("started")
        runtime.status = "Scene Engine generation started."
    reader.start()

    while process.poll() is None:
        drained = _drain_output_queue(output_queue)
        with runtime_lock:
            if (
                runtime.run_token != token
                or runtime.scene_engine_process is not process
            ):
                return
            for line in drained:
                runtime.log_lines.append(line)
                set_runtime_phase_locked(
                    _scene_engine_phase_from_log(line, runtime.phase_key)
                )
            if (output_root / "scene_export" / "scene_config.json").is_file():
                set_runtime_phase_locked("gym_export")
            runtime.status = PHASES[runtime.phase_key].label + "."
        yield _scene_engine_updates(output_root, preview_html)
        time.sleep(0.5)

    reader.join(timeout=1.0)
    with runtime_lock:
        if runtime.run_token != token or runtime.scene_engine_process is not process:
            return
        for line in _drain_output_queue(output_queue):
            runtime.log_lines.append(line)
            set_runtime_phase_locked(
                _scene_engine_phase_from_log(line, runtime.phase_key)
            )
        runtime.scene_engine_process = None

    scene_export = output_root / "scene_export" / "scene_config.json"
    if process.returncode != 0 or not scene_export.is_file():
        detail = (
            f"Scene Engine exited with code {process.returncode}."
            if process.returncode != 0
            else f"Scene Engine did not create {scene_export}."
        )
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked("failed")
            runtime.status = detail
            runtime.last_error = detail
        yield _scene_engine_updates(output_root, preview_html)
        return

    port = SCENE_ENGINE_VISER_PORT
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
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked("failed")
            runtime.status = f"Viser preview start failed: {exc}"
            runtime.last_error = str(exc)
        yield _scene_engine_updates(output_root, preview_html)
        return

    with runtime_lock:
        if runtime.run_token != token:
            terminate_process_group(preview_process)
            return
        runtime.scene_preview_process = preview_process
        runtime.log_lines.append("$ " + " ".join(preview_command))
        set_runtime_phase_locked("preview")
        runtime.status = "Starting Viser preview..."
    yield _scene_engine_updates(output_root, preview_html)

    if not _wait_for_viser(port, preview_process):
        terminate_process_group(preview_process)
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.scene_preview_process = None
            runtime.is_busy = False
            runtime.scene_engine_is_running = False
            set_runtime_phase_locked("failed")
            runtime.status = "Viser preview did not start."
            runtime.last_error = runtime.status
        yield _scene_engine_updates(output_root, preview_html)
        return

    preview_html = _viser_iframe(port, scene_hash)
    with runtime_lock:
        if (
            runtime.run_token != token
            or runtime.scene_preview_process is not preview_process
        ):
            terminate_process_group(preview_process)
            return
        runtime.is_busy = False
        runtime.scene_engine_is_running = False
        set_runtime_phase_locked("complete")
        runtime.status = "Scene generated successfully. Viser preview is ready."
        runtime.last_error = None
    yield _scene_engine_updates(output_root, preview_html)


def _action_scene_is_available() -> bool:
    return FAST_GYM_CONFIG.is_file() and AGENT_CONFIG.is_file()


def _action_agent_cli_is_available() -> bool:
    try:
        return importlib.util.find_spec(COMMANDS["agent"]["module"]) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def run_action_engine_from_current(task_text: str, robot_profile: str | None):
    """Launch DexSim for the existing ``current`` Gym scene."""
    task_text = (task_text or "").strip()
    with runtime_lock:
        if not task_text:
            failure = "Enter a task description first."
        elif not _action_scene_is_available():
            failure = "Current Gym scene/config is unavailable."
        elif runtime.is_busy or runtime.sim_process is not None:
            failure = "Another engine is already running."
        elif not _action_agent_cli_is_available():
            failure = "Action-agent CLI is unavailable in this environment."
        else:
            failure = None
            token = uuid.uuid4().hex
            runtime.run_token = token
            runtime.is_busy = True
            runtime.task_text = task_text
            runtime.status = "Starting DexSim action simulation..."
            runtime.last_error = None
            runtime.log_lines.clear()
            set_runtime_phase_locked("started")

        if failure is not None:
            runtime.status = failure
            runtime.last_error = failure

    if failure is None:
        error = _launch_current_simulation(token, robot_profile=robot_profile)
        if error:
            with runtime_lock:
                runtime.is_busy = False
                set_runtime_phase_locked("failed")
                runtime.status = error
                runtime.last_error = error
    return ui_snapshot()


def _launch_current_simulation(
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
        args=(token, process, output_queue, reader, started_at_ns),
        daemon=True,
    )

    with runtime_lock:
        if runtime.run_token != token:
            stale = True
        else:
            stale = False
            runtime.sim_process = process
            runtime.log_lines.append("$ " + " ".join(command))

    if stale:
        terminate_process_group(process)
        return None

    reader.start()
    monitor.start()
    return None


def _monitor_simulation(
    token: str,
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
    reader: threading.Thread,
    started_at_ns: int,
) -> None:
    while process.poll() is None:
        _append_simulation_logs(token, process, _drain_output_queue(output_queue))
        time.sleep(0.5)

    reader.join(timeout=1.0)
    _append_simulation_logs(token, process, _drain_output_queue(output_queue))
    display_video = latest_audience_output_video(min_mtime_ns=started_at_ns)

    with runtime_lock:
        if runtime.run_token != token or runtime.sim_process is not process:
            return
        runtime.sim_process = None
        runtime.is_busy = False
        runtime.video_path = display_video
        if process.returncode == 0:
            set_runtime_phase_locked("complete")
            runtime.status = "DexSim simulation finished successfully."
            runtime.last_error = None
            if display_video is None:
                runtime.log_lines.append("No simulation preview video was found.")
        else:
            set_runtime_phase_locked("failed")
            runtime.status = f"DexSim exited with return code {process.returncode}."
            runtime.last_error = runtime.status


def _append_simulation_logs(
    token: str,
    process: subprocess.Popen[str],
    lines: list[str],
) -> None:
    if not lines:
        return
    with runtime_lock:
        if runtime.run_token == token and runtime.sim_process is process:
            runtime.log_lines.extend(lines)


def ui_snapshot(extra_status: str | None = None):
    """Return the current Action-engine widget values."""
    with runtime_lock:
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
