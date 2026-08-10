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

"""Standalone SimReady asset-engine workflow used by the engine workspace.

The upstream SimReady CLI works on a directory, while Gradio uploads files.
This adapter creates an isolated directory for every run, keeps material
sidecars together with the mesh, and exposes GLB previews before and after
processing.  It deliberately has no DexSim dependency.
"""

from __future__ import annotations

import queue
import shutil
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import trimesh

from app_articraft import build_articraft_panel, cleanup_articraft_session
from app_config import GEN_SIM_ASSET_ROOT, SIMREADY_MESH_SUFFIXES
from app_processes import (
    SessionProcessRegistry,
    get_request_session_id,
    read_process_output,
    start_pipeline,
    terminate_process_group,
)

__all__ = [
    "build_asset_engine_panel",
    "cleanup_asset_engine_session",
    "prepare_asset_input_preview",
    "reset_simready_asset",
    "run_simready_asset",
]

_simready_runs = SessionProcessRegistry()
_SIMREADY_IDLE_STATUS = "**Status:** waiting for an asset."


def reset_simready_asset(
    request: gr.Request,
) -> tuple[None, str, None, None, None, str, str]:
    """Clear SimReady widgets and stop only the requesting session's run.

    Args:
        request: Gradio request for the browser session initiating Reset.

    Returns:
        Reset values for the SimReady panel widgets.
    """
    _simready_runs.reset(get_request_session_id(request))
    return None, "rigid_object", None, None, None, _SIMREADY_IDLE_STATUS, ""


def cleanup_asset_engine_session(request: gr.Request) -> None:
    """Stop Asset-engine subprocesses owned by a disconnected session.

    Args:
        request: Gradio request for the disconnecting browser session.
    """
    session_id = get_request_session_id(request)
    _simready_runs.reset(session_id)
    cleanup_articraft_session(session_id)


def _as_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    values: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
    paths: list[Path] = []
    for item in values:
        if isinstance(item, str):
            paths.append(Path(item))
        elif isinstance(item, dict) and item.get("path"):
            paths.append(Path(item["path"]))
    return [path for path in paths if path.is_file()]


def _mesh_path(paths: Iterable[Path]) -> Path:
    meshes = [path for path in paths if path.suffix.lower() in SIMREADY_MESH_SUFFIXES]
    if not meshes:
        supported = ", ".join(sorted(SIMREADY_MESH_SUFFIXES))
        raise ValueError(
            f"Upload one mesh file ({supported}) and optional material files."
        )
    return meshes[0]


def _safe_copy_uploads(upload_paths: list[Path], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    for index, source in enumerate(upload_paths):
        # Upload file names are untrusted. Keep only their basename and avoid
        # collisions without ever interpreting a supplied relative path.
        name = source.name or f"upload_{index}"
        target = destination / name
        if target.exists():
            target = destination / f"{target.stem}_{index}{target.suffix}"
        shutil.copy2(source, target)
        copied.append(target)
    return _mesh_path(copied)


def _export_preview(mesh_path: Path, destination: Path) -> Path:
    """Convert every supported mesh type to GLB for one consistent viewer."""
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        scene = loaded
    else:
        raise ValueError(f"Unsupported mesh payload: {type(loaded)!r}")
    if not scene.geometry:
        raise ValueError("The uploaded asset contains no renderable geometry.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination)
    return destination


def prepare_asset_input_preview(upload_value: Any):
    """Validate an upload and return a normalized GLB preview without running SimReady."""
    try:
        source = _mesh_path(_as_paths(upload_value))
        preview = GEN_SIM_ASSET_ROOT / "previews" / f"{uuid.uuid4().hex}.glb"
        _export_preview(source, preview)
        return (
            preview.as_posix(),
            "**Asset input ready.** Review the model, then run SimReady.",
        )
    except Exception as exc:
        return None, f"**Input error:** {exc}"


def _find_simready_output(output_root: Path) -> Path:
    candidates = sorted(
        output_root.rglob("asset_simready.glb"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            output_root.rglob("asset_simready.obj"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            "SimReady completed without asset_simready.glb or asset_simready.obj."
        )
    return candidates[0]


def run_simready_asset(
    upload_value: Any,
    category: str,
    request: gr.Request,
) -> Iterator[tuple[Any, ...]]:
    """Run one upstream SimReady job and stream concise subprocess progress.

    Args:
        upload_value: Gradio upload value containing the asset and sidecars.
        category: SimReady asset category.
        request: Gradio request identifying the owning browser session.

    Yields:
        Updated preview, output, status, and log values for the panel.
    """
    session_id = get_request_session_id(request)
    token = _simready_runs.begin(session_id)
    category = (category or "").strip()
    if not category:
        if _simready_runs.is_active(session_id, token):
            yield None, None, None, "**Input error:** enter an asset category.", ""
        return
    try:
        uploads = _as_paths(upload_value)
        _mesh_path(uploads)
        run_root = GEN_SIM_ASSET_ROOT / "runs" / uuid.uuid4().hex
        input_dir = run_root / "input"
        output_root = run_root / "output"
        source_mesh = _safe_copy_uploads(uploads, input_dir)
        input_preview = _export_preview(source_mesh, run_root / "input_preview.glb")
    except Exception as exc:
        if _simready_runs.is_active(session_id, token):
            yield None, None, None, f"**Input error:** {exc}", ""
        return

    command = [
        sys.executable,
        "-m",
        "embodichain.gen_sim.simready_pipeline.cli.start",
        "--input_dir",
        str(input_dir),
        "--output_root",
        str(output_root),
        "--category",
        category,
    ]
    log_lines = ["$ " + " ".join(command)]
    if not _simready_runs.is_active(session_id, token):
        return
    yield input_preview.as_posix(), None, None, "**SimReady is running…**", "\n".join(
        log_lines
    )

    try:
        process = start_pipeline(command, use_simready_llm=True)
    except Exception as exc:
        if _simready_runs.is_active(session_id, token):
            yield input_preview.as_posix(), None, None, f"**Pipeline start failed:** {exc}", "\n".join(
                log_lines
            )
        return

    if not _simready_runs.attach(session_id, token, process):
        terminate_process_group(process)
        return

    try:
        output_queue: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(
            target=read_process_output, args=(process, output_queue), daemon=True
        )
        reader.start()
        while process.poll() is None:
            if not _simready_runs.is_active(session_id, token, process):
                return
            try:
                while True:
                    log_lines.append(output_queue.get_nowait())
            except queue.Empty:
                pass
            # Keep the browser responsive while the Blender/LLM stages run.
            yield input_preview.as_posix(), None, None, "**SimReady is running…**", "\n".join(
                log_lines[-160:]
            )
            time.sleep(0.5)
        reader.join(timeout=1)
        try:
            while True:
                log_lines.append(output_queue.get_nowait())
        except queue.Empty:
            pass
        if not _simready_runs.is_active(session_id, token, process):
            return

        if process.returncode != 0:
            yield input_preview.as_posix(), None, None, f"**SimReady failed** (exit code {process.returncode}).", "\n".join(
                log_lines[-220:]
            )
            return
        try:
            result = _find_simready_output(output_root)
            preview = (
                result
                if result.suffix.lower() == ".glb"
                else _export_preview(result, run_root / "output_preview.glb")
            )
            yield input_preview.as_posix(), preview.as_posix(), result.as_posix(), "**SimReady completed.**", "\n".join(
                log_lines[-220:]
            )
        except Exception as exc:
            yield input_preview.as_posix(), None, None, f"**Output error:** {exc}", "\n".join(
                log_lines[-220:]
            )
    finally:
        _simready_runs.finish(session_id, token, process)


def build_asset_engine_panel() -> dict[str, Any]:
    """Create the Asset-engine panel and return its event endpoints."""
    with gr.Column(visible=True) as panel:
        gr.Markdown(
            "## Asset engine\nConvert an existing mesh with SimReady, or generate a new articulated asset through Articraft and Codex. DexSim is not started in this engine."
        )
        with gr.Tabs():
            with gr.Tab("SimReady"):
                with gr.Row():
                    uploads = gr.File(
                        label="3D asset and optional material files",
                        file_count="multiple",
                        type="filepath",
                        file_types=[
                            ".glb",
                            ".gltf",
                            ".obj",
                            ".ply",
                            ".stl",
                            ".mtl",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".bin",
                        ],
                    )
                    category = gr.Textbox(
                        label="Asset category",
                        value="rigid_object",
                        placeholder="e.g. cup, chair, bottle",
                    )
                with gr.Row():
                    input_model = gr.Model3D(
                        label="Input asset preview",
                        height=440,
                        clear_color=(0.94, 0.94, 0.94, 1.0),
                    )
                    output_model = gr.Model3D(
                        label="SimReady asset preview",
                        height=440,
                        clear_color=(0.94, 0.94, 0.94, 1.0),
                    )
                with gr.Row():
                    run_button = gr.Button("Run SimReady", variant="primary")
                    reset_button = gr.Button("Reset SimReady", variant="stop")
                    output_file = gr.File(
                        label="SimReady asset output", interactive=False
                    )
                status = gr.Markdown(_SIMREADY_IDLE_STATUS)
                log = gr.Textbox(label="Pipeline log", lines=10, interactive=False)
            with gr.Tab("Articulation"):
                build_articraft_panel()

    uploads.change(
        prepare_asset_input_preview,
        inputs=[uploads],
        outputs=[input_model, status],
        queue=False,
    )
    run_button.click(
        run_simready_asset,
        inputs=[uploads, category],
        outputs=[input_model, output_model, output_file, status, log],
    )
    reset_button.click(
        reset_simready_asset,
        outputs=[
            uploads,
            category,
            input_model,
            output_model,
            output_file,
            status,
            log,
        ],
        queue=False,
    )
    return {"panel": panel}
