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

import hashlib
import html
import io
import json
import importlib.util
import math
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from random_input import IMAGE_DIR as AUTO_BASE_IMAGE_DIR
from random_input import (
    auto_image_directories,
    available_auto_task_indices,
    generate_auto_scene_description,
    generate_auto_text_input,
    get_prebuilt_scene_dir,
    parse_task_id,
)
from app_config import *  # noqa: F403 - services intentionally consume central config.
from app_env import SCENE_ENGINE_VISER_PORT, configure_direct_network_env
from app_processes import (
    build_pipeline_env,
    build_run_agent_command,
    detect_phase_from_files,
    read_process_output,
    run_agent_cli_supports_robot_profile,
    start_pipeline,
    terminate_process_group,
    update_phase_from_log,
)
from app_media import *  # noqa: F403 - workflow consumes media service helpers.

configure_direct_network_env()

import gradio as gr
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps

_RUN_AGENT_SUPPORTS_ROBOT_PROFILE: bool | None = None
VIDEO_SYNC_JS = r"""
() => {
    const audienceRootId = "embodichain-audience-video";
    const lerobotRootId = "embodichain-lerobot-video";
    let syncing = false;

    function findVideo(rootId) {
        const root = document.getElementById(rootId);
        return root ? root.querySelector("video") : null;
    }

    function sourceLoaded(video) {
        return Boolean(video && (video.currentSrc || video.src));
    }

    function copyTime(source, target) {
        if (!sourceLoaded(source) || !sourceLoaded(target)) {
            return;
        }
        const sourceTime = source.currentTime || 0;
        if (!Number.isFinite(sourceTime)) {
            return;
        }
        const duration = Number.isFinite(target.duration) ? target.duration : sourceTime;
        const targetTime = Math.min(sourceTime, duration);
        if (Math.abs((target.currentTime || 0) - targetTime) > 0.35) {
            try {
                target.currentTime = targetTime;
            } catch (_) {
                // Some browsers reject seeking before metadata is fully available.
            }
        }
    }

    function syncPlayback(sourceRootId, targetRootId, shouldPlay) {
        if (syncing) {
            return;
        }
        const source = findVideo(sourceRootId);
        const target = findVideo(targetRootId);
        if (!sourceLoaded(source) || !sourceLoaded(target)) {
            return;
        }

        syncing = true;
        copyTime(source, target);

        const release = () => {
            window.setTimeout(() => {
                syncing = false;
            }, 0);
        };

        if (shouldPlay) {
            const result = target.play();
            if (result && typeof result.finally === "function") {
                result.catch(() => {}).finally(release);
            } else {
                release();
            }
        } else {
            target.pause();
            release();
        }
    }

    function bindOne(rootId, peerRootId) {
        const video = findVideo(rootId);
        if (!sourceLoaded(video) || video.dataset.embodichainSyncBound === "true") {
            return;
        }
        video.dataset.embodichainSyncBound = "true";
        video.addEventListener("play", () => syncPlayback(rootId, peerRootId, true));
        video.addEventListener("pause", () => syncPlayback(rootId, peerRootId, false));
    }

    function bindVideos() {
        bindOne(audienceRootId, lerobotRootId);
        bindOne(lerobotRootId, audienceRootId);
    }

    bindVideos();
    window.setInterval(bindVideos, 1000);
    const observer = new MutationObserver(bindVideos);
    observer.observe(document.body, { childList: true, subtree: true });
}
"""


# Runtime ownership lives in app_state; this module only orchestrates it.
from app_state import (
    PHASES,
    Phase,
    RuntimeState,
    clear_run_timing_locked,
    format_duration_ns,
    format_timing_lines,
    record_phase_transition_locked,
    record_simulation_finished_locked,
    record_simulation_started_locked,
    runtime,
    runtime_lock,
    set_runtime_phase_locked,
    snapshot_timing_locked,
    start_run_timing_locked,
)


@dataclass(frozen=True)
class ScenePaths:
    scene_id: str
    image_path: Path
    prompt_root: Path
    config_dir: Path

    @property
    def fast_gym_config(self) -> Path:
        return self.config_dir / "fast_gym_config.json"

    @property
    def agent_config(self) -> Path:
        return self.config_dir / "agent_config.json"

    @property
    def gradio_scene_dir(self) -> Path:
        return self.config_dir / "gradio_scene"

    @property
    def gradio_scene_glb(self) -> Path:
        return self.gradio_scene_dir / "scene_current.glb"

    @property
    def gradio_object_preview_glb(self) -> Path:
        return self.gradio_scene_dir / "object_preview.glb"

    @property
    def scene_manifest(self) -> Path:
        return self.gradio_scene_dir / "scene_manifest.json"

    @property
    def object_preview_manifest(self) -> Path:
        return self.gradio_scene_dir / "object_preview_manifest.json"


CURRENT_PATHS = ScenePaths(
    scene_id=SCENE_ID,
    image_path=IMAGE_PATH,
    prompt_root=PROMPT2SCENE_ROOT,
    config_dir=CONFIG_DIR,
)


def make_stage_paths(run_token: str) -> ScenePaths:
    scene_id = f"{PENDING_PREFIX}{run_token[:12]}"
    return ScenePaths(
        scene_id=scene_id,
        image_path=IMAGE_DIR / f"{scene_id}.png",
        prompt_root=GYM_PROJECT_ROOT / scene_id,
        config_dir=ACTION_AGENT_ROOT / "configs" / scene_id,
    )


def make_replaced_paths(run_token: str) -> ScenePaths:
    scene_id = f"{REPLACED_PREFIX}{run_token[:12]}"
    return ScenePaths(
        scene_id=scene_id,
        image_path=IMAGE_DIR / f"{scene_id}.png",
        prompt_root=GYM_PROJECT_ROOT / scene_id,
        config_dir=ACTION_AGENT_ROOT / "configs" / scene_id,
    )


def save_input(
    image_value: str | np.ndarray | Image.Image,
    task_text: str,
    image_path: Path,
) -> Path:
    if image_value is None:
        raise ValueError("Please upload an image first.")
    if not task_text.strip():
        raise ValueError("Please enter a task description.")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, str):
        image = Image.open(image_value)
    elif isinstance(image_value, np.ndarray):
        image = Image.fromarray(image_value)
    elif isinstance(image_value, Image.Image):
        image = image_value
    else:
        raise TypeError(f"Unsupported image input type: {type(image_value)!r}")

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.save(image_path, format="PNG")
    return image_path


def reset_current_scene() -> list[str]:
    process: subprocess.Popen[str] | None = None
    sim_process: subprocess.Popen[str] | None = None
    with runtime_lock:
        runtime.run_token = uuid.uuid4().hex
        runtime.auto_loop_active = False
        runtime.auto_loop_token = None
        runtime.auto_round = 0
        process = runtime.process
        sim_process = runtime.sim_process
        runtime.process = None
        runtime.sim_process = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.is_busy = False
        runtime.phase_key = "idle"
        runtime.status = "Idle."
        runtime.task_text = ""
        runtime.input_task_text = ""
        runtime.input_scene_text = ""
        runtime.image_path = None
        runtime.video_path = None
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.object_model_path = None
        runtime.scene_model_path = None
        runtime.edited_scene_model_path = None
        runtime.last_error = None
        runtime.log_lines.clear()
        clear_run_timing_locked()

    if process is not None:
        terminate_process_group(process)
    if sim_process is not None:
        terminate_process_group(sim_process)

    return cleanup_current_and_staging()


def cleanup_current_and_staging() -> list[str]:
    errors: list[str] = []
    paths: list[Path] = [
        PROMPT2SCENE_ROOT,
        CONFIG_DIR,
        IMAGE_PATH,
        *pending_artifact_paths(),
    ]
    for path in paths:
        errors.extend(remove_path(path))
    errors.extend(cleanup_outputs_preserving_videos())
    return errors


def cleanup_auto_generated_artifacts(extra_image_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    paths: list[Path] = [
        PROMPT2SCENE_ROOT,
        CONFIG_DIR,
        IMAGE_PATH,
        *pending_artifact_paths(),
    ]
    if extra_image_path is not None:
        paths.append(extra_image_path)

    for path in paths:
        if is_protected_auto_base_image(path):
            continue
        errors.extend(remove_path(path))

    with runtime_lock:
        runtime.image_path = None
        runtime.input_task_text = ""
        runtime.input_scene_text = ""
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.object_model_path = None
        runtime.scene_model_path = None
        runtime.edited_scene_model_path = None
    errors.extend(cleanup_outputs_preserving_videos())
    return errors


def is_protected_auto_base_image(path: Path) -> bool:
    try:
        path.resolve().relative_to(AUTO_BASE_IMAGE_DIR.resolve())
    except ValueError:
        return False
    except FileNotFoundError:
        return False
    return True


def pending_artifact_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (GYM_PROJECT_ROOT, ACTION_AGENT_ROOT / "configs"):
        if root.is_dir():
            paths.extend(root.glob(f"{PENDING_PREFIX}*"))
            paths.extend(root.glob(f"{REPLACED_PREFIX}*"))
    if IMAGE_DIR.is_dir():
        paths.extend(IMAGE_DIR.glob(f"{PENDING_PREFIX}*.png"))
        paths.extend(IMAGE_DIR.glob(f"{REPLACED_PREFIX}*.png"))
    return paths


def remove_path(path: Path) -> list[str]:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except Exception as exc:
        return [f"Failed to remove {path}: {exc}"]
    return []


def cleanup_outputs_preserving_videos() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    if OUTPUTS_DIR.is_file():
        if OUTPUTS_DIR.suffix.lower() in VIDEO_SUFFIXES:
            return []
        return remove_path(OUTPUTS_DIR)

    errors: list[str] = []
    for path in sorted(
        OUTPUTS_DIR.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_file() and path.suffix.lower() not in VIDEO_SUFFIXES:
            errors.extend(remove_path(path))

    for path in sorted(
        OUTPUTS_DIR.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            pass
        except Exception as exc:
            errors.append(f"Failed to remove empty output directory {path}: {exc}")
    return errors


from app_commands import (
    build_config_command_for_paths,
    build_initial_pipeline_command,
    build_scene_edit_pipeline_command,
    robot_profile_cli_value,
)


def build_edit_pipeline_command(
    task_text: str,
    env_text: str,
    robot_profile: str | None = None,
    load_template_material: bool = False,
) -> list[str]:
    return build_scene_edit_pipeline_command(
        task_text, env_text, CURRENT_PATHS, robot_profile, load_template_material
    )


def build_task_only_config_command(
    task_text: str,
    robot_profile: str | None = None,
    load_template_material: bool = False,
) -> list[str]:
    return build_config_command_for_paths(
        task_text, CURRENT_PATHS, robot_profile, load_template_material
    )


def format_current_task(task_text: str, env_text: str = "") -> str:
    return "\n".join(
        part for part in ((task_text or "").strip(), (env_text or "").strip()) if part
    )


def build_gradio_scene_from_fast_config(
    config_path: Path,
    scene_dir: Path | None = None,
) -> Path:
    config_dir = config_path.parent
    if scene_dir is None:
        scene_dir = config_dir / "gradio_scene"
    scene_glb = scene_dir / "scene_current.glb"
    scene_manifest = scene_dir / "scene_manifest.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config_stat = config_path.stat()

    scene = trimesh.Scene()
    manifest: dict[str, Any] = {
        "source_config": os.path.relpath(config_path, scene_dir),
        "source_config_size": config_stat.st_size,
        "source_config_mtime_ns": config_stat.st_mtime_ns,
        "transform_policy": GRADIO_SCENE_TRANSFORM_POLICY,
        "objects": [],
    }

    object_count = 0
    for role, obj in iter_scene_objects(config):
        shape = obj.get("shape") if isinstance(obj, dict) else None
        if not isinstance(shape, dict) or shape.get("shape_type") != "Mesh":
            continue
        raw_fpath = shape.get("fpath")
        if not raw_fpath:
            continue
        mesh_path = resolve_mesh_path(config_dir, str(raw_fpath))
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Mesh file not found for {obj.get('uid')}: {mesh_path}"
            )

        transform = object_transform(obj)
        frame_transform = gltf_to_sim_frame_transform(mesh_path)
        if frame_transform is not None:
            transform = transform @ frame_transform
        add_mesh_to_scene(scene, mesh_path, transform, str(obj.get("uid", "object")))
        manifest["objects"].append(
            {
                "uid": obj.get("uid"),
                "role": role,
                "source_mesh": os.path.relpath(mesh_path, scene_dir),
                "source_mesh_size": mesh_path.stat().st_size,
                "source_mesh_mtime_ns": mesh_path.stat().st_mtime_ns,
                "gltf_to_sim_frame": frame_transform is not None,
            }
        )
        object_count += 1

    if object_count == 0:
        raise ValueError(f"No mesh objects found in {config_path}")

    scene_dir.mkdir(parents=True, exist_ok=True)
    scene.export(scene_glb)
    scene_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return scene_glb


def gradio_scene_is_current(
    scene_glb: Path,
    manifest_path: Path,
    config_path: Path,
) -> bool:
    if (
        not scene_glb.is_file()
        or not manifest_path.is_file()
        or not config_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_stat = config_path.stat()
    except Exception:
        return False

    if manifest.get("source_config") != os.path.relpath(
        config_path, manifest_path.parent
    ):
        return False
    if manifest.get("source_config_size") != config_stat.st_size:
        return False
    if manifest.get("source_config_mtime_ns") != config_stat.st_mtime_ns:
        return False
    if manifest.get("transform_policy") != GRADIO_SCENE_TRANSFORM_POLICY:
        return False

    expected_objects = []
    try:
        for role, obj in iter_scene_objects(config):
            shape = obj.get("shape") if isinstance(obj, dict) else None
            if not isinstance(shape, dict) or shape.get("shape_type") != "Mesh":
                continue
            raw_fpath = shape.get("fpath")
            if not raw_fpath:
                continue
            mesh_path = resolve_mesh_path(config_path.parent, str(raw_fpath))
            mesh_stat = mesh_path.stat()
            frame_transform = gltf_to_sim_frame_transform(mesh_path)
            expected_objects.append(
                {
                    "uid": obj.get("uid"),
                    "role": role,
                    "source_mesh": os.path.relpath(mesh_path, manifest_path.parent),
                    "source_mesh_size": mesh_stat.st_size,
                    "source_mesh_mtime_ns": mesh_stat.st_mtime_ns,
                    "gltf_to_sim_frame": frame_transform is not None,
                }
            )
    except OSError:
        return False
    return manifest.get("objects") == expected_objects


def collect_generated_object_glbs(paths: ScenePaths) -> list[Path]:
    if not paths.prompt_root.is_dir():
        return []

    glb_paths: list[Path] = []
    seen: set[Path] = set()
    for glb_dir in sorted(paths.prompt_root.rglob("glb_gen")):
        if not glb_dir.is_dir():
            continue
        candidates = [
            path for path in glb_dir.rglob("*_simready.glb") if is_previewable_glb(path)
        ]
        if not candidates:
            candidates = [
                path for path in glb_dir.rglob("*.glb") if is_previewable_glb(path)
            ]
        for path in sorted(candidates):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            glb_paths.append(path)
    return glb_paths


def is_previewable_glb(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    return not any(part.startswith(".") for part in path.relative_to(path.anchor).parts)


def build_object_preview_scene(
    glb_paths: list[Path],
    scene_dir: Path,
) -> Path:
    if not glb_paths:
        raise ValueError("No generated object GLBs found")

    scene_dir.mkdir(parents=True, exist_ok=True)
    preview_glb = scene_dir / "object_preview.glb"
    preview_manifest = scene_dir / "object_preview_manifest.json"
    scene = trimesh.Scene()
    manifest: dict[str, Any] = {"objects": []}

    cursor = 0.0
    spacing = 0.35
    added_count = 0
    for object_index, mesh_path in enumerate(glb_paths):
        meshes = load_mesh_geometries(mesh_path)
        if not meshes:
            continue

        bounds = combined_bounds(meshes)
        extents = bounds[1] - bounds[0]
        max_extent = float(max(extents.max(), 1e-6))
        scale = 1.0 / max_extent
        scaled_width = max(float(extents[0]) * scale, 0.2)
        placement_x = cursor + scaled_width / 2.0
        cursor += scaled_width + spacing

        transform = (
            trimesh.transformations.translation_matrix(
                [
                    placement_x,
                    0.0,
                    0.0,
                ]
            )
            @ trimesh.transformations.scale_matrix(scale)
            @ trimesh.transformations.translation_matrix(
                [
                    -float((bounds[0][0] + bounds[1][0]) / 2.0),
                    -float((bounds[0][1] + bounds[1][1]) / 2.0),
                    -float(bounds[0][2]),
                ]
            )
        )

        for mesh_index, mesh in enumerate(meshes):
            mesh.apply_transform(transform)
            name = f"object_{object_index}_{mesh_index}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            added_count += 1

        manifest["objects"].append(
            {
                "source_mesh": os.path.relpath(mesh_path, scene_dir),
                "size": mesh_path.stat().st_size,
                "mtime_ns": mesh_path.stat().st_mtime_ns,
            }
        )

    if added_count == 0:
        raise ValueError("No renderable meshes found in generated object GLBs")

    if cursor > spacing:
        scene.apply_transform(
            trimesh.transformations.translation_matrix(
                [-(cursor - spacing) / 2.0, 0.0, 0.0]
            )
        )
    scene.export(preview_glb)
    preview_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return preview_glb


def object_preview_is_current(
    manifest_path: Path,
    glb_paths: list[Path],
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = []
    for path in glb_paths:
        try:
            stat = path.stat()
        except OSError:
            return False
        expected.append(
            {
                "source_mesh": os.path.relpath(path, manifest_path.parent),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return manifest.get("objects") == expected


def load_mesh_geometries(mesh_path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    gltf_to_sim_transform = gltf_to_sim_frame_transform(mesh_path)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        if gltf_to_sim_transform is not None:
            mesh.apply_transform(gltf_to_sim_transform)
        return [mesh]
    if isinstance(loaded, trimesh.Scene):
        meshes: list[trimesh.Trimesh] = []
        for geometry in loaded.dump(concatenate=False):
            if isinstance(geometry, trimesh.Trimesh):
                mesh = geometry.copy()
                if gltf_to_sim_transform is not None:
                    mesh.apply_transform(gltf_to_sim_transform)
                meshes.append(mesh)
        return meshes
    raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(loaded)!r}")


def gltf_to_sim_frame_transform(mesh_path: Path) -> np.ndarray | None:
    if mesh_path.suffix.lower() not in {".glb", ".gltf"}:
        return None
    # Match DexSim's native GLTF Y-up to simulation Z-up conversion.
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return transform


def combined_bounds(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    valid_bounds = [
        mesh.bounds
        for mesh in meshes
        if mesh.vertices is not None and len(mesh.vertices) > 0
    ]
    if not valid_bounds:
        raise ValueError("Mesh has no vertices")
    bounds = np.asarray(valid_bounds, dtype=float)
    return np.stack([bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)])


def iter_scene_objects(config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for role in ("background", "rigid_object"):
        value = config.get(role, [])
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            continue
        for obj in value:
            if isinstance(obj, dict):
                yield role, obj


def resolve_mesh_path(config_dir: Path, raw_fpath: str) -> Path:
    mesh_path = Path(raw_fpath).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = config_dir / mesh_path
    return mesh_path.resolve()


def object_transform(obj: dict[str, Any]) -> np.ndarray:
    scale = vector3(obj.get("body_scale"), [1.0, 1.0, 1.0])

    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]

    init_local_pose = matrix4(obj.get("init_local_pose"))
    if init_local_pose is not None:
        return init_local_pose @ scale_matrix

    position = vector3(obj.get("init_pos"), [0.0, 0.0, 0.0])
    rotation_degrees = vector3(obj.get("init_rot"), [0.0, 0.0, 0.0])
    root_matrix = euler_xyz_degrees_matrix(rotation_degrees, position)
    return root_matrix @ scale_matrix


def euler_xyz_degrees_matrix(
    rotation_degrees: list[float],
    position: list[float],
) -> np.ndarray:
    rx, ry, rz = (math.radians(value) for value in rotation_degrees)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cx, -sx, 0.0],
            [0.0, sx, cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_y = np.array(
        [
            [cy, 0.0, sy, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sy, 0.0, cy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_z = np.array(
        [
            [cz, -sz, 0.0, 0.0],
            [sz, cz, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    matrix = rot_x @ rot_y @ rot_z
    matrix[:3, 3] = position
    return matrix


def matrix4(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def vector3(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return list(default)
    return [float(value[0]), float(value[1]), float(value[2])]


def add_mesh_to_scene(
    scene: trimesh.Scene,
    mesh_path: Path,
    transform: np.ndarray,
    uid: str,
) -> None:
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded.apply_transform(transform)
        scene.add_geometry(loaded, node_name=uid, geom_name=uid)
        return

    if isinstance(loaded, trimesh.Scene):
        loaded.apply_transform(transform)
        for index, geometry in enumerate(loaded.dump(concatenate=False)):
            if isinstance(geometry, trimesh.Trimesh):
                scene.add_geometry(
                    geometry,
                    node_name=f"{uid}_{index}",
                    geom_name=f"{uid}_{index}",
                )
        return

    raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(loaded)!r}")


def promote_stage_to_current(stage: ScenePaths, run_token: str) -> list[str]:
    backup = make_replaced_paths(run_token)
    promotion_errors: list[str] = []
    cleanup_errors: list[str] = []

    for required_path in (stage.prompt_root, stage.config_dir, stage.image_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Generated artifact missing: {required_path}")

    cleanup_errors.extend(remove_path(backup.prompt_root))
    cleanup_errors.extend(remove_path(backup.config_dir))
    cleanup_errors.extend(remove_path(backup.image_path))

    moved_to_backup: list[tuple[Path, Path]] = []
    moved_to_current: list[tuple[Path, Path]] = []
    try:
        move_if_exists(PROMPT2SCENE_ROOT, backup.prompt_root, moved_to_backup)
        move_if_exists(CONFIG_DIR, backup.config_dir, moved_to_backup)
        move_if_exists(IMAGE_PATH, backup.image_path, moved_to_backup)

        move_required(stage.prompt_root, PROMPT2SCENE_ROOT, moved_to_current)
        move_required(stage.config_dir, CONFIG_DIR, moved_to_current)
        move_required(stage.image_path, IMAGE_PATH, moved_to_current)
        rewrite_promoted_paths(stage)
    except Exception as exc:
        promotion_errors.append(f"Failed to promote generated scene: {exc}")
        restore_promoted_paths(moved_to_current, moved_to_backup, promotion_errors)
        raise RuntimeError("\n".join(promotion_errors)) from exc

    cleanup_errors.extend(remove_path(backup.prompt_root))
    cleanup_errors.extend(remove_path(backup.config_dir))
    cleanup_errors.extend(remove_path(backup.image_path))
    return cleanup_errors


def move_if_exists(src: Path, dst: Path, moved: list[tuple[Path, Path]]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    moved.append((src, dst))


def move_required(src: Path, dst: Path, moved: list[tuple[Path, Path]]) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    moved.append((dst, src))


def restore_promoted_paths(
    moved_to_current: list[tuple[Path, Path]],
    moved_to_backup: list[tuple[Path, Path]],
    errors: list[str],
) -> None:
    for current_path, original_stage_path in reversed(moved_to_current):
        try:
            if current_path.exists():
                original_stage_path.parent.mkdir(parents=True, exist_ok=True)
                current_path.rename(original_stage_path)
        except Exception as exc:
            errors.append(
                f"Failed to restore staging artifact {original_stage_path}: {exc}"
            )

    for original_current_path, backup_path in reversed(moved_to_backup):
        try:
            if backup_path.exists() and not original_current_path.exists():
                original_current_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.rename(original_current_path)
        except Exception as exc:
            errors.append(
                f"Failed to restore previous scene {original_current_path}: {exc}"
            )


def rewrite_promoted_paths(stage: ScenePaths) -> None:
    replacements = [
        (str(stage.config_dir), str(CONFIG_DIR)),
        (str(stage.prompt_root), str(PROMPT2SCENE_ROOT)),
        (str(stage.image_path), str(IMAGE_PATH)),
        (stage.scene_id, SCENE_ID),
    ]
    for root in (PROMPT2SCENE_ROOT, CONFIG_DIR):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_REWRITE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            new_text = text
            for old, new in replacements:
                new_text = new_text.replace(old, new)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")


def ensure_initial_scene_snapshot(*, overwrite: bool = False) -> Path:
    if not GRADIO_SCENE_GLB.is_file():
        build_gradio_scene_from_fast_config(FAST_GYM_CONFIG, GRADIO_SCENE_DIR)
    if overwrite or not GRADIO_INITIAL_SCENE_GLB.is_file():
        GRADIO_SCENE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GRADIO_SCENE_GLB, GRADIO_INITIAL_SCENE_GLB)
    return GRADIO_INITIAL_SCENE_GLB


def prepare_current_scene_for_edit() -> Path:
    scene_state = PROMPT2SCENE_ROOT / "gym_export" / "scene_state" / "result.json"
    if not scene_state.is_file():
        raise FileNotFoundError(
            f"Current prompt2scene scene state not found: {scene_state}"
        )
    if not FAST_GYM_CONFIG.is_file():
        raise FileNotFoundError(f"Current gym config not found: {FAST_GYM_CONFIG}")

    initial_scene_path = ensure_initial_scene_snapshot()
    errors = remove_path(GRADIO_SCENE_GLB)
    errors.extend(remove_path(SCENE_MANIFEST))
    if errors:
        raise RuntimeError("\n".join(errors))
    return initial_scene_path


def prebuilt_scene_dir_for_image_value(
    image_value: str | np.ndarray | Image.Image,
) -> Path | None:
    if not isinstance(image_value, str):
        return None
    image_path = Path(image_value).expanduser()
    task_index = parse_task_id(image_path.name)
    if task_index is None:
        return None

    try:
        resolved_image = image_path.resolve()
    except FileNotFoundError:
        return None
    filename = image_path.name
    matches_auto_image = False
    for image_dir in auto_image_directories():
        candidate = image_dir / filename
        if not candidate.is_file():
            continue
        try:
            if candidate.resolve() == resolved_image:
                matches_auto_image = True
                break
        except FileNotFoundError:
            continue
    if not matches_auto_image:
        return None

    scene_dir = get_prebuilt_scene_dir(task_index)
    return scene_dir if scene_dir.is_dir() else None


def copy_prebuilt_scene_to_stage(prebuilt_scene_dir: Path, stage: ScenePaths) -> None:
    required_paths = [
        prebuilt_scene_dir / "gym_export" / "gym_config.json",
        prebuilt_scene_dir / "gym_export" / "scene_state" / "result.json",
        prebuilt_scene_dir / "gym_export" / "scene_state" / "unified_scene.json",
        prebuilt_scene_dir / "gym_export" / "scene_state" / "unified_scene_gen.json",
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Prebuilt scene is incomplete: {missing_text}")

    cleanup_errors = []
    cleanup_errors.extend(remove_path(stage.prompt_root))
    cleanup_errors.extend(remove_path(stage.config_dir))
    if cleanup_errors:
        raise RuntimeError("\n".join(cleanup_errors))
    stage.prompt_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prebuilt_scene_dir, stage.prompt_root)


def build_interact_random_initial_preview(prebuilt_scene_dir: Path) -> Path:
    scene_id = prebuilt_scene_dir.name
    preview_dir = INTERACT_RANDOM_PREVIEW_DIR / scene_id
    config_path = prebuilt_scene_dir / "gym_export" / "gym_config.json"
    return build_gradio_scene_from_fast_config(config_path, preview_dir)


def current_scene_available_for_task_only() -> bool:
    return CURRENT_GYM_EXPORT_CONFIG.is_file()


def rerun_simulation_is_available() -> bool:
    return (
        CURRENT_PATHS.fast_gym_config.is_file() and CURRENT_PATHS.agent_config.is_file()
    )


def run_generate(
    image_value: str | np.ndarray | Image.Image,
    task_text: str,
    env_text: str,
    *,
    force_initial: bool = False,
    scene_mode: str = SCENE_MODE_INITIAL,
    parallel_env: bool = False,
    robot_profile: str | None = None,
    load_template_material: bool = False,
    run_log_mode: str = RUN_LOG_MODE_INTERACT,
    prebuilt_scene_dir: Path | None = None,
    launch_simulation: bool = True,
):
    task_text = (task_text or "").strip()
    env_text = (env_text or "").strip()
    if force_initial or scene_mode == SCENE_MODE_INITIAL:
        mode = PIPELINE_MODE_INITIAL
    elif scene_mode == SCENE_MODE_EDIT:
        mode = PIPELINE_MODE_EDIT
    elif scene_mode == SCENE_MODE_TASK_ONLY:
        mode = PIPELINE_MODE_TASK_ONLY
    else:
        raise ValueError(f"Unsupported scene mode: {scene_mode}")
    requested_mode = mode
    if requested_mode == PIPELINE_MODE_TASK_ONLY:
        env_text = ""
    resolved_prebuilt_scene_dir = (
        prebuilt_scene_dir or prebuilt_scene_dir_for_image_value(image_value)
    )
    use_prebuilt_scene = resolved_prebuilt_scene_dir is not None
    supervisor_mode = PIPELINE_MODE_INITIAL if use_prebuilt_scene else mode
    old_sim_process: subprocess.Popen[str] | None = None
    with runtime_lock:
        if runtime.is_busy:
            yield ui_snapshot(extra_status="A pipeline run is already in progress.")
            return
        old_sim_process = runtime.sim_process
        runtime.sim_process = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None

    if old_sim_process is not None:
        terminate_process_group(old_sim_process)

    token = uuid.uuid4().hex
    stage = (
        CURRENT_PATHS
        if supervisor_mode in {PIPELINE_MODE_EDIT, PIPELINE_MODE_TASK_ONLY}
        else make_stage_paths(token)
    )
    initial_scene_path: Path | None = None
    existing_object_preview_path = (
        GRADIO_OBJECT_PREVIEW_GLB
        if supervisor_mode in {PIPELINE_MODE_EDIT, PIPELINE_MODE_TASK_ONLY}
        and GRADIO_OBJECT_PREVIEW_GLB.is_file()
        else None
    )
    prebuilt_initial_scene_dir: Path | None = None
    try:
        if use_prebuilt_scene:
            if not task_text:
                raise ValueError("Please enter a task description.")
            if requested_mode == PIPELINE_MODE_EDIT and not env_text:
                raise ValueError("Please enter a scene description to edit.")
            image_path = save_input(image_value, task_text, stage.image_path)
            prebuilt_initial_scene_dir = resolved_prebuilt_scene_dir
            copy_prebuilt_scene_to_stage(prebuilt_initial_scene_dir, stage)
            initial_scene_path = build_interact_random_initial_preview(
                prebuilt_initial_scene_dir
            )
        elif mode == PIPELINE_MODE_EDIT:
            if not task_text:
                raise ValueError("Please enter a task description.")
            if not env_text:
                raise ValueError("Please enter a scene description to edit.")
            initial_scene_path = prepare_current_scene_for_edit()
            image_path = IMAGE_PATH if IMAGE_PATH.is_file() else None
        elif mode == PIPELINE_MODE_TASK_ONLY:
            if not task_text:
                raise ValueError("Please enter a task description.")
            if not CURRENT_GYM_EXPORT_CONFIG.is_file():
                raise FileNotFoundError(
                    f"Current gym export not found: {CURRENT_GYM_EXPORT_CONFIG}"
                )
            if GRADIO_INITIAL_SCENE_GLB.is_file():
                initial_scene_path = GRADIO_INITIAL_SCENE_GLB
            elif GRADIO_SCENE_GLB.is_file():
                initial_scene_path = GRADIO_SCENE_GLB
            image_path = IMAGE_PATH if IMAGE_PATH.is_file() else None
        else:
            image_path = save_input(image_value, task_text, stage.image_path)
    except Exception as exc:
        with runtime_lock:
            runtime.phase_key = "failed"
            runtime.status = f"Input error: {exc}"
            runtime.last_error = str(exc)
            runtime.log_lines.clear()
            clear_run_timing_locked()
            runtime.log_lines.append(runtime.status)
        if run_log_mode == RUN_LOG_MODE_INTERACT:
            archive_run_log(
                mode=RUN_LOG_MODE_INTERACT,
                task_description=task_text,
                scene_description=env_text,
                outcome="input_error",
            )
        yield ui_snapshot()
        return

    should_edit_prebuilt_scene = (
        prebuilt_initial_scene_dir is not None
        and requested_mode != PIPELINE_MODE_TASK_ONLY
        and bool(env_text)
    )
    if mode == PIPELINE_MODE_EDIT and prebuilt_initial_scene_dir is None:
        command = build_edit_pipeline_command(
            task_text,
            env_text,
            robot_profile,
            load_template_material,
        )
    elif mode == PIPELINE_MODE_TASK_ONLY and prebuilt_initial_scene_dir is None:
        command = build_task_only_config_command(
            task_text,
            robot_profile,
            load_template_material,
        )
    elif should_edit_prebuilt_scene:
        command = build_scene_edit_pipeline_command(
            task_text,
            env_text,
            stage,
            robot_profile,
            load_template_material,
        )
    elif prebuilt_initial_scene_dir is not None:
        command = build_config_command_for_paths(
            task_text,
            stage,
            robot_profile,
            load_template_material,
        )
    else:
        command = build_initial_pipeline_command(
            task_text,
            stage,
            env_text,
            robot_profile,
            load_template_material,
        )
    display_task_text = format_current_task(task_text, env_text)
    with runtime_lock:
        runtime.run_token = token
        runtime.is_busy = True
        runtime.phase_key = "received"
        if mode == PIPELINE_MODE_EDIT:
            runtime.status = "Starting scene edit..."
        elif should_edit_prebuilt_scene:
            runtime.status = "Prebuilt scene loaded. Starting scene edit..."
        elif mode == PIPELINE_MODE_TASK_ONLY:
            runtime.status = "Current scene found. Regenerating action config only..."
        else:
            runtime.status = "Input saved. Starting local pipeline..."
        runtime.task_text = display_task_text
        runtime.input_task_text = task_text
        runtime.input_scene_text = env_text
        runtime.image_path = image_path
        runtime.submitted_input_revision += 1
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.object_model_path = existing_object_preview_path
        runtime.scene_model_path = initial_scene_path
        runtime.edited_scene_model_path = None
        runtime.last_error = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.log_lines.clear()
        clear_run_timing_locked()
        runtime.log_lines.append("$ " + " ".join(command))
    yield ui_snapshot()

    try:
        process = start_pipeline(command)
    except Exception as exc:
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.is_busy = False
            runtime.process = None
            runtime.phase_key = "failed"
            runtime.status = f"Pipeline start failed: {exc}"
            runtime.last_error = str(exc)
        if run_log_mode == RUN_LOG_MODE_INTERACT:
            archive_run_log(
                mode=RUN_LOG_MODE_INTERACT,
                task_description=task_text,
                scene_description=env_text,
                outcome="pipeline_start_failed",
            )
        yield ui_snapshot()
        return

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue),
        daemon=True,
    )
    supervisor = threading.Thread(
        target=supervise_pipeline,
        args=(
            token,
            stage,
            supervisor_mode,
            process,
            display_task_text,
            task_text,
            env_text,
            output_queue,
            reader,
            parallel_env,
            robot_profile,
            run_log_mode,
            initial_scene_path,
            should_edit_prebuilt_scene,
            launch_simulation,
        ),
        daemon=True,
    )

    with runtime_lock:
        if runtime.run_token != token:
            terminate_process_group(process)
            return
        runtime.process = process
        start_run_timing_locked("started")
        runtime.phase_key = "started"
        runtime.status = "Local pipeline started."
    reader.start()
    supervisor.start()
    yield ui_snapshot()

    while True:
        with runtime_lock:
            still_current = runtime.run_token == token
            busy = runtime.is_busy
        if not still_current or not busy:
            break
        time.sleep(1.0)
        yield ui_snapshot()
    yield ui_snapshot()


def start_auto_loop_state() -> str | None:
    process: subprocess.Popen[str] | None = None
    sim_process: subprocess.Popen[str] | None = None
    with runtime_lock:
        if runtime.auto_loop_active or runtime.is_busy:
            runtime.status = "A pipeline run is already in progress."
            return None

        available_tasks = available_auto_task_indices()
        if not available_tasks:
            image_dirs = ", ".join(str(path) for path in auto_image_directories())
            message = (
                "Auto cannot start: no task input images were found. "
                "Add task1_0.png through task5_3.png to one of: "
                f"{image_dirs}"
            )
            runtime.phase_key = "failed"
            runtime.status = message
            runtime.last_error = message
            runtime.log_lines.clear()
            clear_run_timing_locked()
            runtime.log_lines.append(message)
            return None

        token = uuid.uuid4().hex
        process = runtime.process
        sim_process = runtime.sim_process
        runtime.process = None
        runtime.sim_process = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.auto_loop_active = True
        runtime.auto_loop_token = token
        runtime.auto_round = 0
        runtime.auto_scene_mode = SCENE_MODE_INITIAL
        runtime.auto_parallel_env = False
        runtime.auto_robot_profile = DEFAULT_ROBOT_PROFILE
        runtime.phase_key = "received"
        runtime.status = "Auto loop starting."
        runtime.video_path = None
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.last_error = None
        runtime.log_lines.clear()
        clear_run_timing_locked()

    if process is not None:
        terminate_process_group(process)
    if sim_process is not None:
        terminate_process_group(sim_process)
    return token


def auto_loop_is_active(loop_token: str) -> bool:
    with runtime_lock:
        return runtime.auto_loop_active and runtime.auto_loop_token == loop_token


def finish_auto_loop(loop_token: str, status_text: str | None = None) -> None:
    with runtime_lock:
        if runtime.auto_loop_token != loop_token:
            return
        runtime.auto_loop_active = False
        runtime.auto_loop_token = None
        runtime.auto_round = 0
        runtime.auto_scene_mode = SCENE_MODE_INITIAL
        runtime.auto_parallel_env = False
        runtime.auto_robot_profile = DEFAULT_ROBOT_PROFILE
        if status_text is not None:
            runtime.status = status_text


def stop_auto_loop_if_running() -> bool:
    process: subprocess.Popen[str] | None = None
    sim_process: subprocess.Popen[str] | None = None
    with runtime_lock:
        if not runtime.auto_loop_active:
            return False
        runtime.run_token = uuid.uuid4().hex
        runtime.auto_loop_active = False
        runtime.auto_loop_token = None
        runtime.auto_round = 0
        runtime.auto_scene_mode = SCENE_MODE_INITIAL
        runtime.auto_parallel_env = False
        runtime.auto_robot_profile = DEFAULT_ROBOT_PROFILE
        process = runtime.process
        sim_process = runtime.sim_process
        runtime.process = None
        runtime.sim_process = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.is_busy = False
        runtime.phase_key = "idle"
        runtime.status = "Stopped."
        runtime.task_text = ""
        runtime.input_task_text = ""
        runtime.input_scene_text = ""
        runtime.image_path = None
        runtime.video_path = None
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.object_model_path = None
        runtime.scene_model_path = None
        runtime.edited_scene_model_path = None
        runtime.last_error = None
        runtime.log_lines.clear()
        clear_run_timing_locked()

    if process is not None:
        terminate_process_group(process)
    if sim_process is not None:
        terminate_process_group(sim_process)
    return True


def wait_for_current_simulation_to_exit(
    loop_token: str,
    base_image: str,
    auto_task: str,
    auto_scene: str,
):
    while auto_loop_is_active(loop_token):
        with runtime_lock:
            sim_running = runtime.sim_process is not None
        if not sim_running:
            break
        time.sleep(1.0)
        yield (
            base_image,
            auto_task,
            auto_scene,
            *ui_snapshot(extra_status="Auto waiting for Dexsim to exit."),
        )


def run_generate_for_top_mode(
    run_mode: str,
    action_mode: str | None,
    scene_mode: str,
    robot_profile: str | None,
    image_value: str | np.ndarray | Image.Image,
    task_text: str,
    env_text: str,
    interact_prebuilt_scene_dir: str | None,
    language: str | None,
):
    parallel_env = action_mode == TOP_MODE_PARALLEL_ENV
    if run_mode != TOP_MODE_AUTO:
        selected_prebuilt_scene_dir = (
            Path(interact_prebuilt_scene_dir) if interact_prebuilt_scene_dir else None
        )
        for snapshot in run_generate(
            image_value,
            task_text,
            env_text,
            force_initial=False,
            scene_mode=scene_mode,
            parallel_env=parallel_env,
            robot_profile=robot_profile,
            load_template_material=False,
            run_log_mode=RUN_LOG_MODE_INTERACT,
            prebuilt_scene_dir=selected_prebuilt_scene_dir,
        ):
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                *snapshot,
            )
        return

    loop_token = start_auto_loop_state()
    if loop_token is None:
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            *ui_snapshot(),
        )
        return

    with runtime_lock:
        runtime.language = language or LANGUAGE_EN

    def set_auto_control_state(
        scene_mode: str,
        parallel_env: bool,
        robot_profile: str | None,
    ) -> None:
        with runtime_lock:
            runtime.auto_scene_mode = scene_mode
            runtime.auto_parallel_env = parallel_env
            runtime.auto_robot_profile = robot_profile or DEFAULT_ROBOT_PROFILE

    def run_auto_phase(
        phase_name: str,
        base_image: str,
        task_text: str,
        scene_text: str,
        *,
        scene_mode: str,
        parallel_env: bool,
        robot_profile: str | None,
        force_initial: bool = False,
        prebuilt_scene_dir: Path | None = None,
    ):
        for snapshot in run_generate(
            base_image,
            task_text,
            scene_text,
            force_initial=force_initial,
            scene_mode=scene_mode,
            parallel_env=parallel_env,
            robot_profile=robot_profile,
            load_template_material=False,
            run_log_mode=RUN_LOG_MODE_AUTO,
            prebuilt_scene_dir=prebuilt_scene_dir,
        ):
            yield (
                base_image,
                task_text,
                scene_text,
                *snapshot,
            )
            if not auto_loop_is_active(loop_token):
                break

        if not auto_loop_is_active(loop_token):
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=task_text,
                scene_description=scene_text,
                outcome="stopped",
            )
            return "stopped"

        with runtime_lock:
            pipeline_failed = runtime.phase_key == "failed"
            pipeline_error = runtime.last_error
        if pipeline_failed:
            cleanup_auto_generated_artifacts()
            if pipeline_error:
                with runtime_lock:
                    runtime.last_error = pipeline_error
                    runtime.log_lines.append(
                        f"{phase_name} generation failed: {pipeline_error}"
                    )
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=task_text,
                scene_description=scene_text,
                outcome="pipeline_failed",
            )
            return "pipeline_failed"

        for snapshot in wait_for_current_simulation_to_exit(
            loop_token,
            base_image,
            task_text,
            scene_text,
        ):
            yield snapshot

        if not auto_loop_is_active(loop_token):
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=task_text,
                scene_description=scene_text,
                outcome="stopped",
            )
            return "stopped"

        with runtime_lock:
            simulation_completed = (
                runtime.sim_started
                and runtime.sim_finished
                and runtime.sim_process is None
            )
            round_outcome = "completed" if simulation_completed else "simulation_failed"
        archive_run_log(
            mode=RUN_LOG_MODE_AUTO,
            task_description=task_text,
            scene_description=scene_text,
            outcome=round_outcome,
        )
        yield (
            base_image,
            task_text,
            scene_text,
            *ui_snapshot(extra_status=f"{phase_name}: {round_outcome}."),
        )
        return round_outcome

    def run_auto_parallel_simulation(
        base_image: str,
        task_text: str,
        scene_text: str,
        *,
        robot_profile: str | None,
    ):
        with runtime_lock:
            simulation_token = runtime.run_token
            runtime.sim_started = False
            runtime.sim_finished = False
            runtime.sim_returncode = None
            runtime.last_error = None
            runtime.status = "Starting parallel simulation..."
            clear_run_timing_locked()
            runtime.log_lines.append("Auto phase: starting parallel simulation.")

        simulation_error = launch_current_simulation(
            simulation_token,
            parallel_env=True,
            robot_profile=robot_profile,
            run_log_mode=RUN_LOG_MODE_AUTO,
            task_description=task_text,
            scene_description=scene_text,
        )
        if simulation_error is not None:
            with runtime_lock:
                runtime.phase_key = "failed"
                runtime.status = (
                    f"Parallel simulation launch failed: {simulation_error}"
                )
                runtime.last_error = simulation_error
                runtime.log_lines.append(runtime.status)
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=task_text,
                scene_description=scene_text,
                outcome="simulation_launch_failed",
            )
            yield (
                base_image,
                task_text,
                scene_text,
                *ui_snapshot(),
            )
            return "simulation_failed"

        yield (
            base_image,
            task_text,
            scene_text,
            *ui_snapshot(extra_status="Parallel simulation started."),
        )
        for snapshot in wait_for_current_simulation_to_exit(
            loop_token,
            base_image,
            task_text,
            scene_text,
        ):
            yield snapshot

        if not auto_loop_is_active(loop_token):
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=task_text,
                scene_description=scene_text,
                outcome="stopped",
            )
            return "stopped"

        with runtime_lock:
            simulation_completed = (
                runtime.sim_started
                and runtime.sim_finished
                and runtime.sim_process is None
            )
            round_outcome = "completed" if simulation_completed else "simulation_failed"
        archive_run_log(
            mode=RUN_LOG_MODE_AUTO,
            task_description=task_text,
            scene_description=scene_text,
            outcome=round_outcome,
        )
        yield (
            base_image,
            task_text,
            scene_text,
            *ui_snapshot(extra_status=f"Parallel simulation: {round_outcome}."),
        )
        return round_outcome

    while auto_loop_is_active(loop_token):
        auto_task = ""
        auto_scene = ""
        task_label = "unknown"
        with runtime_lock:
            runtime.auto_round += 1
            auto_round = runtime.auto_round
            runtime.status = f"Auto round {auto_round}: cleaning previous artifacts."
            runtime.last_error = None
            runtime.sim_started = False
            runtime.sim_finished = False
            runtime.sim_returncode = None
            runtime.log_lines.clear()
            clear_run_timing_locked()
            runtime.log_lines.append(f"Auto round {auto_round} started.")

        cleanup_errors = cleanup_auto_generated_artifacts()
        if cleanup_errors:
            with runtime_lock:
                runtime.log_lines.extend(cleanup_errors)

        if not auto_loop_is_active(loop_token):
            break

        try:
            with runtime_lock:
                selected_language = runtime.language
            auto_input = generate_auto_text_input(
                language=selected_language,
                include_scene=False,
            )
        except Exception as exc:
            if not auto_loop_is_active(loop_token):
                break
            with runtime_lock:
                runtime.phase_key = "failed"
                runtime.status = f"Auto text generation failed: {exc}"
                runtime.last_error = str(exc)
                clear_run_timing_locked()
                runtime.log_lines.append(runtime.status)
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                *ui_snapshot(),
            )
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=auto_task,
                scene_description=auto_scene,
                outcome="text_generation_failed",
            )
            continue

        base_image = auto_input.base_image_path.as_posix()
        auto_task = auto_input.task_description
        task_label = f"task{auto_input.task_index[0]}_{auto_input.task_index[1]}"
        with runtime_lock:
            runtime.task_text = format_current_task(auto_task, auto_scene)
            runtime.input_task_text = auto_task
            runtime.input_scene_text = auto_scene
            runtime.image_path = auto_input.base_image_path
            runtime.video_path = None
            runtime.lerobot_video_path = None
            runtime.lerobot_dataset_path = None
            runtime.phase_key = "received"
            runtime.status = (
                f"Auto round {auto_round}: selected {task_label}. "
                "Starting prompt2scene pipeline."
            )
            runtime.last_error = None
            runtime.log_lines.append(
                f"Auto selected {task_label}: task={auto_task!r}, scene={auto_scene!r}"
            )
            if auto_input.prebuilt_scene_dir is not None:
                runtime.log_lines.append(
                    f"Auto prebuilt scene: {auto_input.prebuilt_scene_dir}"
                )
        yield (
            base_image,
            auto_task,
            auto_scene,
            *ui_snapshot(extra_status=f"Auto text generated: {task_label}."),
        )

        if not auto_loop_is_active(loop_token):
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=auto_task,
                scene_description=auto_scene,
                outcome="stopped",
            )
            break

        with runtime_lock:
            selected_language = runtime.language
        phase_results: list[str] = []

        phase_results.append("stopped")
        set_auto_control_state(SCENE_MODE_INITIAL, False, robot_profile)
        phase_generator = run_auto_phase(
            "Initial generation",
            base_image,
            auto_task,
            auto_scene,
            scene_mode=SCENE_MODE_INITIAL,
            parallel_env=False,
            robot_profile=robot_profile,
            force_initial=True,
            prebuilt_scene_dir=auto_input.prebuilt_scene_dir,
        )
        try:
            while True:
                yield next(phase_generator)
        except StopIteration as exc:
            phase_results[0] = str(exc.value)

        if phase_results[0] == "stopped":
            break
        if phase_results[0] == "pipeline_failed":
            continue

        try:
            auto_edit_scene = generate_auto_scene_description(
                task_index=auto_input.task_index,
                language=selected_language,
                ensure_scene=True,
            )
        except Exception as exc:
            with runtime_lock:
                runtime.phase_key = "failed"
                runtime.status = f"Auto scene description generation failed: {exc}"
                runtime.last_error = str(exc)
                clear_run_timing_locked()
                runtime.log_lines.append(runtime.status)
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                *ui_snapshot(),
            )
            archive_run_log(
                mode=RUN_LOG_MODE_AUTO,
                task_description=auto_task,
                scene_description=auto_scene,
                outcome="text_generation_failed",
            )
            continue

        phase_results.append("stopped")
        set_auto_control_state(SCENE_MODE_EDIT, False, robot_profile)
        phase_generator = run_auto_phase(
            "Scene edit",
            base_image,
            auto_task,
            auto_edit_scene,
            scene_mode=SCENE_MODE_EDIT,
            parallel_env=False,
            robot_profile=robot_profile,
            force_initial=False,
        )
        try:
            while True:
                yield next(phase_generator)
        except StopIteration as exc:
            phase_results[1] = str(exc.value)

        if phase_results[1] == "stopped":
            break
        if phase_results[1] == "pipeline_failed":
            continue

        phase_results.append("stopped")
        set_auto_control_state(SCENE_MODE_EDIT, True, robot_profile)
        phase_generator = run_auto_parallel_simulation(
            base_image,
            auto_task,
            auto_edit_scene,
            robot_profile=robot_profile,
        )
        try:
            while True:
                yield next(phase_generator)
        except StopIteration as exc:
            phase_results[2] = str(exc.value)

        if phase_results[2] == "stopped":
            break

        phase_results.append("stopped")
        set_auto_control_state(SCENE_MODE_TASK_ONLY, False, ROBOT_PROFILE_FRANKA)
        phase_generator = run_auto_phase(
            "Task-only Franka",
            base_image,
            auto_task,
            "",
            scene_mode=SCENE_MODE_TASK_ONLY,
            parallel_env=False,
            robot_profile=ROBOT_PROFILE_FRANKA,
            force_initial=False,
        )
        try:
            while True:
                yield next(phase_generator)
        except StopIteration as exc:
            phase_results[3] = str(exc.value)

        if phase_results[3] == "stopped":
            break
        if phase_results[3] == "pipeline_failed":
            continue

        phase_results.append("stopped")
        set_auto_control_state(SCENE_MODE_TASK_ONLY, True, ROBOT_PROFILE_FRANKA)
        phase_generator = run_auto_parallel_simulation(
            base_image,
            auto_task,
            "",
            robot_profile=ROBOT_PROFILE_FRANKA,
        )
        try:
            while True:
                yield next(phase_generator)
        except StopIteration as exc:
            phase_results[4] = str(exc.value)

        if phase_results[4] == "stopped":
            break

        cleanup_errors = cleanup_auto_generated_artifacts()
        if cleanup_errors:
            with runtime_lock:
                runtime.log_lines.extend(cleanup_errors)

    finish_auto_loop(loop_token)


def _scene_engine_phase_from_log(line: str, current_key: str) -> str:
    """Map the standalone Scene Engine's stage names to the shared progress UI."""
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
    output_root = DEBUG_SCENE_ENGINE_ROOT / scene_hash
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / "input.png"
    image_path.write_bytes(image_bytes.getvalue())
    return scene_hash, output_root, image_path


def _wait_for_viser(port: int, process: subprocess.Popen[str]) -> bool:
    """Wait briefly for Viser's HTTP listener, without treating Ctrl-C as success."""
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
    """Embed the Viser service using the same hostname as the Gradio page."""
    srcdoc = (
        "<script>window.location.replace(window.top.location.protocol + '//' + "
        f"window.top.location.hostname + ':{port}');</script>"
    )
    return (
        f"<div style='margin-top:0.5rem'><strong>Viser preview: {html.escape(scene_hash)}</strong>"
        f"<iframe title='Viser scene preview {html.escape(scene_hash)}' "
        f'srcdoc="{html.escape(srcdoc, quote=True)}" '
        "style='width:100%; height:680px; border:1px solid #d1d5db; border-radius:8px; margin-top:0.5rem;'></iframe>"
        "</div>"
    )


def reset_scene_engine():
    """Clear Scene Engine widgets and stop its generator and Viser process groups."""
    with runtime_lock:
        generator_process = runtime.scene_engine_process
        preview_process = runtime.scene_preview_process
        owns_runtime = runtime.scene_engine_is_running
        other_workflow_running = not owns_runtime and runtime.is_busy
        if owns_runtime or not runtime.is_busy:
            if owns_runtime:
                runtime.run_token = uuid.uuid4().hex
            if runtime.process is generator_process:
                runtime.process = None
            runtime.is_busy = False
            set_runtime_phase_locked("idle")
            runtime.status = "Scene Engine reset."
            runtime.image_path = None
            runtime.last_error = None
            runtime.log_lines.clear()
            clear_run_timing_locked()
        runtime.scene_engine_process = None
        runtime.scene_preview_process = None
        runtime.scene_engine_is_running = False

    for process in {generator_process, preview_process}:
        if process is not None:
            terminate_process_group(process)

    return (
        None,
        PHASES["idle"].progress,
        format_status(
            "Scene Engine reset."
            if not other_workflow_running
            else "Scene Engine preview reset; another workflow is still running."
        ),
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
            runtime.status = "Another pipeline is already running."
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
            clear_run_timing_locked()
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

    old_preview: subprocess.Popen[str] | None = None
    old_generator: subprocess.Popen[str] | None = None
    with runtime_lock:
        if runtime.run_token != token:
            return
        old_preview = runtime.scene_preview_process
        old_generator = runtime.scene_engine_process
        runtime.scene_engine_process = None
        runtime.scene_preview_process = None
        runtime.status = f"Image saved. Generating Scene Engine output {scene_hash}."
        runtime.image_path = image_path

    if old_preview is not None:
        terminate_process_group(old_preview)
    if old_generator is not None:
        terminate_process_group(old_generator)

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
        target=read_process_output, args=(process, output_queue), daemon=True
    )
    with runtime_lock:
        if runtime.run_token != token:
            terminate_process_group(process)
            return
        runtime.process = process
        runtime.scene_engine_process = process
        start_run_timing_locked("started")
        set_runtime_phase_locked("started")
        runtime.status = "Scene Engine generation started."
    reader.start()

    while process.poll() is None:
        drained = drain_output_queue(output_queue)
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
        for line in drain_output_queue(output_queue):
            runtime.log_lines.append(line)
            set_runtime_phase_locked(
                _scene_engine_phase_from_log(line, runtime.phase_key)
            )
        runtime.process = None
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
        runtime.scene_preview_process = preview_process
        runtime.is_busy = False
        runtime.scene_engine_is_running = False
        set_runtime_phase_locked("complete")
        runtime.status = "Scene generated successfully. Viser preview is ready."
        runtime.last_error = None
    yield _scene_engine_updates(output_root, preview_html)


def run_action_engine_from_current(task_text: str, robot_profile: str | None):
    """Launch DexSim for the Gym scene most recently generated by Scene engine."""
    task_text = (task_text or "").strip()
    failure: str | None = None
    with runtime_lock:
        if not task_text:
            runtime.status = "Enter a task description first."
            runtime.last_error = "Task description is required."
            failure = runtime.status
        elif not rerun_simulation_is_available():
            runtime.status = "Generate a scene first."
            runtime.last_error = "Current Gym scene/config is unavailable."
            failure = runtime.status
        elif (
            runtime.process is not None
            or runtime.sim_process is not None
            or runtime.is_busy
        ):
            runtime.status = "Another pipeline or simulation is already running."
            runtime.last_error = "Busy."
            failure = runtime.status
        elif not action_agent_cli_is_available():
            runtime.status = (
                "Action-agent CLI is unavailable in this EmbodiChain environment."
            )
            runtime.last_error = (
                "Missing embodichain.gen_sim.action_agent_pipeline.cli.run_agent"
            )
            failure = runtime.status
        else:
            token = uuid.uuid4().hex
            runtime.run_token = token
            runtime.task_text = task_text
            runtime.input_task_text = task_text
            runtime.input_scene_text = ""
            runtime.status = "Starting DexSim action simulation..."
            runtime.last_error = None
            runtime.log_lines.append(runtime.status)

    if failure:
        return ui_snapshot()

    error = launch_current_simulation(
        token,
        robot_profile=robot_profile,
        run_log_mode=RUN_LOG_MODE_INTERACT,
        task_description=task_text,
    )
    if error:
        with runtime_lock:
            runtime.status = error
            runtime.last_error = error
    return ui_snapshot()


def action_agent_cli_is_available() -> bool:
    """Avoid spawning a subprocess when the optional action-agent package is absent."""
    try:
        return importlib.util.find_spec(COMMANDS["agent"]["module"]) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def supervise_pipeline(
    token: str,
    stage: ScenePaths,
    mode: str,
    process: subprocess.Popen[str],
    display_task_text: str,
    task_description: str,
    scene_description: str,
    output_queue: queue.Queue[str],
    reader: threading.Thread,
    parallel_env: bool,
    robot_profile: str | None,
    run_log_mode: str,
    initial_scene_path: Path | None,
    show_generated_scene_as_edit: bool,
    launch_simulation: bool = True,
) -> None:
    is_edit = mode == PIPELINE_MODE_EDIT
    is_task_only = mode == PIPELINE_MODE_TASK_ONLY
    scene_build_error: str | None = None
    simulation_error: str | None = None
    simulation_started = False
    try:
        while True:
            with runtime_lock:
                still_current = runtime.run_token == token
            if not still_current:
                terminate_process_group(process)
                return

            drained = drain_output_queue(output_queue)
            if drained:
                with runtime_lock:
                    for line in drained:
                        runtime.log_lines.append(line)
                        set_runtime_phase_locked(
                            update_phase_from_log(line, runtime.phase_key)
                        )

            with runtime_lock:
                detected_key = detect_phase_from_files(runtime.phase_key, stage)
                set_runtime_phase_locked(detected_key)
                if detected_key in PHASES and runtime.phase_key != "failed":
                    runtime.status = PHASES[detected_key].label + "."

            glb_paths = collect_generated_object_glbs(stage)
            if glb_paths and (
                not stage.gradio_object_preview_glb.is_file()
                or not object_preview_is_current(
                    stage.object_preview_manifest,
                    glb_paths,
                )
            ):
                try:
                    object_preview_path = build_object_preview_scene(
                        glb_paths,
                        stage.gradio_scene_dir,
                    )
                    with runtime_lock:
                        runtime.object_model_path = object_preview_path
                        set_runtime_phase_locked(
                            _choose_later_phase(
                                runtime.phase_key,
                                PHASES.get(runtime.phase_key, PHASES["idle"]).progress,
                                "asset_generation",
                            )[0]
                        )
                        runtime.status = (
                            f"Generated object GLB preview loaded "
                            f"({len(glb_paths)} files)."
                        )
                except Exception as exc:
                    with runtime_lock:
                        runtime.log_lines.append(f"Object preview pending: {exc}")

            if (
                not is_edit
                and not is_task_only
                and stage.fast_gym_config.is_file()
                and not gradio_scene_is_current(
                    stage.gradio_scene_glb,
                    stage.scene_manifest,
                    stage.fast_gym_config,
                )
            ):
                try:
                    scene_path = build_gradio_scene_from_fast_config(
                        stage.fast_gym_config,
                        stage.gradio_scene_dir,
                    )
                    scene_build_error = None
                    with runtime_lock:
                        if show_generated_scene_as_edit:
                            if initial_scene_path is not None:
                                runtime.scene_model_path = initial_scene_path
                            runtime.edited_scene_model_path = scene_path
                        else:
                            runtime.scene_model_path = scene_path
                        set_runtime_phase_locked("preview")
                        runtime.status = "3D preview loaded."
                        runtime.last_error = None
                except Exception as exc:
                    scene_build_error = str(exc)
                    with runtime_lock:
                        runtime.log_lines.append(
                            f"3D preview error: {scene_build_error}"
                        )
                        runtime.last_error = scene_build_error

            if process.poll() is not None:
                break
            time.sleep(0.5)

        reader.join(timeout=1.0)
        drained = drain_output_queue(output_queue)
        with runtime_lock:
            for line in drained:
                runtime.log_lines.append(line)
                set_runtime_phase_locked(update_phase_from_log(line, runtime.phase_key))

        glb_paths = collect_generated_object_glbs(stage)
        if glb_paths and (
            not stage.gradio_object_preview_glb.is_file()
            or not object_preview_is_current(stage.object_preview_manifest, glb_paths)
        ):
            try:
                object_preview_path = build_object_preview_scene(
                    glb_paths,
                    stage.gradio_scene_dir,
                )
                with runtime_lock:
                    runtime.object_model_path = object_preview_path
            except Exception as exc:
                with runtime_lock:
                    runtime.log_lines.append(f"Object preview skipped: {exc}")

        if is_task_only and process.returncode == 0 and stage.fast_gym_config.is_file():
            try:
                scene_path = build_gradio_scene_from_fast_config(
                    stage.fast_gym_config,
                    stage.gradio_scene_dir,
                )
                scene_build_error = None
                if GRADIO_SCENE_GLB.is_file():
                    GRADIO_SCENE_DIR.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(GRADIO_SCENE_GLB, GRADIO_INITIAL_SCENE_GLB)
                with runtime_lock:
                    runtime.scene_model_path = scene_path
                    runtime.edited_scene_model_path = None
                    set_runtime_phase_locked("preview")
                    runtime.status = "3D preview loaded."
                    runtime.last_error = None
            except Exception as exc:
                scene_build_error = str(exc)
        elif (
            stage.fast_gym_config.is_file()
            and (not is_edit or process.returncode == 0)
            and not gradio_scene_is_current(
                stage.gradio_scene_glb,
                stage.scene_manifest,
                stage.fast_gym_config,
            )
        ):
            try:
                scene_path = build_gradio_scene_from_fast_config(
                    stage.fast_gym_config,
                    stage.gradio_scene_dir,
                )
                scene_build_error = None
                with runtime_lock:
                    if is_edit or show_generated_scene_as_edit:
                        runtime.edited_scene_model_path = scene_path
                    else:
                        runtime.scene_model_path = scene_path
                    set_runtime_phase_locked("preview")
                    runtime.status = "3D preview loaded."
                    runtime.last_error = None
            except Exception as exc:
                scene_build_error = str(exc)

        cleanup_errors: list[str] = []
        promotion_error: str | None = None
        pipeline_output_ready = (
            stage.fast_gym_config.is_file() and stage.agent_config.is_file()
            if is_task_only
            else stage.fast_gym_config.is_file()
        )
        missing_output_name = (
            f"{stage.fast_gym_config.name} and/or {stage.agent_config.name}"
            if is_task_only
            else FAST_GYM_CONFIG.name
        )
        pipeline_succeeded = (
            process.returncode == 0 and pipeline_output_ready and not scene_build_error
        )
        if pipeline_succeeded:
            if is_edit:
                with runtime_lock:
                    if runtime.run_token == token:
                        runtime.image_path = (
                            IMAGE_PATH if IMAGE_PATH.is_file() else None
                        )
                        if GRADIO_OBJECT_PREVIEW_GLB.is_file():
                            runtime.object_model_path = GRADIO_OBJECT_PREVIEW_GLB
                        if GRADIO_INITIAL_SCENE_GLB.is_file():
                            runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                        if GRADIO_SCENE_GLB.is_file():
                            runtime.edited_scene_model_path = GRADIO_SCENE_GLB
                if launch_simulation:
                    simulation_error = launch_current_simulation(
                        token,
                        parallel_env=parallel_env,
                        robot_profile=robot_profile,
                        run_log_mode=run_log_mode,
                        task_description=task_description,
                        scene_description=scene_description,
                    )
                    simulation_started = simulation_error is None
            elif is_task_only:
                with runtime_lock:
                    if runtime.run_token == token:
                        runtime.image_path = (
                            IMAGE_PATH if IMAGE_PATH.is_file() else None
                        )
                        if GRADIO_OBJECT_PREVIEW_GLB.is_file():
                            runtime.object_model_path = GRADIO_OBJECT_PREVIEW_GLB
                        if GRADIO_INITIAL_SCENE_GLB.is_file():
                            runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                        elif GRADIO_SCENE_GLB.is_file():
                            runtime.scene_model_path = GRADIO_SCENE_GLB
                        runtime.edited_scene_model_path = None
                if launch_simulation:
                    simulation_error = launch_current_simulation(
                        token,
                        parallel_env=parallel_env,
                        robot_profile=robot_profile,
                        run_log_mode=run_log_mode,
                        task_description=task_description,
                        scene_description=scene_description,
                    )
                    simulation_started = simulation_error is None
            else:
                try:
                    cleanup_errors = promote_stage_to_current(stage, token)
                except Exception as exc:
                    promotion_error = str(exc)
                else:
                    initial_scene_error: str | None = None
                    try:
                        if show_generated_scene_as_edit and initial_scene_path:
                            GRADIO_SCENE_DIR.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(initial_scene_path, GRADIO_INITIAL_SCENE_GLB)
                        else:
                            ensure_initial_scene_snapshot(overwrite=True)
                    except Exception as exc:
                        initial_scene_error = str(exc)
                    with runtime_lock:
                        if runtime.run_token == token:
                            runtime.image_path = IMAGE_PATH
                            if GRADIO_OBJECT_PREVIEW_GLB.is_file():
                                runtime.object_model_path = GRADIO_OBJECT_PREVIEW_GLB
                            if show_generated_scene_as_edit:
                                if GRADIO_INITIAL_SCENE_GLB.is_file():
                                    runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                                elif initial_scene_path is not None:
                                    runtime.scene_model_path = initial_scene_path
                                if GRADIO_SCENE_GLB.is_file():
                                    runtime.edited_scene_model_path = GRADIO_SCENE_GLB
                            elif GRADIO_INITIAL_SCENE_GLB.is_file():
                                runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                            elif GRADIO_SCENE_GLB.is_file():
                                runtime.scene_model_path = GRADIO_SCENE_GLB
                            if not show_generated_scene_as_edit:
                                runtime.edited_scene_model_path = None
                            if initial_scene_error:
                                runtime.log_lines.append(
                                    f"Initial scene snapshot skipped: {initial_scene_error}"
                                )
                    if launch_simulation:
                        simulation_error = launch_current_simulation(
                            token,
                            parallel_env=parallel_env,
                            robot_profile=robot_profile,
                            run_log_mode=run_log_mode,
                            task_description=task_description,
                            scene_description=scene_description,
                        )
                        simulation_started = simulation_error is None

        archive_after_status = False
        archive_outcome = "completed"
        with runtime_lock:
            if runtime.run_token != token:
                return
            runtime.is_busy = False
            runtime.process = None
            if pipeline_succeeded and not promotion_error:
                set_runtime_phase_locked("complete")
                runtime.status = "Pipeline completed successfully."
                runtime.task_text = display_task_text
                runtime.image_path = IMAGE_PATH if IMAGE_PATH.is_file() else None
                if GRADIO_OBJECT_PREVIEW_GLB.is_file():
                    runtime.object_model_path = GRADIO_OBJECT_PREVIEW_GLB
                if is_edit or show_generated_scene_as_edit:
                    if GRADIO_INITIAL_SCENE_GLB.is_file():
                        runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                    elif initial_scene_path is not None:
                        runtime.scene_model_path = initial_scene_path
                    if GRADIO_SCENE_GLB.is_file():
                        runtime.edited_scene_model_path = GRADIO_SCENE_GLB
                elif GRADIO_INITIAL_SCENE_GLB.is_file():
                    runtime.scene_model_path = GRADIO_INITIAL_SCENE_GLB
                    runtime.edited_scene_model_path = None
                elif GRADIO_SCENE_GLB.is_file():
                    runtime.scene_model_path = GRADIO_SCENE_GLB
                    runtime.edited_scene_model_path = None
                if simulation_started:
                    runtime.status += "\nDexsim simulation launched."
                if cleanup_errors:
                    runtime.status += "\nCleanup completed with errors; see Last error."
                    runtime.last_error = "\n".join(cleanup_errors)
                if simulation_error:
                    runtime.status += (
                        "\nDexsim launch failed; Gradio preview is still available."
                    )
                    runtime.last_error = simulation_error
            elif process.returncode == 0 and not pipeline_output_ready:
                set_runtime_phase_locked("failed")
                runtime.status = f"Pipeline ended without {missing_output_name}."
                runtime.last_error = runtime.status
                archive_outcome = "pipeline_output_missing"
            elif scene_build_error:
                set_runtime_phase_locked("failed")
                runtime.status = f"3D preview failed: {scene_build_error}"
                runtime.last_error = scene_build_error
                archive_outcome = "preview_failed"
            elif promotion_error:
                set_runtime_phase_locked("failed")
                runtime.status = f"Scene promotion failed: {promotion_error}"
                runtime.last_error = promotion_error
                archive_outcome = "promotion_failed"
            else:
                set_runtime_phase_locked("failed")
                runtime.status = (
                    f"Pipeline failed with return code {process.returncode}."
                )
                runtime.last_error = runtime.status
                archive_outcome = "pipeline_failed"
            if pipeline_succeeded and not promotion_error:
                archive_outcome = (
                    "dexsim_launch_failed" if simulation_error else "completed"
                )
            archive_after_status = (
                run_log_mode == RUN_LOG_MODE_INTERACT and not simulation_started
            )
        if archive_after_status:
            archive_run_log(
                mode=RUN_LOG_MODE_INTERACT,
                task_description=task_description or display_task_text,
                scene_description=scene_description,
                outcome=archive_outcome,
            )
    except Exception as exc:
        should_archive_exception = False
        with runtime_lock:
            if runtime.run_token == token:
                runtime.is_busy = False
                runtime.process = None
                set_runtime_phase_locked("failed")
                runtime.status = f"Pipeline supervision failed: {exc}"
                runtime.last_error = str(exc)
                runtime.log_lines.append(runtime.status)
                should_archive_exception = run_log_mode == RUN_LOG_MODE_INTERACT
        if should_archive_exception:
            archive_run_log(
                mode=RUN_LOG_MODE_INTERACT,
                task_description=task_description or display_task_text,
                scene_description=scene_description,
                outcome="pipeline_supervision_failed",
            )


def launch_current_simulation(
    token: str,
    *,
    parallel_env: bool = False,
    robot_profile: str | None = None,
    run_log_mode: str = RUN_LOG_MODE_INTERACT,
    task_description: str = "",
    scene_description: str = "",
) -> str | None:
    if not CURRENT_PATHS.fast_gym_config.is_file():
        return f"Dexsim launch skipped; missing {CURRENT_PATHS.fast_gym_config}"
    if not CURRENT_PATHS.agent_config.is_file():
        return f"Dexsim launch skipped; missing {CURRENT_PATHS.agent_config}"

    command = build_run_agent_command(
        CURRENT_PATHS,
        parallel_env=parallel_env,
        robot_profile=robot_profile,
    )
    started_at_ns = time.time_ns()
    try:
        process = start_pipeline(command)
    except Exception as exc:
        return f"Dexsim launch failed: {exc}"

    output_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=read_process_output,
        args=(process, output_queue),
        daemon=True,
    )
    monitor = threading.Thread(
        target=monitor_simulation,
        args=(
            token,
            process,
            output_queue,
            reader,
            started_at_ns,
            run_log_mode,
            task_description,
            scene_description,
            parallel_env,
        ),
        daemon=True,
    )

    with runtime_lock:
        if runtime.run_token != token:
            stale = True
        else:
            stale = False
            runtime.sim_process = process
            runtime.sim_started = True
            runtime.sim_finished = False
            runtime.sim_returncode = None
            record_simulation_started_locked()
            runtime.log_lines.append("$ " + " ".join(command))

    if stale:
        terminate_process_group(process)
        return None

    reader.start()
    monitor.start()
    return None


def monitor_simulation(
    token: str,
    process: subprocess.Popen[str],
    output_queue: queue.Queue[str],
    reader: threading.Thread,
    started_at_ns: int,
    run_log_mode: str,
    task_description: str,
    scene_description: str,
    parallel_env: bool,
) -> None:
    while process.poll() is None:
        append_simulation_logs(token, process, drain_output_queue(output_queue))
        time.sleep(0.5)

    reader.join(timeout=1.0)
    append_simulation_logs(token, process, drain_output_queue(output_queue))
    latest_video = latest_audience_output_video(min_mtime_ns=started_at_ns)
    latest_dataset = latest_lerobot_dataset(min_mtime_ns=started_at_ns)
    lerobot_video = (
        build_lerobot_preview_video(latest_dataset)
        if latest_dataset is not None
        else None
    )
    combined_video = (
        build_single_env_combined_video(latest_video, lerobot_video)
        if not parallel_env
        else None
    )
    display_video = combined_video or latest_video

    should_archive = False
    archive_outcome = "completed"
    with runtime_lock:
        if runtime.run_token != token or runtime.sim_process is not process:
            return
        record_simulation_finished_locked()
        runtime.sim_process = None
        runtime.sim_finished = True
        runtime.sim_returncode = process.returncode
        runtime.video_path = display_video
        runtime.lerobot_dataset_path = latest_dataset
        runtime.lerobot_video_path = (
            None if combined_video is not None else lerobot_video
        )
        if process.returncode == 0:
            runtime.status = (
                "Pipeline completed successfully.\nDexsim simulation finished."
            )
            if latest_video is None:
                runtime.log_lines.append("Audience video not found in outputs.")
            if latest_dataset is None:
                runtime.log_lines.append(
                    "LeRobot dataset with recorded frames not found."
                )
            elif lerobot_video is None:
                runtime.log_lines.append(
                    f"LeRobot dataset found, but preview was not generated: {latest_dataset}"
                )
            elif combined_video is not None:
                runtime.log_lines.append(
                    f"Single-env combined video created: {combined_video}"
                )
        else:
            runtime.status = (
                "Pipeline completed successfully.\n"
                f"Dexsim simulation exited with return code {process.returncode}."
            )
            runtime.log_lines.append(
                f"Dexsim simulation exited with return code {process.returncode}."
            )
            archive_outcome = "simulation_failed"
        should_archive = run_log_mode == RUN_LOG_MODE_INTERACT
    if should_archive:
        archive_run_log(
            mode=RUN_LOG_MODE_INTERACT,
            task_description=task_description,
            scene_description=scene_description,
            outcome=archive_outcome,
            audience_video=display_video,
        )


def append_simulation_logs(
    token: str,
    process: subprocess.Popen[str],
    lines: list[str],
) -> None:
    if not lines:
        return
    with runtime_lock:
        if runtime.run_token != token or runtime.sim_process is not process:
            return
        for line in lines:
            runtime.log_lines.append(line)


def drain_output_queue(output_queue: queue.Queue[str]) -> list[str]:
    lines: list[str] = []
    while True:
        try:
            lines.append(output_queue.get_nowait())
        except queue.Empty:
            return lines


def run_reset():
    cleanup_errors = reset_current_scene()
    last_error = "\n".join(cleanup_errors) if cleanup_errors else None
    status_text = (
        "Reset complete."
        if not cleanup_errors
        else "Reset completed, but some cleanup failed."
    )
    return (
        None,
        "",
        "",
        None,
        "",
        PHASES["idle"].progress,
        format_status(status_text, last_error=last_error),
        None,
        None,
        None,
    )


def stop_current_run_without_cleanup():
    process: subprocess.Popen[str] | None = None
    sim_process: subprocess.Popen[str] | None = None
    with runtime_lock:
        runtime.run_token = uuid.uuid4().hex
        runtime.auto_loop_active = False
        runtime.auto_loop_token = None
        runtime.auto_round = 0
        process = runtime.process
        sim_process = runtime.sim_process
        runtime.process = None
        runtime.sim_process = None
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.is_busy = False
        runtime.phase_key = "idle"
        runtime.status = "Stopped."
        runtime.task_text = ""
        runtime.input_task_text = ""
        runtime.input_scene_text = ""
        runtime.image_path = None
        runtime.video_path = None
        runtime.lerobot_video_path = None
        runtime.lerobot_dataset_path = None
        runtime.object_model_path = None
        runtime.scene_model_path = None
        runtime.edited_scene_model_path = None
        runtime.last_error = None
        runtime.log_lines.clear()

    if process is not None:
        terminate_process_group(process)
    if sim_process is not None:
        terminate_process_group(sim_process)

    return (
        None,
        "",
        "",
        None,
        "",
        PHASES["idle"].progress,
        format_status("Stopped."),
        None,
        None,
        None,
    )


def run_reset_or_stop(run_mode: str):
    if run_mode == TOP_MODE_AUTO:
        return stop_current_run_without_cleanup()
    return run_reset()


def rerun_current_simulation(
    run_mode: str | None,
    action_mode: str | None,
    robot_profile: str | None,
):
    def _rerun_outputs():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            *ui_snapshot(),
        )

    if run_mode != TOP_MODE_INTERACT:
        with runtime_lock:
            runtime.status = "Rerun 3D is only available in Interact mode."
            runtime.last_error = runtime.status
            runtime.log_lines.append(runtime.status)
        return _rerun_outputs()

    if not rerun_simulation_is_available():
        with runtime_lock:
            runtime.status = (
                "Current simulation files are not available. Generate once first."
            )
            runtime.last_error = runtime.status
            runtime.log_lines.append(runtime.status)
        return _rerun_outputs()

    token = uuid.uuid4().hex
    with runtime_lock:
        if runtime.process is not None:
            runtime.status = "Another pipeline run is in progress. Stop it first."
            runtime.last_error = runtime.status
            runtime.log_lines.append(runtime.status)
            return _rerun_outputs()
        if runtime.sim_process is not None:
            runtime.status = "Another Dexsim process is running. Stop it first."
            runtime.last_error = runtime.status
            runtime.log_lines.append(runtime.status)
            return _rerun_outputs()
        if runtime.is_busy:
            runtime.status = "Another run is in progress. Stop it first."
            runtime.last_error = runtime.status
            runtime.log_lines.append(runtime.status)
            return _rerun_outputs()

        runtime.run_token = token
        runtime.sim_started = False
        runtime.sim_finished = False
        runtime.sim_returncode = None
        runtime.last_error = None
        clear_run_timing_locked()
        runtime.log_lines.append("Starting Dexsim rerun (run_agent only).")

    simulation_error = launch_current_simulation(
        runtime.run_token,
        parallel_env=action_mode == TOP_MODE_PARALLEL_ENV,
        robot_profile=robot_profile,
        run_log_mode=RUN_LOG_MODE_INTERACT,
        task_description=runtime.task_text,
        scene_description=runtime.input_scene_text,
    )
    if simulation_error is not None:
        with runtime_lock:
            runtime.status = f"Dexsim rerun launch failed: {simulation_error}"
            runtime.last_error = simulation_error
            runtime.log_lines.append(runtime.status)

    return _rerun_outputs()


def randomize_interact_task_input(run_mode: str | None, language: str | None):
    """Fill the Interact form with one available template task."""
    if run_mode != TOP_MODE_INTERACT:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            gr.update(),
            None,
            None,
        )
    auto_input = generate_auto_text_input(
        language=language or LANGUAGE_EN,
        include_scene=False,
    )
    initial_preview = None
    if auto_input.prebuilt_scene_dir is not None:
        initial_preview = build_interact_random_initial_preview(
            auto_input.prebuilt_scene_dir
        ).as_posix()
    return (
        auto_input.base_image_path.as_posix(),
        gr.update(value=auto_input.task_description, interactive=True),
        gr.update(),
        SCENE_MODE_INITIAL,
        (
            auto_input.prebuilt_scene_dir.as_posix()
            if auto_input.prebuilt_scene_dir
            else None
        ),
        initial_preview,
        None,
        None,
    )


def randomize_interact_scene_input(run_mode: str | None, language: str | None):
    """Fill only the scene text in the Interact form."""
    if run_mode != TOP_MODE_INTERACT:
        return gr.update()
    scene_description = generate_auto_scene_description(
        language=language or LANGUAGE_EN,
        ensure_scene=True,
    )
    return gr.update(value=scene_description, interactive=True)


def clear_interact_prebuilt_scene() -> None:
    return None


def button_updates(
    language: str | None,
    run_mode: str | None,
    action_mode: str | None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Build localized labels while preserving the selected button variants."""
    labels = BUTTON_LABELS.get(language or LANGUAGE_EN, BUTTON_LABELS[LANGUAGE_EN])
    is_auto = run_mode == TOP_MODE_AUTO
    is_interact = run_mode != TOP_MODE_AUTO
    is_parallel_env = action_mode == TOP_MODE_PARALLEL_ENV
    can_rerun = (
        run_mode == TOP_MODE_INTERACT
        and rerun_simulation_is_available()
        and not runtime.is_busy
        and runtime.process is None
        and runtime.sim_process is None
    )
    return (
        gr.update(
            value=labels["auto"],
            variant="primary" if is_auto else "secondary",
        ),
        gr.update(
            value=labels["interact"],
            variant="primary" if is_interact else "secondary",
        ),
        gr.update(
            value=labels["parallel_env"],
            variant="primary" if is_parallel_env else "secondary",
            interactive=not is_auto,
        ),
        gr.update(value=labels["start"] if is_auto else labels["generate"]),
        gr.update(
            value=labels["rerun_simulation"],
            visible=is_interact,
            interactive=can_rerun,
        ),
        gr.update(value=labels["random_input"], visible=is_interact),
        gr.update(value=labels["random_scene_input"], visible=is_interact),
        gr.update(value=labels["stop"] if is_auto else labels["reset"]),
    )


def auto_control_updates(
    run_mode: str | None,
    action_mode: str | None,
) -> tuple[Any, Any, Any, str | None]:
    if run_mode != TOP_MODE_AUTO:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(),
            action_mode,
        )

    with runtime_lock:
        scene_mode = runtime.auto_scene_mode
        parallel_env = runtime.auto_parallel_env
        robot_profile = runtime.auto_robot_profile
        labels = BUTTON_LABELS.get(runtime.language, BUTTON_LABELS[LANGUAGE_EN])
    return (
        gr.update(value=scene_mode, interactive=False),
        gr.update(value=robot_profile, interactive=False),
        gr.update(
            value=labels["parallel_env"],
            variant="primary" if parallel_env else "secondary",
            interactive=False,
        ),
        TOP_MODE_PARALLEL_ENV if parallel_env else None,
    )


def video_preview_label(language: str | None, action_mode: str | None) -> str:
    text = UI_TEXT.get(language or LANGUAGE_EN, UI_TEXT[LANGUAGE_EN])
    key = (
        "parallel_video_preview"
        if action_mode == TOP_MODE_PARALLEL_ENV
        else "single_video_preview"
    )
    return text[key]


def scene_mode_choices(language: str | None) -> list[tuple[str, str]]:
    text = UI_TEXT.get(language or LANGUAGE_EN, UI_TEXT[LANGUAGE_EN])
    return [
        (text["scene_mode_initial"], SCENE_MODE_INITIAL),
        (text["scene_mode_edit"], SCENE_MODE_EDIT),
        (text["scene_mode_task_only"], SCENE_MODE_TASK_ONLY),
    ]


def scene_mode_input_updates(scene_mode: str | None) -> tuple[Any, Any]:
    """Set field availability for the selected scene operation."""
    is_task_only = scene_mode == SCENE_MODE_TASK_ONLY
    return (
        gr.update(interactive=True),
        gr.update(interactive=not is_task_only),
    )


def localized_ui_updates(
    language: str | None,
    action_mode: str | None,
) -> tuple[Any, ...]:
    """Return updates for every non-button, user-facing static UI string."""
    text = UI_TEXT.get(language or LANGUAGE_EN, UI_TEXT[LANGUAGE_EN])
    instruction_html = (
        "<div style='font-size: 20px; font-weight: 700; "
        "line-height: 1.35; min-height: 86px; display: flex; "
        f"align-items: center;'>{text['instruction']}</div>"
    )
    return (
        gr.update(value=text["heading"]),
        gr.update(value=instruction_html),
        gr.update(label=text["robot"]),
        gr.update(label=text["input_image"]),
        gr.update(
            label=text["task_description"],
            placeholder=text["task_placeholder"],
        ),
        gr.update(
            label=text["scene_description"],
            placeholder=text["scene_placeholder"],
        ),
        gr.update(
            label=text["scene_mode"],
            choices=scene_mode_choices(language),
        ),
        gr.update(label=video_preview_label(language, action_mode)),
        gr.update(label=text["current_task"]),
        gr.update(label=text["progress"]),
        gr.update(label=text["initial_preview"]),
        gr.update(label=text["edited_preview"]),
        gr.update(label=text["object_preview"]),
    )


def toggle_language(
    language: str | None,
    run_mode: str | None,
    action_mode: str | None,
):
    next_language = LANGUAGE_ZH if language != LANGUAGE_ZH else LANGUAGE_EN
    with runtime_lock:
        runtime.language = next_language
    labels = BUTTON_LABELS[next_language]
    return (
        *button_updates(next_language, run_mode, action_mode),
        gr.update(value=labels["language"]),
        *localized_ui_updates(next_language, action_mode),
        next_language,
    )


def select_top_mode(
    selected_run_mode: str | None,
    selected_action_mode: str | None,
    current_run_mode: str,
    current_action_mode: str | None,
    language: str | None,
):
    run_mode = selected_run_mode or current_run_mode or TOP_MODE_INTERACT
    action_mode = current_action_mode
    if selected_action_mode == TOP_MODE_PARALLEL_ENV:
        action_mode = (
            None if action_mode == TOP_MODE_PARALLEL_ENV else TOP_MODE_PARALLEL_ENV
        )
    elif selected_action_mode:
        action_mode = selected_action_mode
    if (
        run_mode != current_run_mode
        or action_mode != current_action_mode
        or run_mode != TOP_MODE_AUTO
    ):
        stop_auto_loop_if_running()
    return (
        *button_updates(language, run_mode, action_mode),
        gr.update(label=video_preview_label(language, action_mode)),
        run_mode,
        action_mode,
    )


def ui_snapshot(extra_status: str | None = None):
    with runtime_lock:
        phase = PHASES.get(runtime.phase_key, PHASES["idle"])
        video_value = None
        video_signature = None
        if runtime.video_path and runtime.video_path.is_file():
            video_value = runtime.video_path.as_posix()
            video_signature = (video_value, runtime.video_path.stat().st_mtime_ns)
        if runtime.auto_loop_active:
            video_update = video_value
        elif video_signature != runtime.last_sent_video_signature:
            runtime.last_sent_video_signature = video_signature
            video_update = video_value
        else:
            video_update = gr.update()
        object_model_value = (
            runtime.object_model_path.as_posix()
            if runtime.object_model_path and runtime.object_model_path.is_file()
            else None
        )
        model_value = (
            runtime.scene_model_path.as_posix()
            if runtime.scene_model_path and runtime.scene_model_path.is_file()
            else None
        )
        edited_model_value = (
            runtime.edited_scene_model_path.as_posix()
            if runtime.edited_scene_model_path
            and runtime.edited_scene_model_path.is_file()
            else None
        )
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
        model_value,
        edited_model_value,
        object_model_value,
    )


def synced_ui_snapshot(
    run_mode: str | None = None,
    action_mode: str | None = None,
    last_seen_input_revision: int | None = None,
):
    sync_inputs = False
    with runtime_lock:
        submitted_input_revision = runtime.submitted_input_revision
        sync_inputs = (
            runtime.auto_loop_active
            or run_mode == TOP_MODE_AUTO
            or submitted_input_revision != (last_seen_input_revision or 0)
        )
        image_value = (
            runtime.image_path.as_posix()
            if runtime.image_path and runtime.image_path.is_file()
            else None
        )
        input_task_text = runtime.input_task_text
        input_scene_text = runtime.input_scene_text
        can_rerun = (
            runtime.process is None
            and runtime.sim_process is None
            and not runtime.is_busy
            and rerun_simulation_is_available()
        )

    if sync_inputs:
        input_values = (image_value, input_task_text, input_scene_text)
    else:
        input_values = (gr.update(), gr.update(), gr.update())
    return (
        *input_values,
        *ui_snapshot(),
        gr.update(
            visible=run_mode == TOP_MODE_INTERACT,
            interactive=run_mode == TOP_MODE_INTERACT and can_rerun,
        ),
        submitted_input_revision,
        *auto_control_updates(run_mode, action_mode),
    )


def format_status(
    status_text: str,
    *,
    phase: Phase | None = None,
    busy: bool = False,
    last_error: str | None = None,
) -> str:
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
