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

"""Optional local display metadata, never execution or TaskSpec evidence."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any

__all__: list[str] = []


def _read_optional(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _thumbnail(path: Path, root: Path, *, render: bool) -> str | None:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    try:
        if not path.resolve().is_relative_to(root.resolve()):
            return None
        if path.stat().st_size > 8 * 1024 * 1024:
            return None
        with Image.open(path) as source:
            if source.width * source.height > 16_000_000:
                return None
            image = ImageOps.exif_transpose(source).convert("RGBA")
            image.thumbnail((768, 384) if render else (192, 192))
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _graph_presentation(
    graph: Mapping[str, Any],
    *,
    bundle_root: Path | None = None,
    output: Path | None = None,
    execution_report: Mapping[str, Any] | None = None,
    run_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(run_summary or {})
    report = execution_report if isinstance(execution_report, Mapping) else {}
    if summary.get("task_id") != graph["task_id"]:
        summary = {}
    elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        elapsed = None
    result: dict[str, Any] = {
        "run_id": summary.get("run_id", ""),
        "created_at": summary.get("created_at", ""),
        "robot": "未记录",
        "wall_seconds": elapsed,
        "wall_source": summary.get("elapsed_scope", "未记录"),
        "acceptance_mode": (
            report.get("runtime_result", {}).get("acceptance_mode", "未记录")
            if isinstance(report.get("runtime_result"), Mapping)
            else "未记录"
        ),
        "assets": [],
        "video_href": None,
    }
    if output is not None:
        # A relative link keeps the HTML offline and relocatable with its videos.
        videos = sorted((output.parent / "videos").glob("*.mp4"))
        if videos:
            video = next((v for v in videos if "audience_view" in v.name), videos[0])
            result["video_href"] = Path(
                os.path.relpath(video, output.parent)
            ).as_posix()
    if bundle_root is None:
        return result
    scene = _read_optional(bundle_root / "scene_manifest.json")
    static = _read_optional(bundle_root / "static_scene_manifest.json")
    result["robot"] = scene.get("robot_profile", static.get("robot_profile", "未记录"))
    source = static.get("source", {})
    config_path = source.get("config_path") if isinstance(source, Mapping) else None
    project = None
    if isinstance(config_path, str) and config_path:
        config = Path(config_path).expanduser()
        if not config.is_absolute():
            config = bundle_root / config
        parent = config.parent
        project = parent.parent if parent.name == "scene_export" else parent
    referenced: set[str] = set()
    for node in graph["nodes"]:
        call = node["call"]
        arguments = call.get("arguments", {})
        for value in (
            call.get("object"),
            arguments.get("object"),
            arguments.get("reference"),
        ):
            if isinstance(value, str):
                referenced.add(value)
    objects = scene.get("objects", static.get("objects", []))
    if not isinstance(objects, list):
        return result
    static_rows = static.get("objects", [])
    if not isinstance(static_rows, list):
        static_rows = []
    static_objects = {
        item["uid"]: item
        for item in static_rows
        if isinstance(item, dict) and isinstance(item.get("uid"), str)
    }
    for item in objects:
        if not isinstance(item, dict) or item.get("uid") not in referenced:
            continue
        uid = item["uid"]
        source_uid = static_objects.get(uid, {}).get("source_uid", uid)
        if (
            not isinstance(source_uid, str)
            or source_uid in {".", ".."}
            or Path(source_uid).name != source_uid
            or "\\" in source_uid
        ):
            source_uid = None
        asset = {
            key: item.get(key, "")
            for key in ("uid", "name", "description", "category", "role")
        }
        asset.update(cutout=None, render=None, cutout_source=None, render_source=None)
        geometry = static_objects.get(uid, {}).get("geometry", {})
        if not isinstance(geometry, Mapping):
            geometry = {}
        shape = geometry.get("shape", {})
        asset["glb_source"] = (
            shape.get("fpath", "未提供") if isinstance(shape, Mapping) else "未提供"
        )
        asset["glb_sha256"] = geometry.get("asset_sha256")
        if project is not None and source_uid is not None:
            for kind, directory, suffix in (
                ("cutout", "scene_understanding/object_images", "_rgba.png"),
                ("render", "scene_generation/debug/vlm_views", ".png"),
            ):
                path = project / directory / (source_uid + suffix)
                asset[kind] = _thumbnail(path, project, render=kind == "render")
                if asset[kind] is not None:
                    asset[kind + "_source"] = path.as_posix()
        result["assets"].append(asset)
    return result
