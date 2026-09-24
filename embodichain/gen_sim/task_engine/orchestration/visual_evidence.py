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

"""Render current scene assets as UID-linked visual grounding evidence."""

from __future__ import annotations

import colorsys
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.spatial.transform import Rotation

from .scene_source import scene_revision_id
from .source_scene import prepare_scene

__all__ = ["render_scene_visual_evidence"]

_SCHEMA = "gen_sim.visual-grounding-evidence/v1"
_WIDTH = 960
_HEIGHT = 720
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def render_scene_visual_evidence(
    source: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Render in an isolated EGL process and return the auditable manifest."""
    scene_path = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", __name__, scene_path.as_posix(), output.as_posix()],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Visual evidence rendering failed: "
            + (result.stderr or result.stdout)[-2000:]
        )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("Visual evidence manifest schema is invalid.")
    manifest["catalog_path"] = (output / manifest["catalog_path"]).as_posix()
    for view in manifest["views"]:
        for key in ("image_path", "annotated_path"):
            view[key] = (output / view[key]).as_posix()
    for item in manifest["objects"]:
        for key in ("crop_paths", "mask_paths"):
            item[key] = {
                name: (output / path).as_posix() for name, path in item[key].items()
            }
    return manifest


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(config: dict[str, Any], *, proxy: bool) -> np.ndarray:
    if not proxy and config.get("init_local_pose") is not None:
        return np.asarray(config["init_local_pose"], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler(
        "XYZ", config.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
    ).as_matrix()
    matrix[:3, 3] = np.asarray(
        config.get("proxy_init_pos" if proxy else "init_pos", config.get("init_pos")),
        dtype=float,
    )
    return matrix


def _mesh_path(config: dict[str, Any], scene_dir: Path, *, proxy: bool) -> Path:
    raw = (
        config.get("proxy_glb_fpath") if proxy else config.get("shape", {}).get("fpath")
    )
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Scene entity {config.get('uid')!r} has no visual mesh.")
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (scene_dir / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Scene visual mesh is missing: {resolved}")
    return resolved


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    actual_up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, actual_up, backward))
    pose[:3, 3] = eye
    return pose


def _safe_name(uid: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", uid)[:48]
    return f"{cleaned}_{hashlib.sha256(uid.encode()).hexdigest()[:8]}"


def _render(source: Path, output: Path) -> None:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    from PIL import Image, ImageDraw, ImageFont
    import pyrender
    import trimesh

    prepared = prepare_scene(source)
    scene = pyrender.Scene(bg_color=[250, 250, 250, 255], ambient_light=[0.65] * 3)
    color_by_uid: dict[str, tuple[int, int, int]] = {}
    asset_by_uid: dict[str, Path] = {}
    proxy_by_uid: dict[str, bool] = {}
    world_vertices: list[np.ndarray] = []
    seg_node_map: dict[Any, tuple[int, int, int]] = {}
    entries = [
        *((item, False) for item in prepared.background),
        *((item, False) for item in prepared.rigid_objects),
        *((item, True) for item in prepared.articulations),
    ]
    for index, (config, proxy) in enumerate(entries, start=1):
        uid = str(config["uid"])
        path = _mesh_path(config, prepared.scene_dir, proxy=proxy)
        loaded = trimesh.load(path.as_posix(), force="scene")
        scale = np.asarray(
            config.get("proxy_body_scale" if proxy else "body_scale", [1.0] * 3),
            dtype=float,
        )
        transform = _pose(config, proxy=proxy) @ np.diag([*scale, 1.0]) @ _Y_UP_TO_Z_UP
        color = tuple(
            int(channel * 255)
            for channel in colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.95, 1.0)
        )
        color_by_uid[uid] = color
        asset_by_uid[uid] = path
        proxy_by_uid[uid] = proxy
        for node_name in loaded.graph.nodes_geometry:
            node_transform, geometry_name = loaded.graph.get(node_name)
            mesh = loaded.geometry[geometry_name]
            world = transform @ node_transform
            node = scene.add(
                pyrender.Mesh.from_trimesh(mesh, smooth=False),
                pose=world,
                name=f"{uid}/{node_name}",
            )
            seg_node_map[node] = color
            vertices = np.asarray(mesh.vertices, dtype=float)
            world_vertices.append(vertices @ world[:3, :3].T + world[:3, 3])
    if not world_vertices:
        raise ValueError("Visual evidence scene has no renderable geometry.")
    bounds = np.concatenate(world_vertices, axis=0)
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0)) / 2.0)
    distance = max(1.0, radius * 3.2)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    renderer = pyrender.OffscreenRenderer(_WIDTH, _HEIGHT)
    object_rows = {
        uid: {
            "uid": uid,
            "asset_path": path.as_posix(),
            "asset_sha256": _file_hash(path),
            "geometry_kind": (
                "articulation_proxy_glb" if proxy_by_uid[uid] else "rigid_glb"
            ),
            "visible_views": [],
            "crop_paths": {},
            "mask_paths": {},
        }
        for uid, path in asset_by_uid.items()
    }
    views = []
    try:
        for name, direction, up in (
            ("oblique", [1.0, -1.0, 0.85], [0.0, 0.0, 1.0]),
            ("top", [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ):
            direction_array = np.asarray(direction, dtype=float)
            eye = center + direction_array / np.linalg.norm(direction_array) * distance
            pose = _look_at(eye, center, np.asarray(up, dtype=float))
            projected = (bounds - center) @ pose[:3, :2]
            spans = np.ptp(projected, axis=0)
            ymag = max(
                0.1, float(spans[1]) * 0.58, float(spans[0]) * 0.58 * _HEIGHT / _WIDTH
            )
            camera = pyrender.OrthographicCamera(
                xmag=ymag * _WIDTH / _HEIGHT, ymag=ymag
            )
            camera_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)
            try:
                rgb, _ = renderer.render(scene)
                segmented, _ = renderer.render(
                    scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map
                )
            finally:
                scene.remove_node(camera_node)
                scene.remove_node(light_node)
            image_path = output / f"{name}.png"
            Image.fromarray(rgb).save(image_path)
            annotated = Image.fromarray(rgb)
            draw = ImageDraw.Draw(annotated)
            font = ImageFont.load_default()
            for uid, color in color_by_uid.items():
                mask = np.all(segmented == color, axis=2)
                components, count = connected_components(mask)
                if count:
                    areas = np.bincount(components.ravel())[1:]
                    keep = np.flatnonzero(areas >= max(8, int(areas.max() * 0.01))) + 1
                    mask = np.isin(components, keep)
                if int(mask.sum()) < 8:
                    continue
                ys, xs = np.nonzero(mask)
                left, top = max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8)
                right = min(_WIDTH, int(xs.max()) + 9)
                bottom = min(_HEIGHT, int(ys.max()) + 9)
                text_width = int(draw.textlength(uid, font=font))
                label_x = min(left, _WIDTH - text_width - 6)
                label_y = max(0, top - 19)
                draw.rectangle(
                    (label_x, label_y, label_x + text_width + 6, label_y + 17),
                    fill="black",
                )
                draw.text((label_x + 3, label_y + 2), uid, fill="white", font=font)
                row = object_rows[uid]
                mask_path = output / f"{_safe_name(uid)}_{name}_mask.png"
                crop_path = output / f"{_safe_name(uid)}_{name}_crop.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                crop = np.where(mask[..., None], rgb, 255)[top:bottom, left:right]
                Image.fromarray(crop).save(crop_path)
                row["visible_views"].append(name)
                row["mask_paths"][name] = mask_path.name
                row["crop_paths"][name] = crop_path.name
            annotated_path = output / f"{name}_labeled.png"
            annotated.save(annotated_path)
            views.append(
                {
                    "name": name,
                    "image_path": image_path.name,
                    "image_sha256": _file_hash(image_path),
                    "annotated_path": annotated_path.name,
                    "annotated_sha256": _file_hash(annotated_path),
                    "camera_eye": eye.tolist(),
                    "camera_target": center.tolist(),
                }
            )
    finally:
        renderer.delete()
    catalog_items = [row for row in object_rows.values() if row["uid"] != "table"]
    cell_width, cell_height, columns = 190, 170, 4
    rows = max(1, (len(catalog_items) + columns - 1) // columns)
    catalog = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    catalog_draw = ImageDraw.Draw(catalog)
    font = ImageFont.load_default()
    for index, item in enumerate(catalog_items):
        crop_path = item["crop_paths"].get("oblique") or item["crop_paths"].get("top")
        if crop_path is None:
            continue
        thumbnail = Image.open(output / crop_path).convert("RGB")
        thumbnail.thumbnail((cell_width - 16, cell_height - 36))
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        catalog.paste(thumbnail, (x + (cell_width - thumbnail.width) // 2, y + 6))
        catalog_draw.text(
            (x + 8, y + cell_height - 25), item["uid"], fill="black", font=font
        )
    catalog_path = output / "uid_catalog.png"
    catalog.save(catalog_path)
    config_path = prepared.source_config_path
    manifest = {
        "schema_version": _SCHEMA,
        "source_config_path": config_path.as_posix(),
        "source_config_sha256": _file_hash(config_path),
        "scene_revision_id": scene_revision_id(source),
        "render_kind": "configured_static_glb_proxies",
        "state_scope": "configured_pose_not_live_joint_state",
        "articulation_state_untrusted_uids": [
            uid for uid, proxy in proxy_by_uid.items() if proxy
        ],
        "catalog_path": catalog_path.name,
        "catalog_sha256": _file_hash(catalog_path),
        "views": views,
        "objects": list(object_rows.values()),
        "unrendered_uids": [
            uid for uid, row in object_rows.items() if not row["visible_views"]
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    _render(Path(sys.argv[1]), Path(sys.argv[2]))
