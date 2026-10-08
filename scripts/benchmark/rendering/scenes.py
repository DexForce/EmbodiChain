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

"""Frozen, renderer-independent scenes for visual review and R-series workloads.

Scene preparation resolves robot assets and forward kinematics once. Renderers
consume the same triangle buffers and world transforms, without importing a
robot articulation or advancing a physics solver. JSON loading stays independent
of NumPy, Torch, DexSim and Isaac Lab.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from scripts.benchmark.core.artifacts import stable_hash, write_json

__all__ = [
    "SCENE_NAMES",
    "build_scene",
    "camera_intrinsics",
    "camera_world_pose",
    "load_scene",
    "scene_fingerprint",
    "validate_scene",
]

SCENE_NAMES = ("calibration", "workshop", "franka_workstation")


def _pose(position: list[float]) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, position[0]],
        [0.0, 1.0, 0.0, position[1]],
        [0.0, 0.0, 1.0, position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _box(
    uid: str,
    size: list[float],
    position: list[float],
    color: list[float],
    *,
    roughness: float = 0.7,
    metallic: float = 0.0,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "geometry": {"kind": "box", "size": size},
        "world_from_object": _pose(position),
        "material": {
            "base_color_linear": color,
            "roughness": roughness,
            "metallic": metallic,
        },
    }


def _table() -> list[dict[str, Any]]:
    objects = [
        _box("floor", [4.0, 4.0, 0.06], [0.0, 0.0, -0.80], [0.16] * 3),
        _box("table", [1.5, 1.1, 0.06], [0.0, 0.0, -0.03], [0.42] * 3),
    ]
    for index, (x, y) in enumerate(
        ((-0.65, -0.43), (-0.65, 0.43), (0.65, -0.43), (0.65, 0.43))
    ):
        objects.append(
            _box(f"table_leg_{index}", [0.06, 0.06, 0.71], [x, y, -0.415], [0.11] * 3)
        )
    return objects


def _chart(prefix: str, x: float, y: float) -> list[dict[str, Any]]:
    colors = (
        [0.65, 0.08, 0.06],
        [0.07, 0.48, 0.12],
        [0.06, 0.15, 0.65],
        [0.7, 0.48, 0.05],
        [0.38, 0.08, 0.48],
        [0.06, 0.5, 0.55],
        [0.02] * 3,
        [0.08] * 3,
        [0.18] * 3,
        [0.36] * 3,
        [0.60] * 3,
        [0.85] * 3,
    )
    objects = [_box(prefix + "_board", [0.59, 0.022, 0.22], [x, y, 0.16], [0.12] * 3)]
    for index, color in enumerate(colors):
        objects.append(
            _box(
                f"{prefix}_{index:02d}",
                [0.08, 0.004, 0.08],
                [
                    x + (index % 6 - 2.5) * 0.095,
                    y - 0.014,
                    0.205 - (index // 6) * 0.095,
                ],
                color,
                roughness=1.0,
            )
        )
    return objects


def _franka(
    destination: Path, urdf_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import numpy as np
    import pytorch_kinematics as pk
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation

    source = urdf_path.read_bytes()
    chain = pk.build_chain_from_urdf(source)
    qpos = dict(
        zip(
            [f"fr3_joint{i}" for i in range(1, 8)]
            + ["fr3_finger_joint1", "fr3_finger_joint2"],
            [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04],
            strict=True,
        )
    )
    joint_names = chain.get_joint_parameter_names()
    transforms = chain.forward_kinematics(
        torch.tensor([[qpos[name] for name in joint_names]])
    )
    root_pose = np.eye(4)
    root_pose[:3, 3] = [-0.43, 0.08, 0.0]
    objects: list[dict[str, Any]] = []
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source_assets = []
    for link in ET.fromstring(source).findall("link"):
        name = link.attrib["name"]
        for index, visual in enumerate(link.findall("visual")):
            element = visual.find("geometry/mesh")
            if element is None:
                raise ValueError(f"Expected mesh geometry in Franka visual {name}")
            mesh_path = (urdf_path.parent / element.attrib["filename"]).resolve()
            mesh = trimesh.load(str(mesh_path), force="scene", process=False).to_mesh()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            scale = [float(v) for v in element.get("scale", "1 1 1").split()]
            mesh.apply_scale(scale)
            uid = f"robot_{name}_{index}"
            asset_path = assets / (uid + ".npz")
            normals = np.array(mesh.vertex_normals, dtype=np.float32, copy=True)
            lengths = np.linalg.norm(normals, axis=1)
            invalid = ~np.isfinite(lengths) | (lengths < 1e-6)
            # Resolve degenerate source normals once, before either renderer can
            # apply a different native repair to the same asset.
            for vertex in np.flatnonzero(invalid):
                incident = np.flatnonzero(np.any(mesh.faces == vertex, axis=1))
                normals[vertex] = mesh.face_normals[incident[0]]
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            np.savez_compressed(
                asset_path,
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                triangles=np.asarray(mesh.faces, dtype=np.int32),
                normals=normals,
            )
            visual_pose = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                visual_pose[:3, 3] = [
                    float(v) for v in origin.get("xyz", "0 0 0").split()
                ]
                visual_pose[:3, :3] = Rotation.from_euler(
                    "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]
                ).as_matrix()
            pose = root_pose @ transforms[name].get_matrix()[0].numpy() @ visual_pose
            color = [0.08] * 3 if "finger" in name else [0.72] * 3
            objects.append(
                {
                    "uid": uid,
                    "geometry": {
                        "kind": "mesh",
                        "path": str(asset_path.relative_to(destination)),
                        "sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                        "vertices": len(mesh.vertices),
                        "triangles": len(mesh.faces),
                        "repaired_source_normals": int(invalid.sum()),
                    },
                    "world_from_object": pose.tolist(),
                    "material": {
                        "base_color_linear": color,
                        "roughness": 0.4,
                        "metallic": 0.0,
                    },
                }
            )
            source_assets.append(
                {
                    "path": element.attrib["filename"],
                    "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                }
            )
    hand = root_pose @ transforms["fr3_hand"].get_matrix()[0].numpy()
    eye = hand @ np.array([0.045, -0.035, 0.015, 1.0])
    target = hand @ np.array([0.045, -0.035, 0.5, 1.0])
    up = hand[:3, :3] @ np.array([0.0, -1.0, 0.0])
    wrist = {
        "name": "wrist",
        "eye": eye[:3].tolist(),
        "target": target[:3].tolist(),
        "up": up.tolist(),
    }
    provenance = {
        "urdf_sha256": hashlib.sha256(source).hexdigest(),
        "mesh_sources": source_assets,
        "joint_positions": qpos,
        "world_from_robot": root_pose.tolist(),
        "appearance": "shared neutral PBR overrides; source textures are not used",
        "state_source": "frozen forward kinematics; no physical rollout",
        "wrist_parent": "fr3_hand",
        "wrist_eye_offset_m": [0.045, -0.035, 0.015],
    }
    return objects, wrist, provenance


def build_scene(
    name: str, destination: Path, *, franka_urdf: Path | None = None
) -> Path:
    """Freeze a scene and canonical mesh assets for both renderers.

    Args:
        name: One of :data:`SCENE_NAMES`.
        destination: New directory receiving ``scene.json`` and mesh arrays.
        franka_urdf: Optional explicit asset; otherwise use the project registry.

    Returns:
        Absolute path of the validated scene manifest.
    """
    if name not in SCENE_NAMES:
        raise ValueError(f"Unknown scene: {name}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    objects = _table()
    objects += _chart("chart", 0.0, 0.46)
    objects += [
        _box("red_cube", [0.09] * 3, [0.18, -0.16, 0.045], [0.65, 0.055, 0.025]),
        _box(
            "blue_block", [0.12, 0.07, 0.15], [0.38, 0.16, 0.075], [0.035, 0.12, 0.62]
        ),
        _box(
            "green_block", [0.08, 0.12, 0.06], [0.34, -0.22, 0.03], [0.035, 0.4, 0.07]
        ),
        _box(
            "metal_block",
            [0.10, 0.10, 0.13],
            [0.52, -0.05, 0.065],
            [0.58] * 3,
            roughness=0.14,
            metallic=0.9,
        ),
        _box("tray_base", [0.24, 0.20, 0.014], [0.05, 0.18, 0.007], [0.10] * 3),
    ]
    cameras = [
        {
            "name": "external_front",
            "eye": [1.35, -1.65, 1.25],
            "target": [0.0, 0.0, 0.22],
            "up": [0.0, 0.0, 1.0],
        },
        {
            "name": "external_side",
            "eye": [-1.4, -1.3, 1.0],
            "target": [0.02, 0.02, 0.24],
            "up": [0.0, 0.0, 1.0],
        },
    ]
    robot_source = None
    if name != "calibration":
        if franka_urdf is None:
            from embodichain.data import get_data_path

            franka_urdf = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
        robot_objects, wrist, robot_source = _franka(destination, Path(franka_urdf))
        objects += robot_objects
        cameras.append(wrist)
    else:
        objects += [
            _box(
                "shadow_step_low", [0.10, 0.15, 0.04], [-0.40, -0.15, 0.02], [0.65] * 3
            ),
            _box(
                "shadow_step_mid", [0.10, 0.15, 0.08], [-0.29, -0.15, 0.04], [0.65] * 3
            ),
            _box(
                "shadow_step_high", [0.10, 0.15, 0.16], [-0.18, -0.15, 0.08], [0.65] * 3
            ),
        ]
    if name == "workshop":
        objects += [
            _box("back_wall", [3.5, 0.08, 2.4], [0.0, 1.35, 0.4], [0.30, 0.33, 0.36]),
            _box("cabinet", [0.65, 0.5, 1.35], [-1.0, 0.9, -0.08], [0.09, 0.20, 0.28]),
            _box(
                "cabinet_door",
                [0.55, 0.02, 1.12],
                [-1.0, 0.635, -0.04],
                [0.14, 0.3, 0.39],
            ),
            _box(
                "cabinet_handle",
                [0.025, 0.035, 0.23],
                [-0.79, 0.61, 0.0],
                [0.5] * 3,
                metallic=0.8,
                roughness=0.2,
            ),
        ]
        for shelf in range(3):
            z = -0.35 + shelf * 0.45
            objects.append(
                _box(
                    f"shelf_{shelf}",
                    [1.25, 0.38, 0.04],
                    [0.6, 1.05, z],
                    [0.2] * 3,
                    metallic=0.6,
                )
            )
            for column in range(4):
                objects.append(
                    _box(
                        f"bin_{shelf}_{column}",
                        [0.20, 0.25, 0.14],
                        [0.17 + column * 0.28, 1.02, z + 0.09],
                        [0.065, 0.16 + shelf * 0.06, 0.32],
                    )
                )
        cameras[0] = {
            "name": "external_front",
            "eye": [2.35, -2.8, 1.65],
            "target": [-0.05, 0.35, 0.15],
            "up": [0.0, 0.0, 1.0],
        }
    for index, obj in enumerate(objects, 1):
        obj["instance_id"] = index
    scene = {
        "schema_version": 1,
        "scene_id": name,
        "scene_version": "1.0",
        "units": "metres",
        "up_axis": "Z",
        "color_space": "linear_rec709",
        "objects": objects,
        "cameras": cameras,
        "camera_model": {"horizontal_fov_deg": 55.0, "near_m": 0.01, "far_m": 8.0},
        "lighting": {
            "direction": [-0.4, 0.3, -1.0],
            "color": [1.0, 1.0, 1.0],
            "dexsim_directional_intensity": 3.0,
            "isaac_directional_intensity": 3000.0,
            "dexsim_environment_intensity": 100.0,
            "isaac_dome_intensity": 150.0,
            "photometric_calibration": "not_qualified",
        },
        "robot_source": robot_source,
        "physics_steps": 0,
        "time_s": 0.0,
        "qualification": "scene_preview_only",
    }
    validate_scene(scene)
    path = destination / "scene.json"
    write_json(path, scene)
    return path


def validate_scene(scene: dict[str, Any]) -> None:
    """Reject incompatible units, duplicate IDs, and invalid camera geometry."""
    if (
        scene.get("schema_version") != 1
        or scene.get("units") != "metres"
        or scene.get("up_axis") != "Z"
    ):
        raise ValueError("Expected scene schema 1 in metres with Z up")
    if scene.get("physics_steps") != 0:
        raise ValueError("Rendering scene must not advance physics")
    objects = scene.get("objects", [])
    cameras = scene.get("cameras", [])
    if not objects or not cameras:
        raise ValueError("Scene requires objects and cameras")
    for field in ("uid", "instance_id"):
        values = [obj[field] for obj in objects]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate object {field}")
    names = [camera["name"] for camera in cameras]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate camera name")
    for camera in cameras:
        vectors = [camera[key] for key in ("eye", "target", "up")]
        if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("Camera vectors must have three finite values")
        direction = [
            b - a for a, b in zip(camera["eye"], camera["target"], strict=True)
        ]
        up = camera["up"]
        cross = [
            direction[1] * up[2] - direction[2] * up[1],
            direction[2] * up[0] - direction[0] * up[2],
            direction[0] * up[1] - direction[1] * up[0],
        ]
        if sum(x * x for x in cross) < 1e-12:
            raise ValueError("Camera look direction and up must be nondegenerate")


def load_scene(path: Path) -> dict[str, Any]:
    """Read a frozen scene and verify every referenced geometry hash."""
    path = Path(path).resolve()
    scene = json.loads(path.read_text(encoding="utf-8"))
    validate_scene(scene)
    for obj in scene["objects"]:
        geometry = obj["geometry"]
        if geometry["kind"] == "mesh":
            asset = (path.parent / geometry["path"]).resolve()
            if not asset.is_relative_to(path.parent):
                raise ValueError("Geometry must stay inside the frozen scene directory")
            if hashlib.sha256(asset.read_bytes()).hexdigest() != geometry["sha256"]:
                raise ValueError(f"Geometry hash mismatch: {obj['uid']}")
    return scene


def scene_fingerprint(scene: dict[str, Any]) -> str:
    """Return a portable fingerprint including geometry, poses and camera views."""
    return stable_hash(scene)


def camera_intrinsics(
    scene: dict[str, Any], width: int, height: int
) -> list[list[float]]:
    """Return square-pixel pinhole intrinsics for the specified output size."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    fx = width / (
        2.0 * math.tan(math.radians(scene["camera_model"]["horizontal_fov_deg"]) / 2.0)
    )
    return [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]]


def camera_world_pose(camera: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Convert a frozen look-at view to position and OpenGL ``xyzw`` rotation.

    Args:
        camera: Eye, target and up vectors in world coordinates.

    Returns:
        Position in metres and quaternion mapping camera axes into world axes.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    eye = np.asarray(camera["eye"], dtype=float)
    forward = np.asarray(camera["target"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], dtype=float))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    quaternion = Rotation.from_matrix(np.column_stack([right, up, -forward])).as_quat()
    return eye.tolist(), quaternion.tolist()
