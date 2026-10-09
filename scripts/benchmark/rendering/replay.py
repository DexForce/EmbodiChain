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

"""Frozen visual motion for pure rendering, without advancing a simulator.

Preparation uses the shared PyTorch Kinematics FK/IK implementation on CPU.
Loading ordinary rigid/robot frames uses only the standard library. The scripted
grasp is a kinematic visual fixture, not evidence of collision-free execution,
stable contact, or physical task success. External deformation input must keep
the approved topology and explicitly supply every vertex and normal frame.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json
from scripts.benchmark.rendering.scenes import (
    load_scene,
    scene_fingerprint,
    validate_scene,
)

__all__ = [
    "frame_at",
    "load_deformation_replay",
    "load_replay",
    "prepare_replay",
    "validate_replay",
]

_SCRIPTED = "scripted_kinematic_visual_replay"
_DEFORMATION = "external_frozen_deformation_replay"
_PHASES = (
    "approach",
    "descend",
    "close",
    "lift",
    "transfer",
    "place",
    "release",
    "retreat",
)
_KNOTS = (0.0, 0.20, 0.30, 0.36, 0.52, 0.72, 0.84, 0.90, 1.0)


def _phase(frame_id: int, count: int) -> tuple[int, float]:
    knots = [round(value * (count - 1)) for value in _KNOTS]
    index = next((i for i in range(8) if frame_id < knots[i + 1]), 7)
    fraction = (frame_id - knots[index]) / (knots[index + 1] - knots[index])
    return index, fraction * fraction * (3.0 - 2.0 * fraction)


def _identities(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        obj["uid"]: {
            "instance_id": obj["instance_id"],
            "geometry_sha256": stable_hash(obj["geometry"]),
            "material_sha256": stable_hash(obj["material"]),
        }
        for obj in scene["objects"]
    }


def _base_replay(scene: dict[str, Any], count: int, fps: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scene_sha256": scene_fingerprint(scene),
        "kind": _SCRIPTED,
        "fps": fps,
        "frame_count": count,
        "physics_steps": 0,
        "physical_validation": "not_performed",
        "source": {
            "method": "scripted rigid lift/transfer/place; no physical rollout",
            "scene_id": scene["scene_id"],
            "robot_source": deepcopy(scene.get("robot_source")),
        },
        "lineage": {
            "objects": _identities(scene),
            "frozen_cameras": {
                view["name"]: deepcopy(view) for view in scene["cameras"]
            },
            "camera_model": deepcopy(scene["camera_model"]),
            "joint_names": [],
            "joint_limits": {},
            "robot_link_names": [],
            "robot_visual_bindings": {},
            "camera_attachments": {},
            "state_shapes": {
                "object_count": len(scene["objects"]),
                "camera_count": len(scene["cameras"]),
                "object_transform": [4, 4],
                "robot_link_transform": [4, 4],
                "camera_vector": [3],
                "joint_count": 0,
                "robot_link_count": 0,
            },
            "random_seed": 0,
            "random_sampling": False,
            "motion_object_uid": "blue_block",
            "placement_target_uid": "tray_base",
            "phase_knots": [round(value * (count - 1)) for value in _KNOTS],
            "phase_names": list(_PHASES),
        },
        "frames": [],
    }


def _object_path(scene: dict[str, Any]) -> tuple[str, list[list[float]]]:
    objects = {obj["uid"]: obj for obj in scene["objects"]}
    if "blue_block" not in objects or "tray_base" not in objects:
        raise ValueError("Replay requires the approved blue_block and tray_base")
    block, tray = objects["blue_block"], objects["tray_base"]
    if block["geometry"].get("kind") != "box" or tray["geometry"].get("kind") != "box":
        raise ValueError("Replay grasp and placement fixtures must be boxes")
    start = [row[3] for row in block["world_from_object"][:3]]
    destination = [row[3] for row in tray["world_from_object"][:3]]
    destination[2] += (tray["geometry"]["size"][2] + block["geometry"]["size"][2]) / 2
    lifted = [start[0], start[1], start[2] + 0.18]
    above_tray = [destination[0], destination[1], destination[2] + 0.18]
    return "blue_block", [
        start,
        start,
        start,
        start,
        lifted,
        above_tray,
        destination,
        destination,
        destination,
    ]


def _empty_frame(
    scene: dict[str, Any], index: int, count: int, fps: float
) -> dict[str, Any]:
    phase, _ = _phase(index, count)
    return {
        "frame_id": index,
        "time_s": index / fps,
        "phase": _PHASES[phase],
        "objects": {
            obj["uid"]: deepcopy(obj["world_from_object"]) for obj in scene["objects"]
        },
        "cameras": {view["name"]: deepcopy(view) for view in scene["cameras"]},
        "joint_positions": {},
        "robot_links": {},
    }


def _resolve_urdf(path: Path | None) -> Path:
    if path is None:
        data_root = Path(
            os.environ.get(
                "EMBODICHAIN_DATA_ROOT", Path.home() / ".cache/embodichain_data"
            )
        )
        path = data_root / "extract/Franka/Panda/PandaWithHand.urdf"
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Frozen robot replay requires a local franka_urdf; preparation never downloads assets"
        )
    return path


def _robot_frames(scene: dict[str, Any], replay: dict[str, Any], urdf: Path) -> None:
    source = urdf.read_bytes()
    robot_source = scene["robot_source"]
    if hashlib.sha256(source).hexdigest() != robot_source["urdf_sha256"]:
        raise ValueError("Robot source URDF hash mismatch")
    import numpy as np
    import pytorch_kinematics as pk
    import torch
    from scipy.spatial.transform import Rotation, Slerp

    chain = pk.build_chain_from_urdf(source).to(dtype=torch.float64, device="cpu")
    parent = robot_source["wrist_parent"]
    serial = pk.build_serial_chain_from_urdf(source, parent).to(
        dtype=torch.float64, device="cpu"
    )
    names = chain.get_joint_parameter_names()
    arm_names = serial.get_joint_parameter_names()
    initial = robot_source["joint_positions"]
    if set(names) != set(initial):
        raise ValueError("Frozen joint names do not match the source URDF")
    world_from_robot = np.asarray(robot_source["world_from_robot"], dtype=float)
    initial_fk = chain.forward_kinematics(
        torch.tensor([[initial[name] for name in names]], dtype=torch.float64)
    )
    initial_links = {
        name: world_from_robot @ pose.get_matrix()[0].numpy()
        for name, pose in initial_fk.items()
    }
    initial_hand = initial_links[parent]
    lineage = replay["lineage"]
    lineage["joint_names"] = names
    lineage["robot_link_names"] = list(initial_links)
    lineage["state_shapes"].update(
        {"joint_count": len(names), "robot_link_count": len(initial_links)}
    )
    lineage["ik"] = {
        "solver": "pytorch_kinematics.PseudoInverseIK",
        "device": "cpu",
        "compile": False,
        "sequential_seed": "previous_frame",
        "initial_seed": deepcopy(initial),
        "random_retries": 0,
        "position_tolerance_m": 1e-5,
        "rotation_tolerance_rad": 1e-5,
        "maximum_iterations": 250,
        "learning_rate": 0.25,
        "joint_limits_enforced": True,
        "velocity_limits_enforced": False,
        "collision_validation": "not_performed",
    }
    root = ET.fromstring(source)
    lineage["joint_limits"] = {
        joint.attrib["name"]: [
            float(joint.find("limit").attrib[key]) for key in ("lower", "upper")
        ]
        for joint in root.findall("joint")
        if joint.attrib["name"] in names
    }
    lineage["joint_units"] = {
        joint.attrib["name"]: (
            "metres" if joint.attrib["type"] == "prismatic" else "radians"
        )
        for joint in root.findall("joint")
        if joint.attrib["name"] in names
    }
    objects = {obj["uid"]: obj for obj in scene["objects"]}
    for link in root.findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            uid = f"robot_{link.attrib['name']}_{index}"
            if uid not in objects:
                raise ValueError(f"Frozen robot visual missing: {uid}")
            local = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                local[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                local[:3, :3] = Rotation.from_euler(
                    "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]
                ).as_matrix()
            if not np.allclose(
                initial_links[link.attrib["name"]] @ local,
                objects[uid]["world_from_object"],
                atol=2e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"Frozen robot pose does not match source joints: {uid}"
                )
            lineage["robot_visual_bindings"][uid] = {
                "link": link.attrib["name"],
                "link_from_object": local.tolist(),
            }
    wrist = lineage["frozen_cameras"].get("wrist")
    if wrist is None:
        raise ValueError("Robot replay requires its frozen wrist camera")
    inverse_hand = np.linalg.inv(initial_hand)
    attachment = {
        "parent_link": parent,
        "eye": (inverse_hand @ np.r_[wrist["eye"], 1])[:3].tolist(),
        "target": (inverse_hand @ np.r_[wrist["target"], 1])[:3].tolist(),
        "up": (inverse_hand[:3, :3] @ wrist["up"]).tolist(),
    }
    lineage["camera_attachments"]["wrist"] = attachment
    uid, object_knots = _object_path(scene)
    block = objects[uid]
    grip_width = block["geometry"]["size"][1]
    if not 0 < grip_width < 0.08:
        raise ValueError("The scripted grasp must fit inside the 80 mm jaw opening")
    count, fps = replay["frame_count"], replay["fps"]
    rotation = Rotation.from_euler("y", -30, degrees=True).as_matrix() @ np.diag(
        [1.0, -1.0, -1.0]
    )
    hand_knots = []
    for index, position in enumerate(object_knots):
        hand = np.eye(4)
        hand[:3, :3] = rotation
        # TCP pinches near the top of the tall block; its 103.4 mm hand offset
        # is read from the source URDF's fixed hand-to-TCP transform.
        tcp_height = block["geometry"]["size"][2] / 2 - 0.025
        hand[:3, 3] = np.asarray(position) + [0.0, 0.0, tcp_height]
        tcp_local = np.linalg.inv(initial_hand) @ initial_links["fr3_hand_tcp"]
        hand[:3, 3] -= rotation @ tcp_local[:3, 3]
        if index in (1, 8):
            hand[2, 3] += 0.18
        hand_knots.append(hand)
    hand_knots[0] = initial_hand.copy()
    inverse_robot = np.linalg.inv(world_from_robot)
    seed = torch.tensor([[initial[name] for name in arm_names]], dtype=torch.float64)
    limits = torch.tensor(serial.get_joint_limits(), dtype=torch.float64).T
    max_position_error, max_rotation_error = 0.0, 0.0
    grasp_frame, release_frame = lineage["phase_knots"][3], lineage["phase_knots"][6]
    attached_from_object = None
    placed_object = None
    for index in range(count):
        phase, alpha = _phase(index, count)
        target = hand_knots[phase].copy()
        target[:3, 3] = (1 - alpha) * hand_knots[phase][:3, 3] + alpha * hand_knots[
            phase + 1
        ][:3, 3]
        target[:3, :3] = Slerp(
            [0, 1],
            Rotation.from_matrix(
                np.stack([hand_knots[phase][:3, :3], hand_knots[phase + 1][:3, :3]])
            ),
        )(alpha).as_matrix()
        if index:
            solver = pk.PseudoInverseIK(
                serial,
                retry_configs=seed,
                joint_limits=limits,
                max_iterations=250,
                lr=0.25,
                pos_tolerance=1e-5,
                rot_tolerance=1e-5,
                use_compile=False,
                early_stopping_no_improvement_patience=40,
            )
            result = solver.solve(
                pk.Transform3d(
                    matrix=torch.tensor(
                        (inverse_robot @ target)[None], dtype=torch.float64
                    ),
                    dtype=torch.float64,
                    device="cpu",
                )
            )
            if not bool(result.converged[0, 0]):
                raise ValueError(
                    f"Kinematic fixture IK failed at frame {index}; no replay was written"
                )
            seed = result.solutions[0, :1].clone()
        qpos = dict(initial)
        qpos.update(zip(arm_names, seed[0].tolist()))
        finger = initial["fr3_finger_joint1"]
        if phase == 2:
            finger = (1 - alpha) * finger + alpha * grip_width / 2
        elif 3 <= phase <= 5:
            finger = grip_width / 2
        elif phase == 6:
            finger = (1 - alpha) * grip_width / 2 + alpha * finger
        qpos["fr3_finger_joint1"] = qpos["fr3_finger_joint2"] = finger
        poses = chain.forward_kinematics(
            torch.tensor([[qpos[name] for name in names]], dtype=torch.float64)
        )
        links = {
            name: world_from_robot @ pose.get_matrix()[0].numpy()
            for name, pose in poses.items()
        }
        hand = links[parent]
        position_error = float(np.linalg.norm(hand[:3, 3] - target[:3, 3]))
        rotation_error = float(
            Rotation.from_matrix(hand[:3, :3] @ target[:3, :3].T).magnitude()
        )
        if position_error > 1e-5 or rotation_error > 1e-5:
            raise ValueError(f"FK verification failed at frame {index}")
        max_position_error = max(max_position_error, position_error)
        max_rotation_error = max(max_rotation_error, rotation_error)
        frame = _empty_frame(scene, index, count, fps)
        frame["joint_positions"] = qpos
        frame["robot_links"] = {name: pose.tolist() for name, pose in links.items()}
        for object_uid, binding in lineage["robot_visual_bindings"].items():
            frame["objects"][object_uid] = (
                links[binding["link"]] @ binding["link_from_object"]
            ).tolist()
        frame["cameras"]["wrist"] = {
            "name": "wrist",
            "eye": (hand @ np.r_[attachment["eye"], 1])[:3].tolist(),
            "target": (hand @ np.r_[attachment["target"], 1])[:3].tolist(),
            "up": (hand[:3, :3] @ attachment["up"]).tolist(),
        }
        if index == grasp_frame:
            attached_from_object = np.linalg.inv(hand) @ np.asarray(
                block["world_from_object"]
            )
            lineage["grasp"] = {
                "object_uid": uid,
                "parent_link": parent,
                "attach_frame": grasp_frame,
                "release_frame": release_frame,
                "hand_from_object": attached_from_object.tolist(),
                "jaw_opening_m": grip_width,
                "forward_tilt_deg": 30.0,
                "contact_validation": "not_performed",
            }
        if grasp_frame <= index <= release_frame:
            placed_object = hand @ attached_from_object
            frame["objects"][uid] = placed_object.tolist()
        elif index > release_frame:
            frame["objects"][uid] = placed_object.tolist()
        replay["frames"].append(frame)
    replay["source"].update(
        {
            "method": "CPU PyTorch Kinematics sequential IK and FK; scripted grasp attachment; no physical rollout",
            "pytorch_kinematics_version": version("pytorch-kinematics"),
            "torch_version": torch.__version__,
            "ik_device": "cpu",
            "ik_random_retries": 0,
            "maximum_fk_position_error_m": max_position_error,
            "maximum_fk_rotation_error_rad": max_rotation_error,
        }
    )


def prepare_replay(
    scene_path: Path,
    output: Path,
    frames: int = 100,
    fps: float = 30,
    *,
    franka_urdf: Path | None = None,
) -> Path:
    """Freeze the approved scene's deterministic kinematic visual sequence.

    Args:
        scene_path: Approved scene manifest, including immutable mesh hashes.
        output: New JSON file, or a directory receiving ``replay.json``.
        frames: Number of samples; at least twenty for distinct motion phases.
        fps: Timestamp cadence in frames per second, without wall-clock sleeps.
        franka_urdf: Local source URDF; its hash must match the frozen manifest.

    Returns:
        Absolute path to the validated replay. Existing files are not replaced.
    """
    if type(frames) is not int or frames < 20:
        raise ValueError("Replay requires at least twenty frames")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Replay fps must be finite and positive")
    path = Path(output).resolve()
    if path.suffix != ".json":
        path = path / "replay.json"
    if path.exists():
        raise FileExistsError(path)
    scene = load_scene(Path(scene_path))
    replay = _base_replay(scene, frames, float(fps))
    if scene.get("robot_source"):
        import torch

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            _robot_frames(scene, replay, _resolve_urdf(franka_urdf))
        finally:
            torch.set_num_threads(previous_threads)
    else:
        uid, knots = _object_path(scene)
        for index in range(frames):
            phase, alpha = _phase(index, frames)
            frame = _empty_frame(scene, index, frames, fps)
            for axis in range(3):
                frame["objects"][uid][axis][3] = (1 - alpha) * knots[phase][
                    axis
                ] + alpha * knots[phase + 1][axis]
            replay["frames"].append(frame)
    validate_replay(replay, scene)
    return write_json(path, replay)


def _vector(value: Any, size: int, label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError(f"{label} must have {size} finite numbers")


def _matrix(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a rigid 4x4 matrix")
    for row in value:
        _vector(row, 4, label)
    if value[3] != [0, 0, 0, 1]:
        raise ValueError(f"{label} must be a homogeneous transform")
    for i in range(3):
        for j in range(3):
            if abs(sum(value[k][i] * value[k][j] for k in range(3)) - (i == j)) > 2e-5:
                raise ValueError(f"{label} rotation must be orthonormal")
    determinant = sum(
        value[0][i]
        * (
            value[1][(i + 1) % 3] * value[2][(i + 2) % 3]
            - value[1][(i + 2) % 3] * value[2][(i + 1) % 3]
        )
        for i in range(3)
    )
    if abs(determinant - 1) > 2e-5:
        raise ValueError(f"{label} rotation must be right handed")


def _transform(
    matrix: list[list[float]], vector: list[float], weight: int
) -> list[float]:
    return [
        sum(row[i] * vector[i] for i in range(3)) + weight * row[3]
        for row in matrix[:3]
    ]


def validate_replay(
    replay: dict[str, Any], scene: dict[str, Any] | None = None
) -> None:
    """Validate cadence, identities, rigid poses, and frozen camera attachment.

    Args:
        replay: Parsed replay JSON, with full state at every frame.
        scene: Optional approved scene for source and geometry identity checks.

    Raises:
        ValueError: A source, frame, identity, pose, or timestamp is inconsistent.
    """
    if replay.get("schema_version") != 1 or replay.get("kind") not in (
        _SCRIPTED,
        _DEFORMATION,
    ):
        raise ValueError("Unsupported replay schema or kind")
    if replay.get("physics_steps") != 0:
        raise ValueError("Rendering replay cannot advance physics")
    if (
        replay["kind"] == _SCRIPTED
        and replay.get("physical_validation") != "not_performed"
    ):
        raise ValueError("Scripted visual replay cannot claim physical validation")
    fps, count = replay.get("fps"), replay.get("frame_count")
    _vector([fps], 1, "fps")
    if (
        fps <= 0
        or type(count) is not int
        or count < 1
        or not isinstance(replay.get("frames"), list)
        or count != len(replay["frames"])
    ):
        raise ValueError("Invalid replay cadence or frame_count")
    lineage = replay.get("lineage", {})
    if not lineage.get("objects") or not lineage.get("frozen_cameras"):
        raise ValueError("Replay requires object and camera lineage")
    if scene is not None:
        if replay.get("scene_sha256") != scene_fingerprint(scene):
            raise ValueError("Replay scene source hash mismatch")
        if (
            lineage["objects"] != _identities(scene)
            or lineage["camera_model"] != scene["camera_model"]
        ):
            raise ValueError("Replay immutable scene identity mismatch")
        if lineage["frozen_cameras"] != {
            view["name"]: view for view in scene["cameras"]
        }:
            raise ValueError("Replay frozen camera source mismatch")
        if replay.get("source", {}).get("robot_source") != scene.get("robot_source"):
            raise ValueError("Replay robot source mismatch")
    camera_names = set(lineage["frozen_cameras"])
    for index, frame in enumerate(replay["frames"]):
        if type(frame.get("frame_id")) is not int or frame["frame_id"] != index:
            raise ValueError("Replay frame IDs must be contiguous and ordered")
        _vector([frame.get("time_s")], 1, "time_s")
        if abs(frame["time_s"] - index / fps) > 1e-12:
            raise ValueError("Replay timestamps must follow the declared fps")
        for key, expected in (
            ("objects", lineage["objects"]),
            ("cameras", camera_names),
            ("joint_positions", lineage.get("joint_names", [])),
            ("robot_links", lineage.get("robot_link_names", [])),
        ):
            if not isinstance(frame.get(key), dict) or set(frame[key]) != set(expected):
                raise ValueError(f"Replay {key} identities changed at frame {index}")
        for key in ("objects", "robot_links"):
            for name, matrix in frame[key].items():
                _matrix(matrix, f"{key}/{name}")
        for name, value in frame["joint_positions"].items():
            _vector([value], 1, f"joint/{name}")
            limits = lineage["joint_limits"][name]
            if not limits[0] - 1e-8 <= value <= limits[1] + 1e-8:
                raise ValueError(f"Joint limit exceeded: {name}")
        validate_scene(
            {
                "schema_version": 1,
                "units": "metres",
                "up_axis": "Z",
                "physics_steps": 0,
                "objects": [{"uid": "placeholder", "instance_id": 1}],
                "cameras": list(frame["cameras"].values()),
            }
        )
        for name, camera in frame["cameras"].items():
            if camera.get("name") != name:
                raise ValueError("Camera name does not match its replay identity")
            attachment = lineage.get("camera_attachments", {}).get(name)
            if attachment is None:
                if camera != lineage["frozen_cameras"][name]:
                    raise ValueError(f"External camera must remain fixed: {name}")
            else:
                parent = frame["robot_links"][attachment["parent_link"]]
                for key in ("eye", "target", "up"):
                    expected = _transform(parent, attachment[key], int(key != "up"))
                    if any(abs(a - b) > 2e-6 for a, b in zip(camera[key], expected)):
                        raise ValueError(f"Wrist camera attachment changed: {name}")


def load_replay(path: Path, scene_path: Path | None = None) -> dict[str, Any]:
    """Read a replay and optionally bind it to an approved scene manifest.

    Args:
        path: Frozen replay JSON.
        scene_path: Scene manifest whose hash and assets must match the replay.

    Returns:
        Validated replay state. No simulator or kinematics dependency is loaded.
    """
    replay = read_json_object(Path(path))
    scene = load_scene(Path(scene_path)) if scene_path is not None else None
    validate_replay(replay, scene)
    if replay["kind"] == _DEFORMATION:
        if scene_path is None:
            raise ValueError("Deformation replay validation requires its frozen scene")
        _validate_deformations(replay, Path(scene_path), scene)
    return replay


def frame_at(replay: dict[str, Any], frame_id: int) -> dict[str, Any]:
    """Return an independent frozen frame without interpolation or wraparound.

    Args:
        replay: A replay already accepted by :func:`load_replay`.
        frame_id: Exact zero-based frame identity.

    Returns:
        Full object poses, camera poses, robot links, and joint positions.
    """
    if type(frame_id) is not int or not 0 <= frame_id < replay["frame_count"]:
        raise IndexError("Replay frame is outside the frozen sequence")
    return deepcopy(replay["frames"][frame_id])


def _validate_deformations(
    replay: dict[str, Any], scene_path: Path, scene: dict[str, Any]
) -> None:
    import numpy as np

    assets = replay.get("deformation_assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Deformation replay requires immutable topology assets")
    objects = {obj["uid"]: obj for obj in scene["objects"]}
    assigned_objects = set()
    for asset_id, asset in assets.items():
        uid = asset.get("object_uid")
        if not asset_id or uid not in objects or uid in assigned_objects:
            raise ValueError("Deformation assets must identify distinct frozen objects")
        assigned_objects.add(uid)
        geometry = objects[uid]["geometry"]
        if (
            geometry.get("kind") != "mesh"
            or asset.get("geometry_sha256") != geometry["sha256"]
        ):
            raise ValueError("Deformation topology asset hash mismatch")
        with np.load(scene_path.parent / geometry["path"], allow_pickle=False) as data:
            vertices, triangles = data["vertices"], data["triangles"]
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.isfinite(vertices).all()
            ):
                raise ValueError("Invalid frozen deformation vertices")
            if (
                triangles.ndim != 2
                or triangles.shape[1] != 3
                or triangles.dtype.kind not in "iu"
                or np.any(triangles < 0)
                or np.any(triangles >= len(vertices))
            ):
                raise ValueError("Invalid frozen deformation topology")
            if type(asset.get("vertex_count")) is not int or asset[
                "vertex_count"
            ] != len(vertices):
                raise ValueError(
                    "Deformation vertex_count differs from frozen topology"
                )
    for frame in replay["frames"]:
        states = frame.get("deformations", {})
        if set(states) != set(assets):
            raise ValueError("Every deformation frame must supply every topology asset")
        for asset_id, state in states.items():
            # Frame buffers are in object-local metres. Faces, IDs, materials,
            # and geometry paths remain owned by the approved scene manifest.
            if set(state) != {"vertices", "normals"}:
                raise ValueError(
                    "Deformation frames supply vertices and normals only; topology is immutable"
                )
            for key in ("vertices", "normals"):
                values = np.asarray(state[key], dtype=float)
                if (
                    values.shape != (assets[asset_id]["vertex_count"], 3)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        f"Invalid deformation {key} shape or finite values"
                    )
                if key == "normals" and not np.allclose(
                    np.linalg.norm(values, axis=1), 1, atol=1e-3, rtol=0
                ):
                    raise ValueError("Deformation normals must be unit vectors")


def load_deformation_replay(path: Path | None, scene_path: Path) -> dict[str, Any]:
    """Load external frozen vertices, retaining missing input as unsupported.

    Each ``deformation_assets[asset_id]`` names a frozen ``object_uid``, its
    mesh ``geometry_sha256``, and ``vertex_count``. Every frame provides
    ``deformations[asset_id]`` with object-local ``vertices`` and ``normals``
    arrays, both shape ``(vertex_count, 3)``. Triangle indices never vary.

    Args:
        path: External replay JSON; absent input never creates a synthetic run.
        scene_path: Approved scene that owns topology and segmentation IDs.

    Returns:
        Validated external replay, or an explicit unsupported/not-run record.
        Invalid existing data raises instead of being classified as missing.
    """
    if path is None or not Path(path).exists():
        return {
            "status": "unsupported",
            "run_status": "not_run",
            "reason": "missing_external_frozen_deformation_sequence",
        }
    if not Path(path).is_file():
        raise ValueError("External deformation input must be a regular JSON file")
    replay = load_replay(Path(path), Path(scene_path))
    if replay["kind"] != _DEFORMATION:
        raise ValueError(
            "Soft-body rendering requires an external frozen deformation sequence"
        )
    return replay
