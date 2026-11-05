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
"""Dynamic Franka floor pickup and single-post rope wrapping.

Only robot joint drive targets move during physics. The rope's first segment
is fixed at the visible floor clamp; its free end is grasped by finger contact.
No robot/rope attachment or rope state assignment is used during simulation.
Material and fingertip parameters are demonstration settings, not calibration.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch

__all__ = ["main", "create_replay_scene", "restore_replay_frame", "replay_camera_views"]
_ARM_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
_JOINT_NAMES = _ARM_NAMES + ["fr3_finger_joint1", "fr3_finger_joint2"]


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _texture_reference(tokens: list[str]) -> str:
    """Remove standard MTL map options before resolving the texture filename."""
    one_value = {
        "-blendu",
        "-blendv",
        "-cc",
        "-clamp",
        "-imfchan",
        "-texres",
        "-type",
        "-bm",
        "-boost",
    }
    index = 1
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        if option in one_value:
            index += 1
        elif option == "-mm":
            index += 2
        elif option in {"-o", "-s", "-t"}:
            count = 0
            while index < len(tokens) and count < 3:
                try:
                    float(tokens[index])
                except ValueError:
                    break
                index += 1
                count += 1
        else:
            raise ValueError(f"Unsupported MTL texture option: {option}")
    if index >= len(tokens):
        raise ValueError("MTL texture declaration has no filename")
    return " ".join(tokens[index:])


def _mesh_paths(root: ET.Element, source: Path) -> list[Path]:
    """Resolve the generated model's local mesh and material dependency closure.

    The generated model preserves the source directory layout. References must
    therefore stay beneath the source URDF directory; package and remote URIs
    are not supported by this standalone example.
    """
    directory = source.parent.resolve()
    pending = [(directory, mesh.get("filename", "")) for mesh in root.iter("mesh")]
    dependencies: set[Path] = set()
    while pending:
        parent, reference = pending.pop()
        uri = urlsplit(reference)
        if uri.scheme or uri.netloc or not uri.path:
            raise ValueError(
                f"Asset references must be local relative paths: {reference!r}"
            )
        relative = Path(unquote(uri.path))
        if relative.is_absolute():
            raise ValueError(f"Asset references must be relative: {reference!r}")
        path = (parent / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(
                f"Asset dependency escapes the source directory: {reference!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in dependencies:
            continue
        dependencies.add(path)
        if path.suffix.lower() in {".obj", ".mtl"}:
            for line in path.read_text().splitlines():
                tokens = shlex.split(line, comments=True)
                if not tokens:
                    continue
                if path.suffix.lower() == ".obj" and tokens[0] == "mtllib":
                    pending.extend((path.parent, name) for name in tokens[1:])
                elif path.suffix.lower() == ".mtl" and (
                    tokens[0].startswith("map_")
                    or tokens[0].lower() in {"bump", "disp", "decal", "norm", "refl"}
                ):
                    pending.append((path.parent, _texture_reference(tokens)))
        elif path.suffix.lower() == ".dae":
            for image in ET.parse(path).findall(".//{*}image"):
                for value in image.iter():
                    if value.tag.rsplit("}", 1)[-1] in {"init_from", "ref"}:
                        reference = (value.text or "").strip()
                        if reference and not reference.startswith("#"):
                            pending.append((path.parent, reference))
    return sorted(dependencies)


def _asset_archives(reference: str) -> list[dict[str, str]]:
    """Record available registry archive identities without initiating downloads."""
    if Path(reference).is_absolute():
        return []
    from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

    prefix = reference.split("/", 1)[0]
    directory = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "download" / prefix
    archives = []
    try:
        paths = sorted(directory.glob("*.zip"))
    except OSError as error:
        return [{"metadata_error": str(error)}]
    for path in paths:
        try:
            if path.is_file():
                archives.append(
                    {
                        "filename": path.name,
                        "md5": _hash(path, "md5"),
                        "sha256": _hash(path),
                    }
                )
        except OSError as error:
            archives.append({"filename": path.name, "metadata_error": str(error)})
    return archives


def _runtime_versions() -> dict[str, Any]:
    """Collect reproducibility metadata without constructing a simulation world."""
    packages = {}
    metadata_errors = {}
    for name in (
        "embodichain",
        "dexsim-engine",
        "newton",
        "warp-lang",
        "torch",
        "scipy",
        "pytorch-kinematics",
    ):
        try:
            packages[name] = metadata.version(name)
        except Exception as error:
            packages[name] = None
            metadata_errors[name] = str(error)
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "metadata_errors": metadata_errors,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [],
    }
    try:
        import dexsim

        result["dexsim_commit"] = getattr(dexsim, "__commit_id__", None)
    except Exception as error:
        result["dexsim_metadata_error"] = str(error)
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if query.returncode == 0:
            for row in query.stdout.splitlines():
                values = [item.strip() for item in row.split(",")]
                if len(values) == 4:
                    result["gpus"].append(
                        dict(zip(("index", "name", "uuid", "driver_version"), values))
                    )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def _source_revision() -> str | None:
    """Report the source checkout revision when available, including from any cwd."""
    source = Path(__file__).resolve()
    for parent in source.parents:
        if not (parent / "pyproject.toml").is_file():
            continue
        if source.relative_to(parent).parts[0] != "examples":
            continue
        try:
            return subprocess.check_output(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
    return None


def _validate_config(config: dict[str, Any]) -> None:
    """Reject inputs that break the standalone control and geometry contracts."""

    def finite_values(value: Any, name: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                finite_values(item, f"{name}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite_values(item, f"{name}[{index}]")
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    finite_values(config, "config")
    for name, minimum in (
        ("control_hz", 1),
        ("substeps", 4),
        ("iterations", 1),
        ("cable_segments", 8),
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["substeps"] % 4:
        raise ValueError("substeps must be divisible by four for contact sampling")
    for name in (
        "cable_radius_m",
        "cable_density_kg_m3",
        "post_radius_m",
        "post_height_m",
        "finger_pad_width_m",
        "duration_s",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
    for name in (
        "pre_roll_s",
        "linear_stiffness_N_m",
        "bend_stiffness_Nm_rad",
        "twist_stiffness_Nm_rad",
        "bend_damping_Nms_rad",
        "contact_ke_N_m",
        "contact_kd_Ns_m",
        "contact_gap_m",
        "contact_match_distance_m",
    ):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative number")
    if config["duration_s"] < 36:
        raise ValueError("duration_s must include the full 36-second motion and hold")
    if not np.isclose(
        config["duration_s"] * config["control_hz"],
        round(config["duration_s"] * config["control_hz"]),
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("duration_s must align to the control grid")
    if (
        not 0
        <= config["grasp_complete_s"]
        < config["lift_complete_s"]
        <= config["duration_s"] - 2
    ):
        raise ValueError("Grasp and lift checks must precede the final two-second hold")
    for name, shape in (
        ("home_qpos", (7,)),
        ("post_center_xy_m", (2,)),
        ("finger_pad_center_m", (3,)),
        ("finger_pad_size_m", (3,)),
    ):
        vector = np.asarray(config[name], dtype=float)
        if vector.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain finite values")
    if np.any(np.asarray(config["finger_pad_size_m"], dtype=float) <= 0):
        raise ValueError("finger_pad_size_m must contain positive dimensions")
    points = np.asarray(config["cable_waypoints_m"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("cable_waypoints_m must contain at least three XYZ waypoints")
    if not np.isfinite(points).all():
        raise ValueError("cable_waypoints_m must contain finite values")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
        raise ValueError("Consecutive cable waypoints must be distinct")
    if (
        np.linalg.norm(
            points[0, :2] - np.asarray(config["post_center_xy_m"], dtype=float)
        )
        <= 0
    ):
        raise ValueError(
            "The cable anchor and post center must have distinct XY coordinates"
        )
    total_length = float(
        np.linalg.norm(np.diff(_cable_points(config), axis=0), axis=1).sum()
    )
    inset = config["grasp_inset_m"]
    if (
        isinstance(inset, bool)
        or not isinstance(inset, (int, float))
        or not 0 < inset < total_length
    ):
        raise ValueError(
            "grasp_inset_m must lie strictly inside the sampled cable length"
        )
    thresholds = config["validation"]
    if not isinstance(thresholds, dict):
        raise ValueError("validation must be a mapping of acceptance thresholds")
    for name, value in thresholds.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"validation.{name} must be a finite nonnegative number")
    if not isinstance(config["robot_urdf"], str) or not config["robot_urdf"].strip():
        raise ValueError("robot_urdf must be an asset reference or absolute URDF path")


def _box(
    link: ET.Element,
    name: str,
    xyz: Any,
    size: Any,
    color: str,
    *,
    collision: bool = True,
) -> None:
    for kind in ("visual", "collision") if collision else ("visual",):
        element = ET.SubElement(link, kind, name=name + "_" + kind)
        ET.SubElement(element, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
        ET.SubElement(
            ET.SubElement(element, "geometry"), "box", size=" ".join(map(str, size))
        )
        if kind == "visual":
            ET.SubElement(
                ET.SubElement(element, "material", name=name + "_material"),
                "color",
                rgba=color,
            )


def _prepare_models(config: dict[str, Any], out: Path) -> dict[str, Path]:
    from embodichain.data import get_data_path

    generated = out / "generated_assets"
    generated.mkdir()
    robot_directory = generated / "franka"
    robot_directory.mkdir()
    reference = config["robot_urdf"]
    source = Path(get_data_path(reference)).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Robot URDF does not exist: {source}")
    tree = ET.parse(source)
    root = tree.getroot()
    for name in ("fr3_leftfinger", "fr3_rightfinger"):
        link = root.find(f"link[@name='{name}']")
        assert link is not None
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
        _box(
            link,
            name + "_stem",
            (0, 0.004, 0.02),
            (0.018, 0.008, 0.04),
            ".08 .09 .11 1",
        )
        _box(
            link,
            name + "_tip",
            config["finger_pad_center_m"],
            config["finger_pad_size_m"],
            ".16 .18 .21 1",
        )
    dependencies = []
    for path in _mesh_paths(root, source):
        relative = path.relative_to(source.parent)
        target = robot_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        dependencies.append(
            {"path": target.relative_to(out).as_posix(), "sha256": _hash(target)}
        )
    paths = {"robot": robot_directory / "franka_rope.urdf"}
    tree.write(paths["robot"], encoding="utf-8", xml_declaration=True)
    _json(
        generated / "provenance.json",
        {
            "asset_reference": reference,
            "cached_registry_archives": _asset_archives(reference),
            "robot_source": str(source.resolve()),
            "robot_source_sha256": _hash(source),
            "generated_robot": paths["robot"].relative_to(out).as_posix(),
            "generated_robot_sha256": _hash(paths["robot"]),
            "dependencies": dependencies,
            "modifications": "Flat finger pads for direct floor rope grasp; original joint names, origins, axes, limits and masses retained. No moving rope attachment.",
            "finger_pad_width_m": config["finger_pad_width_m"],
            "finger_pad_center_m": config["finger_pad_center_m"],
            "finger_pad_size_m": config["finger_pad_size_m"],
        },
    )
    return paths


def _cable_points(config: dict[str, Any]) -> np.ndarray:
    points = np.asarray(config["cable_waypoints_m"], float)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    curve = make_interp_spline(distances, points, k=2)(
        np.linspace(0, distances[-1], 4000)
    )
    curve[:, 2] = np.maximum(
        curve[:, 2], config["floor_height_m"] + config["cable_radius_m"]
    )
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    result = np.column_stack(
        [
            np.interp(
                np.linspace(0, arc[-1], config["cable_segments"] + 1), arc, curve[:, i]
            )
            for i in range(3)
        ]
    )
    # The fixed root capsule exits its clamp toward the post. Preserve the
    # sampled first-segment length while removing spline endpoint tangent error.
    direction = np.r_[np.asarray(config["post_center_xy_m"]) - result[0, :2], 0.0]
    direction /= np.linalg.norm(direction)
    result[1] = result[0] + direction * np.linalg.norm(result[1] - result[0])
    return result.astype(np.float32)


def _scene(
    config: dict[str, Any],
    paths: dict[str, Path],
    controls: np.ndarray,
    out: Path,
    rendered: bool,
) -> tuple[Any, ...]:
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        CableObjectCfg,
        CablePhysicsCfg,
        DenoisingCfg,
        JointDrivePropertiesCfg,
        LightCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        RobotCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.shapes import CubeCfg, CylinderCfg
    from embodichain.lab.sim.sensors import (
        CameraCfg,
        ContactSensorCfg,
        ArticulationContactFilterCfg,
    )

    dt = 1 / (config["control_hz"] * config["substeps"])
    physics = NewtonPhysicsCfg(
        physics_dt=dt,
        num_substeps=1,
        gravity=(0, 0, -9.81),
        solver_cfg={
            "solver_type": "dexuni",
            "joint_mode": "dynamic",
            "coupling": "two_way",
            "mujoco_articulation_paths": ("arena_0/franka",),
            "vbd_options": {
                "iterations": config["iterations"],
                "rigid_contact_history": config.get("rigid_contact_history", True),
                "rigid_contact_hard": config.get("rigid_contact_hard", False),
                "rigid_joint_linear_ke": config.get("rigid_joint_linear_ke", 100000.0),
                "rigid_joint_angular_ke": config.get("rigid_joint_angular_ke", 10.0),
                "friction_epsilon": config.get("friction_epsilon", 1e-05),
            },
            "collision_options": {
                "contact_matching": "sticky",
                "contact_matching_pos_threshold": config["contact_match_distance_m"],
                "rigid_contact_max": 4096,
            },
        },
        collision_cfg=None,
        sync_to_renderer=False,
        use_cuda_graph=True,
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            gpu_id=0,
            num_envs=1,
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="compact",
            physics_cfg=physics,
            render_cfg=RenderCfg(
                renderer=(
                    config["render"].get("renderer", "hybrid")
                    if rendered
                    else "no-render"
                ),
                spp=config["render"]["spp"],
                denoising=DenoisingCfg(window="off", offscreen="optix"),
                tone_mapping_enabled=config["render"].get(
                    "tone_mapping_enabled", False
                ),
                tone_mapping_exposure=config["render"].get("exposure", 1.0),
                min_bounces=config["render"].get("min_bounces", 4),
                max_bounces=config["render"].get("max_bounces", 8),
            ),
        )
    )

    robot = cable = sensors = None
    cameras = []
    try:

        def contact(mu: float, ke: float | None = None) -> Any:
            return RigidBodyPhysicsCfg.from_dict(
                {
                    "material_props": {
                        "backend": "newton",
                        "dynamic_friction": mu,
                        "ke": config["contact_ke_N_m"] if ke is None else ke,
                        "kd": config["contact_kd_Ns_m"],
                    },
                    "collision_props": {
                        "backend": "newton",
                        "margin": 0.0,
                        "gap": config["contact_gap_m"],
                    },
                }
            )

        def box(uid: str, p: Any, size: Any, color: Any, yaw_deg: float = 0.0) -> Any:
            return sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    body_type="static",
                    init_pos=tuple(p),
                    init_rot=(0.0, 0.0, yaw_deg),
                    shape=CubeCfg(
                        size=tuple(size),
                        visual_material=VisualMaterialCfg(
                            uid=uid + "_mat", base_color=color, roughness=0.65
                        ),
                    ),
                    attrs=contact(config.get("floor_friction", 0.2)),
                )
            )

        # A raised physical slab clears the hidden default plane's contact skin.
        # Robot and all authored world positions share this base-height offset.
        sim.set_ground_plane_visibility(False)
        floor = config.get("floor_height_m", 0.0)
        box("table", (0.35, 0, floor - 0.025), (1.2, 1.05, 0.05), (0.33, 0.37, 0.41, 1))
        anchor = np.asarray(config["cable_waypoints_m"][0])
        center = np.asarray(config["post_center_xy_m"])
        anchor_direction = np.r_[center - anchor[:2], 0.0]
        anchor_direction /= np.linalg.norm(anchor_direction)
        box(
            "anchor_clamp",
            anchor - 0.006 * anchor_direction - [0, 0, 0.004],
            (0.028, 0.018, 0.008),
            (0.08, 0.11, 0.15, 1),
            yaw_deg=float(
                np.rad2deg(np.arctan2(anchor_direction[1], anchor_direction[0]))
            ),
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="post",
                body_type="static",
                init_pos=(
                    float(center[0]),
                    float(center[1]),
                    floor + config["post_height_m"] / 2,
                ),
                shape=CylinderCfg(
                    radius=config["post_radius_m"],
                    height=config["post_height_m"],
                    visual_material=VisualMaterialCfg(
                        uid="post_mat",
                        base_color=(0.025, 0.45, 0.39, 1),
                        roughness=0.38,
                    ),
                ),
                attrs=contact(config["post_friction"]),
            )
        )
        robot = sim.add_robot(
            RobotCfg(
                uid="franka",
                fpath=str(paths["robot"]),
                init_pos=(0.0, 0.0, floor),
                init_qpos=controls[0],
                control_parts={
                    "arm": _ARM_NAMES,
                    "eef": _JOINT_NAMES[7:9],
                },
                solver_cfg=None,
                joint_drive_props=JointDrivePropertiesCfg(
                    stiffness={"arm": 6000.0, "eef": 6000.0},
                    damping={"arm": 150.0, "eef": 35.0},
                    max_effort={"arm": 87.0, "eef": 20.0},
                    target_mode="position_velocity",
                ),
                newton_gravity_compensation=1.0,
                attrs=contact(1.2),
            )
        )
        points = _cable_points(config)
        cable = sim.add_cable_object(
            CableObjectCfg(
                uid="cable",
                points=points,
                radius=config["cable_radius_m"],
                fixed_segments=(0,),
                collision_filter_span=2,
                attrs=CablePhysicsCfg(
                    density=config["cable_density_kg_m3"],
                    stretch_stiffness=config["linear_stiffness_N_m"],
                    shear_stiffness=config["linear_stiffness_N_m"],
                    bend_stiffness=config["bend_stiffness_Nm_rad"],
                    twist_stiffness=config["twist_stiffness_Nm_rad"],
                    stretch_damping=0.03,
                    shear_damping=0.03,
                    bend_damping=config["bend_damping_Nms_rad"],
                    twist_damping=config["bend_damping_Nms_rad"],
                    collision_props=contact(config["rope_friction"]).collision_props,
                    material_props=contact(config["rope_friction"]).material_props,
                ),
                visual_material=VisualMaterialCfg(
                    uid="white_rope", base_color=(0.93, 0.94, 0.92, 1), roughness=0.6
                ),
            )
        )
        cameras = []
        if rendered:
            for name, view, fov in (
                ("overview", config["camera"], 48),
                ("detail", config["detail_camera"], 42),
            ):
                width, height = (config["render"]["width"], config["render"]["height"])
                focal = height / (2 * np.tan(np.deg2rad(fov / 2)))
                cameras.append(
                    sim.add_sensor(
                        CameraCfg(
                            uid=name,
                            width=width,
                            height=height,
                            near=0.01,
                            far=4,
                            intrinsics=(focal, focal, width / 2, height / 2),
                            enable_color=True,
                            enable_depth=False,
                            enable_mask=False,
                            extrinsics=CameraCfg.ExtrinsicsCfg(
                                eye=tuple(view["eye_m"]), target=tuple(view["target_m"])
                            ),
                        )
                    )
                )
            for uid, position, power in (
                ("key", (0.4, -0.6, 1.6), 14),
                ("fill", (0.1, 0.7, 1.1), 9),
            ):
                sim.add_light(
                    LightCfg(
                        uid=uid, init_pos=position, light_type="point", intensity=power
                    )
                )
        sim.prepare()
        sensors = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["post"],
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid="franka", link_name_list=[]
                    ),
                ],
                filter_need_both_actor=False,
                max_contacts_per_env=4096,
            )
        )
        if robot.joint_names != _JOINT_NAMES:
            raise RuntimeError(f"Runtime joint order differs: {robot.joint_names}")
        np.savez_compressed(
            out / "scene_inputs.npz", points=points, anchor=anchor, post_center=center
        )
        return (sim, robot, cable, sensors, cameras, points)
    except BaseException:
        cameras.clear()
        robot = cable = sensors = None
        try:
            sim.destroy(exit_process=False)
        finally:
            sim = None
            SimulationManager.flush_cleanup_queue()
        raise


def _motion(
    config: dict[str, Any], robot_path: Path, out: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Plan floor pickup, a configured orbit, a small tangent pull and held finish.

    Scene coordinates use home_qpos, cable_waypoints_m and post_center_xy_m.
    Optional keys and defaults are written to motion_validation.json. Returned
    controls contain no pre-roll samples. Targets are world-frame hand TCP poses.
    """
    import pytorch_kinematics as pk

    torch.set_num_threads(1)
    arm_names = [f"fr3_joint{i}" for i in range(1, 8)]
    joint_names = arm_names + ["fr3_finger_joint1", "fr3_finger_joint2"]
    floor = float(config.get("floor_height_m", 0.002))
    fps = int(config.get("control_hz", 60))
    duration = float(config.get("duration_s", 36.0))
    if duration < 36.0:
        raise ValueError(
            "At least 36 seconds are required, including a two-second hold"
        )
    scale = duration / 36.0
    radius = float(config.get("orbit_radius_m", 0.12))
    wrap_degrees = float(config.get("orbit_degrees", 450.0))
    direction = int(config.get("orbit_direction", 1))
    if direction not in (-1, 1):
        raise ValueError("orbit_direction must be +1 (CCW) or -1 (CW)")
    theta_start = np.deg2rad(float(config.get("orbit_start_deg", -90.0)))
    theta_end = theta_start + direction * np.deg2rad(wrap_degrees)
    center = np.asarray(config["post_center_xy_m"], dtype=float)
    pickup_end = np.asarray(config["cable_waypoints_m"][-1], dtype=float)
    grasp_offset = np.asarray(
        config.get("grasp_offset_world_m", [-0.020, 0, 0]), dtype=float
    )
    z_start = float(config.get("orbit_start_z_m", floor + 0.055))
    z_end = float(config.get("orbit_end_z_m", floor + 0.075))
    lift_height = float(config.get("pickup_lift_m", 0.09))
    radial_pull = float(config.get("tighten_radial_m", 0.018))
    tangent_pull = float(config.get("tighten_tangent_m", 0.020))
    open_gap = float(config.get("open_half_gap_m", 0.04))
    closed_gap = float(config.get("grip_half_gap_m", config["cable_radius_m"] - 0.0005))
    if not 0 < closed_gap < open_gap <= 0.04:
        raise ValueError("Invalid Franka gripper half gaps")
    hand_r = Rotation.from_euler(
        "z", float(config.get("grasp_yaw_deg", 0.0)), degrees=True
    ).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    chain = pk.build_serial_chain_from_urdf(robot_path.read_bytes(), "fr3_hand_tcp").to(
        dtype=torch.float64, device="cpu"
    )
    if chain.get_joint_parameter_names() != arm_names:
        raise ValueError("Unexpected Franka arm joint order")
    limits = np.asarray(chain.get_joint_limits(), dtype=float).T
    lo, hi = limits[:, 0] + 1e-5, limits[:, 1] - 1e-5
    home = np.asarray(config["home_qpos"], dtype=float)

    def pose(p: np.ndarray, rotation: np.ndarray = hand_r) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = rotation, p
        return result

    def fk(q: np.ndarray) -> np.ndarray:
        result = (
            chain.forward_kinematics(torch.tensor(q, dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        result[2, 3] += floor
        return result

    def ik(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.copy()
        for _ in range(45):
            current = fk(q)
            dp = target[:3, 3] - current[:3, 3]
            dr = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            if np.linalg.norm(dp) < 2e-7 and np.linalg.norm(dr) < 2e-6:
                return q
            jac = chain.jacobian(torch.tensor(q, dtype=torch.float64))[0].numpy()
            jac[3:] *= 0.15
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + np.eye(6) * 1e-8, np.r_[dp, 0.15 * dr]
            )
            q = np.clip(
                q + delta * min(1.0, 0.1 / max(np.max(np.abs(delta)), 1e-9)), lo, hi
            )

        def residual(candidate: np.ndarray) -> np.ndarray:
            current = fk(candidate)
            return np.r_[
                current[:3, 3] - target[:3, 3],
                0.15
                * Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]

        solved = least_squares(
            residual,
            q,
            bounds=(lo, hi),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=160,
        )
        if np.linalg.norm(residual(solved.x)) > 1e-5:
            raise RuntimeError(
                f"Unreachable wrap TCP {target[:3, 3]}: {residual(solved.x)}"
            )
        return solved.x

    def smooth(u: float) -> float:
        u = float(np.clip(u, 0, 1))
        return u**3 * (10 - 15 * u + 6 * u * u)

    def end_at(theta: float, height: float) -> np.ndarray:
        return np.r_[center + radius * np.array([np.cos(theta), np.sin(theta)]), height]

    home_pose = fk(home)
    pickup = pose(pickup_end + grasp_offset)
    above = pickup.copy()
    above[2, 3] += lift_height
    orbit_start = pose(end_at(theta_start, z_start) + grasp_offset)
    orbit_end = pose(end_at(theta_end, z_end) + grasp_offset)
    radial = np.array([np.cos(theta_end), np.sin(theta_end), 0.0])
    tangent = direction * np.array([-np.sin(theta_end), np.cos(theta_end), 0.0])
    tightened = orbit_end.copy()
    tightened[:3, 3] += radial_pull * radial + tangent_pull * tangent
    pickup_keys = [
        (0.0, home_pose, open_gap, "HOME"),
        (1.0, home_pose, open_gap, "HOME"),
        (3.0, above, open_gap, "APPROACH FLOOR END"),
        (4.0, pickup, open_gap, "LOWER TO ROPE"),
        (5.0, pickup, closed_gap, "GRASP ROPE"),
        (7.0, above, closed_gap, "LIFT AND ORGANIZE SLACK"),
        (10.0, orbit_start, closed_gap, "TANGENTIAL APPROACH"),
    ]
    times = np.arange(round(duration * fps) + 1) / fps
    targets, labels, gaps = [], [], []
    for t in times:
        clock = t / scale
        if clock <= 10.0:
            index = min(
                max(
                    int(
                        np.searchsorted(
                            [k[0] for k in pickup_keys], clock, side="right"
                        )
                    )
                    - 1,
                    0,
                ),
                len(pickup_keys) - 2,
            )
            a, b = pickup_keys[index : index + 2]
            u = smooth((clock - a[0]) / (b[0] - a[0]))
            target = pose(
                (1 - u) * a[1][:3, 3] + u * b[1][:3, 3],
                Slerp([0, 1], Rotation.from_matrix([a[1][:3, :3], b[1][:3, :3]]))(
                    [u]
                ).as_matrix()[0],
            )
            gap, label = (1 - u) * a[2] + u * b[2], b[3]
        elif clock <= 28.0:
            u = smooth((clock - 10.0) / 18.0)
            theta = theta_start + (theta_end - theta_start) * u
            target = pose(end_at(theta, (1 - u) * z_start + u * z_end) + grasp_offset)
            gap, label = closed_gap, "WRAP ROPE AROUND PILLAR"
        elif clock <= 32.0:
            u = smooth((clock - 28.0) / 4.0)
            target = pose((1 - u) * orbit_end[:3, 3] + u * tightened[:3, 3])
            gap, label = closed_gap, "GENTLE TANGENTIAL TIGHTEN"
        elif clock < 34.0:
            target, gap, label = tightened.copy(), closed_gap, "SETTLE WRAP"
        else:
            target, gap, label = tightened.copy(), closed_gap, "HOLD COMPLETED WRAP"
        targets.append(target)
        gaps.append(gap)
        labels.append(label)
    targets = np.asarray(targets)
    qpath = np.zeros((len(times), 9), dtype=np.float32)
    q = home.copy()
    for i, target in enumerate(targets):
        q = ik(target, q)
        qpath[i, :7], qpath[i, 7:] = q, gaps[i]
    actual = (
        chain.forward_kinematics(torch.tensor(qpath[:, :7], dtype=torch.float64))
        .get_matrix()
        .numpy()
    )
    actual[:, 2, 3] += floor
    pe = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=1)
    re = Rotation.from_matrix(
        targets[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    vmax = np.max(np.abs(np.diff(qpath[:, :7], axis=0) * fps), axis=0)
    root = ET.parse(robot_path).getroot()
    velocity_limits = np.array(
        [
            float(root.find(f"joint[@name='{name}']/limit").get("velocity"))
            for name in arm_names
        ]
    )
    joint_limits_ok = bool(
        np.all(qpath[:, :7] >= lo - 1e-6) and np.all(qpath[:, :7] <= hi + 1e-6)
    )
    passed = bool(
        pe.max() < 1e-5
        and re.max() < 1e-4
        and np.all(vmax <= velocity_limits)
        and joint_limits_ok
    )
    report = {
        "passed": passed,
        "ik_position_max_m": float(pe.max()),
        "ik_rotation_max_rad": float(re.max()),
        "joint_velocity_max_rad_s": vmax.tolist(),
        "joint_velocity_limits_rad_s": velocity_limits.tolist(),
        "joint_limits_ok": joint_limits_ok,
        "minimum_joint_limit_margin_rad": float(
            np.min(np.minimum(qpath[:, :7] - lo, hi - qpath[:, :7]))
        ),
        "duration_s": duration,
        "hold_start_s": 34.0 * scale,
        "hold_duration_s": 2.0 * scale,
        "orbit_start_s": 10.0 * scale,
        "orbit_end_s": 28.0 * scale,
        "orbit_degrees": wrap_degrees,
        "orbit_direction": direction,
        "orbit_radius_m": radius,
        "orbit_start_z_m": z_start,
        "orbit_end_z_m": z_end,
        "grasp_offset_world_m": grasp_offset.tolist(),
        "grasp_yaw_deg": float(config.get("grasp_yaw_deg", 0.0)),
        "grip_half_gap_m": closed_gap,
        "final_end_target_m": (tightened[:3, 3] - grasp_offset).tolist(),
        "physical_success_requires_actual_centerline_and_contacts": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "motion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"Wrap motion validation failed: {report}")
    np.savez_compressed(
        out / "robot_motion.npz",
        times=times,
        joint_positions=qpath,
        joint_names=joint_names,
        tcp_poses=targets,
        phases=labels,
    )
    return qpath, targets, labels


def _capsule_pad_gap(
    state: np.ndarray,
    lengths: np.ndarray,
    finger_pose: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Exact capsule-axis / finger-box distance, less the physical rope radius."""
    axes = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :, 2]
    starts = state[:, :3] - axes * lengths[:, None] / 2
    ends = state[:, :3] + axes * lengths[:, None] / 2
    half = np.asarray(config["finger_pad_size_m"]) / 2
    a = (starts - finger_pose[:3, 3]) @ finger_pose[:3, :3] - config[
        "finger_pad_center_m"
    ]
    b = (ends - finger_pose[:3, 3]) @ finger_pose[:3, :3] - config[
        "finger_pad_center_m"
    ]
    best = np.inf
    for origin, delta in zip(a, b - a):
        breaks = [0.0, 1.0]
        for k in range(3):
            if abs(delta[k]) > 1e-12:
                breaks.extend(
                    float(v)
                    for v in (
                        (-half[k] - origin[k]) / delta[k],
                        (half[k] - origin[k]) / delta[k],
                    )
                    if 0 < v < 1
                )
        breaks = sorted(breaks)
        for left, right in zip(breaks[:-1], breaks[1:]):
            midpoint = origin + delta * ((left + right) / 2)
            active = np.abs(midpoint) > half
            offset = origin[active] - np.sign(midpoint[active]) * half[active]
            velocity = delta[active]
            denominator = float(velocity @ velocity)
            t = (
                float(np.clip(-(offset @ velocity) / denominator, left, right))
                if denominator > 1e-20
                else left
            )
            distance = np.linalg.norm(np.maximum(np.abs(origin + t * delta) - half, 0))
            best = min(best, float(distance))
    return best - config["cable_radius_m"]


def _rope_geometry(
    state: np.ndarray, lengths: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Measure actual capsule centerline geometry, independently of commanded orbit."""
    axis = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :, 2]
    starts = state[:, :3] - axis * lengths[:, None] / 2
    ends = state[:, :3] + axis * lengths[:, None] / 2
    nodes = np.vstack([starts[:1], (ends[:-1] + starts[1:]) / 2, ends[-1:]])
    center = np.asarray(config["post_center_xy_m"])
    theta = np.unwrap(np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0]))
    samples = (
        starts[:, None]
        + (ends - starts)[:, None] * np.linspace(0, 1, 11)[None, :, None]
    )
    radial = np.linalg.norm(samples[:, :, :2] - center, axis=-1)
    gaps = radial - config["post_radius_m"] - config["cable_radius_m"]
    in_height = (
        samples[:, :, 2] > config["floor_height_m"] - config["cable_radius_m"]
    ) & (
        samples[:, :, 2]
        < config["floor_height_m"] + config["post_height_m"] + config["cable_radius_m"]
    )
    valid_gaps = gaps[in_height]
    near = (
        np.min(np.where(in_height, gaps, np.inf), axis=1)
        <= config["validation"]["contact_band_m"]
    )
    # Union on the circle: revisiting one sector must not inflate coverage.
    intervals = []
    for low, high, include in zip(
        np.minimum(theta[:-1], theta[1:]), np.maximum(theta[:-1], theta[1:]), near
    ):
        if not include:
            continue
        width = high - low
        start = low % (2 * np.pi)
        end = start + width
        intervals.append((start, min(end, 2 * np.pi)))
        if end > 2 * np.pi:
            intervals.append((0.0, end - 2 * np.pi))
    coverage, right = 0.0, 0.0
    for low, high in sorted(intervals):
        coverage += max(0.0, high - max(low, right))
        right = max(right, high)
    joint_gap = np.linalg.norm(ends[:-1] - starts[1:], axis=1)
    anchor_direction = np.r_[center - starts[0, :2], 0.0]
    anchor_direction /= np.linalg.norm(anchor_direction)
    anchor_angle = np.rad2deg(np.arccos(np.clip(axis[0] @ anchor_direction, -1.0, 1.0)))
    anchor_bends = np.rad2deg(
        np.arccos(np.clip(np.sum(axis[:4] * axis[1:5], axis=-1), -1.0, 1.0))
    )
    return {
        "nodes": nodes,
        "anchor_direction_error_deg": float(anchor_angle),
        "anchor_exit_bend_max_deg": float(anchor_bends.max()),
        "winding_deg": float(np.rad2deg(theta[-1] - theta[0])),
        "contact_arc_deg": float(np.rad2deg(coverage)),
        "post_contact_segments": int(near.sum()),
        "post_penetration_m": (
            float(max(0.0, -valid_gaps.min())) if valid_gaps.size else 0.0
        ),
        "joint_gap_max_m": float(joint_gap.max()),
        "centerline_length_m": float(
            np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()
        ),
        "free_end_m": ends[-1].tolist(),
        "rope_min_z_m": float(samples[:, :, 2].min() - config["cable_radius_m"]),
        "rope_max_z_m": float(samples[:, :, 2].max() + config["cable_radius_m"]),
    }


def _validate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["validation"]
    hold = [r for r in rows if r["time_s"] >= config["duration_s"] - 2 - 1e-8]
    grasped = [r for r in rows if r["time_s"] >= config["grasp_complete_s"]]
    lifted = [
        r
        for r in rows
        if config["lift_complete_s"] <= r["time_s"] <= config["lift_complete_s"] + 0.5
    ]
    checks = {
        "completed": bool(
            rows
            and abs(rows[-1]["time_s"] - config["duration_s"]) < 1e-8
            and len(hold) >= round(2 * config["control_hz"]) + 1
        ),
        "finite": bool(rows and all(r["finite"] for r in rows)),
        "joint_limits": bool(
            rows
            and max(r["arm_joint_limit_violation_rad"] for r in rows) < 1e-4
            and max(r["finger_joint_limit_violation_m"] for r in rows) < 1e-4
        ),
        "direct_floor_pickup": bool(
            lifted
            and rows[0]["free_end_m"][2]
            <= config["floor_height_m"] + 2 * config["cable_radius_m"]
            and min(r["free_end_m"][2] for r in lifted)
            > config["floor_height_m"] + 0.035
        ),
        "grasp_retained": bool(
            grasped
            and all(r["grasp_confirmed"] for r in grasped)
            and max(r["tail_tcp_distance_m"] for r in grasped)
            <= thresholds["tail_tcp_max_m"]
        ),
        "material_grasp_slip": bool(
            grasped
            and max(r["material_slip_m"] for r in grasped)
            <= thresholds.get("grasp_slip_max_m", 0.008)
        ),
        "bilateral_finger_contact": bool(
            grasped and np.mean([r["bilateral_rope_contact"] for r in grasped]) >= 0.95
        ),
        "actual_winding": bool(
            hold
            and min(abs(r["winding_deg"]) for r in hold)
            >= thresholds["winding_min_deg"]
        ),
        "post_contact_arc": bool(
            hold
            and min(r["contact_arc_deg"] for r in hold)
            >= thresholds["contact_arc_min_deg"]
        ),
        "hold_stable": bool(
            hold
            and np.ptp([r["winding_deg"] for r in hold])
            <= thresholds["hold_winding_drift_max_deg"]
            and np.max(
                np.linalg.norm(
                    np.asarray([r["free_end_m"] for r in hold]) - hold[0]["free_end_m"],
                    axis=1,
                )
            )
            <= thresholds["hold_end_drift_max_m"]
        ),
        "rope_below_post_top": bool(
            rows
            and max(r["rope_max_z_m"] for r in rows)
            < config["floor_height_m"] + config["post_height_m"]
        ),
        "post_nonpenetration": bool(
            rows
            and max(r["post_penetration_m"] for r in rows)
            <= thresholds["post_penetration_max_m"]
        ),
        "robot_post_clearance": bool(
            rows
            and max(r["robot_post_penetration_m"] for r in rows)
            <= thresholds["robot_post_penetration_max_m"]
        ),
        "anchor_points_toward_post": bool(
            rows and max(r["anchor_direction_error_deg"] for r in rows) < 0.1
        ),
        "smooth_anchor_exit": bool(
            hold
            and max(r["anchor_exit_bend_max_deg"] for r in hold)
            <= thresholds.get("anchor_exit_bend_max_deg", 20.0)
        ),
        "anchor_fixed": bool(rows and max(r["anchor_error_m"] for r in rows) < 1e-5),
        "length_preserved": bool(
            rows
            and max(r["joint_gap_max_m"] for r in rows) <= thresholds["joint_gap_max_m"]
        ),
        "contacts_not_dropped": bool(
            rows and max(r["dropped_contacts"] for r in rows) == 0
        ),
        "floor_nonpenetration": bool(
            rows
            and min(r["rope_min_z_m"] for r in rows)
            >= config["floor_height_m"] - thresholds["floor_penetration_max_m"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            "hold_anchor_exit_bend_max_deg": max(
                [r["anchor_exit_bend_max_deg"] for r in hold], default=None
            ),
            "material_grasp_slip_max_m": max(
                [r["material_slip_m"] for r in grasped], default=None
            ),
            "final_winding_deg": rows[-1]["winding_deg"] if rows else None,
            "hold_winding_min_deg": min(
                [abs(r["winding_deg"]) for r in hold], default=None
            ),
            "hold_contact_arc_min_deg": min(
                [r["contact_arc_deg"] for r in hold], default=None
            ),
            "tail_tcp_max_m": max(
                [r["tail_tcp_distance_m"] for r in grasped], default=None
            ),
            "post_penetration_max_m": max(
                [r["post_penetration_m"] for r in rows], default=None
            ),
            "robot_post_penetration_max_m": max(
                [r["robot_post_penetration_m"] for r in rows], default=None
            ),
            "hold_winding_drift_deg": (
                float(np.ptp([r["winding_deg"] for r in hold])) if hold else None
            ),
            "hold_end_drift_m": (
                float(
                    np.max(
                        np.linalg.norm(
                            np.asarray([r["free_end_m"] for r in hold])
                            - hold[0]["free_end_m"],
                            axis=1,
                        )
                    )
                )
                if hold
                else None
            ),
            "joint_gap_max_m": max([r["joint_gap_max_m"] for r in rows], default=None),
            "bilateral_contact_fraction": (
                float(np.mean([r["bilateral_rope_contact"] for r in grasped]))
                if grasped
                else None
            ),
        },
        "thresholds": thresholds,
    }


def _run(
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: dict[str, Path],
    controls: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    from embodichain.lab.sim import SimulationManager

    sim = robot = cable = contact = None
    cameras, rows, states, joint_states, tcp_states = [], [], [], [], []
    lengths = np.linalg.norm(np.diff(_cable_points(config), axis=0), axis=1)
    material_arc = lengths.sum() - config["grasp_inset_m"]
    material_index = int(np.searchsorted(np.cumsum(lengths), material_arc))
    material_offset = (
        material_arc - lengths[:material_index].sum() - lengths[material_index] / 2
    )
    material_baseline = None
    actor_paths: dict[int, str] = {}
    started = time.perf_counter()
    limit_s = (
        min(config["duration_s"], args.seconds)
        if args.seconds is not None
        else config["duration_s"]
    )
    tree = ET.parse(paths["robot"])
    lo = np.array(
        [
            float(tree.getroot().find(f"joint[@name='{n}']/limit").get("lower"))
            for n in _JOINT_NAMES
        ]
    )
    hi = np.array(
        [
            float(tree.getroot().find(f"joint[@name='{n}']/limit").get("upper"))
            for n in _JOINT_NAMES
        ]
    )
    try:
        sim, robot, cable, contact, cameras, points = _scene(
            config, paths, controls, args.output, False
        )
        robot.set_qpos(torch.tensor(controls[:1], device=sim.device), target=True)
        for _ in range(
            round(config["pre_roll_s"] * config["control_hz"] * config["substeps"])
        ):
            sim.update(step=1, render_final_step=False)
        baseline = cable.data.segment_state_w[0, 0].cpu().numpy()
        _json(
            args.output / "settled_state.json",
            {
                "robot_qpos": robot.get_qpos()[0].cpu().tolist(),
                "cable_state": cable.data.segment_state_w[0].cpu().tolist(),
            },
        )
        previous_command = controls[0].copy()
        for frame in range(round(limit_s * config["control_hz"]) + 1):
            t = frame / config["control_hz"]
            robot_post_peak, penetration, dropped = 0.0, 0.0, 0
            finger_contacts = [False, False]
            bilateral_contact = False
            worst = None
            if frame:
                command = controls[frame]
                robot.set_qpos(
                    torch.tensor(command[None], device=sim.device), target=True
                )
                robot.set_qvel(
                    torch.tensor(
                        ((command - previous_command) * config["control_hz"])[None],
                        device=sim.device,
                    ),
                    target=True,
                )
                previous_command = command.copy()
                for sub in range(config["substeps"]):
                    sim.update(step=1, render_final_step=False)
                    if (sub + 1) % (config["substeps"] // 4) == 0:
                        contact.update()
                        data = contact.get_data()
                        valid = data["is_valid"]
                        candidate_positions = [[], []]
                        for ids, separation, position in zip(
                            data["user_ids"][valid].cpu().tolist(),
                            data["distance"][valid].cpu().tolist(),
                            data["position"][valid].cpu().tolist(),
                        ):
                            pair = []
                            for actor_id in ids:
                                if actor_id < 0:
                                    continue
                                if actor_id not in actor_paths:
                                    info = contact.get_actor_info(actor_id)
                                    actor_paths[actor_id] = (
                                        f"{info.path}/{info.link_name or ''}"
                                    )
                                pair.append(actor_paths[actor_id])
                            is_rope = any(actor_id < 0 for actor_id in ids)
                            for k, name in enumerate(
                                ("fr3_leftfinger", "fr3_rightfinger")
                            ):
                                if (
                                    is_rope
                                    and any(name in p for p in pair)
                                    and separation <= 1e-5
                                ):
                                    candidate_positions[k].append(position)
                            if any("/post/" in p for p in pair) and any(
                                "/franka/" in p for p in pair
                            ):
                                robot_post_peak = max(robot_post_peak, -separation)
                            if -separation > penetration:
                                penetration = -separation
                                worst = pair
                        matched = [False, False]
                        if any(candidate_positions):
                            sampled_state = (
                                cable.data.segment_state_w[0, -8:].cpu().numpy()
                            )
                            axes = Rotation.from_quat(
                                sampled_state[:, 3:7]
                            ).as_matrix()[:, :, 2]
                            starts = (
                                sampled_state[:, :3] - axes * lengths[-8:, None] / 2
                            )
                            delta = axes * lengths[-8:, None]
                            for k, name in enumerate(
                                ("fr3_leftfinger", "fr3_rightfinger")
                            ):
                                if not candidate_positions[k]:
                                    continue
                                contact_points = np.asarray(candidate_positions[k])
                                fraction = np.clip(
                                    np.sum(
                                        (contact_points[:, None] - starts) * delta,
                                        axis=-1,
                                    )
                                    / np.sum(delta * delta, axis=-1),
                                    0.0,
                                    1.0,
                                )
                                closest = starts + fraction[..., None] * delta
                                point_gap = (
                                    np.linalg.norm(
                                        contact_points[:, None] - closest, axis=-1
                                    ).min()
                                    - config["cable_radius_m"]
                                )
                                pose = (
                                    robot.get_link_pose(name, to_matrix=True)[0]
                                    .cpu()
                                    .numpy()
                                )
                                pad_gap = _capsule_pad_gap(
                                    sampled_state, lengths[-8:], pose, config
                                )
                                matched[k] = bool(
                                    point_gap <= 0.0003 and pad_gap <= 0.0002
                                )
                                finger_contacts[k] |= matched[k]
                        bilateral_contact |= all(matched)
                        dropped += contact.dropped_contacts
            state = cable.data.segment_state_w[0].cpu().numpy()
            qpos = robot.get_qpos()[0].cpu().numpy()
            tcp = robot.get_link_pose("fr3_hand_tcp", to_matrix=True)[0].cpu().numpy()
            pad_gaps = []
            for name in ("fr3_leftfinger", "fr3_rightfinger"):
                pose = robot.get_link_pose(name, to_matrix=True)[0].cpu().numpy()
                pad_gaps.append(
                    _capsule_pad_gap(state[-8:], lengths[-8:], pose, config)
                )
            finger_contacts = [
                present and gap <= 0.0002
                for present, gap in zip(finger_contacts, pad_gaps)
            ]
            material_axis = Rotation.from_quat(state[material_index, 3:7]).as_matrix()[
                :, 2
            ]
            material_world = state[material_index, :3] + material_offset * material_axis
            material_local = tcp[:3, :3].T @ (material_world - tcp[:3, 3])
            if (
                material_baseline is None
                and t >= config["grasp_complete_s"]
                and bilateral_contact
            ):
                material_baseline = material_local.copy()
            material_slip = (
                float(np.linalg.norm(material_local - material_baseline))
                if material_baseline is not None
                else 0.0
            )
            geometry = _rope_geometry(state, lengths, config)
            nodes = geometry.pop("nodes")
            row = {
                "time_s": t,
                "phase": labels[frame],
                **geometry,
                "actual_tcp_pose": tcp.tolist(),
                "command_tcp_pose": targets[frame].tolist(),
                "tcp_error_m": float(
                    np.linalg.norm(tcp[:3, 3] - targets[frame, :3, 3])
                ),
                "tail_tcp_distance_m": float(np.linalg.norm(nodes[-1] - tcp[:3, 3])),
                "grasp_region_tcp_distance_m": float(
                    np.linalg.norm(state[-6:, :3] - tcp[:3, 3], axis=1).min()
                ),
                "grasp_confirmed": material_baseline is not None,
                "bilateral_rope_contact": bilateral_contact,
                "material_point_tcp_m": material_local.tolist(),
                "material_slip_m": material_slip,
                "finger_pad_gaps_m": pad_gaps,
                "left_rope_contact": finger_contacts[0],
                "right_rope_contact": finger_contacts[1],
                "finger_gap_m": float(qpos[7] + qpos[8]),
                "anchor_error_m": float(np.linalg.norm(state[0, :3] - baseline[:3])),
                "arm_joint_limit_violation_rad": float(
                    max(0.0, (lo[:7] - qpos[:7]).max(), (qpos[:7] - hi[:7]).max())
                ),
                "finger_joint_limit_violation_m": float(
                    max(0.0, (lo[7:] - qpos[7:]).max(), (qpos[7:] - hi[7:]).max())
                ),
                "robot_post_penetration_m": robot_post_peak,
                "penetration_m": penetration,
                "worst_contact": worst,
                "dropped_contacts": dropped,
                "finite": bool(np.isfinite(state).all() and np.isfinite(qpos).all()),
            }
            rows.append(row)
            states.append(state)
            joint_states.append(qpos)
            tcp_states.append(tcp)
            if not row["finite"]:
                raise RuntimeError(f"Nonfinite physical state at {t}")
            if frame % config["control_hz"] == 0:
                print(
                    f"t={t:.1f} {row['phase']} winding={row['winding_deg']:.1f} arc={row['contact_arc_deg']:.1f} tail={row['tail_tcp_distance_m']*1000:.1f}mm fingers={finger_contacts} wall={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        report = _validate(rows, config)
        report["elapsed_s"] = time.perf_counter() - started
        _json(args.output / "validation.json", report)
        outcome = "DEMO_COMPLETE" if report["checks"]["completed"] else "DEMO_PARTIAL"
        print(outcome + " " + json.dumps(report), flush=True)
        return report
    finally:
        try:
            _json(
                args.output / "metrics.json",
                {
                    "rows": rows,
                    "actor_paths": actor_paths,
                    "contact_sample_hz": config["control_hz"] * 4,
                    "rope_contact_evidence": "Both selected fingers must have anonymous contact rows in the same diagnostic substep; each contact position must match the actual rope capsule surface and its fingertip geometry at that substep. This is contact/geometry evidence, not calibrated force measurement.",
                },
            )
            if rows:
                np.savez_compressed(
                    args.output / "states.npz",
                    times=[r["time_s"] for r in rows],
                    cable_states=states,
                    robot_qpos=joint_states,
                    tcp_poses=tcp_states,
                    joint_names=_JOINT_NAMES,
                )
        finally:
            cameras.clear()
            robot = cable = contact = None
            try:
                if sim is not None:
                    sim.destroy(exit_process=False)
            finally:
                sim = None
                SimulationManager.flush_cleanup_queue()


def create_replay_scene(
    config: dict[str, Any], run_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Rebuild the recorded scene for rendering without advancing dynamics."""
    paths = {"robot": run_dir / "generated_assets/franka/franka_rope.urdf"}
    with np.load(run_dir / "states.npz", allow_pickle=False) as data:
        controls = data["robot_qpos"][:1]
    sim, robot, cable, contact, cameras, points = _scene(
        config, paths, controls, output_dir, True
    )
    return dict(sim=sim, robot=robot, cable=cable, contact=contact, cameras=cameras)


def restore_replay_frame(
    scene: dict[str, Any], states: dict[str, np.ndarray], index: int
) -> None:
    """Restore an exact recorded state; this function never steps physics."""
    device = scene["sim"].device
    scene["robot"].set_qpos(
        torch.as_tensor(
            states["robot_qpos"][index : index + 1], dtype=torch.float32, device=device
        ),
        target=False,
    )
    scene["cable"].write_segment_state_to_sim(
        torch.as_tensor(
            states["cable_states"][index : index + 1],
            dtype=torch.float32,
            device=device,
        ).contiguous()
    )


def replay_camera_views(
    t: float, config: dict[str, Any], states: dict[str, np.ndarray], index: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Use stable overview and high oblique views to expose the actual winding."""
    return {
        name: (np.asarray(view["eye_m"]), np.asarray(view["target_m"]))
        for name, view in (
            ("overview", config["camera"]),
            ("detail", config["detail_camera"]),
        )
    }


def main() -> int:
    """Run a reproducible contact-driven floor pickup and rope wrapping demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Scene configuration; defaults to the config shipped beside this example",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        help="Stop early for diagnostics; an incomplete run does not pass acceptance",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Prepare assets and validate the motion without stepping physics",
    )
    args = parser.parse_args()
    if args.seconds is not None and (
        not np.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("--seconds must be a finite positive number")
    args.config = args.config.expanduser().resolve()
    try:
        config = json.loads(args.config.read_text())
        _validate_config(config)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"Invalid configuration: {error}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output / "config.json")
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    manifest = {
        "run_id": args.output.name,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "EmbodiChain",
        "robot": "Franka FR3 / Panda hand with flat rope pads",
        "physics": "Newton / DexUni",
        "rendered": False,
        "control": "Dynamic position/velocity joint drives; direct finger contact on the rope; no rope state writes or gripper attachment during physics",
        "floor": {
            "default_ground_hidden": True,
            "surface_height_m": config["floor_height_m"],
            "pickup_stand": False,
        },
        "source_sha256": _hash(Path(__file__)),
        "config_sha256": _hash(args.config),
        "git_head": _source_revision(),
        "runtime": _runtime_versions(),
    }
    _json(args.output / "manifest.json", manifest)
    try:
        paths = _prepare_models(config, args.output)
        controls, targets, labels = _motion(config, paths["robot"], args.output)
        if args.plan_only:
            manifest["status"] = "planned"
            return 0
        report = _run(args, config, paths, controls, targets, labels)
        manifest.update(
            status="completed" if report["checks"]["completed"] else "partial",
            passed=report["passed"],
        )
        return 0 if report["passed"] else 1
    except Exception:
        manifest["status"] = "failed"
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        _json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
