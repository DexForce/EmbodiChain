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

"""JSON contracts for grasping an existing, verified assembly."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.tools.assemble._geometry import load_mesh
from scripts.tools.assemble._json_io import pose_matrix, read_json

__all__ = [
    "load_config",
    "load_assembly",
    "action_schema",
    "execution_settings",
    "layout_settings",
    "motion_settings",
    "resolve_scene",
    "resolve_task_plan",
]

_MOTION_DEFAULTS = {
    "strategy": "toppra",
    "clearance": 0.12,
    "pre_grasp_distance": 0.10,
    "retract_distance": 0.12,
    "cartesian_step": 0.01,
    "rotation_step_degrees": 3.0,
    "velocity_limit": 0.6,
    "acceleration_limit": 1.2,
    "jerk_limit": 6.0,
    "duration_scale": 1.0,
    "grasp_hold_seconds": 0.24,
    "hand_duration": 0.5,
    "joint_cost_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 0.2],
}


def motion_settings(value: dict | None = None) -> dict:
    """Validate direct waypoint timing and sampling settings.

    Args:
        value: Motion configuration.

    Returns:
        Resolved TOPPRA and smooth time-scaling settings.
    """
    if value is not None and not isinstance(value, dict):
        raise ValueError("Invalid motion settings")
    value = {} if value is None else value.copy()
    if value.keys() - _MOTION_DEFAULTS.keys():
        raise ValueError(
            "Unknown motion settings; use the direct waypoint configuration"
        )
    result = _MOTION_DEFAULTS | value
    if result["strategy"] != "toppra":
        raise ValueError("motion.strategy must be toppra")
    weights = result["joint_cost_weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != 6
        or any(
            type(w) not in (int, float) or not math.isfinite(w) or w < 0
            for w in weights
        )
        or not any(w > 0 for w in weights)
    ):
        raise ValueError(
            "motion.joint_cost_weights must contain six finite nonnegative numbers "
            "with at least one positive weight, in arm joint1 through joint6 order"
        )
    result["joint_cost_weights"] = [float(w) for w in weights]
    for key, low, high in (
        ("clearance", 0.02, 0.5),
        ("pre_grasp_distance", 0.02, 0.3),
        ("retract_distance", 0.02, 0.3),
        ("cartesian_step", 0.002, 0.03),
        ("rotation_step_degrees", 0.5, 10),
        ("velocity_limit", 0.05, 2),
        ("acceleration_limit", 0.1, 10),
        ("jerk_limit", 0.1, 100),
        ("duration_scale", 1, 10),
        ("grasp_hold_seconds", 0.1, 3),
        ("hand_duration", 0.2, 3),
    ):
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"Invalid motion.{key}")
    return result


_EXECUTION_DEFAULTS = {
    "grasp_mode": "contact",
    "release_hold_steps": 100,
    "settle_steps": 240,
    "rotation_tolerance_degrees": 20.0,
}

_LAYOUT_DEFAULTS = {
    "mode": "fixed",
    "x_bounds": [-0.70, -0.35],
    "y_bounds": [-0.40, 0.40],
    "yaw_offset_bounds_degrees": [-90.0, 90.0],
    "base_min_robot_distance": 0.55,
    "assemble_min_robot_distance": 0.40,
    "max_robot_distance": 0.80,
    "min_object_distance": 0.23,
    "min_candidates": 3,
    "max_joint_step_degrees": 12.0,
    "max_joint_range_degrees": 240.0,
}


def layout_settings(value: dict | None = None) -> dict:
    """Validate bounded robot-relative layout search settings.

    Args:
        value: Optional layout section. Omission preserves fixed legacy poses.

    Returns:
        Complete workspace bounds and joint-motion acceptance limits.
    """
    value = {} if value is None else value
    if not isinstance(value, dict) or value.keys() - _LAYOUT_DEFAULTS.keys():
        raise ValueError("Invalid layout settings")
    result = _LAYOUT_DEFAULTS | value
    if result["mode"] not in ("fixed", "optimize"):
        raise ValueError("layout.mode must be fixed or optimize")
    for key in ("x_bounds", "y_bounds", "yaw_offset_bounds_degrees"):
        pair = result[key]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in pair)
            or pair[0] >= pair[1]
        ):
            raise ValueError(f"layout.{key} must be an increasing finite pair")
        result[key] = list(pair)
    if (
        type(result["min_candidates"]) is not int
        or not 2 <= result["min_candidates"] <= 12
    ):
        raise ValueError("layout.min_candidates must be an integer in [2, 12]")
    for key in (
        "base_min_robot_distance",
        "assemble_min_robot_distance",
        "max_robot_distance",
        "min_object_distance",
        "max_joint_step_degrees",
        "max_joint_range_degrees",
    ):
        number = result[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"layout.{key} must be positive and finite")
    if (
        max(result["base_min_robot_distance"], result["assemble_min_robot_distance"])
        >= result["max_robot_distance"]
    ):
        raise ValueError("layout robot distance bounds must be increasing")
    return result


def execution_settings(value: dict | None = None) -> dict:
    """Validate execution options, including legacy results without this section.

    Args:
        value: Optional execution object from a config or saved result.

    Returns:
        Resolved grasp mode, simulation hold durations, and final pose tolerance.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict) or value.keys() - _EXECUTION_DEFAULTS.keys():
        raise ValueError("Invalid execution settings")
    result = _EXECUTION_DEFAULTS | value
    if result["grasp_mode"] not in ("contact", "fixed_constraint"):
        raise ValueError("execution.grasp_mode must be contact or fixed_constraint")
    for key in ("release_hold_steps", "settle_steps"):
        if type(result[key]) is not int or not 1 <= result[key] <= 2000:
            raise ValueError(f"execution.{key} must be an integer in [1, 2000]")
    angle = result["rotation_tolerance_degrees"]
    if (
        type(angle) not in (int, float)
        or not math.isfinite(angle)
        or not 0 < angle <= 180
    ):
        raise ValueError("execution.rotation_tolerance_degrees must be in (0, 180]")
    return result


def load_config(path: Path) -> dict:
    """Resolve a grasp job's existing assembly, world placement and run settings.

    Args:
        path: JSON input. Relative paths use its parent directory.

    Returns:
        Validated configuration with absolute paths.
    """
    path = path.resolve()
    config = read_json(path)
    defaults = {
        "output_dir": str(
            Path(__file__).resolve().parents[3]
            / "outputs"
            / "grasp_assemble"
            / path.stem
        ),
        "robot": "ur5_pgi",
        "codex": {"model": None, "max_turns": 12, "timeout_seconds": 300},
        "execution": deepcopy(_EXECUTION_DEFAULTS),
        "layout": _LAYOUT_DEFAULTS,
        "motion": _MOTION_DEFAULTS,
        "instruction": "",
    }
    required = {"assembly_result"}
    pose_keys = {"T_world_base", "T_world_assemble_initial"}
    if (
        not required <= config.keys()
        or config.keys() - required - defaults.keys() - pose_keys
    ):
        raise ValueError(
            f"Required fields: {sorted(required)}; unknown fields are forbidden"
        )
    parameter_overrides = {
        "motion": deepcopy(config.get("motion", {})),
        "assemble_initial_pose": "T_world_assemble_initial" in config,
    }
    default_layout = deepcopy(_LAYOUT_DEFAULTS)
    if not pose_keys <= config.keys():
        default_layout["mode"] = "optimize"
        defaults["execution"]["rotation_tolerance_degrees"] = 10.0
    defaults["layout"] = default_layout
    for key, default in defaults.items():
        if isinstance(default, dict):
            supplied = config.get(key, {})
            if not isinstance(supplied, dict) or supplied.keys() - default.keys():
                raise ValueError(f"Invalid {key} settings")
            config[key] = default | supplied
        else:
            config.setdefault(key, default)
    config["execution"] = execution_settings(config["execution"])
    config["layout"] = layout_settings(config["layout"])
    config["motion"] = motion_settings(config["motion"])
    for key in ("assembly_result", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
        config[key] = str((path.parent / config[key]).resolve())
    if path.is_relative_to(Path(config["output_dir"])):
        raise ValueError("Keep the configuration outside the output directory")
    if config["robot"] != "ur5_pgi":
        raise ValueError("This calibrated harness currently supports robot='ur5_pgi'")
    if not isinstance(config["instruction"], str):
        raise ValueError("instruction must be a string")
    for key in pose_keys & config.keys():
        config[key] = pose_matrix(config[key]).tolist()
    # The verified assembly is gravity-supported in its exported Z-up frame.
    if "T_world_base" in config and any(
        abs(x - y) > 1e-6
        for x, y in zip([row[2] for row in config["T_world_base"][:3]], [0, 0, 1])
    ):
        raise ValueError("T_world_base must preserve the upward Z axis")
    for section, key, low, high, integer in (
        ("codex", "max_turns", 2, 30, True),
        ("codex", "timeout_seconds", 1, 1800, False),
    ):
        number = config[section][key]
        if (
            type(number) not in (int, float)
            or not math.isfinite(number)
            or not low <= number <= high
            or (integer and type(number) is not int)
        ):
            raise ValueError(f"Invalid {section}.{key}")
    model = config["codex"]["model"]
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("codex.model must be a model name or null")
    if (
        config["layout"]["mode"] == "optimize"
        and config["codex"]["max_turns"] <= config["layout"]["min_candidates"]
    ):
        raise ValueError(
            "codex.max_turns must allow layout evaluations plus a finish action"
        )
    config["config_path"] = str(path)
    config["parameter_overrides"] = parameter_overrides
    return config


def _grounded_pose(path: Path, transform: np.ndarray) -> np.ndarray:
    """Place the lowest transformed mesh vertex 1 mm above the ground."""
    transform = transform.copy()
    vertices = np.asarray(load_mesh(path).vertices, dtype=float)
    rotated_z = vertices @ transform[2, :3]
    transform[2, 3] = 0.001 - float(rotated_z.min())
    return transform


def resolve_scene(config: dict, assembly: dict) -> dict:
    """Supply grounded seed poses while preserving explicit initial transforms.

    Args:
        config: Loaded job, whose world poses are optional.
        assembly: Verified source assets with resolved mesh paths.

    Returns:
        Independent configuration containing both initial world transforms.
    """
    resolved = deepcopy(config)
    for role, key, xy in (
        ("base", "T_world_base", (-0.55, -0.10)),
        ("assemble", "T_world_assemble_initial", (-0.35, -0.30)),
    ):
        if key in resolved:
            continue
        transform = np.eye(4)
        transform[:2, 3] = xy
        resolved[key] = _grounded_pose(
            Path(assembly["assets"][role]["path"]), transform
        ).tolist()
    return resolved


_TASK_DISTANCE_LIMITS = {
    "clearance": (0.02, 0.5),
    "pre_grasp_distance": (0.02, 0.3),
    "insertion_distance": (0.02, 0.5),
    "retract_distance": (0.02, 0.3),
}
_TASK_VECTOR_KEYS = {
    "assemble_initial_rpy_degrees",
    "insertion_direction_base",
    "retract_direction_base",
}


def resolve_task_plan(config: dict, assembly: dict, proposal: dict | None) -> dict:
    """Validate agent-selected geometric parameters and apply user constraints.

    Args:
        config: Scene configuration with resolved world poses. Explicit user
            motion fields and initial transforms take precedence over proposals.
        assembly: Verified assembly containing the moving object's mesh.
        proposal: Agent plan; null preserves historical saved configurations.
            RPY is an extrinsic XYZ rotation from exported object coordinates
            into world coordinates, before the layout's world-Z yaw offset.
            Direction vectors are expressed in the base object's frame.

    Returns:
        Independent configuration with the applied ``task_plan`` and motion
        settings. Automatically oriented objects are grounded from mesh vertices.
    """
    resolved = deepcopy(config)
    if proposal is None:
        return resolved
    keys = _TASK_VECTOR_KEYS | _TASK_DISTANCE_LIMITS.keys() | {"reason"}
    if not isinstance(proposal, dict) or set(proposal) != keys:
        raise ValueError(f"task_plan requires exactly these fields: {sorted(keys)}")
    plan = deepcopy(proposal)
    for key in _TASK_VECTOR_KEYS:
        value = plan[key]
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
        ):
            raise ValueError(f"task_plan.{key} must contain three finite numbers")
        vector = np.asarray(value, dtype=float)
        if key != "assemble_initial_rpy_degrees":
            scale = float(np.max(np.abs(vector)))
            if scale == 0:
                raise ValueError(f"task_plan.{key} must be nonzero")
            vector = vector / scale
            vector /= np.linalg.norm(vector)
        plan[key] = vector.tolist()
    for key, (low, high) in _TASK_DISTANCE_LIMITS.items():
        value = plan[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"task_plan.{key} must be in [{low}, {high}]")
    if not isinstance(plan["reason"], str) or not plan["reason"].strip():
        raise ValueError("task_plan.reason must be a nonempty string")
    overrides = config.get("parameter_overrides", {})
    motion = motion_settings(config.get("motion"))
    for key in ("clearance", "pre_grasp_distance", "retract_distance"):
        if key in overrides.get("motion", {}):
            plan[key] = motion[key]
        else:
            motion[key] = plan[key]
    resolved["motion"] = motion_settings(motion)
    transform = pose_matrix(resolved["T_world_assemble_initial"]).copy()
    if overrides.get("assemble_initial_pose", False):
        plan["assemble_initial_rpy_degrees"] = (
            Rotation.from_matrix(transform[:3, :3])
            .as_euler("xyz", degrees=True)
            .tolist()
        )
    else:
        transform[:3, :3] = Rotation.from_euler(
            "xyz", plan["assemble_initial_rpy_degrees"], degrees=True
        ).as_matrix()
        resolved["T_world_assemble_initial"] = _grounded_pose(
            Path(assembly["assets"]["assemble"]["path"]), transform
        ).tolist()
    resolved["task_plan"] = plan
    return resolved


def load_assembly(config: dict) -> dict:
    """Require a successful assembly and verify the exact source asset hashes.

    Args:
        config: Normalized grasp job.

    Returns:
        Assembly result whose source meshes and transform remain intact.
    """
    source = read_json(Path(config["assembly_result"]))
    if (
        source.get("schema") != "codex-assemble/v1"
        or source.get("success") is not True
        or source.get("validation", {}).get("accepted") is not True
    ):
        raise ValueError(
            "assembly_result must reference a successful assemble result.json; a failed latest.json is not usable"
        )
    pose_matrix(source["T_base_assemble"])
    for role in ("base", "assemble"):
        asset = source["assets"][role]
        path = Path(asset["path"])
        if not path.is_absolute():
            path = Path(config["assembly_result"]).parent / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if asset.get("sha256") != digest:
            raise ValueError(f"{role} mesh changed after assembly validation")
        asset["path"] = str(path.resolve())
    return source


def action_schema() -> dict:
    """Return the grasp-specific structured action schema."""
    row = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    matrix = {"type": "array", "items": row, "minItems": 4, "maxItems": 4}
    xy = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    xyz = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    task_properties = {key: deepcopy(xyz) for key in sorted(_TASK_VECTOR_KEYS)}
    task_properties.update(
        {
            key: {"type": "number", "minimum": low, "maximum": high}
            for key, (low, high) in _TASK_DISTANCE_LIMITS.items()
        }
    )
    task_properties["reason"] = {"type": "string"}
    task_plan = {
        "type": "object",
        "properties": task_properties,
        "required": list(task_properties),
        "additionalProperties": False,
    }
    layout = {
        "type": "object",
        "properties": {
            "base_xy": xy,
            "assemble_xy": xy,
            "base_yaw_offset_degrees": {"type": "number"},
            "assemble_yaw_offset_degrees": {"type": "number"},
        },
        "required": [
            "base_xy",
            "assemble_xy",
            "base_yaw_offset_degrees",
            "assemble_yaw_offset_degrees",
        ],
        "additionalProperties": False,
    }
    properties = {
        "action": {"type": "string", "enum": ["evaluate", "finish", "fail"]},
        "reason": {"type": "string"},
        "T_assemble_tcp": {"anyOf": [matrix, {"type": "null"}]},
        "layout": {"anyOf": [layout, {"type": "null"}]},
        "task_plan": {"anyOf": [task_plan, {"type": "null"}]},
        "candidate_id": {"type": ["integer", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
