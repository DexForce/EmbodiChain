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

"""Build the simulator and Action Engine artifact manifests."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.config import (
    ACTION_ENGINE_DEFAULTS_SCHEMA,
    default_runtime_policy,
    generation_defaults,
    runtime_policy_hash,
)
from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    ACTION_ENGINE_ENV_ID,
    EXECUTION_PROGRAM_FILENAME,
    SCENE_REQUIREMENTS_FILENAME,
    TASK_SPEC_FILENAME,
)

from .models import PreparedScene

__all__ = [
    "build_agent_config",
    "build_fast_gym_config",
    "canonical_robot_profile",
    "VLM_CAMERA_UIDS",
    "validate_fast_gym_config",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_GENERATION_DEFAULTS = generation_defaults()
_DEFAULT_TABLETOP_Z = float(_GENERATION_DEFAULTS["scene"]["default_tabletop_z"])

_ARM_SLOTS = {
    "left": {"arm": "left_arm", "eef": "left_eef"},
    "right": {"arm": "right_arm", "eef": "right_eef"},
}

# These IDs are part of the A/B runtime contract.  Keep the order stable so
# visual-fact payloads and comparison reports are reproducible across runs.
VLM_CAMERA_UIDS = (
    "vlm_front",
    "vlm_left",
    "vlm_rear",
    "vlm_right",
)


def canonical_robot_profile(profile: str) -> str:
    """Normalize the supported CLI aliases to one runtime profile ID."""
    normalized = str(profile).strip().lower().replace("-", "_")
    profiles = _robot_profiles()
    if normalized in profiles:
        return normalized
    for profile_id, value in profiles.items():
        if normalized in value["aliases"]:
            return profile_id
    raise ValueError(
        f"Unsupported robot profile {profile!r}; expected one of: "
        f"{', '.join(sorted(profiles))}"
    )


def build_agent_config(
    *,
    task_name: str,
    robot_profile: str,
    execution_program_hash: str,
    source_config_path: Path,
    uid_map: dict[str, str],
    planning_mode: str = "offline",
    seed_task_graph_path: str | Path | None = EXECUTION_PROGRAM_FILENAME,
    vlm_model: str | None = None,
    vlm_camera_uids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the small manifest consumed by ``run_agent``."""
    profile = canonical_robot_profile(robot_profile)
    runtime_policy = default_runtime_policy(profile)
    _validate_planning_mode(planning_mode)
    graph_path = _validate_seed_graph_path(seed_task_graph_path)
    if planning_mode == "ab" and graph_path == EXECUTION_PROGRAM_FILENAME:
        graph_path = f"offline/{EXECUTION_PROGRAM_FILENAME}"
    result = {
        "schema_version": ACTION_ENGINE_CONFIG_SCHEMA,
        "task_name": task_name,
        "robot_profile": profile,
        "planning_mode": planning_mode,
        "task_spec": TASK_SPEC_FILENAME,
        "scene_requirements": SCENE_REQUIREMENTS_FILENAME,
        "seed_task_graph": graph_path,
        "seed_task_graph_hash": execution_program_hash,
        "runtime_policy": runtime_policy.as_mapping(),
        "runtime_policy_hash": runtime_policy_hash(runtime_policy),
        "source": {
            "gym_config": source_config_path.as_posix(),
            "uid_map": dict(sorted(uid_map.items())),
        },
    }
    if planning_mode == "ab":
        camera_uids = _normalize_vlm_camera_uids(vlm_camera_uids)
        configured_model = _optional_model(vlm_model)
        # Retain concise top-level aliases for early A/B bundles while keeping
        # the nested section as the canonical runtime namespace.
        result["offline_seed_task_graph"] = graph_path
        result["vlm_model"] = configured_model
        result["vlm_camera_uids"] = list(camera_uids)
        result["online_planning"] = {
            # Model names are deliberately persisted only when explicitly
            # supplied by the generator.  Runtime resolution can then apply
            # the documented ACTION_ENGINE_VLM_MODEL/OPENAI_MODEL fallback.
            "vlm_model": configured_model,
            "camera_uids": camera_uids,
        }
    return result


def build_fast_gym_config(
    scene: PreparedScene,
    *,
    task_name: str,
    task_description: str,
    robot_profile: str,
    execution_program_hash: str,
    max_episodes: int,
    max_episode_steps: int,
    randomize_scene: bool = False,
    randomize_table_material: bool = False,
    planning_mode: str = "offline",
    seed_task_graph_path: str | Path | None = EXECUTION_PROGRAM_FILENAME,
) -> dict[str, Any]:
    """Build a runnable EmbodiChain gym config from a prepared source scene."""
    if max_episodes < 1:
        raise ValueError("max_episodes must be at least 1.")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1.")
    _validate_planning_mode(planning_mode)
    graph_path = _validate_seed_graph_path(seed_task_graph_path)
    if planning_mode == "ab" and graph_path == EXECUTION_PROGRAM_FILENAME:
        graph_path = f"offline/{EXECUTION_PROGRAM_FILENAME}"
    profile = canonical_robot_profile(robot_profile)

    profile_config = _profile(profile)
    robot = _make_robot(profile, profile_config, scene.table_top_z)
    observations = _make_observations(robot)
    # These two template fields describe serialization order to generation, not
    # RobotCfg. Remove them after deriving observation IDs to avoid parser noise.
    robot.pop("observation_joint_parts", None)
    robot.pop("qpos_control_part_order", None)
    sensors = _load_template("default_sensors.json")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("Default sensor template must define at least one camera.")
    environment_policy = _GENERATION_DEFAULTS["environment"]
    viewer_camera_uid = str(environment_policy["viewer_camera_uid"])
    sensors[0]["uid"] = viewer_camera_uid
    if planning_mode == "ab":
        vlm_sensors = _load_template("vlm_sensors.json")
        if not isinstance(vlm_sensors, list) or len(vlm_sensors) != len(
            VLM_CAMERA_UIDS
        ):
            raise ValueError("A/B planning requires exactly four VLM cameras.")
        _validate_vlm_sensors(vlm_sensors)
        _anchor_vlm_sensors(vlm_sensors, scene)
        sensors.extend(vlm_sensors)
    light = _load_template("default_lights.json")

    rigid_uids = [str(config["uid"]) for config in scene.rigid_objects]
    background_uids = [str(config["uid"]) for config in scene.background]
    engine_extension = {
        "schema_version": "action_engine_runtime_v2",
        "defaults_schema_version": ACTION_ENGINE_DEFAULTS_SCHEMA,
        "task_name": task_name,
        "robot_profile": profile,
        "planning_mode": planning_mode,
        "task_spec": TASK_SPEC_FILENAME,
        "scene_requirements": SCENE_REQUIREMENTS_FILENAME,
        "seed_task_graph": graph_path,
        "seed_task_graph_hash": execution_program_hash,
        "source_gym_config": scene.source_config_path.as_posix(),
        "source_scene_z_rotation_degrees": scene.z_rotation_degrees,
        "body_scale_policy": scene.body_scale_policy,
        "body_scale": list(scene.body_scale),
        "asset_hashes": dict(sorted(scene.asset_hashes.items())),
        "asset_provenance": [deepcopy(value) for value in scene.asset_provenance],
        "uid_map": dict(sorted(scene.uid_map.items())),
    }
    extensions = {
        "action_engine": engine_extension,
        "agent_robot_profile": profile,
        "agent_arm_slots": deepcopy(_ARM_SLOTS),
        "agent_static_obstacle_uids": background_uids,
        "gripper_open_state": list(profile_config["gripper_open_state"]),
        "gripper_close_state": list(profile_config["gripper_close_state"]),
        "arm_aim_yaw_offset": deepcopy(environment_policy["arm_aim_yaw_offset"]),
        "ignore_terminations_during_agent": bool(
            environment_policy["ignore_terminations_during_agent"]
        ),
        "viewer_camera_uid": viewer_camera_uid,
    }

    config: dict[str, Any] = {
        "id": ACTION_ENGINE_ENV_ID,
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": extensions,
            "events": _make_events(
                sensors[0],
                rigid_uids,
                randomize_scene=randomize_scene,
                randomize_table_material=randomize_table_material,
            ),
            "observations": observations,
            "dataset": _make_dataset(
                task_name=task_name,
                task_description=task_description,
                source_config_path=scene.source_config_path,
                robot_type=str(robot["uid"]),
            ),
        },
        "robot": robot,
        "sensor": sensors,
        "light": light,
        "background": [deepcopy(obj_config) for obj_config in scene.background],
        "rigid_object": [deepcopy(obj_config) for obj_config in scene.rigid_objects],
    }
    if scene.articulations:
        config["articulation"] = [
            deepcopy(articulation) for articulation in scene.articulations
        ]
    validate_fast_gym_config(config)
    return config


def validate_fast_gym_config(config: dict[str, Any]) -> None:
    """Check the cross-file and simulator-facing invariants generation owns."""
    if config.get("id") != ACTION_ENGINE_ENV_ID:
        raise ValueError(f"Gym config id must be {ACTION_ENGINE_ENV_ID!r}.")
    if not isinstance(config.get("robot"), dict) or not config["robot"].get("uid"):
        raise ValueError("Gym config requires a concrete robot template.")
    if not config.get("sensor"):
        raise ValueError("Gym config requires at least one sensor.")
    if not all(isinstance(sensor, dict) for sensor in config["sensor"]):
        raise ValueError("Generated sensors must be object mappings.")
    sensor_uids = [str(sensor.get("uid", "")) for sensor in config["sensor"]]
    if not all(sensor_uids) or len(sensor_uids) != len(set(sensor_uids)):
        raise ValueError("Generated sensor UIDs must be non-empty and unique.")
    if not config.get("background"):
        raise ValueError("Gym config requires at least one background object.")

    objects = [
        *config.get("background", []),
        *config.get("rigid_object", []),
        *config.get("articulation", []),
    ]
    uids = [str(obj.get("uid", "")) for obj in objects]
    if not all(uids) or len(uids) != len(set(uids)):
        raise ValueError("Generated scene object UIDs must be non-empty and unique.")
    if "table" not in uids:
        raise ValueError("Generated tabletop scene must expose runtime UID 'table'.")

    for obj in objects:
        shape = obj.get("shape")
        fpath = shape.get("fpath") if isinstance(shape, dict) else obj.get("fpath")
        if fpath is None:
            continue
        path = Path(str(fpath))
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                f"Generated asset path for {obj.get('uid')!r} is not an "
                f"existing absolute file: {path}"
            )

    action_engine = config.get("env", {}).get("extensions", {}).get("action_engine", {})
    if action_engine.get("defaults_schema_version") != ACTION_ENGINE_DEFAULTS_SCHEMA:
        raise ValueError("Gym config has an unexpected defaults schema version.")
    if action_engine.get("task_spec") != TASK_SPEC_FILENAME:
        raise ValueError("Gym config points to an unexpected TaskSpec artifact.")
    if action_engine.get("scene_requirements") != SCENE_REQUIREMENTS_FILENAME:
        raise ValueError("Gym config points to unexpected SceneRequirements.")
    graph_path = action_engine.get("seed_task_graph")
    if (
        not isinstance(graph_path, str)
        or Path(graph_path).name != EXECUTION_PROGRAM_FILENAME
    ):
        raise ValueError("Gym config points to an unexpected SeedGraph artifact.")

    planning_mode = action_engine.get("planning_mode", "offline")
    _validate_planning_mode(planning_mode)
    if planning_mode == "ab":
        sensors = config["sensor"]
        vlm_sensors = [
            sensor
            for sensor in sensors
            if isinstance(sensor, dict)
            and str(sensor.get("uid", "")).startswith("vlm_")
        ]
        _validate_vlm_sensors(vlm_sensors)

    registered = {
        entry.get("entity_cfg", {}).get("uid")
        for entry in (
            config.get("env", {})
            .get("events", {})
            .get("register_info_to_env", {})
            .get("params", {})
            .get("registry", [])
        )
    }
    rigid_uids = {obj["uid"] for obj in config.get("rigid_object", [])}
    if registered != rigid_uids:
        raise ValueError("Every rigid object must have one live-pose registry entry.")


def _make_robot(
    profile_id: str,
    profile: dict[str, Any],
    table_top_z: float | None,
) -> dict[str, Any]:
    robot = _load_template(str(profile["template"]))
    tabletop_z = _DEFAULT_TABLETOP_Z if table_top_z is None else float(table_top_z)
    robot["init_pos"][2] = round(
        tabletop_z
        + float(profile["tabletop_clearance"])
        - float(profile["arm_component_z"]),
        6,
    )
    family = str(profile["robot_family"])
    if family.startswith("ur"):
        display = family.upper()
        urdf_dir = display
        robot["uid"] = f"Dual{display}"
        robot["urdf_cfg"]["fname"] = f"dual_{family}_robotiq_arg2f_140_basket"
        for component in robot["urdf_cfg"]["components"]:
            if str(component.get("component_type", "")).endswith("_arm"):
                component["urdf_path"] = f"UniversalRobots/{urdf_dir}/{urdf_dir}.urdf"
                component["transform"][0][3] = float(profile["arm_base_x"])
                component["transform"][2][3] = float(profile["arm_component_z"])
        for arm in ("left_arm", "right_arm"):
            robot["solver_cfg"][arm]["ur_type"] = family
            robot["drive_pros"]["max_effort"][arm] = float(profile["max_effort"])
        robot["qpos_control_part_order"] = [
            "left_arm",
            "right_arm",
            "left_eef",
            "right_eef",
        ]
        robot["observation_joint_parts"] = ["left_eef", "right_eef"]
    if profile_id != canonical_robot_profile(profile_id):
        raise ValueError(f"Invalid canonical robot profile {profile_id!r}.")
    return robot


@lru_cache(maxsize=1)
def _robot_profiles() -> dict[str, dict[str, Any]]:
    value = _read_template("robot_profiles.json")
    if not isinstance(value, dict) or not value:
        raise ValueError("robot_profiles.json must contain a non-empty object.")
    return value


def _profile(profile_id: str) -> dict[str, Any]:
    profile = deepcopy(_robot_profiles()[profile_id])
    required = {
        "aliases",
        "template",
        "robot_family",
        "tabletop_clearance",
        "arm_component_z",
        "gripper_open_state",
        "gripper_close_state",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Robot profile {profile_id!r} is missing fields: {missing}.")
    return profile


def _make_events(
    camera: dict[str, Any],
    rigid_uids: list[str],
    *,
    randomize_scene: bool = False,
    randomize_table_material: bool = False,
) -> dict[str, Any]:
    extrinsics = camera["extrinsics"]
    eye = list(extrinsics["eye"])
    target = list(extrinsics["target"])
    # The recording view mirrors the interactive viewer around its target.
    audience_eye = [
        2.0 * float(target[0]) - float(eye[0]),
        2.0 * float(target[1]) - float(eye[1]),
        float(eye[2]),
    ]
    events = {
        "record_camera": {
            "func": "record_camera_data",
            "mode": "interval",
            "interval_step": 1,
            "params": {
                "name": "record_cam_audience_view",
                "resolution": [int(camera["width"]), int(camera["height"])],
                "intrinsics": list(camera["intrinsics"]),
                "eye": audience_eye,
                "target": target,
                "up": [
                    -float(extrinsics["up"][0]),
                    -float(extrinsics["up"][1]),
                    float(extrinsics["up"][2]),
                ],
            },
        },
        "validation_cameras": {
            "func": "validation_cameras",
            "mode": "trigger",
            "params": {},
        },
        "prepare_extra_attr": {
            "func": "prepare_extra_attr",
            "mode": "reset",
            "params": {
                "attrs": [
                    {
                        "name": "object_lengths",
                        "mode": "callable",
                        "entity_uids": "all_objects",
                        "func_name": "compute_object_length",
                        "func_kwargs": {
                            "is_svd_frame": True,
                            "sample_points": int(
                                _GENERATION_DEFAULTS["scene"][
                                    "object_length_sample_points"
                                ]
                            ),
                        },
                    }
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    {
                        "entity_cfg": {"uid": uid},
                        "pose_register_params": {
                            "compute_relative": False,
                            "compute_pose_object_to_arena": True,
                            "to_matrix": True,
                        },
                    }
                    for uid in sorted(rigid_uids)
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }
    if randomize_table_material:
        material = _GENERATION_DEFAULTS["randomization"]["table_material"]
        events["randomize_table_material"] = {
            "func": "randomize_visual_material",
            "mode": "reset",
            "params": {
                "entity_cfg": {"uid": "table"},
                "random_texture_prob": float(material["random_texture_prob"]),
                "base_color_range": deepcopy(material["base_color_range"]),
                "metallic_range": list(material["metallic_range"]),
                "roughness_range": list(material["roughness_range"]),
            },
        }
    if randomize_scene:
        randomization = _GENERATION_DEFAULTS["randomization"]
        for uid in sorted(rigid_uids):
            events[f"randomize_{uid}_pose"] = {
                "func": "randomize_rigid_object_pose",
                "mode": "reset",
                "params": {
                    "entity_cfg": {"uid": uid},
                    "position_range": deepcopy(
                        randomization["rigid_object_position_range"]
                    ),
                    "rotation_range": deepcopy(
                        randomization["rigid_object_rotation_range"]
                    ),
                    "relative_position": True,
                    "relative_rotation": True,
                },
            }
        events["randomize_table_height"] = {
            "func": "randomize_anchor_height",
            "mode": "reset",
            "params": {
                "anchor_uid": "table",
                "height_delta_range": deepcopy(
                    randomization["table_height_delta_range"]
                ),
            },
        }
    return events


def _make_observations(robot: dict[str, Any]) -> dict[str, Any]:
    control_parts = robot["control_parts"]
    qpos_order = robot["qpos_control_part_order"]
    observed_parts = set(robot["observation_joint_parts"])
    offset = 0
    joint_ids: list[int] = []
    for part in qpos_order:
        count = len(control_parts[part])
        if part in observed_parts:
            joint_ids.extend(range(offset, offset + count))
        offset += count
    return {
        "norm_robot_eef_joint": {
            "func": "normalize_robot_joint_data",
            "mode": "modify",
            "name": "robot/qpos",
            "params": {"joint_ids": joint_ids},
        }
    }


def _make_dataset(
    *,
    task_name: str,
    task_description: str,
    source_config_path: Path,
    robot_type: str,
) -> dict[str, Any]:
    dataset_policy = _GENERATION_DEFAULTS["dataset"]
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "save_failed_episodes": bool(dataset_policy["save_failed_episodes"]),
            "params": {
                "robot_meta": {
                    "robot_type": robot_type,
                    "control_freq": int(dataset_policy["control_frequency"]),
                },
                "instruction": {"lang": task_description},
                "extra": {
                    "scene_type": source_config_path.parent.name,
                    "task_name": task_name,
                    # LeRobotRecorder uses this legacy field as a directory label.
                    "task_description": task_name,
                    "data_type": "sim",
                },
                "use_videos": bool(dataset_policy["use_videos"]),
            },
        }
    }


def _load_template(name: str) -> Any:
    return deepcopy(_read_template(name))


@lru_cache(maxsize=None)
def _read_template(name: str) -> Any:
    path = _TEMPLATE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Action Engine template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_planning_mode(value: Any) -> str:
    """Validate and return the two supported generation/runtime modes."""
    if value not in {"offline", "ab"}:
        raise ValueError("planning_mode must be 'offline' or 'ab'.")
    return str(value)


def _validate_seed_graph_path(value: str | Path | None) -> str:
    """Validate a relative or absolute path while preserving caller spelling."""
    if value is None:
        return EXECUTION_PROGRAM_FILENAME
    if not isinstance(value, (str, Path)):
        raise ValueError("seed_task_graph_path must be a non-empty path string.")
    path = str(value).strip()
    if not path:
        raise ValueError("seed_task_graph_path must be a non-empty path string.")
    if Path(path).name != EXECUTION_PROGRAM_FILENAME:
        raise ValueError("seed_task_graph_path must point to seed_task_graph.json.")
    return path


def _optional_model(value: Any) -> str | None:
    """Normalize optional model names without serializing blank strings."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Model name must be a string or None.")
    normalized = value.strip()
    return normalized or None


def _normalize_vlm_camera_uids(value: Sequence[str] | None) -> list[str]:
    """Return the canonical four-camera list used by A/B execution."""
    if value is None:
        return list(VLM_CAMERA_UIDS)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("vlm_camera_uids must be a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise TypeError("vlm_camera_uids must be a list of strings.")
    normalized = [item.strip() for item in value]
    if normalized != list(VLM_CAMERA_UIDS):
        raise ValueError(
            "A/B planning requires VLM cameras in canonical order: "
            f"{list(VLM_CAMERA_UIDS)}."
        )
    return normalized


def _validate_vlm_sensors(value: list[dict[str, Any]]) -> None:
    """Validate camera template fields needed by visual fact extraction."""
    if len(value) != len(VLM_CAMERA_UIDS):
        raise ValueError("A/B planning requires exactly four VLM cameras.")
    if not all(isinstance(sensor, dict) for sensor in value):
        raise ValueError("VLM sensors must be object mappings.")
    uids = [str(sensor.get("uid", "")) for sensor in value]
    if uids != list(VLM_CAMERA_UIDS):
        raise ValueError("VLM camera UIDs must be exactly " f"{list(VLM_CAMERA_UIDS)}.")
    for sensor in value:
        if sensor.get("sensor_type", "Camera") != "Camera":
            raise ValueError(f"VLM sensor {sensor.get('uid')!r} must be a Camera.")
        if int(sensor.get("width", 0)) != 640 or int(sensor.get("height", 0)) != 480:
            raise ValueError("VLM cameras must use 640x480 resolution.")
        if not bool(sensor.get("enable_color")) or not bool(sensor.get("enable_depth")):
            raise ValueError("VLM cameras must enable RGB and depth.")
        extrinsics = sensor.get("extrinsics")
        if not isinstance(extrinsics, dict) or not all(
            key in extrinsics for key in ("eye", "target", "up")
        ):
            raise ValueError(
                f"VLM sensor {sensor.get('uid')!r} requires eye/target/up extrinsics."
            )
        for name in ("eye", "target", "up"):
            vector = extrinsics[name]
            if (
                not isinstance(vector, Sequence)
                or isinstance(vector, (str, bytes, bytearray))
                or len(vector) != 3
            ):
                raise ValueError(
                    f"VLM sensor {sensor.get('uid')!r} {name} must be a 3-vector."
                )
            try:
                values = [float(item) for item in vector]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"VLM sensor {sensor.get('uid')!r} {name} must be numeric."
                ) from exc
            if not all(math.isfinite(item) for item in values):
                raise ValueError(
                    f"VLM sensor {sensor.get('uid')!r} {name} must be finite."
                )


def _anchor_vlm_sensors(sensors: list[dict[str, Any]], scene: PreparedScene) -> None:
    """Aim the fixed high views at the normalized tabletop center."""
    table = next(
        (
            item
            for item in scene.background
            if isinstance(item, dict) and str(item.get("uid")) == "table"
        ),
        None,
    )
    init_pos = table.get("init_pos", [0.0, 0.0, 0.0]) if table else [0.0, 0.0, 0.0]
    if not isinstance(init_pos, Sequence) or len(init_pos) != 3:
        init_pos = [0.0, 0.0, 0.0]
    center = [
        float(init_pos[0]),
        float(init_pos[1]),
        float(scene.table_top_z if scene.table_top_z is not None else 0.75),
    ]
    for sensor in sensors:
        extrinsics = sensor["extrinsics"]
        eye = [float(value) for value in extrinsics["eye"]]
        target = [float(value) for value in extrinsics["target"]]
        offset = [target[index] - 0.0 for index in range(3)]
        extrinsics["target"] = list(center)
        extrinsics["eye"] = [
            center[index] + eye[index] - offset[index] for index in range(3)
        ]
