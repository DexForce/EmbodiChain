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

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    ACTION_ENGINE_ENV_ID,
    EXECUTION_PROGRAM_FILENAME,
    TASK_AGENT_FILENAME,
)

from .models import PreparedScene

__all__ = [
    "build_agent_config",
    "build_fast_gym_config",
    "canonical_robot_profile",
    "validate_fast_gym_config",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_VIEWER_CAMERA_UID = "cam_high"
_DEFAULT_TABLETOP_Z = 0.7

_ARM_SLOTS = {
    "left": {"arm": "right_arm", "eef": "right_eef"},
    "right": {"arm": "left_arm", "eef": "left_eef"},
}
_GRIPPER_OPEN_STATE = [0.0] * 6
_GRIPPER_CLOSE_STATE = [0.7, -0.7, 0.7, -0.7, -0.7, 0.7]


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
) -> dict[str, Any]:
    """Build the small manifest consumed by ``run_agent``."""
    return {
        "schema_version": ACTION_ENGINE_CONFIG_SCHEMA,
        "task_name": task_name,
        "robot_profile": canonical_robot_profile(robot_profile),
        "task_agent": TASK_AGENT_FILENAME,
        "execution_program": EXECUTION_PROGRAM_FILENAME,
        "execution_program_hash": execution_program_hash,
        "source": {
            "gym_config": source_config_path.as_posix(),
            "uid_map": dict(sorted(uid_map.items())),
        },
    }


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
) -> dict[str, Any]:
    """Build a runnable EmbodiChain gym config from a prepared source scene."""
    if max_episodes < 1:
        raise ValueError("max_episodes must be at least 1.")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be at least 1.")
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
    sensors[0]["uid"] = _VIEWER_CAMERA_UID
    light = _load_template("default_lights.json")

    rigid_uids = [str(config["uid"]) for config in scene.rigid_objects]
    engine_extension = {
        "schema_version": "action_engine_runtime_v2",
        "task_name": task_name,
        "robot_profile": profile,
        "task_agent": TASK_AGENT_FILENAME,
        "execution_program": EXECUTION_PROGRAM_FILENAME,
        "execution_program_hash": execution_program_hash,
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
        "gripper_open_state": list(profile_config["gripper_open_state"]),
        "gripper_close_state": list(profile_config["gripper_close_state"]),
        "arm_aim_yaw_offset": {"left": 3.141592653589793, "right": 0.0},
        "agent_grasp_runtime_defaults": {
            "max_open_length": 0.115,
            "min_open_length": 0.01,
            "finger_length": 0.13,
        },
        "ignore_terminations_during_agent": True,
        "viewer_camera_uid": _VIEWER_CAMERA_UID,
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
    if action_engine.get("task_agent") != TASK_AGENT_FILENAME:
        raise ValueError("Gym config points to an unexpected Task Agent artifact.")
    if action_engine.get("execution_program") != EXECUTION_PROGRAM_FILENAME:
        raise ValueError(
            "Gym config points to an unexpected Execution Program artifact."
        )

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
                component["transform"][1][3] = float(profile["arm_base_y"])
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
                            "sample_points": 5000,
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
        events["randomize_table_material"] = {
            "func": "randomize_visual_material",
            "mode": "reset",
            "params": {
                "entity_cfg": {"uid": "table"},
                "random_texture_prob": 0.0,
                "base_color_range": [
                    [0.55, 0.55, 0.55],
                    [0.95, 0.95, 0.95],
                ],
                "metallic_range": [0.0, 0.15],
                "roughness_range": [0.45, 0.95],
            },
        }
    if randomize_scene:
        for uid in sorted(rigid_uids):
            events[f"randomize_{uid}_pose"] = {
                "func": "randomize_rigid_object_pose",
                "mode": "reset",
                "params": {
                    "entity_cfg": {"uid": uid},
                    "position_range": [
                        [-0.04, -0.04, 0.0],
                        [0.04, 0.04, 0.0],
                    ],
                    "rotation_range": [
                        [0.0, 0.0, -30.0],
                        [0.0, 0.0, 30.0],
                    ],
                    "relative_position": True,
                    "relative_rotation": True,
                },
            }
        events["randomize_table_height"] = {
            "func": "randomize_anchor_height",
            "mode": "reset",
            "params": {
                "anchor_uid": "table",
                "height_delta_range": [[-0.05], [0.05]],
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
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "save_failed_episodes": True,
            "params": {
                "robot_meta": {
                    "robot_type": robot_type,
                    "control_freq": 25,
                },
                "instruction": {"lang": task_description},
                "extra": {
                    "scene_type": source_config_path.parent.name,
                    "task_name": task_name,
                    "task_description": task_description,
                    "data_type": "sim",
                },
                "use_videos": True,
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
