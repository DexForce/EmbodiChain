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

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_engine.config import (
    ACTION_ENGINE_DEFAULTS_SCHEMA,
    RuntimePolicyCfg,
    default_runtime_policy,
    generation_defaults,
    runtime_policy_hash,
)
from embodichain.gen_sim.action_engine.config.runtime_policy import (
    _resolve_planner_policy,
)
from embodichain.gen_sim.action_engine.gripper_profiles import (
    GripperProfile,
    get_gripper_profile,
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
    "canonical_gripper_model",
    "canonical_robot_profile",
    "VLM_CAMERA_UIDS",
    "validate_fast_gym_config",
]

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_GENERATION_DEFAULTS = generation_defaults()
_DEFAULT_TABLETOP_Z = float(_GENERATION_DEFAULTS["scene"]["default_tabletop_z"])
_DEFAULT_GRIPPER_MODEL = str(_GENERATION_DEFAULTS["task"]["default_gripper_model"])

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


def canonical_gripper_model(model: str) -> str:
    """Validate and return one exact GenSim gripper model ID."""
    return get_gripper_profile(model).model.value


def build_agent_config(
    *,
    task_name: str,
    robot_profile: str,
    execution_program_hash: str,
    source_config_path: Path,
    uid_map: dict[str, str],
    gripper_model: str = _DEFAULT_GRIPPER_MODEL,
    static_obstacle_uids: Sequence[str] | None = None,
    dynamic_obstacle_uids: Sequence[str] | None = None,
    table_top_z: float | None = None,
    articulation_settings: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    planning_mode: str = "offline",
    seed_task_graph_path: str | Path | None = EXECUTION_PROGRAM_FILENAME,
    vlm_model: str | None = None,
    vlm_camera_uids: Sequence[str] | None = None,
    planner_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the small manifest consumed by ``run_agent``."""
    profile = canonical_robot_profile(robot_profile)
    selected_gripper = canonical_gripper_model(gripper_model)
    runtime_policy = default_runtime_policy(profile)
    explicit_dynamic_collision = (
        planner_policy is not None and "dynamic_collision" in planner_policy
    )
    if planner_policy is not None:
        policy = runtime_policy.as_mapping()
        policy["planner"] = _resolve_planner_policy(
            planner_policy,
            robot_profile=profile,
        )
        runtime_policy = RuntimePolicyCfg.from_mapping(policy)
    if (
        static_obstacle_uids is not None
        or dynamic_obstacle_uids is not None
        or table_top_z is not None
    ):
        policy = runtime_policy.as_mapping()
        planner = policy["planner"]
        if static_obstacle_uids is not None:
            planner["static_obstacle_uids"] = [str(uid) for uid in static_obstacle_uids]
        if dynamic_obstacle_uids is not None:
            planner["dynamic_obstacle_uids"] = [
                str(uid) for uid in dynamic_obstacle_uids
            ]
            if not explicit_dynamic_collision:
                planner["dynamic_collision"] = bool(dynamic_obstacle_uids) and (
                    planner["backend"] == "curobo"
                )
        if table_top_z is not None:
            tabletop = float(table_top_z)
            if not math.isfinite(tabletop):
                raise ValueError("table_top_z must be finite when provided.")
            height_offset = tabletop - _DEFAULT_TABLETOP_Z
            height_policies = (
                policy["grounding"]["semantic_defaults"],
                policy["grounding"]["handover"],
                policy["motion_defaults"]["MoveEndEffector"],
                policy["motion_modifiers"]["orientation"]["upright"]["MoveEndEffector"],
            )
            for height_policy in height_policies:
                height_policy["maximum_eef_height"] = round(
                    float(height_policy["maximum_eef_height"]) + height_offset,
                    6,
                )
        runtime_policy = RuntimePolicyCfg.from_mapping(policy)
    _validate_planning_mode(planning_mode)
    graph_path = _validate_seed_graph_path(seed_task_graph_path)
    if planning_mode == "ab" and graph_path == EXECUTION_PROGRAM_FILENAME:
        graph_path = f"offline/{EXECUTION_PROGRAM_FILENAME}"
    result = {
        "schema_version": ACTION_ENGINE_CONFIG_SCHEMA,
        "task_name": task_name,
        "robot_profile": profile,
        "gripper_model": selected_gripper,
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
        "articulation_settings": _normalize_articulation_settings(
            articulation_settings or {}
        ),
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


def _normalize_articulation_settings(
    value: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Own finite per-joint ordinal setting calibrations for runtime grounding."""
    result: dict[str, dict[str, list[float]]] = {}
    for uid, joints in value.items():
        if not isinstance(uid, str) or not uid or not isinstance(joints, Mapping):
            raise ValueError("articulation_settings must map UIDs to joint mappings.")
        normalized_joints = {}
        for joint_name, settings in joints.items():
            if (
                not isinstance(joint_name, str)
                or not joint_name
                or not isinstance(settings, Sequence)
                or isinstance(settings, (str, bytes, bytearray))
                or not settings
            ):
                raise ValueError(
                    "articulation_settings joints require non-empty setting lists."
                )
            normalized = [float(item) for item in settings]
            if any(not math.isfinite(item) for item in normalized):
                raise ValueError("articulation setting values must be finite.")
            normalized_joints[joint_name] = normalized
        result[uid] = normalized_joints
    return dict(sorted(result.items()))


def build_fast_gym_config(
    scene: PreparedScene,
    *,
    task_name: str,
    task_description: str,
    robot_profile: str,
    execution_program_hash: str,
    max_episodes: int,
    max_episode_steps: int,
    gripper_model: str = _DEFAULT_GRIPPER_MODEL,
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
    gripper_profile = get_gripper_profile(gripper_model)

    profile_config = _profile(profile)
    robot = _make_robot(
        profile,
        profile_config,
        scene.table_top_z,
        gripper_profile=gripper_profile,
    )
    observations = _make_observations(robot, gripper_profile)
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
        "gripper_model": gripper_profile.model.value,
        "planning_mode": planning_mode,
        "task_spec": TASK_SPEC_FILENAME,
        "scene_requirements": SCENE_REQUIREMENTS_FILENAME,
        "seed_task_graph": graph_path,
        "seed_task_graph_hash": execution_program_hash,
        "source_gym_config": scene.source_config_path.as_posix(),
        "source_scene_z_rotation_degrees": scene.z_rotation_degrees,
        "source_scene_xy_translation": list(scene.source_scene_xy_translation),
        "body_scale_policy": scene.body_scale_policy,
        "body_scale": list(scene.body_scale),
        "asset_hashes": dict(sorted(scene.asset_hashes.items())),
        "asset_provenance": [deepcopy(value) for value in scene.asset_provenance],
        "uid_map": dict(sorted(scene.uid_map.items())),
    }
    extensions = {
        "action_engine": engine_extension,
        "agent_robot_profile": profile,
        "agent_gripper_model": gripper_profile.model.value,
        "agent_arm_slots": deepcopy(_ARM_SLOTS),
        "agent_static_obstacle_uids": background_uids,
        "agent_dynamic_obstacle_uids": rigid_uids,
        "gripper_open_state": list(gripper_profile.open_positions),
        "gripper_close_state": list(gripper_profile.close_positions),
        "gripper_profile": gripper_profile.runtime_manifest(
            tcp_parent_frames={
                "left": str(robot["solver_cfg"]["left_arm"]["end_link_name"]),
                "right": str(robot["solver_cfg"]["right_arm"]["end_link_name"]),
            }
        ),
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
                planning_mode=planning_mode,
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
            {
                key: deepcopy(value)
                for key, value in articulation.items()
                if key not in {"attributes", "role"}
            }
            for articulation in scene.articulations
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
    gripper_profile = get_gripper_profile(action_engine.get("gripper_model"))
    extensions = config["env"]["extensions"]
    if extensions.get("agent_gripper_model") != gripper_profile.model.value:
        raise ValueError("Gym config gripper model fields do not match.")
    _validate_robot_gripper_contract(config["robot"], gripper_profile)
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
    *,
    gripper_profile: GripperProfile,
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
        robot["urdf_cfg"][
            "fname"
        ] = f"dual_{family}_{gripper_profile.assembly_name}_basket"
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
    else:
        robot["urdf_cfg"][
            "fname"
        ] = f"dual_{family}_{gripper_profile.assembly_name}_basket"
    _apply_gripper_profile(robot, gripper_profile)
    if profile_id != canonical_robot_profile(profile_id):
        raise ValueError(f"Invalid canonical robot profile {profile_id!r}.")
    return robot


def _apply_gripper_profile(
    robot: dict[str, Any],
    profile: GripperProfile,
) -> None:
    """Apply one profile atomically to simulator, controller, and solver config."""
    control_parts = robot.get("control_parts")
    init_qpos = robot.get("init_qpos")
    if not isinstance(control_parts, dict) or not isinstance(init_qpos, list):
        raise ValueError("Robot template requires control_parts and init_qpos.")
    arm_dof = sum(
        len(control_parts.get(f"{side}_arm", ())) for side in ("left", "right")
    )
    if arm_dof <= 0 or len(init_qpos) < arm_dof:
        raise ValueError("Robot template has an invalid initial arm posture.")
    arm_init_qpos = list(init_qpos[:arm_dof])

    components = robot.get("urdf_cfg", {}).get("components")
    if not isinstance(components, list):
        raise ValueError("Robot template requires a URDF component list.")
    hands = {
        str(component.get("component_type")): component
        for component in components
        if str(component.get("component_type", "")).endswith("_hand")
    }
    if set(hands) != {"left_hand", "right_hand"}:
        raise ValueError("Robot template requires exactly one left and right hand.")
    for component in hands.values():
        component["urdf_path"] = profile.asset_path

    for side in ("left", "right"):
        control_parts[f"{side}_eef"] = list(profile.control_joint_names(side))
    robot["init_qpos"] = (
        arm_init_qpos + list(profile.simulated_joint_initial_positions) * 2
    )

    drive = robot.get("drive_pros")
    if not isinstance(drive, dict):
        raise ValueError("Robot template requires drive_pros.")
    for section, value in (
        ("stiffness", profile.drive_stiffness),
        ("damping", profile.drive_damping),
        ("max_effort", profile.drive_max_effort),
    ):
        values = drive.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Robot drive_pros.{section} must be a mapping.")
        for side in ("left", "right"):
            values[f"{side}_eef"] = value

    solvers = robot.get("solver_cfg")
    if not isinstance(solvers, dict):
        raise ValueError("Robot template requires solver_cfg.")
    tcp = [list(row) for row in profile.tcp_transform]
    for arm in ("left_arm", "right_arm"):
        if not isinstance(solvers.get(arm), dict):
            raise ValueError(f"Robot template requires solver_cfg.{arm}.")
        solvers[arm]["tcp"] = deepcopy(tcp)


def _validate_robot_gripper_contract(
    robot: Mapping[str, Any],
    profile: GripperProfile,
) -> None:
    """Reject generated artifacts whose physical and planning profiles drift."""
    components = robot.get("urdf_cfg", {}).get("components", [])
    hand_assets = {
        str(component.get("urdf_path"))
        for component in components
        if isinstance(component, Mapping)
        and str(component.get("component_type", "")).endswith("_hand")
    }
    if hand_assets != {profile.asset_path}:
        raise ValueError("Robot hand assets do not match the selected gripper profile.")
    control_parts = robot.get("control_parts", {})
    for side in ("left", "right"):
        if control_parts.get(f"{side}_eef") != list(profile.control_joint_names(side)):
            raise ValueError(
                f"Robot {side} gripper controls do not match the selected profile."
            )
    expected_tcp = [list(row) for row in profile.tcp_transform]
    for arm in ("left_arm", "right_arm"):
        if robot.get("solver_cfg", {}).get(arm, {}).get("tcp") != expected_tcp:
            raise ValueError(
                f"Robot {arm} TCP does not match the selected gripper profile."
            )


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
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Robot profile {profile_id!r} is missing fields: {missing}.")
    return profile


def _make_events(
    camera: dict[str, Any],
    rigid_uids: list[str],
    *,
    planning_mode: str,
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
    recording_enabled, recording_resolution, recording_interval = _recording_policy(
        planning_mode
    )
    source_width = int(camera["width"])
    source_height = int(camera["height"])
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Recording source camera resolution must be positive.")
    intrinsics = camera.get("intrinsics")
    if (
        not isinstance(intrinsics, Sequence)
        or isinstance(intrinsics, (str, bytes, bytearray))
        or len(intrinsics) != 4
    ):
        raise ValueError("Recording source camera intrinsics must be a 4-vector.")
    scale_x = recording_resolution[0] / source_width
    scale_y = recording_resolution[1] / source_height
    recording_intrinsics = [
        float(intrinsics[0]) * scale_x,
        float(intrinsics[1]) * scale_y,
        float(intrinsics[2]) * scale_x,
        float(intrinsics[3]) * scale_y,
    ]
    events = {
        "record_camera": {
            "func": "record_camera_data",
            "mode": "interval",
            "interval_step": recording_interval,
            "params": {
                "name": "record_cam_audience_view",
                "resolution": list(recording_resolution),
                "intrinsics": recording_intrinsics,
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
                        "entity_uids": list(rigid_uids),
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
    if not recording_enabled:
        events.pop("record_camera")
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


def _recording_policy(planning_mode: str) -> tuple[bool, tuple[int, int], int]:
    """Resolve the bounded GenSim audience-recording policy."""
    value = _GENERATION_DEFAULTS["environment"].get("recording")
    required = {"enabled", "resolution", "interval_step"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "generation.environment.recording must define enabled, resolution, "
            "and interval_step."
        )
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("generation.environment.recording.enabled must be a boolean.")
    resolution = value["resolution"]
    if (
        not isinstance(resolution, Sequence)
        or isinstance(resolution, (str, bytes, bytearray))
        or len(resolution) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in resolution
        )
        or any(int(item) <= 0 for item in resolution)
    ):
        raise ValueError(
            "generation.environment.recording.resolution must contain two "
            "positive integers."
        )
    interval_step = value["interval_step"]
    if (
        isinstance(interval_step, bool)
        or not isinstance(interval_step, int)
        or interval_step <= 0
    ):
        raise ValueError(
            "generation.environment.recording.interval_step must be positive."
        )
    return (
        bool(enabled or planning_mode == "ab"),
        (int(resolution[0]), int(resolution[1])),
        int(interval_step),
    )


def _make_observations(
    robot: dict[str, Any],
    gripper_profile: GripperProfile,
) -> dict[str, Any]:
    per_hand_dof = len(gripper_profile.simulated_joint_initial_positions)
    arm_dof = len(robot["init_qpos"]) - 2 * per_hand_dof
    if arm_dof <= 0:
        raise ValueError("Robot initial posture does not contain arm joints.")
    joint_ids: list[int] = []
    for side_index, side in enumerate(("left", "right")):
        simulated = gripper_profile.simulated_joint_names(side)
        base = arm_dof + side_index * per_hand_dof
        joint_ids.extend(
            base + simulated.index(name)
            for name in gripper_profile.control_joint_names(side)
        )
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
