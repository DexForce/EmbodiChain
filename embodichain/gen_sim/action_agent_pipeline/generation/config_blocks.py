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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import copy
import re

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _ArrangementLineSpec,
    _BasketTaskRoles,
    _RelativePlacementSpec,
    _ResolvedTargetReplacement,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_frame_normalization import (
    MeshFrameNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _left_target_text,
    _normalize_runtime_uid,
    _right_target_text,
    _target_task_description_text,
)

__all__ = [
    "_make_background_config",
    "_make_arrangement_dataset_config",
    "_make_arrangement_events_config",
    "_make_dataset_config",
    "_make_events_config",
    "_make_extra_background_config",
    "_make_extra_rigid_object_config",
    "_make_observations_config",
    "_make_container_background_config",
    "_make_relative_background_object_config",
    "_make_relative_dataset_config",
    "_make_relative_events_config",
    "_make_relative_rigid_object_config",
    "_make_target_object_config",
    "_relative_rigid_object_max_convex_hull_num",
    "_relative_static_background_max_convex_hull_num",
    "_source_body_scale",
    "_target_body_scale_vector",
]

_BACKGROUND_MAX_CONVEX_HULL_NUM = 1
_TARGET_MAX_CONVEX_HULL_NUM = 16
_CONTAINER_MAX_CONVEX_HULL_NUM = 8
_EXTRA_RIGID_MAX_CONVEX_HULL_NUM = 1

_BACKGROUND_ATTRS = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
}

_RIGID_OBJECT_ATTRS = {
    "mass": 0.01,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.01,
    "max_depenetration_velocity": 10.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
}


def _target_body_scale_vector(
    target_body_scale: float | list[float] | tuple[float, float, float],
) -> list[float]:
    if isinstance(target_body_scale, (int, float)):
        value = float(target_body_scale)
        return [value, value, value]
    return _clean_vector3(target_body_scale)


def _source_body_scale(obj: _SceneObject) -> list[float]:
    return _clean_vector3(obj.config.get("body_scale", [1.0, 1.0, 1.0]))


def _make_relative_events_config(
    spec: _RelativePlacementSpec,
    registered_runtime_uids: list[str],
    *,
    sensor_config_factory: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "record_camera": _record_camera_event_config(sensor_config_factory),
        "validation_cameras": _validation_cameras_event_config(),
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
                    },
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    _object_registry_entry(uid)
                    for uid in sorted(registered_runtime_uids)
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }


def _make_arrangement_events_config(
    registered_runtime_uids: list[str],
    *,
    sensor_config_factory: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "record_camera": _record_camera_event_config(sensor_config_factory),
        "validation_cameras": _validation_cameras_event_config(),
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
                    },
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    _object_registry_entry(uid)
                    for uid in sorted(registered_runtime_uids)
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }


def _make_events_config(
    roles: _BasketTaskRoles,
    *,
    sensor_config_factory: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "record_camera": _record_camera_event_config(sensor_config_factory),
        "validation_cameras": _validation_cameras_event_config(),
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
                    },
                ]
            },
        },
        "register_info_to_env": {
            "func": "register_info_to_env",
            "mode": "reset",
            "params": {
                "registry": [
                    _object_registry_entry(roles.left_target_runtime_uid),
                    _object_registry_entry(roles.right_target_runtime_uid),
                    _object_registry_entry(roles.container_runtime_uid),
                ],
                "registration": "affordance_datas",
                "sim_update": True,
            },
        },
    }


def _record_camera_event_config(
    sensor_config_factory: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    camera = sensor_config_factory()[0]
    extrinsics = camera["extrinsics"]
    return {
        "func": "record_camera_data",
        "mode": "interval",
        "interval_step": 1,
        "params": {
            "name": "record_cam_high",
            "resolution": [camera["width"], camera["height"]],
            "intrinsics": camera["intrinsics"],
            "eye": extrinsics["eye"],
            "target": extrinsics["target"],
            "up": extrinsics["up"],
        },
    }


def _validation_cameras_event_config() -> dict[str, Any]:
    return {
        "func": "validation_cameras",
        "mode": "trigger",
        "params": {},
    }


def _object_registry_entry(uid: str) -> dict[str, Any]:
    return {
        "entity_cfg": {
            "uid": uid,
        },
        "pose_register_params": {
            "compute_relative": False,
            "compute_pose_object_to_arena": True,
            "to_matrix": True,
        },
    }


def _make_observations_config(
    robot_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    joint_ids = (
        _derive_observation_joint_ids(robot_config) if robot_config is not None else []
    )
    return {
        "norm_robot_eef_joint": {
            "func": "normalize_robot_joint_data",
            "mode": "modify",
            "name": "robot/qpos",
            "params": {
                "joint_ids": joint_ids,
            },
        }
    }


def _derive_observation_joint_ids(robot_config: Mapping[str, Any]) -> list[int]:
    explicit_joint_ids = robot_config.get("observation_joint_ids")
    if explicit_joint_ids is not None:
        return [int(joint_id) for joint_id in explicit_joint_ids]

    control_parts = robot_config.get("control_parts", {})
    if not isinstance(control_parts, Mapping):
        raise ValueError("robot.control_parts must be a mapping to derive joint_ids.")

    observation_parts = robot_config.get("observation_joint_parts")
    if observation_parts is None:
        observation_parts = _default_observation_joint_parts(control_parts)
    observation_parts = [str(part) for part in observation_parts]
    if not observation_parts:
        return []

    qpos_order = robot_config.get("qpos_control_part_order")
    if qpos_order is None:
        qpos_order = _default_qpos_control_part_order(control_parts, observation_parts)
    qpos_order = [str(part) for part in qpos_order]

    missing_parts = set(observation_parts) - set(control_parts)
    if missing_parts:
        raise ValueError(
            "robot.observation_joint_parts contains unknown control parts: "
            f"{', '.join(sorted(missing_parts))}."
        )

    joint_ids: list[int] = []
    offset = 0
    for part in qpos_order:
        part_joint_count = _control_part_joint_count(control_parts.get(part, []))
        if part in observation_parts:
            joint_ids.extend(range(offset, offset + part_joint_count))
        offset += part_joint_count

    init_qpos = robot_config.get("init_qpos")
    if isinstance(init_qpos, Sequence) and not isinstance(init_qpos, (str, bytes)):
        dof = len(init_qpos)
        out_of_range = [joint_id for joint_id in joint_ids if joint_id >= dof]
        if out_of_range:
            raise ValueError(
                "Derived observation joint_ids exceed robot.init_qpos length "
                f"{dof}: {out_of_range}."
            )

    return joint_ids


def _default_observation_joint_parts(
    control_parts: Mapping[str, Any],
) -> list[str]:
    return [
        str(name)
        for name in control_parts
        if any(token in str(name).lower() for token in ("eef", "hand", "gripper"))
    ]


def _default_qpos_control_part_order(
    control_parts: Mapping[str, Any],
    observation_parts: Sequence[str],
) -> list[str]:
    observation_part_set = set(observation_parts)
    non_observation_parts = [
        str(name) for name in control_parts if str(name) not in observation_part_set
    ]
    observation_parts_in_order = [
        str(name) for name in control_parts if str(name) in observation_part_set
    ]
    return non_observation_parts + observation_parts_in_order


def _control_part_joint_count(patterns: Any) -> int:
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, Sequence):
        raise ValueError(
            f"control part joint patterns must be a sequence: {patterns!r}"
        )
    return sum(_joint_pattern_count(str(pattern)) for pattern in patterns)


def _joint_pattern_count(pattern: str) -> int:
    count = 1
    for start, end in re.findall(r"\[([0-9]+)-([0-9]+)\]", pattern):
        count *= abs(int(end) - int(start)) + 1
    return count


def _make_dataset_config(
    project_name: str,
    roles: _BasketTaskRoles,
) -> dict[str, Any]:
    left_target_text = _left_target_text(roles)
    right_target_text = _right_target_text(roles)
    target_description = _target_task_description_text(roles)
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": (
                        f"Use the left UR5 to place the left {left_target_text} into "
                        f"the {roles.container_runtime_uid}, then use the right "
                        f"UR5 to place the right {right_target_text} into the "
                        f"{roles.container_runtime_uid}."
                    ),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": (
                        f"Dual UR5 {target_description}-to-container placement"
                    ),
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _make_relative_dataset_config(
    project_name: str,
    spec: _RelativePlacementSpec,
    *,
    relation_phrase: Callable[[str], str],
) -> dict[str, Any]:
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": _relative_dataset_instruction(
                        spec,
                        relation_phrase=relation_phrase,
                    ),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": spec.task_description,
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _make_arrangement_dataset_config(
    project_name: str,
    spec: _ArrangementLineSpec,
) -> dict[str, Any]:
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": _arrangement_dataset_instruction(spec),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": spec.task_description,
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _arrangement_dataset_instruction(spec: _ArrangementLineSpec) -> str:
    ordered = ", ".join(step.runtime_uid for step in spec.steps)
    return (
        "Move the selected objects to the table center and arrange them "
        f"left-to-right as: {ordered}."
    )


def _relative_dataset_instruction(
    spec: _RelativePlacementSpec,
    *,
    relation_phrase: Callable[[str], str],
) -> str:
    if spec.intent == "hold_hover":
        return " ".join(
            f"Use the {placement.active_side} UR5 to pick up "
            f"{placement.moved_runtime_uid} and keep it hovering in a closed "
            "gripper."
            for placement in spec.placements
        )
    if len(spec.placements) == 1:
        placement = spec.placements[0]
        return (
            f"Use the {placement.active_side} UR5 to move "
            f"{placement.moved_runtime_uid} "
            f"{relation_phrase(placement.relation)} "
            f"{placement.reference_runtime_uid}."
        )
    return " ".join(
        f"Use the {placement.active_side} UR5 to move "
        f"{placement.moved_runtime_uid} "
        f"{relation_phrase(placement.relation)} "
        f"{placement.reference_runtime_uid}."
        for placement in spec.placements
    )


def _make_background_config(
    scene_dir: Path,
    obj: _SceneObject,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    shape = _make_shape_config(scene_dir, obj.config, mesh_normalizer=mesh_normalizer)
    return {
        "uid": "table",
        "shape": shape,
        "attrs": dict(_BACKGROUND_ATTRS),
        "body_scale": _clean_vector3(obj.config.get("body_scale", [1.0, 1.0, 1.0])),
        "body_type": "kinematic",
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "max_convex_hull_num": _role_limited_max_convex_hull_num(
            obj,
            _BACKGROUND_MAX_CONVEX_HULL_NUM,
        ),
    }


def _make_extra_background_config(
    scene_dir: Path,
    obj: _SceneObject,
    mesh_normalizer: MeshFrameNormalizer | None,
    body_scale: Any | None = None,
    runtime_uid: str | None = None,
) -> dict[str, Any]:
    shape = _make_shape_config(scene_dir, obj.config, mesh_normalizer=mesh_normalizer)
    config = {
        "uid": runtime_uid or _normalize_runtime_uid(obj.source_uid),
        "shape": shape,
        "attrs": copy.deepcopy(dict(obj.config.get("attrs", _BACKGROUND_ATTRS))),
        "body_scale": _clean_vector3(
            obj.config.get("body_scale", [1.0, 1.0, 1.0])
            if body_scale is None
            else body_scale
        ),
        "body_type": str(obj.config.get("body_type", "static")),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "max_convex_hull_num": _role_limited_max_convex_hull_num(
            obj,
            _BACKGROUND_MAX_CONVEX_HULL_NUM,
        ),
    }
    return config


def _make_target_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    target_scale: list[float],
    mesh_normalizer: MeshFrameNormalizer | None,
    replacement: _ResolvedTargetReplacement | None = None,
) -> dict[str, Any]:
    config = _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        target_scale,
        max_convex_hull_num=_TARGET_MAX_CONVEX_HULL_NUM,
        mesh_fpath=replacement.mesh_path if replacement else None,
        mesh_normalizer=mesh_normalizer,
    )
    config["body_type"] = "dynamic"
    return config


def _make_container_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            _CONTAINER_MAX_CONVEX_HULL_NUM,
        ),
        mesh_normalizer=mesh_normalizer,
    )


def _make_container_background_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    config = _make_container_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        mesh_normalizer,
    )
    config["body_type"] = "kinematic"
    return config


def _make_relative_background_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    *,
    body_scale: Any | None = None,
    max_convex_hull_num: int,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    config = _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        _source_body_scale(obj) if body_scale is None else body_scale,
        max_convex_hull_num=max_convex_hull_num,
        mesh_normalizer=mesh_normalizer,
    )
    config["body_type"] = "kinematic"
    return config


def _make_extra_rigid_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    body_scale: Any,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    return _make_rigid_object_config(
        scene_dir,
        obj,
        _normalize_runtime_uid(obj.source_uid),
        body_scale,
        max_convex_hull_num=_role_limited_max_convex_hull_num(
            obj,
            _EXTRA_RIGID_MAX_CONVEX_HULL_NUM,
        ),
        mesh_normalizer=mesh_normalizer,
    )


def _make_relative_rigid_object_config(
    *,
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    max_convex_hull_num: int,
    mesh_normalizer: MeshFrameNormalizer | None,
) -> dict[str, Any]:
    if max_convex_hull_num == _TARGET_MAX_CONVEX_HULL_NUM:
        resolved_max_convex_hull_num = max_convex_hull_num
    else:
        resolved_max_convex_hull_num = _role_limited_max_convex_hull_num(
            obj,
            max_convex_hull_num,
        )
    config = _make_rigid_object_config(
        scene_dir,
        obj,
        runtime_uid,
        body_scale,
        max_convex_hull_num=resolved_max_convex_hull_num,
        mesh_normalizer=mesh_normalizer,
    )
    config["body_type"] = "dynamic"
    return config


def _make_rigid_object_config(
    scene_dir: Path,
    obj: _SceneObject,
    runtime_uid: str,
    body_scale: Any,
    max_convex_hull_num: int,
    mesh_fpath: str | Path | None = None,
    mesh_normalizer: MeshFrameNormalizer | None = None,
) -> dict[str, Any]:
    shape = _make_shape_config(
        scene_dir,
        obj.config,
        mesh_fpath=mesh_fpath,
        mesh_normalizer=mesh_normalizer,
    )
    config = {
        "uid": runtime_uid,
        "shape": shape,
        "attrs": dict(_RIGID_OBJECT_ATTRS),
        "init_pos": _clean_vector3(obj.config.get("init_pos", [0.0, 0.0, 0.0])),
        "init_rot": _clean_vector3(obj.config.get("init_rot", [0.0, 0.0, 0.0])),
        "body_scale": _clean_vector3(body_scale),
        "max_convex_hull_num": int(max_convex_hull_num),
    }
    if "body_type" in obj.config:
        config["body_type"] = str(obj.config["body_type"])
    return config


def _role_limited_max_convex_hull_num(
    obj: _SceneObject,
    role_max_convex_hull_num: int,
) -> int:
    source_max_convex_hull_num = obj.config.get("max_convex_hull_num")
    if source_max_convex_hull_num is None:
        return role_max_convex_hull_num
    return max(1, min(int(source_max_convex_hull_num), role_max_convex_hull_num))


def _relative_rigid_object_max_convex_hull_num(
    runtime_uid: str,
    spec: _RelativePlacementSpec,
) -> int:
    for placement in spec.placements:
        if (
            placement.relation == "inside"
            and runtime_uid == placement.reference_runtime_uid
        ):
            return _CONTAINER_MAX_CONVEX_HULL_NUM
    task_uids = {
        uid
        for placement in spec.placements
        for uid in (placement.moved_runtime_uid, placement.reference_runtime_uid)
    }
    if runtime_uid in task_uids:
        return _TARGET_MAX_CONVEX_HULL_NUM
    return _EXTRA_RIGID_MAX_CONVEX_HULL_NUM


def _relative_static_background_max_convex_hull_num(
    runtime_uid: str,
    spec: _RelativePlacementSpec,
) -> int:
    for placement in spec.placements:
        if (
            placement.relation == "inside"
            and runtime_uid == placement.reference_runtime_uid
        ):
            return _CONTAINER_MAX_CONVEX_HULL_NUM
    return _BACKGROUND_MAX_CONVEX_HULL_NUM


def _make_shape_config(
    scene_dir: Path,
    source_config: Mapping[str, Any],
    *,
    mesh_fpath: str | Path | None = None,
    mesh_normalizer: MeshFrameNormalizer | None = None,
) -> dict[str, Any]:
    shape = copy.deepcopy(dict(source_config.get("shape", {})))
    if mesh_fpath is not None:
        shape["shape_type"] = "Mesh"
        shape["fpath"] = str(mesh_fpath)
    if shape.get("shape_type") == "Mesh" and "fpath" in shape:
        mesh_path = Path(_asset_path_for_config(scene_dir, str(shape["fpath"])))
        if mesh_normalizer is not None:
            mesh_path = mesh_normalizer.normalize_path(mesh_path)
        shape["fpath"] = mesh_path.as_posix()
    shape.setdefault("compute_uv", False)
    return shape


def _asset_path_for_config(scene_dir: Path, fpath: str) -> str:
    raw_path = Path(fpath)
    if raw_path.is_absolute():
        return raw_path.resolve().as_posix()
    return (scene_dir / raw_path).resolve().as_posix()
