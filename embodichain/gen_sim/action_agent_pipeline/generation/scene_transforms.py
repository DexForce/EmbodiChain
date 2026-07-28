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

"""Apply source-scene transforms and object body-scale policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any
import warnings

from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _clean_vector3,
    _source_body_scale,
    _target_body_scale_vector,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _apply_tabletop_z_placement,
    _mesh_config_world_z_bounds,
    _mesh_config_world_zmax,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    iter_scene_object_configs,
)

__all__ = [
    "_apply_scene_z_rotation",
    "_apply_source_scene_transforms",
    "_maybe_apply_source_scene_body_scale",
    "_maybe_apply_source_scene_xy_scale",
    "_maybe_apply_tabletop_z_placement",
    "_maybe_preserve_source_scene_vertical_contacts",
    "_relative_generated_object_body_scale",
    "_source_objects_by_runtime_uid",
    "_source_scene_body_scale_override",
    "_validate_source_scene_body_scale_mode",
]

_SOURCE_SCENE_BODY_SCALE_MODES = {"preserve", "multiply", "absolute"}


def _apply_source_scene_transforms(
    gym_config: dict[str, Any],
    *,
    runtime_uids: Mapping[str, str],
    by_uid: Mapping[str, SceneObject],
    table_top_z: float,
    preserve_source_scene_geometry: bool,
    source_scene_body_scale_mode: str | None,
    source_scene_z_rotation_degrees: float,
    robot_profile: RobotProfile,
) -> None:
    """Apply the shared source-scene transform pipeline in canonical order.

    Arrangement and stacking bundles must apply these four steps in exactly
    this order, otherwise prompt2scene metric exports drift: XY scale first
    (anchors stay put), then vertical-contact preservation, then tabletop
    z-snap, then world-Z rotation. Centralizing the call sequence prevents the
    two routes from diverging on transform ordering.
    """
    source_objects_by_runtime_uid = _source_objects_by_runtime_uid(
        runtime_uids, by_uid=by_uid
    )
    _maybe_apply_source_scene_xy_scale(
        gym_config,
        source_objects_by_runtime_uid,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    _maybe_preserve_source_scene_vertical_contacts(
        gym_config,
        source_objects_by_runtime_uid,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        robot_profile=robot_profile,
    )
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)


def _maybe_apply_tabletop_z_placement(
    gym_config: dict[str, Any],
    table_top_z: float | None,
    *,
    preserve_source_scene_geometry: bool,
) -> None:
    if preserve_source_scene_geometry:
        return
    _apply_tabletop_z_placement(gym_config, table_top_z)


def _source_objects_by_runtime_uid(
    runtime_uids_by_source_uid: Mapping[str, str],
    *,
    by_uid: Mapping[str, SceneObject],
) -> dict[str, SceneObject]:
    return {
        runtime_uid: by_uid[source_uid]
        for source_uid, runtime_uid in runtime_uids_by_source_uid.items()
        if source_uid in by_uid
    }


def _maybe_apply_source_scene_xy_scale(
    gym_config: dict[str, Any],
    source_objects_by_runtime_uid: Mapping[str, SceneObject],
    *,
    source_scene_body_scale_mode: str | None,
) -> None:
    if source_scene_body_scale_mode in {None, "preserve"}:
        return

    anchor_xy = _scene_xy_scale_anchor(gym_config)
    for obj_config in _iter_scene_pose_configs(gym_config):
        runtime_uid = str(obj_config.get("uid", ""))
        source_obj = source_objects_by_runtime_uid.get(runtime_uid)
        if source_obj is None:
            continue
        _scale_scene_init_pos_xy_about_anchor(obj_config, source_obj, anchor_xy)


def _scene_xy_scale_anchor(gym_config: Mapping[str, Any]) -> list[float]:
    table_config = next(
        (
            obj_config
            for obj_config in _iter_scene_pose_configs(gym_config)
            if obj_config.get("uid") == "table"
        ),
        None,
    )
    if table_config is None:
        return [0.0, 0.0]
    init_pos = _clean_vector3(table_config.get("init_pos", [0.0, 0.0, 0.0]))
    return [init_pos[0], init_pos[1]]


def _scale_scene_init_pos_xy_about_anchor(
    obj_config: dict[str, Any],
    source_obj: SceneObject,
    anchor_xy: Sequence[float],
) -> None:
    source_scale = _source_body_scale(source_obj)
    current_scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    ratio_x = _scale_ratio(current_scale[0], source_scale[0])
    ratio_y = _scale_ratio(current_scale[1], source_scale[1])
    if math.isclose(ratio_x, 1.0, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
        ratio_y, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        return

    init_pos = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    new_x = float(anchor_xy[0]) + (init_pos[0] - anchor_xy[0]) * ratio_x
    new_y = float(anchor_xy[1]) + (init_pos[1] - anchor_xy[1]) * ratio_y
    obj_config["init_pos"] = [
        _round_pose_value(new_x),
        _round_pose_value(new_y),
        _round_pose_value(init_pos[2]),
    ]


def _scale_ratio(current: float, source: float) -> float:
    if math.isclose(float(source), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 1.0
    return float(current) / float(source)


def _maybe_preserve_source_scene_vertical_contacts(
    gym_config: dict[str, Any],
    source_objects_by_runtime_uid: Mapping[str, SceneObject],
    *,
    preserve_source_scene_geometry: bool,
    source_scene_body_scale_mode: str | None,
    robot_profile: RobotProfile | str | None = None,
) -> None:
    if not preserve_source_scene_geometry:
        return
    if source_scene_body_scale_mode in {None, "preserve"}:
        return

    for obj_config in _iter_scene_pose_configs(gym_config):
        runtime_uid = str(obj_config.get("uid", ""))
        source_obj = source_objects_by_runtime_uid.get(runtime_uid)
        if source_obj is None:
            continue
        _preserve_source_scene_vertical_boundary(obj_config, source_obj)
    _sync_robot_init_z_to_current_tabletop(gym_config, robot_profile=robot_profile)


def _preserve_source_scene_vertical_boundary(
    obj_config: dict[str, Any],
    source_obj: SceneObject,
) -> None:
    source_scale = _source_body_scale(source_obj)
    current_scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    if all(
        math.isclose(source, current, rel_tol=0.0, abs_tol=1e-12)
        for source, current in zip(source_scale, current_scale)
    ):
        return

    source_config = dict(obj_config)
    source_config["body_scale"] = source_scale
    source_bounds = _mesh_config_world_z_bounds(source_config)
    current_bounds = _mesh_config_world_z_bounds(obj_config)
    if source_bounds is None or current_bounds is None:
        return

    boundary_index = 1 if obj_config.get("uid") == "table" else 0
    delta_z = source_bounds[boundary_index] - current_bounds[boundary_index]
    if math.isclose(delta_z, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return

    init_pos = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    init_pos[2] = _round_pose_value(init_pos[2] + delta_z)
    obj_config["init_pos"] = init_pos


def _sync_robot_init_z_to_current_tabletop(
    gym_config: dict[str, Any],
    *,
    robot_profile: RobotProfile | str | None = None,
) -> None:
    robot_config = gym_config.get("robot")
    if not isinstance(robot_config, dict):
        return

    table_config = next(
        (
            obj_config
            for obj_config in _iter_scene_pose_configs(gym_config)
            if obj_config.get("uid") == "table"
        ),
        None,
    )
    if table_config is None:
        return

    table_top_z = _mesh_config_world_zmax(table_config)
    if table_top_z is None:
        return

    profile = resolve_robot_profile(
        robot_profile
        or gym_config.get("env", {}).get("extensions", {}).get("agent_robot_profile")
    )
    init_pos = _clean_vector3(robot_config.get("init_pos", [0.0, 0.0, 0.0]))
    init_pos[2] = profile.robot_init_z_from_table_top(table_top_z)
    robot_config["init_pos"] = init_pos


def _apply_scene_z_rotation(
    gym_config: dict[str, Any],
    rotation_degrees: float,
) -> None:
    if not rotation_degrees:
        return
    for obj in _iter_scene_pose_configs(gym_config):
        _rotate_pose_about_world_z(obj, rotation_degrees)


def _iter_scene_pose_configs(gym_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return iter_scene_object_configs(gym_config)


def _rotate_pose_about_world_z(
    obj_config: dict[str, Any],
    rotation_degrees: float,
) -> None:
    position = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    theta = math.radians(float(rotation_degrees))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    obj_config["init_pos"] = [
        _round_pose_value(position[0] * cos_theta - position[1] * sin_theta),
        _round_pose_value(position[0] * sin_theta + position[1] * cos_theta),
        _round_pose_value(position[2]),
    ]

    from scipy.spatial.transform import Rotation

    # Prompt2Scene exports and RigidObject.reset both use intrinsic XYZ Euler angles.
    rotation = _clean_vector3(obj_config.get("init_rot", [0.0, 0.0, 0.0]))
    original = Rotation.from_euler("XYZ", rotation, degrees=True)
    world_z = Rotation.from_rotvec([0.0, 0.0, theta])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        rotated_euler = (world_z * original).as_euler("XYZ", degrees=True)
    obj_config["init_rot"] = [_round_pose_value(value) for value in rotated_euler]


def _round_pose_value(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded


def _validate_source_scene_body_scale_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in _SOURCE_SCENE_BODY_SCALE_MODES:
        expected = ", ".join(sorted(_SOURCE_SCENE_BODY_SCALE_MODES))
        raise ValueError(f"source_scene_body_scale_mode must be one of: {expected}")
    return normalized


def _source_scene_body_scale(
    obj: SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    mode: str,
) -> list[float]:
    if mode == "preserve":
        return _source_body_scale(obj)
    if mode == "multiply":
        target_scale = _target_body_scale_vector(target_body_scale)
        return [
            _round_pose_value(source * multiplier)
            for source, multiplier in zip(_source_body_scale(obj), target_scale)
        ]
    if mode == "absolute":
        return _target_body_scale_vector(target_body_scale)
    raise AssertionError(f"Unhandled source scene body_scale mode: {mode}")


def _source_scene_body_scale_override(
    obj: SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    source_scene_body_scale_mode: str | None,
) -> list[float] | None:
    if source_scene_body_scale_mode is None:
        return None
    return _source_scene_body_scale(
        obj,
        target_body_scale=target_body_scale,
        mode=source_scene_body_scale_mode,
    )


def _maybe_apply_source_scene_body_scale(
    obj_config: dict[str, Any],
    obj: SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    source_scene_body_scale_mode: str | None,
) -> None:
    body_scale = _source_scene_body_scale_override(
        obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    if body_scale is not None:
        obj_config["body_scale"] = body_scale


def _relative_target_body_scale(
    obj: SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
) -> list[float]:
    if source_scene_body_scale_mode is not None:
        return _source_scene_body_scale(
            obj,
            target_body_scale=target_body_scale,
            mode=source_scene_body_scale_mode,
        )
    if source_target_body_scale_multiplier is not None:
        multiplier = float(source_target_body_scale_multiplier)
        return [
            _round_pose_value(value * multiplier) for value in _source_body_scale(obj)
        ]
    if preserve_source_target_body_scale:
        return _source_body_scale(obj)
    return _target_body_scale_vector(target_body_scale)


def _relative_generated_object_body_scale(
    obj: SceneObject,
    *,
    moved_source_uids: set[str],
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
) -> list[float]:
    if obj.source_uid in moved_source_uids:
        return _relative_target_body_scale(
            obj,
            target_body_scale=target_body_scale,
            preserve_source_target_body_scale=preserve_source_target_body_scale,
            source_target_body_scale_multiplier=source_target_body_scale_multiplier,
            source_scene_body_scale_mode=source_scene_body_scale_mode,
        )
    return _source_scene_body_scale_override(
        obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    ) or _source_body_scale(obj)
