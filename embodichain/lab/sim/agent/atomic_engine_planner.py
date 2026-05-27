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

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.lab.sim.agent.action_plan import ActionPlan
from embodichain.lab.sim.agent.atomic_action_adapter import (
    _as_pose_tensor,
    _build_legacy_grasp_pose,
    _build_pickup_cfg,
    _build_public_grasp_semantics,
    _object_geometry_bounds,
    _public_grasp_approach_direction,
    _public_grasp_approach_direction_candidates,
    _public_grasp_legacy_reference_required,
    _public_grasp_lift_height,
    _public_grasp_roll_offsets,
    _record_public_grasp_attempt,
    _register_public_grasp_physical_validation,
    _select_arm,
    _state_to_hand_qpos,
    _store_public_grasp_relation,
    _with_public_grasp_strategy_defaults,
)
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    MoveActionCfg,
    PlaceActionCfg,
)
from embodichain.lab.sim.atomic_actions.semantic_grasp import (
    format_semantic_candidate_message,
    semantic_candidate_record_label,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.utils.logger import log_warning

__all__ = ["AtomicEnginePlanner"]


class AtomicEnginePlanner:
    """Plan graph-authored atomic actions through ``AtomicActionEngine``."""

    def plan(self, spec: Mapping[str, Any], *, env, **kwargs) -> ActionPlan:
        if env is None:
            raise RuntimeError("Atomic graph action execution requires env.")

        action_specs = _flatten_action_specs(spec)
        if not action_specs:
            raise RuntimeError("Atomic graph action spec has no actions.")

        robot_names = [_robot_name_from_action_spec(action) for action in action_specs]
        robot_names = [
            robot_name for robot_name in robot_names if robot_name is not None
        ]
        if not robot_names:
            raise RuntimeError("Atomic graph action spec has no control_part.")
        if len(set(robot_names)) != 1:
            raise RuntimeError(
                "Atomic graph action execution currently supports one arm per edge."
            )
        robot_name = robot_names[0]

        if len(action_specs) == 1:
            orientation_plan = _single_eef_orientation_move_plan(
                action_specs[0],
                env=env,
                robot_name=robot_name,
                kwargs=_action_runtime_kwargs(action_specs[0], kwargs),
            )
            if orientation_plan is not None:
                return orientation_plan

        action_kwargs_list = [
            _action_runtime_kwargs(action, kwargs) for action in action_specs
        ]
        cfg_list = [
            _build_action_cfg(
                action,
                env=env,
                robot_name=robot_name,
                kwargs=action_kwargs,
            )
            for action, action_kwargs in zip(action_specs, action_kwargs_list)
        ]
        target_list = [
            _resolve_action_target(
                action,
                env=env,
                robot_name=robot_name,
                kwargs=action_kwargs,
            )
            for action, action_kwargs in zip(action_specs, action_kwargs_list)
        ]

        engine = _create_engine(env, cfg_list)
        is_success, trajectory = engine.execute_static(target_list=target_list)
        if not is_success or trajectory.numel() == 0:
            raise RuntimeError("AtomicActionEngine failed to produce a trajectory.")
        _register_pick_up_metadata(
            env=env,
            robot_name=robot_name,
            action_specs=action_specs,
            action_kwargs_list=action_kwargs_list,
            engine=engine,
        )

        joint_ids = _controlled_joint_ids(env, action_specs, robot_name)
        return ActionPlan(
            is_success=True,
            trajectory=trajectory[:, :, joint_ids],
            joint_ids=joint_ids,
            action_name=str(spec.get("name", spec.get("kind", "atomic_graph_action"))),
        )


def _flatten_action_specs(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    kind = spec.get("kind")
    if kind == "atomic_action":
        return [spec]
    if kind == "atomic_sequence":
        parent_runtime_kwargs = dict(spec.get("runtime_kwargs", {}))
        actions = []
        for action in spec.get("actions", []):
            if not isinstance(action, Mapping) or action.get("kind") != "atomic_action":
                continue
            if parent_runtime_kwargs:
                action = deepcopy(dict(action))
                runtime_kwargs = dict(parent_runtime_kwargs)
                runtime_kwargs.update(dict(action.get("runtime_kwargs", {})))
                action["runtime_kwargs"] = runtime_kwargs
            actions.append(action)
        return actions
    return []


def _action_runtime_kwargs(
    action: Mapping[str, Any],
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(runtime_kwargs)
    action_kwargs = action.get("runtime_kwargs")
    if isinstance(action_kwargs, Mapping):
        merged.update(dict(action_kwargs))
    if bool(runtime_kwargs.get("_edge_is_recovery", False)):
        _apply_recovery_public_grasp_overrides(merged, runtime_kwargs)
    return merged


_RECOVERY_PUBLIC_GRASP_OVERRIDE_KEYS = {
    "recovery_public_grasp_strategy": "public_grasp_strategy",
    "recovery_public_grasp_candidate_num": "public_grasp_candidate_num",
    "recovery_public_grasp_pre_grasp_distance": "public_grasp_pre_grasp_distance",
    "recovery_public_grasp_lift_height": "public_grasp_lift_height",
    "recovery_public_grasp_pose_offset_world": "public_grasp_pose_offset_world",
    "recovery_public_grasp_pose_offset_along_approach": (
        "public_grasp_pose_offset_along_approach"
    ),
    "recovery_validate_public_grasp_after_action": (
        "validate_public_grasp_after_action"
    ),
    "recovery_public_grasp_validation_min_object_lift": (
        "public_grasp_validation_min_object_lift"
    ),
    "recovery_public_grasp_validation_max_object_lift": (
        "public_grasp_validation_max_object_lift"
    ),
    "recovery_public_grasp_validation_max_object_xy_displacement": (
        "public_grasp_validation_max_object_xy_displacement"
    ),
    "recovery_public_grasp_rank_by_legacy_pose": ("public_grasp_rank_by_legacy_pose"),
    "recovery_public_grasp_use_legacy_orientation": (
        "public_grasp_use_legacy_orientation"
    ),
    "recovery_public_grasp_auto_approach_direction": (
        "public_grasp_auto_approach_direction"
    ),
    "recovery_public_grasp_try_approach_directions": (
        "public_grasp_try_approach_directions"
    ),
    "recovery_public_grasp_approach_direction": "public_grasp_approach_direction",
    "recovery_public_grasp_approach_directions": "public_grasp_approach_directions",
}


def _apply_recovery_public_grasp_overrides(
    merged: dict[str, Any],
    runtime_kwargs: dict[str, Any],
) -> None:
    for recovery_key, public_key in _RECOVERY_PUBLIC_GRASP_OVERRIDE_KEYS.items():
        value = runtime_kwargs.get(recovery_key)
        if value is not None:
            merged[public_key] = value
            if recovery_key == "recovery_public_grasp_candidate_num":
                merged["_recovery_public_grasp_candidate_num_override"] = True


def _robot_name_from_action_spec(action: Mapping[str, Any]) -> str | None:
    cfg = dict(action.get("cfg", {}))
    robot_name = cfg.get("arm_control_part", cfg.get("control_part"))
    if robot_name is not None:
        robot_name = str(robot_name)
        if robot_name.endswith("_eef"):
            return "right_arm" if "right" in robot_name else "left_arm"
        return robot_name
    return None


def _build_action_cfg(
    action: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
):
    action_name = action.get("name")
    cfg_payload = dict(action.get("cfg", {}))
    is_left, default_arm_part, default_hand_part = _select_arm(robot_name)
    arm_part = str(cfg_payload.get("control_part", default_arm_part))
    hand_part = str(cfg_payload.get("hand_control_part", default_hand_part))
    hand_joints = env.left_eef_joints if is_left else env.right_eef_joints
    hand_dof = len(hand_joints)
    device = getattr(env.robot, "device", None)

    if action_name == "pick_up":
        obj_name = _target_object_name(action.get("target", {}))
        if obj_name is None:
            raise RuntimeError("pick_up target requires obj_name.")
        graph_kwargs = _prepare_pick_up_rank_kwargs(
            action=action,
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            kwargs=kwargs,
            device=device,
        )
        cfg_payload = dict(action.get("cfg", {}))
        approach_direction = cfg_payload.get("approach_direction")
        if approach_direction is None:
            directions = graph_kwargs.get("_public_grasp_approach_directions") or []
            if directions:
                approach_direction = directions[0][1]
            else:
                approach_direction = _public_grasp_approach_direction(
                    env,
                    robot_name,
                    obj_name,
                    graph_kwargs,
                    device=device,
                )
        else:
            approach_direction = torch.as_tensor(
                approach_direction,
                dtype=torch.float32,
                device=device,
            )
        pre_grasp_distance = _float_option(
            cfg_payload.get(
                "pre_grasp_distance",
                cfg_payload.get("pre_grasp_dis"),
            ),
            default=0.1,
        )
        return _build_pickup_cfg(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_distance,
            kwargs=graph_kwargs,
            approach_direction=approach_direction,
        )

    if action_name == "place":
        return PlaceActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(
                env.open_state,
                hand_dof,
                device=device,
            ),
            hand_close_qpos=_state_to_hand_qpos(
                env.close_state,
                hand_dof,
                device=device,
            ),
            lift_height=_float_option(
                cfg_payload.get("lift_height", cfg_payload.get("pre_place_distance")),
                default=0.08,
            ),
            sample_interval=_int_option(
                cfg_payload.get(
                    "sample_interval",
                    kwargs.get("sample_interval", kwargs.get("sample_num")),
                ),
                default=45,
            ),
            hand_interp_steps=_int_option(
                cfg_payload.get("hand_interp_steps", kwargs.get("hand_interp_steps")),
                default=15,
            ),
            post_open_wait_steps=_int_option(
                cfg_payload.get(
                    "post_open_wait_steps",
                    kwargs.get("public_place_post_open_wait_steps"),
                ),
                default=20,
            ),
        )

    if action_name == "move":
        return MoveActionCfg(
            name="move",
            control_part=arm_part,
            sample_interval=_int_option(
                cfg_payload.get(
                    "sample_interval",
                    kwargs.get("sample_interval", kwargs.get("sample_num")),
                ),
                default=50,
            ),
        )

    raise RuntimeError(f"Unsupported atomic graph action '{action_name}'.")


def _prepare_pick_up_rank_kwargs(
    action: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    obj_name: str,
    kwargs: dict[str, Any],
    device,
) -> dict[str, Any]:
    graph_kwargs = _with_public_grasp_strategy_defaults(kwargs)
    graph_kwargs = dict(graph_kwargs)
    graph_kwargs["obj_name"] = obj_name
    cfg_payload = dict(action.get("cfg", {}))
    for key in ("sample_interval", "hand_interp_steps", "lift_height"):
        if cfg_payload.get(key) is not None:
            graph_kwargs[key] = cfg_payload[key]
    configured_direction = cfg_payload.get("approach_direction")
    if configured_direction is not None:
        approach_directions = [
            (
                "configured",
                torch.as_tensor(
                    configured_direction, dtype=torch.float32, device=device
                ),
            )
        ]
    else:
        approach_directions = _public_grasp_approach_direction_candidates(
            env,
            robot_name,
            obj_name,
            graph_kwargs,
            device=device,
        )
    if not approach_directions:
        approach_directions = [
            (
                "default",
                _public_grasp_approach_direction(
                    env,
                    robot_name,
                    obj_name,
                    graph_kwargs,
                    device=device,
                ),
            )
        ]

    graph_kwargs["_public_grasp_approach_directions"] = approach_directions
    graph_kwargs["_public_grasp_geometry_bounds"] = _object_geometry_bounds(
        env,
        obj_name,
        device=device,
    )
    graph_kwargs["_public_grasp_roll_offsets"] = _public_grasp_roll_offsets(
        env,
        graph_kwargs,
    )
    if _public_grasp_legacy_reference_required(graph_kwargs):
        reference_pose = _build_legacy_grasp_pose(env, robot_name, obj_name)
        if reference_pose is None:
            raise RuntimeError(
                "Public semantic grasp candidate selection requires legacy "
                f"grasp_pose_obj reference for '{obj_name}'."
            )
        graph_kwargs["_public_grasp_reference_pose"] = reference_pose
    return graph_kwargs


def _resolve_action_target(
    action: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
):
    action_name = action.get("name")
    target = action.get("target", {})

    if action_name == "pick_up":
        obj_name = _target_object_name(target)
        if obj_name is None:
            raise RuntimeError("pick_up target requires obj_name.")
        strict = bool(kwargs.get("require_atomic_action_graph", False))
        semantics = _build_public_grasp_semantics(
            env,
            obj_name,
            kwargs,
            strict=strict,
        )
        if semantics is None:
            raise RuntimeError(f"Cannot build ObjectSemantics for '{obj_name}'.")
        return semantics

    if action_name == "place":
        return _resolve_place_target_pose(
            target,
            env=env,
            robot_name=robot_name,
            kwargs=kwargs,
        )

    if action_name == "move":
        if isinstance(target, Mapping) and target.get("kind") == "gripper_state":
            return _resolve_gripper_target(
                target,
                env=env,
                robot_name=robot_name,
            )
        return _resolve_pose_target(target, env=env, robot_name=robot_name)

    raise RuntimeError(f"Unsupported atomic graph action '{action_name}'.")


def _target_object_name(target: Any) -> str | None:
    if not isinstance(target, Mapping):
        return None
    obj_name = target.get("obj_name", target.get("object"))
    return str(obj_name) if obj_name is not None else None


def _float_option(value: Any, *, default: float) -> float:
    return float(default if value is None else value)


def _int_option(value: Any, *, default: int) -> int:
    return int(default if value is None else value)


def _controlled_joint_ids(
    env,
    action_specs: list[Mapping[str, Any]],
    robot_name: str,
) -> list[int]:
    is_left, _, _ = _select_arm(robot_name)
    arm_ids = env.left_arm_joints if is_left else env.right_arm_joints
    hand_ids = env.left_eef_joints if is_left else env.right_eef_joints
    names = {str(action.get("name")) for action in action_specs}
    if _controls_only_hand(action_specs):
        return list(hand_ids)
    if names <= {"move"}:
        if _controls_any_hand(action_specs):
            return list(dict.fromkeys(list(arm_ids) + list(hand_ids)))
        return list(arm_ids)
    return list(dict.fromkeys(list(arm_ids) + list(hand_ids)))


def _controls_only_hand(action_specs: list[Mapping[str, Any]]) -> bool:
    if not action_specs:
        return False
    for action in action_specs:
        cfg = dict(action.get("cfg", {}))
        control_part = str(cfg.get("control_part", ""))
        target = action.get("target", {})
        target_kind = target.get("kind") if isinstance(target, Mapping) else None
        if not control_part.endswith("_eef") and target_kind != "gripper_state":
            return False
    return True


def _controls_any_hand(action_specs: list[Mapping[str, Any]]) -> bool:
    for action in action_specs:
        cfg = dict(action.get("cfg", {}))
        control_part = str(cfg.get("control_part", ""))
        target = action.get("target", {})
        target_kind = target.get("kind") if isinstance(target, Mapping) else None
        if control_part.endswith("_eef") or target_kind == "gripper_state":
            return True
    return False


def _register_pick_up_metadata(
    *,
    env,
    robot_name: str,
    action_specs: list[Mapping[str, Any]],
    action_kwargs_list: list[dict[str, Any]],
    engine: AtomicActionEngine,
) -> None:
    action_sequence = getattr(engine, "_action_sequence", [])
    for index, (action, kwargs) in enumerate(zip(action_specs, action_kwargs_list)):
        if action.get("name") != "pick_up":
            continue
        if index >= len(action_sequence):
            continue
        _action_name, atom_action = action_sequence[index]
        selected = getattr(atom_action, "last_selected_grasp", None)
        if selected is None:
            continue
        obj_name = _target_object_name(action.get("target", {}))
        if obj_name is None:
            continue
        label = semantic_candidate_record_label(selected)
        _store_public_grasp_relation(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            grasp_pose=selected.grasp_pose,
            source=f"semantic:{label}",
        )
        _record_public_grasp_attempt(
            env=env,
            kwargs=kwargs,
            obj_name=obj_name,
            label=label,
            direction=selected.direction,
            status="selected",
            message=format_semantic_candidate_message(selected),
        )
        reference_pose = None
        cfg_options = getattr(
            getattr(atom_action, "cfg", None), "grasp_rank_options", {}
        )
        if isinstance(cfg_options, Mapping):
            reference_pose = cfg_options.get("reference_grasp_pose")
        _register_public_grasp_physical_validation(
            env=env,
            kwargs=kwargs,
            robot_name=robot_name,
            obj_name=obj_name,
            label=label,
            direction=selected.direction,
            lift_height=_public_grasp_lift_height(kwargs),
            legacy_reference_pose=reference_pose,
        )


def _single_eef_orientation_move_plan(
    action: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
) -> ActionPlan | None:
    if action.get("name") != "move":
        return None
    target = action.get("target", {})
    if not isinstance(target, Mapping) or target.get("kind") != "eef_orientation":
        return None
    if target.get("rotation") is not None or target.get("matrix") is not None:
        return None
    if str(target.get("direction", "front")) != "down":
        return None

    hold_plan = _current_eef_down_hold_plan(
        action,
        env=env,
        robot_name=robot_name,
        kwargs=kwargs,
    )
    if hold_plan is not None:
        return hold_plan

    cfg_list = [
        _build_action_cfg(
            action,
            env=env,
            robot_name=robot_name,
            kwargs=kwargs,
        )
    ]
    target_candidates = _eef_orientation_down_pose_candidates(
        target,
        env=env,
        robot_name=robot_name,
        kwargs=kwargs,
    )
    joint_ids = _controlled_joint_ids(env, [action], robot_name)
    attempted_labels: list[str] = []
    for candidate_index, (label, target_pose) in enumerate(target_candidates):
        attempted_labels.append(label)
        engine = _create_engine(env, cfg_list)
        is_success, trajectory = engine.execute_static(target_list=[target_pose])
        if is_success and trajectory.numel() > 0:
            if candidate_index > 0:
                log_warning(
                    "Atomic graph move(eef_orientation=down) succeeded with "
                    f"candidate '{label}' for {robot_name} after failed "
                    f"candidates: {attempted_labels[:-1]}."
                )
            return ActionPlan(
                is_success=True,
                trajectory=trajectory[:, :, joint_ids],
                joint_ids=joint_ids,
                action_name="move",
            )

    raise RuntimeError(
        "AtomicActionEngine failed to produce a trajectory for "
        f"move(eef_orientation=down) on {robot_name}; tried candidates "
        f"{attempted_labels}."
    )


def _current_eef_down_hold_plan(
    action: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
) -> ActionPlan | None:
    target = action.get("target", {})
    if not isinstance(target, Mapping):
        return None

    is_left, arm_part, _ = _select_arm(robot_name)
    device = getattr(env.robot, "device", None)
    current_pose = _as_pose_tensor(
        env.left_arm_current_xpos if is_left else env.right_arm_current_xpos,
        device=device,
    )
    rotation = current_pose[:3, :3]
    min_dot = float(
        target.get(
            "axis_min_dot",
            kwargs.get("eef_orientation_down_axis_min_dot", 0.85),
        )
    )
    axis_scores = {
        axis_name: float((-rotation[:, axis_index].detach().cpu())[2].item())
        for axis_name, axis_index in (("x", 0), ("y", 1), ("z", 2))
    }
    best_axis, best_dot = max(
        axis_scores.items(),
        key=lambda item: item[1],
    )
    if best_dot < min_dot:
        return None

    cfg_payload = dict(action.get("cfg", {}))
    sample_interval = _int_option(
        cfg_payload.get(
            "sample_interval",
            kwargs.get("sample_interval", kwargs.get("sample_num")),
        ),
        default=50,
    )
    current_qpos = env.robot.get_qpos(name=arm_part)
    if current_qpos.ndim == 1:
        current_qpos = current_qpos.unsqueeze(0)
    trajectory = current_qpos.unsqueeze(1).repeat(1, sample_interval, 1)
    log_warning(
        "Atomic graph move(eef_orientation=down) using current-pose hold for "
        f"{robot_name}: local {best_axis}-axis down dot={best_dot:.3f}."
    )
    return ActionPlan(
        is_success=True,
        trajectory=trajectory,
        joint_ids=list(env.left_arm_joints if is_left else env.right_arm_joints),
        action_name="move",
    )


def _eef_orientation_down_pose_candidates(
    target: Mapping[str, Any],
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    device = getattr(env.robot, "device", None)
    is_left, _, _ = _select_arm(robot_name)
    current_pose = _as_pose_tensor(
        env.left_arm_current_xpos if is_left else env.right_arm_current_xpos,
        device=device,
    )
    z_offsets = _float_sequence_option(
        target.get(
            "z_offsets",
            target.get(
                "z_offset",
                kwargs.get("eef_orientation_down_z_offsets", (0.0, 0.03, 0.06)),
            ),
        ),
        default=(0.0, 0.03, 0.06),
    )
    rotations = _eef_down_rotation_candidates(
        current_pose[:3, :3],
        target=target,
        kwargs=kwargs,
        device=device,
    )

    candidates: list[tuple[str, torch.Tensor]] = []
    for z_offset in z_offsets:
        for rotation_label, rotation in rotations:
            pose = current_pose.clone()
            pose[:3, :3] = rotation.to(dtype=pose.dtype, device=pose.device)
            pose[2, 3] = pose[2, 3] + float(z_offset)
            label = rotation_label
            if abs(float(z_offset)) > 1e-9:
                label = f"{label},z_offset={float(z_offset):.3f}"
            candidates.append((label, _as_pose_tensor(pose, device=device)))
    return candidates


def _eef_down_rotation_candidates(
    current_rotation: torch.Tensor,
    *,
    target: Mapping[str, Any],
    kwargs: dict[str, Any],
    device,
) -> list[tuple[str, torch.Tensor]]:
    base_rotation = _orientation_matrix("down", device=device)
    candidates: list[tuple[str, torch.Tensor]] = [("down_default", base_rotation)]
    configured_yaws = target.get(
        "yaw_candidates_degrees",
        target.get(
            "yaw_degrees",
            kwargs.get("eef_orientation_down_yaw_candidates_degrees"),
        ),
    )
    if configured_yaws is None:
        yaw_degrees = _default_eef_down_yaw_candidates(current_rotation)
    else:
        yaw_degrees = _float_sequence_option(configured_yaws, default=())

    for yaw in yaw_degrees:
        rotation = (
            _rotation_matrix_xyz_degrees(
                (0.0, 0.0, float(yaw)),
                device=device,
            )
            @ base_rotation
        )
        if _has_similar_rotation(rotation, candidates):
            continue
        candidates.append((f"yaw={float(yaw):.1f}", rotation))
    return candidates


def _default_eef_down_yaw_candidates(current_rotation: torch.Tensor) -> list[float]:
    yaw_candidates: list[float] = []
    current_yaw = _horizontal_yaw_degrees(current_rotation)
    if current_yaw is not None:
        yaw_candidates.extend(
            [
                current_yaw,
                current_yaw + 90.0,
                current_yaw - 90.0,
                current_yaw + 180.0,
            ]
        )
    yaw_candidates.extend([0.0, 90.0, -90.0, 180.0])
    return _dedupe_degrees(yaw_candidates)


def _horizontal_yaw_degrees(rotation: torch.Tensor) -> float | None:
    x_axis = rotation[:2, 0].detach().cpu()
    norm = float(torch.linalg.norm(x_axis).item())
    if norm < 1e-6:
        return None
    return math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0])))


def _dedupe_degrees(values: list[float]) -> list[float]:
    seen: set[float] = set()
    result: list[float] = []
    for value in values:
        normalized = ((float(value) + 180.0) % 360.0) - 180.0
        key = round(normalized, 4)
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _has_similar_rotation(
    rotation: torch.Tensor,
    candidates: list[tuple[str, torch.Tensor]],
) -> bool:
    for _, candidate in candidates:
        if torch.allclose(
            rotation.to(device=candidate.device, dtype=candidate.dtype),
            candidate,
            atol=1e-5,
            rtol=1e-5,
        ):
            return True
    return False


def _float_sequence_option(value: Any, *, default: tuple[float, ...]) -> list[float]:
    if value is None:
        value = default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [float(part) for part in parts]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def _interpolate_qpos(
    start_qpos: torch.Tensor,
    target_qpos: torch.Tensor,
    *,
    n_waypoints: int,
) -> torch.Tensor:
    if n_waypoints < 1:
        raise RuntimeError("sample_interval must be >= 1.")
    weights = torch.linspace(0, 1, steps=n_waypoints, device=start_qpos.device)
    return torch.stack(
        [torch.lerp(start_qpos, target_qpos, weight) for weight in weights],
        dim=1,
    )


def _resolve_place_target_pose(
    target: Any,
    *,
    env,
    robot_name: str,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if isinstance(target, Mapping) and target.get("pose") is not None:
        return _as_pose_tensor(
            target["pose"], device=getattr(env.robot, "device", None)
        )

    is_left, _, _ = _select_arm(robot_name)
    current_pose = env.left_arm_current_xpos if is_left else env.right_arm_current_xpos
    place_pose = deepcopy(current_pose)
    obj_name = _target_object_name(target)
    obj_info = getattr(env, "obj_info", {}).get(obj_name, {}) if obj_name else {}
    height = obj_info.get("height")
    if height is None and obj_name is not None:
        env.update_obj_info()
        height = getattr(env, "obj_info", {}).get(obj_name, {}).get("height")
    if height is not None:
        place_pose[2, 3] = height + kwargs.get("eps", 0.03)
    if isinstance(target, Mapping):
        if target.get("x") is not None:
            place_pose[0, 3] = float(target["x"])
        if target.get("y") is not None:
            place_pose[1, 3] = float(target["y"])
        if target.get("z") is not None:
            place_pose[2, 3] = float(target["z"])
    return _as_pose_tensor(place_pose, device=getattr(env.robot, "device", None))


def _resolve_pose_target(
    target: Any,
    *,
    env,
    robot_name: str,
) -> torch.Tensor:
    device = getattr(env.robot, "device", None)
    if isinstance(target, Mapping):
        for key in ("pose", "matrix", "target_pose"):
            if target.get(key) is not None:
                return _as_pose_tensor(target[key], device=device)
        target_kind = target.get("kind")
        is_left, _, _ = _select_arm(robot_name)
        current_pose = _as_pose_tensor(
            env.left_arm_current_xpos if is_left else env.right_arm_current_xpos,
            device=device,
        )
        if target_kind == "current_pose":
            return _as_pose_tensor(current_pose, device=device)
        if target_kind == "initial_pose":
            init_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
            return torch.as_tensor(init_qpos, dtype=torch.float32, device=device)
        if target_kind in {"joint_delta", "eef_rotation_delta"}:
            arm_part = "left_arm" if is_left else "right_arm"
            target_qpos = env.robot.get_qpos(name=arm_part)
            if target_qpos.ndim == 1:
                target_qpos = target_qpos.unsqueeze(0)
            joint_index = int(target.get("joint_index", target.get("joint", 5)))
            degree = _float_option(
                target.get("degree", target.get("degrees")),
                default=0.0,
            )
            radian = target.get("radian", target.get("radians"))
            delta = float(radian) if radian is not None else math.radians(degree)
            target_qpos[:, joint_index] += delta
            return target_qpos
        if target_kind == "object_relative_pose":
            pose = current_pose.clone()
            obj_name = _target_object_name(target)
            if obj_name is None:
                raise RuntimeError("object_relative_pose target requires obj_name.")
            obj_pose = _object_pose(env, obj_name, device=device)
            pose[:3, 3] = obj_pose[:3, 3] + _target_offset(target, device=device)
            return _as_pose_tensor(pose, device=device)
        if target_kind == "absolute_position":
            pose = current_pose.clone()
            for index, key in enumerate(("x", "y", "z")):
                if target.get(key) is not None:
                    pose[index, 3] = float(target[key])
            return _as_pose_tensor(pose, device=device)
        if target_kind == "relative_offset":
            pose = current_pose.clone()
            offset = _target_offset(target, device=device)
            mode = str(target.get("mode", target.get("frame", "extrinsic")))
            if mode in {"intrinsic", "eef", "tool"}:
                offset = pose[:3, :3] @ offset
            pose[:3, 3] = pose[:3, 3] + offset
            return _as_pose_tensor(pose, device=device)
        if target_kind == "eef_orientation":
            pose = current_pose.clone()
            if target.get("rotation") is not None:
                rotation = torch.as_tensor(
                    target["rotation"], dtype=torch.float32, device=device
                )
            elif target.get("matrix") is not None:
                matrix = torch.as_tensor(
                    target["matrix"], dtype=torch.float32, device=device
                )
                rotation = matrix[:3, :3]
            else:
                rotation = _orientation_matrix(
                    str(target.get("direction", "front")),
                    device=device,
                )
            pose[:3, :3] = rotation.to(dtype=pose.dtype, device=pose.device)
            return _as_pose_tensor(pose, device=device)
    return _as_pose_tensor(target, device=device)


def _resolve_gripper_target(
    target: Any,
    *,
    env,
    robot_name: str,
) -> torch.Tensor:
    is_left, _, _ = _select_arm(robot_name)
    hand_dof = len(env.left_eef_joints if is_left else env.right_eef_joints)
    device = getattr(env.robot, "device", None)
    if isinstance(target, Mapping):
        for key in ("qpos", "target_qpos", "state_qpos"):
            if target.get(key) is not None:
                return _state_to_hand_qpos(target[key], hand_dof, device=device)
        state = str(target.get("state", "")).lower()
    else:
        state = ""
    if not state:
        raise RuntimeError("gripper_state target requires state 'open' or 'close'.")
    if state in {"open", "opened"}:
        return _state_to_hand_qpos(env.open_state, hand_dof, device=device)
    if state in {"close", "closed"}:
        return _state_to_hand_qpos(env.close_state, hand_dof, device=device)
    raise RuntimeError(f"Unsupported gripper target state '{state}'.")


def _object_pose(env, obj_name: str, *, device) -> torch.Tensor:
    obj_info = getattr(env, "obj_info", None)
    if isinstance(obj_info, Mapping):
        cached = obj_info.get(obj_name, {}).get("pose")
        if cached is not None:
            return _as_pose_tensor(cached, device=device)

    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        return _as_pose_tensor(
            env.sim.get_rigid_object(obj_name)
            .get_local_pose(to_matrix=True)
            .squeeze(0),
            device=device,
        )
    raise RuntimeError(f"No matched object '{obj_name}'.")


def _target_offset(target: Mapping[str, Any], *, device) -> torch.Tensor:
    if target.get("offset") is not None:
        return torch.as_tensor(target["offset"], dtype=torch.float32, device=device)
    return torch.tensor(
        [
            float(target.get("x_offset", target.get("dx", 0.0))),
            float(target.get("y_offset", target.get("dy", 0.0))),
            float(target.get("z_offset", target.get("dz", 0.0))),
        ],
        dtype=torch.float32,
        device=device,
    )


def _orientation_matrix(direction: str, *, device) -> torch.Tensor:
    if direction == "down":
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )
    if direction == "front":
        return _rotation_matrix_xyz_degrees((180.0, -90.0, 0.0), device=device)
    raise RuntimeError("EEF orientation direction must be 'front' or 'down'.")


def _rotation_matrix_xyz_degrees(
    xyz_degrees: tuple[float, float, float],
    *,
    device,
) -> torch.Tensor:
    x, y, z = [math.radians(value) for value in xyz_degrees]
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=torch.float32,
        device=device,
    )
    ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=torch.float32,
        device=device,
    )
    rz = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return rz @ ry @ rx


def _create_engine(env, cfg_list: list[Any]) -> AtomicActionEngine:
    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )
    return AtomicActionEngine(
        motion_generator=motion_generator,
        actions_cfg_list=cfg_list,
    )
