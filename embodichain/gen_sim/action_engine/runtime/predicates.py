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

"""Evaluate canonical closed-loop predicates against live environment state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from .robot_parts import arm_control_part

__all__ = ["PREDICATE_TYPES", "evaluate_predicate"]

PREDICATE_TYPES = frozenset(
    {
        "both_arms_at_initial_qpos",
        "both_grippers_open",
        "coordinated_placed",
        "grippers_clear_of_object",
        "held_by_both_grippers",
        "object_axis_near",
        "object_axis_offset_near",
        "object_held",
        "object_held_by_both_grippers",
        "object_held_by_gripper",
        "object_in_container",
        "object_lifted",
        "object_not_fallen",
        "object_on_object",
        "object_position_near",
        "object_upright",
        "object_xy_near",
        "objects_collinear",
        "objects_ordered",
        "pressed",
    }
)


def _constant(env: Any, value: bool) -> torch.Tensor:
    return torch.full(
        (int(env.num_envs),),
        value,
        dtype=torch.bool,
        device=env.device,
    )


def _pose(env: Any, uid: str) -> torch.Tensor:
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown rigid object {uid!r}.")
    pose = torch.as_tensor(
        entity.get_local_pose(to_matrix=True),
        dtype=torch.float32,
        device=env.device,
    )
    if pose.ndim == 2:
        pose = pose.unsqueeze(0).repeat(int(env.num_envs), 1, 1)
    return pose


def _position(env: Any, uid: str) -> torch.Tensor:
    return _pose(env, uid)[:, :3, 3]


def _objects(spec: Mapping[str, Any]) -> list[str]:
    values = spec.get("objects", spec.get("object_uids"))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Predicate requires a non-empty objects list.")
    return [str(value) for value in values]


def _object(spec: Mapping[str, Any]) -> str:
    value = spec.get("object", spec.get("object_uid"))
    if not isinstance(value, str) or not value:
        raise ValueError("Predicate requires a non-empty object uid.")
    return value


def _local_axis_index(env: Any, uid: str, axis: Any) -> int:
    name = str(axis).lower()
    if name in {"x", "y", "z"}:
        return {"x": 0, "y": 1, "z": 2}[name]
    if name not in {"long", "long_axis", "longest"}:
        raise ValueError(f"Unsupported upright local axis {axis!r}.")
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown rigid object {uid!r}.")
    vertices = entity.get_vertices(env_ids=[0], scale=True)
    if isinstance(vertices, (tuple, list)):
        vertices = vertices[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=env.device)
    if vertices.ndim == 3:
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError(f"Rigid object {uid!r} has invalid mesh vertices.")
    extents = vertices.max(dim=0).values - vertices.min(dim=0).values
    return int(torch.argmax(extents).item())


def _arm_values(
    env: Any, kind: str
) -> tuple[torch.Tensor | None, torch.Tensor | None] | None:
    getter = getattr(env, f"get_current_{kind}_agent", None)
    if callable(getter):
        left, right = getter()
        values = []
        for value in (left, right):
            if value is None:
                values.append(None)
                continue
            item = torch.as_tensor(value, device=env.device)
            if kind == "xpos" and item.ndim == 2:
                item = item.unsqueeze(0).repeat(int(env.num_envs), 1, 1)
            elif kind == "gripper_state" and item.ndim == 1:
                item = item.unsqueeze(0)
            values.append(item)
        return values[0], values[1]
    if kind != "gripper_state":
        return None
    qpos = env.robot.get_qpos()
    values = []
    for side in ("left", "right"):
        ids = list(getattr(env, f"{side}_eef_joints", ()))
        if not ids:
            return None
        values.append(qpos[:, ids])
    return values[0], values[1]


def _gripper_has_closed(
    env: Any,
    gripper: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    """Check closure intent without requiring an impossible empty-gripper pose."""
    gripper = gripper.to(device=env.device, dtype=torch.float32)
    open_state = getattr(env, "open_state", None)
    close_state = getattr(env, "close_state", None)
    reference = open_state if open_state is not None else close_state
    if reference is None:
        return _constant(env, False)
    expected = torch.as_tensor(
        reference,
        dtype=torch.float32,
        device=env.device,
    ).flatten()
    repeats = (gripper.shape[-1] + expected.numel() - 1) // expected.numel()
    expected = expected.repeat(repeats)[: gripper.shape[-1]]
    distance = torch.linalg.vector_norm(gripper - expected, dim=-1)
    if open_state is not None:
        return distance > tolerance
    return distance <= tolerance


def _object_held(
    env: Any,
    uid: str,
    *,
    owners: Mapping[str, Sequence[str | None]] | None,
    states: Mapping[tuple[str, str], Any] | None,
    position_tolerance: float,
    gripper_tolerance: float,
    required_arm: str | None = None,
) -> torch.Tensor:
    """Verify registry ownership against live object, TCP, and gripper state."""
    result = _constant(env, False)
    if owners is None or states is None or uid not in owners:
        return result
    eef_values = _arm_values(env, "xpos")
    gripper_values = _arm_values(env, "gripper_state")
    if eef_values is None or gripper_values is None:
        return result

    object_pose = _pose(env, uid)
    for arm_index, arm in enumerate(("left_arm", "right_arm")):
        if required_arm is not None and arm != required_arm:
            continue
        state = states.get((uid, arm))
        held = (
            None if state is None else state.get_held_object(arm_control_part(env, arm))
        )
        actual_eef = eef_values[arm_index]
        gripper = gripper_values[arm_index]
        if held is None or actual_eef is None or gripper is None:
            continue
        label = getattr(held.semantics, "label", None)
        if not label and held.semantics.entity is not None:
            label = getattr(held.semantics.entity, "uid", None)
        if label != uid:
            continue
        actual_eef = actual_eef.to(device=env.device, dtype=object_pose.dtype)
        expected_eef = torch.bmm(
            object_pose,
            held.object_to_eef.to(device=env.device, dtype=object_pose.dtype),
        )
        position_ok = (
            torch.linalg.vector_norm(
                actual_eef[:, :3, 3] - expected_eef[:, :3, 3], dim=-1
            )
            <= position_tolerance
        )
        closed = _gripper_has_closed(
            env,
            gripper,
            tolerance=gripper_tolerance,
        )
        owned = torch.tensor(
            [item == arm for item in owners[uid]],
            dtype=torch.bool,
            device=env.device,
        )
        result |= owned & position_ok & closed
    return result


def _coordinated_held(
    env: Any,
    uid: str,
    state: Any,
    *,
    position_tolerance: float,
    gripper_tolerance: float,
) -> torch.Tensor:
    result = _constant(env, False)
    if state is None:
        return result
    held = state.get_coordinated_held_object(
        arm_control_part(env, "left_arm"),
        arm_control_part(env, "right_arm"),
    )
    if held is None:
        return result
    label = getattr(held.semantics, "label", None)
    if not label and getattr(held.semantics, "entity", None) is not None:
        label = getattr(held.semantics.entity, "uid", None)
    if label != uid:
        return result
    eef_values = _arm_values(env, "xpos")
    gripper_values = _arm_values(env, "gripper_state")
    if eef_values is None or gripper_values is None:
        return result

    object_pose = _pose(env, uid)
    result = _constant(env, True)
    for arm_index, transform_name in enumerate(
        ("left_object_to_eef", "right_object_to_eef")
    ):
        actual_eef = eef_values[arm_index]
        gripper = gripper_values[arm_index]
        if actual_eef is None or gripper is None:
            return _constant(env, False)
        transform = getattr(held, transform_name).to(
            device=env.device,
            dtype=object_pose.dtype,
        )
        expected_eef = torch.bmm(object_pose, transform)
        actual_eef = actual_eef.to(device=env.device, dtype=object_pose.dtype)
        position_ok = (
            torch.linalg.vector_norm(
                actual_eef[:, :3, 3] - expected_eef[:, :3, 3],
                dim=-1,
            )
            <= position_tolerance
        )
        closed = _gripper_has_closed(
            env,
            gripper,
            tolerance=gripper_tolerance,
        )
        result &= position_ok & closed
    return result


def evaluate_predicate(
    env: Any,
    spec: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    held_owners: Mapping[str, Sequence[str | None]] | None = None,
    held_states: Mapping[tuple[str, str], Any] | None = None,
    coordinated_state: Any | None = None,
) -> torch.Tensor:
    """Evaluate one typed predicate or a boolean predicate tree."""
    runtime = {
        "held_owners": held_owners,
        "held_states": held_states,
        "coordinated_state": coordinated_state,
    }
    if spec is None:
        return _constant(env, True)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, Mapping)):
        result = _constant(env, True)
        for term in spec:
            result &= evaluate_predicate(env, term, **runtime)
        return result
    if not isinstance(spec, Mapping):
        raise TypeError("Predicate must be a mapping or a sequence of mappings.")
    op = str(spec.get("op", "")).lower()
    if not op and "terms" in spec:
        op = "all"
    if op in {"all", "and"}:
        return evaluate_predicate(env, list(spec.get("terms", ())), **runtime)
    if op in {"any", "or"}:
        result = _constant(env, False)
        for term in spec.get("terms", ()):
            result |= evaluate_predicate(env, term, **runtime)
        return result
    if op == "not":
        return ~evaluate_predicate(env, spec.get("term"), **runtime)

    kind = str(spec.get("type", spec.get("kind", ""))).lower()
    if kind in {"semantic_goal", "line_member_placed", "stack_layer_supported"}:
        raise ValueError(
            f"Predicate {kind!r} is a compiler marker and requires the "
            "executor's grounded target."
        )
    if kind in {"object_held", "object_held_by_gripper"}:
        required_arm = spec.get("arm")
        if required_arm in {"left", "right"}:
            required_arm = f"{required_arm}_arm"
        return _object_held(
            env,
            _object(spec),
            owners=held_owners,
            states=held_states,
            position_tolerance=float(spec.get("position_tolerance", 0.06)),
            gripper_tolerance=float(spec.get("gripper_tolerance", 0.01)),
            required_arm=str(required_arm) if required_arm else None,
        )
    if kind in {"held_by_both_grippers", "object_held_by_both_grippers"}:
        return _coordinated_held(
            env,
            _object(spec),
            coordinated_state,
            position_tolerance=float(spec.get("position_tolerance", 0.06)),
            gripper_tolerance=float(spec.get("gripper_tolerance", 0.01)),
        )
    if kind in {"object_position_near", "position_near"}:
        position = _position(env, _object(spec))
        target = torch.as_tensor(
            spec.get("target_position", spec.get("target")),
            dtype=position.dtype,
            device=position.device,
        )
        if target.ndim == 1:
            target = target.unsqueeze(0)
        return torch.linalg.vector_norm(position - target, dim=-1) <= float(
            spec.get("tolerance", 0.05)
        )
    if kind in {"object_xy_near", "xy_near"}:
        position = _position(env, _object(spec))[:, :2]
        target = torch.as_tensor(
            spec.get("target_xy", spec.get("target")),
            dtype=position.dtype,
            device=position.device,
        ).reshape(-1, 2)
        return torch.linalg.vector_norm(position - target, dim=-1) <= float(
            spec.get("tolerance", 0.05)
        )
    if kind in {"object_in_container", "inside"}:
        position = _position(env, _object(spec))
        container = _position(
            env, str(spec.get("container", spec.get("reference_object")))
        )
        xy = torch.linalg.vector_norm(position[:, :2] - container[:, :2], dim=-1)
        z = position[:, 2] - container[:, 2]
        return (
            (xy <= float(spec.get("xy_radius", 0.20)))
            & (z >= float(spec.get("min_z_offset", -0.05)))
            & (z <= float(spec.get("max_z_offset", 0.35)))
        )
    if kind in {"object_on_object", "on"}:
        position = _position(env, _object(spec))
        support = _position(
            env,
            str(
                spec.get(
                    "support",
                    spec.get("reference_object", spec.get("reference")),
                )
            ),
        )
        xy = torch.linalg.vector_norm(position[:, :2] - support[:, :2], dim=-1)
        z = position[:, 2] - support[:, 2]
        return (
            (xy <= float(spec.get("xy_radius", 0.08)))
            & (z >= float(spec.get("min_z_offset", 0.02)))
            & (z <= float(spec.get("max_z_offset", 0.35)))
        )
    if kind == "object_not_fallen":
        axis = _pose(env, _object(spec))[:, :3, 2]
        cosine = axis[:, 2].clamp(-1.0, 1.0)
        return torch.arccos(cosine) <= float(spec.get("max_tilt", math.radians(45.0)))
    if kind == "object_upright":
        uid = _object(spec)
        local_axis = spec.get("local_axis", "long_axis")
        axis_index = _local_axis_index(
            env,
            uid,
            local_axis,
        )
        axis = _pose(env, uid)[:, :3, axis_index]
        cosine = axis[:, 2].clamp(-1.0, 1.0)
        if str(local_axis).lower() in {"long", "long_axis", "longest"}:
            cosine = cosine.abs()
        return torch.arccos(cosine) <= float(spec.get("max_tilt", math.radians(15.0)))
    if kind in {"object_axis_offset_near", "object_axis_near"}:
        object_position = _position(env, _object(spec))
        axis = _axis_index(spec.get("axis", "x"))
        reference_uid = spec.get(
            "reference_object",
            spec.get("reference", spec.get("support")),
        )
        if isinstance(reference_uid, str) and reference_uid:
            values = object_position[:, axis] - _position(env, reference_uid)[:, axis]
        else:
            values = object_position[:, axis]
        target = spec.get(
            "target_offset",
            spec.get("offset", spec.get("target", 0.0)),
        )
        target_value = torch.as_tensor(
            target,
            dtype=values.dtype,
            device=values.device,
        )
        return torch.abs(values - target_value) <= float(spec.get("tolerance", 0.03))
    if kind in {"objects_collinear", "collinear"}:
        positions = torch.stack(
            [_position(env, uid) for uid in _objects(spec)],
            dim=1,
        )
        axis = 0 if str(spec.get("axis", "x")) in {"x", "world_x"} else 1
        values = positions[:, :, 1 - axis]
        return values.max(dim=1).values - values.min(dim=1).values <= float(
            spec.get("tolerance", 0.03)
        )
    if kind in {"objects_ordered", "ordered"}:
        positions = torch.stack(
            [_position(env, uid) for uid in _objects(spec)],
            dim=1,
        )
        axis = 0 if str(spec.get("axis", "x")) in {"x", "world_x"} else 1
        differences = torch.diff(positions[:, :, axis], dim=1)
        tolerance = float(spec.get("tolerance", 0.02))
        if str(spec.get("direction", "ascending")) == "descending":
            return torch.all(differences <= tolerance, dim=1)
        return torch.all(differences >= -tolerance, dim=1)
    if kind == "object_lifted":
        position = _position(env, _object(spec))[:, 2]
        initial = spec.get("initial_height")
        if initial is None:
            initial_pose = getattr(env, "agent_initial_object_poses", {}).get(
                _object(spec)
            )
            if initial_pose is None:
                raise ValueError("object_lifted requires an initial object pose.")
            initial = initial_pose[:, 2, 3]
        initial = torch.as_tensor(initial, device=position.device)
        return position >= initial + float(spec.get("min_height", 0.08))
    if kind in {"both_arms_at_initial_qpos", "arms_home"}:
        current = env.robot.get_qpos()
        initial = getattr(env, "init_qpos", current)
        return torch.all(
            torch.abs(current - initial) <= float(spec.get("tolerance", 0.05)),
            dim=-1,
        )
    if kind in {"both_grippers_open", "grippers_open"}:
        if not hasattr(env, "get_current_gripper_state_agent"):
            return _constant(env, False)
        left, right = env.get_current_gripper_state_agent()
        expected = torch.as_tensor(
            env.open_state,
            dtype=torch.float32,
            device=env.device,
        )
        results = []
        for value in (left, right):
            value = torch.as_tensor(value, dtype=torch.float32, device=env.device)
            if value.ndim == 1:
                value = value.unsqueeze(0).repeat(int(env.num_envs), 1)
            results.append(
                torch.linalg.vector_norm(value - expected, dim=-1)
                <= float(spec.get("tolerance", 0.001))
            )
        return results[0] & results[1]
    if kind == "grippers_clear_of_object":
        eef_values = _arm_values(env, "xpos")
        if eef_values is None:
            return _constant(env, False)
        object_position = _position(env, _object(spec))
        clearance = float(spec.get("min_distance", spec.get("clearance", 0.08)))
        result = _constant(env, True)
        for eef in eef_values:
            if eef is None:
                return _constant(env, False)
            result &= (
                torch.linalg.vector_norm(
                    eef[:, :3, 3] - object_position,
                    dim=-1,
                )
                >= clearance
            )
        return result
    if kind == "pressed":
        checker = getattr(env, "is_object_pressed", None)
        if callable(checker):
            value = checker(_object(spec), spec.get("terminal_state", "activated"))
            result = torch.as_tensor(value, dtype=torch.bool, device=env.device)
            return (
                result.repeat(int(env.num_envs))
                if result.ndim == 0
                else result.reshape(-1)
            )
        states = getattr(env, "action_engine_semantic_states", {})
        value = states.get((_object(spec), "pressed"))
        if value is None:
            return _constant(env, False)
        result = torch.as_tensor(value, dtype=torch.bool, device=env.device)
        return (
            result.repeat(int(env.num_envs)) if result.ndim == 0 else result.reshape(-1)
        )
    if kind == "coordinated_placed":
        relation = str(spec.get("relation", "on"))
        reference = spec.get("support_object", spec.get("reference_object"))
        translated = {
            "type": (
                "object_in_container" if relation == "inside" else "object_on_object"
            ),
            "object": _object(spec),
            ("container" if relation == "inside" else "support"): reference,
        }
        return evaluate_predicate(env, translated, **runtime)
    raise ValueError(f"Unsupported execution predicate {kind!r}.")


def _axis_index(value: Any) -> int:
    axis = str(value).lower().replace("world_", "")
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported predicate axis {value!r}.")
    return {"x": 0, "y": 1, "z": 2}[axis]
