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
from typing import Any

import torch

from .geometry_axes import analyze_local_geometry_axes

from embodichain.gen_sim.action_engine.config import default_runtime_policy

from .frames import relation_axes
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
        "object_supported_by",
        "object_position_near",
        "object_relative_position",
        "object_upright",
        "object_xy_near",
        "objects_collinear",
        "objects_ordered",
        "pressed",
        "poured",
    }
)
_DEFAULT_PREDICATE_FALLBACKS = default_runtime_policy("dual_ur10").predicate_fallbacks


def _predicate_fallbacks(env: Any) -> Mapping[str, Any]:
    policy = getattr(env, "runtime_policy", None)
    value = getattr(policy, "predicate_fallbacks", None)
    return value if isinstance(value, Mapping) else _DEFAULT_PREDICATE_FALLBACKS


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


def _world_vertices(env: Any, uid: str, env_id: int) -> torch.Tensor:
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown rigid object {uid!r}.")
    value = entity.get_vertices(env_ids=[env_id], scale=True)
    if isinstance(value, (tuple, list)):
        value = value[0]
    vertices = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError(f"Rigid object {uid!r} has invalid mesh vertices.")
    pose = _pose(env, uid)[env_id]
    return vertices @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]


def _projected_center_of_mass(
    env: Any,
    uid: str,
    env_id: int,
    world_vertices: torch.Tensor,
) -> torch.Tensor:
    """Return the live COM projection, with a geometry-center fallback."""
    entity = env.sim.get_rigid_object(uid)
    body_data = None if entity is None else getattr(entity, "body_data", None)
    com_pose = None if body_data is None else getattr(body_data, "com_pose", None)
    if callable(com_pose):
        com_pose = com_pose()
    if com_pose is not None:
        local_com = torch.as_tensor(
            com_pose,
            dtype=torch.float32,
            device=env.device,
        )
        if local_com.ndim == 1:
            local_com = local_com.unsqueeze(0).repeat(int(env.num_envs), 1)
        if local_com.ndim == 2 and local_com.shape[0] == int(env.num_envs):
            pose = _pose(env, uid)[env_id]
            return (pose[:3, :3] @ local_com[env_id, :3] + pose[:3, 3])[:2]
    return (
        world_vertices[:, :2].min(dim=0).values
        + world_vertices[:, :2].max(dim=0).values
    ) * 0.5


def _object_supported_by(
    env: Any,
    spec: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> torch.Tensor:
    """Evaluate one-frame geometric support without advancing simulation."""
    object_uid = _object(spec)
    support_uid = str(
        spec.get(
            "support",
            spec.get("reference_object", spec.get("reference", "")),
        )
    )
    if not support_uid:
        raise ValueError("Support predicate requires a support object uid.")
    margin = float(spec.get("com_margin", defaults["support_com_margin"]))
    max_gap = float(spec.get("max_vertical_gap", defaults["support_max_vertical_gap"]))
    max_penetration = float(
        spec.get("max_penetration", defaults["support_max_penetration"])
    )
    min_overlap = float(
        spec.get("min_overlap_ratio", defaults["support_min_overlap_ratio"])
    )
    result = _constant(env, False)
    for env_id in range(int(env.num_envs)):
        moved = _world_vertices(env, object_uid, env_id)
        support = _world_vertices(env, support_uid, env_id)
        moved_lower = moved[:, :2].min(dim=0).values
        moved_upper = moved[:, :2].max(dim=0).values
        support_lower = support[:, :2].min(dim=0).values
        support_upper = support[:, :2].max(dim=0).values
        overlap_extent = torch.clamp(
            torch.minimum(moved_upper, support_upper)
            - torch.maximum(moved_lower, support_lower),
            min=0.0,
        )
        moved_extent = torch.clamp(moved_upper - moved_lower, min=1e-6)
        overlap_ratio = torch.prod(overlap_extent) / torch.prod(moved_extent)
        projected_center = _projected_center_of_mass(
            env,
            object_uid,
            env_id,
            moved,
        )
        center_supported = torch.all(
            projected_center >= support_lower + margin
        ) & torch.all(projected_center <= support_upper - margin)
        local_mask = torch.all(
            (support[:, :2] >= moved_lower - margin)
            & (support[:, :2] <= moved_upper + margin),
            dim=1,
        )
        if bool(local_mask.any()):
            local_support_height = support[local_mask, 2].max()
        else:
            # Sparse meshes may have no vertex exactly under a small payload.
            # Nearest vertices are a local fallback; using the mesh-wide peak
            # would confuse a remote protrusion with the candidate support pose.
            distances = torch.linalg.vector_norm(
                support[:, :2] - projected_center,
                dim=1,
            )
            count = min(8, int(support.shape[0]))
            local_support_height = support[
                torch.topk(distances, count, largest=False).indices, 2
            ].max()
        vertical_gap = moved[:, 2].min() - local_support_height
        result[env_id] = bool(
            center_supported
            and overlap_ratio >= min_overlap
            and vertical_gap >= -max_penetration
            and vertical_gap <= max_gap
        )
    return result


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
    geometry_axis = {
        "long": "long",
        "long_axis": "long",
        "longest": "long",
        "short": "short",
        "short_axis": "short",
        "shortest": "short",
    }.get(name)
    if geometry_axis is None:
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
    axes = analyze_local_geometry_axes(vertices)
    return axes.long_axis_index if geometry_axis == "long" else axes.short_axis_index


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
    held_relations = tuple(
        state.get_held_object(arm_control_part(env, arm))
        for arm in ("left_arm", "right_arm")
    )
    if any(held is None for held in held_relations):
        return result
    for held in held_relations:
        assert held is not None
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
    for arm_index, held in enumerate(held_relations):
        assert held is not None
        if held.env_mask is not None:
            result &= held.env_mask.to(device=env.device)
        actual_eef = eef_values[arm_index]
        gripper = gripper_values[arm_index]
        if actual_eef is None or gripper is None:
            return _constant(env, False)
        transform = held.object_to_eef.to(
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
    defaults = _predicate_fallbacks(env)
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
            position_tolerance=float(
                spec.get("position_tolerance", defaults["held_position_tolerance"])
            ),
            gripper_tolerance=float(
                spec.get("gripper_tolerance", defaults["held_gripper_tolerance"])
            ),
            required_arm=str(required_arm) if required_arm else None,
        )
    if kind == "handover_complete":
        required_arm = spec.get("arm", "right_arm")
        return _object_held(
            env,
            _object(spec),
            owners=held_owners,
            states=held_states,
            position_tolerance=float(
                spec.get("position_tolerance", defaults["held_position_tolerance"])
            ),
            gripper_tolerance=float(
                spec.get("gripper_tolerance", defaults["held_gripper_tolerance"])
            ),
            required_arm=str(required_arm),
        )
    if kind in {"held_by_both_grippers", "object_held_by_both_grippers"}:
        return _coordinated_held(
            env,
            _object(spec),
            coordinated_state,
            position_tolerance=float(
                spec.get("position_tolerance", defaults["held_position_tolerance"])
            ),
            gripper_tolerance=float(
                spec.get("gripper_tolerance", defaults["held_gripper_tolerance"])
            ),
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
            spec.get("tolerance", defaults["position_tolerance"])
        )
    if kind in {"object_xy_near", "xy_near"}:
        position = _position(env, _object(spec))[:, :2]
        target = torch.as_tensor(
            spec.get("target_xy", spec.get("target")),
            dtype=position.dtype,
            device=position.device,
        ).reshape(-1, 2)
        return torch.linalg.vector_norm(position - target, dim=-1) <= float(
            spec.get("tolerance", defaults["xy_tolerance"])
        )
    if kind in {"object_relative_position", "relative_position"}:
        reference_uid = spec.get("reference_object", spec.get("reference"))
        if not isinstance(reference_uid, str) or not reference_uid:
            raise ValueError("Relative-position predicate requires a reference object.")
        relation = str(spec.get("relation", ""))
        axes = relation_axes(
            env,
            relation,
            frame=str(spec.get("relation_frame", "world")),
        )
        if not axes:
            raise ValueError(f"Unsupported directional relation {relation!r}.")
        delta = (
            _position(env, _object(spec))[:, :2] - _position(env, reference_uid)[:, :2]
        )
        minimum_distance = float(spec.get("minimum_distance", 0.0))
        result = _constant(env, True)
        for axis in axes:
            projection = torch.sum(
                delta * axis.to(dtype=delta.dtype, device=delta.device), dim=1
            )
            result &= projection >= minimum_distance
        return result
    if kind in {"object_in_container", "inside"}:
        position = _position(env, _object(spec))
        container = _position(
            env, str(spec.get("container", spec.get("reference_object")))
        )
        xy = torch.linalg.vector_norm(position[:, :2] - container[:, :2], dim=-1)
        z = position[:, 2] - container[:, 2]
        return (
            (xy <= float(spec.get("xy_radius", defaults["container_xy_radius"])))
            & (z >= float(spec.get("min_z_offset", defaults["container_min_z_offset"])))
            & (z <= float(spec.get("max_z_offset", defaults["container_max_z_offset"])))
        )
    if kind in {"object_supported_by", "object_on_object", "on"}:
        return _object_supported_by(env, spec, defaults)
    if kind == "object_not_fallen":
        axis = _pose(env, _object(spec))[:, :3, 2]
        cosine = axis[:, 2].clamp(-1.0, 1.0)
        return torch.arccos(cosine) <= float(
            spec.get("max_tilt", defaults["not_fallen_max_tilt"])
        )
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
        directed = spec.get(
            "directed",
            str(local_axis).lower()
            not in {
                "long",
                "long_axis",
                "longest",
                "short",
                "short_axis",
                "shortest",
            },
        )
        if not isinstance(directed, bool):
            raise ValueError("object_upright directed must be a boolean.")
        if not directed:
            cosine = cosine.abs()
        return torch.arccos(cosine) <= float(
            spec.get("max_tilt", defaults["upright_max_tilt"])
        )
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
        return torch.abs(values - target_value) <= float(
            spec.get("tolerance", defaults["axis_tolerance"])
        )
    if kind in {"objects_collinear", "collinear"}:
        positions = torch.stack(
            [_position(env, uid) for uid in _objects(spec)],
            dim=1,
        )
        axis = 0 if str(spec.get("axis", "x")) in {"x", "world_x"} else 1
        values = positions[:, :, 1 - axis]
        return values.max(dim=1).values - values.min(dim=1).values <= float(
            spec.get("tolerance", defaults["collinearity_tolerance"])
        )
    if kind in {"objects_ordered", "ordered"}:
        positions = torch.stack(
            [_position(env, uid) for uid in _objects(spec)],
            dim=1,
        )
        axis = 0 if str(spec.get("axis", "x")) in {"x", "world_x"} else 1
        differences = torch.diff(positions[:, :, axis], dim=1)
        tolerance = float(spec.get("tolerance", defaults["ordering_tolerance"]))
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
        return position >= initial + float(
            spec.get("min_height", defaults["minimum_lift_height"])
        )
    if kind in {"both_arms_at_initial_qpos", "arms_home"}:
        current = env.robot.get_qpos()
        initial = getattr(env, "init_qpos", current)
        return torch.all(
            torch.abs(current - initial)
            <= float(spec.get("tolerance", defaults["arm_initial_qpos_tolerance"])),
            dim=-1,
        )
    if kind in {"both_grippers_open", "grippers_open"}:
        if not hasattr(env, "get_current_gripper_state_agent"):
            return _constant(env, False)
        left, right = env.get_current_gripper_state_agent()
        results = []
        for side, value in zip(("left", "right"), (left, right)):
            value = torch.as_tensor(value, dtype=torch.float32, device=env.device)
            if value.ndim == 1:
                value = value.unsqueeze(0).repeat(int(env.num_envs), 1)
            expected = getattr(env, f"{side}_arm_init_gripper_state", env.open_state)
            expected = torch.as_tensor(
                expected,
                dtype=torch.float32,
                device=env.device,
            )
            if expected.ndim == 1:
                expected = expected.unsqueeze(0).repeat(int(env.num_envs), 1)
            results.append(
                torch.linalg.vector_norm(value - expected, dim=-1)
                <= float(spec.get("tolerance", defaults["gripper_state_tolerance"]))
            )
        return results[0] & results[1]
    if kind == "grippers_clear_of_object":
        eef_values = _arm_values(env, "xpos")
        if eef_values is None:
            return _constant(env, False)
        object_position = _position(env, _object(spec))
        clearance = float(
            spec.get(
                "min_distance",
                spec.get("clearance", defaults["gripper_clear_min_distance"]),
            )
        )
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
        return _constant(env, False)
    if kind == "poured":
        if spec.get("verification") == "action_completion":
            # Reaching semantic-step verification means every required E3 edge
            # already completed without a fatal planning or execution failure.
            return _constant(env, True)

        raw_contents = spec.get("contents", ())
        if not isinstance(raw_contents, Sequence) or isinstance(
            raw_contents, (str, bytes, bytearray)
        ):
            raise ValueError("poured contents must be a list of observable objects.")
        contents = [
            item.get("object") if isinstance(item, Mapping) else item
            for item in raw_contents
        ]
        if not contents or any(not isinstance(uid, str) or not uid for uid in contents):
            raise ValueError(
                "poured requires at least one independently observable content object."
            )
        if len(contents) != len(set(contents)):
            raise ValueError("poured content objects must be unique.")
        target = spec.get("reference_object", spec.get("container"))
        if not isinstance(target, str) or not target:
            raise ValueError("poured requires a target reference_object.")
        transferred = _constant(env, True)
        for uid in contents:
            transferred &= evaluate_predicate(
                env,
                {
                    "type": "object_in_container",
                    "object": uid,
                    "container": target,
                },
                **runtime,
            )
        return transferred
    if kind == "articulation_joint_near":
        uid = _object(spec)
        articulation = getattr(env.sim, "get_articulation", lambda _uid: None)(uid)
        if articulation is None:
            raise ValueError(f"Unknown articulation {uid!r}.")
        backend_entities = getattr(
            articulation,
            "_entities",
            getattr(articulation, "entities", ()),
        )
        if not backend_entities:
            raise ValueError("Articulation backend does not expose joint metadata.")
        backend = backend_entities[0]
        expected_joint_type = "revolute" if "target_setting" in spec else "prismatic"
        candidates = []
        for joint_id in getattr(
            articulation,
            "active_joint_ids",
            range(len(articulation.joint_names)),
        ):
            joint_name = str(articulation.joint_names[int(joint_id)])
            info = backend.get_joint_info(joint_name)
            joint_type = (
                str(getattr(getattr(info, "joint_type", None), "name", info.joint_type))
                .rsplit(".", maxsplit=1)[-1]
                .lower()
            )
            if joint_type == expected_joint_type:
                candidates.append((int(joint_id), joint_name))
        requested = spec.get("joint_name")
        if requested is not None:
            candidates = [item for item in candidates if item[1] == str(requested)]
        if len(candidates) != 1:
            raise ValueError(
                "articulation_joint_near requires exactly one matching "
                f"{expected_joint_type} joint."
            )
        joint_id, _ = candidates[0]
        limits = articulation.get_qpos_limits(joint_ids=[joint_id])[:, 0]
        qpos = articulation.get_qpos()[:, joint_id]
        target_state = spec.get("target_state")
        if target_state == "open":
            target = limits[:, 1]
        elif target_state == "closed":
            target = limits[:, 0]
        elif "target_qpos" in spec:
            target = torch.as_tensor(
                spec["target_qpos"], dtype=torch.float32, device=env.device
            ).expand_as(qpos)
        elif "target_setting" in spec:
            raw_values = spec.get("setting_values", ())
            if (
                not isinstance(raw_values, Sequence)
                or isinstance(raw_values, (str, bytes, bytearray))
                or not raw_values
            ):
                raise ValueError(
                    "articulation_joint_near target_setting requires setting_values."
                )
            values = torch.as_tensor(raw_values, dtype=torch.float32, device=env.device)
            setting = int(spec["target_setting"])
            if setting < 0 or setting >= values.numel():
                raise ValueError(
                    "articulation_joint_near target_setting is outside setting_values."
                )
            target = values[setting].expand_as(qpos)
        else:
            raise ValueError(
                "articulation_joint_near requires open/closed target_state or "
                "target_setting with setting_values."
            )
        tolerance = float(spec.get("tolerance", defaults["axis_tolerance"]))
        return torch.isfinite(qpos) & (torch.abs(qpos - target) <= tolerance)
    if kind == "coordinated_placed":
        relation = str(spec.get("relation", "on"))
        reference = spec.get("support_object", spec.get("reference_object"))
        translated = {
            "type": (
                "object_in_container" if relation == "inside" else "object_supported_by"
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
