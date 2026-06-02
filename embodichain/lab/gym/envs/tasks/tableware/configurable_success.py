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

from collections.abc import Mapping, Sequence
from typing import Any

import torch

__all__ = ["evaluate_configured_success"]


def evaluate_configured_success(
    env, spec: Mapping[str, Any] | None = None
) -> torch.Tensor:
    """Evaluate a config-defined task success predicate."""

    success_spec = spec or getattr(env, "agent_success", None)
    if success_spec is None:
        success_spec = _legacy_success_spec(env)
    if success_spec is None:
        return _constant(env, False)
    return _evaluate_spec(env, success_spec)


def _legacy_success_spec(env) -> Mapping[str, Any] | None:
    object_name = getattr(env, "agent_success_object", None)
    container_name = getattr(env, "agent_success_container", None)
    if object_name is not None and container_name is not None:
        return {
            "type": "object_in_container",
            "object": object_name,
            "container": container_name,
            "radius": getattr(env, "container_success_radius", 0.1),
            "min_z_offset": getattr(env, "container_success_min_z_offset", -0.03),
            "max_z_offset": getattr(env, "container_success_max_z_offset", 0.25),
        }

    target_position = getattr(env, "agent_success_position", None)
    if object_name is not None and target_position is not None:
        return {
            "type": "object_position_near",
            "object": object_name,
            "target_position": target_position,
            "tolerance": getattr(env, "agent_success_tolerance", 0.05),
        }

    target_xy = getattr(env, "target_xy", None)
    if target_xy is not None:
        return {
            "type": "object_xy_near",
            "object": getattr(env, "success_object", "cup"),
            "target_xy": target_xy,
            "tolerance": getattr(env, "success_tolerance", 0.04),
        }

    return None


def _evaluate_spec(
    env, spec: Mapping[str, Any] | Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, Mapping)):
        return _evaluate_all(env, spec)
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"Success spec must be a mapping or sequence, got {type(spec)}."
        )

    op = str(spec.get("op", "")).lower()
    if not op and "terms" in spec and "type" not in spec and "func" not in spec:
        op = "all"

    if op in {"all", "and"}:
        return _evaluate_all(env, spec.get("terms", []))
    if op in {"any", "or"}:
        return _evaluate_any(env, spec.get("terms", []))
    if op == "not":
        term = spec.get("term", None)
        if term is None:
            terms = spec.get("terms", [])
            if len(terms) != 1:
                raise ValueError("Success op='not' requires exactly one term.")
            term = terms[0]
        return ~_evaluate_spec(env, term)

    term_type = str(spec.get("type", spec.get("func", ""))).lower()
    if term_type in {"object_position_near", "object_near_position"}:
        return _object_position_near(env, spec)
    if term_type in {"object_xy_near", "object_near_xy"}:
        return _object_xy_near(env, spec)
    if term_type == "object_in_container":
        return _object_in_container(env, spec)
    if term_type in {"object_not_fallen", "not_fallen"}:
        return _object_not_fallen(env, spec)
    if term_type in {"object_axis_offset_near", "object_relative_axis_near"}:
        return _object_axis_offset_near(env, spec)
    if term_type in {"object_axis_near", "object_coordinate_near"}:
        return _object_axis_near(env, spec)
    if term_type in {"object_lifted", "object_height_above_initial"}:
        return _object_lifted(env, spec)

    raise ValueError(f"Unsupported success term type: {term_type!r}.")


def _evaluate_all(env, terms: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    success = _constant(env, True)
    for term in terms:
        success = success & _evaluate_spec(env, term)
    return success


def _evaluate_any(env, terms: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    success = _constant(env, False)
    for term in terms:
        success = success | _evaluate_spec(env, term)
    return success


def _constant(env, value: bool) -> torch.Tensor:
    return torch.full((env.num_envs,), value, dtype=torch.bool, device=env.device)


def _pose(env, uid: str) -> torch.Tensor:
    return env.sim.get_rigid_object(uid).get_local_pose(to_matrix=True)


def _position(env, uid: str) -> torch.Tensor:
    return _pose(env, uid)[:, :3, 3]


def _tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _object_name(spec: Mapping[str, Any]) -> str:
    return str(spec.get("object", spec.get("object_uid")))


def _object_position_near(env, spec: Mapping[str, Any]) -> torch.Tensor:
    position = _position(env, _object_name(spec))
    target = _tensor(
        spec.get("target_position", spec.get("position", spec.get("target"))),
        dtype=position.dtype,
        device=position.device,
    ).flatten()
    if target.numel() == 2:
        return _object_xy_near(env, {**spec, "target_xy": target})
    target = target.reshape(1, 3)
    tolerance = float(spec.get("tolerance", 0.05))
    return torch.linalg.norm(position - target, dim=-1) <= tolerance


def _object_xy_near(env, spec: Mapping[str, Any]) -> torch.Tensor:
    position = _position(env, _object_name(spec))
    target_xy = _tensor(
        spec.get("target_xy", spec.get("xy", spec.get("target"))),
        dtype=position.dtype,
        device=position.device,
    ).flatten()[:2]
    tolerance = float(spec.get("tolerance", spec.get("xy_tolerance", 0.05)))
    return (
        torch.linalg.norm(position[:, :2] - target_xy.reshape(1, 2), dim=-1)
        <= tolerance
    )


def _object_in_container(env, spec: Mapping[str, Any]) -> torch.Tensor:
    object_position = _position(env, _object_name(spec))
    container_position = _position(
        env, str(spec.get("container", spec.get("container_uid")))
    )
    radius = float(spec.get("xy_radius", spec.get("radius", 0.1)))
    min_z_offset = float(spec.get("min_z_offset", -0.03))
    max_z_offset = float(spec.get("max_z_offset", 0.25))

    xy_distance = torch.linalg.norm(
        object_position[:, :2] - container_position[:, :2], dim=-1
    )
    z_offset = object_position[:, 2] - container_position[:, 2]
    return (
        (xy_distance <= radius)
        & (z_offset >= min_z_offset)
        & (z_offset <= max_z_offset)
    )


def _object_not_fallen(env, spec: Mapping[str, Any]) -> torch.Tensor:
    pose = _pose(env, _object_name(spec))
    pose_z_axis = pose[:, :3, 2]
    world_z_axis = torch.tensor([0, 0, 1], dtype=pose.dtype, device=pose.device)
    dot_product = torch.sum(pose_z_axis * world_z_axis, dim=-1).clamp(-1.0, 1.0)
    angle = torch.arccos(dot_product)
    return angle < float(spec.get("max_tilt", torch.pi / 4))


def _object_axis_offset_near(env, spec: Mapping[str, Any]) -> torch.Tensor:
    object_position = _position(env, _object_name(spec))
    reference_position = _position(
        env, str(spec.get("reference", spec.get("reference_uid")))
    )
    axis = _axis_index(str(spec.get("axis", "y")))
    target_value = reference_position[:, axis] + float(spec.get("offset", 0.0))
    tolerance = float(spec.get("tolerance", 0.02))
    return torch.abs(object_position[:, axis] - target_value) <= tolerance


def _object_axis_near(env, spec: Mapping[str, Any]) -> torch.Tensor:
    object_position = _position(env, _object_name(spec))
    axis = _axis_index(str(spec.get("axis", "y")))
    target_value = float(spec.get("target", spec.get("value")))
    tolerance = float(spec.get("tolerance", 0.02))
    return torch.abs(object_position[:, axis] - target_value) <= tolerance


def _object_lifted(env, spec: Mapping[str, Any]) -> torch.Tensor:
    object_name = _object_name(spec)
    position = _position(env, object_name)
    initial_height = spec.get("initial_height")
    if initial_height is None:
        initial_height = getattr(env, "obj_info", {}).get(object_name, {}).get("height")
    if initial_height is None:
        initial_height = position[:, 2]
    initial_height = _tensor(
        initial_height, dtype=position.dtype, device=position.device
    )
    return position[:, 2] >= initial_height + float(spec.get("min_height", 0.1))


def _axis_index(axis: str) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    if axis not in axes:
        raise ValueError(f"Unsupported axis {axis!r}; expected one of x, y, z.")
    return axes[axis]
