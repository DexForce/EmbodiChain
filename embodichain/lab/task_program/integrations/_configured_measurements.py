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
"""Checked declarations for episode-scoped physical measurement channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from embodichain.lab.evaluation._episode_reduction import _OPERATIONS, _UNITS

__all__: list[str] = []

_COMMON_FIELDS = {"frame", "finite", "anchor_error_m", "dropped_contacts"}
_WINDING_FIELDS = _COMMON_FIELDS | {
    "free_end_m",
    "free_end_z_m",
    "rope_min_z_m",
    "rope_max_z_m",
    "arm_joint_limit_violation_rad",
    "finger_joint_limit_violation_m",
    "grasp_confirmed",
    "tail_tcp_distance_m",
    "material_slip_m",
    "bilateral_rope_contact",
    "winding_deg",
    "contact_arc_deg",
    "post_penetration_m",
    "robot_post_penetration_m",
    "anchor_direction_error_deg",
    "anchor_exit_bend_max_deg",
    "joint_gap_max_m",
}
_CONNECTOR_FIELDS = _COMMON_FIELDS | {
    "tip_z_m",
    "tool_axis_z_abs",
    "finger_latch_contact",
    "latch_socket_contact",
    "depth_m",
    "lateral_m",
    "orientation_error_deg",
    "home_error_rad",
    "latch_angle_deg",
    "latch_angle_abs_deg",
    "latch_limit_violation_rad",
    "return_error_m",
    "return_orientation_error_deg",
    "finger_gap_m",
    "attachment_error_m",
    "penetration_m",
}
_PRESS_FIELDS = {"depth_m", "latch_angle_deg"}


@dataclass(frozen=True, slots=True)
class _EpisodeMeasurementPlan:
    kind: str
    cable_uid: str
    contact_sensor: str
    tool_link: str
    arm: str
    steps: int
    samples_per_control: int
    anchor_at_reset: bool
    parameters: Mapping[str, Any]
    reductions: Mapping[str, Mapping[str, Any]]
    identity: str

    def channels(self) -> dict[str, str]:
        return {
            f"measurements/{name}": f"measurements/{self.identity}/{name}"
            for name in self.reductions
        }


def _decode_measurement_plan(
    value: object, scene, *, path: str = "evaluation.measurements"
) -> _EpisodeMeasurementPlan:
    from .configured import (
        _finite_tuple,
        _identifier,
        _integer,
        _mapping,
        _real,
        _sequence,
    )

    cfg = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "kind",
                "cable",
                "contact_sensor",
                "tool_link",
                "arm",
                "steps",
                "samples_per_control",
                "parameters",
                "reductions",
            }
        ),
        optional=frozenset({"anchor_at_reset"}),
    )
    kind = cfg["kind"]
    if kind not in ("cable_winding", "connector"):
        raise ValueError(f"{path}.kind must select cable_winding or connector.")
    cables = {item.entity_id: item.simulation_uid for item in scene.cables}
    cable = _identifier(cfg["cable"], path=f"{path}.cable")
    if cable not in cables:
        raise ValueError(f"{path}.cable must bind a declared scene cable.")
    steps = _integer(cfg["steps"], path=f"{path}.steps", minimum=1)
    samples = _integer(
        cfg["samples_per_control"], path=f"{path}.samples_per_control", minimum=1
    )
    anchor_at_reset = cfg.get("anchor_at_reset", False)
    if type(anchor_at_reset) is not bool:
        raise ValueError(f"{path}.anchor_at_reset must be boolean.")
    fields = _WINDING_FIELDS if kind == "cable_winding" else _CONNECTOR_FIELDS
    params = (
        _winding_parameters(cfg["parameters"], scene, f"{path}.parameters")
        if kind == "cable_winding"
        else _connector_parameters(cfg["parameters"], scene, f"{path}.parameters")
    )
    if not isinstance(cfg["reductions"], Mapping) or not cfg["reductions"]:
        raise ValueError(f"{path}.reductions must contain named scalar measurements.")
    reductions = {}
    for name, item in cfg["reductions"].items():
        key = f"{path}.reductions.{name}"
        _identifier(name, path=key)
        if "/" in name:
            raise ValueError(f"{key} cannot contain a path separator.")
        declaration = _mapping(
            item,
            path=key,
            required=frozenset({"operation", "unit"}),
            optional=frozenset({"field", "window", "absolute", "samples", "where"}),
        )
        operation = _identifier(declaration["operation"], path=f"{key}.operation")
        unit = _identifier(declaration["unit"], path=f"{key}.unit")
        if operation not in _OPERATIONS or unit not in _UNITS:
            raise ValueError(f"{key} needs a supported scalar reduction and unit.")
        normalized = {"operation": operation, "unit": unit}
        if "window" in declaration:
            window = _sequence(declaration["window"], path=f"{key}.window")
            if len(window) != 2:
                raise ValueError(f"{key}.window needs two inclusive control frames.")
            first, last = (
                _integer(frame, path=f"{key}.window", minimum=0, maximum=steps)
                for frame in window
            )
            if last < first:
                raise ValueError(f"{key}.window ends before it begins.")
            normalized["window"] = (first, last)
        if operation == "count":
            if declaration.keys() - {"operation", "unit", "window"}:
                raise ValueError(
                    f"{key} counts frames and takes only a unit and window."
                )
            reductions[name] = normalized
            continue
        child = declaration.get("samples")
        if child is not None and (kind != "connector" or child != "press_samples"):
            raise ValueError(
                f"{key}.samples must name a declared substep sample stream."
            )
        field = _identifier(declaration.get("field"), path=f"{key}.field")
        field_set = _PRESS_FIELDS if child else fields
        if field not in field_set or (field == "free_end_m") != (
            operation == "displacement"
        ):
            raise ValueError(
                f"{key} uses an incompatible measurement field and reduction."
            )
        if type(declaration.get("absolute", False)) is not bool:
            raise ValueError(f"{key}.absolute must be boolean.")
        normalized.update(field=field, absolute=declaration.get("absolute", False))
        if child:
            normalized["samples"] = child
        if "where" in declaration:
            where = declaration["where"]
            if not isinstance(where, Mapping) or not set(where) <= field_set - {
                "free_end_m"
            }:
                raise ValueError(
                    f"{key}.where must bind ranges on available scalar fields."
                )
            limits = {
                name: _finite_tuple(
                    interval, path=f"{key}.where.{name}", expected_length=2
                )
                for name, interval in where.items()
            }
            if any(lo > hi for lo, hi in limits.values()):
                raise ValueError(f"{key}.where ranges must be ordered.")
            normalized["where"] = limits
        reductions[name] = normalized
    identity = sha256(
        json.dumps(
            {"spec": cfg, "cable_uid": cables[cable], "parameters": params},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return _EpisodeMeasurementPlan(
        kind,
        cables[cable],
        _identifier(cfg["contact_sensor"], path=f"{path}.contact_sensor"),
        _identifier(cfg["tool_link"], path=f"{path}.tool_link"),
        _identifier(cfg["arm"], path=f"{path}.arm"),
        steps,
        samples,
        anchor_at_reset,
        _freeze(params),
        _freeze(reductions),
        identity,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _winding_parameters(value: object, scene, path: str) -> dict[str, Any]:
    from .configured import (
        _finite_tuple,
        _identifier,
        _integer,
        _mapping,
        _real,
        _sequence,
    )

    scalars = {
        "floor_height_m",
        "post_radius_m",
        "post_height_m",
        "cable_radius_m",
        "contact_band_m",
        "grasp_inset_m",
        "contact_distance_m",
        "contact_point_gap_m",
        "pad_gap_m",
    }
    props = _mapping(
        value,
        path=path,
        required=frozenset(
            scalars
            | {
                "post",
                "post_center_xy_m",
                "fingers",
                "finger_pad_center_m",
                "finger_pad_size_m",
                "grasp_start",
                "tail_segments",
            }
        ),
    )
    post = _identifier(props["post"], path=f"{path}.post")
    objects = {item.entity_id: item.simulation_uid for item in scene.rigid_objects}
    if post not in objects:
        raise ValueError(f"{path}.post must bind a declared rigid object.")
    fingers = tuple(
        _identifier(name, path=f"{path}.fingers")
        for name in _sequence(props["fingers"], path=f"{path}.fingers")
    )
    if len(fingers) != 2 or len(set(fingers)) != 2:
        raise ValueError(f"{path}.fingers requires two distinct robot links.")
    result = {
        name: _real(
            props[name],
            path=f"{path}.{name}",
            minimum=0 if name != "floor_height_m" else -float("inf"),
            strict_minimum=name
            in {"post_radius_m", "post_height_m", "cable_radius_m", "grasp_inset_m"},
        )
        for name in scalars
    }
    for name, count in (
        ("post_center_xy_m", 2),
        ("finger_pad_center_m", 3),
        ("finger_pad_size_m", 3),
    ):
        result[name] = _finite_tuple(
            props[name], path=f"{path}.{name}", expected_length=count
        )
    if min(result["finger_pad_size_m"]) <= 0:
        raise ValueError(f"{path}.finger_pad_size_m must be positive.")
    result.update(
        post_uid=objects[post],
        fingers=fingers,
        grasp_start=_integer(
            props["grasp_start"], path=f"{path}.grasp_start", minimum=0
        ),
        tail_segments=_integer(
            props["tail_segments"], path=f"{path}.tail_segments", minimum=1
        ),
    )
    return result


def _connector_parameters(value: object, scene, path: str) -> dict[str, Any]:
    from .configured import _finite_tuple, _identifier, _integer, _mapping, _sequence
    import numpy as np

    identifiers = {
        "connectable",
        "receiver_uid",
        "connectable_link",
        "latch_link",
        "latch_joint",
        "receiver_link",
    }
    vectors = {"socket_mouth_m", "attachment_point_m", "return_tip_m"}
    matrices = {"insertion_rotation", "return_rotation"}
    props = _mapping(
        value,
        path=path,
        required=frozenset(
            identifiers
            | vectors
            | matrices
            | {"home_qpos", "vertical_tool_axis", "fingers", "latch_range_rad"}
        ),
    )
    entities = {item.entity_id: item.simulation_uid for item in scene.articulations}
    result = {
        name: _identifier(props[name], path=f"{path}.{name}") for name in identifiers
    }
    if result["connectable"] not in entities:
        raise ValueError(f"{path}.connectable must bind a declared articulation.")
    result["connectable"] = entities[result["connectable"]]
    for name in vectors:
        result[name] = _finite_tuple(
            props[name], path=f"{path}.{name}", expected_length=3
        )
    for name in matrices:
        rotation = tuple(
            _finite_tuple(row, path=f"{path}.{name}", expected_length=3)
            for row in _sequence(props[name], path=f"{path}.{name}")
        )
        if (
            len(rotation) != 3
            or not np.allclose(np.asarray(rotation).T @ rotation, np.eye(3))
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError(f"{path}.{name} must be a proper rotation.")
        result[name] = rotation
    lo, hi = _finite_tuple(
        props["latch_range_rad"], path=f"{path}.latch_range_rad", expected_length=2
    )
    if lo >= hi:
        raise ValueError(f"{path}.latch_range_rad must be strictly ordered.")
    result["latch_range_rad"] = (lo, hi)
    result["home_qpos"] = _finite_tuple(props["home_qpos"], path=f"{path}.home_qpos")
    result["vertical_tool_axis"] = _integer(
        props["vertical_tool_axis"],
        path=f"{path}.vertical_tool_axis",
        minimum=0,
        maximum=2,
    )
    fingers = tuple(
        _identifier(name, path=f"{path}.fingers")
        for name in _sequence(props["fingers"], path=f"{path}.fingers")
    )
    if len(fingers) != 2 or len(set(fingers)) != 2:
        raise ValueError(f"{path}.fingers requires two distinct robot links.")
    result["fingers"] = fingers
    return result
