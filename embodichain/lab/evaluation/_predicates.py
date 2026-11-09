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

"""Trusted physical predicates for the evaluation registry and legacy adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TYPE_CHECKING, Any

from embodichain.compute._evaluation import (
    _corners_in_frame,
    _evaluation_pose_matrix,
    _inside_region,
    _measure_lift,
    _measure_relative_position,
    _measure_upright,
    _speeds_within,
    _within_target,
)
from embodichain.compute.geometry._distance import (
    _measure_near_target,
    _positions,
    _validate_near_target,
)
from ._graph import PredicateDefinition, PredicateMeasurement, PredicateRegistry

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


def _keys(params, required, optional=()):
    if not set(required) <= params.keys() or params.keys() - {*required, *optional}:
        raise ValueError("Predicate parameters do not match the declared contract.")


def _finite(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be finite.")
    return float(value)


def _near_params(params):
    _keys(params, ("tolerance",), ("axes", "inclusive"))
    inclusive = params.get("inclusive", True)
    axes = _validate_near_target(
        params["tolerance"], params.get("axes", (0, 1, 2)), inclusive
    )
    return {
        "tolerance": float(params["tolerance"]),
        "axes": axes,
        "inclusive": inclusive,
    }


def _near(arguments, params):
    position = arguments["position"]
    _positions(position)
    satisfied, valid, distance = _measure_near_target(
        position,
        arguments["target"],
        params["tolerance"],
        params["axes"],
        params["inclusive"],
        portable=True,
    )
    return PredicateMeasurement(satisfied, valid, {"distance": distance})


def _height_params(params):
    _keys(params, ("height",))
    return {"height": _finite(params["height"], "height")}


def _height(arguments, params):
    import torch

    position = arguments["position"]
    _positions(position)
    height = position[:, 2]
    return PredicateMeasurement(
        height < params["height"],
        torch.isfinite(position).all(dim=-1),
        {"height": height},
    )


def _range_params(params):
    _keys(params, (), ("lower", "upper"))
    lower = params.get("lower")
    upper = params.get("upper")
    if lower is None and upper is None:
        raise ValueError("Joint range requires at least one bound.")
    lower = _finite(lower, "lower") if lower is not None else None
    upper = _finite(upper, "upper") if upper is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Joint range lower bound must not exceed the upper bound.")
    return {"lower": lower, "upper": upper}


def _joint_range(arguments, params):
    import torch

    position = arguments["position"]
    if position.ndim != 1 or not position.is_floating_point():
        raise ValueError("Joint position must be one scalar per environment.")
    valid = torch.isfinite(position)
    passed = valid.clone()
    if params["lower"] is not None:
        passed &= position >= params["lower"]
    if params["upper"] is not None:
        passed &= position <= params["upper"]
    return PredicateMeasurement(passed, valid, {"position": position})


def _joint_near_params(params):
    _keys(params, ("target", "tolerance"))
    result = {name: _finite(value, name) for name, value in params.items()}
    if result["tolerance"] < 0:
        raise ValueError("Joint tolerance must be nonnegative.")
    return result


def _joint_near(arguments, params):
    import torch

    position = arguments["position"]
    if position.ndim != 1 or not position.is_floating_point():
        raise ValueError("Joint position must be one scalar per environment.")
    return PredicateMeasurement(
        _within_target(position, params["target"], params["tolerance"]),
        torch.isfinite(position),
        {"position": position},
    )


def _command_params(params):
    _keys(params, ("tolerance",))
    return {
        "tolerance": _joint_near_params(
            {"target": 0, "tolerance": params["tolerance"]}
        )["tolerance"]
    }


def _command_match(arguments, params):
    import torch

    measured, command = arguments["position"], arguments["command"]
    if measured.ndim != 2 or measured.shape != command.shape or not measured.shape[1]:
        raise ValueError(
            "Measured and commanded joint vectors must share a nonempty layout."
        )
    valid = torch.isfinite(measured).all(-1) & torch.isfinite(command).all(-1)
    return PredicateMeasurement(
        _within_target(measured, command, params["tolerance"]).all(-1),
        valid,
        {"maximum_error": (measured - command).abs().amax(-1)},
    )


def _preset_params(params):
    _keys(params, ("tolerance", "preset", "positions"))
    positions = params["positions"]
    preset = params["preset"]
    if not isinstance(preset, str) or not preset or preset != preset.strip():
        raise ValueError("Command preset requires an exact preset name.")
    if not isinstance(positions, (tuple, list)) or not positions:
        raise ValueError("Command preset requires a nonempty joint vector.")
    return {
        **_command_params({"tolerance": params["tolerance"]}),
        "preset": preset,
        "positions": tuple(
            _finite(position, "preset joint position") for position in positions
        ),
    }


def _preset_match(arguments, params):
    import torch

    position = arguments["position"]
    if position.ndim != 2 or position.shape[1] != len(params["positions"]):
        unavailable = torch.zeros(
            len(position), device=position.device, dtype=torch.bool
        )
        return PredicateMeasurement(unavailable, unavailable, {})
    command = position.new_tensor(params["positions"]).expand_as(position)
    return _command_match({"position": position, "command": command}, params)


def _corner_params(params):
    _keys(params, ("lower", "upper", "corners"), ("tolerance",))
    bounds = _region_params({key: params[key] for key in ("lower", "upper")})
    raw = params["corners"]
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(not isinstance(point, (list, tuple)) or len(point) != 3 for point in raw)
    ):
        raise ValueError("Containment corners must be a nonempty list of XYZ triples.")
    corners = tuple(tuple(_finite(x, "corner") for x in point) for point in raw)
    tolerance = _finite(params.get("tolerance", 0), "tolerance")
    if tolerance < 0:
        raise ValueError("Containment tolerance must be nonnegative.")
    return {**bounds, "corners": corners, "tolerance": tolerance}


def _corner_containment(arguments, params):
    import torch

    pose, frame = arguments["pose"], arguments["frame"]
    if pose.ndim != 2 or pose.shape[-1] != 7 or pose.shape != frame.shape:
        raise ValueError(
            "Containment requires object and reference poses in XYZ/XYZW format."
        )
    dtype = torch.promote_types(pose.dtype, frame.dtype)
    corners = _corners_in_frame(
        pose.to(dtype), frame.to(dtype), pose.new_tensor(params["corners"], dtype=dtype)
    )
    tolerance = params["tolerance"]
    inside = _inside_region(
        corners,
        corners.new_tensor(params["lower"]) - tolerance,
        corners.new_tensor(params["upper"]) + tolerance,
    )
    return PredicateMeasurement(
        inside.all(-1),
        torch.isfinite(corners).all(dim=(1, 2)),
        {"inside_count": inside.sum(-1)},
    )


def _lift(pose, baseline, minimum):
    passed, valid, measured = _measure_lift(pose, baseline, minimum)
    return PredicateMeasurement(passed, valid, {"lift": measured})


def _relative_position(pose, reference, offset, tolerances, *, parent_axes):
    passed, valid, measured = _measure_relative_position(
        pose, reference, offset, tolerances, parent_axes=parent_axes
    )
    return PredicateMeasurement(passed, valid, {"position_error": measured})


def _upright(pose, maximum_tilt):
    passed, valid, measured = _measure_upright(pose, maximum_tilt)
    return PredicateMeasurement(passed, valid, {"tilt_degrees": measured})


def _xyz(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be an XYZ triple.")
    return tuple(_finite(item, name) for item in value)


def _near_relative_params(params):
    _keys(params, ("tolerance", "displacement"), ("axes", "inclusive"))
    return {
        **_near_params(
            {key: value for key, value in params.items() if key != "displacement"}
        ),
        "displacement": _xyz(params["displacement"], "displacement"),
    }


def _near_relative(arguments, params):
    import torch

    position, reference = arguments["position"], arguments["reference"]
    target = reference + position.new_tensor(params["displacement"])
    dtype = torch.promote_types(position.dtype, target.dtype)
    return _near({"position": position.to(dtype), "target": target.to(dtype)}, params)


def _lift_params(params):
    _keys(params, ("minimum_lift",))
    minimum = _finite(params["minimum_lift"], "minimum_lift")
    if minimum <= 0:
        raise ValueError("Minimum lift must be positive.")
    return {"minimum_lift": minimum}


def _lift_predicate(arguments, params):
    return _lift(
        _evaluation_pose_matrix(arguments["pose"]),
        _evaluation_pose_matrix(arguments["baseline"]),
        params["minimum_lift"],
    )


def _relative_params(params):
    _keys(params, ("offset", "tolerances"), ("parent_axes",))
    tolerances = params["tolerances"]
    parent_axes = params.get("parent_axes", False)
    if (
        not isinstance(tolerances, (tuple, list))
        or len(tolerances) != 3
        or all(value is None for value in tolerances)
        or type(parent_axes) is not bool
    ):
        raise ValueError(
            "Relative position requires XYZ tolerances and a boolean parent_axes flag."
        )
    tolerances = tuple(
        None if value is None else _finite(value, "tolerance") for value in tolerances
    )
    if any(value is not None and value <= 0 for value in tolerances):
        raise ValueError("Relative position tolerances must be positive.")
    return {
        "offset": _xyz(params["offset"], "offset"),
        "tolerances": tolerances,
        "parent_axes": parent_axes,
    }


def _relative_predicate(arguments, params):
    import torch

    pose, reference = arguments["pose"], arguments["reference"]
    dtype = torch.promote_types(pose.dtype, reference.dtype)
    return _relative_position(
        _evaluation_pose_matrix(pose.to(dtype)),
        _evaluation_pose_matrix(reference.to(dtype)),
        pose.new_tensor(params["offset"], dtype=dtype),
        params["tolerances"],
        parent_axes=params["parent_axes"],
    )


def _upright_params(params):
    _keys(params, ("maximum_tilt_degrees",))
    maximum = _finite(params["maximum_tilt_degrees"], "maximum_tilt_degrees")
    if not 0 < maximum <= 180:
        raise ValueError("Upright tilt limit must be in (0, 180].")
    return {"maximum_tilt_degrees": maximum}


def _upright_predicate(arguments, params):
    return _upright(
        _evaluation_pose_matrix(arguments["pose"]), params["maximum_tilt_degrees"]
    )


def _stable_params(params):
    _keys(params, ("linear_speed", "angular_speed"))
    limits = {key: _finite(value, key) for key, value in params.items()}
    if min(limits.values()) < 0:
        raise ValueError("Stability speed bounds must be nonnegative.")
    return limits


def _stable(arguments, params):
    import torch

    linear, angular = arguments["linear"], arguments["angular"]
    _positions(linear)
    _positions(angular)
    linear_speed = torch.linalg.vector_norm(linear, dim=-1)
    angular_speed = torch.linalg.vector_norm(angular, dim=-1)
    valid = torch.isfinite(linear_speed) & torch.isfinite(angular_speed)
    return PredicateMeasurement(
        _speeds_within(
            linear_speed, angular_speed, params["linear_speed"], params["angular_speed"]
        ),
        valid,
        {"linear_speed": linear_speed, "angular_speed": angular_speed},
    )


def _region_params(params):
    _keys(params, ("lower", "upper"))
    if any(not isinstance(values, (tuple, list)) for values in params.values()):
        raise ValueError("Region bounds must specify ordered XYZ triples.")
    bounds = {
        name: tuple(_finite(x, name) for x in values) for name, values in params.items()
    }
    if any(len(values) != 3 for values in bounds.values()) or any(
        lo > hi for lo, hi in zip(bounds["lower"], bounds["upper"], strict=True)
    ):
        raise ValueError("Region bounds must specify ordered XYZ triples.")
    return bounds


def _region(arguments, params):
    import torch

    position = arguments["position"]
    _positions(position)
    return PredicateMeasurement(
        _inside_region(
            position,
            position.new_tensor(params["lower"]),
            position.new_tensor(params["upper"]),
        ),
        torch.isfinite(position).all(dim=-1),
        {"position": position},
    )


_NEAR = PredicateDefinition(
    2, ("position", "target"), _near_params, _near, ("position", "position")
)
_LINK_NEAR = PredicateDefinition(
    2, ("position", "target"), _near_params, _near, ("link_position", "position")
)
_HEIGHT = PredicateDefinition(1, ("position",), _height_params, _height, ("position",))
_JOINT_RANGE = PredicateDefinition(
    1, ("position",), _range_params, _joint_range, ("joint_position",)
)
_JOINT_NEAR = PredicateDefinition(
    1, ("position",), _joint_near_params, _joint_near, ("joint_position",)
)
_COMMAND = PredicateDefinition(
    1,
    ("position", "command"),
    _command_params,
    _command_match,
    ("joint_positions", "joint_command"),
    shared_layout=True,
)
_PRESET = PredicateDefinition(
    1, ("position",), _preset_params, _preset_match, ("joint_positions",)
)
_STABLE = PredicateDefinition(
    1,
    ("linear", "angular"),
    _stable_params,
    _stable,
    ("linear_velocity", "angular_velocity"),
)
_REGION = PredicateDefinition(1, ("position",), _region_params, _region, ("position",))
_CORNERS = PredicateDefinition(
    1,
    ("pose", "frame"),
    _corner_params,
    _corner_containment,
    ("object_frame", "link_frame"),
)


def builtin_predicates() -> PredicateRegistry:
    """Return an independent registry with the shared built-in predicates."""
    registry = PredicateRegistry()
    for kind, definition in (
        ("near_target", _NEAR),
        (
            "near_relative_target",
            PredicateDefinition(
                1,
                ("position", "reference"),
                _near_relative_params,
                _near_relative,
                ("position", "position"),
            ),
        ),
        (
            "lift",
            PredicateDefinition(
                1,
                ("pose", "baseline"),
                _lift_params,
                _lift_predicate,
                ("object_frame", "object_frame"),
            ),
        ),
        (
            "relative_position",
            PredicateDefinition(
                1,
                ("pose", "reference"),
                _relative_params,
                _relative_predicate,
                ("object_frame", "object_frame"),
            ),
        ),
        (
            "upright",
            PredicateDefinition(
                1, ("pose",), _upright_params, _upright_predicate, ("object_frame",)
            ),
        ),
        ("below_height", _HEIGHT),
        ("joint_range", _JOINT_RANGE),
        ("joint_near", _JOINT_NEAR),
        ("command_position_match", _COMMAND),
        ("command_preset_match", _PRESET),
        ("stable", _STABLE),
        ("inside_region", _REGION),
        ("corner_containment", _CORNERS),
        ("link_object_distance", _LINK_NEAR),
    ):
        registry.register(kind, definition)
    return registry
