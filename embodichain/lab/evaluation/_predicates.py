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

from embodichain.compute._evaluation import _within_target
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
    return _joint_near_params({"target": 0, "tolerance": params["tolerance"]})


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


def _corner_containment(arguments, params):
    import torch

    corners = arguments["corners"]
    if corners.ndim != 3 or not corners.shape[1] or corners.shape[2] != 3:
        raise ValueError(
            "Containment requires a nonempty batch of qualified XYZ corners."
        )
    inside = _inside_region(
        corners,
        corners.new_tensor(params["lower"]),
        corners.new_tensor(params["upper"]),
    )
    return PredicateMeasurement(
        inside.all(-1),
        torch.isfinite(corners).all(dim=(1, 2)),
        {"inside_count": inside.sum(-1)},
    )


def _lift(pose, baseline, minimum):
    import torch

    lift = pose[:, 2, 3] - baseline[:, 2, 3]
    valid = torch.isfinite(pose).all(dim=(1, 2)) & torch.isfinite(baseline).all(
        dim=(1, 2)
    )
    return PredicateMeasurement(lift >= minimum, valid, {"lift": lift})


def _relative_position(pose, reference, offset, tolerances, *, parent_axes):
    import torch

    displacement = pose[:, :3, 3] - reference[:, :3, 3]
    if parent_axes:
        displacement = torch.bmm(
            reference[:, :3, :3].transpose(1, 2), displacement.unsqueeze(-1)
        ).squeeze(-1)
    error = torch.abs(displacement - offset)
    valid = torch.isfinite(pose).all(dim=(1, 2)) & torch.isfinite(reference).all(
        dim=(1, 2)
    )
    passed = valid.clone()
    for axis, tolerance in enumerate(tolerances):
        if tolerance is not None:
            passed &= error[:, axis] < tolerance
    return PredicateMeasurement(passed, valid, {"position_error": error})


def _upright(pose, maximum_tilt):
    import torch

    tilt = torch.rad2deg(torch.acos(pose[:, 2, 2].clamp(-1.0, 1.0)))
    return PredicateMeasurement(
        tilt < maximum_tilt,
        torch.isfinite(pose).all(dim=(1, 2)),
        {"tilt_degrees": tilt},
    )


def _speeds_within(linear, angular, linear_limit, angular_limit):
    return (linear <= linear_limit) & (angular <= angular_limit)


def _advance_hold(previous, stable, *, increment=1, maximum=None):
    import torch

    count = previous + increment
    if maximum is not None:
        count = count.clamp(max=maximum)
    return torch.where(stable, count, 0)


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
    bounds = {
        name: tuple(_finite(x, name) for x in values) for name, values in params.items()
    }
    if any(len(values) != 3 for values in bounds.values()) or any(
        lo > hi for lo, hi in zip(bounds["lower"], bounds["upper"], strict=True)
    ):
        raise ValueError("Region bounds must specify ordered XYZ triples.")
    return bounds


def _inside_region(position, lower, upper):
    return ((position >= lower) & (position <= upper)).all(dim=-1)


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


_NEAR = PredicateDefinition(1, ("position", "target"), _near_params, _near)
_HEIGHT = PredicateDefinition(1, ("position",), _height_params, _height)
_JOINT_RANGE = PredicateDefinition(1, ("position",), _range_params, _joint_range)
_JOINT_NEAR = PredicateDefinition(1, ("position",), _joint_near_params, _joint_near)
_COMMAND = PredicateDefinition(
    1, ("position", "command"), _command_params, _command_match
)
_STABLE = PredicateDefinition(1, ("linear", "angular"), _stable_params, _stable)
_REGION = PredicateDefinition(1, ("position",), _region_params, _region)
_CORNERS = PredicateDefinition(1, ("corners",), _region_params, _corner_containment)


def builtin_predicates() -> PredicateRegistry:
    """Return an independent registry with the shared built-in predicates."""
    registry = PredicateRegistry()
    for kind, definition in (
        ("near_target", _NEAR),
        ("below_height", _HEIGHT),
        ("joint_range", _JOINT_RANGE),
        ("joint_near", _JOINT_NEAR),
        ("command_position_match", _COMMAND),
        ("stable", _STABLE),
        ("inside_region", _REGION),
        ("corner_containment", _CORNERS),
        ("link_object_distance", _NEAR),
    ):
        registry.register(kind, definition)
    return registry
