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

"""Closed decoding for explicitly declared control operation services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, MISSING
import math
import re
import numpy as np

from .simulation.control_assembly import ControlOperationDeclaration
from .simulation.control_lowering import (
    PressCallDeclaration,
    TwistCallDeclaration,
    PressPrepareFactory,
    PressFactory,
    PressApproachFactory,
    TwistPrepareFactory,
    TwistFactory,
    specialize_control_factory,
)
from embodichain.lab.sim.scene.press_inputs import PressBinding
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding

__all__: list[str] = []


def _closed(value, *, required, optional=(), path):
    if (
        not isinstance(value, Mapping)
        or not set(required) <= set(value)
        or set(value) - set(required) - set(optional)
    ):
        raise ValueError(f"{path} has missing or unexpected fields.")
    return dict(value)


def _binding(value, *, press, path):
    cls = PressBinding if press else RevoluteControlBinding
    required = {field.name for field in fields(cls) if field.default is MISSING} - {
        "source_path"
    }
    optional = {field.name for field in fields(cls)} - required - {"source_path"}
    data = _closed(value, required=required, optional=optional, path=path)
    vectors = (
        {"axis": 3, "limits": 2, "press_position": 3}
        if press
        else {"axis": 3, "limits": 2, "origin": 3, "outer_point": 3}
    )
    strings = {
        "object_id",
        "joint",
        "link",
        "parent",
        "source_sha256",
        "collision_mesh",
        "calibration_sha256",
        "mass_lineage_sha256",
    }
    for name, item in tuple(data.items()):
        if name in vectors:
            if (
                type(item) not in (list, tuple)
                or len(item) != vectors[name]
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in item
                )
            ):
                raise ValueError(f"{path}.{name} requires finite coordinates.")
            data[name] = tuple(item)
        elif name in strings:
            if (
                type(item) is not str
                or item != item.strip()
                or (not item and name != "mass_lineage_sha256")
            ):
                raise ValueError(f"{path}.{name} requires an exact identity.")
        elif name == "target_setting":
            if type(item) is not int:
                raise TypeError(f"{path}.{name} requires an integer setting.")
        elif type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(f"{path}.{name} requires a finite scalar.")
    if press:
        data["source_path"] = ""
        if (
            data["scale"] <= 0
            or data["limits"][0] >= data["limits"][1]
            or abs(data["released_position"] - data["pressed_position"]) <= 1e-5
            or {data["released_position"], data["pressed_position"]}
            != set(data["limits"])
            or not np.isclose(np.linalg.norm(data["axis"]), 1.0)
            or not re.fullmatch("[a-f0-9]{64}", data["source_sha256"])
            or not data["collision_mesh"].startswith("/")
            or data["link"] == data["parent"]
        ):
            raise ValueError(
                f"{path} requires positive scale and distinct ordered travel."
            )
    return cls(**data)


def _decode_control_operation(
    value: object, *, path: str
) -> ControlOperationDeclaration:
    common = {
        "kind",
        "operation_id",
        "entity_id",
        "binding",
        "sensor_uid",
        "grasp_target_id",
        "finger_link_names",
        "arm_label",
    }
    optional = {
        "variant",
        "extra_press_distance",
        "moving_mass",
        "table_uid",
        "chunk_count",
        "success_criterion",
        "grasp_depth",
        "chunk_angle",
        "settle_steps",
        "fixed_roll",
    }
    config = _closed(value, required=common, optional=optional, path=path)
    kind = config["kind"]
    if kind not in ("press", "twist"):
        raise ValueError(f"{path}.kind must be press or twist.")
    press = kind == "press"
    extra = (
        {"moving_mass", "extra_press_distance"}
        if press
        else {
            "table_uid",
            "chunk_count",
            "success_criterion",
            "grasp_depth",
            "chunk_angle",
            "settle_steps",
            "fixed_roll",
        }
    )
    config = _closed(
        value,
        required=common | ({"moving_mass"} if press else {"table_uid", "chunk_count"}),
        optional=extra | {"variant"},
        path=path,
    )
    cls = PressCallDeclaration if press else TwistCallDeclaration
    call_fields = {field.name for field in fields(cls)}
    call = cls(**{key: item for key, item in config.items() if key in call_fields})
    names = config["finger_link_names"]
    if type(names) not in (list, tuple):
        raise TypeError(f"{path}.finger_link_names requires a sequence.")
    return ControlOperationDeclaration(
        call=call,
        binding=_binding(config["binding"], press=press, path=f"{path}.binding"),
        sensor_uid=config["sensor_uid"],
        grasp_target_id=config["grasp_target_id"],
        finger_link_names=tuple(names),
        arm_label=config["arm_label"],
        **{
            key: config[key]
            for key in ("moving_mass", "table_uid", "chunk_count", "success_criterion")
            if key in config
        },
    )


def _decode_control_lowerer(
    value: object, *, operations: tuple[ControlOperationDeclaration, ...], path: str
):
    config = _closed(
        value,
        required={"kind", "operation_id"},
        optional={"operation_suffix"},
        path=path,
    )
    selected = [
        operation
        for operation in operations
        if operation.call.operation_id == config["operation_id"]
    ]
    if len(selected) != 1:
        raise ValueError(f"{path} requires one matching control-operation declaration.")
    call = selected[0].call
    kind = config["kind"]
    if kind in ("press_prepare", "press") and type(call) is PressCallDeclaration:
        cls = (
            PressPrepareFactory
            if kind == "press_prepare"
            else (
                PressApproachFactory
                if call.variant == "approach_only"
                else PressFactory
            )
        )
    elif kind in ("twist_prepare", "twist") and type(call) is TwistCallDeclaration:
        cls = TwistPrepareFactory if kind == "twist_prepare" else TwistFactory
    else:
        raise ValueError(f"{path} control family differs from its operation.")
    return specialize_control_factory(cls(call), config.get("operation_suffix", ""))
