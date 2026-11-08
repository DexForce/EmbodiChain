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

"""Resolve robot configurations from lightweight entry-point catalogs.

Each ``embodichain.robots`` entry point provides a mapping from robot IDs to
specifications with a ``class_path`` (``module:RobotCfgSubclass``) and optional
``defaults`` mapping. Catalogs must defer imports of their robot implementations.
Optional control metadata describes independent default control DOFs, raw
per-part joint counts, mimic relations, and declared variants. Catalogs can
advertise compatible single-arm bases through ``supports_dual_arm``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = ["list_robot_cfgs", "resolve_robot_cfg", "resolve_gripper_defaults"]


def _validate_control_metadata(name: str, spec: Mapping[str, Any]) -> None:
    """Validate optional display metadata before exposing it to CLI consumers."""
    for field in ("vendor", "morphology"):
        value = spec.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise TypeError(
                f"Robot {name!r} {field} must be a nonempty string or None."
            )
    arm_count = spec.get("arm_count")
    if arm_count is not None and (
        not isinstance(arm_count, int) or isinstance(arm_count, bool) or arm_count < 0
    ):
        raise TypeError(
            f"Robot {name!r} arm_count must be a nonnegative integer or None."
        )
    if "defaults" in spec and not isinstance(spec["defaults"], Mapping):
        raise TypeError(f"Robot {name!r} defaults must be a mapping.")
    dof = spec.get("dof")
    if dof is not None and (
        not isinstance(dof, int) or isinstance(dof, bool) or dof < 0
    ):
        raise TypeError(f"Robot {name!r} dof must be a nonnegative integer.")
    parts = spec.get("control_parts")
    if parts is not None and (
        not isinstance(parts, Mapping)
        or any(
            not isinstance(part, str)
            or not part
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for part, count in parts.items()
        )
    ):
        raise TypeError(
            f"Robot {name!r} control_parts must map nonempty names "
            "to nonnegative joint counts."
        )
    mimics = spec.get("mimic_joints")
    if mimics is not None and (
        not isinstance(mimics, Mapping)
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in mimics.items()
        )
    ):
        raise TypeError(
            f"Robot {name!r} mimic_joints must map follower names to leader names."
        )
    part_mimics = spec.get("control_part_mimics")
    if part_mimics is not None:
        if not isinstance(part_mimics, Mapping):
            raise TypeError(f"Robot {name!r} control_part_mimics must be a mapping.")
        for part, followers in part_mimics.items():
            if (
                not isinstance(part, str)
                or not part
                or not isinstance(followers, Sequence)
                or isinstance(followers, (str, bytes))
                or any(not isinstance(joint, str) or not joint for joint in followers)
                or (
                    parts is not None
                    and (part not in parts or len(set(followers)) > parts[part])
                )
                or (
                    mimics is not None
                    and any(joint not in mimics for joint in followers)
                )
            ):
                raise TypeError(f"Robot {name!r} has invalid control_part_mimics.")
    if "supports_dual_arm" in spec and not isinstance(spec["supports_dual_arm"], bool):
        raise TypeError(f"Robot {name!r} supports_dual_arm must be a boolean.")
    for key in ("variant_of", "default_variant"):
        if key in spec and (not isinstance(spec[key], str) or not spec[key]):
            raise TypeError(f"Robot {name!r} {key} must be a nonempty string.")
    variants = spec.get("variants")
    if variants is not None:
        if not isinstance(variants, Mapping):
            raise TypeError(f"Robot {name!r} variants must be a mapping.")
        for variant, profile in variants.items():
            if (
                not isinstance(variant, str)
                or not variant
                or not isinstance(profile, Mapping)
            ):
                raise TypeError(f"Robot {name!r} has an invalid variant profile.")
            if "variants" in profile:
                raise TypeError(
                    f"Robot {name!r} nested variant profiles are not supported."
                )
            _validate_control_metadata(f"{name}/{variant}", profile)
        if (
            spec.get("default_variant") is not None
            and spec["default_variant"] not in variants
        ):
            raise ValueError(
                f"Robot {name!r} default_variant is not declared in variants."
            )


def _dual_arm_variants(
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Summarize declared compatible single-arm bases without constructing them."""
    variants: dict[str, Mapping[str, Any]] = {}
    for name, base in specs.items():
        if not base.get("supports_dual_arm") or base.get("variants"):
            continue
        parts = base.get("control_parts") or {}
        controls = {
            f"{side}_{part}": count
            for part, count in parts.items()
            for side in ("left", "right")
        }
        if "arm" in parts:
            controls["dual_arm"] = 2 * parts["arm"]
        mimics = base.get("mimic_joints")
        part_mimics = base.get("control_part_mimics")
        variants[name] = {
            "dof": 2 * base["dof"] if base.get("dof") is not None else None,
            "control_parts": controls,
            "mimic_joints": (
                {
                    f"{side}_{follower}": f"{side}_{leader}"
                    for follower, leader in mimics.items()
                    for side in ("left", "right")
                }
                if mimics is not None
                else None
            ),
            "control_part_mimics": (
                {
                    f"{side}_{part}": tuple(
                        f"{side}_{follower}" for follower in followers
                    )
                    for part, followers in part_mimics.items()
                    for side in ("left", "right")
                }
                if part_mimics is not None
                else None
            ),
            "defaults": {"base_robot": name},
        }
        if "arm" in parts and part_mimics and part_mimics.get("arm"):
            variants[name]["control_part_mimics"]["dual_arm"] = tuple(
                f"{side}_{follower}"
                for side in ("left", "right")
                for follower in part_mimics["arm"]
            )
    return variants


@lru_cache(maxsize=1)
def _robot_specs() -> dict[str, Mapping[str, Any]]:
    specs: dict[str, Mapping[str, Any]] = {
        "DualArmRobot": {
            "class_path": "embodichain.lab.sim.robots.dual_arm:DualArmRobotCfg",
            "vendor": None,
            "morphology": "fixed_manipulator",
            "arm_count": 2,
            "dof": 12,
            "control_parts": {"left_arm": 6, "right_arm": 6, "dual_arm": 12},
            "mimic_joints": {},
            "control_part_mimics": {},
        }
    }
    for entry_point in metadata.entry_points(group="embodichain.robots"):
        catalog = entry_point.load()
        if not isinstance(catalog, Mapping):
            raise TypeError(
                f"Robot catalog {entry_point.name!r} must provide a mapping."
            )
        for name, spec in catalog.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Robot catalog IDs must be nonempty strings.")
            if name in specs:
                raise ValueError(
                    f"Duplicate robot ID {name!r} in catalog {entry_point.name!r}."
                )
            if (
                not isinstance(spec, Mapping)
                or not isinstance(spec.get("class_path"), str)
                or not isinstance(spec.get("defaults", {}), Mapping)
            ):
                raise TypeError(
                    f"Robot {name!r} must declare a class_path string and "
                    "optional defaults mapping."
                )
            _validate_control_metadata(name, spec)
            specs[name] = spec
    dual_variants = _dual_arm_variants(specs)
    if dual_variants:
        specs["DualArmRobot"] = {
            **specs["DualArmRobot"],
            "variants": dual_variants,
            **({"default_variant": "ur10"} if "ur10" in dual_variants else {}),
        }
    return specs


def list_robot_cfgs() -> tuple[str, ...]:
    """List registered robot IDs without importing configuration classes.

    Returns:
        Sorted robot IDs, including variant presets and core compositions.
    """
    return tuple(sorted(_robot_specs()))


def resolve_robot_cfg(
    name: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    variant: str | None = None,
) -> RobotCfg:
    """Construct a robot config using catalog defaults and caller overrides.

    Args:
        name: Exact registered robot ID.
        overrides: Robot-specific configuration fields. Caller values take
            precedence over catalog and variant defaults. Input mappings are
            not mutated; nested values follow the existing top-level merge.
        variant: Exact declared variant ID. When omitted, the existing catalog
            and configuration-class defaults apply; ``default_variant`` is
            display metadata and does not implicitly select a profile.

    Returns:
        The selected ``RobotCfg`` subclass decoded through its ``from_dict``.

    Raises:
        ValueError: If the robot or variant ID is unknown, or an ID is
            registered more than once.
        TypeError: If a catalog or its selected class violates the contract.
    """
    specs = _robot_specs()
    if not isinstance(name, str) or not name:
        raise ValueError("Robot ID must be a nonempty string.")
    key = name
    if key not in specs:
        raise ValueError(
            f"Unknown robot {name!r}. Available robot IDs: {list_robot_cfgs()}. "
            "Install the robot provider or refresh the EmbodiChain editable install."
        )
    spec = specs[key]
    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("Robot overrides must be a mapping.")
    variant_defaults: Mapping[str, Any] = {}
    if variant is not None:
        variants = spec.get("variants") or {}
        if not isinstance(variant, str) or not variant or variant not in variants:
            raise ValueError(
                f"Unknown variant {variant!r} for robot {name!r}. "
                f"Available variants: {tuple(variants)}."
            )
        variant_defaults = variants[variant].get("defaults", {})
    module_name, separator, class_name = spec["class_path"].partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(
            f"Robot {key!r} class_path must use the 'module:Class' format."
        )
    cfg_cls = getattr(import_module(module_name), class_name)
    from embodichain.lab.sim.cfg import RobotCfg

    if not isinstance(cfg_cls, type) or not issubclass(cfg_cls, RobotCfg):
        raise TypeError(f"Robot {key!r} must resolve to a RobotCfg subclass.")
    params = deepcopy(
        {**spec.get("defaults", {}), **variant_defaults, **(overrides or {})}
    )
    cfg = cfg_cls.from_dict(params)
    return cfg


def resolve_gripper_defaults(
    name: str, *, variant: str | None = None
) -> dict[str, object]:
    """Read owned parallel-jaw defaults through the existing metadata-only catalog.

    Args:
        name: Exact robot-catalog ID exposing a gripper description.
        variant: Explicit model variant, including native left/right selection.

    Returns:
        Detached primitive configuration; asset references remain unresolved.

    Raises:
        ValueError: The ID or variant has no explicit gripper description.
        TypeError: The provider returns incomplete or non-serializable data.
    """
    spec = _robot_specs().get(name)
    if spec is None or not spec.get("description_path"):
        raise ValueError(f"Robot {name!r} has no parallel-jaw description.")
    module, separator, symbol = spec["description_path"].partition(":")
    if not separator or not module or not symbol:
        raise ValueError("description_path must use module:function format.")
    if variant is not None and variant not in (spec.get("variants") or {}):
        raise ValueError(f"Unknown gripper variant {variant!r} for {name!r}.")
    result = getattr(import_module(module), symbol)(variant)
    return _validate_gripper_description(result)


def _validate_gripper_transform(value: object) -> tuple[tuple[float, ...], ...]:
    """Normalize a finite SE(3) matrix without importing numerical backends."""
    import math

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Gripper transform must have shape (4, 4).")
    try:
        rows = tuple(tuple(float(x) for x in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper transform must contain finite numbers.") from exc
    if any(len(row) != 4 or any(not math.isfinite(x) for x in row) for row in rows):
        raise ValueError("Gripper transform must have finite shape (4, 4).")
    if any(abs(a - b) > 1e-7 for a, b in zip(rows[3], (0, 0, 0, 1))):
        raise ValueError("Gripper transform must have homogeneous last row.")
    if any(
        abs(sum(rows[k][i] * rows[k][j] for k in range(3)) - (i == j)) > 1e-6
        for i in range(3)
        for j in range(3)
    ):
        raise ValueError("Gripper transform rotation must be orthonormal.")
    r = rows
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(determinant - 1) > 1e-6:
        raise ValueError("Gripper transform must be a proper rotation.")
    return rows


def _validate_gripper_description(value: object) -> dict[str, object]:
    """Validate and detach model-owned primitives without importing Lab or GraspKit."""
    import json
    import math

    if not isinstance(value, Mapping):
        raise TypeError("Gripper description must be a mapping.")
    required = {
        "fpath",
        "model_id",
        "root_link",
        "collision_links",
        "root_from_contact_frame",
        "joint_commands",
        "default_root_from_tcp",
        "control_parts",
        "min_opening_width",
        "max_opening_width",
        "grasp_generators",
    }
    if required - value.keys():
        raise ValueError(
            f"Gripper description is incomplete: {sorted(required - value.keys())}"
        )
    data = deepcopy(dict(value))
    for name in ("fpath", "model_id", "root_link"):
        if (
            type(data[name]) is not str
            or not data[name]
            or data[name] != data[name].strip()
        ):
            raise ValueError(f"Gripper {name} must be an explicit nonempty string.")
    for name in ("collision_links", "grasp_generators"):
        items = data[name]
        if (
            not isinstance(items, (list, tuple))
            or len(set(items)) != len(items)
            or any(type(x) is not str or not x or x != x.strip() for x in items)
        ):
            raise ValueError(f"Gripper {name} must contain distinct explicit names.")
        data[name] = list(items)
    if data["grasp_generators"] and not data["collision_links"]:
        raise ValueError(
            "Grasp generation requires explicit root_link and collision_links."
        )
    pads = data.get("pad_links")
    if pads is not None and (
        not isinstance(pads, (list, tuple))
        or len(pads) != 2
        or len(set(pads)) != 2
        or not set(pads) <= set(data["collision_links"])
    ):
        raise ValueError("pad_links must identify two distinct collision links.")
    for name in ("min_opening_width", "max_opening_width"):
        number = data[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
        data[name] = float(number)
    if data["min_opening_width"] >= data["max_opening_width"]:
        raise ValueError("min_opening_width must be less than max_opening_width.")
    for name in ("root_from_contact_frame", "default_root_from_tcp"):
        data[name] = [list(row) for row in _validate_gripper_transform(data[name])]
    parts = data["control_parts"]
    names = parts.get("eef") if isinstance(parts, Mapping) else None
    if (
        not isinstance(names, (list, tuple))
        or not names
        or len(set(names)) != len(names)
        or any(type(x) is not str or not x for x in names)
    ):
        raise ValueError("Gripper native eef joints must be explicit and unique.")
    commands = data["joint_commands"]
    if not isinstance(commands, Mapping) or not {"open", "grasp"} <= commands.keys():
        raise ValueError("Gripper commands must provide open and grasp.")
    for command in commands.values():
        if (
            not isinstance(command, Mapping)
            or set(command) != set(names)
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                for x in command.values()
            )
        ):
            raise ValueError(
                "Gripper commands must cover all native moving joints with finite positions."
            )
    json.dumps(data, allow_nan=False)
    return data
