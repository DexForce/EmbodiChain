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

__all__ = ["list_robot_cfgs", "resolve_robot_cfg"]


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
    return cfg_cls.from_dict(params)
