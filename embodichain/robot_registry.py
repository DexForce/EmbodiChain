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
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = ["list_robot_cfgs", "resolve_robot_cfg"]


@lru_cache(maxsize=1)
def _robot_specs() -> dict[str, Mapping[str, Any]]:
    specs: dict[str, Mapping[str, Any]] = {
        "DualArmRobot": {
            "class_path": "embodichain.lab.sim.robots.dual_arm:DualArmRobotCfg"
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
            specs[name] = spec
    return specs


def list_robot_cfgs() -> tuple[str, ...]:
    """List registered robot IDs without importing configuration classes.

    Returns:
        Sorted robot IDs, including compatibility aliases and core compositions.
    """
    return tuple(sorted(_robot_specs()))


def resolve_robot_cfg(
    name: str, overrides: Mapping[str, Any] | None = None
) -> RobotCfg:
    """Construct a robot config using catalog defaults and caller overrides.

    Args:
        name: Registered robot ID or its legacy ``Cfg``-suffixed spelling.
        overrides: Robot-specific configuration fields. Caller values take
            precedence over catalog defaults; neither input mapping is mutated.

    Returns:
        The selected ``RobotCfg`` subclass decoded through its ``from_dict``.

    Raises:
        ValueError: If the robot ID is unknown or registered more than once.
        TypeError: If a catalog or its selected class violates the contract.
    """
    specs = _robot_specs()
    if not isinstance(name, str) or not name:
        raise ValueError("Robot ID must be a nonempty string.")
    key = name if name in specs else name.removesuffix("Cfg")
    if key not in specs:
        raise ValueError(
            f"Unknown robot {name!r}. Available robot IDs: {list_robot_cfgs()}. "
            "Install the robot provider or refresh the EmbodiChain editable install."
        )
    spec = specs[key]
    module_name, separator, class_name = spec["class_path"].partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(
            f"Robot {key!r} class_path must use the 'module:Class' format."
        )
    cfg_cls = getattr(import_module(module_name), class_name)
    from embodichain.lab.sim.cfg import RobotCfg

    if not isinstance(cfg_cls, type) or not issubclass(cfg_cls, RobotCfg):
        raise TypeError(f"Robot {key!r} must resolve to a RobotCfg subclass.")
    params = deepcopy({**spec.get("defaults", {}), **(overrides or {})})
    return cfg_cls.from_dict(params)
