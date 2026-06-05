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

"""Global robot registry for name-based lookup and instantiation.

The registry allows robot definitions to be registered by a unique string
name so that callers do not need to know the full import path.  Typical
usage::

    from embodichain.lab.sim.robots.registry import register_robot, get_robot_def

    @register_robot("my_robot")
    class MyRobotDef:
        ...

    robot = get_robot_def("my_robot")
    cfg = robot.build_cfg()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = ["register_robot", "get_robot_def", "build_robot_cfg"]

# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

_ROBOT_REGISTRY: dict[str, Type] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_robot(name: str):
    """Class decorator that registers a robot definition under *name*.

    If *name* is already present in the registry a warning is logged and the
    existing entry is overwritten.

    Args:
        name: Unique string identifier for the robot definition.

    Returns:
        The original class, unmodified.
    """

    def _decorator(cls: Type) -> Type:
        if name in _ROBOT_REGISTRY:
            logger.warning(
                "Robot '%s' is already registered; overwriting with %s.",
                name,
                cls.__qualname__,
            )
        _ROBOT_REGISTRY[name] = cls
        return cls

    return _decorator


def get_robot_def(name: str, **variant_kwargs: object) -> object:
    """Look up a robot definition by name and instantiate it.

    Args:
        name: Registered robot name.
        **variant_kwargs: Keyword arguments forwarded to the robot class
            constructor.

    Returns:
        An instance of the registered robot class.

    Raises:
        ValueError: If *name* is not found in the registry.
    """
    if name not in _ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot '{name}'. Available: {sorted(_ROBOT_REGISTRY.keys())}"
        )
    cls = _ROBOT_REGISTRY[name]
    return cls(**variant_kwargs)


def build_robot_cfg(name: str, **kwargs: object) -> RobotCfg:
    """Convenience helper that builds a :class:`RobotCfg` from a registered robot.

    The ``overrides`` key is extracted from *kwargs* (if present) and passed
    to the robot's :meth:`build_cfg` method.  All remaining kwargs are
    forwarded to :func:`get_robot_def` for instantiation.

    Args:
        name: Registered robot name.
        **kwargs: Forwarded to the constructor and/or ``build_cfg``.  The
            special key ``overrides`` is a dict passed to ``build_cfg``.

    Returns:
        A :class:`~embodichain.lab.sim.cfg.RobotCfg` produced by the
        robot's ``build_cfg`` method.

    Raises:
        ValueError: If *name* is not found in the registry.
    """
    from embodichain.lab.sim.cfg import (
        RobotCfg,
    )  # noqa: F811 – avoid circular import at module level

    overrides = kwargs.pop("overrides", {})
    robot = get_robot_def(name, **kwargs)
    cfg = robot.build_cfg(**overrides)
    assert isinstance(cfg, RobotCfg)
    return cfg
