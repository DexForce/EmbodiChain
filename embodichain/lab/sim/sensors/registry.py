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

"""Shared sensor configuration/factory registry and lazy package discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_sensor import BaseSensor, SensorCfg

__all__ = [
    "SensorRegistration",
    "register_sensor",
    "discover_sensor_plugins",
    "get_sensor_registration",
    "list_sensor_types",
]


@dataclass(frozen=True)
class SensorRegistration:
    """Pair one configuration class with its construction contract.

    Args:
        sensor_type: Case-sensitive identifier, also the config's sensor_type.
        config_cls: SensorCfg subclass used by Python and YAML decoding.
        factory: Callable accepting ``factory(cfg, *, context=SensorContext)``
            and returning a BaseSensor. Built-in constructors retain their
            existing manager-owned adapters.
        legacy_factory: Explicit opt-in to the old ``factory(cfg, device)``
            convention, including existing camera/contact owner handling.
        api_version: Extension contract version. Only version 1 is supported.
    """

    sensor_type: str
    config_cls: type[SensorCfg]
    factory: Callable[..., BaseSensor]
    legacy_factory: bool = False
    api_version: int = 1


_registrations: dict[str, SensorRegistration] = {}
_entry_points: dict[str, metadata.EntryPoint] = {}
_discovered = False
_loading: set[str] = set()


def _validate(registration: SensorRegistration) -> None:
    from .base_sensor import SensorCfg

    if not isinstance(registration, SensorRegistration):
        raise TypeError("Sensor entry points must export a SensorRegistration value.")
    if (
        not isinstance(registration.sensor_type, str)
        or not registration.sensor_type.strip()
    ):
        raise ValueError("A sensor registration requires a non-empty sensor_type.")
    if registration.api_version != 1:
        raise ValueError(
            f"Sensor {registration.sensor_type!r} requires extension API "
            f"{registration.api_version}; this EmbodiChain supports API 1."
        )
    if not isinstance(registration.config_cls, type) or not issubclass(
        registration.config_cls, SensorCfg
    ):
        raise TypeError("A registered sensor config must inherit SensorCfg.")
    if registration.config_cls().sensor_type != registration.sensor_type:
        raise ValueError("The registration and configuration sensor_type must match.")
    if not callable(registration.factory):
        raise TypeError("A sensor factory must be callable.")


def register_sensor(registration: SensorRegistration) -> None:
    """Register a sensor explicitly without modifying built-in modules.

    Args:
        registration: Configuration class and factory to register.

    Raises:
        ValueError: The identifier already exists or the contract is incompatible.
        TypeError: The registration has an invalid config or factory.
    """
    _validate(registration)
    name = registration.sensor_type
    if name in _registrations or name in _entry_points:
        raise ValueError(f"Sensor type {name!r} is already registered.")
    _registrations[name] = registration


def discover_sensor_plugins() -> None:
    """Discover installed sensor names without importing their implementations.

    Each ``embodichain.sensors`` entry-point name is a sensor_type; its value
    points to a SensorRegistration object. Discovery is deterministic and
    cached for the process lifetime. Install packages before starting Python.

    Raises:
        ValueError: Installed packages declare duplicate or reserved names.
    """
    global _discovered
    if _discovered:
        return
    candidates: dict[str, metadata.EntryPoint] = {}
    for entry in sorted(
        metadata.entry_points(group="embodichain.sensors"),
        key=lambda item: (item.name, item.value),
    ):
        if entry.name in _registrations or entry.name in candidates:
            raise ValueError(
                f"Duplicate sensor type {entry.name!r} from entry point "
                f"{entry.value!r}; choose a unique name in embodichain.sensors."
            )
        candidates[entry.name] = entry
    _entry_points.update(candidates)
    _discovered = True


def get_sensor_registration(sensor_type: str) -> SensorRegistration:
    """Resolve a sensor's config and factory, loading only that plugin.

    Args:
        sensor_type: Case-sensitive registered identifier.

    Returns:
        The validated configuration/factory pair.

    Raises:
        KeyError: No sensor with this identifier is installed or registered.
        RuntimeError: Loading the selected plugin fails or recurses.
        ValueError: Discovery encounters duplicate identifiers.
    """
    discover_sensor_plugins()
    if sensor_type in _registrations:
        return _registrations[sensor_type]
    entry = _entry_points.get(sensor_type)
    if entry is None:
        raise KeyError(
            f"Unknown sensor type {sensor_type!r}. Available types: "
            f"{list_sensor_types()}. Install its package or call register_sensor()."
        )
    if sensor_type in _loading:
        raise RuntimeError(f"Recursive loading of sensor {sensor_type!r}.")
    _loading.add(sensor_type)
    try:
        registration = entry.load()
        _validate(registration)
        if registration.sensor_type != sensor_type:
            raise ValueError(
                "The entry-point name must match registration.sensor_type."
            )
        _registrations[sensor_type] = registration
        return registration
    except Exception as error:
        raise RuntimeError(
            f"Could not load sensor {sensor_type!r} from {entry.value!r}. "
            f"Check the plugin's dependencies and extension API compatibility: {error}"
        ) from error
    finally:
        _loading.remove(sensor_type)


def list_sensor_types() -> list[str]:
    """Return sorted installed/registered identifiers without loading plugins.

    Returns:
        Case-sensitive sensor identifiers.
    """
    discover_sensor_plugins()
    return sorted(_registrations.keys() | _entry_points.keys())


def _resolve_sensor_factory(
    sensor_type: str, legacy_factories: Mapping[str, Callable[..., BaseSensor]]
) -> tuple[SensorRegistration | None, Callable[..., BaseSensor], bool]:
    """Share registry and legacy override precedence with all factory consumers."""
    try:
        registration = get_sensor_registration(sensor_type)
    except KeyError as error:
        if sensor_type not in legacy_factories:
            raise ValueError(
                f"Unsupported sensor type {sensor_type!r}. Supported types: "
                f"{list_sensor_types()}."
            ) from error
        return None, legacy_factories[sensor_type], True
    factory = legacy_factories.get(sensor_type, registration.factory)
    legacy = registration.legacy_factory or factory is not registration.factory
    return registration, factory, legacy
