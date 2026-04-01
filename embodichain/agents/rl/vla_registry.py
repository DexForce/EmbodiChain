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

from importlib.metadata import entry_points
from typing import Any, Callable

from embodichain.utils.logger import log_warning

__all__ = [
    "get_vla_backend",
    "get_registered_vla_backend_names",
    "create_vla_backend",
]


_VLA_BACKENDS: dict[str, Callable[..., Any]] = {}
_ENTRY_POINTS_DISCOVERED = False
_ENTRY_POINTS_ENUM_LOGGED = False


def _discover_entry_points() -> None:
    """Discover and register VLA backends from entry_points."""
    global _ENTRY_POINTS_DISCOVERED, _ENTRY_POINTS_ENUM_LOGGED
    if _ENTRY_POINTS_DISCOVERED:
        return
    try:
        eps = entry_points(group="embodichain.vla_backends")
    except (OSError, ValueError, TypeError) as exc:
        if not _ENTRY_POINTS_ENUM_LOGGED:
            log_warning(
                "Could not enumerate 'embodichain.vla_backends' entry points: "
                f"{type(exc).__name__}: {exc}"
            )
            _ENTRY_POINTS_ENUM_LOGGED = True
        return

    for ep in eps:
        try:
            factory = ep.load()
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            log_warning(
                f"Failed to load VLA backend entry point name={ep.name!r} "
                f"value={ep.value!r}: {type(exc).__name__}: {exc}"
            )
            continue
        except Exception as exc:
            log_warning(
                f"Unexpected error loading VLA backend entry point name={ep.name!r} "
                f"value={ep.value!r}: {type(exc).__name__}: {exc}"
            )
            continue
        name = str(ep.name).lower()
        if name not in _VLA_BACKENDS:
            _VLA_BACKENDS[name] = factory

    _ENTRY_POINTS_DISCOVERED = True


def get_vla_backend(name: str) -> Callable[..., Any] | None:
    """Get a registered backend factory by name.

    This checks the in-memory registry first, and then lazily triggers
    entry-point discovery if needed.

    Args:
        name: Backend identifier (case-insensitive).

    Returns:
        The backend factory callable if found, otherwise ``None``.
    """
    name = str(name).lower()
    if name in _VLA_BACKENDS:
        return _VLA_BACKENDS[name]
    _discover_entry_points()
    return _VLA_BACKENDS.get(name)


def get_registered_vla_backend_names() -> list[str]:
    """List all currently discoverable VLA backend names.

    Returns:
        A list of backend names after lazy entry-point discovery.
    """
    _discover_entry_points()
    return list(_VLA_BACKENDS.keys())


def create_vla_backend(name: str, **kwargs) -> Any:
    """Instantiate a VLA backend by name.

    Args:
        name: Backend identifier (case-insensitive).
        **kwargs: Keyword arguments forwarded to the backend factory.

    Returns:
        The instantiated backend object (factory-defined type).

    Raises:
        ValueError: If the backend name is unknown.
    """
    factory = get_vla_backend(name)
    if factory is None:
        available = get_registered_vla_backend_names()
        raise ValueError(
            f"Unknown VLA backend '{name}'. Available: {available}. "
            "Ensure a package providing the 'embodichain.vla_backends' entry point "
            "group is installed."
        )
    return factory(**kwargs)
