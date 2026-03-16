# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable

__all__ = [
    "get_vla_backend",
    "get_registered_vla_backend_names",
    "create_vla_backend",
]


_VLA_BACKENDS: dict[str, Callable[..., Any]] = {}
_ENTRY_POINTS_DISCOVERED = False


def _discover_entry_points() -> None:
    """Discover and register VLA backends from entry_points."""
    global _ENTRY_POINTS_DISCOVERED
    if _ENTRY_POINTS_DISCOVERED:
        return
    _ENTRY_POINTS_DISCOVERED = True
    try:
        eps = entry_points(group="embodichain.vla_backends")
        for ep in eps:
            try:
                factory = ep.load()
                name = str(ep.name).lower()
                if name not in _VLA_BACKENDS:
                    _VLA_BACKENDS[name] = factory
            except Exception:
                pass
    except Exception:
        pass


def get_vla_backend(name: str) -> Callable[..., Any] | None:
    name = str(name).lower()
    if name in _VLA_BACKENDS:
        return _VLA_BACKENDS[name]
    _discover_entry_points()
    return _VLA_BACKENDS.get(name)


def get_registered_vla_backend_names() -> list[str]:
    _discover_entry_points()
    return list(_VLA_BACKENDS.keys())


def create_vla_backend(name: str, **kwargs) -> Any:
    factory = get_vla_backend(name)
    if factory is None:
        available = get_registered_vla_backend_names()
        raise ValueError(
            f"Unknown VLA backend '{name}'. Available: {available}. "
            "Ensure dexechain is installed (pip install dexechain)."
        )
    return factory(**kwargs)
