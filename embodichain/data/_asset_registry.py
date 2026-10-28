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

"""Shared core and entry-point asset discovery; constructors run only on demand."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import import_module, metadata
import inspect
from typing import Any

__all__ = []

_CATEGORY_MODULES: dict[str, tuple[str, ...]] = {
    "demo": ("embodichain.data.assets.demo_assets",),
    "eef": ("embodichain.data.assets.eef_assets",),
    "materials": ("embodichain.data.assets.materials",),
    "obj": ("embodichain.data.assets.obj_assets",),
    "robot": (
        "embodichain.data.assets.robot_assets",
        "embodichain.data.assets.locomotion_assets",
        "embodichain.data.assets.so101_assets",
    ),
    "scene": ("embodichain.data.assets.scene_assets",),
    "task": ("embodichain.data.assets.task_assets",),
    "w1": ("embodichain.data.assets.w1_assets",),
}

# Retain the public, single-module compatibility view used by existing callers.
_COMPAT_CATEGORY_MODULES: dict[str, str] = {
    category: modules[0] for category, modules in _CATEGORY_MODULES.items()
}


@lru_cache(maxsize=1)
def _asset_specs() -> dict[str, dict[str, str]]:
    """Merge declared asset sources, excluding forwarded compatibility aliases."""
    import open3d

    result: dict[str, dict[str, str]] = {}
    names: set[str] = set()

    def add(name: str, spec: Mapping[str, Any], provider: str) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError(f"Asset catalog {provider!r} IDs must be nonempty strings.")
        if name.casefold() in names:
            raise ValueError(f"Duplicate asset ID {name!r} in catalog {provider!r}.")
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Asset {name!r} must declare class_path and category strings."
            )
        path, category = spec.get("class_path"), spec.get("category")
        if not isinstance(path, str) or not isinstance(category, str) or not category:
            raise TypeError(
                f"Asset {name!r} must declare class_path and category strings."
            )
        module, separator, cls = path.partition(":")
        if not separator or not module or not cls or ":" in cls:
            raise ValueError(f"Asset {name!r} class_path must use 'module:Class'.")
        names.add(name.casefold())
        result[name] = {"class_path": path, "category": category}

    for category, modules in _CATEGORY_MODULES.items():
        for module_name in modules:
            module = import_module(module_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    not name.startswith("_")
                    and issubclass(cls, open3d.data.DownloadDataset)
                    and cls is not open3d.data.DownloadDataset
                    and cls.__module__ == module_name
                ):
                    add(
                        name,
                        {
                            "class_path": f"{module_name}:{cls.__name__}",
                            "category": category,
                        },
                        module_name,
                    )
    for entry in metadata.entry_points(group="embodichain.assets"):
        catalog = entry.load()
        if not isinstance(catalog, Mapping):
            raise TypeError(f"Asset catalog {entry.name!r} must provide a mapping.")
        for name, spec in catalog.items():
            add(name, spec, entry.name)
    return result


def _get_asset_class(name: str) -> type | None:
    """Resolve a registered class without constructing or downloading it."""
    spec = _asset_specs().get(name)
    if spec is None:
        return None
    module, _, class_name = spec["class_path"].partition(":")
    cls = getattr(import_module(module), class_name)
    import open3d

    if not isinstance(cls, type) or not issubclass(cls, open3d.data.DownloadDataset):
        raise TypeError(f"Asset {name!r} must resolve to a DownloadDataset subclass.")
    return cls


def _get_registry() -> dict[str, list[tuple[str, type]]]:
    """Return fresh category lists backed by the common asset index."""
    result: dict[str, list[tuple[str, type]]] = {
        category: [] for category in _CATEGORY_MODULES
    }
    for name, spec in sorted(_asset_specs().items()):
        result.setdefault(spec["category"], []).append((name, _get_asset_class(name)))
    return result


def _categories() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(_CATEGORY_MODULES)
            | {spec["category"] for spec in _asset_specs().values()}
        )
    )
