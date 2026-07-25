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

"""Recursive sub-package importer for auto-registration of task environments.

This follows the same pattern as IsaacLab's ``isaaclab_tasks`` — recursively
import every sub-package so that each task's ``__init__.py`` triggers its
``@register_env`` → ``gym.register()`` call chain.
"""

from __future__ import annotations

import importlib
import pkgutil

__all__ = ["import_packages"]


def import_packages(package_name: str, blacklist: list[str] | None = None) -> None:
    """Recursively import all sub-packages of *package_name*.

    Each imported sub-package executes its ``__init__.py``, which triggers
    ``@register_env`` decorators that call ``gym.register()``.

    Args:
        package_name: The fully-qualified package name (e.g. ``"embodichain_tasks"``).
        blacklist: Sub-package names to skip (e.g. ``["utils"]``).
    """
    blacklist = blacklist or []
    package = importlib.import_module(package_name)

    for _, name, is_pkg in pkgutil.walk_packages(
        package.__path__, prefix=package_name + "."
    ):
        if any(
            name.endswith("." + b) or name == package_name + "." + b for b in blacklist
        ):
            continue
        importlib.import_module(name)
