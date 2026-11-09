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

"""Stable path resolution for user and packaged configuration files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TypeVar

__all__ = ["resolve_config_path", "resolve_local_resource_paths"]

_ConfigValue = TypeVar("_ConfigValue")


def resolve_config_path(
    path: str | Path, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve one configuration path without opening the target file.

    Existing, absolute, and ordinary relative paths preserve their normal
    filesystem meaning. Repository-style paths below
    ``extensions/embodichain_tasks/configs`` (or the legacy
    ``embodichain_tasks/configs`` prefix) are redirected to the packaged task-config
    resource so the same configuration reference works from an installed
    wheel.

    With an explicit owner directory, ordinary relative references resolve
    from that directory even when a same-named CWD file exists. Official-task
    references bind to this SDK's packaged configs and ignore CWD shadows.

    Args:
        path: User path or repository-style official-task configuration path.
        base_dir: Owning configuration directory for nested references. ``None``
            preserves the existing filesystem and CWD behavior.

    Returns:
        Expanded filesystem path, resolved through the packaged task resource
        only when the input uses the official-task configuration prefix.

    Raises:
        TypeError: If ``path`` is not path-like.
        ValueError: If a packaged config path escapes the config root.
    """
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute() or (base_dir is None and resolved_path.exists()):
        return resolved_path

    task_prefixes = (
        ("extensions", "embodichain_tasks", "configs"),
        ("embodichain_tasks", "configs"),
    )
    task_prefix = next(
        (
            prefix
            for prefix in task_prefixes
            if resolved_path.parts[: len(prefix)] == prefix
        ),
        None,
    )
    if task_prefix is None:
        return (
            resolved_path
            if base_dir is None
            else Path(base_dir).expanduser() / resolved_path
        )

    relative_path = Path(*resolved_path.parts[len(task_prefix) :])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Config path must stay within the package: {relative_path}")

    # Prefer configs co-located with the imported core package. This keeps a
    # source worktree self-contained even when another checkout is installed
    # in editable mode, and resolves to the same location in a wheel install.
    sdk_root = Path(__file__).resolve().parents[2]
    for prefix in task_prefixes:
        colocated_root = sdk_root / Path(*prefix)
        colocated_path = colocated_root / relative_path
        if colocated_path.exists() or (
            base_dir is not None and colocated_root.is_dir()
        ):
            return colocated_path

    from embodichain_tasks.configs import get_config_path

    return get_config_path(relative_path)


def resolve_local_resource_paths(
    value: _ConfigValue, *, base_dir: str | Path
) -> _ConfigValue:
    """Own a declaration and resolve its explicit local resource files.

    Only string values under ``fpath``, ``urdf_path`` or ``xml_path`` prefixed
    with ``./`` or ``../`` select owner-relative files. Bare registry references,
    absolute strings and Path values keep their existing meaning. Dictionaries
    and lists are traversed with the existing rules; other containers are copied
    without adding resource interpretation. This performs no registry, network
    or SDK lookup.

    Args:
        value: Declaration to copy, retaining its container types.
        base_dir: Directory of the component or deployment declaring the files.

    Returns:
        Independent declaration with explicit local resource strings resolved to
        absolute file paths. The input remains unchanged, including on failure.

    Raises:
        FileNotFoundError: An explicit owner-relative resource is not a file.
    """
    owned = deepcopy(value)
    owner = Path(base_dir).expanduser()

    def resolve(current: object) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                if (
                    key in {"fpath", "urdf_path", "xml_path"}
                    and isinstance(child, str)
                    and child.startswith(("./", "../"))
                ):
                    selected = (owner / child).resolve()
                    if not selected.is_file():
                        raise FileNotFoundError(selected)
                    current[key] = str(selected)
                else:
                    resolve(child)
        elif isinstance(current, list):
            for child in current:
                resolve(child)

    resolve(owned)
    return owned
