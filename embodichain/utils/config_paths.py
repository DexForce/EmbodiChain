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

from pathlib import Path

__all__ = ["resolve_config_path"]


def resolve_config_path(
    path: str | Path, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve one configuration path without opening the target file.

    Existing, absolute, and ordinary relative paths preserve their normal
    filesystem meaning. Repository-style paths below
    ``embodichain_tasks/configs`` are redirected to the packaged task-config
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

    task_prefix = ("embodichain_tasks", "configs")
    if resolved_path.parts[: len(task_prefix)] != task_prefix:
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
    colocated_root = Path(__file__).resolve().parents[2] / Path(*task_prefix)
    colocated_path = colocated_root / relative_path
    if colocated_path.exists() or (base_dir is not None and colocated_root.is_dir()):
        return colocated_path

    from embodichain_tasks.configs import get_config_path

    return get_config_path(relative_path)
