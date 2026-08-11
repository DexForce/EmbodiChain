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


def resolve_config_path(path: str | Path) -> Path:
    """Resolve one configuration path without opening the target file.

    Existing, absolute, and ordinary relative paths preserve their normal
    filesystem meaning. Repository-style paths below
    ``embodichain_tasks/configs`` are redirected to the packaged task-config
    resource so the same configuration reference works from an installed
    wheel.

    Args:
        path: User path or repository-style official-task configuration path.

    Returns:
        Expanded filesystem path, resolved through the packaged task resource
        only when the input uses the official-task configuration prefix.

    Raises:
        TypeError: If ``path`` is not path-like.
    """
    resolved_path = Path(path).expanduser()
    if resolved_path.exists() or resolved_path.is_absolute():
        return resolved_path

    task_prefix = ("embodichain_tasks", "configs")
    if resolved_path.parts[: len(task_prefix)] != task_prefix:
        return resolved_path

    from embodichain_tasks.configs import get_config_path

    relative_path = Path(*resolved_path.parts[len(task_prefix) :])
    return get_config_path(relative_path)
