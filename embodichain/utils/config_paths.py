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

_OFFICIAL_TASK_CONFIGS = Path("embodichain/extensions/tasks/configs")


def resolve_config_path(
    path: str | Path, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve one configuration path without opening the target file.

    Existing, absolute, and ordinary relative paths preserve their normal
    filesystem meaning. Repository-style paths below
    ``embodichain/extensions/tasks/configs`` resolve to the packaged task
    resources when opening them from outside the checkout.

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

    if not resolved_path.is_relative_to(_OFFICIAL_TASK_CONFIGS):
        return (
            resolved_path
            if base_dir is None
            else Path(base_dir).expanduser() / resolved_path
        )

    relative_path = resolved_path.relative_to(_OFFICIAL_TASK_CONFIGS)
    if ".." in relative_path.parts:
        raise ValueError(f"Config path must stay within the package: {relative_path}")

    # Bind packaged configs to the same checkout or wheel as the imported SDK.
    sdk_root = Path(__file__).resolve().parents[2]
    return sdk_root / _OFFICIAL_TASK_CONFIGS / relative_path
