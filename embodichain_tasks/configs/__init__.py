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

"""Paths for official task configuration resources."""

from __future__ import annotations

from pathlib import Path

__all__ = ["get_config_path"]

_CONFIG_ROOT = Path(__file__).resolve().parent


def get_config_path(relative_path: str | Path = ".") -> Path:
    """Return an installed official-task config path.

    Args:
        relative_path: Path relative to ``embodichain_tasks/configs``.

    Returns:
        Resolved filesystem path inside the installed task config package.

    Raises:
        ValueError: If ``relative_path`` is absolute or escapes the config root.
    """
    relative_path = Path(relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Config path must stay within the package: {relative_path}")
    return _CONFIG_ROOT / relative_path
