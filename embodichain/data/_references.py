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

"""Closed references to files in registered, verified asset bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from embodichain.utils.config_paths import resolve_config_path

__all__: list[str] = []


def _resolve_data_reference(value: object, *, base_dir: Path) -> Path:
    """Resolve an owner-relative filename or a registered asset file reference."""
    if isinstance(value, (str, Path)):
        if not str(value).strip():
            raise ValueError("resource path must be nonempty")
        return resolve_config_path(value, base_dir=base_dir).resolve()
    if not isinstance(value, Mapping) or set(value) != {"asset", "path"}:
        raise ValueError("asset reference requires exactly asset and path")
    asset, relative = value["asset"], value["path"]
    if not isinstance(asset, str) or not asset.isidentifier():
        raise ValueError("asset reference requires a registered asset ID")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "\x00" in relative
        or PurePosixPath(relative).is_absolute()
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError("asset path must be a relative file below the bundle root")
    from embodichain.data import get_data_class

    root = Path(get_data_class(asset)().extract_dir).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("asset file is missing or escapes the verified bundle")
    return result
