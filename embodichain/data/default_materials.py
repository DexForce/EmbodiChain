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

"""Offline catalog for the versioned DexSim default material collection."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = ["list_default_materials", "get_default_material_info"]


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    return json.loads(
        files("embodichain.data")
        .joinpath("default_materials.json")
        .read_text(encoding="utf-8")
    )


def list_default_materials() -> list[str]:
    """List the 18 public presets without downloading assets or creating a World.

    Returns:
        Preset names in catalog order. Auxiliary resources are omitted.
    """
    return [item["name"] for item in _catalog()["materials"] if not item["auxiliary"]]


def get_default_material_info(name: str) -> dict[str, Any]:
    """Read independent metadata for a public preset or auxiliary resource.

    Args:
        name: Case-sensitive catalog name, including ``SoapBubbleInterior``.

    Returns:
        Description, supported renderers, dependencies, suggested texture size,
        geometry guidance and source revision. Suggested sizes do not change UVs.

    Raises:
        ValueError: The name is not in the built-in catalog.
    """
    for item in _catalog()["materials"]:
        if item["name"] == name:
            result = copy.deepcopy(item)
            result["texture_resolution"] = list(_catalog()["texture_resolution"])
            result["revision"] = _catalog()["bundle"]["revision"]
            return result
    raise ValueError(
        f"Unknown default material {name!r}. Available presets: {', '.join(list_default_materials())}."
    )
