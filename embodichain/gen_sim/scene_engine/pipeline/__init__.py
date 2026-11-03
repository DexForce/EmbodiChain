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

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SCENE_BLUEPRINT_SCHEMA",
    "SCENE_EDIT_BLUEPRINT_SCHEMA",
    "SceneBlueprintPackage",
    "SceneEditBlueprintPackage",
    "SceneMaterialization",
    "analyze_edit",
    "analyze_image",
    "materialize_blueprint",
    "materialize_edit",
]


def __getattr__(name: str) -> Any:
    """Load generation APIs only when their public exports are requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.api"), name)
    globals()[name] = value
    return value
