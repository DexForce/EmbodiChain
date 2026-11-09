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

"""Dataset resolution, asset-download helpers, preset asset registries, and shared constants/enums used by simulation tasks and training pipelines."""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

from .constants import EMBODICHAIN_DEFAULT_DATABASE_ROOT

database_dir = EMBODICHAIN_DEFAULT_DATABASE_ROOT
database_2d_dir = os.path.join(database_dir, "2dasset")
database_agent_prompt_dir = os.path.join(database_dir, "agent_prompt")
database_demo_dir = os.path.join(database_dir, "demostration")

from .dataset import *
from .default_materials import list_default_materials, get_default_material_info

__all__ = [
    "assets",
    "EMBODICHAIN_DEFAULT_DATABASE_ROOT",
    "database_dir",
    "database_2d_dir",
    "database_agent_prompt_dir",
    "database_demo_dir",
    "EmbodiChainDataset",
    "DEFAULT_DATA_MODULES",
    "get_data_class",
    "get_data_path",
    "list_default_materials",
    "get_default_material_info",
]


def __getattr__(name: str) -> ModuleType:
    # Asset implementations import this package's dataset base. Loading all
    # asset re-exports here would re-enter a partially initialized provider.
    if name != "assets":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".assets", __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
