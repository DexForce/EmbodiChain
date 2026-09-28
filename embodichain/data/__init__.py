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

import importlib
import os

from .constants import EMBODICHAIN_DEFAULT_DATABASE_ROOT

database_dir = EMBODICHAIN_DEFAULT_DATABASE_ROOT
database_2d_dir = os.path.join(database_dir, "2dasset")
database_agent_prompt_dir = os.path.join(database_dir, "agent_prompt")
database_demo_dir = os.path.join(database_dir, "demostration")

from .dataset import get_data_class, get_data_path

__all__ = [
    "EmbodiChainDataset",
    "get_data_class",
    "get_data_path",
    "assets",
    "database_dir",
    "database_2d_dir",
    "database_agent_prompt_dir",
    "database_demo_dir",
]


def __getattr__(name: str):
    if name == "assets":
        value = importlib.import_module(".assets", __name__)
    elif name == "EmbodiChainDataset":
        from .dataset import EmbodiChainDataset

        value = EmbodiChainDataset
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
