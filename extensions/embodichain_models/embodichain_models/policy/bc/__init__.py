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

"""State and visual behavior cloning with the shared supervised learner.

Imports stay lazy: numeric inference does not load HDF5 or torchvision.
Task target construction and controller decoding belong to host applications.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ChunkBCModelCfg",
    "ChunkBCModel",
    "ChunkBCExecutor",
    "ResNet10Encoder",
    "ChunkBCObjective",
    "BCWindowDataset",
    "HDF5BCSource",
    "PrefetchedBCSource",
    "build_supervised_trainer",
]

_EXPORTS = {
    "ChunkBCModelCfg": "model",
    "ChunkBCModel": "model",
    "ChunkBCExecutor": "model",
    "ResNet10Encoder": "vision",
    "ChunkBCObjective": "supervised",
    "BCWindowDataset": "data",
    "HDF5BCSource": "supervised",
    "PrefetchedBCSource": "supervised",
    "build_supervised_trainer": "sl_recipe",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
