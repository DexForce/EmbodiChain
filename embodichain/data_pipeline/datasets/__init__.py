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

"""Online datasets and offline inspection with lazy simulation dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .online_data import OnlineDataset
    from .sampler import ChunkSizeSampler, GMMChunkSampler, UniformChunkSampler
    from .sequence import (
        BatchSource,
        DataBudgetExhaustedError,
        DataUnavailableError,
        SequenceSpecCfg,
    )
    from .sources import OnlineDataSource, TensorDataSource
    from .lerobot import LeRobotDataSource, LeRobotFieldMappingCfg
    from .offline_data import OfflineDataset
    from .sequence_data import SequenceDataset
    from .schema import validate_data_schema, data_schema_fingerprint

__all__ = [
    "ChunkSizeSampler",
    "GMMChunkSampler",
    "OnlineDataset",
    "UniformChunkSampler",
    "BatchSource",
    "DataBudgetExhaustedError",
    "SequenceSpecCfg",
    "OnlineDataSource",
    "TensorDataSource",
    "LeRobotDataSource",
    "LeRobotFieldMappingCfg",
    "OfflineDataset",
    "SequenceDataset",
    "DataUnavailableError",
    "validate_data_schema",
    "data_schema_fingerprint",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = {
        "OnlineDataset": "online_data",
        "OfflineDataset": "offline_data",
        "SequenceDataset": "sequence_data",
        "BatchSource": "sequence",
        "DataBudgetExhaustedError": "sequence",
        "DataUnavailableError": "sequence",
        "validate_data_schema": "schema",
        "data_schema_fingerprint": "schema",
        "SequenceSpecCfg": "sequence",
        "OnlineDataSource": "sources",
        "TensorDataSource": "sources",
        "LeRobotDataSource": "lerobot",
        "LeRobotFieldMappingCfg": "lerobot",
    }.get(name, "sampler")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
