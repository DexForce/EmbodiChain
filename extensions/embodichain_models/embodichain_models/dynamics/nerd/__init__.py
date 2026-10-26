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

"""Lazy NeRD networks, physical codecs and supervised training integration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NeRDModelCfg",
    "NeRDModel",
    "FeatureNormalizer",
    "export_model",
    "load_model",
    "convert_legacy_checkpoint",
    "NeRDStateCodec",
    "NeRDObjective",
    "make_nerd_batch",
    "NeRDWindowSource",
    "build_supervised_trainer",
]

_EXPORT_MODULES = {
    "NeRDModelCfg": "model",
    "NeRDModel": "model",
    "FeatureNormalizer": "model",
    "export_model": "checkpoint",
    "load_model": "checkpoint",
    "convert_legacy_checkpoint": "checkpoint",
    "NeRDStateCodec": "codec",
    "NeRDObjective": "supervised",
    "make_nerd_batch": "supervised",
    "NeRDWindowSource": "sl_recipe",
    "build_supervised_trainer": "sl_recipe",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
