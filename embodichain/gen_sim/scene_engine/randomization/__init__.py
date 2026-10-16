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

"""Prepare reusable geometry and support-group baselines for randomization.

Load a mandatory YAML configuration with :func:`load_randomization_config`,
then pass it to :func:`prepare_scene_baseline`. Baseline preparation validates
the portable scene and persists derived data; layout sampling is a later step.
Mesh loading dependencies are loaded only when a baseline API is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .cfg import SceneRandomizationCfg, load_randomization_config

if TYPE_CHECKING:
    from .baseline import PreparedSceneBaseline, prepare_scene_baseline

__all__ = [
    "SceneRandomizationCfg",
    "load_randomization_config",
    "PreparedSceneBaseline",
    "prepare_scene_baseline",
]


def __getattr__(name: str) -> Any:
    if name not in {"PreparedSceneBaseline", "prepare_scene_baseline"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.baseline"), name)
    globals()[name] = value
    return value
