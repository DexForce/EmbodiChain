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

"""Scene proposal contracts and candidate sources shared by simulation hosts.

Candidates describe arena-local changes. Task semantics, scene materialization,
reset, execution, task acceptance, and persistence belong to host integrations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .cfg import SceneExpansionCfg
from .contracts import ScenePoseChange, SceneVariant
from .source import FixedSceneCandidateSource

__all__ = [
    "SceneExpansionCfg",
    "ScenePoseChange",
    "SceneVariant",
    "FixedSceneCandidateSource",
    "SceneLayoutInput",
    "PreparedSceneRandomization",
    "SceneRandomizationBatch",
    "SceneVariantApplicationResult",
    "SceneRandomizationPolicyCfg",
    "SceneRandomizationJobCfg",
    "MemberRandomizationCfg",
    "IntraGroupRandomizationCfg",
    "SupportOverrideCfg",
    "scene_layout_input_from_sim",
    "prepare_scene_randomization",
    "generate_scene_variants",
    "apply_scene_variants",
    "LoadedSceneVariants",
    "save_scene_variants",
    "load_scene_variants",
]

_POLICY_EXPORTS = {
    "SceneRandomizationPolicyCfg",
    "SceneRandomizationJobCfg",
    "MemberRandomizationCfg",
    "IntraGroupRandomizationCfg",
    "SupportOverrideCfg",
}
_IO_EXPORTS = {"LoadedSceneVariants", "save_scene_variants", "load_scene_variants"}


def __getattr__(name: str) -> Any:
    """Load geometry and persistence APIs only when requested."""
    if name not in __all__:
        raise AttributeError(name)
    module = (
        "_policy"
        if name in _POLICY_EXPORTS
        else "persistence" if name in _IO_EXPORTS else "layout"
    )
    return getattr(import_module(f"{__name__}.{module}"), name)
