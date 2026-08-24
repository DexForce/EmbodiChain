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

"""Independent config generation for Action Engine."""

from __future__ import annotations

from .config_builder import VLM_CAMERA_UIDS, canonical_robot_profile
from .assets import normalize_scene_assets
from .generator import generate_action_engine_config
from .models import GeneratedConfigPaths, PreparedScene

__all__ = [
    "GeneratedConfigPaths",
    "PreparedScene",
    "VLM_CAMERA_UIDS",
    "canonical_robot_profile",
    "generate_action_engine_config",
    "normalize_scene_assets",
]
