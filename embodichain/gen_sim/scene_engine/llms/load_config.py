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


"""Compatibility access to the existing Scene VLM configuration."""

from __future__ import annotations
from embodichain.gen_sim.scene_engine.configs.environment import (
    read_scene_engine_env_values,
)
from embodichain.gen_sim.services.adapters.vlm_config import (
    LLMConfig,
    load_llm_config as _load,
)

__all__: list[str] = []


def load_llm_config() -> LLMConfig:
    """Read the same required Scene settings through the compatibility reader."""
    return _load(reader=read_scene_engine_env_values)
