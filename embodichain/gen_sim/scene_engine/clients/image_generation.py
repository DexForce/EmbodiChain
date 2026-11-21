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

"""Compatibility facade; provider implementation lives in services."""

from __future__ import annotations

from embodichain.gen_sim.services.adapters.normalized import (
    ImageAdapter as ImageGenerationClient,
)
from embodichain.gen_sim.services.adapters.image_generation import (
    _load_dotenv_config,
    requests,
    read_scene_engine_env_values,
)

__all__: list[str] = []
