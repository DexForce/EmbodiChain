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

"""Compatibility facade for the shared articulated-generation adapter."""

from __future__ import annotations

from embodichain.gen_sim.services.adapters.normalized import (
    ArticulationAdapter as ArticulatedGenerationClient,
)
from embodichain.gen_sim.services.adapters.articulated_generation import (
    _validate_articulated_usdc,
    _load_dotenv_config,
    _positive_int,
    _validate_base_url,
    _validate_relative_path,
    _resolve_optional_image_path,
    requests,
    read_asset_engine_env_values,
)

__all__ = ["ArticulatedGenerationClient"]
