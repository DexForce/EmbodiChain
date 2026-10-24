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

from __future__ import annotations

import pytest

from embodichain.gen_sim.asset_engine.clients import articulated_generation


def test_asset_engine_articulation_client_owns_canonical_environment_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "ASSET_ENGINE_ARTICULATED_GENERATION_BASE_URL": "http://asset-articulation/",
        "ASSET_ENGINE_ARTICULATED_GENERATION_TIMEOUT_S": "7200",
        "ASSET_ENGINE_ARTICULATED_GENERATION_MAX_ATTEMPTS": "2",
        "ASSET_ENGINE_ARTICULATED_GENERATION_HEALTH_PATH": "/health",
        "ASSET_ENGINE_ARTICULATED_GENERATION_GENERATE_PATH": "/generate_articulation",
    }
    monkeypatch.setattr(
        articulated_generation,
        "read_asset_engine_env_values",
        lambda *_: values,
    )

    client = articulated_generation.ArticulatedGenerationClient.from_dotenv()

    assert client._base_url == "http://asset-articulation/"
    assert client._timeout_s == 7200
    assert client._max_attempts == 2
    assert client._health_path == "/health"
    assert client._generate_path == "/generate_articulation"
