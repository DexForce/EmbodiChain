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

from pathlib import Path

import pytest

from embodichain.gen_sim.asset_engine.pipeline.articulation import (
    generate_articulated_usdcs,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.services.adapters import articulated_generation


@pytest.mark.parametrize("is_articulated", [False, True])
def test_articulation_generation_obeys_explicit_flag_for_puzzle_cube(
    tmp_path: Path, is_articulated: bool
) -> None:
    cube = SceneObject(
        id="evidence_001",
        kind="asset",
        category="puzzle_cube",
        name="Rubik's cube",
        description="A colorful puzzle cube.",
        is_articulated=is_articulated,
        visible_rgba_path=str(tmp_path / "cube.png"),
    )
    generated = []

    class Client:
        def generate_articulated_usdc(
            self, *, prompt: str, image_path: str, output_path: Path
        ) -> Path:
            generated.append(output_path)
            output_path.write_bytes(b"USDC")
            return output_path

    generate_articulated_usdcs(
        asset_objects=[cube],
        output_root=tmp_path / "articulated",
        coarse_scales_y_up_by_id={cube.id: [1.0, 1.0, 1.0]},
        articulated_generation_client=Client() if is_articulated else None,
        canonicalize_fn=Path,
    )
    assert cube.is_articulated is is_articulated
    if is_articulated:
        assert generated == [tmp_path / "articulated" / "evidence_001.usdc"]
        assert Path(cube.articulated_usdc_path).read_bytes() == b"USDC"
    else:
        assert generated == []
        assert cube.articulated_usdc_path is None


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
