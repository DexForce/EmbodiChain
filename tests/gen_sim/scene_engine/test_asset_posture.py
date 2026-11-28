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

import json
from pathlib import Path

import numpy as np
import pytest

from embodichain.gen_sim.scene_engine.pipeline.utils import (
    _asset_posture as asset_posture,
)


class _Vision:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path | None = None,
    ) -> str:
        assert image_path is not None and Path(image_path).is_file()
        self.prompts.append(json.loads(user_prompt))
        return next(self._responses)


@pytest.fixture
def rendered_rotations(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    rendered: list[np.ndarray] = []

    def render(operation: str, arguments: dict[str, object], *, timeout: float) -> None:
        assert operation == "asset_posture" and timeout > 0
        rendered.append(np.asarray(arguments["rotation_y_up"], dtype=float))
        Path(str(arguments["output_path"])).write_bytes(b"preview")

    monkeypatch.setattr(asset_posture, "_run_blender_job", render)
    return rendered


def test_review_applies_only_the_accepted_candidate(
    tmp_path: Path, rendered_rotations: list[np.ndarray]
) -> None:
    vision = _Vision(
        [
            '{"accepted":false,"reason":"The mug opening faces up."}',
            '{"axis":"y","angle_degrees":90}',
            '{"axis":"x","angle_degrees":90}',
            '{"accepted":false,"reason":"The mug is on its side."}',
            '{"axis":"z","angle_degrees":180}',
            '{"accepted":true,"reason":"The mug rests upside down on its rim."}',
        ]
    )

    rotation = asset_posture.review_asset_posture(
        glb_path=tmp_path / "mug.glb",
        description="Mug",
        pose_description="Place upside down.",
        vlm_client=vision,
        debug_output_root=tmp_path,
    )

    assert rotation is not None
    np.testing.assert_allclose(rotation, np.diag([-1, -1, 1]), atol=1e-12)
    assert len(rendered_rotations) == 3
    np.testing.assert_allclose(rendered_rotations[0], np.eye(3))
    np.testing.assert_allclose(rendered_rotations[-1], rotation)
    assert vision.prompts[-2]["tried_rotations"] == [["x", 90]]
    assert "on its side" in str(vision.prompts[-2]["latest_review"])


@pytest.mark.parametrize("accepted", [True, False])
def test_review_keeps_identity_or_returns_no_rejected_rotation(
    accepted: bool, tmp_path: Path, rendered_rotations: list[np.ndarray]
) -> None:
    verdict = json.dumps({"accepted": accepted, "reason": "Reviewed posture."})
    responses = [verdict]
    if not accepted:
        responses.extend(
            [
                '{"axis":"x","angle_degrees":90}',
                verdict,
                '{"axis":"z","angle_degrees":90}',
                verdict,
                '{"axis":"x","angle_degrees":180}',
                verdict,
            ]
        )

    rotation = asset_posture.review_asset_posture(
        glb_path=tmp_path / "asset.glb",
        description="A solid object.",
        pose_description=None,
        vlm_client=_Vision(responses),
        debug_output_root=tmp_path,
    )

    if accepted:
        np.testing.assert_allclose(rotation, np.eye(3))
        assert len(rendered_rotations) == 1
    else:
        assert rotation is None
        assert len(rendered_rotations) == 4
