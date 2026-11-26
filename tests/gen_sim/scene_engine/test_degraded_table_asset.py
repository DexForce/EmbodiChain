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

from itertools import groupby
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from embodichain.gen_sim.scene_engine.pipeline.utils.degraded_table_asset import (
    generate_degraded_table_asset,
)
from embodichain.gen_sim.services.contracts import (
    GeometryRequest,
    GeometryResult,
    ReconstructedObject,
)
from embodichain.gen_sim.services.errors import CapabilityError


def test_standalone_reconstruction_uses_complete_table_mask(tmp_path: Path) -> None:
    observed_path = tmp_path / "table_rgba.png"
    observed = Image.new("RGBA", (3, 1))
    observed.putdata([(147, 93, 45, 255), (130, 78, 31, 255), (255, 0, 0, 0)])
    observed.save(observed_path)

    table_mask = Image.new("L", (32, 32))
    draw = ImageDraw.Draw(table_mask)
    draw.rectangle((8, 9, 23, 13), fill=255)
    tabletop_only = table_mask.copy()
    draw.rectangle((8, 12, 10, 22), fill=255)
    draw.rectangle((21, 12, 23, 22), fill=255)

    class ImageClient:
        def generate_image_by_image(
            self, *, image_path, prompt, output_path, size, seed
        ) -> Path:
            assert image_path == tmp_path / "reference_table.png"
            with Image.open(image_path) as reference:
                assert reference.mode == "RGB"
                assert reference.getpixel((0, 0)) == (147, 93, 45)
                assert reference.getpixel((1, 0)) == (130, 78, 31)
                assert reference.getpixel((2, 0)) == (128, 128, 128)
            assert "matte brown" in prompt
            assert "1.50" in prompt
            assert "grain direction" in prompt
            assert (tmp_path / "prompt.txt").read_text(
                encoding="utf-8"
            ) == prompt + "\n"
            Image.new("RGB", table_mask.size).save(output_path)
            return output_path

    class SegmentationClient:
        def segment_single_object(self, *, image_path, prompt):
            assert image_path == tmp_path / "generated_table.png"
            assert prompt == "table"
            return [
                {
                    "size": [32, 32],
                    "counts": [len(list(run)) for _, run in groupby(mask.tobytes())],
                }
                for mask in (tabletop_only, table_mask)
            ]

    class GeometryClient:
        def reconstruct(self, request: GeometryRequest) -> GeometryResult:
            assert request.image_path == tmp_path / "generated_table.png"
            assert request.object_masks == (
                ("table", tmp_path / "generated_table_mask.png"),
            )
            with Image.open(request.object_masks[0][1]) as mask:
                assert mask.tobytes() == table_mask.tobytes()
            mesh_path = request.output_root / "table.glb"
            mesh_path.touch()
            return GeometryResult(
                (
                    ReconstructedObject(
                        "table",
                        mesh_path,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ),
                )
            )

    path = generate_degraded_table_asset(
        description="A rectangular matte brown surface.",
        support_width_depth=(1.2, 0.8),
        reference_image_path=observed_path,
        output_root=tmp_path,
        image_generation_client=ImageClient(),
        image_segmentation_client=SegmentationClient(),
        geometry_generation_client=GeometryClient(),
    )
    assert path == tmp_path / "geometry" / "table.glb"


def test_degraded_table_does_not_fall_back_to_text_generation(tmp_path: Path) -> None:
    class TextOnlyClient:
        def generate_image_by_prompt(self, **kwargs) -> Path:
            pytest.fail("Text generation cannot preserve the observed tabletop.")

    with pytest.raises(CapabilityError, match="image-to-image"):
        generate_degraded_table_asset(
            description="A textured wood surface.",
            support_width_depth=(1.0, 1.0),
            reference_image_path=tmp_path / "table_rgba.png",
            output_root=tmp_path,
            image_generation_client=TextOnlyClient(),
            image_segmentation_client=None,
            geometry_generation_client=None,
        )
