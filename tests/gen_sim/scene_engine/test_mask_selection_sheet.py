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

from copy import deepcopy
from pathlib import Path
import re

from PIL import Image, ImageDraw
import pytest

from embodichain.gen_sim.scene_engine.pipeline.utils import (
    image_segmentation_utils as segmentation,
)


def _candidate(index: int, mask: Image.Image) -> segmentation.MaskCandidate:
    return segmentation.MaskCandidate(
        index=index,
        mask_rle=segmentation._encode_binary_mask_rle(mask),
    )


def test_cards_keep_exact_source_pixels_and_excluded_holes(tmp_path: Path) -> None:
    image_size = (120, 120)
    source_color = (37, 91, 143)
    source = tmp_path / "source.png"
    overview = tmp_path / "overview.png"
    Image.new("RGB", image_size, source_color).save(source)
    Image.new("RGB", image_size, (16, 16, 16)).save(overview)
    mask = Image.new("L", image_size)
    mask.paste(255, (30, 30, 90, 90))
    mask.paste(0, (50, 50, 70, 70))
    candidate = _candidate(8, mask)
    original_rle = deepcopy(candidate.mask_rle)
    original_image = source.read_bytes()

    path = segmentation.render_numbered_mask_candidates(
        image_path=overview,
        detail_image_path=source,
        candidates=[candidate],
        output_path=tmp_path / "selection.png",
        include_candidate_details=True,
    )
    with Image.open(path) as image:
        selected = [
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if image.getpixel((x, y)) == source_color
        ]
        # The 60x60 selected square has an excluded 20x20 central hole.
        assert len(selected) == 60 * 60 - 20 * 20
        left = min(x for x, _ in selected)
        top = min(y for _, y in selected)
        assert image.getpixel((left + 29, top + 29)) == (200, 203, 208)
        assert image.getpixel((left - 1, top + 29)) == (200, 203, 208)
        assert image.getpixel((left + 59, top + 59)) == source_color
    assert candidate.mask_rle == original_rle
    assert source.read_bytes() == original_image


def test_every_sparse_id_has_its_own_card_and_matching_overview_color(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    image_size = (480, 360)
    Image.new("RGB", image_size, "white").save(source)
    mask = Image.new("L", image_size)
    mask.paste(255, (80, 80, 400, 280))
    # Enough overlapping candidates to require multiple card columns; IDs are
    # intentionally sparse so positional renumbering would break the contract.
    candidate_ids = [2, 5, 9, 13, 16, 20, 24, 27, 31, 35, 39]
    candidates = [_candidate(index, mask) for index in candidate_ids]
    titles: dict[int, tuple[int, int, int]] = {}
    labels: dict[int, tuple[int, int, int]] = {}
    current_badge_color: tuple[int, int, int] | None = None
    original_text = ImageDraw.ImageDraw.text
    original_rectangle = ImageDraw.ImageDraw.rounded_rectangle

    def record_rectangle(self, xy, radius=0, fill=None, **kwargs):
        nonlocal current_badge_color
        current_badge_color = fill
        return original_rectangle(self, xy, radius=radius, fill=fill, **kwargs)

    def record_text(self, xy, text, fill=None, **kwargs):
        if str(text).isdecimal():
            assert current_badge_color is not None
            labels[int(text)] = current_badge_color
        else:
            title_id = re.search(r"\b(\d+)\b", str(text))
            if title_id:
                titles[int(title_id.group(1))] = fill
        return original_text(self, xy, text, fill=fill, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "rounded_rectangle", record_rectangle)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    segmentation.render_numbered_mask_candidates(
        image_path=source,
        candidates=candidates,
        output_path=tmp_path / "selection.png",
        include_candidate_details=True,
        mask_style="outline",
    )
    assert set(titles) == set(labels) == set(candidate_ids)
    assert titles == labels
    assert titles[2] != titles[5]
    assert [candidate.index for candidate in candidates] == candidate_ids


@pytest.mark.parametrize("indices", [[1, 1], [0], [True]])
def test_selection_sheet_rejects_ambiguous_candidate_ids(
    tmp_path: Path, indices: list[int]
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 80), "white").save(source)
    mask = Image.new("L", (80, 80), 255)
    with pytest.raises(ValueError, match="candidate IDs"):
        segmentation.render_numbered_mask_candidates(
            image_path=source,
            candidates=[_candidate(index, mask) for index in indices],
            output_path=tmp_path / "selection.png",
            include_candidate_details=True,
        )


def test_detail_source_must_match_mask_coordinates(tmp_path: Path) -> None:
    source, detail = tmp_path / "source.png", tmp_path / "detail.png"
    Image.new("RGB", (80, 80), "white").save(source)
    Image.new("RGB", (80, 100), "white").save(detail)
    mask = Image.new("L", (80, 80), 255)
    with pytest.raises(ValueError, match="size does not match"):
        segmentation.render_numbered_mask_candidates(
            image_path=source,
            detail_image_path=detail,
            candidates=[_candidate(1, mask)],
            output_path=tmp_path / "selection.png",
            include_candidate_details=True,
        )


def test_default_rendering_keeps_original_image_dimensions(tmp_path: Path) -> None:
    image_size = (240, 180)
    source = tmp_path / "source.png"
    Image.new("RGB", image_size, "white").save(source)
    mask = Image.new("L", image_size)
    mask.paste(255, (40, 40, 160, 140))
    result = segmentation.render_numbered_mask_candidates(
        image_path=source,
        candidates=[_candidate(1, mask)],
        output_path=tmp_path / "legacy.png",
    )
    with Image.open(result) as image:
        assert image.size == image_size
