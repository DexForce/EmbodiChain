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
    _image_segmentation_utils as segmentation,
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
    leaders: list[tuple[tuple[int, int], tuple[int, int]]] = []
    badges: list[tuple[int, int, int, int]] = []
    current_badge_color: tuple[int, int, int] | None = None
    original_text = ImageDraw.ImageDraw.text
    original_rectangle = ImageDraw.ImageDraw.rounded_rectangle
    original_line = ImageDraw.ImageDraw.line

    def record_rectangle(self, xy, radius=0, fill=None, **kwargs):
        nonlocal current_badge_color
        assert xy[0] >= image_size[0]
        badges.append(xy)
        current_badge_color = fill
        return original_rectangle(self, xy, radius=radius, fill=fill, **kwargs)

    def record_line(self, xy, **kwargs):
        leaders.append(xy)
        return original_line(self, xy, **kwargs)

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
    monkeypatch.setattr(ImageDraw.ImageDraw, "line", record_line)
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
    assert leaders
    assert all(start[0] < image_size[0] <= end[0] for start, end in leaders)
    assert len(badges) == len(candidate_ids)
    assert all(
        first[3] <= second[1] or second[3] <= first[1]
        for position, first in enumerate(badges)
        for second in badges[position + 1 :]
    )


@pytest.mark.parametrize(
    ("image_size", "number_size", "legend_size"),
    [((320, 240), 24, 18), ((1920, 1080), 40, 28)],
)
def test_scene_overview_keeps_full_ids_and_names_outside_the_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_size: tuple[int, int],
    number_size: int,
    legend_size: int,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", image_size, "black").save(source)
    table, asset = tmp_path / "table.png", tmp_path / "asset.png"
    table_mask = Image.new("L", image_size)
    table_mask.paste(
        255, (10, image_size[1] // 2, image_size[0] - 10, image_size[1] - 10)
    )
    table_mask.save(table)
    asset_mask = Image.new("L", image_size)
    asset_mask.putpixel((40, 60), 255)
    asset_mask.save(asset)
    long_id = "opaque_task_entity_" + "x" * 65
    texts: list[tuple[tuple[int, int], str]] = []
    number_fonts: list[int] = []
    legend_fonts: list[int] = []
    glyph_bounds: list[tuple[int, int, int, int]] = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(self, xy, text, **kwargs):
        texts.append((xy, str(text)))
        (number_fonts if str(text).isdecimal() else legend_fonts).append(
            kwargs["font"].size
        )
        glyph_bounds.append(self.textbbox(xy, text, font=kwargs["font"]))
        return original_text(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    result = segmentation.render_asset_mask_id_overlay(
        image_path=source,
        asset_masks=[("table", table), (long_id, asset)],
        asset_names={"table": "workbench", long_id: "metal_can"},
        output_path=tmp_path / "scene_overview.png",
    )
    with Image.open(result) as image:
        assert image.width > image_size[0]
        assert image.height >= image_size[1]
        # Both the broad table and a one-pixel asset remain visible at their
        # original coordinates; neither is downsampled to fit the legend.
        assert image.getpixel((8, image_size[1] // 2 + 20)) != (0, 0, 0)
        assert image.getpixel((40, 60)) != (0, 0, 0)
        assert all(
            0 <= left < right <= image.width and 0 <= top < bottom <= image.height
            for left, top, right, bottom in glyph_bounds
        )
    assert all(position[0] >= image_size[0] for position, _ in texts)
    rendered_text = "".join(text for _, text in texts)
    assert long_id in rendered_text
    assert "workbench" in rendered_text
    assert "metal_can" in rendered_text
    assert set(number_fonts) == {number_size}
    assert set(legend_fonts) == {legend_size}


@pytest.mark.parametrize(
    ("image_size", "number_size"),
    [((320, 240), 24), ((1920, 1080), 34)],
)
def test_candidate_typography_uses_the_rendered_overview_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_size: tuple[int, int],
    number_size: int,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", image_size, "white").save(source)
    mask = Image.new("L", image_size)
    mask.paste(255, (40, 40, 80, 100))
    fonts = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(self, xy, text, **kwargs):
        if str(text).isdecimal():
            fonts.append(kwargs["font"].size)
        return original_text(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    segmentation.render_numbered_mask_candidates(
        image_path=source,
        candidates=[_candidate(7, mask)],
        output_path=tmp_path / "candidate.png",
        include_candidate_details=True,
    )
    # The 1920px source becomes a 960px overview. Source-based sizing would
    # incorrectly choose 40px, while the rendered reference calls for 34px.
    assert fonts == [number_size]


def test_dense_legend_rows_preserve_long_ids_without_overlap_or_clipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_size = (320, 240)
    source = tmp_path / "source.png"
    Image.new("RGB", image_size, "black").save(source)
    asset_masks = []
    for index in range(12):
        mask = Image.new("L", image_size)
        mask.paste(255, (10 + 25 * index, 100, 20 + 25 * index, 110))
        mask_path = tmp_path / f"mask_{index}.png"
        mask.save(mask_path)
        asset_masks.append((f"task_entity_{index:02d}_" + "x" * 60, mask_path))
    badges, texts, glyph_bounds = [], [], []
    original_rectangle = ImageDraw.ImageDraw.rounded_rectangle
    original_text = ImageDraw.ImageDraw.text

    def record_rectangle(self, xy, **kwargs):
        badges.append(xy)
        return original_rectangle(self, xy, **kwargs)

    def record_text(self, xy, text, **kwargs):
        texts.append(str(text))
        glyph_bounds.append(self.textbbox(xy, text, font=kwargs["font"]))
        return original_text(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "rounded_rectangle", record_rectangle)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    result = segmentation.render_asset_mask_id_overlay(
        image_path=source,
        asset_masks=asset_masks,
        output_path=tmp_path / "dense.png",
    )
    with Image.open(result) as image:
        assert image.height > image_size[1]
        assert all(
            image_size[0] <= left < right <= image.width
            and 0 <= top < bottom <= image.height
            for left, top, right, bottom in glyph_bounds
        )
    assert len(badges) == len(asset_masks)
    assert all(first[3] < second[1] for first, second in zip(badges, badges[1:]))
    assert all(asset_id in "".join(texts) for asset_id, _ in asset_masks)


def test_empty_candidate_sheet_keeps_the_source_for_visibility_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    source_color = (37, 91, 143)
    image_size = (240, 180)
    Image.new("RGB", image_size, source_color).save(source)
    result = segmentation.render_numbered_mask_candidates(
        image_path=source,
        candidates=[],
        output_path=tmp_path / "empty.png",
        include_candidate_details=True,
    )
    with Image.open(result) as image:
        # The header may grow, but the image remains intact underneath it.
        source_region = image.crop(
            (0, image.height - image_size[1], image_size[0], image.height)
        )
        assert (
            source_region.tobytes()
            == Image.new("RGB", image_size, source_color).tobytes()
        )


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
