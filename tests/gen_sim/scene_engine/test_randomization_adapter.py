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

from dataclasses import replace

import pytest

from embodichain.gen_sim.scene_engine.randomization.adapter import (
    convert_layouts,
    randomize_scene_export,
)
from embodichain.gen_sim.scene_engine.randomization.generation import randomize_scene
from embodichain.gen_sim.scene_engine.randomization.loading import load_layouts
from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts
from .test_randomization_baseline import _export


def test_export_adapter_generates_source_independent_layouts(tmp_path) -> None:
    export, _ = _export(tmp_path)
    workspace = export.parent
    original = (workspace / "scene_export/scene_config.json").read_bytes()
    run = randomize_scene_export(
        workspace,
        {
            "candidate_count": 3,
            "max_attempts": 200,
            "policy": {"group_targets": ["bottle"]},
        },
    )
    loaded = load_layouts(run / "layouts")
    assert len(loaded.variants) == 3
    assert (workspace / "scene_export/scene_config.json").read_bytes() == original
    assert all(
        {change.entity_uid for change in variant.pose_changes} == {"bottle"}
        for variant in loaded.variants
    )


@pytest.mark.parametrize("backend,storage", [("slsqp", "json"), ("batched", "packed")])
def test_legacy_conversion_preserves_every_identity(
    tmp_path, monkeypatch, backend, storage
) -> None:
    export, cfg = _export(tmp_path)
    workspace = export.parent
    original = randomize_scene(replace(cfg, backend=backend), candidate_count=3)
    run = save_layouts(original, output_root=workspace, layout_format=storage)
    import embodichain.gen_sim.scene_engine.randomization.generation as generation

    monkeypatch.setattr(
        generation,
        "randomize_scene",
        lambda *args, **kwargs: pytest.fail("Conversion resampled"),
    )
    converted = convert_layouts(run / "layouts", tmp_path / "converted")
    replay = load_layouts(converted / "layouts")
    assert [variant.variant_id for variant in replay.variants] == [
        candidate.variant.variant_id for candidate in original.candidates
    ]


def test_generate_and_edit_trigger_one_post_export_randomization(
    tmp_path, monkeypatch
) -> None:
    from embodichain.gen_sim.scene_engine.cli import start
    import embodichain.gen_sim.scene_engine.randomization.adapter as adapter

    events = []
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(
        start, "generate_scene_from_image", lambda **kwargs: events.append("generate")
    )
    monkeypatch.setattr(start, "edit_scene", lambda **kwargs: events.append("edit"))
    monkeypatch.setattr(
        adapter, "randomize_scene_export", lambda *args: events.append("randomize")
    )
    start.cli_scene_engine(
        image, tmp_path, edit_prompt="move", scene_randomization={"seed": 7}
    )
    assert events == ["generate", "edit", "randomize"]
