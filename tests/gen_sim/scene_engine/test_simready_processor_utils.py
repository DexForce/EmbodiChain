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
import trimesh

from embodichain.gen_sim.scene_engine.core.scene import Scene, SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    DEFAULT_NEEDED_LAYOUT,
    compute_uniform_xy_scale_for_target,
    query_vlm_pose_switch_candidate,
    query_vlm_object_pose_and_target_size,
)


def test_simready_pose_layout_uses_graph_pose_descriptions(
    tmp_path: Path,
) -> None:
    class VLM:
        def complete(self, **_: object) -> str:
            raise AssertionError("This selection test must not call the VLM.")

    processor = SimReadyProcessor(
        scene=Scene(),
        coarse_layout_by_id={},
        coarse_geometry_root=tmp_path / "coarse",
        simready_geometry_root=tmp_path / "simready",
        config=SimReadyProcessorConfig(
            pose_descriptions_by_id={
                "bottle_001": "Stand upright on its base.",
                "fork_001": "Lie flat on the support surface.",
                "knife_001": None,
            },
        ),
        vlm_client=VLM(),  # type: ignore[arg-type]
    )

    assert (
        processor._needed_layout_for_object("bottle_001")
        == "Stand upright on its base."
    )
    assert (
        processor._needed_layout_for_object("fork_001")
        == "Lie flat on the support surface."
    )
    assert processor._needed_layout_for_object("knife_001") == DEFAULT_NEEDED_LAYOUT


def test_vlm_transform_query_retries_an_empty_response(tmp_path: Path) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = [
                "",
                '{"pose_action": "keep_current", "reason": "already upright", '
                '"target_xy_size_cm": [8.0, 8.0]}',
            ]

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    vlm_client = VLM()
    decision = query_vlm_object_pose_and_target_size(
        scene_object_description="small blue bottle",
        needed_layout="Stand upright on its base.",
        rendered_views_path=tmp_path / "views.png",
        vlm_client=vlm_client,  # type: ignore[arg-type]
    )

    assert decision == {
        "pose_action": "keep_current",
        "reason": "already upright",
        "target_xy_size_cm": [8.0, 8.0],
    }
    assert vlm_client.responses == []


def test_vlm_pose_candidate_query_returns_selected_rotation(tmp_path: Path) -> None:
    class VLM:
        def complete(self, **_: object) -> str:
            return '{"selected_candidate": "b", "reason": "pan opening faces up"}'

    selected_rotation_degrees, reason = query_vlm_pose_switch_candidate(
        scene_object_description="small saucepan",
        needed_layout="Rest stably with the cooking surface facing upward.",
        rendered_candidates_path=tmp_path / "candidates.png",
        vlm_client=VLM(),  # type: ignore[arg-type]
    )

    assert selected_rotation_degrees == -90.0
    assert reason == "pan opening faces up"


def test_simready_pose_switch_uses_the_vlm_selected_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_object = SceneObject(
        id="pan_001",
        kind="asset",
        category="pan",
        name="pan",
        description="small saucepan",
    )
    processor = SimReadyProcessor(
        scene=Scene(objects=[scene_object]),
        coarse_layout_by_id={"pan_001": {}},
        coarse_geometry_root=tmp_path / "coarse",
        simready_geometry_root=tmp_path / "simready",
        config=SimReadyProcessorConfig(
            pose_descriptions_by_id={
                "pan_001": "Rest stably with the cooking surface facing upward."
            }
        ),
        vlm_client=object(),  # type: ignore[arg-type]
    )
    calls: dict[str, object] = {}

    def fake_render_candidates(**kwargs: object) -> Path:
        calls["candidate_render"] = kwargs
        return tmp_path / "candidates.png"

    def fake_query_candidate(**kwargs: object) -> tuple[float, str]:
        calls["candidate_query"] = kwargs
        return -90.0, "opening up"

    def fake_rotate(**kwargs: object) -> Path:
        calls["rotate"] = kwargs
        return tmp_path / "rotated.glb"

    monkeypatch.setattr(
        processor,
        "_vlm_transform_for_object",
        lambda *_args, **_kwargs: {
            "pose_action": "rotate_to_required_pose",
            "reason": "current pan is upright",
            "target_xy_size_cm": [20.0, 20.0],
        },
    )
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor.render_object_pose_switch_candidates",
        fake_render_candidates,
    )
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor.query_vlm_pose_switch_candidate",
        fake_query_candidate,
    )
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor.rotate_glb_about_x_axis",
        fake_rotate,
    )

    rotated_path, vlm_scale = processor._prepare_vlm_rotated_glb(scene_object)

    assert rotated_path == tmp_path / "rotated.glb"
    assert vlm_scale is None
    assert calls["candidate_render"] == {
        "glb_path": tmp_path / "coarse" / "pan_001.glb",
        "output_path": tmp_path / "debug" / "vlm_pose_candidates" / "pan_001.png",
    }
    assert calls["candidate_query"] == {
        "scene_object_description": "small saucepan",
        "needed_layout": "Rest stably with the cooking surface facing upward.",
        "rendered_candidates_path": tmp_path / "candidates.png",
        "vlm_client": processor.vlm_client,
        "debug_output_path": tmp_path
        / "debug"
        / "vlm_pose_candidates"
        / "pan_001.json",
    }
    assert calls["rotate"] == {
        "input_path": tmp_path / "coarse" / "pan_001.glb",
        "output_path": tmp_path / "simready" / "vlm_rotated" / "pan_001.glb",
        "rotation_degrees": -90.0,
    }


def test_simready_null_pose_description_checks_the_default_stable_pose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_object = SceneObject(
        id="book_001",
        kind="asset",
        category="book",
        name="book",
        description="small book",
    )
    processor = SimReadyProcessor(
        scene=Scene(objects=[scene_object]),
        coarse_layout_by_id={"book_001": {}},
        coarse_geometry_root=tmp_path / "coarse",
        simready_geometry_root=tmp_path / "simready",
        config=SimReadyProcessorConfig(pose_descriptions_by_id={"book_001": None}),
        vlm_client=object(),  # type: ignore[arg-type]
    )
    requested_layouts: list[str] = []

    def fake_transform(*_args: object, needed_layout: str) -> dict[str, object]:
        requested_layouts.append(needed_layout)
        return {
            "pose_action": "keep_current",
            "reason": "book is already flat",
            "target_xy_size_cm": [20.0, 15.0],
        }

    monkeypatch.setattr(processor, "_vlm_transform_for_object", fake_transform)
    monkeypatch.setattr(
        "embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor.rotate_glb_about_x_axis",
        lambda **_kwargs: tmp_path / "rotated.glb",
    )

    processor._prepare_vlm_rotated_glb(scene_object)

    assert requested_layouts == [DEFAULT_NEEDED_LAYOUT]


def test_uniform_scale_uses_the_z_up_tabletop_footprint(tmp_path: Path) -> None:
    """Measure y-up GLBs against the VLM's z-up XY target footprint."""
    glb_path = tmp_path / "flat_fork.glb"
    # In y-up, the thin vertical axis is y; in z-up it becomes the z axis.
    trimesh.creation.box(extents=[2.0, 0.01, 0.5]).export(glb_path)

    scale = compute_uniform_xy_scale_for_target(
        glb_path=glb_path,
        target_xy_size_cm=[200.0, 50.0],
        rotate_about_x=False,
    )

    assert scale == pytest.approx(1.0)
