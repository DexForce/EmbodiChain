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

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor import (
    SimReadyProcessor,
    SimReadyProcessorConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    DEFAULT_NEEDED_LAYOUT,
    LYING_NEEDED_LAYOUT,
    STANDING_NEEDED_LAYOUT,
    compute_uniform_xy_scale_for_target,
    query_vlm_object_rotation_and_target_size,
)


def test_simready_pose_layout_uses_graph_orientation_states(
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
            orientation_states_by_id={"bottle_001": "standing", "fork_001": "lying"},
        ),
        vlm_client=VLM(),  # type: ignore[arg-type]
    )

    assert processor._orientation_state_for_object("bottle_001") == "standing"
    assert processor._orientation_state_for_object("fork_001") == "lying"
    assert processor._orientation_state_for_object("knife_001") is None
    assert processor._needed_layout_for_object("bottle_001") == STANDING_NEEDED_LAYOUT
    assert processor._needed_layout_for_object("fork_001") == LYING_NEEDED_LAYOUT
    assert processor._needed_layout_for_object("knife_001") == DEFAULT_NEEDED_LAYOUT


def test_vlm_transform_query_retries_an_empty_response(tmp_path: Path) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = [
                "",
                '{"rotate_about_x": false, "target_xy_size_cm": [8.0, 8.0]}',
            ]

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    vlm_client = VLM()
    decision = query_vlm_object_rotation_and_target_size(
        scene_object_description="small blue bottle",
        needed_layout=STANDING_NEEDED_LAYOUT,
        rendered_views_path=tmp_path / "views.png",
        vlm_client=vlm_client,  # type: ignore[arg-type]
    )

    assert decision == {"rotate_about_x": False, "target_xy_size_cm": [8.0, 8.0]}
    assert vlm_client.responses == []


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
