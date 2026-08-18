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
    compute_uniform_xy_scale_for_target,
)


def test_simready_long_axis_standardization_uses_graph_selected_ids(
    tmp_path: Path,
) -> None:
    processor = SimReadyProcessor(
        scene=Scene(),
        coarse_layout_by_id={},
        coarse_geometry_root=tmp_path / "coarse",
        simready_geometry_root=tmp_path / "simready",
        config=SimReadyProcessorConfig(
            long_axis_object_ids=frozenset({"rolling_pin_001"}),
        ),
    )

    assert processor._requires_long_axis_standardization("rolling_pin_001")
    assert not processor._requires_long_axis_standardization("bottle_001")


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
