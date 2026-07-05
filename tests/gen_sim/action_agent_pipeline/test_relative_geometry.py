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

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _RelativePlacementSpec,
    _RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_geometry import (
    _with_coordinated_side_release_height_offsets,
)


def test_coordinated_side_release_height_preserves_moved_object_world_z() -> None:
    placement = _RelativePlacementStepSpec(
        intent="coordinated_pickment",
        moved_source_uid="tray_src",
        reference_source_uid="table_src",
        moved_runtime_uid="tray",
        reference_runtime_uid="table",
        relation="right_of",
        active_side="left",
        release_offset=[0.0, -0.16, 0.12],
        high_offset=[0.0, -0.16, 0.22],
    )
    spec = _RelativePlacementSpec(
        intent="coordinated_pickment",
        table_source_uid="table_src",
        moved_source_uid="tray_src",
        reference_source_uid="table_src",
        moved_runtime_uid="tray",
        reference_runtime_uid="table",
        relation="right_of",
        active_side="left",
        task_description="move tray right",
        task_prompt_summary="Move tray right.",
        basic_background_notes="",
        action_sketch=[],
        release_offset=placement.release_offset,
        high_offset=placement.high_offset,
        placements=(placement,),
    )
    gym_config = {
        "background": [
            {"uid": "table", "init_pos": [0.0, 0.0, 0.0]},
        ],
        "rigid_object": [
            {"uid": "tray", "init_pos": [0.0, 0.0, 0.58826]},
        ],
    }

    updated = _with_coordinated_side_release_height_offsets(spec, gym_config)

    assert updated.release_offset == pytest.approx([0.0, -0.16, 0.58826])
    assert updated.high_offset == pytest.approx([0.0, -0.16, 0.68826])
    assert updated.placements[0].release_offset == pytest.approx([0.0, -0.16, 0.58826])
