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
import torch

from embodichain.lab.sim.motion.expansion import (
    MinimumJerkBridgePlanner,
    TrajectoryBridge,
    TrajectoryPhase,
    TrajectorySegmentBank,
    TrajectoryTemplate,
    compare_trajectory_segments,
    extract_trajectory_segments,
    stitch_trajectory_segments,
)

_CONTROL_DT = 0.1


def _template(
    template_id: str,
    *,
    offset: float = 0.0,
    second_phase_start: float = 1.0,
) -> TrajectoryTemplate:
    positions = torch.tensor(
        [
            [offset],
            [offset + 0.5],
            [offset + 1.0],
            [second_phase_start],
            [second_phase_start + 0.5],
            [second_phase_start + 1.0],
        ],
        dtype=torch.float64,
    )
    dt = torch.full((6,), _CONTROL_DT, dtype=torch.float64)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id=f"source_{template_id}",
        source_revision="one",
        template_id=template_id,
        joint_names=("joint",),
        positions=positions,
        dt=dt,
        phases=(
            TrajectoryPhase("reach", 0, 3, "free", ("joint_residual",)),
            TrajectoryPhase("retreat", 3, 6, "free", ("via_points",)),
        ),
        controlled_joint_indices=(0,),
        validator_id="path_and_task",
    )


def test_segment_bank_filters_by_phase_contract_and_selects_nearest_start() -> None:
    reference = extract_trajectory_segments(_template("reference"))[0]
    near = extract_trajectory_segments(_template("near", offset=0.1))[0]
    far = extract_trajectory_segments(_template("far", offset=2.0))[0]
    wrong_phase = replace(near, segment_id="wrong", phase_id="retreat")
    bank = TrajectorySegmentBank((far, wrong_phase, near))

    assert bank.candidates(reference) == (far, near)
    selected = bank.select(
        reference,
        strategy="nearest_start",
        current_position=torch.tensor([0.05], dtype=torch.float64),
    )
    assert selected.segment_id == near.segment_id
    assert compare_trajectory_segments(reference, wrong_phase).reasons == ("phase_id",)


def test_segment_identity_includes_the_full_source_identity() -> None:
    first = _template("shared")
    second = replace(first, source_id="another_source")
    first_segment = extract_trajectory_segments(first)[0]
    second_segment = extract_trajectory_segments(second)[0]

    bank = TrajectorySegmentBank((first_segment, second_segment))

    assert first_segment.segment_id != second_segment.segment_id
    assert len(bank) == 2


def test_stitch_exact_boundary_deduplicates_shared_sample_and_keeps_lineage() -> None:
    reach, retreat = extract_trajectory_segments(_template("nominal"))

    result = stitch_trajectory_segments(
        (reach, retreat),
        source_id="composite",
        source_revision="policy:one",
        template_id="stitched",
    )

    assert result.template.positions[:, 0].tolist() == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    assert result.bridge_phase_ids == ()
    assert [phase.phase_id for phase in result.template.phases] == [
        "segment_0:reach",
        "segment_1:retreat",
    ]
    assert result.placements[0].source_template_id == "nominal"
    assert result.placements[1].source_start_index == 3


def test_discontinuous_segments_require_and_record_a_minimum_jerk_bridge() -> None:
    reach = extract_trajectory_segments(_template("left"))[0]
    retreat = extract_trajectory_segments(_template("right", second_phase_start=1.4))[1]
    planner = MinimumJerkBridgePlanner(
        control_dt=_CONTROL_DT,
        velocity_limits=torch.tensor([2.0], dtype=torch.float64),
        acceleration_limits=torch.tensor([10.0], dtype=torch.float64),
        minimum_sample_count=3,
    )

    result = stitch_trajectory_segments(
        (reach, retreat),
        source_id="composite",
        source_revision="policy:one",
        template_id="with_bridge",
        bridge_planner=planner,
    )

    assert result.bridge_phase_ids == ("bridge_0_1",)
    bridge_phase = next(
        phase
        for phase in result.template.phases
        if phase.phase_id == result.bridge_phase_ids[0]
    )
    assert result.template.positions[bridge_phase.stop_index - 1, 0] == pytest.approx(
        1.4
    )
    assert result.template.positions[-1, 0] == pytest.approx(2.4)
    assert torch.all(result.template.dt[1:] > 0)


def test_discontinuous_segments_fail_without_implicit_interpolation() -> None:
    reach = extract_trajectory_segments(_template("left"))[0]
    retreat = extract_trajectory_segments(_template("right", second_phase_start=1.4))[1]

    with pytest.raises(ValueError, match="explicit bridge planner"):
        stitch_trajectory_segments(
            (reach, retreat),
            source_id="composite",
            source_revision="policy:one",
            template_id="missing_bridge",
        )


def test_stitch_rejects_a_bridge_with_a_different_tensor_contract() -> None:
    reach = extract_trajectory_segments(_template("left"))[0]
    retreat = extract_trajectory_segments(_template("right", second_phase_start=1.4))[1]

    class FloatBridgePlanner:
        def plan(
            self, start_position: torch.Tensor, goal_position: torch.Tensor
        ) -> TrajectoryBridge:
            positions = torch.stack((start_position, goal_position)).float()
            return TrajectoryBridge(positions, torch.tensor([0.0, _CONTROL_DT]))

    with pytest.raises(ValueError, match="share dtype/device"):
        stitch_trajectory_segments(
            (reach, retreat),
            source_id="composite",
            source_revision="policy:one",
            template_id="wrong_bridge_dtype",
            bridge_planner=FloatBridgePlanner(),
        )


def test_segment_extraction_rejects_single_sample_phases() -> None:
    source = _template("short")
    source = replace(
        source,
        phases=(
            TrajectoryPhase("instant", 0, 1),
            TrajectoryPhase("rest", 1, 6),
        ),
    )

    with pytest.raises(ValueError, match="at least two samples"):
        extract_trajectory_segments(source)
