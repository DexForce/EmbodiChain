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

"""Control-grid retiming protects contact events and derivative stencils."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.trajectory import differentiate_positions
from embodichain.lab.sim.motion.expansion import (
    TrajectoryPhase,
    TrajectoryTemplate,
    retime_with_time_map,
)


def _template() -> TrajectoryTemplate:
    q = torch.zeros(18, 2, dtype=torch.float64)
    q[:14, 0] = torch.linspace(0, 1, 14, dtype=q.dtype)
    q[14:, 0] = 1
    dt = torch.full((18,), 0.1, dtype=q.dtype)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id="source",
        source_revision="1",
        template_id="unit",
        joint_names=("arm", "hand"),
        positions=q,
        dt=dt,
        phases=(
            TrajectoryPhase("approach", 0, 14, "free", ("retime",)),
            TrajectoryPhase("close", 14, 18, "contact"),
        ),
        allowed_operators=("retime",),
        controlled_joint_indices=(0,),
    )


def test_mapped_retime_preserves_contact_and_its_derivative_neighbors() -> None:
    source = _template()
    result = retime_with_time_map(
        source, duration_scale=1.5, control_dt=0.1, protected_indices=(5,)
    )
    assert result.template.positions.shape[0] > source.positions.shape[0]
    assert (
        result.template.phases[0].stop_index - result.template.phases[0].start_index - 1
        == 20
    )
    mapping = dict(
        zip(
            result.time_map.source_event_indices,
            result.time_map.execution_event_indices,
        )
    )
    for index in (4, 5, 6, 13, 14, 15, 16, 17):
        assert torch.equal(
            result.template.positions[mapping[index]], source.positions[index]
        )
    old_velocity = differentiate_positions(source.positions[None], source.dt[None])[0]
    new_velocity = differentiate_positions(
        result.template.positions[None], result.template.dt[None]
    )[0]
    for index in (5, 13, 14, 15, 16, 17):
        assert torch.allclose(new_velocity[mapping[index]], old_velocity[index])
    assert (
        result.template.phases[1].stop_index - result.template.phases[1].start_index
        == 4
    )


def test_mapped_retime_rejects_changing_uncontrolled_joint() -> None:
    source = _template()
    positions = source.positions.clone()
    positions[6, 1] = 0.2
    from dataclasses import replace

    with pytest.raises(ValueError, match="uncontrolled joint"):
        retime_with_time_map(
            replace(source, positions=positions), duration_scale=1.5, control_dt=0.1
        )


def test_mapped_retime_rejects_unrepresentable_short_phase_duration() -> None:
    with pytest.raises(ValueError, match="cannot preserve protected stencils"):
        retime_with_time_map(_template(), duration_scale=0.1, control_dt=0.1)
    with pytest.raises(ValueError, match="sample budget"):
        retime_with_time_map(_template(), duration_scale=1e300, control_dt=0.1)
