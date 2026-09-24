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

"""Source-neutral Task Program generation integration tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    ActionPlanTemplateAdapter,
)
from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    SourceAdapter,
    SourceContext,
    TrajectoryTemplate,
)
from embodichain.lab.task_program.integrations import TaskProgramSourceAdapter


def _trajectory_template() -> TrajectoryTemplate:
    return TrajectoryTemplate(
        source_id="program",
        source_revision="revision",
        template_id="place:0",
        joint_names=("j0", "j1", "j2"),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        dt=torch.tensor([0.0, 0.1]),
    )


def test_task_program_source_adapter_delegates_to_atomic_adapter() -> None:
    atomic = ActionPlanTemplateAdapter(
        joint_names=("j0", "j1", "j2"),
        phase_permissions={"place": ("joint_residual",)},
        phase_kinds={"place": "free"},
    )
    adapter = TaskProgramSourceAdapter(atomic)
    context = SourceContext(
        "program",
        "revision",
        "place:0",
        SceneCase("case", "initial", "signature", "task", "robot"),
        0.1,
    )
    expected = _trajectory_template()
    source = Mock(spec=ActionPlan)

    assert adapter.kind == "task_program"
    assert isinstance(adapter, SourceAdapter)
    with patch.object(
        ActionPlanTemplateAdapter,
        "export_template",
        return_value=expected,
    ) as export_template:
        actual = adapter.export_template(source, context=context)

    assert actual is expected
    export_template.assert_called_once_with(source, context=context)
