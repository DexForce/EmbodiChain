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

"""Terminal runtime fixture for tests of the production Task Program bridge."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import torch

from embodichain.lab.sim.atomic_skills import TaskState
from embodichain.lab.task_program.runtime.results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)


def completed_runtime(env_ids: torch.Tensor) -> Mock:
    """Record start arguments and complete without issuing motion commands."""
    selected = torch.ones(env_ids.numel(), dtype=torch.bool, device=env_ids.device)
    runtime = Mock(
        spec=[
            "result",
            "status",
            "start",
            "step",
            "cancel",
            "adopt_verified_task_state",
        ]
    )
    runtime.status = SemanticExecutionStatus.IDLE
    runtime.result = SemanticExecutionResult(
        status=runtime.status,
        workflow_id=None,
        current_call_index=None,
        env_ids=env_ids,
        success_mask=torch.zeros_like(selected),
        failure_mask=torch.zeros_like(selected),
        cancelled_mask=torch.zeros_like(selected),
        eligible_mask=selected,
        task_state=TaskState.empty(env_ids.numel(), env_ids.device),
    )

    def start(
        *calls,
        workflow_id="semantic_workflow",
        eligible_mask=None,
        execution_prefix_length=None,
    ):
        mask = selected.clone() if eligible_mask is None else eligible_mask.clone()
        runtime.status = SemanticExecutionStatus.COMPLETED
        runtime.result = replace(
            runtime.result,
            status=runtime.status,
            workflow_id=workflow_id,
            success_mask=mask,
            eligible_mask=mask,
        )
        return runtime.result

    runtime.start.side_effect = start
    runtime.step.side_effect = AssertionError("A terminal runtime must not be stepped.")
    runtime.cancel.side_effect = AssertionError(
        "A completed runtime cannot be cancelled."
    )
    runtime.adopt_verified_task_state.side_effect = lambda state: runtime.result
    return runtime
