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

"""One declared fixed grasp qualified by the shared complete candidate path."""

from __future__ import annotations
from dataclasses import replace
import torch
from embodichain.lab.sim.motion.search import PlanningRandomState
from ._pick_recovery import _qualify_candidate

__all__: list[str] = []


def plan_fixed_grasp(self, request, context, selection):
    """Qualify one declared grasp as the main route, without resampling."""
    from embodichain.lab.sim.atomic_actions.goals import _resolve_object_pose

    state = PlanningRandomState.capture((self.device, self.robot.device))
    record = {}
    candidate = None
    try:
        if (
            self.planning_services.free_motion_query is None
            or selection.closing_query is None
        ):
            raise ValueError("Fixed Pick requires scene and native closing screens.")
        object_pose = _resolve_object_pose(
            request.goal.semantics, context, name="pickup_object_pose"
        ).to(device=self.device, dtype=context.robot.qpos.dtype)
        pose = object_pose @ request.skill_options.fixed_object_to_eef.to(object_pose)
        direction = request.skill_options.approach_direction.to(pose)
        direction = (direction / torch.linalg.vector_norm(direction))[None]
        selection.object_pose = object_pose
        record["selected_pose"] = pose.detach().cpu().tolist()
        record["approach_direction"] = direction.detach().cpu().tolist()
        if self._approach_alignment_mask(
            pose[:, None, None], request.skill_options, direction[:, None, None]
        ).all():
            candidate, qualification = _qualify_candidate(
                self, request, context, selection, pose, direction
            )
            record.update(qualification)
        else:
            record["reason"] = "approach_alignment_rejected"
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        record["unavailable"] = str(error)
    finally:
        state.restore()
    accepted = candidate is not None
    if not accepted:
        candidate = self.failed_plan(
            request,
            context,
            message="The declared fixed Pick was not qualified.",
            failure_code="gen_sim_fixed_pick_unqualified",
            retryable=False,
        )
    return replace(
        candidate,
        diagnostics=replace(
            candidate.diagnostics,
            metadata={
                **candidate.diagnostics.metadata,
                "gen_sim_fixed_pick": {
                    "accepted": accepted,
                    "fixed_object_to_eef": request.skill_options.fixed_object_to_eef.detach()
                    .cpu()
                    .tolist(),
                    **record,
                    "resampled": False,
                    "physical_evidence": False,
                },
            },
        ),
    )
