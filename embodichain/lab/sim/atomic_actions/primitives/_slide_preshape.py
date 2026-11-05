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

"""Bounded aperture-only Slide retry around the complete engine plan boundary."""

from __future__ import annotations
from dataclasses import replace
import math
from embodichain.lab.sim.motion.search import PlanningRandomState
from ..articulation_continuation import SlidePlanningRecovery

__all__: list[str] = []


def plan_complete_preshape(action, request, context, *, build, external_nominal):
    rule = request.recovery_policy.planning_recovery
    enabled = (
        isinstance(rule, SlidePlanningRecovery) and rule.preshape_aperture_recovery
    )
    if not request.skill_options.qualify_hand_clearance:
        if enabled:
            raise ValueError(
                "Aperture recovery requires qualified hand/table grasp filtering."
            )
        return build(request)
    if (
        external_nominal
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or request.skill_options.preshape_fraction != 0.85
        or request.motion_policy.strategy != "ik_interp"
    ):
        raise ValueError(
            "Source preshape filtering requires B1/env0 nominal85% ik_interp and no external nominal source."
        )
    provider = action.planning_services.slide_hand_clearance_provider
    if provider is None:
        raise ValueError(
            "Slide hand clearance requires its explicit geometry provider."
        )
    nominal = provider.nominal(request, context)
    facts = []
    before = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    original = build(
        request,
        planner=lambda selected, current: action._plan_core(
            selected, current, clearance=nominal, candidate_facts=facts
        ),
    )
    required = min((value for value, _ in facts), default=math.inf)
    width_fit = any(fit for _, fit in facts)
    if (
        not enabled
        or original.plan_success.any()
        or width_fit
        or not math.isfinite(required)
    ):
        return original
    after = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    accepted = False
    measured = []
    try:
        for percent in range(84, 74, -1):
            fraction = percent / 100
            clearance = provider.measure(request, context, fraction)
            measured.append(
                {"fraction": fraction, "usable_aperture_m": clearance.aperture}
            )
            if clearance.aperture < required:
                continue
            selected = replace(
                request,
                skill_options=replace(
                    request.skill_options, preshape_fraction=fraction
                ),
            )
            before.restore()
            candidate = build(
                selected,
                planner=lambda chosen, current: action._plan_core(
                    chosen, current, clearance=clearance, candidate_facts=[]
                ),
            )
            viable = bool(candidate.plan_success.all())
            plan = candidate if viable else original
            accepted = viable and bool(plan.plan_success.all())
            return replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_e6_preshape": {
                            "nominal_fraction": 0.85,
                            "nominal_usable_aperture_m": nominal.aperture,
                            "required_width_m": required,
                            "selected_fraction": fraction,
                            "measurements": measured,
                            "accepted_plan": accepted,
                            "physical_success": None,
                            **(
                                {
                                    "rebuild_failure": {
                                        "code": getattr(
                                            candidate.diagnostics.failure, "code", None
                                        ),
                                        "messages": list(
                                            candidate.diagnostics.messages
                                        ),
                                        "metadata": dict(
                                            candidate.diagnostics.metadata
                                        ),
                                    }
                                }
                                if not viable
                                else {}
                            ),
                        },
                    },
                ),
            )
        return original
    finally:
        if not accepted:
            after.restore()
