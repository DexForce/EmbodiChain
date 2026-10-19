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

"""One aperture-only E6 retry with matched hand geometry and command options."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from .e6_clearance import HandClearance, PRESHAPE_FRACTION

__all__: list[str] = []
E6_PRESHAPE_REVISION = 1


@dataclass
class _Scope:
    provider: Any
    clearance: HandClearance
    invocation_id: str
    required_width: float = math.inf
    width_fit: bool = False


_SCOPE: ContextVar[_Scope | None] = ContextVar("gen_sim_e6_preshape", default=None)


def defer_initial_capture(invocation_id: str) -> bool:
    """Publish the selected plan, never an intermediate aperture rejection."""
    scope = _SCOPE.get()
    return scope is not None and scope.invocation_id == invocation_id


def filter_candidates(
    provider: Any,
    clearance: HandClearance,
    poses: torch.Tensor,
    widths: torch.Tensor,
    costs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observe only finite candidates, retaining the complete clearance filter."""
    scope = _SCOPE.get()
    if scope is not None and scope.provider is provider:
        clearance = scope.clearance
        valid = torch.isfinite(costs) & torch.isfinite(widths) & (widths >= 0)
        valid &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
        if valid.any():
            scope.required_width = min(scope.required_width, float(widths[valid].min()))
            scope.width_fit |= bool((widths[valid] <= clearance.aperture).any())
    return clearance.filter(poses, widths, costs)


def plan_with_preshape(
    engine: Any, request: Any, context: Any, build: Callable[[Any], Any]
) -> Any:
    """Keep nominal plans exact; widen only an aperture-rejected bound Slide.

    Measure at most five one-percent opening increments, within the calibrated
    80-85% closure band. Rebuild only once, with the smallest sufficient opening.
    Full hand/table screening and the original planner/runner remain authoritative.
    """
    from embodichain.lab.sim.atomic_actions import JointPositionTarget, SlideOptions
    from .grasp_filter import E6ApproachGraspPoseGenerator
    from .held_alignment_recovery import _RandomState, _cuda_devices
    from .planning_probe import initial_plan_capture

    if (
        request.skill_id != "slide"
        or getattr(engine, "_articulation_calls", {}).get(request.invocation_id)
        != "slide"
        or getattr(context, "batch_size", None) != 1
        or type(request.skill_options) is not SlideOptions
        or request.skill_options.preshape_fraction != PRESHAPE_FRACTION
        or request.motion_policy.strategy != "ik_interp"
    ):
        return build(request)
    hand = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    provider = engine.grasp_pose_generators.get(hand.target_id)
    if (
        not isinstance(provider, E6ApproachGraspPoseGenerator)
        or provider._clearance is None
        or provider._clearance_factory is None
    ):
        return build(request)
    devices = _cuda_devices(engine.robot.device, context.robot.qpos.device)
    before = _RandomState.capture(devices)
    scope = _Scope(provider, provider._clearance, request.invocation_id)
    capture = initial_plan_capture(request.invocation_id)

    def finish(selected: Any, plan: Any) -> Any:
        return plan if capture is None else capture.accept(selected, plan, engine.robot)

    token = _SCOPE.set(scope)
    try:
        original = build(request)
        if (
            original.plan_success.any()
            or scope.width_fit
            or not math.isfinite(scope.required_width)
        ):
            return finish(request, original)
        required_width = scope.required_width
        measured = []
        after_nominal = _RandomState.capture(devices)
        accepted = False
        try:
            for percent in range(84, 79, -1):
                fraction = percent / 100
                clearance = provider._clearance_factory(fraction)
                measured.append(
                    {"fraction": fraction, "usable_aperture_m": clearance.aperture}
                )
                if clearance.aperture < required_width:
                    continue
                scope.clearance = clearance
                selected = replace(
                    request,
                    skill_options=replace(
                        request.skill_options, preshape_fraction=fraction
                    ),
                )
                before.restore()
                candidate = build(selected)
                viable = bool(candidate.plan_success.all())
                plan = finish(
                    selected if viable else request, candidate if viable else original
                )
                accepted = viable and bool(plan.plan_success.all())
                return replace(
                    plan,
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_e6_preshape": {
                                "nominal_fraction": PRESHAPE_FRACTION,
                                "nominal_usable_aperture_m": provider._clearance.aperture,
                                "required_width_m": required_width,
                                "selected_fraction": fraction,
                                "measurements": measured,
                                "accepted_plan": accepted,
                                "physical_success": None,
                            },
                        },
                    ),
                )
            return finish(request, original)
        finally:
            if not accepted:
                after_nominal.restore()
    except Exception as error:
        if capture is not None:
            capture.failed(error)
        raise
    finally:
        _SCOPE.reset(token)
