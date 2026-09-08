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

"""Task-owned stack placement identity with a distinct, declared option preset."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
from embodichain.lab.task_program.compiler.lowering import SemanticLowering

from .services import (
    _PickLowerer,
    _PickLowererFactory,
    _PickRoute,
    _RelativePlaceLowerer,
    _RelativePlaceLowererFactory,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
    SkillPolicyPreset,
)

__all__: list[str] = []

STACK_PLACE_CALL = "gen_sim.stack_place"
STACK_PICK_CALL = "gen_sim.stack_pick"
_RELATIVE_PLACE_CALL = "simulation.place_relative"


class _StackPickLowerer(_PickLowerer):
    call_id: ClassVar[str] = STACK_PICK_CALL


@dataclass(frozen=True, slots=True)
class _StackPickFactory(_PickLowererFactory):
    call_id: ClassVar[str] = STACK_PICK_CALL

    def create(self, **kwargs: Any) -> _StackPickLowerer:
        canonical = _PickLowererFactory.create(self, **kwargs)
        return _StackPickLowerer(
            tuple(canonical._routes.values()), tuple(canonical._semantics.values())
        )


class _StackPlaceLowerer(_RelativePlaceLowerer):
    """Reuse the original goal/effect binding without overriding action options."""

    call_id: ClassVar[str] = STACK_PLACE_CALL

    def __init__(
        self,
        routes: tuple[Any, ...],
        *,
        approach: float,
        release: float,
        nominal: float,
    ) -> None:
        super().__init__(routes)
        self._approach = approach
        self._release = release
        self._nominal = nominal

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: Any,
        bound: Any,
        option_template: Any,
    ) -> SemanticLowering:
        result = super().lower(
            call, context=context, bound=bound, option_template=option_template
        )
        # Freeze this observed, stable support for a precision descent. The
        # final task policy still checks the actually observed support relation.
        release = resolve_pose_goal(
            result.goal.xpos, context, name="stack_release"
        ).clone()
        if release.ndim == 2:
            release = release.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        approach = release.clone()
        approach[:, 2, 3] += self._approach - self._nominal
        release[:, 2, 3] += self._release - self._nominal
        return replace(
            result, goal=PlaceGoal(xpos=torch.stack((approach, release), dim=1))
        )

    def pick_lookahead_targets(
        self, call: RegisteredSemanticCall, **kwargs: Any
    ) -> Any:
        targets = super().pick_lookahead_targets(call, **kwargs)
        if targets is None:
            return None
        return tuple(
            replace(
                target,
                pose=replace(
                    target.pose,
                    world_displacement=(
                        target.pose.world_displacement
                        + target.pose.world_displacement.new_tensor(
                            [0.0, 0.0, self._release - self._nominal]
                        )
                    ),
                ),
            )
            for target in targets
        )


@dataclass(frozen=True, slots=True)
class _StackPlaceFactory(_RelativePlaceLowererFactory):
    """Register a separate task preset while retaining canonical route validation."""

    call_id: ClassVar[str] = STACK_PLACE_CALL
    revision: ClassVar[str] = "2"
    approach_clearance: float = 0.02
    release_clearance: float = 0.003
    nominal_clearance: float = 0.01

    def __post_init__(self) -> None:
        _RelativePlaceLowererFactory.__post_init__(self)
        values = (
            self.approach_clearance,
            self.release_clearance,
            self.nominal_clearance,
        )
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in values
        ):
            raise ValueError("Stack clearances must be finite positive lengths.")
        if self.release_clearance >= self.approach_clearance:
            raise ValueError("Stack approach must be above its release target.")

    def create(self, **kwargs: Any) -> _StackPlaceLowerer:
        canonical = _RelativePlaceLowererFactory.create(self, **kwargs)
        return _StackPlaceLowerer(
            tuple(canonical._routes.values()),
            approach=self.approach_clearance,
            release=self.release_clearance,
            nominal=self.nominal_clearance,
        )


def with_stack_placement(
    registration: Any,
    *,
    targets: frozenset[tuple[str, str]],
) -> Any:
    """Give stacking a full retract, independently of low table-placement caps."""
    relative_factory = next(
        factory
        for factory in registration.registered_semantic_lowerer_factories
        if factory.call_id == _RELATIVE_PLACE_CALL
    )
    factory = _StackPlaceFactory(
        tuple(
            route
            for route in relative_factory.routes
            if (route.object_id, route.reference_entity_id) in targets
            and route.relation in {"on", "above"}
        )
    )
    catalog = registration.call_catalog.with_descriptor(
        SemanticCallDescriptor(
            call_id=STACK_PLACE_CALL,
            spec_type=RegisteredSemanticCall,
            target_descriptor=factory.target_descriptor,
        )
    )
    pick_factory = _StackPickFactory(
        tuple(_PickRoute(obj, target) for obj, target in sorted(targets))
    )
    catalog = catalog.with_descriptor(
        SemanticCallDescriptor(
            call_id=STACK_PICK_CALL,
            spec_type=RegisteredSemanticCall,
            target_descriptor=pick_factory.target_descriptor,
        )
    )
    presets = []
    for preset in registration.robot_profile_binding.presets:
        options = dict(preset.action_option_templates)
        monitors = dict(preset.effect_monitors)
        if "pick" in options:
            options[STACK_PICK_CALL] = replace(options["pick"], pick_object_part="top")
            if "pick" in monitors:
                monitors[STACK_PICK_CALL] = monitors["pick"]
        if _RELATIVE_PLACE_CALL in options:
            options[STACK_PLACE_CALL] = replace(
                options[_RELATIVE_PLACE_CALL],
                max_approach_retract_z=None,
                hand_interp_steps=40,
            )
            if _RELATIVE_PLACE_CALL in monitors:
                monitors[STACK_PLACE_CALL] = monitors[_RELATIVE_PLACE_CALL]
        presets.append(
            SkillPolicyPreset(
                preset_id=preset.preset_id,
                required_planner=preset.required_planner,
                action_option_templates=options,
                effect_monitors=monitors,
                effect_assurance=preset.effect_assurance,
                motion_policy=preset.motion_policy,
                tracking_policy=preset.tracking_policy,
                recovery_policy=preset.recovery_policy,
                workflow_recovery_policy=preset.workflow_recovery_policy,
                runner_cfg=preset.runner_cfg,
            )
        )
    profile = replace(registration.robot_profile_binding, presets=tuple(presets))
    return replace(
        registration,
        robot_profile_binding=profile,
        call_catalog=catalog,
        registered_semantic_lowerer_factories=(
            *registration.registered_semantic_lowerer_factories,
            factory,
            pick_factory,
        ),
    )
