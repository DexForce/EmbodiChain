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

"""Explicit registered Pick targets using immutable shared grasp constraints."""

from __future__ import annotations
from dataclasses import dataclass, make_dataclass
from functools import lru_cache
from typing import ClassVar
import hashlib, math
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    PickUp,
    PickUpOptions,
    GraspGoal,
    ActionOptions,
    PlanningContext,
    SkillDescriptor,
    ObjectSemantics,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    RegisteredSemanticEffect,
    RegisteredHeldObjectEffect,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    BoundSemanticCall,
    SceneObjectRef,
    SceneRegistry,
    SceneManifest,
    GRASP_AFFORDANCE_CAPABILITY,
    HeldObjectRelation,
    SemanticEffectKind,
)
from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.toolkits.graspkit.pg_grasp.constraints import (
    GraspGeometryConstraint,
    grasp_geometry_digest,
)
from ..extensions import RegisteredSemanticLowererFactory
from .targets import AbsolutePoseTarget, _identifier
from .release_clearance import NominalHandBinding

__all__ = ["PickTargetRoute", "ConfiguredPickLowererFactory", "make_pick_alias_factory"]


@dataclass(frozen=True, slots=True)
class PickTargetRoute:
    """Target-qualified source region/release operands, separate from action options."""

    object_id: str
    target_id: str
    release_clearance_object_pose: AbsolutePoseTarget | None = None
    release_clearance_plane_z: float | None = None
    release_clearance_safety_margin: float = 0.0
    grasp_region: str | None = None

    def __post_init__(self) -> None:
        for name in ("object_id", "target_id"):
            _identifier(getattr(self, name), field_name=name)
        if self.grasp_region not in (None, "upper_half"):
            raise ValueError("Grasp region must be upper_half or omitted.")
        if (self.release_clearance_object_pose is None) != (
            self.release_clearance_plane_z is None
        ):
            raise ValueError(
                "Release-clearance pose and plane must be provided together."
            )
        if self.release_clearance_object_pose is not None:
            if type(self.release_clearance_object_pose) is not AbsolutePoseTarget:
                raise TypeError("Release clearance requires an explicit absolute pose.")
            object.__setattr__(
                self,
                "release_clearance_object_pose",
                self.release_clearance_object_pose.snapshot(),
            )
        for name in ("release_clearance_plane_z", "release_clearance_safety_margin"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite.")
        if self.release_clearance_safety_margin < 0:
            raise ValueError("Release-clearance margin must be nonnegative.")

    @property
    def constrained(self) -> bool:
        """Return whether this target explicitly declares region/release qualification."""
        return (
            self.grasp_region is not None
            or self.release_clearance_object_pose is not None
        )


class _ConfiguredPickLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "simulation.pick"
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

    def __init__(self, routes, semantics, generators, hands, verify_phases):
        self._routes = {(r.object_id, r.target_id): r for r in routes}
        self._semantics = {value.entity_id: value for value in semantics}
        self._generators = dict(generators)
        self._hands = {hand.grasp_target_id: hand for hand in hands}
        self._verify_phases = verify_phases

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        if type(option_template) is not PickUpOptions:
            raise TypeError("Configured Pick requires exact PickUpOptions.")
        args = dict(call.arguments)
        if set(args) != {"object", "target"}:
            raise ValueError(
                "Configured Pick arguments must contain only object and target."
            )
        route = self._routes.get((args["object"], args["target"]))
        if route is None:
            raise ValueError("Configured Pick has no matching explicit route.")
        semantics = self._semantics[route.object_id]
        constraint = None
        if route.constrained:
            if (
                not option_template.constrained_grasp
                or option_template.pick_object_part != "center"
                or option_template.rotate_upright is not None
                or option_template.fixed_object_to_eef is not None
            ):
                raise ValueError(
                    "Constrained Pick requires its explicit unrotated center-sampling options."
                )
            target = (
                bound.binding.resources["primary"].endpoints["grasp"].runtime_target
            )
            generator = self._generators[target.target_id]
            if not isinstance(generator, ParallelJawGraspPoseGenerator):
                raise TypeError(
                    "Constrained Pick requires a calibrated parallel-jaw provider."
                )
            affordance = semantics.affordance
            axis = getattr(affordance, "internal_axis", None)
            if route.grasp_region is not None and axis is None:
                raise ValueError("Upper-half filtering requires a declared local axis.")
            axis = (
                torch.tensor([0.0, 0.0, 1.0]) if axis is None else axis.detach().cpu()
            )
            axis = torch.nn.functional.normalize(axis, dim=-1)
            projection = affordance.mesh_vertices.detach().cpu() @ axis
            release = route.release_clearance_object_pose
            hand = None
            if release is not None:
                hand = self._hands.get(target.target_id)
                if hand is None or hand.model_id != generator.gripper_model.model_id:
                    raise ValueError(
                        "Constrained release requires its exact declared nominal hand model."
                    )
            constraint = GraspGeometryConstraint(
                geometry_digest=grasp_geometry_digest(
                    affordance.mesh_vertices, affordance.mesh_triangles
                ),
                local_axis=tuple(float(x) for x in axis),
                midpoint=float((projection.amin() + projection.amax()) * 0.5),
                upper_half=route.grasp_region == "upper_half",
                release_pose=(
                    None
                    if release is None
                    else tuple(float(x) for x in release.to_matrix().reshape(-1))
                ),
                plane_z=route.release_clearance_plane_z,
                margin=route.release_clearance_safety_margin,
                hand_envelope=None if hand is None else hand.envelope,
                model_id=None if hand is None else hand.model_id,
            )
        elif option_template.constrained_grasp:
            raise ValueError("Constrained options require declared target geometry.")
        handover = (
            self.call_id == "gen_sim.pick.handover_source"
            or self.call_id.startswith("gen_sim.pick.handover_horizontal_source.")
        )
        if handover and option_template.adaptive_grasp != "handover_source":
            raise ValueError(
                "Handover-source aliases require explicit handover_source action options."
            )
        return SemanticLowering(
            goal=GraspGoal(semantics=semantics, grasp_constraint=constraint),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=route.object_id,
                        slot_id="primary",
                        acquire_before_segment="lift" if self._verify_phases else None,
                        retain_during_segments=("lift",) if self._verify_phases else (),
                    ),
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class ConfiguredPickLowererFactory(RegisteredSemanticLowererFactory):
    """Expose target-qualified pickup through ordinary PickUp and public contracts."""

    call_id: ClassVar[str] = "simulation.pick"
    revision: ClassVar[str] = "typed-constraints-1"
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    lowerer_type: ClassVar[type[_ConfiguredPickLowerer]] = _ConfiguredPickLowerer
    routes: tuple[PickTargetRoute, ...]
    verify_phases: bool = False
    hands: tuple[NominalHandBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not PickTargetRoute for route in self.routes)
        ):
            raise TypeError(
                "Pick routes require explicit immutable target declarations."
            )
        if len({(route.object_id, route.target_id) for route in self.routes}) != len(
            self.routes
        ):
            raise ValueError("Pick routes must be unique.")
        if type(self.verify_phases) is not bool:
            raise TypeError("verify_phases must be bool.")
        if (
            type(self.hands) is not tuple
            or any(type(hand) is not NominalHandBinding for hand in self.hands)
            or len({hand.grasp_target_id for hand in self.hands}) != len(self.hands)
        ):
            raise TypeError(
                "Pick hand envelopes require unique immutable target bindings."
            )
        if (
            any(
                route.release_clearance_object_pose is not None for route in self.routes
            )
            and not self.hands
        ):
            raise ValueError(
                "Release-constrained Pick requires explicit nominal hand envelopes."
            )

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Resolve explicit object identities before runtime assembly.

        Args:
            scene: Canonical immutable scene declarations.
        """
        for route in self.routes:
            scene.resolve(route.object_id, expected_type=SceneObjectRef)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Bind source object semantics without sampling or planning.

        Args:
            simulation: Borrowed owner; unused here.
            robot: Exact engine robot.
            scene_registry: Existing object/grasp registry.
            engine: Ordinary Atomic engine owning the calibrated sampling services.

        Returns:
            Fresh lowerer selecting immutable goal constraints and effect gates.
        """
        if engine.robot is not robot:
            raise ValueError("Configured Pick requires the engine's exact robot.")
        semantics = []
        for object_id in dict.fromkeys(route.object_id for route in self.routes):
            ref = scene_registry.resolve(object_id, expected_type=SceneObjectRef)
            grasp = scene_registry.resolve_affordance(
                ref, capability=GRASP_AFFORDANCE_CAPABILITY
            )
            semantics.append(scene_registry.object_semantics(ref, affordance=grasp))
        return self.lowerer_type(
            self.routes,
            tuple(semantics),
            engine.grasp_pose_generators,
            self.hands,
            self.verify_phases,
        )


@lru_cache(maxsize=1024)
def _alias_factory_type(call_id: str) -> type[ConfiguredPickLowererFactory]:
    suffix = hashlib.sha256(call_id.encode()).hexdigest()[:16]
    lowerer = type(
        f"_PickAliasLowerer_{suffix}",
        (_ConfiguredPickLowerer,),
        {"call_id": call_id, "__module__": __name__, "__slots__": ()},
    )
    return make_dataclass(
        f"_PickAliasFactory_{suffix}",
        [],
        bases=(ConfiguredPickLowererFactory,),
        namespace={"call_id": call_id, "lowerer_type": lowerer, "__module__": __name__},
        frozen=True,
        slots=True,
    )


def make_pick_alias_factory(
    routes: tuple[PickTargetRoute, ...],
    call_id: str,
    *,
    verify_phases: bool = False,
    hands: tuple[NominalHandBinding, ...] = (),
) -> ConfiguredPickLowererFactory:
    """Bind an allowlisted stable source alias, never executable code from data.

    Args:
        routes: Immutable target-qualified route declarations.
        call_id: ``simulation.pick``, ``gen_sim.stack_pick`` or a ``gen_sim.pick.`` alias.
        verify_phases: Require source acquisition/retention gates before/during lift.
        hands: Explicit nominal hand envelopes when release filtering is requested.

    Returns:
        Immutable registered factory with a deterministic class-level alias identity.
    """
    _identifier(call_id, field_name="call_id")
    if call_id == "simulation.pick":
        return ConfiguredPickLowererFactory(routes, verify_phases, hands)
    if call_id != "gen_sim.stack_pick" and (
        not call_id.startswith("gen_sim.pick.")
        or not call_id.removeprefix("gen_sim.pick.")
    ):
        raise ValueError("Pick aliases must use the declared source namespaces.")
    return _alias_factory_type(call_id)(routes, verify_phases, hands)
