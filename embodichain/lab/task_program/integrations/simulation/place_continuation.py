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

"""Exact program-occurrence binding for private canonical release forecasts."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING
from embodichain.lab.sim.atomic_actions.place_continuation import PlaceContinuationRoute
from embodichain.lab.sim.atomic_actions import (
    PickUpOptions,
    PickPlanningRecovery,
    MoveHeldObjectPlanningRecovery,
    MoveHeldObjectOptions,
)
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    EffectAssurance,
    SemanticEffectKind,
    SemanticIntegrationManifest,
    SupportSurfaceAffordance,
)
from embodichain.lab.task_program.semantics.scene import (
    SceneObjectRef,
    SceneAffordanceRef,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.place_continuation import (
        PlaceContinuationSession,
    )
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        PlanningContext,
        ResolvedActionRequest,
        ActionPlan,
    )
    from embodichain.lab.sim.atomic_actions.planning_probes import PlaceSceneInputs
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
    from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
    from ..catalog import SimulationTaskProgramRegistration

__all__ = [
    "PlaceContinuationBinding",
    "HeldAlignmentContinuationBinding",
    "SimulationPlaceContinuationProvider",
]


@dataclass(frozen=True, slots=True)
class PlaceContinuationBinding:
    """Select exact sequential occurrences under a frozen complete program identity."""

    program_fingerprint: str
    source_invocation_id: str
    place_invocation_id: str
    transport_invocation_id: str | None = None

    def __post_init__(self) -> None:
        value = self.program_fingerprint
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(
                "Continuation requires a canonical SHA256 program fingerprint."
            )
        PlaceContinuationRoute(
            self.source_invocation_id,
            self.place_invocation_id,
            "declaration",
            self.transport_invocation_id,
        )


@dataclass(frozen=True, slots=True)
class HeldAlignmentContinuationBinding(PlaceContinuationBinding):
    """Bind one declared retained free-yaw alignment directly to its release."""

    def __post_init__(self) -> None:
        PlaceContinuationBinding.__post_init__(self)
        if self.transport_invocation_id is not None:
            raise ValueError(
                "Held alignment continuation does not admit an intermediate transport."
            )


@dataclass(frozen=True, slots=True)
class _BoundRoute:
    route: PlaceContinuationRoute
    calls: tuple
    presets: tuple[str, ...]


def _ordinary_on(call, subject, scene, rigid_entity_ids):
    if (
        subject not in rigid_entity_ids
        or type(scene.resolve(subject)) is not SceneObjectRef
    ):
        return False
    parent = None
    if type(call) is Place and call.object.entity_id == subject:
        target = call.on
        if type(target) is SceneAffordanceRef:
            entry = next(
                value
                for value in scene.entries
                if value.ref.entity_id == target.entity_id
            )
            if entry.affordance_payload_type is not SupportSurfaceAffordance:
                return False
            parent = None if entry.parent is None else entry.parent.entity_id
            if target.entity_id != f"on__{parent}__{subject}":
                return False
        elif type(target) is SceneObjectRef:
            parent = target.entity_id
    elif (
        type(call) is RegisteredSemanticCall
        and call.call_id == "simulation.place_relative"
    ):
        args = call.arguments
        if args.get("relation") == "on" and args.get("reference_part") is None:
            parent = args.get("reference")
    return bool(
        parent in rigid_entity_ids
        and parent != subject
        and type(scene.resolve(parent)) is SceneObjectRef
    )


def _program_fingerprint(program):
    from ..invocation_policy import task_program_fingerprint

    return task_program_fingerprint(program)


def _bind_place_continuation_declarations(
    program,
    manifest,
    policy_bindings,
    declarations,
    *,
    held_staging_bindings=(),
    rigid_entity_ids=frozenset(),
):
    from ..invocation_policy import BoundInvocationPolicies

    if not declarations and not any(
        type(preset.recovery_policy.planning_recovery)
        in (PickPlanningRecovery, MoveHeldObjectPlanningRecovery)
        and preset.recovery_policy.planning_recovery.place_continuation
        for preset in manifest.robot_profile.presets.values()
    ):
        return ()
    fingerprint = _program_fingerprint(program)
    if any(
        type(value) not in (PlaceContinuationBinding, HeldAlignmentContinuationBinding)
        for value in declarations
    ):
        raise TypeError("Continuation bindings must be typed declarations.")
    if len({value.source_invocation_id for value in declarations}) != len(declarations):
        raise ValueError("Duplicate continuation source occurrence.")
    policies = BoundInvocationPolicies(program, policy_bindings)
    policies.validate_manifest(program, manifest)
    selected = {value.invocation_id: value.preset_id for value in policies.bindings}
    segments = tuple(program.iter_segments())
    occurrences = {}
    enabled = set()
    for index, segment in enumerate(segments):
        for call in segment.calls:
            identity = (
                f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            linked = manifest.link_call(
                call.call, path=call.source_path, preset_id=selected.get(identity)
            )
            preset = manifest.robot_profile.presets[linked.preset_id]
            occurrences[identity] = (
                index,
                segment,
                linked.call,
                linked.preset_id,
                preset,
            )
            rule = preset.recovery_policy.planning_recovery
            if (
                type(rule) in (PickPlanningRecovery, MoveHeldObjectPlanningRecovery)
                and rule.place_continuation
            ):
                enabled.add(identity)
    if enabled != {value.source_invocation_id for value in declarations}:
        raise ValueError(
            "Pick continuation policies and occurrence bindings must agree exactly."
        )
    result = []
    for declaration in declarations:
        if declaration.program_fingerprint != fingerprint:
            raise ValueError(
                "Continuation program fingerprint differs from the declaration."
            )
        ids = (
            (declaration.source_invocation_id,)
            + (
                (declaration.transport_invocation_id,)
                if declaration.transport_invocation_id is not None
                else ()
            )
            + (declaration.place_invocation_id,)
        )
        if any(value not in occurrences for value in ids):
            raise ValueError("Continuation contains an unknown execution occurrence.")
        entries = [occurrences[value] for value in ids]
        start = entries[0][0]
        if any(
            entry[0] != start + offset
            or len(entry[1].calls) != 1
            or entry[1].parallel_block is not None
            for offset, entry in enumerate(entries)
        ):
            raise ValueError(
                "Continuation requires adjacent single-call serial occurrences."
            )
        source, place = entries[0][2], entries[-1][2]
        if type(declaration) is HeldAlignmentContinuationBinding:
            result.append(
                _bind_held_alignment_route(
                    entries,
                    ids,
                    manifest,
                    selected,
                    policy_bindings,
                    held_staging_bindings,
                )
            )
            continue
        if type(source) is not Pick or set(source.resources) != {"primary"}:
            raise ValueError(
                "Prospective acquisition requires a canonical single-resource Pick."
            )
        subject = source.object.entity_id
        if type(place) is Place:
            place_subject = place.object.entity_id
        elif (
            type(place) is RegisteredSemanticCall
            and place.call_id == "simulation.place_relative"
            and place.arguments.get("reference_part") is None
        ):
            args = place.arguments
            required = {"object", "reference", "relation"}
            if (
                not required <= set(args)
                or set(args) - required - {"stage_id", "reference_part"}
                or (
                    "stage_id" in args
                    and (
                        type(args["stage_id"]) is not str
                        or args["stage_id"] != args["stage_id"].strip()
                    )
                )
                or any(
                    type(args[k]) is not str
                    or not args[k]
                    or args[k] != args[k].strip()
                    for k in required
                )
            ):
                raise ValueError(
                    "Prospective release has unsupported relative-place arguments."
                )
            place_subject = args["object"]
        else:
            raise ValueError(
                "Prospective Pick release must be canonical Place or unqualified relative Place."
            )
        if subject != place_subject or any(
            dict(entry[2].resources) != dict(source.resources) for entry in entries
        ):
            raise ValueError("Continuation object/resources differ across occurrences.")
        if len(entries) == 3:
            transport = entries[1][2]
            if (
                type(transport) is not RegisteredSemanticCall
                or transport.call_id != "simulation.move_held_object"
            ):
                raise ValueError(
                    "Continuation transport must be canonical held-object motion."
                )
            args = transport.arguments
            if (
                not {"object", "target"} <= set(args)
                or set(args) - {"object", "target", "reference"}
                or args["object"] != subject
                or any(
                    type(v) is not str or not v or v != v.strip() for v in args.values()
                )
            ):
                raise ValueError(
                    "Continuation transport changed its declared object/target intent."
                )
        for entry in entries:
            if (
                entry[4].effect_assurance is not EffectAssurance.VERIFIED
                or entry[2].semantic_id not in entry[4].effect_monitors
            ):
                raise ValueError(
                    "Continuation requires canonical verified effect monitors for every occurrence."
                )
        option = entries[0][4].action_option_templates.get("pick")
        motion = entries[0][4].motion_policy
        if (
            type(option) is not PickUpOptions
            or option.linear_segments
            or option.constrained_grasp
            or option.fixed_object_to_eef is not None
            or option.rotate_upright is not None
            or option.obj_upright_direction is not None
            or option.pick_object_part != "center"
            or option.adaptive_grasp != "ordinary"
            or motion.strategy != "ik_interp"
            or motion.dynamic_collision_mode != "off"
        ):
            raise ValueError(
                "Continuation policy is outside the source prospective Pick capability."
            )
        deferred = option.adaptive_grasp == "ordinary" and _ordinary_on(
            place, subject, manifest.scene, rigid_entity_ids
        )
        result.append(
            _BoundRoute(
                PlaceContinuationRoute(
                    ids[0],
                    ids[-1],
                    subject,
                    ids[1] if len(ids) == 3 else None,
                    not deferred,
                ),
                tuple(entry[2] for entry in entries),
                tuple(entry[3] for entry in entries),
            )
        )
    return tuple(result)


def _bind_held_alignment_route(
    entries, ids, manifest, selected, policies, staging_bindings
):
    source, place = entries[0][2], entries[-1][2]
    if (
        len(entries) != 2
        or type(source) is not RegisteredSemanticCall
        or source.call_id != "gen_sim.align_held"
        or source.arguments.get("preserve_yaw") is not False
        or type(place) is not RegisteredSemanticCall
        or place.call_id
        not in {
            "simulation.place_relative",
            "gen_sim.place_upright",
            "gen_sim.stack_place",
        }
        or set(source.resources) != {"primary"}
        or dict(source.resources) != dict(place.resources)
        or type(source.arguments.get("object")) is not str
        or source.arguments["object"] != place.arguments.get("object")
    ):
        raise ValueError(
            "Held continuation requires the exact adjacent free-yaw alignment and registered Place."
        )
    subject = source.arguments["object"]
    binding = next(
        (value for value in staging_bindings if value.invocation_id == ids[0]), None
    )
    if (
        binding is None
        or binding.rigid_entity_id != subject
        or binding.predecessor_invocation_id is None
        or ids[0] not in selected
        or binding.predecessor_invocation_id not in selected
    ):
        raise ValueError(
            "Held continuation requires its explicitly selected staging/retention authority."
        )
    from .held_planning import _declared_authorities

    if ids[0] not in _declared_authorities(
        manifest.robot_profile, policies, (binding,)
    ):
        raise ValueError(
            "Held continuation canonical retention authority is unavailable."
        )
    for entry in entries:
        if (
            entry[4].effect_assurance is not EffectAssurance.VERIFIED
            or entry[2].semantic_id not in entry[4].effect_monitors
        ):
            raise ValueError("Held continuation requires verified canonical monitors.")
        if entry[4].motion_policy.sample_count != 260:
            raise ValueError(
                "Held continuation source and Place require unchanged 260-sample policies."
            )
    preset = entries[0][4]
    option = preset.action_option_templates.get(source.call_id)
    rule = preset.recovery_policy.planning_recovery
    if (
        type(option) is not MoveHeldObjectOptions
        or not option.report_retained_attachment
        or type(rule) is not MoveHeldObjectPlanningRecovery
        or not rule.staging
        or not rule.place_continuation
        or preset.motion_policy.strategy != "ik_interp"
        or preset.motion_policy.dynamic_collision_mode != "off"
    ):
        raise ValueError(
            "Held continuation requires its declared retained staging capability."
        )
    return _BoundRoute(
        PlaceContinuationRoute(ids[0], ids[-1], subject),
        tuple(entry[2] for entry in entries),
        tuple(entry[3] for entry in entries),
    )


def _require_monitor(grounded, kind, identity, entity, task_key):
    if (
        grounded.analyzed.effect_assurance is not EffectAssurance.VERIFIED
        or grounded.effect_monitor is None
        or grounded.effect_spec is None
        or grounded.effect_spec.effect_kind is not kind
        or grounded.effect_spec.invocation_id != identity
        or grounded.invocation.invocation_id != identity
        or len(grounded.effect_spec.state_expectations) != 1
        or grounded.effect_spec.state_expectations[0].object_id != entity
        or grounded.effect_spec.state_expectations[0].task_state_key != task_key
    ):
        raise ValueError("canonical_effect_monitor_unavailable")


class SimulationPlaceContinuationProvider:
    """One engine/compiled program owner, bound once to its canonical compiler."""

    def __init__(
        self,
        robot: Robot,
        program: CompiledTaskProgram,
        registration: SimulationTaskProgramRegistration,
        *,
        bindings: tuple[PlaceContinuationBinding, ...],
        require_open: Callable[[], None],
    ) -> None:
        """Validate frozen declarations without observations, native queries or IK."""
        if type(bindings) is not tuple:
            raise TypeError("Continuation bindings must be an immutable typed tuple.")
        if not callable(require_open):
            raise TypeError("Continuation requires an explicit runtime owner guard.")
        require_open()
        registration.assert_unchanged()
        manifest = SemanticIntegrationManifest(
            scene=registration.catalog.scene,
            robot_profile=registration.robot_profile_binding.declare(),
            call_catalog=registration.call_catalog,
            runtime_preset=program.integration.runtime_preset,
        )
        bound = _bind_place_continuation_declarations(
            program,
            manifest,
            registration.invocation_policy_bindings,
            bindings,
            rigid_entity_ids=frozenset(
                value.entity_id for value in registration.scene_binding.rigid_objects
            ),
            held_staging_bindings=(
                ()
                if getattr(registration, "planning_services", None) is None
                else registration.planning_services.held_staging_bindings
            ),
        )
        self._routes = {entry.route.source_invocation_id: entry for entry in bound}
        self._robot = robot
        self._program = program
        self._fingerprint = _program_fingerprint(program)
        self._registration = registration
        self._require_owner = require_open
        self._compiler = None
        self._engine = None
        self._workflows = {}
        self._closed = False

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        return self._robot

    def _require_open(self):
        self._require_owner()
        if self._closed:
            raise RuntimeError("Place continuation provider is closed.")

    def bind_compiler(
        self, engine: AtomicActionEngine, compiler: SemanticCallCompiler
    ) -> None:
        """Bind once to the already assembled compiler and exact owning engine."""
        self._require_open()
        self._registration.assert_unchanged()
        if _program_fingerprint(self._program) != self._fingerprint:
            raise ValueError(
                "Continuation compiled program changed before compiler binding."
            )
        if self._compiler is not None:
            raise RuntimeError("Place continuation compiler is already bound.")
        if engine.robot is not self.robot or compiler.integration.engine is not engine:
            raise ValueError("Place continuation compiler/engine ownership differs.")
        workflows = {}
        for identity, entry in self._routes.items():
            ids = (
                (entry.route.source_invocation_id,)
                + (
                    (entry.route.transport_invocation_id,)
                    if entry.route.transport_invocation_id
                    else ()
                )
                + (entry.route.place_invocation_id,)
            )
            workflows[identity] = tuple(
                compiler.analyze(
                    (call,),
                    workflow_id=occurrence.rsplit(":", 1)[0],
                    preset_ids=(preset,),
                )
                for occurrence, call, preset in zip(ids, entry.calls, entry.presets)
            )
        self._workflows = workflows
        self._engine, self._compiler = engine, compiler

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceContinuationSession | None:
        """Resolve only an exact declared Pick occurrence under this live owner."""
        self._require_open()
        if self._compiler is None:
            raise RuntimeError("Place continuation compiler is not bound.")
        entry = self._routes.get(request.invocation_id)
        if entry is None:
            return None
        pick = type(entry.calls[0]) is Pick
        key = request.binding.endpoint("primary", "motion").task_state_key
        held = None if pick else context.get_held_object(key)
        subject = (
            getattr(getattr(request.goal, "semantics", None), "entity_id", None)
            if pick
            else (None if held is None else held.semantics.entity_id)
        )
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or request.skill_id != ("pick_up" if pick else "move_held_object")
            or request.binding.owner_id
            != self._engine.planning_services.binding_owner_id
            or subject != entry.route.object_id
        ):
            raise ValueError(
                "Continuation source requires its exact B1 object and engine owner."
            )
        return _Session(
            self,
            entry.route,
            request.binding.endpoint("primary", "motion").task_state_key,
            self._workflows[request.invocation_id],
        )

    def close(self) -> None:
        """Invalidate existing sessions without closing a borrowed engine/compiler."""
        self._closed = True
        self._workflows.clear()
        self._compiler = self._engine = None


@dataclass(frozen=True, slots=True)
class _Session:
    owner: SimulationPlaceContinuationProvider
    route: PlaceContinuationRoute
    task_key: str
    workflows: tuple

    def _ground(self, index, context, identity, kind):
        self.owner._require_open()
        grounded = self.owner._compiler.ground(self.workflows[index], 0, context)
        _require_monitor(grounded, kind, identity, self.route.object_id, self.task_key)
        return self.owner._engine.resolve(grounded.invocation)

    def validate_source(self, context: PlanningContext) -> None:
        self._ground(
            0, context, self.route.source_invocation_id, SemanticEffectKind.ATTACH
        )

    def resolve_transport(self, context: PlanningContext) -> ResolvedActionRequest:
        if self.route.transport_invocation_id is None:
            raise ValueError("Continuation has no transport occurrence.")
        return self._ground(
            1, context, self.route.transport_invocation_id, SemanticEffectKind.ATTACH
        )

    def resolve_place(self, context: PlanningContext) -> ResolvedActionRequest:
        return self._ground(
            -1, context, self.route.place_invocation_id, SemanticEffectKind.RELEASE
        )

    def plan(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        self.owner._require_open()
        if not context.is_preview:
            raise ValueError(
                "Canonical continuation planning requires an explicit private context."
            )
        return self.owner._engine.plan_request(request, context)

    def place_scene_inputs(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PlaceSceneInputs:
        self.owner._require_open()
        provider = self.owner._engine.planning_services.place_scene_provider
        if provider is None:
            raise ValueError("Place continuation scene qualification is unavailable.")
        return provider.prepare(request, context)
