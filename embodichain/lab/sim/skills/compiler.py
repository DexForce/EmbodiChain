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

"""Static workflow analysis and JIT semantic-call lowering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar
from uuid import uuid4

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionControlOverrides,
    ActionInvocation,
    ActionOptions,
    Affordance,
    ArticulationOperationAffordance,
    GraspGoal,
    HandOverOptions,
    HeldObjectState,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    OperateArticulationGoal,
    PlanningContext,
    PoseGoalValue,
    SceneArticulationOperationGeometry,
    SceneEntityPose,
)
from .calls import (
    HandOver,
    OperateArticulation,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
    SemanticPose,
)
from .effects import (
    ArticulationJointStateExpectation,
    CONTACT_EFFECT_CHANNEL,
    CONSTRAINT_EFFECT_CHANNEL,
    POSE_RELATION_EFFECT_CHANNEL,
    BinaryEffectClause,
    BinaryEvidenceKind,
    CompositeEffectMonitorFactory,
    CoordinatedHeldObjectCleanupExpectation,
    EffectClause,
    EffectMonitor,
    EffectMonitorRef,
    EffectMonitorRegistry,
    EffectEvidenceSourceRef,
    EffectStateExpectation,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    JointStateEffectClause,
    PoseRelationClause,
    PoseRelationExpectation,
    SemanticEffectKind,
    SemanticEffectSpec,
    SymbolicStateKey,
)
from .integration import (
    BoundSemanticCall,
    BoundSemanticIntegration,
    PathPart,
    SemanticDiagnostic,
    SemanticValidationError,
)
from .scene import (
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
    ArticulationJointEvidenceAddress,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneObjectRef,
)


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Return one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _diagnostic(
    code: str,
    path: tuple[PathPart, ...],
    message: str,
    candidates: tuple[str, ...] = (),
) -> SemanticValidationError:
    """Build one pathful semantic compiler error."""
    return SemanticValidationError(SemanticDiagnostic(code, path, message, candidates))


@dataclass(frozen=True, slots=True)
class SemanticRelationTarget:
    """Statically selected relation affordance awaiting typed grounding."""

    capability: str
    affordance: SceneAffordanceRef
    payload_type: type[Affordance]
    payload_revision: str

    def __post_init__(self) -> None:
        _validate_identifier(self.capability, field_name="relation capability")
        if type(self.affordance) is not SceneAffordanceRef:
            raise TypeError("affordance must be exactly SceneAffordanceRef.")
        if not isinstance(self.payload_type, type) or not issubclass(
            self.payload_type, Affordance
        ):
            raise TypeError("payload_type must be an Affordance subclass.")
        _validate_identifier(
            self.payload_revision,
            field_name="relation payload_revision",
        )

    @property
    def grounder_key(self) -> tuple[str, type[Affordance], str]:
        """Return the exact typed/versioned grounder lookup key."""
        return self.capability, self.payload_type, self.payload_revision


class RelationTargetGrounder(ABC):
    """Shared implementation that converts one relation into object pose."""

    capability: ClassVar[str]
    affordance_type: ClassVar[type[Affordance]]
    affordance_revision: ClassVar[str]

    @abstractmethod
    def ground(
        self,
        relation: SemanticRelationTarget,
        *,
        affordance: Affordance,
        context: PlanningContext,
    ) -> PoseGoalValue:
        """Return an object-space target from current state and typed payload.

        Args:
            relation: Statically selected relation metadata.
            affordance: Owned exact-type affordance payload.
            context: Latest immutable planning observation.

        Returns:
            Direct or scene-relative desired object pose.
        """


@dataclass(frozen=True, slots=True)
class SemanticObjectTarget:
    """One object-space look-ahead target.

    Relation targets remain late-bound and require an explicitly installed
    typed/versioned grounder. Handover targets defer to the
    embodiment-selected provider and are used only for workflow look-ahead.
    """

    value: (
        SemanticPose | SceneEntityPose | SemanticRelationTarget | SemanticHandOverTarget
    )

    def __post_init__(self) -> None:
        if type(self.value) in (SemanticPose, SceneEntityPose):
            object.__setattr__(self, "value", self.value.snapshot())
        elif type(self.value) not in (
            SemanticRelationTarget,
            SemanticHandOverTarget,
        ):
            raise TypeError(
                "value must be exactly SemanticPose, SceneEntityPose, "
                "SemanticRelationTarget, or SemanticHandOverTarget."
            )

    @property
    def pose(self) -> SemanticPose | SceneEntityPose | None:
        """Return the direct pose variant, when selected."""
        if type(self.value) in (SemanticPose, SceneEntityPose):
            return self.value  # type: ignore[return-value]
        return None

    @property
    def relation(self) -> SemanticRelationTarget | None:
        """Return the relation variant, when selected."""
        return self.value if type(self.value) is SemanticRelationTarget else None

    @property
    def handover(self) -> SemanticHandOverTarget | None:
        """Return the deferred handover variant, when selected."""
        return self.value if type(self.value) is SemanticHandOverTarget else None


@dataclass(frozen=True, slots=True)
class SemanticHandOverTarget:
    """Deferred middle pose selected by one named embodiment provider."""

    provider_id: str
    bound: BoundSemanticCall

    def __post_init__(self) -> None:
        _validate_identifier(self.provider_id, field_name="handover provider_id")
        if type(self.bound) is not BoundSemanticCall:
            raise TypeError("bound must be exactly BoundSemanticCall.")
        if type(self.bound.linked.call) is not HandOver:
            raise TypeError("bound must contain an exact HandOver call.")


@dataclass(frozen=True, slots=True)
class SemanticEffectDependency:
    """A consumer's verified-held-state dependency on an earlier call."""

    producer_index: int | None
    consumer_index: int
    object: SceneObjectRef

    def __post_init__(self) -> None:
        if self.producer_index is not None and (
            type(self.producer_index) is not int or self.producer_index < 0
        ):
            raise ValueError("producer_index must be non-negative or None.")
        if type(self.consumer_index) is not int or self.consumer_index < 0:
            raise ValueError("consumer_index must be non-negative.")
        if self.producer_index is not None and (
            self.producer_index >= self.consumer_index
        ):
            raise ValueError("producer_index must precede consumer_index.")
        if type(self.object) is not SceneObjectRef:
            raise TypeError("object must be exactly SceneObjectRef.")


@dataclass(frozen=True, slots=True)
class AnalyzedSemanticCall:
    """One statically linked call plus workflow-derived lowering metadata."""

    index: int
    bound: BoundSemanticCall
    effect_kind: SemanticEffectKind
    symbolic_writes: frozenset[SymbolicStateKey] = frozenset()
    opaque_symbolic_effect: bool = False
    effect_monitor_ref: EffectMonitorRef | None = None
    downstream_object_targets: tuple[SemanticObjectTarget, ...] = ()
    requires_verified_held_object: bool = False
    requires_fresh_observation: bool = True

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a non-negative integer.")
        if type(self.bound) is not BoundSemanticCall:
            raise TypeError("bound must be exactly BoundSemanticCall.")
        if not isinstance(self.effect_kind, SemanticEffectKind):
            raise TypeError("effect_kind must be a SemanticEffectKind.")
        if type(self.symbolic_writes) is not frozenset or not all(
            type(write) is SymbolicStateKey for write in self.symbolic_writes
        ):
            raise TypeError(
                "symbolic_writes must be an exact frozenset of "
                "SymbolicStateKey values."
            )
        if type(self.opaque_symbolic_effect) is not bool:
            raise TypeError("opaque_symbolic_effect must be a bool.")
        if self.opaque_symbolic_effect and self.symbolic_writes:
            raise ValueError(
                "Opaque symbolic effects cannot also claim inferred exact keys."
            )
        if self.effect_monitor_ref is not None:
            if not isinstance(self.effect_monitor_ref, EffectMonitorRef):
                raise TypeError(
                    "effect_monitor_ref must be an EffectMonitorRef or None."
                )
            object.__setattr__(
                self,
                "effect_monitor_ref",
                self.effect_monitor_ref.snapshot(),
            )
        targets = tuple(self.downstream_object_targets)
        if not all(type(target) is SemanticObjectTarget for target in targets):
            raise TypeError(
                "downstream_object_targets must contain exact "
                "SemanticObjectTarget values."
            )
        object.__setattr__(self, "downstream_object_targets", targets)
        if type(self.requires_verified_held_object) is not bool:
            raise TypeError("requires_verified_held_object must be a bool.")
        if type(self.requires_fresh_observation) is not bool:
            raise TypeError("requires_fresh_observation must be a bool.")

    @property
    def call(self) -> SemanticCallSpec:
        """Return the canonical linked semantic call."""
        return self.bound.linked.call


@dataclass(frozen=True, slots=True)
class SemanticWorkflow:
    """Immutable result of static workflow analysis."""

    workflow_id: str
    calls: tuple[AnalyzedSemanticCall, ...]
    effect_dependencies: tuple[SemanticEffectDependency, ...] = ()
    _compiler_id: str = field(repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        _validate_identifier(self.workflow_id, field_name="workflow_id")
        calls = tuple(self.calls)
        if not calls:
            raise ValueError("SemanticWorkflow requires at least one call.")
        if not all(type(call) is AnalyzedSemanticCall for call in calls):
            raise TypeError("calls must contain exact AnalyzedSemanticCall values.")
        if tuple(call.index for call in calls) != tuple(range(len(calls))):
            raise ValueError("SemanticWorkflow call indices must be contiguous.")
        dependencies = tuple(self.effect_dependencies)
        if not all(
            type(dependency) is SemanticEffectDependency for dependency in dependencies
        ):
            raise TypeError(
                "effect_dependencies must contain exact "
                "SemanticEffectDependency values."
            )
        _validate_identifier(self._compiler_id, field_name="compiler_id")
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "effect_dependencies", dependencies)


@dataclass(frozen=True, slots=True)
class SemanticLowering:
    """Registered-lowerer output wrapped by compiler-owned invocation policy."""

    goal: object
    skill_options: ActionOptions | None = None
    control_overrides: ActionControlOverrides = field(
        default_factory=ActionControlOverrides
    )

    def __post_init__(self) -> None:
        if self.skill_options is not None and not isinstance(
            self.skill_options, ActionOptions
        ):
            raise TypeError("skill_options must be an ActionOptions or None.")
        if type(self.control_overrides) is not ActionControlOverrides:
            raise TypeError("control_overrides must be exactly ActionControlOverrides.")


class RegisteredSemanticLowerer(ABC):
    """Explicitly installed implementation for one registered call ID."""

    call_id: ClassVar[str]
    schema_version: ClassVar[int]

    @abstractmethod
    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> SemanticLowering:
        """Lower one registered value to goal/options without changing policy."""


@dataclass(frozen=True, slots=True)
class HandOverPoseTargets:
    """Embodiment-owned object-space poses needed by the core handover skill."""

    middle: SemanticObjectTarget
    final: SemanticObjectTarget

    def __post_init__(self) -> None:
        if type(self.middle) is not SemanticObjectTarget:
            raise TypeError("middle must be exactly SemanticObjectTarget.")
        if type(self.final) is not SemanticObjectTarget:
            raise TypeError("final must be exactly SemanticObjectTarget.")


class HandOverPoseProvider(ABC):
    """Integration extension that selects robot-appropriate handover poses."""

    provider_id: ClassVar[str]

    @abstractmethod
    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Return middle and final object-space targets for one handover.

        Args:
            call: Canonical handover semantic value.
            context: Latest immutable planning observation.
            bound: Engine/profile-bound handover call.

        Returns:
            Embodiment-appropriate middle and final object targets.
        """


@dataclass(frozen=True, slots=True)
class GroundedSemanticCall:
    """Call lowered from the latest observed context."""

    analyzed: AnalyzedSemanticCall
    invocation: ActionInvocation
    effect_spec: SemanticEffectSpec | None
    effect_monitor: EffectMonitor | None = field(repr=False, compare=False)
    _eligible_mask: torch.Tensor = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside :class:`SemanticSkillCompiler`."""
        del args, kwargs
        raise TypeError(
            "GroundedSemanticCall values are created by "
            "SemanticSkillCompiler.ground()."
        )

    @classmethod
    def _create(
        cls,
        *,
        analyzed: AnalyzedSemanticCall,
        invocation: ActionInvocation,
        effect_spec: SemanticEffectSpec | None,
        effect_monitor: EffectMonitor | None,
        eligible_mask: torch.Tensor,
    ) -> GroundedSemanticCall:
        """Create one compiler-owned grounded result."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "analyzed", analyzed)
        object.__setattr__(instance, "invocation", invocation)
        object.__setattr__(instance, "effect_spec", effect_spec)
        object.__setattr__(instance, "effect_monitor", effect_monitor)
        object.__setattr__(instance, "_eligible_mask", eligible_mask.clone())
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if type(self.analyzed) is not AnalyzedSemanticCall:
            raise TypeError("analyzed must be exactly AnalyzedSemanticCall.")
        if type(self.invocation) is not ActionInvocation:
            raise TypeError("invocation must be exactly ActionInvocation.")
        if self.invocation.skill_id != self.analyzed.bound.linked.descriptor.skill_id:
            raise ValueError("invocation skill_id must match the analyzed call.")
        if (self.effect_spec is None) != (self.effect_monitor is None):
            raise ValueError(
                "effect_spec and effect_monitor must either both be set or both be None."
            )
        if self.effect_spec is not None:
            if not isinstance(self.effect_spec, SemanticEffectSpec):
                raise TypeError("effect_spec must be a SemanticEffectSpec or None.")
            if not isinstance(self.effect_monitor, EffectMonitor):
                raise TypeError("effect_monitor must be an EffectMonitor or None.")
            if self.effect_spec.semantic_id != self.analyzed.call.semantic_id:
                raise ValueError(
                    "effect_spec semantic_id must match the analyzed call."
                )
            object.__setattr__(self, "effect_spec", self.effect_spec.snapshot())
        if not isinstance(self._eligible_mask, torch.Tensor):
            raise TypeError("eligible_mask must be a torch.Tensor.")
        if self._eligible_mask.dtype != torch.bool or self._eligible_mask.dim() != 1:
            raise ValueError("eligible_mask must be a one-dimensional bool tensor.")
        if self._eligible_mask.numel() == 0:
            raise ValueError("eligible_mask must contain at least one environment.")
        object.__setattr__(self, "_eligible_mask", self._eligible_mask.clone())

    @property
    def eligible_mask(self) -> torch.Tensor:
        """Return an owned mask that the execution session must preserve."""
        return self._eligible_mask.clone()


class SemanticSkillCompiler:
    """Analyze semantic workflows and JIT-lower exactly one call at a time."""

    def __init__(
        self,
        integration: BoundSemanticIntegration,
        *,
        registered_lowerers: Iterable[RegisteredSemanticLowerer] = (),
        relation_grounders: Iterable[RelationTargetGrounder] = (),
        handover_pose_providers: Iterable[HandOverPoseProvider] = (),
        effect_monitor_registry: EffectMonitorRegistry | None = None,
    ) -> None:
        """Install immutable semantic lowering and grounding registries.

        Args:
            integration: Exact live scene, engine, and robot-profile binding.
            registered_lowerers: Explicit implementations for registered calls.
            relation_grounders: Exact capability/payload/revision dispatch entries.
            handover_pose_providers: Named embodiment-owned handover providers.
            effect_monitor_registry: Versioned semantic-effect monitor factories.
        """
        if type(integration) is not BoundSemanticIntegration:
            raise TypeError("integration must be exactly BoundSemanticIntegration.")
        if isinstance(registered_lowerers, (str, bytes)):
            raise TypeError("registered_lowerers must be an iterable of lowerers.")
        try:
            supplied_lowerers = tuple(registered_lowerers)
        except TypeError as exc:
            raise TypeError(
                "registered_lowerers must be an iterable of lowerers."
            ) from exc
        lowerers: dict[str, RegisteredSemanticLowerer] = {}
        for lowerer in supplied_lowerers:
            if not isinstance(lowerer, RegisteredSemanticLowerer):
                raise TypeError(
                    "registered_lowerers must contain RegisteredSemanticLowerer "
                    "instances."
                )
            call_id = _validate_identifier(
                getattr(type(lowerer), "call_id", None),
                field_name="RegisteredSemanticLowerer.call_id",
            )
            if call_id in lowerers:
                raise ValueError(f"Duplicate registered lowerer {call_id!r}.")
            try:
                descriptor = integration.manifest.call_catalog.discover(call_id)
            except KeyError as exc:
                raise ValueError(
                    f"Lowerer {call_id!r} has no registered semantic descriptor."
                ) from exc
            if descriptor.spec_type is not RegisteredSemanticCall:
                raise ValueError(
                    f"Lowerer {call_id!r} cannot replace curated call semantics."
                )
            schema_version = getattr(type(lowerer), "schema_version", None)
            if type(schema_version) is not int or (
                schema_version != descriptor.schema_version
            ):
                raise ValueError(
                    f"Lowerer {call_id!r} schema_version must exactly match "
                    f"descriptor version {descriptor.schema_version}."
                )
            lowerers[call_id] = lowerer
        if isinstance(relation_grounders, (str, bytes)):
            raise TypeError("relation_grounders must be an iterable of grounders.")
        try:
            supplied_grounders = tuple(relation_grounders)
        except TypeError as exc:
            raise TypeError(
                "relation_grounders must be an iterable of grounders."
            ) from exc
        normalized_grounders: dict[
            tuple[str, type[Affordance], str], RelationTargetGrounder
        ] = {}
        for grounder in supplied_grounders:
            if not isinstance(grounder, RelationTargetGrounder):
                raise TypeError(
                    "relation_grounders must contain RelationTargetGrounder "
                    "instances."
                )
            grounder_type = type(grounder)
            capability = _validate_identifier(
                getattr(grounder_type, "capability", None),
                field_name="RelationTargetGrounder.capability",
            )
            affordance_type = getattr(grounder_type, "affordance_type", None)
            if not isinstance(affordance_type, type) or not issubclass(
                affordance_type, Affordance
            ):
                raise TypeError(
                    "RelationTargetGrounder.affordance_type must be an "
                    "Affordance subclass."
                )
            revision = _validate_identifier(
                getattr(grounder_type, "affordance_revision", None),
                field_name="RelationTargetGrounder.affordance_revision",
            )
            key = (capability, affordance_type, revision)
            if key in normalized_grounders:
                raise ValueError(f"Duplicate relation grounder key {key!r}.")
            normalized_grounders[key] = grounder
        if isinstance(handover_pose_providers, (str, bytes)):
            raise TypeError("handover_pose_providers must be an iterable of providers.")
        try:
            supplied_handover_providers = tuple(handover_pose_providers)
        except TypeError as exc:
            raise TypeError(
                "handover_pose_providers must be an iterable of providers."
            ) from exc
        normalized_handover_providers: dict[str, HandOverPoseProvider] = {}
        for provider in supplied_handover_providers:
            if not isinstance(provider, HandOverPoseProvider):
                raise TypeError(
                    "handover_pose_providers must contain "
                    "HandOverPoseProvider instances."
                )
            provider_id = _validate_identifier(
                getattr(type(provider), "provider_id", None),
                field_name="HandOverPoseProvider.provider_id",
            )
            if provider_id in normalized_handover_providers:
                raise ValueError(f"Duplicate handover pose provider {provider_id!r}.")
            normalized_handover_providers[provider_id] = provider
        self._integration = integration
        self._compiler_id = uuid4().hex
        self._registered_lowerers = MappingProxyType(lowerers)
        self._relation_grounders = MappingProxyType(normalized_grounders)
        self._handover_pose_providers = MappingProxyType(normalized_handover_providers)
        selected_monitor_registry = (
            EffectMonitorRegistry((CompositeEffectMonitorFactory(),))
            if effect_monitor_registry is None
            else effect_monitor_registry
        )
        if not isinstance(selected_monitor_registry, EffectMonitorRegistry):
            raise TypeError(
                "effect_monitor_registry must be an EffectMonitorRegistry or None."
            )
        self._effect_monitor_registry = selected_monitor_registry

    @property
    def integration(self) -> BoundSemanticIntegration:
        """Return the exact live integration used for linking and grounding."""
        return self._integration

    @property
    def registered_lowerers(self) -> Mapping[str, RegisteredSemanticLowerer]:
        """Return installed registered-call lowerers by stable call ID."""
        return self._registered_lowerers

    @property
    def relation_grounders(
        self,
    ) -> Mapping[tuple[str, type[Affordance], str], RelationTargetGrounder]:
        """Return exact typed/versioned relation grounders."""
        return self._relation_grounders

    @property
    def handover_pose_providers(self) -> Mapping[str, HandOverPoseProvider]:
        """Return installed handover pose providers by stable provider ID."""
        return self._handover_pose_providers

    @property
    def effect_monitor_registry(self) -> EffectMonitorRegistry:
        """Return the immutable versioned effect-monitor factory registry."""
        return self._effect_monitor_registry

    def analyze(
        self,
        calls: Iterable[SemanticCallSpec],
        *,
        workflow_id: str = "semantic_workflow",
        path: tuple[PathPart, ...] = ("workflow",),
    ) -> SemanticWorkflow:
        """Statically link calls and infer look-ahead/effect dependencies.

        Args:
            calls: Ordered exact semantic call values.
            workflow_id: Stable caller-selected workflow identifier.
            path: Root diagnostic path.

        Returns:
            Factory-owned provider-free workflow analysis.

        Raises:
            SemanticValidationError: If linking, grounding capabilities, or
                object-state flow are invalid.
        """
        _validate_identifier(workflow_id, field_name="workflow_id")
        self._assert_current(path=("integration", "robot_profile"))
        if isinstance(calls, (str, bytes)):
            raise TypeError("calls must be an iterable of semantic call values.")
        try:
            supplied = tuple(calls)
        except TypeError as exc:
            raise TypeError(
                "calls must be an iterable of semantic call values."
            ) from exc
        if not supplied:
            raise ValueError("Semantic workflow requires at least one call.")
        allowed_types = (
            Pick,
            Place,
            HandOver,
            OperateArticulation,
            RegisteredSemanticCall,
        )
        if not all(type(call) in allowed_types for call in supplied):
            raise TypeError("calls must contain exact supported semantic call values.")

        bound_calls: list[BoundSemanticCall] = []
        effect_kinds: list[SemanticEffectKind] = []
        dependencies: list[SemanticEffectDependency] = []
        latest_holder: dict[str, tuple[int, str]] = {}
        for index, call in enumerate(supplied):
            if type(call) is RegisteredSemanticCall and (
                call.call_id not in self._registered_lowerers
            ):
                raise _diagnostic(
                    "semantic_lowerer_not_installed",
                    (*path, index, "kind"),
                    f"Registered semantic call {call.call_id!r} has no explicitly "
                    "installed compiler lowerer.",
                    tuple(self._registered_lowerers),
                )
            call = self._inherit_held_resource(call, latest_holder)
            bound = self._integration.link_call(
                call,
                path=(*path, index, "call"),
            )
            bound_calls.append(bound)
            call = bound.linked.call
            if type(call) is HandOver:
                self._require_handover_pose_provider(
                    call,
                    path=(*path, index, "call"),
                )
            if type(call) is Place and call.at is None:
                target = self._relation_target(bound)
                assert target.relation is not None
                destination_metadata = self._integration.manifest.scene.lookup(
                    target.relation.affordance,
                    expected_type=SceneAffordanceRef,
                )
                if destination_metadata.parent == call.object:
                    raise _diagnostic(
                        "place_self_reference",
                        (*path, index, "call", "destination"),
                        f"Object {call.object.entity_id!r} cannot be placed in a "
                        "relation to its own affordance.",
                    )
                self._require_relation_grounder(
                    target.relation,
                    path=(*path, index, "call", "destination"),
                )
            if type(call) is Pick:
                previous = latest_holder.get(call.object.entity_id)
                if previous is not None:
                    raise _diagnostic(
                        "invalid_object_state_flow",
                        (*path, index, "call", "object"),
                        f"Object {call.object.entity_id!r} is already acquired by "
                        f"call {previous[0]} without an intervening release.",
                    )
                effect_kind = SemanticEffectKind.ATTACH
                latest_holder[call.object.entity_id] = (
                    index,
                    bound.binding.resource_ids["primary"],
                )
            elif type(call) is Place:
                producer = latest_holder.get(call.object.entity_id)
                selected_resource = bound.binding.resource_ids["primary"]
                if producer is not None and producer[1] != selected_resource:
                    raise _diagnostic(
                        "held_resource_mismatch",
                        (*path, index, "call", "resources", "primary"),
                        f"Place selects resource {selected_resource!r}, but the "
                        f"verified producer selects {producer[1]!r}.",
                        (producer[1],),
                    )
                effect_kind = SemanticEffectKind.RELEASE
                dependencies.append(
                    SemanticEffectDependency(
                        producer_index=None if producer is None else producer[0],
                        consumer_index=index,
                        object=call.object,
                    )
                )
                latest_holder.pop(call.object.entity_id, None)
            elif type(call) is HandOver:
                producer = latest_holder.get(call.object.entity_id)
                source_resource = bound.binding.resource_ids["source"]
                if producer is not None and producer[1] != source_resource:
                    raise _diagnostic(
                        "held_resource_mismatch",
                        (*path, index, "call", "resources", "source"),
                        f"HandOver selects source {source_resource!r}, but the "
                        f"verified producer selects {producer[1]!r}.",
                        (producer[1],),
                    )
                effect_kind = SemanticEffectKind.TRANSFER
                dependencies.append(
                    SemanticEffectDependency(
                        producer_index=None if producer is None else producer[0],
                        consumer_index=index,
                        object=call.object,
                    )
                )
                latest_holder[call.object.entity_id] = (
                    index,
                    bound.binding.resource_ids["destination"],
                )
            elif type(call) is OperateArticulation:
                effect_kind = SemanticEffectKind.ARTICULATION
            else:
                effect_kind = SemanticEffectKind.REGISTERED
                # A registered extension has no declarative state-flow contract
                # in Version 1. Treat it as an opaque effect boundary.
                latest_holder.clear()
            effect_kinds.append(effect_kind)

        analyzed: list[AnalyzedSemanticCall] = []
        for index, (bound, effect_kind) in enumerate(
            zip(bound_calls, effect_kinds, strict=True)
        ):
            call = bound.linked.call
            requires_held = type(call) in (Place, HandOver)
            downstream_targets = (
                self._downstream_targets(index, bound_calls)
                if type(call) is Pick
                else ()
            )
            effect_monitor_ref = self._effect_monitor_ref(
                bound,
                effect_kind,
                path=(*path, index, "effect_monitor"),
            )
            symbolic_writes, opaque_symbolic_effect = self._static_symbolic_writes(
                bound,
                path=(*path, index, "call"),
            )
            analyzed.append(
                AnalyzedSemanticCall(
                    index=index,
                    bound=bound,
                    effect_kind=effect_kind,
                    symbolic_writes=symbolic_writes,
                    opaque_symbolic_effect=opaque_symbolic_effect,
                    effect_monitor_ref=effect_monitor_ref,
                    downstream_object_targets=downstream_targets,
                    requires_verified_held_object=requires_held,
                )
            )
        return SemanticWorkflow(
            workflow_id=workflow_id,
            calls=tuple(analyzed),
            effect_dependencies=tuple(dependencies),
            _compiler_id=self._compiler_id,
        )

    @staticmethod
    def _inherit_held_resource(
        call: SemanticCallSpec,
        latest_holder: Mapping[str, tuple[int, str]],
    ) -> SemanticCallSpec:
        """Fill an omitted consumer slot from the workflow's known holder."""
        if type(call) is Place:
            slot_id = "primary"
        elif type(call) is HandOver:
            slot_id = "source"
        else:
            return call
        holder = latest_holder.get(call.object.entity_id)
        if holder is None or slot_id in call.resources:
            return call
        resources = dict(call.resources)
        resources[slot_id] = holder[1]
        return replace(call, resources=resources)

    def _static_symbolic_writes(
        self,
        bound: BoundSemanticCall,
        *,
        path: tuple[PathPart, ...],
    ) -> tuple[frozenset[SymbolicStateKey], bool]:
        """Return exact provider-free ``TaskState`` keys for one linked call.

        Curated calls own these contracts.  Registered calls remain an opaque
        physical-effect boundary until their public descriptor grows an
        explicit static-effect contract; lowering arguments are never guessed.
        Conditional coordinated-held cleanup is likewise omitted because its
        exact pair keys depend on the verified input ``TaskState``.
        """
        call = bound.linked.call
        if type(call) in (Pick, Place):
            return (
                frozenset(
                    {
                        SymbolicStateKey.held_object(
                            self._participant_task_state_key(
                                bound,
                                slot_id="primary",
                                path=(*path, "resources", "primary"),
                            )
                        )
                    }
                ),
                False,
            )
        if type(call) is HandOver:
            return (
                frozenset(
                    SymbolicStateKey.held_object(
                        self._participant_task_state_key(
                            bound,
                            slot_id=slot_id,
                            path=(*path, "resources", slot_id),
                        )
                    )
                    for slot_id in ("source", "destination")
                ),
                False,
            )
        if type(call) is OperateArticulation:
            handle_ref = bound.linked.affordances.get("handle")
            if handle_ref is None:
                raise AssertionError(
                    "Linked articulation call lacks an operation affordance."
                )
            registration = self._integration.scene_registry.lookup(
                handle_ref,
                expected_type=SceneAffordanceRef,
            )
            affordance = registration.affordance
            if (
                type(affordance) is not ArticulationOperationAffordance
                or ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
                not in registration.affordance_capabilities
            ):
                raise _diagnostic(
                    "invalid_articulation_affordance",
                    (*path, "handle"),
                    f"Handle {handle_ref.entity_id!r} must expose an exact "
                    "ArticulationOperationAffordance payload and the articulation "
                    "operation capability.",
                )
            return (
                frozenset(
                    {
                        SymbolicStateKey.articulation_joint(
                            call.articulation.entity_id,
                            affordance.joint_id,
                        )
                    }
                ),
                False,
            )
        if type(call) is RegisteredSemanticCall:
            return frozenset(), True
        raise AssertionError(f"Unsupported linked call {type(call).__name__}.")

    @staticmethod
    def _participant_task_state_key(
        bound: BoundSemanticCall,
        *,
        slot_id: str,
        path: tuple[PathPart, ...],
    ) -> str:
        """Resolve the exact held-object key shared by participant endpoints."""
        resource = bound.binding.resources.get(slot_id)
        if resource is None:
            raise _diagnostic(
                "missing_effect_resource",
                path,
                f"Held-object effects require bound resource slot {slot_id!r}.",
                tuple(bound.binding.resources),
            )
        motion_endpoint = resource.endpoints.get("motion")
        grasp_endpoint = resource.endpoints.get("grasp")
        if motion_endpoint is None or grasp_endpoint is None:
            raise _diagnostic(
                "missing_effect_endpoint",
                (*path, "endpoints"),
                "Held-object effects require bound motion and grasp endpoints.",
                tuple(resource.endpoints),
            )
        task_state_key = motion_endpoint.task_state_key
        assert isinstance(task_state_key, str)
        if grasp_endpoint.task_state_key != task_state_key:
            raise _diagnostic(
                "effect_state_key_mismatch",
                (*path, "task_state_key"),
                "Motion and grasp endpoints for one participant must share one "
                "logical task-state key.",
            )
        return task_state_key

    def ground(
        self,
        workflow: SemanticWorkflow,
        call_index: int,
        context: PlanningContext,
        *,
        eligible_mask: torch.Tensor | None = None,
        revision: int = 0,
        path: tuple[PathPart, ...] = ("workflow",),
    ) -> GroundedSemanticCall:
        """Lower one analyzed call from the latest immutable observation.

        Args:
            workflow: Workflow created by this compiler.
            call_index: Zero-based call index to lower.
            context: Latest immutable planning observation.
            eligible_mask: Rows still eligible to execute this call.
            revision: Monotonic revision for re-grounding the same invocation.
            path: Root diagnostic path.

        Returns:
            Invocation and execution eligibility.

        Raises:
            SemanticValidationError: If workflow ownership, live integration,
                grounding, or verified state is invalid.
        """
        if type(workflow) is not SemanticWorkflow:
            raise TypeError("workflow must be exactly SemanticWorkflow.")
        if type(call_index) is not int or not 0 <= call_index < len(workflow.calls):
            raise IndexError(f"call_index {call_index!r} is outside the workflow.")
        if type(context) is not PlanningContext:
            raise TypeError("context must be exactly PlanningContext.")
        if type(revision) is not int or revision < 0:
            raise ValueError("revision must be a non-negative integer.")
        self._assert_workflow_current(workflow, path=path)
        self._integration.engine._validate_context(context)
        eligible = self._normalize_eligible_mask(eligible_mask, context)
        analyzed = workflow.calls[call_index]
        call = analyzed.call
        if type(call) is Pick:
            lowering = self._lower_pick(analyzed, context)
        elif type(call) is Place:
            lowering = self._lower_place(analyzed, context, eligible, path=path)
        elif type(call) is HandOver:
            lowering = self._lower_handover(analyzed, context, eligible, path=path)
        elif type(call) is OperateArticulation:
            lowering = self._lower_operate_articulation(
                analyzed,
                context,
                path=path,
            )
        elif type(call) is RegisteredSemanticCall:
            lowering = self._lower_registered(analyzed, context, path=path)
        else:  # pragma: no cover - exact workflow construction prevents this
            raise AssertionError(f"Unsupported analyzed call {type(call).__name__}.")

        bound = analyzed.bound
        invocation = ActionInvocation(
            skill_id=bound.linked.descriptor.skill_id,
            goal=lowering.goal,
            binding=bound.binding.action_binding,
            motion_policy=bound.preset.motion_policy,
            tracking_policy=bound.preset.tracking_policy,
            recovery_policy=bound.preset.recovery_policy,
            skill_options=lowering.skill_options,
            control_overrides=lowering.control_overrides,
            invocation_id=f"{workflow.workflow_id}:{call_index}",
            revision=revision,
        )
        effect_spec = self._ground_effect_spec(
            analyzed,
            invocation,
            context,
            path=(*path, call_index, "effect"),
        )
        effect_monitor: EffectMonitor | None = None
        if effect_spec is not None and analyzed.effect_monitor_ref is not None:
            try:
                effect_monitor = self._effect_monitor_registry.create(
                    effect_spec,
                    analyzed.effect_monitor_ref,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise _diagnostic(
                    "effect_monitor_creation_failed",
                    (*path, call_index, "effect_monitor"),
                    f"Could not create the grounded effect monitor: {exc}",
                ) from exc
        return GroundedSemanticCall._create(
            analyzed=analyzed,
            invocation=invocation,
            effect_spec=effect_spec,
            effect_monitor=effect_monitor,
            eligible_mask=eligible,
        )

    def _assert_current(self, *, path: tuple[PathPart, ...]) -> None:
        """Reject a compiler after engine profile/catalog ownership changes."""
        engine = self._integration.engine
        if engine.skill_profile is not self._integration.robot_profile:
            raise _diagnostic(
                "semantic_profile_stale",
                path,
                "The engine's canonical robot profile changed after compiler "
                "construction.",
            )
        try:
            _ = self._integration.robot_profile.skills
        except RuntimeError as exc:
            raise _diagnostic(
                "semantic_catalog_stale",
                path,
                str(exc),
            ) from exc

    def _assert_workflow_current(
        self,
        workflow: SemanticWorkflow,
        *,
        path: tuple[PathPart, ...],
    ) -> None:
        """Ensure a workflow belongs to this still-current engine revision."""
        self._assert_current(path=("integration", "robot_profile"))
        if workflow._compiler_id != self._compiler_id:
            raise _diagnostic(
                "semantic_program_stale",
                path,
                "The workflow belongs to a different compiler/grounder registry.",
            )

    @staticmethod
    def _normalize_eligible_mask(
        eligible_mask: torch.Tensor | None,
        context: PlanningContext,
    ) -> torch.Tensor:
        """Return one owned per-row eligibility mask."""
        if eligible_mask is None:
            return torch.ones(
                context.batch_size,
                dtype=torch.bool,
                device=context.robot.qpos.device,
            )
        if not isinstance(eligible_mask, torch.Tensor):
            raise TypeError("eligible_mask must be a torch.Tensor or None.")
        if eligible_mask.dtype != torch.bool or eligible_mask.shape != (
            context.batch_size,
        ):
            raise ValueError(
                "eligible_mask must be a bool tensor matching the context batch."
            )
        if eligible_mask.device != context.robot.qpos.device:
            raise ValueError("eligible_mask must use the context device.")
        return eligible_mask.clone()

    def _validate_context(self, context: PlanningContext) -> None:
        """Require the grounding observation to match the bound engine batch."""
        engine = self._integration.engine
        if context.robot.robot_dof != engine.robot.dof:
            raise ValueError(
                "PlanningContext robot_dof must match the compiler engine, "
                f"got {context.robot.robot_dof} and {engine.robot.dof}."
            )
        engine_qpos = engine.robot.get_qpos()
        if context.batch_size != int(engine_qpos.shape[0]):
            raise ValueError(
                "PlanningContext batch size must match the compiler engine, "
                f"got {context.batch_size} and {engine_qpos.shape[0]}."
            )
        if context.robot.qpos.device != engine.device:
            raise ValueError("PlanningContext and compiler engine must share a device.")

    def _effect_monitor_ref(
        self,
        bound: BoundSemanticCall,
        effect_kind: SemanticEffectKind,
        *,
        path: tuple[PathPart, ...],
    ) -> EffectMonitorRef | None:
        """Resolve one preset-owned exact monitor reference without creating it."""
        semantic_id = bound.linked.call.semantic_id
        monitor_ref = bound.preset.effect_monitors.get(semantic_id)
        if monitor_ref is None:
            if type(bound.linked.call) in (
                Pick,
                Place,
                HandOver,
                OperateArticulation,
            ):
                raise _diagnostic(
                    "missing_effect_monitor",
                    path,
                    f"Semantic call {semantic_id!r} requires an effect monitor "
                    f"for its {effect_kind.value!r} postcondition.",
                    tuple(bound.preset.effect_monitors),
                )
            return None
        if type(bound.linked.call) is RegisteredSemanticCall:
            raise _diagnostic(
                "registered_effect_contract_not_installed",
                path,
                f"Registered semantic call {semantic_id!r} selects an effect "
                "monitor but no declarative effect-contract grounder is "
                "installed.",
            )
        try:
            self._effect_monitor_registry.validate_ref(monitor_ref)
        except KeyError as exc:
            available = tuple(
                f"{monitor_id}@{revision}"
                for monitor_id, revision in self._effect_monitor_registry.factories
            )
            raise _diagnostic(
                "effect_monitor_not_installed",
                path,
                f"Effect monitor {monitor_ref.monitor_id!r} revision "
                f"{monitor_ref.revision!r} is not installed.",
                available,
            ) from exc
        except (TypeError, ValueError) as exc:
            raise _diagnostic(
                "invalid_effect_monitor_config",
                path,
                f"Effect monitor {monitor_ref.monitor_id!r} revision "
                f"{monitor_ref.revision!r} has invalid configuration: {exc}",
            ) from exc
        return monitor_ref.snapshot()

    def _downstream_targets(
        self,
        pick_index: int,
        bound_calls: list[BoundSemanticCall],
    ) -> tuple[SemanticObjectTarget, ...]:
        """Propagate object targets until the picked object is released."""
        pick = bound_calls[pick_index].linked.call
        assert type(pick) is Pick
        object_id = pick.object.entity_id
        targets: list[SemanticObjectTarget] = []
        for call_index, bound in enumerate(
            bound_calls[pick_index + 1 :],
            start=pick_index + 1,
        ):
            call = bound.linked.call
            if type(call) is RegisteredSemanticCall:
                break
            call_object = getattr(call, "object", None)
            if type(call_object) is not SceneObjectRef or (
                call_object.entity_id != object_id
            ):
                continue
            if type(call) is Pick:
                break
            if type(call) is HandOver:
                provider_id, _ = self._require_handover_pose_provider(
                    call,
                    path=("workflow", call_index, "call"),
                )
                targets.append(
                    SemanticObjectTarget(
                        SemanticHandOverTarget(
                            provider_id=provider_id,
                            bound=bound,
                        )
                    )
                )
                break
            if type(call) is Place:
                if call.at is not None:
                    targets.append(SemanticObjectTarget(call.at))
                else:
                    targets.append(self._relation_target(bound))
                break
        return tuple(targets)

    def _lower_pick(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
    ) -> SemanticLowering:
        """Lower object-centric pickup and its downstream look-ahead."""
        call = analyzed.call
        assert type(call) is Pick
        grasp_ref = analyzed.bound.linked.affordances.get("grasp")
        if grasp_ref is None:
            raise AssertionError("Linked pick call lacks a grasp affordance.")
        semantics = self._integration.scene_registry.object_semantics(
            call.object,
            affordance=grasp_ref,
        )
        return SemanticLowering(
            goal=GraspGoal(semantics=semantics),
            skill_options=PickUpOptions(
                downstream_object_target_poses=tuple(
                    self._ground_object_target(target, context)
                    for target in analyzed.downstream_object_targets
                )
            ),
        )

    def _lower_place(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
        eligible: torch.Tensor,
        *,
        path: tuple[PathPart, ...],
    ) -> SemanticLowering:
        """Convert an object-space place target using verified held state."""
        call = analyzed.call
        assert type(call) is Place
        task_state_key, held = self._require_held_object(
            analyzed,
            context,
            eligible,
            slot_id="primary",
            path=(*path, analyzed.index, "call"),
        )
        del task_state_key
        if call.at is not None:
            object_target = self._broadcast_pose(
                call.at.to_matrix(),
                context,
                name="Place.at",
            )
            xpos: PoseGoalValue = torch.bmm(object_target, held.object_to_eef)
        else:
            object_target = self._ground_object_target(
                self._relation_target(analyzed.bound),
                context,
            )
            xpos = self._compose_object_to_eef(
                object_target, held.object_to_eef, context
            )
        return SemanticLowering(goal=PlaceGoal(xpos=xpos), skill_options=PlaceOptions())

    def _lower_handover(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
        eligible: torch.Tensor,
        *,
        path: tuple[PathPart, ...],
    ) -> SemanticLowering:
        """Lower handover through an explicitly installed embodiment provider."""
        call = analyzed.call
        assert type(call) is HandOver
        self._require_held_object(
            analyzed,
            context,
            eligible,
            slot_id="source",
            path=(*path, analyzed.index, "call"),
        )
        _, provider = self._require_handover_pose_provider(
            call,
            path=(*path, analyzed.index, "call"),
        )
        targets = self._resolve_handover_targets(
            provider,
            call,
            context=context,
            bound=analyzed.bound,
        )
        grasp_ref = analyzed.bound.linked.affordances.get("receiver_grasp")
        if grasp_ref is None:
            raise AssertionError("Linked handover lacks receiver grasp affordance.")
        semantics = self._integration.scene_registry.object_semantics(
            call.object,
            affordance=grasp_ref,
        )
        middle = self._ground_object_target(targets.middle, context)
        final_target = (
            SemanticObjectTarget(call.final_target)
            if call.final_target is not None
            else targets.final
        )
        final = self._ground_object_target(final_target, context)
        return SemanticLowering(
            goal=GraspGoal(semantics=semantics),
            skill_options=HandOverOptions(
                middle_object_pose=middle,
                final_object_pose=final,
            ),
        )

    def _lower_operate_articulation(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
        *,
        path: tuple[PathPart, ...],
    ) -> SemanticLowering:
        """Ground one handle operation from the latest scene snapshot."""
        call = analyzed.call
        assert type(call) is OperateArticulation
        handle_ref = analyzed.bound.linked.affordances.get("handle")
        if handle_ref is None:
            raise AssertionError(
                "Linked articulation call lacks an operation affordance."
            )
        registration = self._integration.scene_registry.lookup(
            handle_ref,
            expected_type=SceneAffordanceRef,
        )
        affordance = registration.affordance
        if (
            type(affordance) is not ArticulationOperationAffordance
            or ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
            not in registration.affordance_capabilities
        ):
            raise _diagnostic(
                "invalid_articulation_affordance",
                (*path, analyzed.index, "call", "handle"),
                f"Handle {handle_ref.entity_id!r} must expose an exact "
                "ArticulationOperationAffordance payload and the articulation "
                "operation capability.",
            )

        if call.target is not None:
            try:
                resolved_target = affordance.resolve_target(call.target)
            except KeyError as exc:
                raise _diagnostic(
                    "unknown_articulation_target",
                    (*path, analyzed.index, "call", "target"),
                    f"Handle {handle_ref.entity_id!r} has no semantic target "
                    f"{call.target!r}.",
                    tuple(affordance.semantic_targets),
                ) from exc
            target_position = resolved_target.target_position
            displacement = resolved_target.displacement
        else:
            assert call.target_position is not None
            assert call.target_displacement is not None
            target_position = call.target_position
            displacement = call.target_displacement

        try:
            handle_state = context.scene.entities[handle_ref.entity_id]
        except KeyError as exc:
            raise _diagnostic(
                "missing_handle_observation",
                (*path, analyzed.index, "call", "handle"),
                f"The current planning snapshot has no pose for handle "
                f"{handle_ref.entity_id!r}.",
            ) from exc
        try:
            self._broadcast_pose(
                handle_state.pose,
                context,
                name=f"handle {handle_ref.entity_id!r}",
            )
        except (TypeError, ValueError) as exc:
            raise _diagnostic(
                "articulation_grounding_failed",
                (*path, analyzed.index, "call", "handle"),
                f"Could not ground articulation handle geometry: {exc}",
            ) from exc
        joint_address = call.articulation.entity_id, affordance.joint_id
        observed_joint = context.scene.get_articulation_joint_state(*joint_address)
        if observed_joint is None:
            raise _diagnostic(
                "missing_articulation_joint_observation",
                (*path, analyzed.index, "call", "articulation"),
                "Recovery-safe articulation grounding requires a live "
                "ObservedArticulationJointState for "
                f"{joint_address!r} in the current scene snapshot.",
            )
        try:
            source_position = self._broadcast_joint_position(
                observed_joint.position,
                context,
                name=f"articulation joint {joint_address!r}",
            )
        except (TypeError, ValueError) as exc:
            raise _diagnostic(
                "invalid_articulation_joint_observation",
                (*path, analyzed.index, "call", "articulation"),
                f"Could not use live articulation joint state: {exc}",
            ) from exc
        if observed_joint.valid_mask is not None:
            valid = observed_joint.valid_mask.to(device=context.robot.qpos.device)
            if bool((~valid).any()):
                rows = (~valid).nonzero(as_tuple=False).flatten().tolist()
                raise _diagnostic(
                    "invalid_articulation_joint_observation",
                    (*path, analyzed.index, "call", "articulation"),
                    "Live articulation joint state is unavailable for planning "
                    f"rows {rows}.",
                )
        target = torch.full(
            (context.batch_size, 1),
            target_position,
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        )
        return SemanticLowering(
            goal=OperateArticulationGoal(
                articulation_id=call.articulation.entity_id,
                joint_id=affordance.joint_id,
                geometry=SceneArticulationOperationGeometry(
                    handle_pose=SceneEntityPose(handle_ref.entity_id),
                    approach_offset=affordance.approach_offset,
                    contact_offset=affordance.contact_offset,
                    operation_offset=affordance.operation_offset,
                    retract_offset=affordance.retract_offset,
                    operation_axis=affordance.operation_axis,
                    position_scale=affordance.position_scale,
                ),
                source_position=source_position,
                target_position=target,
                target_displacement=displacement,
            )
        )

    def _lower_registered(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
        *,
        path: tuple[PathPart, ...],
    ) -> SemanticLowering:
        """Invoke one explicitly installed registered-call lowerer."""
        call = analyzed.call
        assert type(call) is RegisteredSemanticCall
        lowerer = self._registered_lowerers.get(call.call_id)
        if lowerer is None:
            raise _diagnostic(
                "semantic_lowerer_not_installed",
                (*path, analyzed.index, "call", "kind"),
                f"No lowerer is installed for {call.call_id!r}.",
                tuple(self._registered_lowerers),
            )
        lowering = lowerer.lower(
            call,
            context=context,
            bound=analyzed.bound,
        )
        if type(lowering) is not SemanticLowering:
            raise TypeError(
                "RegisteredSemanticLowerer.lower() must return exactly "
                "SemanticLowering."
            )
        descriptor = analyzed.bound.linked.descriptor
        target = descriptor.target_descriptor
        assert target is not None
        expected_goal_types = (
            target.goal_type
            if isinstance(target.goal_type, tuple)
            else (target.goal_type,)
        )
        if type(lowering.goal) not in expected_goal_types:
            raise TypeError(
                f"Lowerer {call.call_id!r} produced {type(lowering.goal).__name__}; "
                f"target skill {target.skill_id!r} expects {target.goal_type!r}."
            )
        if lowering.skill_options is not None and (
            type(lowering.skill_options) is not target.options_type
        ):
            raise TypeError(
                f"Lowerer {call.call_id!r} produced incompatible skill options."
            )
        return lowering

    def _ground_effect_spec(
        self,
        analyzed: AnalyzedSemanticCall,
        invocation: ActionInvocation,
        context: PlanningContext,
        *,
        path: tuple[PathPart, ...],
    ) -> SemanticEffectSpec | None:
        """Ground typed symbolic state and raw-evidence clauses."""
        if analyzed.effect_monitor_ref is None:
            return None
        call = analyzed.call
        state_expectations: list[EffectStateExpectation] = []
        clauses: list[EffectClause] = []
        if type(call) is Pick:
            expectation, grounded_clauses = self._ground_held_effect(
                analyzed,
                expectation_id="destination",
                relation=HeldObjectRelation.ATTACHED,
                slot_id="primary",
                object_id=call.object.entity_id,
                context=context,
                path=(*path, "state_expectations", "destination"),
            )
            state_expectations.append(expectation)
            clauses.extend(grounded_clauses)
            state_expectations.extend(
                self._coordinated_cleanup_expectations(
                    context,
                    task_state_keys=(expectation.task_state_key,),
                )
            )
        elif type(call) is Place:
            expectation, grounded_clauses = self._ground_held_effect(
                analyzed,
                expectation_id="source",
                relation=HeldObjectRelation.DETACHED,
                slot_id="primary",
                object_id=call.object.entity_id,
                context=context,
                path=(*path, "state_expectations", "source"),
            )
            state_expectations.append(expectation)
            clauses.extend(grounded_clauses)
            state_expectations.extend(
                self._coordinated_cleanup_expectations(
                    context,
                    task_state_keys=(expectation.task_state_key,),
                )
            )
        elif type(call) is HandOver:
            source, source_clauses = self._ground_held_effect(
                analyzed,
                expectation_id="source",
                relation=HeldObjectRelation.DETACHED,
                slot_id="source",
                object_id=call.object.entity_id,
                context=context,
                path=(*path, "state_expectations", "source"),
            )
            destination, destination_clauses = self._ground_held_effect(
                analyzed,
                expectation_id="destination",
                relation=HeldObjectRelation.ATTACHED,
                slot_id="destination",
                object_id=call.object.entity_id,
                context=context,
                path=(*path, "state_expectations", "destination"),
            )
            state_expectations.extend((source, destination))
            clauses.extend((*source_clauses, *destination_clauses))
        elif type(call) is OperateArticulation:
            goal = invocation.goal
            if type(goal) is not OperateArticulationGoal:
                raise AssertionError(
                    "OperateArticulation lowering produced an incompatible goal."
                )
            expectation = ArticulationJointStateExpectation(
                expectation_id="joint",
                articulation_id=goal.articulation_id,
                joint_id=goal.joint_id,
                target_position=goal.target_position,
            )
            source = EffectEvidenceSourceRef(
                provider_id=SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
                revision=SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
                address=ArticulationJointEvidenceAddress(
                    articulation_id=goal.articulation_id,
                    joint_id=goal.joint_id,
                ),
            )
            state_expectations.append(expectation)
            clauses.append(
                JointStateEffectClause(
                    clause_id="joint.position",
                    expectation_id=expectation.expectation_id,
                    source=source,
                    target_position=goal.target_position,
                )
            )
        else:  # pragma: no cover - exact workflow construction prevents this
            raise AssertionError(f"Unsupported analyzed call {type(call).__name__}.")
        return SemanticEffectSpec(
            semantic_id=call.semantic_id,
            effect_kind=analyzed.effect_kind,
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            env_ids=context.env_ids,
            state_expectations=tuple(state_expectations),
            clauses=tuple(clauses),
        )

    @staticmethod
    def _coordinated_cleanup_expectations(
        context: PlanningContext,
        *,
        task_state_keys: tuple[str, ...],
    ) -> tuple[CoordinatedHeldObjectCleanupExpectation, ...]:
        """Declare the exact coordinated relations a primitive must remove."""
        related = set(task_state_keys)
        return tuple(
            CoordinatedHeldObjectCleanupExpectation(
                expectation_id=f"cleanup:{resources[0]}:{resources[1]}",
                task_state_keys=resources,
            )
            for resources in context.task.coordinated_held_objects
            if not set(resources).isdisjoint(related)
        )

    @staticmethod
    def _effect_source(
        sources: Mapping[str, EffectEvidenceSourceRef],
        channel: str,
        *,
        path: tuple[PathPart, ...],
    ) -> EffectEvidenceSourceRef:
        """Resolve one exact endpoint-owned observation source."""
        source = sources.get(channel)
        if source is None:
            raise _diagnostic(
                "missing_effect_source",
                (*path, "effect_sources", channel),
                f"The endpoint does not expose required effect channel {channel!r}.",
                tuple(sources),
            )
        return source.snapshot()

    def _ground_held_effect(
        self,
        analyzed: AnalyzedSemanticCall,
        *,
        expectation_id: str,
        relation: HeldObjectRelation,
        slot_id: str,
        object_id: str,
        context: PlanningContext,
        path: tuple[PathPart, ...],
    ) -> tuple[HeldObjectStateExpectation, tuple[EffectClause, ...]]:
        """Bind one held-object state relation to generic endpoint sources."""
        resource = analyzed.bound.binding.resources[slot_id]
        motion_endpoint = resource.endpoints.get("motion")
        grasp_endpoint = resource.endpoints.get("grasp")
        if motion_endpoint is None or grasp_endpoint is None:
            raise _diagnostic(
                "missing_effect_endpoint",
                (*path, "endpoints"),
                "Held-object effects require bound motion and grasp endpoints.",
                tuple(resource.endpoints),
            )
        task_state_key = motion_endpoint.task_state_key
        assert isinstance(task_state_key, str)
        if grasp_endpoint.task_state_key != task_state_key:
            raise _diagnostic(
                "effect_state_key_mismatch",
                (*path, "task_state_key"),
                "Motion and grasp endpoints for one participant must share one "
                "logical task-state key.",
            )
        baseline: torch.Tensor | None = None
        if relation is HeldObjectRelation.DETACHED:
            held = context.task.get_held_object(task_state_key)
            if held is None or held.semantics.entity_id != object_id:
                raise _diagnostic(
                    "verified_held_object_required",
                    (*path, "baseline"),
                    f"Detached relation requires verified object {object_id!r} "
                    f"held under logical state key {task_state_key!r}.",
                )
            baseline = held.object_to_eef
        state_expectation = HeldObjectStateExpectation(
            expectation_id=expectation_id,
            relation=relation,
            object_id=object_id,
            slot_id=slot_id,
            resource_id=resource.resource_id,
            task_state_key=task_state_key,
        )
        pose_source = self._effect_source(
            motion_endpoint.effect_sources,
            POSE_RELATION_EFFECT_CHANNEL,
            path=(*path, "motion"),
        )
        binary_channel = (
            CONSTRAINT_EFFECT_CHANNEL
            if CONSTRAINT_EFFECT_CHANNEL in grasp_endpoint.effect_sources
            else CONTACT_EFFECT_CHANNEL
        )
        binary_source = self._effect_source(
            grasp_endpoint.effect_sources,
            binary_channel,
            path=(*path, "grasp"),
        )
        pose_clause = PoseRelationClause(
            clause_id=f"{expectation_id}.pose",
            expectation_id=expectation_id,
            source=pose_source,
            expectation=(
                PoseRelationExpectation.MATCHED
                if relation is HeldObjectRelation.ATTACHED
                else PoseRelationExpectation.SEPARATED
            ),
            baseline_object_to_endpoint=baseline,
        )
        binary_kind = (
            BinaryEvidenceKind.CONSTRAINT
            if binary_channel == CONSTRAINT_EFFECT_CHANNEL
            else BinaryEvidenceKind.CONTACT
        )
        binary_clause = BinaryEffectClause(
            clause_id=f"{expectation_id}.{binary_kind.value}",
            expectation_id=expectation_id,
            source=binary_source,
            evidence_kind=binary_kind,
            expected=relation is HeldObjectRelation.ATTACHED,
        )
        return state_expectation, (pose_clause, binary_clause)

    def _relation_target(
        self,
        bound: BoundSemanticCall,
    ) -> SemanticObjectTarget:
        """Describe a linked placement relation without observing providers."""
        call = bound.linked.call
        assert type(call) is Place and call.at is None
        capability = (
            PLACE_ON_AFFORDANCE_CAPABILITY
            if call.on is not None
            else PLACE_IN_AFFORDANCE_CAPABILITY
        )
        affordance_ref = bound.linked.affordances.get("destination")
        if affordance_ref is None:
            raise AssertionError("Linked relation place lacks destination affordance.")
        metadata = self._integration.manifest.scene.lookup(
            affordance_ref,
            expected_type=SceneAffordanceRef,
        )
        if (
            metadata.affordance_payload_type is None
            or metadata.affordance_revision is None
        ):
            raise AssertionError(
                "Capability-bearing relation affordance lacks payload metadata."
            )
        return SemanticObjectTarget(
            SemanticRelationTarget(
                capability=capability,
                affordance=affordance_ref,
                payload_type=metadata.affordance_payload_type,
                payload_revision=metadata.affordance_revision,
            )
        )

    def _require_relation_grounder(
        self,
        relation: SemanticRelationTarget | None,
        *,
        path: tuple[PathPart, ...],
    ) -> RelationTargetGrounder:
        """Resolve one exact relation grounder or fail during static analysis."""
        assert relation is not None
        grounder = self._relation_grounders.get(relation.grounder_key)
        if grounder is None:
            candidates = tuple(
                f"{capability}:{payload_type.__name__}:{revision}"
                for capability, payload_type, revision in self._relation_grounders
            )
            raise _diagnostic(
                "relation_grounder_not_installed",
                path,
                "No relation target grounder is installed for "
                f"{relation.capability!r}, {relation.payload_type.__name__}, "
                f"revision {relation.payload_revision!r}.",
                candidates,
            )
        return grounder

    def _ground_object_target(
        self,
        target: SemanticObjectTarget,
        context: PlanningContext,
    ) -> PoseGoalValue:
        """Ground a direct pose or dispatch one typed relation grounder."""
        if type(target.pose) is SemanticPose:
            return target.pose.to_matrix()
        if type(target.pose) is SceneEntityPose:
            return target.pose
        deferred_handover = target.handover
        if deferred_handover is not None:
            call = deferred_handover.bound.linked.call
            assert type(call) is HandOver
            provider_id, provider = self._require_handover_pose_provider(
                call,
                path=("handover", "provider"),
            )
            if provider_id != deferred_handover.provider_id:
                raise _diagnostic(
                    "semantic_program_stale",
                    ("handover", "provider"),
                    "The profile-selected handover provider changed after "
                    "workflow analysis.",
                )
            targets = self._resolve_handover_targets(
                provider,
                call,
                context=context,
                bound=deferred_handover.bound,
            )
            return self._ground_object_target(targets.middle, context)
        relation = target.relation
        assert relation is not None
        grounder = self._require_relation_grounder(
            relation,
            path=("relation", relation.affordance.entity_id),
        )
        registration = self._integration.scene_registry.lookup(
            relation.affordance,
            expected_type=SceneAffordanceRef,
        )
        affordance = registration.affordance
        assert affordance is not None
        if (
            type(affordance) is not relation.payload_type
            or registration.affordance_revision != relation.payload_revision
            or relation.capability not in registration.affordance_capabilities
        ):
            raise TypeError(
                "Semantic relation target does not match the exact live "
                "affordance type, capability, and revision."
            )
        pose_goal = grounder.ground(
            relation,
            affordance=affordance,
            context=context,
        )
        if type(pose_goal) is not SceneEntityPose and not isinstance(
            pose_goal, torch.Tensor
        ):
            raise TypeError(
                "RelationTargetGrounder.ground() must return a torch.Tensor or "
                "exact SceneEntityPose."
            )
        return pose_goal

    def _require_handover_pose_provider(
        self,
        call: HandOver,
        *,
        path: tuple[PathPart, ...],
    ) -> tuple[str, HandOverPoseProvider]:
        """Resolve the profile-selected named handover grounding provider."""
        provider_id = (
            self._integration.robot_profile.source_profile.grounding_providers.get(
                call.semantic_id
            )
        )
        if provider_id is None:
            raise _diagnostic(
                "handover_grounding_unconfigured",
                path,
                "The robot profile must select a named grounding provider for "
                f"semantic call {call.semantic_id!r}.",
                tuple(self._handover_pose_providers),
            )
        provider = self._handover_pose_providers.get(provider_id)
        if provider is None:
            raise _diagnostic(
                "handover_grounding_provider_not_installed",
                path,
                f"Robot profile selects handover provider {provider_id!r}, but "
                "the compiler did not install it.",
                tuple(self._handover_pose_providers),
            )
        return provider_id, provider

    @staticmethod
    def _resolve_handover_targets(
        provider: HandOverPoseProvider,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Run one provider and reject recursive deferred target values."""
        targets = provider.resolve(call, context=context, bound=bound)
        if type(targets) is not HandOverPoseTargets:
            raise TypeError(
                "HandOverPoseProvider.resolve() must return exactly "
                "HandOverPoseTargets."
            )
        if targets.middle.handover is not None or targets.final.handover is not None:
            raise TypeError(
                "HandOverPoseProvider targets cannot recursively defer to another "
                "handover provider."
            )
        return targets

    def _compose_object_to_eef(
        self,
        object_target: PoseGoalValue,
        object_to_eef: torch.Tensor,
        context: PlanningContext,
    ) -> PoseGoalValue:
        """Compose a relation-grounded object target with verified held state."""
        if isinstance(object_target, torch.Tensor):
            return torch.bmm(
                self._broadcast_pose(object_target, context, name="relation target"),
                object_to_eef,
            )
        relative = object_target.relative_pose
        if relative is None:
            composed = object_to_eef.clone()
        else:
            composed = torch.bmm(
                self._broadcast_pose(relative, context, name="relation offset"),
                object_to_eef,
            )
        return SceneEntityPose(
            object_target.entity_id,
            relative_pose=composed,
            minimum_confidence=object_target.minimum_confidence,
        )

    def _require_held_object(
        self,
        analyzed: AnalyzedSemanticCall,
        context: PlanningContext,
        eligible: torch.Tensor,
        *,
        slot_id: str,
        path: tuple[PathPart, ...],
    ) -> tuple[str, HeldObjectState]:
        """Resolve the logical participant key and verify held-object identity."""
        resource = analyzed.bound.binding.resources[slot_id]
        endpoint = resource.endpoints.get("motion")
        if endpoint is None:
            raise _diagnostic(
                "missing_effect_endpoint",
                (*path, "resources", slot_id, "motion"),
                "The semantic lowerer requires a bound motion endpoint.",
                tuple(resource.endpoints),
            )
        task_state_key = endpoint.task_state_key
        assert isinstance(task_state_key, str)
        held = context.task.get_held_object(task_state_key)
        call_object = getattr(analyzed.call, "object", None)
        assert type(call_object) is SceneObjectRef
        if held is None or held.semantics.entity_id != call_object.entity_id:
            raise _diagnostic(
                "verified_held_object_required",
                (*path, "object"),
                f"Call requires verified object {call_object.entity_id!r} held by "
                f"logical state key {task_state_key!r}.",
            )
        assert held.env_mask is not None
        missing = eligible & ~held.env_mask
        if missing.any():
            missing_env_ids = tuple(
                str(value)
                for value in context.env_ids[missing].detach().to("cpu").tolist()
            )
            raise _diagnostic(
                "verified_held_object_required",
                (*path, "object"),
                f"Object {call_object.entity_id!r} is not verified as held in "
                "every eligible environment.",
                missing_env_ids,
            )
        return task_state_key, held

    @staticmethod
    def _broadcast_pose(
        pose: torch.Tensor,
        context: PlanningContext,
        *,
        name: str,
    ) -> torch.Tensor:
        """Move and broadcast one object-space pose to the planning batch."""
        pose = pose.to(device=context.robot.qpos.device, dtype=torch.float32)
        if pose.shape == (4, 4):
            return pose.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        if pose.shape != (context.batch_size, 4, 4):
            raise ValueError(
                f"{name} must have shape (4, 4) or " f"({context.batch_size}, 4, 4)."
            )
        return pose.clone()

    @staticmethod
    def _broadcast_joint_position(
        position: torch.Tensor,
        context: PlanningContext,
        *,
        name: str,
    ) -> torch.Tensor:
        """Move and broadcast one scalar articulation joint observation."""
        if not isinstance(position, torch.Tensor):
            raise TypeError(f"{name} position must be a torch.Tensor.")
        if not position.is_floating_point() or not torch.isfinite(position).all():
            raise ValueError(f"{name} position must be a finite floating tensor.")
        position = position.to(
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        if position.shape == (1,):
            return position.unsqueeze(0).expand(context.batch_size, -1).clone()
        if position.shape != (context.batch_size, 1):
            raise ValueError(
                f"{name} position must have shape (1,) or "
                f"({context.batch_size}, 1)."
            )
        return position.clone()


__all__ = [
    "AnalyzedSemanticCall",
    "GroundedSemanticCall",
    "HandOverPoseProvider",
    "HandOverPoseTargets",
    "RelationTargetGrounder",
    "RegisteredSemanticLowerer",
    "SemanticEffectDependency",
    "SemanticEffectKind",
    "SemanticHandOverTarget",
    "SemanticLowering",
    "SemanticObjectTarget",
    "SemanticRelationTarget",
    "SemanticSkillCompiler",
    "SemanticWorkflow",
]
