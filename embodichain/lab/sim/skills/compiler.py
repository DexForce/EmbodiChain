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
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import ClassVar
from uuid import uuid4

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionControlOverrides,
    ActionInvocation,
    ActionOptions,
    Affordance,
    GraspGoal,
    HandOverOptions,
    JointPositionTarget,
    HeldObjectState,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    PoseGoalValue,
    SceneEntityPose,
)
from .calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
    SemanticPose,
)
from .integration import (
    BoundSemanticCall,
    BoundSemanticIntegration,
    PathPart,
    SemanticDiagnostic,
    SemanticValidationError,
)
from .scene import (
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


class SemanticEffectKind(str, Enum):
    """Symbolic effect boundary inferred for a semantic call."""

    ATTACH = "attach"
    RELEASE = "release"
    TRANSFER = "transfer"
    REGISTERED = "registered"


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
    downstream_object_target: SemanticObjectTarget | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a non-negative integer.")
        if type(self.bound) is not BoundSemanticCall:
            raise TypeError("bound must be exactly BoundSemanticCall.")
        if not isinstance(self.effect_kind, SemanticEffectKind):
            raise TypeError("effect_kind must be a SemanticEffectKind.")
        if self.downstream_object_target is not None and (
            type(self.downstream_object_target) is not SemanticObjectTarget
        ):
            raise TypeError(
                "downstream_object_target must be exactly SemanticObjectTarget "
                "or None."
            )

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
    _eligible_mask: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.analyzed) is not AnalyzedSemanticCall:
            raise TypeError("analyzed must be exactly AnalyzedSemanticCall.")
        if type(self.invocation) is not ActionInvocation:
            raise TypeError("invocation must be exactly ActionInvocation.")
        if self.invocation.skill_id != self.analyzed.bound.linked.descriptor.skill_id:
            raise ValueError("invocation skill_id must match the analyzed call.")
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
    ) -> None:
        """Install immutable semantic lowering and grounding registries.

        Args:
            integration: Exact live scene, engine, and robot-profile binding.
            registered_lowerers: Explicit implementations for registered calls.
            relation_grounders: Exact capability/payload/revision dispatch entries.
            handover_pose_providers: Named embodiment-owned handover providers.
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
        allowed_types = (Pick, Place, HandOver, RegisteredSemanticCall)
        if not all(type(call) in allowed_types for call in supplied):
            raise TypeError("calls must contain exact supported semantic call values.")

        bound_calls: list[BoundSemanticCall] = []
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
            bound_calls.append(
                self._integration.link_call(
                    call,
                    path=(*path, index, "call"),
                )
            )
        for index, bound in enumerate(bound_calls):
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

        dependencies: list[SemanticEffectDependency] = []
        latest_holder: dict[str, tuple[int, str]] = {}
        analyzed: list[AnalyzedSemanticCall] = []
        for index, bound in enumerate(bound_calls):
            call = bound.linked.call
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
            else:
                effect_kind = SemanticEffectKind.REGISTERED
                # A registered extension has no declarative state-flow contract
                # in Version 1. Treat it as an opaque effect boundary.
                latest_holder.clear()
            downstream_target = (
                self._downstream_target(index, bound_calls)
                if type(call) is Pick
                else None
            )
            analyzed.append(
                AnalyzedSemanticCall(
                    index=index,
                    bound=bound,
                    effect_kind=effect_kind,
                    downstream_object_target=downstream_target,
                )
            )
        return SemanticWorkflow(
            workflow_id=workflow_id,
            calls=tuple(analyzed),
            effect_dependencies=tuple(dependencies),
            _compiler_id=self._compiler_id,
        )

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
            recovery_policy=bound.preset.recovery_policy,
            skill_options=lowering.skill_options,
            control_overrides=lowering.control_overrides,
            invocation_id=f"{workflow.workflow_id}:{call_index}",
            revision=revision,
        )
        return GroundedSemanticCall(
            analyzed=analyzed,
            invocation=invocation,
            _eligible_mask=eligible,
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

    def _downstream_target(
        self,
        pick_index: int,
        bound_calls: list[BoundSemanticCall],
    ) -> SemanticObjectTarget | None:
        """Return the first target at which the picked object is released."""
        pick = bound_calls[pick_index].linked.call
        assert type(pick) is Pick
        object_id = pick.object.entity_id
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
                return SemanticObjectTarget(
                    SemanticHandOverTarget(
                        provider_id=provider_id,
                        bound=bound,
                    )
                )
            if type(call) is Place:
                if call.at is not None:
                    return SemanticObjectTarget(call.at)
                return self._relation_target(bound)
        return None

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
                downstream_object_target_poses=(
                    ()
                    if analyzed.downstream_object_target is None
                    else (
                        self._ground_object_target(
                            analyzed.downstream_object_target,
                            context,
                        ),
                    )
                ),
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
        control_part, held = self._require_held_object(
            analyzed,
            context,
            eligible,
            slot_id="primary",
            path=(*path, analyzed.index, "call"),
        )
        del control_part
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
        """Resolve the motion control part and verify its held-object identity."""
        endpoint = analyzed.bound.binding.action_binding.endpoint(slot_id, "motion")
        try:
            target = endpoint.require_target(JointPositionTarget)
        except TypeError as exc:
            raise _diagnostic(
                "unsupported_builtin_endpoint",
                (*path, "resources", slot_id, "motion"),
                "The current built-in semantic lowerer requires a joint-position "
                "motion endpoint.",
            ) from exc
        held = context.task.get_held_object(target.control_part)
        call_object = getattr(analyzed.call, "object", None)
        assert type(call_object) is SceneObjectRef
        if held is None or held.semantics.entity_id != call_object.entity_id:
            raise _diagnostic(
                "verified_held_object_required",
                (*path, "object"),
                f"Call requires verified object {call_object.entity_id!r} held by "
                f"{target.control_part!r}.",
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
        return target.control_part, held

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
