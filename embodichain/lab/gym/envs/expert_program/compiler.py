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

"""Provider-free compilation and lazy expansion of Expert Program ASTs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from embodichain.lab.sim.skills.calls import (
    DeclarativeValue,
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
    SemanticPose,
)
from embodichain.lab.sim.skills.integration import SceneManifest
from embodichain.lab.sim.skills.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRef,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)

from .cfg import (
    EXPERT_PROGRAM_SCHEMA_VERSION,
    REGISTERED_SEMANTIC_CALL_SCHEMA_VERSION,
    MAX_EXPANDED_CALLS,
    MAX_REPEAT_COUNT,
    ArticulationJointPositionValidatorCfg,
    BarrierCfg,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    ParallelCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    ProgramNodeCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from .decoder import ConfigPath, ExpertProgramConfigError, render_config_path

_SEMANTIC_CALL_TYPES = (
    Pick,
    Place,
    HandOver,
    RegisteredSemanticCall,
)
_SCENE_REF_TYPES = (
    SceneEntityRef,
    SceneObjectRef,
    SceneArticulationRef,
    SceneLinkRef,
    SceneAffordanceRef,
)


class ExpertProgramCompileError(ExpertProgramConfigError):
    """Raised when a validated AST cannot lower to canonical semantic calls."""


def _copy_scene_ref(reference: SceneEntityRef) -> SceneEntityRef:
    """Return one independent exact typed scene reference."""
    if type(reference) not in _SCENE_REF_TYPES:
        raise TypeError(f"Unsupported scene reference {type(reference).__name__}.")
    return type(reference)(reference.entity_id)


@dataclass(frozen=True, slots=True)
class CompiledRepeatFrame:
    """One lexical repeat occurrence in a compiled call or segment path."""

    path: ConfigPath
    iteration_index: int
    count: int

    def __post_init__(self) -> None:
        if type(self.path) is not tuple:
            raise TypeError("path must be a ConfigPath tuple.")
        if type(self.iteration_index) is not int or not 0 <= self.iteration_index:
            raise ValueError("iteration_index must be a non-negative integer.")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("count must be a positive integer.")
        if self.iteration_index >= self.count:
            raise ValueError("iteration_index must be smaller than count.")


@dataclass(frozen=True, slots=True)
class CompiledTargetSelection:
    """Deterministic cyclic-target selection metadata for one occurrence."""

    target_id: str
    value_index: int
    repeat_path: ConfigPath | None
    repeat_iteration_index: int | None

    def __post_init__(self) -> None:
        if type(self.target_id) is not str or not self.target_id:
            raise ValueError("target_id must be a non-empty string.")
        if type(self.value_index) is not int or self.value_index < 0:
            raise ValueError("value_index must be a non-negative integer.")
        if (self.repeat_path is None) != (self.repeat_iteration_index is None):
            raise ValueError(
                "repeat_path and repeat_iteration_index must both be set or unset."
            )
        if self.repeat_path is not None and type(self.repeat_path) is not tuple:
            raise TypeError("repeat_path must be a ConfigPath tuple or None.")
        if self.repeat_iteration_index is not None and (
            type(self.repeat_iteration_index) is not int
            or self.repeat_iteration_index < 0
        ):
            raise ValueError("repeat_iteration_index must be non-negative or None.")


def _snapshot_semantic_call(call: SemanticCallSpec) -> SemanticCallSpec:
    """Return one independently owned exact semantic-call value."""
    if type(call) is Pick:
        return Pick(
            object=_copy_scene_ref(call.object),
            grasp=(None if call.grasp is None else _copy_scene_ref(call.grasp)),
            resources=dict(call.resources),
        )
    if type(call) is Place:
        return Place(
            object=_copy_scene_ref(call.object),
            at=None if call.at is None else call.at.snapshot(),
            on=None if call.on is None else _copy_scene_ref(call.on),
            inside=None if call.inside is None else _copy_scene_ref(call.inside),
            resources=dict(call.resources),
        )
    if type(call) is HandOver:
        return HandOver(
            object=_copy_scene_ref(call.object),
            receiver=call.receiver,
            final_target=(
                None if call.final_target is None else call.final_target.snapshot()
            ),
            resources=dict(call.resources),
        )
    if type(call) is RegisteredSemanticCall:
        return RegisteredSemanticCall(
            call_id=call.call_id,
            arguments=call.arguments,
            resources=dict(call.resources),
        )
    raise TypeError("call must be an exact supported SemanticCallSpec value.")


@dataclass(frozen=True, slots=True)
class CompiledProgramCall:
    """One owned semantic call occurrence emitted by lazy program expansion."""

    call_index: int
    segment_call_index: int
    call: SemanticCallSpec
    source_path: ConfigPath
    repeat_frames: tuple[CompiledRepeatFrame, ...] = ()
    target_selections: tuple[CompiledTargetSelection, ...] = ()

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer.")
        if type(self.segment_call_index) is not int or self.segment_call_index < 0:
            raise ValueError("segment_call_index must be a non-negative integer.")
        if type(self.call) not in _SEMANTIC_CALL_TYPES:
            raise TypeError("call must be an exact supported SemanticCallSpec value.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        frames = tuple(self.repeat_frames)
        selections = tuple(self.target_selections)
        if not all(type(frame) is CompiledRepeatFrame for frame in frames):
            raise TypeError("repeat_frames must contain CompiledRepeatFrame values.")
        if not all(
            type(selection) is CompiledTargetSelection for selection in selections
        ):
            raise TypeError(
                "target_selections must contain CompiledTargetSelection values."
            )
        object.__setattr__(self, "call", _snapshot_semantic_call(self.call))
        object.__setattr__(self, "repeat_frames", frames)
        object.__setattr__(self, "target_selections", selections)


@dataclass(frozen=True, slots=True)
class CompiledPostPolicy:
    """Owned post-policy config plus its canonical scene entity and source path."""

    cfg: WaitStablePostCfg
    entity: SceneEntityRef
    source_path: ConfigPath

    def __post_init__(self) -> None:
        if type(self.cfg) is not WaitStablePostCfg:
            raise TypeError("cfg must be exactly WaitStablePostCfg.")
        if type(self.entity) not in _SCENE_REF_TYPES:
            raise TypeError("entity must be an exact SceneEntityRef value.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        object.__setattr__(
            self,
            "cfg",
            WaitStablePostCfg(
                entity=self.cfg.entity,
                preset=self.cfg.preset,
                kind=self.cfg.kind,
            ),
        )
        object.__setattr__(self, "entity", _copy_scene_ref(self.entity))


@dataclass(frozen=True, slots=True)
class CompiledObjectNearTargetValidator:
    """Owned validator config with canonical object and resolved target pose."""

    cfg: ObjectNearTargetValidatorCfg
    object: SceneObjectRef
    target_pose: SemanticPose
    target_selection: CompiledTargetSelection
    source_path: ConfigPath

    def __post_init__(self) -> None:
        if type(self.cfg) is not ObjectNearTargetValidatorCfg:
            raise TypeError("cfg must be exactly ObjectNearTargetValidatorCfg.")
        if type(self.object) is not SceneObjectRef:
            raise TypeError("object must be exactly SceneObjectRef.")
        if type(self.target_pose) is not SemanticPose:
            raise TypeError("target_pose must be exactly SemanticPose.")
        if type(self.target_selection) is not CompiledTargetSelection:
            raise TypeError("target_selection must be exactly CompiledTargetSelection.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        object.__setattr__(
            self,
            "cfg",
            ObjectNearTargetValidatorCfg(
                object=self.cfg.object,
                target=self.cfg.target,
                position_tolerance=self.cfg.position_tolerance,
                kind=self.cfg.kind,
            ),
        )
        object.__setattr__(self, "object", _copy_scene_ref(self.object))
        object.__setattr__(self, "target_pose", self.target_pose.snapshot())


@dataclass(frozen=True, slots=True)
class CompiledArticulationJointPositionValidator:
    """Owned joint-position validator with its canonical articulation."""

    cfg: ArticulationJointPositionValidatorCfg
    articulation: SceneArticulationRef
    source_path: ConfigPath

    def __post_init__(self) -> None:
        if type(self.cfg) is not ArticulationJointPositionValidatorCfg:
            raise TypeError(
                "cfg must be exactly ArticulationJointPositionValidatorCfg."
            )
        if type(self.articulation) is not SceneArticulationRef:
            raise TypeError("articulation must be exactly SceneArticulationRef.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        object.__setattr__(
            self,
            "cfg",
            ArticulationJointPositionValidatorCfg(
                articulation=self.cfg.articulation,
                joint=self.cfg.joint,
                minimum_position=self.cfg.minimum_position,
                maximum_position=self.cfg.maximum_position,
                kind=self.cfg.kind,
            ),
        )
        object.__setattr__(
            self,
            "articulation",
            _copy_scene_ref(self.articulation),
        )


CompiledProgramValidator: TypeAlias = (
    CompiledObjectNearTargetValidator | CompiledArticulationJointPositionValidator
)
_COMPILED_VALIDATOR_TYPES = (
    CompiledObjectNearTargetValidator,
    CompiledArticulationJointPositionValidator,
)


@dataclass(frozen=True, slots=True)
class CompiledBarrier:
    """Explicit schema-v2 join semantics for one compiled parallel block."""

    name: str
    timeout_steps: int
    failure_policy: str
    source_path: ConfigPath

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("barrier name must be non-empty.")
        if type(self.timeout_steps) is not int or self.timeout_steps <= 0:
            raise ValueError("barrier timeout_steps must be positive.")
        if self.failure_policy != "fail_fast":
            raise ValueError("barrier failure_policy must be 'fail_fast'.")
        if type(self.source_path) is not tuple:
            raise TypeError("barrier source_path must be a ConfigPath tuple.")


@dataclass(frozen=True, slots=True)
class CompiledParallelBranch:
    """One ordered semantic-call lane inside a parallel block."""

    branch_index: int
    calls: tuple[CompiledProgramCall, ...]
    source_path: ConfigPath

    def __post_init__(self) -> None:
        if type(self.branch_index) is not int or self.branch_index < 0:
            raise ValueError("branch_index must be non-negative.")
        calls = tuple(self.calls)
        if not calls or not all(type(call) is CompiledProgramCall for call in calls):
            raise TypeError("parallel branch calls must be non-empty compiled calls.")
        if type(self.source_path) is not tuple:
            raise TypeError("parallel branch source_path must be a ConfigPath tuple.")
        object.__setattr__(self, "calls", calls)


@dataclass(frozen=True, slots=True)
class CompiledParallelBlock:
    """Two or more call lanes joined by an explicit deterministic barrier."""

    branches: tuple[CompiledParallelBranch, ...]
    barrier: CompiledBarrier
    source_path: ConfigPath

    def __post_init__(self) -> None:
        branches = tuple(self.branches)
        if len(branches) < 2 or not all(
            type(branch) is CompiledParallelBranch for branch in branches
        ):
            raise TypeError("parallel blocks require at least two compiled branches.")
        if tuple(branch.branch_index for branch in branches) != tuple(
            range(len(branches))
        ):
            raise ValueError("parallel branch indices must be contiguous from zero.")
        if type(self.barrier) is not CompiledBarrier:
            raise TypeError("barrier must be exactly CompiledBarrier.")
        if type(self.source_path) is not tuple:
            raise TypeError("parallel source_path must be a ConfigPath tuple.")
        object.__setattr__(self, "branches", branches)


@dataclass(frozen=True, slots=True)
class CompiledProgramSegment:
    """One independent explicit or implicit logical program segment."""

    segment_index: int
    segment_id: str
    name: str
    calls: tuple[CompiledProgramCall, ...]
    source_path: ConfigPath
    repeat_frames: tuple[CompiledRepeatFrame, ...] = ()
    post_policies: tuple[CompiledPostPolicy, ...] = ()
    validators: tuple[CompiledProgramValidator, ...] = ()
    parallel_block: CompiledParallelBlock | None = None
    implicit: bool = False

    def __post_init__(self) -> None:
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("segment_index must be a non-negative integer.")
        for field_name in ("segment_id", "name"):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"{field_name} must be a non-empty string.")
        calls = tuple(self.calls)
        if not calls or not all(type(call) is CompiledProgramCall for call in calls):
            raise TypeError(
                "calls must contain at least one exact CompiledProgramCall."
            )
        if tuple(call.segment_call_index for call in calls) != tuple(range(len(calls))):
            raise ValueError("segment call indices must be contiguous from zero.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        frames = tuple(self.repeat_frames)
        post = tuple(self.post_policies)
        validators = tuple(self.validators)
        if not all(type(frame) is CompiledRepeatFrame for frame in frames):
            raise TypeError("repeat_frames must contain CompiledRepeatFrame values.")
        if not all(type(value) is CompiledPostPolicy for value in post):
            raise TypeError("post_policies must contain CompiledPostPolicy values.")
        if not all(type(value) in _COMPILED_VALIDATOR_TYPES for value in validators):
            raise TypeError("validators must contain compiled validator values.")
        if type(self.implicit) is not bool:
            raise TypeError("implicit must be a bool.")
        if self.implicit and (post or validators):
            raise ValueError(
                "Implicit segments cannot own post-policies or validators."
            )
        if self.parallel_block is not None:
            if type(self.parallel_block) is not CompiledParallelBlock:
                raise TypeError("parallel_block must be CompiledParallelBlock or None.")
            flattened = tuple(
                call for branch in self.parallel_block.branches for call in branch.calls
            )
            if flattened != calls:
                raise ValueError(
                    "segment calls must equal parallel branch calls in branch order."
                )
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "repeat_frames", frames)
        object.__setattr__(self, "post_policies", post)
        object.__setattr__(self, "validators", validators)


@dataclass(frozen=True, slots=True)
class CompiledProgramAnalysis:
    """One owned canonical semantic-analysis window for a compiled program.

    ``execution_prefix_length`` separates calls that the current segment owns
    from downstream calls included only for static state-flow and target
    look-ahead.  Preflight analyses set the prefix to the complete window.
    """

    analysis_id: str
    kind: str
    calls: tuple[SemanticCallSpec, ...]
    source_path: ConfigPath
    segment_indices: tuple[int, ...]
    execution_prefix_length: int

    def __post_init__(self) -> None:
        if type(self.analysis_id) is not str or not self.analysis_id:
            raise ValueError("analysis_id must be a non-empty string.")
        if self.kind not in {
            "sequential_stretch",
            "parallel_branch",
            "sequential_suffix",
        }:
            raise ValueError("kind must identify a supported program analysis.")
        calls = tuple(self.calls)
        if not calls or not all(type(call) in _SEMANTIC_CALL_TYPES for call in calls):
            raise TypeError("calls must contain supported semantic call values.")
        if type(self.source_path) is not tuple:
            raise TypeError("source_path must be a ConfigPath tuple.")
        indices = tuple(self.segment_indices)
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("segment_indices must contain non-negative integers.")
        if len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices:
            raise ValueError("segment_indices must be unique and ordered.")
        if type(
            self.execution_prefix_length
        ) is not int or not 1 <= self.execution_prefix_length <= len(calls):
            raise ValueError(
                "execution_prefix_length must select a non-empty prefix of calls."
            )
        object.__setattr__(
            self,
            "calls",
            tuple(_snapshot_semantic_call(call) for call in calls),
        )
        object.__setattr__(self, "segment_indices", indices)


@dataclass(frozen=True, slots=True)
class _CallTemplate:
    kind: str
    source_path: ConfigPath
    object: SceneObjectRef | None = None
    grasp: SceneAffordanceRef | None = None
    at_target_id: str | None = None
    on: SceneObjectRef | SceneAffordanceRef | None = None
    inside: SceneObjectRef | SceneAffordanceRef | None = None
    receiver: str | None = None
    final_target_id: str | None = None
    call_id: str | None = None
    arguments: Mapping[str, DeclarativeValue] | None = None
    resources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _InvokeTemplate:
    call: _CallTemplate
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _SequenceTemplate:
    items: tuple[_NodeTemplate, ...]
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _RepeatTemplate:
    count: int
    body: _NodeTemplate
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _BarrierTemplate:
    name: str
    timeout_steps: int
    failure_policy: str
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _ParallelTemplate:
    branches: tuple[_NodeTemplate, ...]
    barrier: _BarrierTemplate
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _PostTemplate:
    cfg: WaitStablePostCfg
    entity: SceneEntityRef
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _ObjectNearTargetValidatorTemplate:
    cfg: ObjectNearTargetValidatorCfg
    object: SceneObjectRef
    target_id: str
    source_path: ConfigPath


@dataclass(frozen=True, slots=True)
class _ArticulationJointPositionValidatorTemplate:
    cfg: ArticulationJointPositionValidatorCfg
    articulation: SceneArticulationRef
    source_path: ConfigPath


_ValidatorTemplate: TypeAlias = (
    _ObjectNearTargetValidatorTemplate | _ArticulationJointPositionValidatorTemplate
)


@dataclass(frozen=True, slots=True)
class _SegmentTemplate:
    name: str
    steps: _NodeTemplate
    post: tuple[_PostTemplate, ...]
    validators: tuple[_ValidatorTemplate, ...]
    source_path: ConfigPath


_NodeTemplate = (
    _InvokeTemplate
    | _SequenceTemplate
    | _RepeatTemplate
    | _SegmentTemplate
    | _ParallelTemplate
)


def _contains_parallel(template: _NodeTemplate) -> bool:
    """Return whether a compiled subtree owns a parallel block."""
    if type(template) is _ParallelTemplate:
        return True
    if type(template) is _SequenceTemplate:
        return any(_contains_parallel(child) for child in template.items)
    if type(template) is _RepeatTemplate:
        return _contains_parallel(template.body)
    if type(template) is _SegmentTemplate:
        return _contains_parallel(template.steps)
    return False


@dataclass(slots=True)
class _ExpansionState:
    segment_index: int = 0
    call_index: int = 0


def _resolve_target(
    target_id: str,
    *,
    targets: Mapping[str, tuple[SemanticPose, ...]],
    repeat_frames: tuple[CompiledRepeatFrame, ...],
) -> tuple[SemanticPose, CompiledTargetSelection]:
    """Select one cyclic target from the nearest lexical repeat frame."""
    values = targets[target_id]
    repeat = repeat_frames[-1] if repeat_frames else None
    value_index = 0 if repeat is None else repeat.iteration_index % len(values)
    selection = CompiledTargetSelection(
        target_id=target_id,
        value_index=value_index,
        repeat_path=None if repeat is None else repeat.path,
        repeat_iteration_index=None if repeat is None else repeat.iteration_index,
    )
    return values[value_index].snapshot(), selection


def _instantiate_call(
    template: _CallTemplate,
    *,
    call_index: int,
    segment_call_index: int,
    targets: Mapping[str, tuple[SemanticPose, ...]],
    repeat_frames: tuple[CompiledRepeatFrame, ...],
) -> CompiledProgramCall:
    """Instantiate one semantic call occurrence from static templates."""
    resources = dict(template.resources)
    selections: list[CompiledTargetSelection] = []
    if template.kind == "pick":
        assert template.object is not None
        call: SemanticCallSpec = Pick(
            object=_copy_scene_ref(template.object),
            grasp=(None if template.grasp is None else _copy_scene_ref(template.grasp)),
            resources=resources,
        )
    elif template.kind == "place":
        assert template.object is not None
        at: SemanticPose | None = None
        if template.at_target_id is not None:
            at, selection = _resolve_target(
                template.at_target_id,
                targets=targets,
                repeat_frames=repeat_frames,
            )
            selections.append(selection)
        call = Place(
            object=_copy_scene_ref(template.object),
            at=at,
            on=None if template.on is None else _copy_scene_ref(template.on),
            inside=(
                None if template.inside is None else _copy_scene_ref(template.inside)
            ),
            resources=resources,
        )
    elif template.kind == "hand_over":
        assert template.object is not None
        final_target: SemanticPose | None = None
        if template.final_target_id is not None:
            final_target, selection = _resolve_target(
                template.final_target_id,
                targets=targets,
                repeat_frames=repeat_frames,
            )
            selections.append(selection)
        call = HandOver(
            object=_copy_scene_ref(template.object),
            receiver=template.receiver,
            final_target=final_target,
            resources=resources,
        )
    elif template.kind == "registered":
        assert template.call_id is not None and template.arguments is not None
        call = RegisteredSemanticCall(
            call_id=template.call_id,
            arguments=template.arguments,
            resources=resources,
        )
    else:  # pragma: no cover - compiler-owned templates prevent this
        raise AssertionError(f"Unknown call template {template.kind!r}.")
    return CompiledProgramCall(
        call_index=call_index,
        segment_call_index=segment_call_index,
        call=call,
        source_path=template.source_path,
        repeat_frames=repeat_frames,
        target_selections=tuple(selections),
    )


def _iter_call_templates(
    template: _NodeTemplate,
    *,
    repeat_frames: tuple[CompiledRepeatFrame, ...],
) -> Iterator[tuple[_CallTemplate, tuple[CompiledRepeatFrame, ...]]]:
    """Expand call templates inside one explicit segment without segment splits."""
    if type(template) is _InvokeTemplate:
        yield template.call, repeat_frames
    elif type(template) is _SequenceTemplate:
        for child in template.items:
            yield from _iter_call_templates(child, repeat_frames=repeat_frames)
    elif type(template) is _RepeatTemplate:
        for iteration_index in range(template.count):
            frame = CompiledRepeatFrame(
                path=template.source_path,
                iteration_index=iteration_index,
                count=template.count,
            )
            yield from _iter_call_templates(
                template.body,
                repeat_frames=(*repeat_frames, frame),
            )
    else:  # pragma: no cover - nested segments are rejected during compilation
        raise AssertionError("A nested segment reached call-only expansion.")


def _instantiate_parallel_block(
    template: _ParallelTemplate,
    *,
    targets: Mapping[str, tuple[SemanticPose, ...]],
    repeat_frames: tuple[CompiledRepeatFrame, ...],
    state: _ExpansionState,
) -> tuple[CompiledParallelBlock, tuple[CompiledProgramCall, ...]]:
    """Instantiate branch-local call order without serializing branch semantics."""
    branches: list[CompiledParallelBranch] = []
    flattened: list[CompiledProgramCall] = []
    segment_call_index = 0
    for branch_index, branch_template in enumerate(template.branches):
        calls: list[CompiledProgramCall] = []
        for call_template, call_repeat_frames in _iter_call_templates(
            branch_template,
            repeat_frames=repeat_frames,
        ):
            call = _instantiate_call(
                call_template,
                call_index=state.call_index,
                segment_call_index=segment_call_index,
                targets=targets,
                repeat_frames=call_repeat_frames,
            )
            calls.append(call)
            flattened.append(call)
            state.call_index += 1
            segment_call_index += 1
        branches.append(
            CompiledParallelBranch(
                branch_index=branch_index,
                calls=tuple(calls),
                source_path=template.branches[branch_index].source_path,
            )
        )
    barrier = CompiledBarrier(
        name=template.barrier.name,
        timeout_steps=template.barrier.timeout_steps,
        failure_policy=template.barrier.failure_policy,
        source_path=template.barrier.source_path,
    )
    return (
        CompiledParallelBlock(
            branches=tuple(branches),
            barrier=barrier,
            source_path=template.source_path,
        ),
        tuple(flattened),
    )


def _segment_identity(
    program_id: str,
    *,
    source_path: ConfigPath,
    repeat_frames: tuple[CompiledRepeatFrame, ...],
    implicit: bool,
) -> str:
    """Build one deterministic segment identity from lexical occurrence data."""
    repeat_suffix = "".join(
        f"@{render_config_path(frame.path)}[{frame.iteration_index}]"
        for frame in repeat_frames
    )
    boundary = "implicit" if implicit else "segment"
    return f"{program_id}:{boundary}:{render_config_path(source_path)}{repeat_suffix}"


def _iter_segments(
    template: _NodeTemplate,
    *,
    program_id: str,
    targets: Mapping[str, tuple[SemanticPose, ...]],
    repeat_frames: tuple[CompiledRepeatFrame, ...],
    state: _ExpansionState,
) -> Iterator[CompiledProgramSegment]:
    """Lazily expand outer program structure into independent segments."""
    if type(template) is _SequenceTemplate:
        for child in template.items:
            yield from _iter_segments(
                child,
                program_id=program_id,
                targets=targets,
                repeat_frames=repeat_frames,
                state=state,
            )
        return
    if type(template) is _RepeatTemplate:
        for iteration_index in range(template.count):
            frame = CompiledRepeatFrame(
                path=template.source_path,
                iteration_index=iteration_index,
                count=template.count,
            )
            yield from _iter_segments(
                template.body,
                program_id=program_id,
                targets=targets,
                repeat_frames=(*repeat_frames, frame),
                state=state,
            )
        return
    if type(template) is _InvokeTemplate:
        call = _instantiate_call(
            template.call,
            call_index=state.call_index,
            segment_call_index=0,
            targets=targets,
            repeat_frames=repeat_frames,
        )
        state.call_index += 1
        segment = CompiledProgramSegment(
            segment_index=state.segment_index,
            segment_id=_segment_identity(
                program_id,
                source_path=template.source_path,
                repeat_frames=repeat_frames,
                implicit=True,
            ),
            name=f"invoke:{call.call.semantic_id}",
            calls=(call,),
            source_path=template.source_path,
            repeat_frames=repeat_frames,
            implicit=True,
        )
        state.segment_index += 1
        yield segment
        return

    if type(template) is _ParallelTemplate:
        parallel_block, calls = _instantiate_parallel_block(
            template,
            targets=targets,
            repeat_frames=repeat_frames,
            state=state,
        )
        segment = CompiledProgramSegment(
            segment_index=state.segment_index,
            segment_id=_segment_identity(
                program_id,
                source_path=template.source_path,
                repeat_frames=repeat_frames,
                implicit=True,
            ),
            name=f"parallel:{parallel_block.barrier.name}",
            calls=calls,
            source_path=template.source_path,
            repeat_frames=repeat_frames,
            parallel_block=parallel_block,
            implicit=True,
        )
        state.segment_index += 1
        yield segment
        return

    assert type(template) is _SegmentTemplate
    parallel_block: CompiledParallelBlock | None = None
    if type(template.steps) is _ParallelTemplate:
        parallel_block, instantiated_calls = _instantiate_parallel_block(
            template.steps,
            targets=targets,
            repeat_frames=repeat_frames,
            state=state,
        )
        calls = list(instantiated_calls)
    else:
        calls = []
        for segment_call_index, (call_template, call_repeat_frames) in enumerate(
            _iter_call_templates(template.steps, repeat_frames=repeat_frames)
        ):
            calls.append(
                _instantiate_call(
                    call_template,
                    call_index=state.call_index,
                    segment_call_index=segment_call_index,
                    targets=targets,
                    repeat_frames=call_repeat_frames,
                )
            )
            state.call_index += 1
    post_policies = tuple(
        CompiledPostPolicy(
            cfg=post.cfg,
            entity=post.entity,
            source_path=post.source_path,
        )
        for post in template.post
    )
    validators: list[CompiledProgramValidator] = []
    for validator in template.validators:
        if type(validator) is _ObjectNearTargetValidatorTemplate:
            target_pose, selection = _resolve_target(
                validator.target_id,
                targets=targets,
                repeat_frames=repeat_frames,
            )
            validators.append(
                CompiledObjectNearTargetValidator(
                    cfg=validator.cfg,
                    object=validator.object,
                    target_pose=target_pose,
                    target_selection=selection,
                    source_path=validator.source_path,
                )
            )
        elif type(validator) is _ArticulationJointPositionValidatorTemplate:
            validators.append(
                CompiledArticulationJointPositionValidator(
                    cfg=validator.cfg,
                    articulation=validator.articulation,
                    source_path=validator.source_path,
                )
            )
        else:
            raise TypeError(
                f"Unsupported internal validator template {type(validator).__name__}."
            )
    segment = CompiledProgramSegment(
        segment_index=state.segment_index,
        segment_id=_segment_identity(
            program_id,
            source_path=template.source_path,
            repeat_frames=repeat_frames,
            implicit=False,
        ),
        name=template.name,
        calls=tuple(calls),
        source_path=template.source_path,
        repeat_frames=repeat_frames,
        post_policies=post_policies,
        validators=tuple(validators),
        parallel_block=parallel_block,
        implicit=False,
    )
    state.segment_index += 1
    yield segment


@dataclass(frozen=True, slots=True, init=False)
class CompiledProgram:
    """Bounded provider-free segment snapshot used by preflight and execution."""

    schema_version: int
    program_id: str
    _integration: ExpertProgramIntegrationCfg = field(repr=False, compare=False)
    _segments: tuple[CompiledProgramSegment, ...] = field(
        repr=False,
        compare=False,
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside :class:`ExpertProgramCompiler`."""
        del args, kwargs
        raise TypeError("CompiledProgram values are created by ExpertProgramCompiler.")

    @classmethod
    def _create(
        cls,
        *,
        schema_version: int,
        program_id: str,
        integration: ExpertProgramIntegrationCfg,
        segments: tuple[CompiledProgramSegment, ...],
    ) -> CompiledProgram:
        """Create one compiler-owned materialized program."""
        if (
            type(schema_version) is not int
            or schema_version != EXPERT_PROGRAM_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_version must be exactly {EXPERT_PROGRAM_SCHEMA_VERSION}."
            )
        if type(program_id) is not str or not program_id:
            raise ValueError("program_id must be a non-empty string.")
        if type(integration) is not ExpertProgramIntegrationCfg:
            raise TypeError("integration must be ExpertProgramIntegrationCfg.")
        values = tuple(segments)
        if not values or not all(
            type(segment) is CompiledProgramSegment for segment in values
        ):
            raise TypeError(
                "segments must contain at least one CompiledProgramSegment."
            )
        if tuple(segment.segment_index for segment in values) != tuple(
            range(len(values))
        ):
            raise ValueError("Materialized segment indices must be contiguous.")
        flattened_calls = tuple(call for segment in values for call in segment.calls)
        if len(flattened_calls) > MAX_EXPANDED_CALLS:
            raise ValueError(
                f"Materialized program exceeds {MAX_EXPANDED_CALLS} calls."
            )
        if tuple(call.call_index for call in flattened_calls) != tuple(
            range(len(flattened_calls))
        ):
            raise ValueError("Materialized call indices must be contiguous.")

        instance = object.__new__(cls)
        object.__setattr__(instance, "schema_version", schema_version)
        object.__setattr__(instance, "program_id", program_id)
        object.__setattr__(
            instance,
            "_integration",
            ExpertProgramIntegrationCfg(
                robot_profile=integration.robot_profile,
                scene_registry=integration.scene_registry,
                runtime_preset=integration.runtime_preset,
            ),
        )
        object.__setattr__(instance, "_segments", values)
        return instance

    @property
    def integration(self) -> ExpertProgramIntegrationCfg:
        """Return an independent integration-selection snapshot."""
        return ExpertProgramIntegrationCfg(
            robot_profile=self._integration.robot_profile,
            scene_registry=self._integration.scene_registry,
            runtime_preset=self._integration.runtime_preset,
        )

    @property
    def segment_count(self) -> int:
        """Return the number of materialized logical segments."""
        return len(self._segments)

    def iter_segments(self) -> Iterator[CompiledProgramSegment]:
        """Iterate the already materialized provider-free segments."""
        return iter(self._segments)

    def preflight_analyses(self) -> tuple[CompiledProgramAnalysis, ...]:
        """Return full-program analyses split only at parallel barriers.

        Consecutive sequential segments form one static workflow, preserving
        their object-state flow and cross-segment target look-ahead.  Each
        parallel branch is analyzed independently; no state or target inference
        crosses the barrier in either direction.
        """
        analyses: list[CompiledProgramAnalysis] = []
        stretch: list[CompiledProgramSegment] = []

        def flush_stretch() -> None:
            if not stretch:
                return
            indices = tuple(segment.segment_index for segment in stretch)
            calls = tuple(call.call for segment in stretch for call in segment.calls)
            analyses.append(
                CompiledProgramAnalysis(
                    analysis_id=(
                        f"{self.program_id}:preflight:sequential:"
                        f"{indices[0]}-{indices[-1]}"
                    ),
                    kind="sequential_stretch",
                    calls=calls,
                    source_path=stretch[0].source_path,
                    segment_indices=indices,
                    execution_prefix_length=len(calls),
                )
            )
            stretch.clear()

        for segment in self._segments:
            block = segment.parallel_block
            if block is None:
                stretch.append(segment)
                continue
            flush_stretch()
            for branch in block.branches:
                calls = tuple(call.call for call in branch.calls)
                analyses.append(
                    CompiledProgramAnalysis(
                        analysis_id=(
                            f"{self.program_id}:preflight:parallel:"
                            f"{segment.segment_index}:{branch.branch_index}"
                        ),
                        kind="parallel_branch",
                        calls=calls,
                        source_path=branch.source_path,
                        segment_indices=(segment.segment_index,),
                        execution_prefix_length=len(calls),
                    )
                )
        flush_stretch()
        return tuple(analyses)

    def sequential_execution_analysis(
        self,
        segment_index: int,
    ) -> CompiledProgramAnalysis:
        """Return current-segment prefix plus downstream sequential look-ahead.

        Args:
            segment_index: Index of the sequential segment about to execute.

        Returns:
            Analysis beginning at the selected segment and ending immediately
            before the next parallel barrier or the end of the program.

        Raises:
            IndexError: If ``segment_index`` is outside this program.
            ValueError: If the selected segment is itself parallel.
        """
        if type(segment_index) is not int:
            raise TypeError("segment_index must be an integer.")
        if not 0 <= segment_index < len(self._segments):
            raise IndexError(f"segment_index {segment_index!r} is outside the program.")
        current = self._segments[segment_index]
        if current.parallel_block is not None:
            raise ValueError("Parallel segments do not have sequential look-ahead.")
        window: list[CompiledProgramSegment] = []
        for segment in self._segments[segment_index:]:
            if segment.parallel_block is not None:
                break
            window.append(segment)
        calls = tuple(call.call for segment in window for call in segment.calls)
        indices = tuple(segment.segment_index for segment in window)
        return CompiledProgramAnalysis(
            analysis_id=(
                f"{self.program_id}:execution:sequential:" f"{indices[0]}-{indices[-1]}"
            ),
            kind="sequential_suffix",
            calls=calls,
            source_path=current.source_path,
            segment_indices=indices,
            execution_prefix_length=len(current.calls),
        )

    def __iter__(self) -> Iterator[CompiledProgramSegment]:
        return self.iter_segments()


def _materialize_program(
    *,
    schema_version: int,
    program_id: str,
    integration: ExpertProgramIntegrationCfg,
    targets: Mapping[str, tuple[SemanticPose, ...]],
    root: _NodeTemplate,
) -> CompiledProgram:
    """Expand one internal template directly into the public bounded snapshot."""
    segments: list[CompiledProgramSegment] = []
    expanded_calls = 0
    for segment in _iter_segments(
        root,
        program_id=program_id,
        targets=targets,
        repeat_frames=(),
        state=_ExpansionState(),
    ):
        expanded_calls += len(segment.calls)
        if expanded_calls > MAX_EXPANDED_CALLS:
            raise ExpertProgramCompileError(
                "expanded_call_limit",
                segment.source_path,
                "Program expansion exceeds the static limit of "
                f"{MAX_EXPANDED_CALLS} semantic calls.",
            )
        segments.append(segment)
    return CompiledProgram._create(
        schema_version=schema_version,
        program_id=program_id,
        integration=integration,
        segments=tuple(segments),
    )


class ExpertProgramCompiler:
    """Compile validated Expert Program ASTs through one static scene manifest."""

    def __init__(self, scene_manifest: SceneManifest) -> None:
        """Create one provider-free compiler.

        Args:
            scene_manifest: Canonical provider-free scene identity catalog.
        """
        if type(scene_manifest) is not SceneManifest:
            raise TypeError("scene_manifest must be exactly SceneManifest.")
        self._scene_manifest = scene_manifest

    @classmethod
    def from_scene_registry(cls, registry: SceneRegistry) -> ExpertProgramCompiler:
        """Create a compiler from a provider-free SceneRegistry identity snapshot."""
        return cls(SceneManifest.from_registry(registry))

    def _resolve_scene(
        self,
        reference: str,
        *,
        expected_types: tuple[type[SceneEntityRef], ...],
        path: ConfigPath,
    ) -> SceneEntityRef:
        """Resolve and validate one exact typed canonical scene reference."""
        try:
            resolved = self._scene_manifest.resolve(
                reference,
                path=path,
            )
        except ExpertProgramConfigError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ExpertProgramCompileError(
                "scene_resolution_failed",
                path,
                str(exc),
            ) from exc
        if type(resolved) not in expected_types:
            raise ExpertProgramCompileError(
                "scene_reference_type_mismatch",
                path,
                f"Scene reference {reference!r} resolves to "
                f"{type(resolved).__name__}, expected one of "
                f"{tuple(value.__name__ for value in expected_types)}.",
            )
        return _copy_scene_ref(resolved)

    @staticmethod
    def _target_id(
        reference: TargetRefCfg,
        *,
        targets: Mapping[str, tuple[SemanticPose, ...]],
        path: ConfigPath,
    ) -> str:
        """Resolve one statically registered target ID."""
        if type(reference) is not TargetRefCfg or reference.kind != "target_ref":
            raise ExpertProgramCompileError(
                "invalid_target_reference",
                path,
                "Expected an exact target_ref configuration.",
            )
        if reference.target not in targets:
            raise ExpertProgramCompileError(
                "unknown_target",
                (*path, "target"),
                f"Unknown target {reference.target!r}.",
            )
        return reference.target

    def _compile_call(
        self,
        cfg: object,
        *,
        targets: Mapping[str, tuple[SemanticPose, ...]],
        path: ConfigPath,
    ) -> _CallTemplate:
        """Lower one config call into a provider-free canonical template."""
        if type(cfg) is PickCfg:
            if cfg.kind != "pick":
                raise ExpertProgramCompileError(
                    "invalid_discriminator", (*path, "kind"), "Expected 'pick'."
                )
            object_ref = self._resolve_scene(
                cfg.object,
                expected_types=(SceneObjectRef,),
                path=(*path, "object"),
            )
            grasp_ref = (
                None
                if cfg.grasp is None
                else self._resolve_scene(
                    cfg.grasp,
                    expected_types=(SceneAffordanceRef,),
                    path=(*path, "grasp"),
                )
            )
            return _CallTemplate(
                kind="pick",
                source_path=path,
                object=object_ref,
                grasp=grasp_ref,
                resources=tuple(sorted(cfg.resources.items())),
            )
        if type(cfg) is PlaceCfg:
            if cfg.kind != "place":
                raise ExpertProgramCompileError(
                    "invalid_discriminator", (*path, "kind"), "Expected 'place'."
                )
            object_ref = self._resolve_scene(
                cfg.object,
                expected_types=(SceneObjectRef,),
                path=(*path, "object"),
            )
            at_target_id = (
                None
                if cfg.at is None
                else self._target_id(
                    cfg.at,
                    targets=targets,
                    path=(*path, "at"),
                )
            )
            on = (
                None
                if cfg.on is None
                else self._resolve_scene(
                    cfg.on,
                    expected_types=(SceneObjectRef, SceneAffordanceRef),
                    path=(*path, "on"),
                )
            )
            inside = (
                None
                if cfg.inside is None
                else self._resolve_scene(
                    cfg.inside,
                    expected_types=(SceneObjectRef, SceneAffordanceRef),
                    path=(*path, "inside"),
                )
            )
            return _CallTemplate(
                kind="place",
                source_path=path,
                object=object_ref,
                at_target_id=at_target_id,
                on=on,
                inside=inside,
                resources=tuple(sorted(cfg.resources.items())),
            )
        if type(cfg) is HandOverCfg:
            if cfg.kind != "hand_over":
                raise ExpertProgramCompileError(
                    "invalid_discriminator",
                    (*path, "kind"),
                    "Expected 'hand_over'.",
                )
            object_ref = self._resolve_scene(
                cfg.object,
                expected_types=(SceneObjectRef,),
                path=(*path, "object"),
            )
            final_target_id = (
                None
                if cfg.final_target is None
                else self._target_id(
                    cfg.final_target,
                    targets=targets,
                    path=(*path, "final_target"),
                )
            )
            return _CallTemplate(
                kind="hand_over",
                source_path=path,
                object=object_ref,
                receiver=cfg.receiver,
                final_target_id=final_target_id,
                resources=tuple(sorted(cfg.resources.items())),
            )
        if type(cfg) is RegisteredSemanticCallCfg:
            if cfg.kind != "registered":
                raise ExpertProgramCompileError(
                    "invalid_discriminator",
                    (*path, "kind"),
                    "Expected 'registered'.",
                )
            if cfg.schema_version != REGISTERED_SEMANTIC_CALL_SCHEMA_VERSION:
                raise ExpertProgramCompileError(
                    "unsupported_registered_schema",
                    (*path, "schema_version"),
                    "Registered call schema_version must be exactly 1.",
                )
            snapshot = RegisteredSemanticCall(
                call_id=cfg.call_id,
                arguments=cfg.arguments,
                resources=cfg.resources,
            )
            return _CallTemplate(
                kind="registered",
                source_path=path,
                call_id=snapshot.call_id,
                arguments=snapshot.arguments,
                resources=tuple(sorted(snapshot.resources.items())),
            )
        raise ExpertProgramCompileError(
            "unsupported_call",
            path,
            f"Unsupported semantic call config {type(cfg).__name__}.",
        )

    def _compile_node(
        self,
        node: ProgramNodeCfg,
        *,
        targets: Mapping[str, tuple[SemanticPose, ...]],
        path: ConfigPath,
        inside_segment: bool,
        inside_parallel: bool,
    ) -> _NodeTemplate:
        """Compile static AST structure without expanding repeats."""
        if type(node) is InvokeCfg:
            if node.kind != "invoke":
                raise ExpertProgramCompileError(
                    "invalid_discriminator", (*path, "kind"), "Expected 'invoke'."
                )
            return _InvokeTemplate(
                call=self._compile_call(
                    node.call,
                    targets=targets,
                    path=(*path, "call"),
                ),
                source_path=path,
            )
        if type(node) is SequenceCfg:
            if node.kind != "sequence":
                raise ExpertProgramCompileError(
                    "invalid_discriminator",
                    (*path, "kind"),
                    "Expected 'sequence'.",
                )
            if not node.items:
                raise ExpertProgramCompileError(
                    "empty_sequence",
                    (*path, "items"),
                    "Sequence items must contain at least one program node.",
                )
            return _SequenceTemplate(
                items=tuple(
                    self._compile_node(
                        child,
                        targets=targets,
                        path=(*path, "items", index),
                        inside_segment=inside_segment,
                        inside_parallel=inside_parallel,
                    )
                    for index, child in enumerate(node.items)
                ),
                source_path=path,
            )
        if type(node) is RepeatCfg:
            if node.kind != "repeat":
                raise ExpertProgramCompileError(
                    "invalid_discriminator", (*path, "kind"), "Expected 'repeat'."
                )
            if type(node.count) is not int or not 1 <= node.count <= MAX_REPEAT_COUNT:
                raise ExpertProgramCompileError(
                    "invalid_repeat_count",
                    (*path, "count"),
                    f"Repeat count must be an integer in [1, {MAX_REPEAT_COUNT}].",
                )
            return _RepeatTemplate(
                count=node.count,
                body=self._compile_node(
                    node.body,
                    targets=targets,
                    path=(*path, "body"),
                    inside_segment=inside_segment,
                    inside_parallel=inside_parallel,
                ),
                source_path=path,
            )
        if type(node) is SegmentCfg:
            if inside_parallel:
                raise ExpertProgramCompileError(
                    "segment_inside_parallel",
                    path,
                    "Parallel branches may contain only Invoke, Sequence, and "
                    "Repeat nodes; wrap the Parallel node in one Segment instead.",
                )
            if inside_segment:
                raise ExpertProgramCompileError(
                    "nested_segment",
                    path,
                    "Nested Segment nodes are ambiguous and forbidden.",
                )
            if node.kind != "segment":
                raise ExpertProgramCompileError(
                    "invalid_discriminator", (*path, "kind"), "Expected 'segment'."
                )
            post: list[_PostTemplate] = []
            for index, cfg in enumerate(node.post):
                post_path = (*path, "post", index)
                if type(cfg) is not WaitStablePostCfg or cfg.kind != "wait_stable":
                    raise ExpertProgramCompileError(
                        "unsupported_post_policy",
                        post_path,
                        "Supported schemas accept only exact wait_stable post policies.",
                    )
                entity = self._resolve_scene(
                    cfg.entity,
                    expected_types=_SCENE_REF_TYPES,
                    path=(*post_path, "entity"),
                )
                post.append(
                    _PostTemplate(
                        cfg=WaitStablePostCfg(
                            entity=cfg.entity,
                            preset=cfg.preset,
                            kind=cfg.kind,
                        ),
                        entity=entity,
                        source_path=post_path,
                    )
                )
            validators: list[_ValidatorTemplate] = []
            for index, cfg in enumerate(node.validators):
                validator_path = (*path, "validators", index)
                if type(cfg) is ObjectNearTargetValidatorCfg:
                    if cfg.kind != "object_near_target":
                        raise ExpertProgramCompileError(
                            "unsupported_validator",
                            validator_path,
                            "ObjectNearTargetValidatorCfg must use kind "
                            "'object_near_target'.",
                        )
                    if cfg.target not in targets:
                        raise ExpertProgramCompileError(
                            "unknown_target",
                            (*validator_path, "target"),
                            f"Unknown target {cfg.target!r}.",
                        )
                    object_ref = self._resolve_scene(
                        cfg.object,
                        expected_types=(SceneObjectRef,),
                        path=(*validator_path, "object"),
                    )
                    validators.append(
                        _ObjectNearTargetValidatorTemplate(
                            cfg=ObjectNearTargetValidatorCfg(
                                object=cfg.object,
                                target=cfg.target,
                                position_tolerance=cfg.position_tolerance,
                                kind=cfg.kind,
                            ),
                            object=object_ref,
                            target_id=cfg.target,
                            source_path=validator_path,
                        )
                    )
                    continue
                if type(cfg) is ArticulationJointPositionValidatorCfg:
                    if cfg.kind != "articulation_joint_position":
                        raise ExpertProgramCompileError(
                            "unsupported_validator",
                            validator_path,
                            "ArticulationJointPositionValidatorCfg must use kind "
                            "'articulation_joint_position'.",
                        )
                    articulation_ref = self._resolve_scene(
                        cfg.articulation,
                        expected_types=(SceneArticulationRef,),
                        path=(*validator_path, "articulation"),
                    )
                    validators.append(
                        _ArticulationJointPositionValidatorTemplate(
                            cfg=ArticulationJointPositionValidatorCfg(
                                articulation=cfg.articulation,
                                joint=cfg.joint,
                                minimum_position=cfg.minimum_position,
                                maximum_position=cfg.maximum_position,
                                kind=cfg.kind,
                            ),
                            articulation=articulation_ref,
                            source_path=validator_path,
                        )
                    )
                    continue
                raise ExpertProgramCompileError(
                    "unsupported_validator",
                    validator_path,
                    f"Unsupported validator {type(cfg).__name__}.",
                )
            steps = self._compile_node(
                node.steps,
                targets=targets,
                path=(*path, "steps"),
                inside_segment=True,
                inside_parallel=False,
            )
            if type(steps) is not _ParallelTemplate and _contains_parallel(steps):
                raise ExpertProgramCompileError(
                    "mixed_parallel_segment",
                    (*path, "steps"),
                    "A Segment may contain either a call-only program or one direct "
                    "Parallel node, not a mixed sequential/parallel tree.",
                )
            return _SegmentTemplate(
                name=node.name,
                steps=steps,
                post=tuple(post),
                validators=tuple(validators),
                source_path=path,
            )
        if type(node) is ParallelCfg:
            if inside_parallel:
                raise ExpertProgramCompileError(
                    "nested_parallel",
                    path,
                    "Nested Parallel nodes are forbidden in schema version 2.",
                )
            if node.kind != "parallel":
                raise ExpertProgramCompileError(
                    "invalid_discriminator",
                    (*path, "kind"),
                    "Expected 'parallel'.",
                )
            if len(node.branches) < 2:
                raise ExpertProgramCompileError(
                    "parallel_branch_count",
                    (*path, "branches"),
                    "Parallel requires at least two branches.",
                )
            if type(node.barrier) is not BarrierCfg:
                raise ExpertProgramCompileError(
                    "parallel_barrier_required",
                    (*path, "barrier"),
                    "Parallel.barrier must be an exact BarrierCfg.",
                )
            branches = tuple(
                self._compile_node(
                    branch,
                    targets=targets,
                    path=(*path, "branches", index),
                    inside_segment=inside_segment,
                    inside_parallel=True,
                )
                for index, branch in enumerate(node.branches)
            )
            if any(_contains_parallel(branch) for branch in branches):
                raise ExpertProgramCompileError(
                    "nested_parallel",
                    (*path, "branches"),
                    "Nested Parallel nodes are forbidden in schema version 2.",
                )
            barrier = node.barrier
            if barrier.kind != "barrier":
                raise ExpertProgramCompileError(
                    "invalid_discriminator",
                    (*path, "barrier", "kind"),
                    "Expected 'barrier'.",
                )
            if barrier.failure_policy != "fail_fast":
                raise ExpertProgramCompileError(
                    "unsupported_failure_policy",
                    (*path, "barrier", "failure_policy"),
                    "Barrier failure_policy must be exactly 'fail_fast'.",
                )
            return _ParallelTemplate(
                branches=branches,
                barrier=_BarrierTemplate(
                    name=barrier.name,
                    timeout_steps=barrier.timeout_steps,
                    failure_policy=barrier.failure_policy,
                    source_path=(*path, "barrier"),
                ),
                source_path=path,
            )
        raise ExpertProgramCompileError(
            "unsupported_program_node",
            path,
            f"Unsupported program node {type(node).__name__}.",
        )

    @staticmethod
    def _compile_targets(
        targets: Mapping[str, CyclicPoseTargetCfg],
    ) -> Mapping[str, tuple[SemanticPose, ...]]:
        """Compile static pose providers without selecting repeat values."""
        compiled: dict[str, tuple[SemanticPose, ...]] = {}
        for target_id, target in targets.items():
            path = ("targets", target_id)
            if type(target) is not CyclicPoseTargetCfg or target.kind != "cyclic_pose":
                raise ExpertProgramCompileError(
                    "unsupported_target",
                    path,
                    "Supported schemas accept only exact cyclic_pose targets.",
                )
            poses: list[SemanticPose] = []
            if not target.values:
                raise ExpertProgramCompileError(
                    "empty_target_values",
                    (*path, "values"),
                    "Cyclic target values must contain at least one pose.",
                )
            for index, pose in enumerate(target.values):
                if type(pose) is not PoseCfg:
                    raise ExpertProgramCompileError(
                        "invalid_pose",
                        (*path, "values", index),
                        "Target values must be exact PoseCfg values.",
                    )
                poses.append(SemanticPose(pose.position, pose.quaternion_wxyz))
            compiled[target_id] = tuple(poses)
        return MappingProxyType(compiled)

    def compile(self, config: ExpertProgramCfg) -> CompiledProgram:
        """Compile one validated AST into a bounded provider-free program.

        Args:
            config: Strict, supported-version Expert Program configuration.

        Returns:
            Immutable segments with repeat-local targets already resolved.

        Raises:
            ExpertProgramCompileError: If typed scene resolution or AST lowering
                fails.
        """
        if type(config) is not ExpertProgramCfg:
            raise TypeError("config must be exactly ExpertProgramCfg.")
        if config.schema_version != EXPERT_PROGRAM_SCHEMA_VERSION:
            raise ExpertProgramCompileError(
                "unsupported_schema_version",
                ("schema_version",),
                "Expert Program schema_version must be exactly "
                f"{EXPERT_PROGRAM_SCHEMA_VERSION}.",
            )
        targets = self._compile_targets(config.targets)
        root = self._compile_node(
            config.program,
            targets=targets,
            path=("program",),
            inside_segment=False,
            inside_parallel=False,
        )
        integration = ExpertProgramIntegrationCfg(
            robot_profile=config.integration.robot_profile,
            scene_registry=config.integration.scene_registry,
            runtime_preset=config.integration.runtime_preset,
        )
        return _materialize_program(
            schema_version=config.schema_version,
            program_id=config.program_id,
            integration=integration,
            targets=targets,
            root=root,
        )


__all__: list[str] = []
