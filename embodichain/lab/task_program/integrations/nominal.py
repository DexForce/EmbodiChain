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

"""Explicit source bindings for one compiled sequential Task Program."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    InstantiatedSkillMotion,
    NominalMotionSource,
    PrototypeMotionProvider,
    ResolvedActionRequest,
    PlanningContext,
)
from embodichain.lab.task_program.compiler.program import CompiledTaskProgram
from embodichain.lab.task_program.runtime.expansion import (
    TaskProgramNominalSourceFactory,
    TaskProgramPlanRequest,
)
from embodichain.lab.task_program.semantics.calls import (
    Pick,
    Place,
    RegisteredSemanticCall,
)
from .catalog import _canonical_value

__all__ = ["BoundTaskProgramNominalSources", "TracingTaskProgramNominalSources"]


@dataclass(frozen=True)
class _ConfiguredPrototypeSource:
    """Provider-free, content-addressed declaration from integration YAML."""

    call_index: int
    program_fingerprint: str
    prototype_manifest: str
    prototype_sha256: str
    object_entity_id: str
    target_entity_id: str | None
    target_entity_to_reference: tuple[float, ...] | None
    authority: str
    gripper_mode: str
    require_collision_validation: bool

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("configured call_index must be nonnegative")
        for name in ("program_fingerprint", "prototype_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.object_entity_id, str)
            or not self.object_entity_id.strip()
        ):
            raise ValueError("object_entity_id must be nonempty")
        if self.target_entity_id is not None and (
            not isinstance(self.target_entity_id, str)
            or not self.target_entity_id.strip()
        ):
            raise ValueError("target_entity_id must be nonempty or None")
        if self.target_entity_to_reference is not None:
            transform = np.asarray(self.target_entity_to_reference, dtype=np.float64)
            if transform.shape != (16,) or not np.isfinite(transform).all():
                raise ValueError(
                    "target_entity_to_reference must be a finite flat 4x4 transform"
                )
            matrix = transform.reshape(4, 4)
            if (
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                or not np.allclose(
                    matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0
                )
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6, rtol=0)
            ):
                raise ValueError(
                    "target_entity_to_reference must be a proper rigid transform"
                )
        if (
            self.authority not in {"qpos", "eef_pose", "hybrid"}
            or self.gripper_mode not in {"recorded", "semantic"}
            or type(self.require_collision_validation) is not bool
        ):
            raise ValueError(
                "invalid configured prototype authority, gripper mode, or collision policy"
            )
        if not self.require_collision_validation:
            raise ValueError(
                "configured prototype collection requires sampled collision validation"
            )

    def validate_source(self) -> None:
        """Verify content now and again at live-source creation."""
        path = Path(self.prototype_manifest)
        if (
            not path.is_file()
            or path.stat().st_size > 1024 * 1024
            or hashlib.sha256(path.read_bytes()).hexdigest() != self.prototype_sha256
        ):
            raise ValueError("configured prototype manifest is missing or has changed")
        load_prototype(path)


class _ConfiguredTaskProgramNominalSources:
    """Create fresh configured sources for one exact compiled program."""

    def __init__(
        self,
        program: CompiledTaskProgram,
        declarations: tuple[_ConfiguredPrototypeSource, ...],
        *,
        host_collision_guard: object | None = None,
    ) -> None:
        if host_collision_guard is not None:
            from .simulation.real2sim_collision import TabletopCollisionGuardFactory

            if not isinstance(host_collision_guard, TabletopCollisionGuardFactory):
                raise TypeError(
                    "host collision guard must be a validated tabletop guard"
                )
        self._program_id = program.program_id
        self._fingerprint = _fingerprint(program)
        self._declarations = declarations
        self._host_collision_guard = host_collision_guard
        self._sources = {item.call_index: item for item in declarations}
        if len(self._sources) != len(declarations):
            raise ValueError("configured nominal source call indices must be unique")
        self.validate_program(program)

    def validate_program(self, program: CompiledTaskProgram) -> None:
        """Check program and prototype revisions before assembling a bridge."""
        if _fingerprint(program) != self._fingerprint:
            raise ValueError(
                "configured nominal sources do not match the compiled program"
            )
        segments = tuple(program.iter_segments())
        if any(segment.parallel_block is not None for segment in segments):
            raise ValueError("configured nominal sources require a sequential program")
        calls = tuple(item.call for segment in segments for item in segment.calls)
        count = len(calls)
        for index, declaration in self._sources.items():
            if (
                not 0 <= index < count
                or declaration.program_fingerprint != self._fingerprint
            ):
                raise ValueError(
                    "configured nominal source call index or program fingerprint does not match"
                )
            declaration.validate_source()
            prototype = load_prototype(declaration.prototype_manifest)
            call = calls[index]
            expected = (
                "pick_up"
                if isinstance(call, Pick)
                else (
                    "place"
                    if isinstance(call, Place)
                    else (
                        "move_held_object"
                        if isinstance(call, RegisteredSemanticCall)
                        and call.call_id == "simulation.move_held_object"
                        else None
                    )
                )
            )
            if expected is None or prototype.unit.source.skill_id != expected:
                raise ValueError(
                    "configured prototype does not match the selected semantic call"
                )
            if declaration.authority != "qpos" and prototype.unit.world_tcp is None:
                raise ValueError("configured Cartesian source requires TCP data")
            if declaration.authority == "hybrid" and any(
                phase.authority is None for phase in prototype.unit.phases
            ):
                raise ValueError(
                    "configured hybrid source requires explicit phase authorities"
                )
            if (
                prototype.unit.source.target_pose is not None
                and declaration.target_entity_id is None
            ):
                raise ValueError("configured prototype requires a target entity")

    def create_nominal_source(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> NominalMotionSource | None:
        """Create a source for the current compiled call occurrence."""
        if not request.workflow_id.startswith(self._program_id + "/"):
            raise ValueError("configured nominal source belongs to a different program")
        declaration = self._sources.get(request.analysis_call_index)
        if declaration is None:
            return None
        declaration.validate_source()
        prototype = load_prototype(declaration.prototype_manifest)
        return PrototypeMotionProvider(
            prototype,
            object_entity_id=declaration.object_entity_id,
            target_entity_id=declaration.target_entity_id,
            target_entity_to_reference=(
                None
                if declaration.target_entity_to_reference is None
                else np.asarray(declaration.target_entity_to_reference).reshape(4, 4)
            ),
            authority=declaration.authority,
            gripper_mode=declaration.gripper_mode,
            motion_generator=(
                engine.motion_generator
                if declaration.authority != "qpos"
                or declaration.require_collision_validation
                else None
            ),
            require_collision_validation=(
                declaration.require_collision_validation
                and self._host_collision_guard is None
            ),
            prototype_sha256=declaration.prototype_sha256,
            object_support_validator=(
                self._host_collision_guard.validator.object_supported_by_target
                if self._host_collision_guard is not None
                and prototype.unit.source.skill_id == "pick_up"
                else None
            ),
            ik_smoothing_trigger_fraction=(
                None
                if self._host_collision_guard is None
                else self._host_collision_guard.ik_smoothing_trigger_fraction
            ),
        )


def _fingerprint(program: CompiledTaskProgram) -> str:
    if type(program) is not CompiledTaskProgram:
        raise TypeError("program must be a CompiledTaskProgram")
    payload = _canonical_value(
        (
            program.program_id,
            program.instruction,
            program.integration,
            tuple(program.iter_segments()),
        )
    )
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class BoundTaskProgramNominalSources:
    """Bind sources to distinct call occurrences of one compiled program.

    The constructor records a fingerprint of the provider-free program and
    validates all declared indices. A fresh bridge must call ``validate_program``
    before use. Unlisted calls retain the ordinary native source.

    Args:
        program: Compiled sequential program used to establish call identity.
        sources: Sources keyed by zero-based global sequential call occurrence.
    """

    def __init__(
        self, program: CompiledTaskProgram, sources: Mapping[int, NominalMotionSource]
    ) -> None:
        fingerprint = _fingerprint(program)
        segments = tuple(program.iter_segments())
        if any(segment.parallel_block is not None for segment in segments):
            raise ValueError(
                "nominal source bindings currently require a sequential program"
            )
        count = sum(len(segment.calls) for segment in segments)
        values = dict(sources)
        if any(type(index) is not int or not 0 <= index < count for index in values):
            raise ValueError("nominal source binding index is outside the program")
        if any(
            not isinstance(source, NominalMotionSource) for source in values.values()
        ):
            raise TypeError("every source must implement NominalMotionSource")
        self._program_id = program.program_id
        self._fingerprint = fingerprint
        self._sources = MappingProxyType(values)

    @property
    def program_fingerprint(self) -> str:
        """Return the bound provider-free program fingerprint."""
        return self._fingerprint

    def validate_program(self, program: CompiledTaskProgram) -> None:
        """Reject a changed program before creating an execution bridge."""
        if _fingerprint(program) != self._fingerprint:
            raise ValueError(
                "nominal source bindings do not match the compiled program"
            )

    def create_nominal_source(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> NominalMotionSource | None:
        """Resolve a source by compiled call occurrence, not by skill name."""
        if not isinstance(engine, AtomicActionEngine) or not isinstance(
            request, TaskProgramPlanRequest
        ):
            raise TypeError("engine and request must be typed runtime values")
        if not request.workflow_id.startswith(self._program_id + "/"):
            raise ValueError("nominal binding belongs to a different program")
        return self._sources.get(request.analysis_call_index)


class TracingTaskProgramNominalSources:
    """Capture instantiated source maps by occurrence for one B=1 episode.

    This wrapper owns source evidence only; it does not execute actions or
    attest physical validity. Repeated instantiation of one occurrence is
    rejected so retries cannot silently replace its original lineage.
    """

    def __init__(
        self, program: CompiledTaskProgram, delegate: TaskProgramNominalSourceFactory
    ) -> None:
        if type(program) is not CompiledTaskProgram:
            raise TypeError("program must be a CompiledTaskProgram")
        if not isinstance(delegate, TaskProgramNominalSourceFactory):
            raise TypeError("delegate must implement TaskProgramNominalSourceFactory")
        if any(
            segment.parallel_block is not None for segment in program.iter_segments()
        ):
            raise ValueError("traced nominal sources require a sequential program")
        self._fingerprint = _fingerprint(program)
        self._delegate = delegate
        self._motions: dict[int, InstantiatedSkillMotion] = {}
        self.validate_program(program)

    @property
    def program_fingerprint(self) -> str:
        """Return the bound compiled-program fingerprint."""
        return self._fingerprint

    @property
    def motions(self) -> Mapping[int, InstantiatedSkillMotion]:
        """Return independent instantiated motions in call-occurrence order."""
        return MappingProxyType(
            {index: replace(motion) for index, motion in self._motions.items()}
        )

    def validate_program(self, program: CompiledTaskProgram) -> None:
        """Reject a changed or unsupported program before source selection."""
        if _fingerprint(program) != self._fingerprint:
            raise ValueError("traced nominal sources do not match the compiled program")
        validator = getattr(self._delegate, "validate_program", None)
        if validator is not None:
            if not callable(validator):
                raise TypeError("delegate validate_program must be callable")
            validator(program)

    def create_nominal_source(
        self, request: TaskProgramPlanRequest, *, engine: AtomicActionEngine
    ) -> NominalMotionSource | None:
        """Wrap the selected source and capture its successful instantiation."""
        source = self._delegate.create_nominal_source(request, engine=engine)
        if source is None:
            return None
        if not isinstance(source, NominalMotionSource):
            raise TypeError("delegate returned an invalid nominal source")
        owner, index = self, request.analysis_call_index

        class _TraceSource:
            def instantiate(
                self, resolved: ResolvedActionRequest, context: PlanningContext
            ) -> InstantiatedSkillMotion:
                if index in owner._motions:
                    raise ValueError(
                        "a traced call occurrence cannot be instantiated twice"
                    )
                motion = source.instantiate(resolved, context)
                if not isinstance(motion, InstantiatedSkillMotion):
                    raise TypeError("nominal source returned an invalid motion")
                if motion.trajectory.batch_size != 1:
                    raise ValueError("traced nominal motions require B=1")
                owner._motions[index] = replace(motion)
                return motion

        return _TraceSource()
