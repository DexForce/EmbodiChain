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

"""Immutable named-preset bindings for compiled sequential call occurrences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from embodichain.lab.task_program.compiler.program import (
    CompiledTaskProgram,
    CompiledTaskProgramAnalysis,
)
from embodichain.lab.task_program.semantics.integration import (
    SemanticIntegrationManifest,
)

from ._program_identity import task_program_fingerprint

__all__ = [
    "InvocationPolicyBinding",
    "BoundInvocationPolicies",
    "task_program_fingerprint",
]


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a non-empty exact string without outer whitespace."
        )


@dataclass(frozen=True, slots=True)
class InvocationPolicyBinding:
    """Select one declared preset for one occurrence of a compiled program.

    Attributes:
        program_fingerprint: Canonical SHA-256 of the complete compiled program.
        invocation_id: Execution ID ``program_id/segment_id:segment_call_index``.
        preset_id: Named policy preset already declared by the robot profile.
    """

    program_fingerprint: str
    invocation_id: str
    preset_id: str

    def __post_init__(self) -> None:
        _identifier(self.program_fingerprint, "program_fingerprint")
        if len(self.program_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in self.program_fingerprint
        ):
            raise ValueError("program_fingerprint must be a lowercase SHA-256 digest.")
        _identifier(self.invocation_id, "invocation_id")
        _identifier(self.preset_id, "preset_id")


@dataclass(frozen=True, slots=True, init=False)
class BoundInvocationPolicies:
    """Own validated occurrence selections without retaining mutable runtime state.

    Sequential calls may be selected in a program that also contains parallel
    segments; binding a parallel-branch call is rejected. Unselected calls retain
    the integration's normal preset selection. ``validate_manifest`` checks the
    selected profile presets before any execution begins.
    """

    program_fingerprint: str
    bindings: tuple[InvocationPolicyBinding, ...]
    _segment_presets: Mapping[int, tuple[str | None, ...]] = field(repr=False)

    def __init__(
        self, program: CompiledTaskProgram, bindings: Iterable[InvocationPolicyBinding]
    ) -> None:
        """Validate identities and own an immutable selection table.

        Args:
            program: Exact provider-free compiled program owning these occurrences.
            bindings: Unique named-preset declarations for sequential occurrences.

        Raises:
            TypeError: If a program or binding has the wrong type.
            ValueError: If a fingerprint, occurrence or uniqueness check fails.
        """
        fingerprint = task_program_fingerprint(program)
        declarations = tuple(bindings)
        if not all(type(item) is InvocationPolicyBinding for item in declarations):
            raise TypeError(
                "bindings must contain exact InvocationPolicyBinding values."
            )
        selected: dict[str, str] = {}
        for item in declarations:
            if item.program_fingerprint != fingerprint:
                raise ValueError("Invocation policy program fingerprint mismatch.")
            if item.invocation_id in selected:
                raise ValueError(
                    f"Duplicate invocation policy binding: {item.invocation_id!r}."
                )
            selected[item.invocation_id] = item.preset_id
        sequential = set()
        parallel = set()
        segment_presets = {}
        for segment in program.iter_segments():
            prefix = f"{program.program_id}/{segment.segment_id}"
            if segment.parallel_block is not None:
                for branch in segment.parallel_block.branches:
                    parallel.update(
                        f"{prefix}:branch_{branch.branch_index}:{index}"
                        for index, _ in enumerate(branch.calls)
                    )
                continue
            identities = tuple(
                f"{prefix}:{call.segment_call_index}" for call in segment.calls
            )
            sequential.update(identities)
            segment_presets[segment.segment_index] = tuple(
                selected.get(identity) for identity in identities
            )
        if set(selected) & parallel:
            raise ValueError(
                "Invocation policy bindings do not support parallel branch calls."
            )
        unknown = set(selected) - sequential
        if unknown:
            raise ValueError(
                f"Unknown sequential invocation policy occurrences: {sorted(unknown)}."
            )
        object.__setattr__(self, "program_fingerprint", fingerprint)
        object.__setattr__(self, "bindings", declarations)
        object.__setattr__(self, "_segment_presets", MappingProxyType(segment_presets))

    def validate_program(self, program: CompiledTaskProgram) -> None:
        """Reject use with a different or mutated compiled program.

        Args:
            program: Program about to execute through a fresh bridge.
        """
        if task_program_fingerprint(program) != self.program_fingerprint:
            raise ValueError("Invocation policy program fingerprint mismatch.")

    def validate_manifest(
        self, program: CompiledTaskProgram, manifest: SemanticIntegrationManifest
    ) -> None:
        """Check selected preset capabilities and unchanged effect assurance.

        Args:
            program: Program owning the selected occurrences.
            manifest: Exact static scene, profile and call catalog to use.
        """
        self.validate_program(program)
        if type(manifest) is not SemanticIntegrationManifest:
            raise TypeError("manifest must be exactly SemanticIntegrationManifest.")
        selection = program.integration
        if (
            manifest.robot_profile.profile_id != selection.robot_profile
            or manifest.runtime_preset != selection.runtime_preset
        ):
            raise ValueError(
                "Invocation policy manifest does not match the program integration selection."
            )
        for segment in program.iter_segments():
            if segment.parallel_block is not None:
                continue
            for call, preset_id in zip(
                segment.calls, self._segment_presets[segment.segment_index]
            ):
                if preset_id is not None:
                    manifest.link_call(
                        call.call, preset_id=preset_id, path=call.source_path
                    )

    def _presets_for_analysis(
        self, analysis: CompiledTaskProgramAnalysis
    ) -> tuple[str | None, ...]:
        """Return occurrence-aligned selections for a sequential analysis window.

        Args:
            analysis: Compiled sequential stretch or suffix from the owning program.

        Returns:
            One optional preset ID per call, preserving repeated occurrences.
        """
        if (
            type(analysis) is not CompiledTaskProgramAnalysis
            or analysis.kind == "parallel_branch"
        ):
            raise ValueError(
                "Policy selection requires a sequential compiled analysis."
            )
        result = tuple(
            value
            for index in analysis.segment_indices
            for value in self._segment_presets[index]
        )
        if len(result) != len(analysis.calls):
            raise ValueError(
                "Policy selections do not match the compiled analysis window."
            )
        return result
