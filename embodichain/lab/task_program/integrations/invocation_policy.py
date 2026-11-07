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
import hashlib
import json
from types import MappingProxyType

from embodichain.lab.task_program.compiler.program import (
    CompiledTaskProgram,
    CompiledTaskProgramAnalysis,
)
from embodichain.lab.task_program.semantics.integration import (
    SemanticIntegrationManifest,
)

from ._program_identity import task_program_fingerprint
from .catalog import _canonical_value

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


def _call_digest(call: object) -> str:
    """Snapshot a call's complete declaration without retaining mutable aliases."""
    encoded = json.dumps(
        _canonical_value(call), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


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


def _snapshot_invocation_policy_bindings(
    values: tuple[InvocationPolicyBinding, ...],
) -> tuple[InvocationPolicyBinding, ...]:
    """Own one exact immutable declaration tuple before program validation."""
    if type(values) is not tuple or not all(
        type(item) is InvocationPolicyBinding for item in values
    ):
        raise TypeError("invocation_policy_bindings must be an exact typed tuple.")
    if len({item.invocation_id for item in values}) != len(values):
        raise ValueError("Duplicate invocation policy binding.")
    return tuple(
        InvocationPolicyBinding(
            item.program_fingerprint, item.invocation_id, item.preset_id
        )
        for item in values
    )


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
    _program_id: str = field(repr=False)
    _segment_calls: Mapping[int, tuple[str, ...]] = field(repr=False)
    _segment_paths: Mapping[int, tuple[str | int, ...]] = field(repr=False)
    _analysis_windows: Mapping[int, tuple[int, ...]] = field(repr=False)

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
        segment_calls = {}
        segment_paths = {}
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
            segment_calls[segment.segment_index] = tuple(
                _call_digest(call.call) for call in segment.calls
            )
            segment_paths[segment.segment_index] = segment.source_path
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
        object.__setattr__(self, "_program_id", program.program_id)
        object.__setattr__(self, "_segment_calls", MappingProxyType(segment_calls))
        object.__setattr__(self, "_segment_paths", MappingProxyType(segment_paths))
        object.__setattr__(
            self,
            "_analysis_windows",
            MappingProxyType(
                {
                    index: analysis.segment_indices
                    for analysis in program.preflight_analyses()
                    if analysis.kind == "sequential_stretch"
                    for index in analysis.segment_indices
                }
            ),
        )

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
            manifest: Static scene metadata, profile and call catalog to use.

        The profile ID and runtime preset must match the compiled selection.
        SceneManifest has no registry-selection ID: the environment adapter (or
        a host constructing a bridge directly) must validate scene ownership.
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
                "Invocation policy manifest does not match the program profile/preset selection."
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
        indices = analysis.segment_indices
        window = self._analysis_windows.get(indices[0])
        if window is None:
            raise ValueError(
                "Policy analysis does not belong to the bound program window."
            )
        is_stretch = analysis.kind == "sequential_stretch"
        expected_indices = (
            window
            if is_stretch
            else tuple(index for index in window if index >= indices[0])
        )
        phase = "preflight" if is_stretch else "execution"
        expected_id = f"{self._program_id}:{phase}:sequential:{indices[0]}-{window[-1]}"
        expected_prefix = (
            sum(len(self._segment_calls[index]) for index in expected_indices)
            if is_stretch
            else len(self._segment_calls[indices[0]])
        )
        if (
            indices != expected_indices
            or analysis.analysis_id != expected_id
            or analysis.source_path != self._segment_paths[indices[0]]
            or analysis.execution_prefix_length != expected_prefix
        ):
            raise ValueError(
                "Policy analysis does not belong to the bound program window."
            )
        expected_calls = tuple(
            value for index in indices for value in self._segment_calls[index]
        )
        if tuple(_call_digest(call) for call in analysis.calls) != expected_calls:
            raise ValueError(
                "Policy analysis calls differ from the bound program window."
            )
        return tuple(
            value for index in indices for value in self._segment_presets[index]
        )
