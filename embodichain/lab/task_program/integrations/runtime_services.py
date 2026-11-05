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

"""Read-only references to the actual bridge-owned runtime services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation.control_policy import ControlOperationRuntime

__all__ = ["TaskProgramRuntimeServiceContext"]


@dataclass(frozen=True, slots=True)
class TaskProgramRuntimeServiceContext:
    """Bind actual operation instances to one program/episode and policy session.

    This is a live assembly value, never a serialized configuration or an active-run
    proxy. The context retains the exact supplied services and checks the owning
    session before lookup; it does not create replacements or mutate bindings.
    """

    program_fingerprint: str
    episode_id: str
    control_operations: Mapping[str, ControlOperationRuntime]
    validate_scope: Callable[[], None] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        from .simulation.control_policy import ControlOperationRuntime

        for value in (self.program_fingerprint, self.episode_id):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Runtime service identity must be exact.")
        if not callable(self.validate_scope):
            raise TypeError("Runtime service context requires its actual owner guard.")
        operations = dict(self.control_operations)
        for key, operation in operations.items():
            if (
                type(operation) is not ControlOperationRuntime
                or key != operation.operation_id
            ):
                raise ValueError(
                    "Runtime services require exact declared control operations."
                )
            scope = operation.observation_scope
            if (
                scope["program_fingerprint"] != self.program_fingerprint
                or scope["episode_id"] != self.episode_id
            ):
                raise ValueError("Runtime service belongs to another program/episode.")
        object.__setattr__(self, "control_operations", MappingProxyType(operations))

    def control_operation(self, operation_id: str) -> ControlOperationRuntime:
        """Return the same actual operation used by this bridge's policy session."""
        self.validate_scope()
        if operation_id not in self.control_operations:
            raise ValueError("No control operation is declared in this runtime scope.")
        operation = self.control_operations[operation_id]
        operation.validate_scope()
        return operation
