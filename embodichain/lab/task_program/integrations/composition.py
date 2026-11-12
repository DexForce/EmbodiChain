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

"""Public composition of configured Task Program deployments.

Composition resolves declarations and constructs lazy adapter factories. It does
not register a Gym environment, instantiate simulation, or execute a program.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from embodichain.lab.evaluation import PredicateRegistry
    from embodichain.lab.task_program.language.schema import TaskProgramCfg

from embodichain.lab.task_program.language.schema import TaskProgramIntegrationCfg

from .catalog import SimulationTaskProgramRegistration
from .environment import TaskProgramAdapterFactory

__all__ = [
    "ConfiguredTaskProgramDeployment",
    "ConfiguredTaskProgramIntegration",
    "load_configured_task_program_deployment",
]


@dataclass(frozen=True, slots=True)
class ConfiguredTaskProgramIntegration:
    """Decoded integration identity and its lazy environment adapter factory.

    Attributes:
        registration: Validated scene, robot-profile and semantic-call catalog.
        adapter_factory: Factory called only after the host creates its live scene.
        integration_fingerprint: Identity of the composed runtime declarations.
    """

    registration: SimulationTaskProgramRegistration
    adapter_factory: TaskProgramAdapterFactory
    integration_fingerprint: str


@dataclass(frozen=True, slots=True)
class ConfiguredTaskProgramDeployment:
    """Resolved component metadata for one configured deployment.

    An inline program is retained as an owned declaration; file selections keep
    their resolved path. ``program_source`` returns the selected representation.

    Attributes:
        integration_id: Task integration identifier from the selected component.
        program_id: Program identifier declared by that integration.
        program_path: Resolved file path, or None for an inline program.
        selection: Trusted scene, robot-profile and execution-preset selection.
        integration: Validated registration and lazy live adapter factory.
        scene_binding (dict): Independently owned semantic scene declaration.
        planner_config (dict | None): Independently owned planner declaration,
            when selected.
    """

    integration_id: str
    program_id: str
    program_path: Path | None
    selection: TaskProgramIntegrationCfg
    integration: ConfiguredTaskProgramIntegration
    scene_binding: dict[str, object]
    planner_config: dict[str, object] | None
    program_declaration: dict[str, Any] | None = None

    @property
    def program_source(self) -> Path | dict[str, Any]:
        """Return a file path or owned inline program for strict decoding.

        Returns:
            pathlib.Path | dict: The selected immutable-file reference or a fresh
            declaration dictionary whose values remain under the strict decoder.
        """
        if self.program_path is not None:
            return self.program_path
        if self.program_declaration is None:
            raise ValueError("Configured deployment has no program source.")
        return deepcopy(self.program_declaration)

    def load_program(self) -> TaskProgramCfg:
        """Decode and preflight the selected program without live runtime creation."""
        from ._configured_composition import _load_deployment_program

        registration = self.integration.registration
        registration.assert_unchanged()
        program = _load_deployment_program(
            self.program_source, selection=self.selection, catalog=registration.catalog
        )
        if program.program_id != self.program_id:
            raise ValueError(
                "Configured deployment requires the integration's program_id."
            )
        registration.catalog.preflight(program)
        return program


def load_configured_task_program_deployment(
    *,
    task_program: Mapping[str, object],
    skill_profile: Mapping[str, object],
    base_dir: str | Path,
    evaluation: Mapping[str, object] | None = None,
    evaluation_registry: PredicateRegistry | None = None,
) -> ConfiguredTaskProgramDeployment:
    """Compose deployment metadata using the shared strict component decoders.

    Args:
        task_program: Program, integration and execution-policy component selectors
            from a runnable deployment's ``task_program`` mapping.
        skill_profile: The resolved embodiment component's ``skill_profile`` mapping.
        base_dir: Owning deployment directory for relative component paths.
        evaluation: Owned rule library selected by the deployment, if any.
        evaluation_registry: Optional trusted registry of built-in and extension predicates.

    Returns:
        A fresh deployment with its validated registration, trusted selection and
        lazy adapter factory. Inputs and previously loaded deployments are not
        mutated. The caller retains responsibility for physical component checks,
        program loading and live environment construction.

    Raises:
        TypeError: If a declaration has an invalid value type.
        ValueError: If fields, component contracts or policy values are invalid.
        FileNotFoundError: If a selected component cannot be resolved.

    .. note::
        Declared nominal motion sources retain their existing static program
        preflight. Composition itself does not register an environment or create
        a simulator, renderer, grasp service or live adapter.
    """
    from ._configured_composition import _load_configured_task_program_deployment

    return _load_configured_task_program_deployment(
        task_program=task_program,
        skill_profile=skill_profile,
        base_dir=base_dir,
        evaluation=evaluation,
        evaluation_registry=evaluation_registry,
    )
