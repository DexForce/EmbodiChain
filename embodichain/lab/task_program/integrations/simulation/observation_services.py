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

"""Own explicitly declared physical collectors for one actual bridge episode."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
from .cargo_observation import SimulationCargoEnvelopeCollector
from .control_observations import SimulationControlObservationCollector
from .observations import (
    SimulationStateObservationCollector,
    SimulationSupportObservationCollector,
)
from .recorded_contacts import SimulationRecordedContactCollector

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from ..catalog import SimulationTaskProgramRegistration

__all__ = ["SimulationObservationServices"]


class SimulationObservationServices:
    """Create actual declared collectors before any execution sample can be lost.

    The host owns the environment, bridge, control runtime and ObservationManager.
    This object closes only its recorded-contact hooks and Cargo collector. It
    never steps, refreshes a sensor, creates a simulator or evaluates a TaskSpec.
    Completed control windows remain owned by the borrowed policy runtime.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        registration: SimulationTaskProgramRegistration,
        *,
        observations: ObservationManager,
    ) -> None:
        """Assemble exact per-episode services from the sealed registration.

        Args:
            environment: Initialized environment whose active bridge is supplied.
            bridge: Actual fresh bridge, bound before its first control step.
            registration: Exact scene, profile and optional collector declarations.
            observations: Actual environment-owned canonical observation manager.

        Raises:
            ValueError: A runtime, episode, declaration or history owner differs.
            TypeError: The bridge, registration or manager has another type.
        """
        from ..catalog import SimulationTaskProgramRegistration

        if (
            type(bridge) is not TaskProgramDemoBridge
            or type(registration) is not SimulationTaskProgramRegistration
            or not isinstance(observations, ObservationManager)
        ):
            raise TypeError("Observation services require their actual typed owners.")
        if (
            environment.active_task_program_bridge is not bridge
            or environment.observation_manager is not observations
            or bridge.clock.step_index != 0
            or bridge.program_completed
        ):
            raise ValueError("Bind observation services to the actual fresh runtime.")
        if getattr(environment, "_closed", False):
            raise RuntimeError(
                "Physical observation collection is unavailable after close."
            )
        registration.assert_unchanged()
        history = observations if observations.has_history else None
        if history is not None:
            history.validate_history_owner(environment.episode_context)
        if history is None and (
            registration.recorded_contacts_cfg is not None
            or registration.cargo_envelope_cfg is not None
        ):
            raise ValueError(
                "Declared recorded observations require enabled raw history."
            )
        self._closed = False
        self._owner_release: Callable[[], None] | None = None
        self._recorded_contacts = None
        self._cargo = None
        self._state = SimulationStateObservationCollector(
            environment, bridge, registration, observations=history
        )
        self._support = SimulationSupportObservationCollector(
            environment, bridge, registration, observations=history
        )
        self._controls = MappingProxyType(
            {
                declaration.call.operation_id: SimulationControlObservationCollector(
                    environment, bridge, declaration.call.operation_id
                )
                for declaration in registration.control_operations
            }
        )
        try:
            if registration.recorded_contacts_cfg is not None:
                self._recorded_contacts = SimulationRecordedContactCollector(
                    environment,
                    bridge,
                    registration,
                    registration.recorded_contacts_cfg,
                    observations=history,
                )
            if registration.cargo_envelope_cfg is not None:
                self._cargo = SimulationCargoEnvelopeCollector(
                    environment,
                    bridge,
                    registration.cargo_envelope_cfg,
                    observations=history,
                )
        except BaseException:
            self.close()
            raise

    @property
    def state(self) -> SimulationStateObservationCollector:
        """Return the actual State collector using the borrowed canonical history."""
        return self._state

    @property
    def support(self) -> SimulationSupportObservationCollector:
        """Return the actual support collector for this policy session."""
        return self._support

    @property
    def controls(self) -> Mapping[str, SimulationControlObservationCollector]:
        """Return actual declared operation collectors, keyed by operation identity."""
        return self._controls

    @property
    def recorded_contacts(self) -> SimulationRecordedContactCollector | None:
        """Return the declared recorded-contact producer, or None when unselected."""
        return self._recorded_contacts

    @property
    def cargo(self) -> SimulationCargoEnvelopeCollector | None:
        """Return the declared initial/recorded/terminal Cargo producer if selected."""
        return self._cargo

    @property
    def closed(self) -> bool:
        """Return whether owned observation hooks have been released."""
        return self._closed

    def close(self) -> None:
        """Close owned collectors without closing borrowed runtime or raw history.

        Repeated cleanup is harmless. Every owned collector is attempted when a
        cleanup fails, and the first error is propagated to the host.
        """
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for collector in (self._recorded_contacts, self._cargo):
            if collector is not None:
                try:
                    collector.close()
                except BaseException as error:
                    errors.append(error)
        if self._owner_release is not None:
            release, self._owner_release = self._owner_release, None
            release()
        if errors:
            raise errors[0]
