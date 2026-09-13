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
"""Read-only access to a Spawn-owned Newton gradient model and state."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["NewtonDifferentiableRuntime"]


class NewtonDifferentiableRuntime:
    """Expose Newton model/state buffers required by kinematic environments.

    The backend provider is resolved for every access so a scene rebuild cannot
    silently return buffers owned by a replaced Spawn backend. This facade does
    not expose controls, contacts, solver stepping, or gradient rollouts.
    """

    def __init__(self, backend_provider: Callable[[], Any]) -> None:
        self._backend_provider = backend_provider

    def _backend(self) -> Any:
        backend = self._backend_provider()
        if backend is None:
            raise RuntimeError(
                "The Spawn-owned Newton backend is unavailable. Call "
                "SimulationManager.prepare() before using differentiable "
                "kinematics."
            )
        return backend

    def _validated_backend(self) -> Any:
        backend = self._backend()
        if backend.model is None:
            raise RuntimeError(
                "The Spawn-owned Newton model is not finalized. Call "
                "SimulationManager.prepare() first."
            )
        if not bool(backend.cfg.requires_grad):
            raise RuntimeError(
                "Differentiable Newton kinematics require requires_grad=True."
            )
        return backend

    @staticmethod
    def _spawn_runtime(backend: Any) -> Any:
        """Return DexSim's runtime facade across the 0.4/0.5 API boundary."""
        runtime = getattr(backend, "runtime", None)
        if runtime is None:
            runtime = getattr(backend, "_runtime", None)
        if runtime is None:
            raise RuntimeError("The Spawn-owned Newton runtime is unavailable.")
        return runtime

    @property
    def model(self) -> Any:
        """Return the finalized differentiable Newton model."""
        return self._validated_backend().model

    @property
    def current_state(self) -> Any:
        """Return the live state currently selected by the Spawn runtime."""
        backend = self._validated_backend()
        return self._spawn_runtime(backend).current_state

    @property
    def live_states(self) -> tuple[Any, Any]:
        """Return both live ping-pong states owned by the Spawn backend."""
        backend = self._validated_backend()
        return backend.state_0, backend.state_1
