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

"""Public functional dynamics facade over SimulationManager's Newton services."""

from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any, Protocol, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import FunctionalSolverState

__all__ = ["FunctionalDynamics", "detach_dynamics_state"]


class _DynamicsManager(Protocol):
    def create_functional_state(self) -> FunctionalSolverState: ...
    def functional_parameters(self) -> Mapping[str, torch.Tensor]: ...
    def functional_step(
        self,
        state: FunctionalSolverState,
        control: Any,
        parameters: Mapping[str, torch.Tensor] | None = None,
        dt: float | None = None,
        *,
        publish: bool = True,
        render_final_step: bool = True,
    ) -> FunctionalSolverState: ...
    def reset_functional_state(
        self, state: FunctionalSolverState, world_mask: torch.Tensor | None = None
    ) -> FunctionalSolverState: ...


def detach_dynamics_state(state: FunctionalSolverState) -> FunctionalSolverState:
    """Create an explicit BPTT boundary without reset or state resampling.

    Args:
        state: Owned functional state with tensor physical/history fields.

    Returns:
        A new state containing detached views. Both states remain immutable;
        native publication owns separate storage and cannot overwrite them.
    """
    from dexsim.engine.newton_physics import FunctionalSolverState

    return FunctionalSolverState(
        physical={k: v.detach() for k, v in state.physical.items()},
        history={k: v.detach() for k, v in state.history.items()},
        valid_history_lengths=(
            None
            if state.valid_history_lengths is None
            else state.valid_history_lengths.detach()
        ),
    )


class FunctionalDynamics:
    """Advance differentiable state through a manager's public Newton interface.

    Args:
        manager: SimulationManager or an object implementing the same public
            functional dynamics methods. No backend or native buffer is accessed.

    A physics step means one configured macro interval. DexSim alone owns its
    internal solver substeps. A control interval can contain several such steps.
    Native state publication is detached; returned state retains its graph.
    """

    def __init__(self, manager: _DynamicsManager) -> None:
        self._manager = manager

    def create_state(self) -> FunctionalSolverState:
        """Snapshot native state and solver history into independent storage."""
        return self._manager.create_functional_state()

    def parameters(self) -> Mapping[str, torch.Tensor]:
        """Return the live named solver parameters without copying or detaching."""
        return self._manager.functional_parameters()

    def step(
        self,
        state: FunctionalSolverState,
        control: Any,
        *,
        physics_steps: int = 1,
        physics_dt: float | None = None,
        parameters: Mapping[str, torch.Tensor] | None = None,
        publish: bool = True,
    ) -> FunctionalSolverState:
        """Hold control across physics_steps macro intervals and preserve BPTT.

        Args:
            state: Initial functional state.
            control: Provider-defined differentiable controls held for the interval.
            physics_steps: Positive macro step count, independent of solver substeps.
            physics_dt: Macro timestep; None uses the configured physics interval.
            parameters: Optional explicit parameter mapping for functional evaluation.
            publish: Whether to publish detached state and advance native time.

        Returns:
            Final owned functional state. Rendering is left to the environment.
        """
        if (
            isinstance(physics_steps, bool)
            or not isinstance(physics_steps, int)
            or physics_steps < 1
        ):
            raise ValueError("physics_steps must be a positive integer")
        if physics_dt is not None and (
            not math.isfinite(physics_dt) or physics_dt <= 0
        ):
            raise ValueError("physics_dt must be positive and finite")
        for _ in range(physics_steps):
            state = self._manager.functional_step(
                state,
                control,
                parameters=parameters,
                dt=physics_dt,
                publish=publish,
                render_final_step=False,
            )
        return state

    def reset(
        self, state: FunctionalSolverState, world_mask: torch.Tensor | None = None
    ) -> FunctionalSolverState:
        """Merge freshly reset native rows into state while preserving other graphs.

        The caller first performs the manager's normal native reset/randomization.
        This method snapshots those new rows and delegates ownership-aware merging
        and detached publication to the manager. It does not advance time.
        """
        return self._manager.reset_functional_state(state, world_mask=world_mask)
