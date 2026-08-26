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
"""Differentiable transactions over a Spawn-owned Newton runtime."""

from __future__ import annotations

import math
from typing import Any, Callable

__all__ = ["NewtonDifferentiableRuntime"]


class NewtonDifferentiableTrajectory:
    """Own the detached buffers for one differentiable Newton trajectory."""

    def __init__(
        self,
        runtime: "NewtonDifferentiableRuntime",
        *,
        physics_steps: int,
        physics_dt: float,
    ) -> None:
        self._runtime = runtime
        self._backend = runtime._validated_backend()
        self.physics_steps = int(physics_steps)
        self.physics_dt = float(physics_dt)
        self.total_solver_steps = self.physics_steps * runtime.num_substeps
        self.solver_dt = self.physics_dt / runtime.num_substeps

        model = self._backend.model
        self.states = [model.state() for _ in range(self.total_solver_steps + 1)]
        self.states[0].assign(self._backend.runtime.current_state)
        self.control = model.control()
        self.contacts = [
            self._backend.collision_pipeline.contacts()
            for _ in range(self.total_solver_steps)
        ]
        self._stepped = False
        self._committed = False
        self._released = False

    @property
    def final_state(self) -> Any:
        """Return the terminal state owned by this trajectory."""
        return self.states[-1]

    def step(self) -> Any:
        """Run the complete trajectory inside the caller's active Warp tape."""
        if self._released:
            raise RuntimeError("Cannot step a released differentiable trajectory.")
        if self._stepped:
            raise RuntimeError("A differentiable trajectory can only be stepped once.")
        if self._runtime._backend() is not self._backend:
            raise RuntimeError(
                "The Spawn-owned Newton backend changed while a differentiable "
                "trajectory was active. Release it and create a fresh trajectory."
            )

        backend = self._backend
        apply_external_wrenches = backend.runtime.has_external_wrenches
        for index, (state_in, state_out, contacts) in enumerate(
            zip(self.states, self.states[1:], self.contacts)
        ):
            state_in.clear_forces()
            if apply_external_wrenches and index < self._runtime.num_substeps:
                backend.runtime.apply_external_wrenches(state_in)
            if backend.cfg.enable_collision_pipeline:
                backend.collision_pipeline.collide(state_in, contacts)
            backend.solver.step(
                state_in,
                state_out,
                self.control,
                contacts,
                self.solver_dt,
            )
        self._stepped = True
        return self.final_state

    def release(self) -> None:
        """Release the runtime lease after the owning Warp tape is reset."""
        if self._released:
            return
        self._runtime._release_differentiable_trajectory(self)
        self._released = True


class NewtonDifferentiableRuntime:
    """Adapt the current Spawn-owned Newton backend to the autograd bridge.

    The provider is resolved for every public operation so a scene rebuild
    cannot silently publish a trajectory into a replaced Newton backend.
    """

    def __init__(self, backend_provider: Callable[[], Any]) -> None:
        self._backend_provider = backend_provider
        self._active_trajectory: NewtonDifferentiableTrajectory | None = None

    def _backend(self) -> Any:
        backend = self._backend_provider()
        if backend is None:
            raise RuntimeError(
                "The Spawn-owned Newton backend is unavailable. Call "
                "SimulationManager.prepare() before using differentiable physics."
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
                "Differentiable Newton physics requires requires_grad=True."
            )
        if backend.cfg.solver_cfg.solver_type != "semi_implicit":
            raise RuntimeError(
                "Differentiable Newton physics requires " "solver_type='semi_implicit'."
            )
        if backend.collision_pipeline is None:
            raise RuntimeError(
                "Differentiable Newton physics requires a collision pipeline."
            )
        if getattr(backend, "_runtime_controls", ()):
            raise RuntimeError(
                "Differentiable trajectories do not support Spawn runtime "
                "controls yet. Remove them before finalizing the scene."
            )
        return backend

    @property
    def model(self) -> Any:
        """Return the finalized Newton model for expert Warp operations."""
        return self._validated_backend().model

    @property
    def current_state(self) -> Any:
        """Return the live state currently selected by the Spawn runtime."""
        return self._validated_backend().runtime.current_state

    @property
    def live_states(self) -> tuple[Any, Any]:
        """Return both live ping-pong states owned by the Spawn backend."""
        backend = self._validated_backend()
        return backend.state_0, backend.state_1

    @property
    def control(self) -> Any:
        """Return the live Spawn control buffer."""
        return self._validated_backend().control

    @property
    def num_substeps(self) -> int:
        """Return the number of Newton solver substeps per physics step."""
        return max(int(self._validated_backend().cfg.num_substeps), 1)

    @property
    def physics_dt(self) -> float:
        """Return the configured outer physics-step duration."""
        return float(self._validated_backend().cfg.dt)

    @property
    def solver_dt(self) -> float:
        """Return the configured Newton solver substep duration."""
        return self.physics_dt / self.num_substeps

    # Compatibility aliases consumed by DexSim's low-level differentiable
    # stepper/rollout helpers. They borrow, but never own, Spawn resources.
    @property
    def _model(self) -> Any:
        return self.model

    @property
    def _state_0(self) -> Any:
        return self._validated_backend().state_0

    @property
    def _state_1(self) -> Any:
        return self._validated_backend().state_1

    @property
    def _control(self) -> Any:
        return self.control

    @property
    def _solver(self) -> Any:
        return self._validated_backend().solver

    @property
    def _collision_pipeline(self) -> Any:
        return self._validated_backend().collision_pipeline

    @property
    def _external_forces(self) -> Any:
        return self._validated_backend().runtime.external_wrenches

    def _ensure_external_force_buffers(self) -> None:
        self._validated_backend()

    def clear_external_forces(self) -> None:
        """Clear pending Spawn runtime wrenches."""
        self._validated_backend().runtime.clear_external_wrenches()

    def create_differentiable_trajectory(
        self,
        *,
        physics_steps: int,
        physics_dt: float,
    ) -> NewtonDifferentiableTrajectory:
        """Allocate one detached trajectory and acquire the runtime lease."""
        if isinstance(physics_steps, bool) or int(physics_steps) != physics_steps:
            raise TypeError("physics_steps must be a positive integer.")
        physics_steps = int(physics_steps)
        if physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer.")
        try:
            physics_dt = float(physics_dt)
        except (TypeError, ValueError) as exc:
            raise TypeError("physics_dt must be a positive finite float.") from exc
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError("physics_dt must be a positive finite float.")
        if self._active_trajectory is not None:
            raise RuntimeError(
                "A differentiable trajectory is still active; release it after "
                "backward before creating another trajectory."
            )

        trajectory = NewtonDifferentiableTrajectory(
            self,
            physics_steps=physics_steps,
            physics_dt=physics_dt,
        )
        self._active_trajectory = trajectory
        return trajectory

    def commit_differentiable_trajectory(
        self,
        trajectory: NewtonDifferentiableTrajectory,
    ) -> None:
        """Publish one detached terminal state back to the live Spawn runtime."""
        if self._active_trajectory is not trajectory:
            raise RuntimeError(
                "The differentiable trajectory is not active on this runtime."
            )
        if trajectory._released:
            raise RuntimeError("Cannot commit a released differentiable trajectory.")
        if trajectory._committed:
            raise RuntimeError(
                "A differentiable trajectory can only be committed once."
            )
        if not trajectory._stepped:
            raise RuntimeError(
                "Step the differentiable trajectory before committing it."
            )

        backend = self._validated_backend()
        if backend is not trajectory._backend:
            raise RuntimeError(
                "The Spawn-owned Newton backend changed before trajectory commit."
            )
        backend.state_0.assign(trajectory.final_state)
        backend.state_1.assign(trajectory.final_state)
        backend.runtime.set_current_state(backend.state_0)
        backend.runtime.clear_external_wrenches()
        backend.set_sim_time(
            backend.sim_time + trajectory.physics_steps * trajectory.physics_dt,
            backend.step_index + trajectory.physics_steps,
        )
        trajectory._committed = True

    def _release_differentiable_trajectory(
        self,
        trajectory: NewtonDifferentiableTrajectory,
    ) -> None:
        if self._active_trajectory is not trajectory:
            raise RuntimeError(
                "The differentiable trajectory is not active on this runtime."
            )
        self._active_trajectory = None

    def create_differentiable_stepper(self) -> Any:
        """Create DexSim's low-level differentiable Newton step primitive."""
        self._validated_backend()
        from dexsim.engine.newton_physics.differentiable_stepper import (
            DifferentiableStepper,
        )

        return DifferentiableStepper(self)

    def create_gradient_rollout(
        self,
        record_steps: int,
        substeps_per_record: int | None = None,
        record_dt: float | None = None,
    ) -> Any:
        """Create DexSim's standalone gradient-rollout buffers."""
        backend = self._validated_backend()
        record_steps = int(record_steps)
        if record_steps <= 0:
            raise ValueError("record_steps must be positive.")
        substeps = (
            self.num_substeps
            if substeps_per_record is None
            else int(substeps_per_record)
        )
        if substeps <= 0:
            raise ValueError("substeps_per_record must be positive.")
        duration = self.physics_dt if record_dt is None else float(record_dt)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("record_dt must be a positive finite float.")

        from dexsim.engine.newton_physics.gradient_rollout import GradientRollout

        total_substeps = record_steps * substeps
        states = [backend.model.state() for _ in range(total_substeps + 1)]
        states[0].assign(backend.runtime.current_state)
        contacts = [
            backend.collision_pipeline.contacts() for _ in range(total_substeps)
        ]
        return GradientRollout(
            self,
            record_steps=record_steps,
            substeps_per_record=substeps,
            record_dt=duration,
            states=states,
            control=backend.model.control(),
            contacts=contacts,
            stepper=self.create_differentiable_stepper(),
        )
