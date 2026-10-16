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
"""World-owned Newton (Warp) physics backend configuration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from collections.abc import Sequence
import weakref

import warp as wp

from .base import PhysicsBackend

if TYPE_CHECKING:
    import dexsim

    from embodichain.lab.sim.sim_manager import SimulationManagerCfg
    from embodichain.lab.sim.objects.fluid import FluidObject

__all__ = ["NewtonPhysicsBackend"]


def is_newton_gradient_mode(result) -> bool:
    """Return whether a finalized Spawn result uses Newton gradients."""
    if result is None or getattr(result, "backend", None) != "newton":
        return False
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    backend = get_newton_backend(result.world)
    if backend is None:
        return False
    return bool(
        backend.cfg.requires_grad
        or (backend.model is not None and backend.model.requires_grad)
    )


class NewtonPhysicsBackend(PhysicsBackend):
    """The Warp-based Newton physics backend integrated through DexSim."""

    name = "newton"

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._differentiable_runtime = None
        self._runtime_device: str | None = None
        self._configured_solver_type: str | None = None

    @property
    def solver_type(self) -> str | None:
        """Return the configured or scene-resolved Newton solver type."""
        world = getattr(self._manager, "_world", None)
        if world is not None:
            from dexsim.engine.newton_physics.backend_registry import (
                get_newton_backend,
            )

            backend = get_newton_backend(world)
            if backend is not None:
                return str(backend.solver_type)
        return self._configured_solver_type

    # -- construction / world-config activation ------------------------- #

    @property
    def cuda_graph_status(self) -> str:
        """Return the World-owned graph state without triggering capture."""
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(self._manager._world)
        return backend.cuda_graph_status if backend is not None else "pending"

    def configure_world(self, world_config, sim_config: "SimulationManagerCfg") -> None:
        importlib.import_module("dexsim.engine.newton_physics")

        newton_physics_cfg = sim_config.physics_cfg
        newton_cfg = newton_physics_cfg.to_dexsim_cfg(
            gpu_id=sim_config.gpu_id,
        )
        self._configured_solver_type = str(newton_cfg.solver_cfg.solver_type)
        self._runtime_device = str(newton_cfg.device)
        world_config.newton_cfg = newton_cfg

    def activate(self, sim_config: "SimulationManagerCfg") -> None:
        del sim_config
        # WorldConfig.newton_cfg registers the World-owned NewtonBackend.
        # SceneBuilder.finalize() completes its model; no second manager-level
        # activation or rebuild domain participates.

    def sync_render_state(self, result: "dexsim.scene.Scene") -> None:
        """Publish Newton state through DexSim's render bridge without stepping."""
        if not self._manager.has_native_renderer:
            return
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(result.world)
        if backend is None:
            raise RuntimeError(
                "Newton backend is unavailable for render-state synchronization."
            )
        fluids = getattr(self._manager, "_fluid_objects", {})
        clipped = [f for f in fluids.values() if f.cfg.render.clip_below_z is not None]
        if not clipped:
            backend.sync_to_dexsim(result.world)
            return
        positions = wp.to_torch(backend.state_0.particle_q).clone()
        for fluid in clipped:
            for entity in fluid._entities:
                binding = entity._physics_binding
                start = binding.particle_start
                selected = positions[start : start + entity.particle_count]
                selected[selected[:, 2] < fluid.cfg.render.clip_below_z] = float("nan")
        backend.render_sync.sync(
            result.world,
            backend.state_0.body_q,
            wp.from_torch(positions, dtype=wp.vec3),
            simulation_time=backend.sim_time,
        )

    @property
    def supports_fluid_worlds(self) -> bool:
        """Whether the loaded SPH artifact supports isolated worlds and reset."""
        if self.solver_type != "sph":
            return False
        from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available

        solver = ensure_solver_available("sph")
        return bool(
            getattr(solver, "supports_world_isolation", False)
            and getattr(solver, "supports_world_reset", False)
        )

    def invalidate_fluid_state(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected SPH history after an explicit state write.

        Args:
            env_ids: Local worlds to clear, or all local worlds when omitted.
                Shared boundary history remains unchanged. Legacy single-world
                artifacts rebaseline their optional SDF history directly.
        """
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(self._manager._world)
        if backend is None or getattr(backend, "sph_solver", None) is None:
            raise RuntimeError("Prepared SPH backend required.")
        backend.invalidate_cuda_graph("fluid state write")
        solver = getattr(backend, "sph_solver", backend.solver)
        if getattr(solver, "supports_world_reset", False):
            mask = [
                env_ids is None or i in env_ids
                for i in range(backend.model.world_count)
            ] + [False]
            solver.reset(
                backend._runtime.current_state,
                world_mask=wp.array(mask, dtype=wp.bool, device=backend.model.device),
                flags=0,
            )
            return
        boundary = solver.sdf_boundary
        if boundary is not None and boundary._body_q_prev is not None:
            boundary._body_q_prev.assign(backend.state_0.body_q)
        reset_warm_start = getattr(solver, "reset_dfsph_warm_start", None)
        if reset_warm_start is not None:
            reset_warm_start()

    def fluid_diagnostics(self, fluid: FluidObject) -> dict[str, object]:
        """Read detached native SPH diagnostics without publishing or stepping."""
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend
        import torch

        backend = get_newton_backend(self._manager._world)
        solver = getattr(backend, "sph_solver", backend.solver)
        density = wp.to_torch(solver.particle_density)
        values = []
        for entity in fluid._entities:
            start = entity._physics_binding.particle_start
            values.append(density[start : start + entity.particle_count].clone())
        return {
            "particle_density": torch.stack(values),
            "volume_map_enabled": solver.volume_map_boundary is not None,
            "sdf_penalty_enabled": solver.sdf_boundary is not None,
            "boundary_shape_count": (
                0
                if solver.volume_map_boundary is None
                else solver.volume_map_boundary.shape_count
            ),
            "rest_density_kg_m3": solver.rest_density,
            "viscosity": solver.viscosity,
        }

    def _step(self, physics_dt: float) -> None:
        """Let the manager batch window and offscreen consumers after stepping."""
        self._manager._world.update(physics_dt, sync_to_dexsim=False)

    @property
    def _requires_step_render_sync(self) -> bool:
        return self._manager.sim_config.physics_cfg.sync_to_renderer is True

    def prepare_for_teardown(self) -> None:
        """Release Newton render views while Spawn still owns their parents."""
        if self._runtime_device is not None and self._runtime_device.startswith("cuda"):
            wp.synchronize_device(self._runtime_device)

        world = getattr(self._manager, "_world", None)
        if world is None:
            return

        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(world)
        if backend is not None:
            # NewtonRenderSync retains native link-node wrappers. They must be
            # released before Scene.close() drops the owning skeletons;
            # otherwise pybind can destruct a child after its native parent.
            backend.render_sync.clear()

    @property
    def newton_manager(self):
        """Reject access to the removed, independently owned Newton manager."""
        raise RuntimeError(
            "NewtonManager is not part of Spawn scene ownership. Use "
            "SimulationManager.spawn_result and its Spawned*/Batch APIs."
        )

    @property
    def differentiable_runtime(self):
        """Return the differentiable facade over the Spawn-owned runtime."""
        if self._differentiable_runtime is None:
            from embodichain.lab.sim.diff.runtime import NewtonDifferentiableRuntime

            owner_ref = weakref.ref(self)

            def backend_provider():
                owner = owner_ref()
                if owner is None:
                    return None
                result = owner._manager.spawn_result
                if result is None:
                    return None
                from dexsim.engine.newton_physics.backend_registry import (
                    get_newton_backend,
                )

                return get_newton_backend(result.world)

            self._differentiable_runtime = NewtonDifferentiableRuntime(backend_provider)
        return self._differentiable_runtime

    # -- scene ---------------------------------------------------------- #
    def get_scene(self):
        raise RuntimeError(
            "Newton Spawn scenes do not expose a PhysicsScene. Use "
            "SimulationManager.spawn_result and its Spawned*/Batch APIs."
        )

    # -- capabilities --------------------------------------------------- #
    @property
    def supports_volume_deformables(self) -> bool:
        return True

    @property
    def supports_surface_deformables(self) -> bool:
        return True

    @property
    def supports_robot(self) -> bool:
        # Robots are SpawnedArticulations in the World-owned Newton model.
        return True

    @property
    def supports_rigid_object_group(self) -> bool:
        # Groups are env-major views over the Scene rigid-body batch, which
        # provides the same state and mass-property API on Newton.
        return True

    @property
    def supports_contact_sensor(self) -> bool:
        # ContactSensor consumes the backend-neutral Scene ContactQuery API,
        # but not every Newton solver/device pair currently publishes that
        # query. Keep this runtime-sensitive so ``solver_type='auto'`` can be
        # rechecked after Spawn finalization resolves the concrete solver.
        solver_type = self.solver_type
        if solver_type is None:
            return True
        solver_type = solver_type.lower()
        if solver_type.endswith("dexuni"):
            return False
        runtime_device = self._runtime_device
        if runtime_device is None:
            runtime_device = str(getattr(self._manager, "device", ""))
        return not (
            solver_type.endswith("mujoco_warp")
            and runtime_device.lower().startswith("cpu")
        )
