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
from collections.abc import Sequence
from typing import TYPE_CHECKING
import weakref

import warp as wp
import numpy as np

from .base import PhysicsBackend

if TYPE_CHECKING:
    import dexsim

    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

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
        self._initial_joint_state_targets: set[str] = set()

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

    def _prepare_scene_build(self) -> None:
        """Preflight cable anchors and freeze episode joint defaults once."""
        manager = self._manager
        cables = getattr(manager, "_cable_objects", {})
        if not cables:
            return
        from embodichain.lab.sim._runtime_controls import _InitialJointState

        articulations = {**manager._robots, **manager._articulations}
        # Normalize the entire input before registration so an invalid config
        # cannot leave an incomplete set of initialization controls behind.
        positions = {
            uid: _InitialJointState.normalize_positions(obj.cfg.init_qpos)
            for uid, obj in articulations.items()
            if obj.cfg.init_qpos is not None
        }
        for cable in cables.values():
            for anchor in (cable.cfg.start_attachment, cable.cfg.end_attachment):
                if anchor is not None and anchor.target in positions:
                    if np.any(positions[anchor.target] != 0):
                        raise NotImplementedError(
                            f"Cable {cable.uid!r} attachment to {anchor.target!r} "
                            "with nonzero init_qpos requires a DexSim authoring "
                            "API that applies initial joint poses before attachment "
                            "construction; the current model-initialization hook "
                            "runs too late. No scene has been committed."
                        )
        scene = manager._spawn_scene
        registered = self._initial_joint_state_targets
        for uid, values in positions.items():
            if not values.size:
                continue
            for arena_name in scene.arena_names:
                target = f"{arena_name}/{uid}"
                if target not in registered:
                    scene.builder.add_runtime_control(
                        _InitialJointState(target, values)
                    )
                    registered.add(target)

    def _reset_scene_state(
        self, env_ids: Sequence[int] | None, excluded_uids: set[str]
    ) -> list[int] | None:
        """Restore cable worlds through the public engine episode boundary."""
        manager = self._manager
        cables = getattr(manager, "_cable_objects", {})
        if not cables:
            return None
        physical_uids = (
            set(manager._robots)
            | set(manager._articulations)
            | set(manager._rigid_objects)
            | set(manager._rigid_object_groups)
            | set(manager._deformable_objects)
            | set(cables)
        )
        if physical_uids.intersection(excluded_uids):
            raise NotImplementedError(
                "Cable episode reset requires every physical asset in selected "
                "environments; no objects have been reset."
            )
        cable = next(iter(cables.values()))
        cable._require_bound()
        ids = cable._resolve_env_ids(env_ids)
        if ids:
            result = manager.spawn_result
            if result is None:
                raise RuntimeError("Cable reset requires a prepared Spawn scene.")
            arenas = [
                result.arenas.index(manager._spawn_scene.arena_names[i]) for i in ids
            ]
            result.reset_arenas(arenas)
        return ids

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
        backend.sync_to_dexsim(result.world)

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
    def supports_cables(self) -> bool:
        """Newton supports cables with explicit DexUni on CUDA."""
        return True

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
