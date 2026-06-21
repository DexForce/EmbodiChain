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
"""Newton (Warp) physics backend.

Wraps DexSim's Newton module (``dexsim.engine.newton_physics``), which itself
runs NVIDIA Newton solvers (MuJoCo-Warp / XPBD / Featherstone / VBD /
semi-implicit) on Warp. The backend owns the lazy finalize/invalidate state
machine that rebuilds the Newton model whenever the scene is mutated.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from embodichain.utils import logger

from .base import PhysicsBackend

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import NewtonManager

    from embodichain.lab.sim.cfg import SimulationManagerCfg

__all__ = ["NewtonPhysicsBackend"]


class NewtonPhysicsBackend(PhysicsBackend):
    """The DexSim Newton physics backend (Warp-based)."""

    name = "newton"

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._newton_manager: "NewtonManager | None" = None
        self._is_finalized = False
        self._arenas_cloned = False

    # -- construction / world-config activation ------------------------- #
    def configure_world(self, world_config, sim_config: "SimulationManagerCfg") -> None:
        importlib.import_module("dexsim.engine.newton_physics")

        newton_physics_cfg = sim_config.physics_cfg
        world_config.newton_cfg = newton_physics_cfg.to_dexsim_cfg(
            gpu_id=sim_config.gpu_id,
        )

    def activate(self, sim_config: "SimulationManagerCfg") -> None:
        from dexsim.engine.newton_physics import get_newton_manager

        self._newton_manager = get_newton_manager(self._manager._world)

    # -- lifecycle ------------------------------------------------------ #
    def invalidate(self) -> None:
        """Mark the Newton scene as needing re-finalization after a mutation."""
        self._is_finalized = False
        self._arenas_cloned = False

    @property
    def is_initialized(self) -> bool:
        return self._is_finalized

    @property
    def newton_manager(self) -> "NewtonManager | None":
        if self._newton_manager is None:
            from dexsim.engine.newton_physics import get_newton_manager

            self._newton_manager = get_newton_manager(self._manager._world)
        return self._newton_manager

    def _lifecycle_state(self) -> str:
        """Return the Newton manager lifecycle state name, or empty string."""
        mgr = self.newton_manager
        return getattr(getattr(mgr, "lifecycle_state", None), "name", "")

    def _reset_entities_after_finalize(self) -> None:
        """Apply deferred initial resets once Newton runtime data is ready."""
        for rigid_obj in self._manager._rigid_objects.values():
            rigid_obj.reset()
        for articulation in self._manager._articulations.values():
            articulation.reset()
        for robot in self._manager._robots.values():
            robot.reset()
        # Rigid object groups are not supported on the Newton backend yet.

    def prepare(self) -> None:
        """Finalize the Newton scene if it has not been finalized yet.

        Implements the unified :meth:`PhysicsBackend.prepare` contract: this is
        both the "finalize" entry point (public
        :meth:`SimulationManager.finalize_newton_physics`) and the "GPU init"
        entry point (:meth:`SimulationManager.init_gpu_physics`) for the Newton
        backend, since Newton's notion of becoming ready to step is finalizing
        the model.
        """
        if self._is_finalized and self._lifecycle_state() == "READY":
            return

        mgr = self.newton_manager
        state = self._lifecycle_state()

        if state != "READY":
            from dexsim.engine.newton_physics.rebuild import (
                ensure_simulation_prepared_lazy,
                rebuild_newton_from_scene,
            )

            safe_to_continue, _ = ensure_simulation_prepared_lazy(
                mgr,
                self._manager._world,
                rebuild_from_scene=rebuild_newton_from_scene,
                warn=True,
            )
            if not safe_to_continue:
                logger.log_error(
                    "Failed to finalize Newton physics: model is not ready to build "
                    f"(lifecycle state {state!r})."
                )
                return

        state = self._lifecycle_state()
        if state != "READY":
            logger.log_error(
                "Failed to finalize Newton physics: lifecycle state is "
                f"{state!r} after simulation preparation."
            )

        self._is_finalized = True
        self._reset_entities_after_finalize()

    def ensure_initialized(self) -> None:
        self.prepare()

    # -- scene ---------------------------------------------------------- #
    def get_scene(self):
        return self.newton_manager.scene

    # -- capabilities --------------------------------------------------- #
    @property
    def supports_robot(self) -> bool:
        # Robots are URDF articulations; the Newton ``load_urdf`` patch builds a
        # NewtonArticulation, and the shared spawn path (add_robot invalidate +
        # _reset_entities_after_finalize) handles the Newton lifecycle. Requires
        # the dexsim fix to ``NewtonArticulation._joint_metas_from_ids`` so that
        # explicit joint_ids use active-joint indexing (matching get_dof()).
        return True

    @property
    def can_disable_manual_update(self) -> bool:
        # Newton cannot switch between manual and automatic update.
        return False
