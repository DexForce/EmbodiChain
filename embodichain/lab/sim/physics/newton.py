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

from .base import PhysicsBackend

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import SimulationManagerCfg

__all__ = ["NewtonPhysicsBackend"]


class NewtonPhysicsBackend(PhysicsBackend):
    """The DexSim Newton physics backend (Warp-based)."""

    name = "newton"

    #: Resolved Newton solver type after world configuration.
    solver_type: str | None = None

    # -- construction / world-config activation ------------------------- #
    def configure_world(self, world_config, sim_config: "SimulationManagerCfg") -> None:
        importlib.import_module("dexsim.engine.newton_physics")

        newton_physics_cfg = sim_config.physics_cfg
        newton_cfg = newton_physics_cfg.to_dexsim_cfg(
            gpu_id=sim_config.gpu_id,
        )
        self.solver_type = newton_cfg.solver_cfg.solver_type
        world_config.newton_cfg = newton_cfg

    def activate(self, sim_config: "SimulationManagerCfg") -> None:
        del sim_config
        # WorldConfig.newton_cfg registers the World-owned NewtonBackend.
        # SceneBuilder.finalize() completes its model; no second manager-level
        # activation or rebuild domain participates.

    @property
    def newton_manager(self):
        """Reject access to the removed, independently owned Newton manager."""
        raise RuntimeError(
            "NewtonManager is not part of Spawn scene ownership. Use "
            "SimulationManager.spawn_result and its Spawned*/Batch APIs."
        )

    # -- scene ---------------------------------------------------------- #
    def get_scene(self):
        raise RuntimeError(
            "Newton Spawn scenes do not expose a PhysicsScene. Use "
            "SimulationManager.spawn_result and its Spawned*/Batch APIs."
        )

    # -- capabilities --------------------------------------------------- #
    @property
    def supports_volume_deformables(self) -> bool:
        # Reserved entry point: add a Newton volume adapter before enabling.
        return False

    @property
    def supports_surface_deformables(self) -> bool:
        # Reserved entry point: add a Newton surface adapter before enabling.
        return False

    @property
    def supports_robot(self) -> bool:
        # Robots are SpawnedArticulations in the World-owned Newton model.
        return True

    @property
    def supports_rigid_object_group(self) -> bool:
        # Groups are env-major views over the Spawn rigid-body batch, which
        # provides the same state and mass-property API on Newton.
        return True

    @property
    def can_disable_manual_update(self) -> bool:
        # Newton cannot switch between manual and automatic update.
        return False
