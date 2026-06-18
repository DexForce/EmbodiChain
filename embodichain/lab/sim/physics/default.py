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
"""DexSim default physics backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dexsim

from embodichain.lab.sim.cfg import DefaultPhysicsCfg
from embodichain.utils import logger

from .base import PhysicsBackend

if TYPE_CHECKING:
    import dexsim as _dexsim  # noqa: F401

    from embodichain.lab.sim.cfg import SimulationManagerCfg

__all__ = ["DefaultPhysicsBackend"]


class DefaultPhysicsBackend(PhysicsBackend):
    """The legacy DexSim default physics backend (GPU or CPU)."""

    name = "default"

    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._is_initialized_gpu_physics = False

    # -- construction / world-config activation ------------------------- #
    def configure_world(self, world_config, sim_config: "SimulationManagerCfg") -> None:
        cfg = sim_config.physics_cfg
        assert isinstance(cfg, DefaultPhysicsCfg)
        world_config.length_tolerance = cfg.length_tolerance
        world_config.speed_tolerance = cfg.speed_tolerance
        if self._manager.device.type == "cuda":
            world_config.enable_gpu_sim = True
            world_config.direct_gpu_api = True

    def activate(self, sim_config: "SimulationManagerCfg") -> None:
        cfg = sim_config.physics_cfg
        assert isinstance(cfg, DefaultPhysicsCfg)
        dexsim.set_physics_config(**cfg.to_dexsim_args())
        dexsim.set_physics_gpu_memory_config(**cfg.gpu_memory.to_dict())

    # -- lifecycle ------------------------------------------------------ #
    def invalidate(self) -> None:
        # The default backend has no dirty/finalize lifecycle.
        pass

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized_gpu_physics

    def prepare(self) -> None:
        """Initialize GPU physics for the default backend.

        Implements the unified :meth:`PhysicsBackend.prepare` contract. For the
        default backend "becoming ready to step" is initializing GPU physics; on
        CPU there is nothing to initialize so this is a no-op.
        """
        if not self._manager.is_use_gpu_physics:
            logger.log_warning(
                "The simulation device is not cuda, cannot initialize GPU physics."
            )
            return

        if self._is_initialized_gpu_physics:
            return

        for art in self._manager._articulations.values():
            art.reallocate_body_data()
        for robot in self._manager._robots.values():
            robot.reallocate_body_data()

        # Re-establish rigid object positions after articulation resets, ensuring
        # no articulation kinematics step has inadvertently corrupted the broadphase
        # state for rigid bodies.
        for rigid_obj in self._manager._rigid_objects.values():
            rigid_obj.reset()

        self._is_initialized_gpu_physics = True

    def ensure_initialized(self) -> None:
        if self._manager.is_use_gpu_physics and not self._is_initialized_gpu_physics:
            logger.log_warning(
                "Using GPU physics, but not initialized yet. Forcing initialization."
            )
            self.prepare()

    # -- scene ---------------------------------------------------------- #
    def get_scene(self):
        return self._manager._world.get_physics_scene()

    # -- capabilities --------------------------------------------------- #
    # The default backend supports soft/cloth on GPU; the GPU
    # precondition itself is enforced separately in SimulationManager.
    @property
    def supports_soft_bodies(self) -> bool:
        return True

    @property
    def supports_cloth(self) -> bool:
        return True

    @property
    def supports_rigid_object_group(self) -> bool:
        return True

    @property
    def supports_robot(self) -> bool:
        return True
