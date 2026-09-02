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
"""Default physics backend implementation integrated through DexSim."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dexsim

from embodichain.lab.sim.cfg import DefaultPhysicsCfg

from .base import PhysicsBackend

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

__all__ = ["DefaultPhysicsBackend"]


class DefaultPhysicsBackend(PhysicsBackend):
    """Default backend using DexSim's native GPU or CPU physics path."""

    name = "default"

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

    def prepare_spawn_runtime(self, result: "dexsim.scene.Scene") -> None:
        """Initialize Direct GPU buffers for a committed CUDA topology."""
        del result
        if self._manager.device.type == "cuda":
            self._manager._world.init_gpu_physics()

    # -- scene ---------------------------------------------------------- #
    def get_scene(self):
        """Return the Default backend's compatibility scene after Spawn is prepared."""
        self._manager.prepare()
        return self._manager._world.get_physics_scene()

    # -- capabilities --------------------------------------------------- #
    # The default backend supports deformables on GPU; the GPU
    # precondition itself is enforced separately in SimulationManager.
    @property
    def supports_volume_deformables(self) -> bool:
        return True

    @property
    def supports_surface_deformables(self) -> bool:
        return True

    @property
    def supports_rigid_object_group(self) -> bool:
        return True

    @property
    def supports_robot(self) -> bool:
        return True

    @property
    def supports_rigid_constraints(self) -> bool:
        return True

    @property
    def supports_contact_sensor(self) -> bool:
        return True
