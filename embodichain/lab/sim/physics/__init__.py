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
"""Physics backend registry and factory.

Selects a concrete :class:`PhysicsBackend` from a physics config via
:func:`embodichain.lab.sim.cfg.physics_backend_from_cfg` and instantiates it
with the owning :class:`SimulationManager` as its back-reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from embodichain.lab.sim.cfg import physics_backend_from_cfg
from embodichain.utils import logger

from .base import PhysicsBackend
from .default import DefaultPhysicsBackend
from .newton import NewtonPhysicsBackend

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "PhysicsBackend",
    "DefaultPhysicsBackend",
    "NewtonPhysicsBackend",
    "make_physics_backend",
]

#: Registry of backend name -> backend class.
_BACKENDS: dict[str, type[PhysicsBackend]] = {
    "default": DefaultPhysicsBackend,
    "newton": NewtonPhysicsBackend,
}


def make_physics_backend(physics_cfg, manager: "SimulationManager") -> PhysicsBackend:
    """Construct the physics backend for ``physics_cfg``.

    The backend subclass is selected by the *type* of ``physics_cfg``
    (via :func:`physics_backend_from_cfg`), so passing a
    :class:`~embodichain.lab.sim.cfg.NewtonPhysicsCfg` activates the Newton
    backend and a
    :class:`~embodichain.lab.sim.cfg.DefaultPhysicsCfg` activates the default
    backend.

    Args:
        physics_cfg: The physics backend configuration.
        manager: The owning :class:`SimulationManager` (passed as the
            backend's back-reference).

    Returns:
        The instantiated :class:`PhysicsBackend`.
    """
    name = physics_backend_from_cfg(physics_cfg)
    cls = _BACKENDS.get(name)
    if cls is None:
        logger.log_error(f"Unknown physics backend: {name!r}.")
    return cls(manager)
