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
"""Swappable physics-backend abstraction for :class:`SimulationManager`.

This module defines the contract that every physics backend (DexSim default,
Newton/Warp, ...) satisfies. The owning :class:`SimulationManager`
holds a single :class:`PhysicsBackend` instance as ``self.physics`` and
delegates the backend-specific lifecycle, scene access, world-config
activation and capability queries to it, instead of branching on a backend
name string throughout the manager.

The design deliberately mirrors IsaacLab's split of an orchestrator
(``SimulationContext``) from a swappable physics manager (``PhysicsManager``),
with one departure: EmbodiChain keeps the backend as a true *instance* member
rather than a class-singleton, because :class:`SimulationManager` is itself a
multiton (one instance per ``instance_id``) and a class-singleton backend
would break that.

.. note::
    This ABC covers the *manager-level* backend surface (lifecycle, scene,
    capabilities, world-config). The per-asset read/write contract lives in
    :mod:`embodichain.lab.sim.objects.backends` (``RigidBodyViewBase`` /
    ``ArticulationViewBase``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dexsim

    from embodichain.lab.sim.cfg import SimulationManagerCfg
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["PhysicsBackend"]


class PhysicsBackend(ABC):
    """Abstract base class for a swappable physics backend.

    A backend is constructed with a back-reference to its owning
    :class:`SimulationManager` (from which it reaches the dexsim world, the
    resolved device, the asset registries and the physics config). All
    backend-specific behaviour is expressed as overrides of the methods and
    properties below; the manager never inspects ``self.physics.name`` to
    decide what to do (it only exposes it for backwards-compatible public
    properties).
    """

    #: Backend identifier, e.g. ``"default"`` or ``"newton"``.
    name: str = ""

    def __init__(self, manager: "SimulationManager") -> None:
        self._manager: "SimulationManager" = manager

    # ------------------------------------------------------------------ #
    # Construction / world-config activation
    # ------------------------------------------------------------------ #
    @abstractmethod
    def configure_world(
        self,
        world_config: "dexsim.WorldConfig",
        sim_config: "SimulationManagerCfg",
    ) -> None:
        """Apply backend-specific fields to the dexsim ``WorldConfig``.

        Called from :meth:`SimulationManager._convert_sim_config` after the
        shared world-config fields and the resolved device have been set, so
        implementations may read ``self._manager.device``.

        Args:
            world_config: The dexsim world config to mutate in place.
            sim_config: The full simulation manager config.
        """

    @abstractmethod
    def activate(self, sim_config: "SimulationManagerCfg") -> None:
        """Perform backend setup immediately after the dexsim World is created.

        This is the counterpart of the backend split that used to live in
        ``SimulationManager.__init__`` (default ``set_physics_config`` vs
        ``get_newton_manager``).
        """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @abstractmethod
    def ensure_initialized(self) -> None:
        """Ensure the backend runtime is ready before a physics step.

        Called at the top of :meth:`SimulationManager.update`. For the default
        backend this lazy-initializes GPU physics; for Newton it finalizes the
        scene (rebuilding if the scene was mutated). Idempotent.
        """

    @abstractmethod
    def invalidate(self) -> None:
        """Mark the backend scene as needing re-initialization.

        Called after any scene mutation (adding/removing assets) so that the
        next :meth:`ensure_initialized` rebuilds as needed. A no-op for
        backends without a dirty/finalize lifecycle.
        """

    @abstractmethod
    def prepare(self) -> None:
        """Force the backend into a ready-to-step state.

        This unifies what the legacy code exposed as two separate operations -
        "GPU physics init" on the default backend and "Newton finalize" - into a
        single backend-agnostic entry point. It is idempotent: a backend that is
        already ready is a no-op, and after :meth:`invalidate` the next call
        re-prepares (re-initializes GPU physics / re-finalizes the Newton scene)
        as needed.

        Called both lazily by :meth:`ensure_initialized` before each step and
        directly by the public :meth:`SimulationManager.init_gpu_physics` and
        :meth:`SimulationManager.finalize_newton_physics` entry points (both of
        which delegate here).
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the backend runtime has been initialized/finalized."""

    # ------------------------------------------------------------------ #
    # Scene access
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_scene(self):
        """Return the active physics scene object (default DexSim or Newton)."""

    @property
    def newton_manager(self):
        """The DexSim Newton manager, or ``None`` if not the Newton backend.

        Returns:
            The :class:`dexsim.engine.newton_physics.NewtonManager` for the
            Newton backend, otherwise ``None``.
        """
        return None

    # ------------------------------------------------------------------ #
    # Capabilities (override in subclasses; defaults are conservative)
    # ------------------------------------------------------------------ #
    @property
    def supports_soft_bodies(self) -> bool:
        """Whether this backend can simulate soft bodies."""
        return False

    @property
    def supports_cloth(self) -> bool:
        """Whether this backend can simulate cloth bodies."""
        return False

    @property
    def supports_rigid_object_group(self) -> bool:
        """Whether this backend supports rigid object groups."""
        return False

    @property
    def supports_robot(self) -> bool:
        """Whether this backend supports robots (articulated URDF assets)."""
        return False

    @property
    def can_disable_manual_update(self) -> bool:
        """Whether ``set_manual_update(False)`` is permitted on this backend."""
        return True
