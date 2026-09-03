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
"""Spawn-aware physics-backend abstraction for :class:`SimulationManager`.

This module defines the contract that every physics backend (Default, Newton,
...) satisfies. The owning :class:`SimulationManager`
holds a single :class:`PhysicsBackend` instance as ``self.physics`` and
delegates backend-specific world configuration, compatibility scene access,
and capability queries to it. Scene topology and runtime readiness are owned
by DexSim's ``SceneBuilder`` and finalized ``Scene``.

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

    from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

__all__ = ["PhysicsBackend"]


class PhysicsBackend(ABC):
    """Abstract base class for a swappable physics backend.

    A backend is constructed with a back-reference to its owning
    :class:`SimulationManager` (from which it reaches the dexsim world, the
    resolved device, the asset registries and the physics config). All
    Manager-level backend behavior is expressed as overrides of the methods
    and properties below. The backend name remains available for diagnostics
    and backwards-compatible public predicates, but operational decisions use
    hooks and capability flags.
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

        Default configures the native DexSim globals. Newton is already
        registered from ``WorldConfig.newton_cfg`` and therefore has no
        additional activation work.
        """

    def prepare_spawn_runtime(self, result: "dexsim.scene.Scene") -> None:
        """Prepare runtime buffers for one committed Spawn topology revision.

        :class:`SimulationManager` calls this hook once per topology revision,
        after source configuration and before facade binding. Backends that do
        not need a separate runtime-preparation step keep the default no-op.

        Args:
            result: The committed Spawn result being prepared.
        """
        del result

    def sync_render_state(self, result: "dexsim.scene.Scene") -> None:
        """Publish the current physics state to render resources without stepping.

        Backends whose physics and render state share native storage require no
        work. Backends with a separate render bridge override this hook.

        Args:
            result: The finalized Spawn result whose state should be published.
        """
        del result

    def prepare_for_teardown(self) -> None:
        """Release backend-owned views before Spawn releases their parents.

        :class:`SimulationManager` calls this during deferred destruction,
        after render workers stop and before it closes the Spawn result. A
        backend can use this boundary to synchronize device work and release
        borrowed render or physics views while their World-owned native
        parents are still alive. Backends without such views keep the default
        no-op implementation.
        """

    # ------------------------------------------------------------------ #
    # Scene access
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_scene(self):
        """Return a backend compatibility scene, or raise if none exists."""

    @property
    def newton_manager(self):
        """Return ``None`` because Spawn does not use ``NewtonManager``.

        The Newton backend overrides this property with an actionable error so
        callers do not accidentally mix the removed manager ownership domain
        with the World-owned Spawn backend.
        """
        return None

    @property
    def differentiable_runtime(self):
        """Return no differentiable runtime for non-Newton backends."""
        return None

    @property
    def solver_type(self) -> str | None:
        """Return the configured or resolved backend solver type, if exposed."""
        return None

    # ------------------------------------------------------------------ #
    # Capabilities (override in subclasses; defaults are conservative)
    # ------------------------------------------------------------------ #
    @property
    def supports_volume_deformables(self) -> bool:
        """Whether this backend has a volume-deformable object adapter."""
        return False

    @property
    def supports_surface_deformables(self) -> bool:
        """Whether this backend has a surface-deformable object adapter."""
        return False

    @property
    def supports_soft_bodies(self) -> bool:
        """Compatibility alias for volume-deformable support."""
        return self.supports_volume_deformables

    @property
    def supports_cloth(self) -> bool:
        """Compatibility alias for surface-deformable support."""
        return self.supports_surface_deformables

    @property
    def supports_rigid_object_group(self) -> bool:
        """Whether this backend supports rigid object groups."""
        return False

    @property
    def supports_robot(self) -> bool:
        """Whether this backend supports robots (articulated URDF assets)."""
        return False

    @property
    def supports_rigid_constraints(self) -> bool:
        """Whether this backend supports native rigid constraints."""
        return False

    @property
    def supports_contact_sensor(self) -> bool:
        """Whether this backend supports the native contact sensor."""
        return False

    @property
    def can_disable_manual_update(self) -> bool:
        """Whether ``set_manual_update(False)`` is permitted on this backend."""
        return True
