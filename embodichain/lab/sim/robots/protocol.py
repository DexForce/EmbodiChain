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

"""RobotDef protocol for EmbodiChain robot definitions.

Every robot in the framework should implement the :class:`RobotDef` protocol.
The :meth:`build_cfg` method converts a ``RobotDef``-compatible object into a
:class:`~embodichain.lab.sim.cfg.RobotCfg` which
:meth:`~embodichain.lab.sim.SimulationManager.add_robot` expects.
"""

from __future__ import annotations

import torch
from typing import Dict, TYPE_CHECKING, runtime_checkable, Protocol

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg

if TYPE_CHECKING:
    pass

__all__ = ["RobotDef"]

# Fields that may exist on both the protocol implementer and RobotCfg and
# should be copied across during ``build_cfg``.
_EXTRA_CFG_FIELDS: tuple[str, ...] = (
    "min_position_iters",
    "min_velocity_iters",
    "fix_base",
    "disable_self_collision",
    "build_pk_chain",
    "init_qpos",
    "body_scale",
)


@runtime_checkable
class RobotDef(Protocol):
    """Structural protocol that every robot definition must satisfy.

    Properties:
        name: Unique identifier string for this robot.
        urdf_cfg: URDF assembly configuration.
        control_parts: Mapping from part name to joint name lists.
        solver_cfg: Solver configuration (single or per-part).
        drive_pros: Joint drive properties.
        attrs: Rigid-body physics attributes.

    Methods:
        build_pk_serial_chain: Build pytorch-kinematics serial chains.
        build_cfg: Convert this definition into a :class:`RobotCfg`.
    """

    # -- required properties --------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this robot definition."""
        ...

    @property
    def urdf_cfg(self) -> URDFCfg | None:
        """URDF assembly configuration."""
        ...

    @property
    def control_parts(self) -> dict[str, list[str]] | None:
        """Mapping from part name to joint name lists."""
        ...

    @property
    def solver_cfg(self) -> SolverCfg | dict[str, SolverCfg] | None:
        """Solver configuration for IK/FK computation."""
        ...

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg:
        """Joint drive properties (stiffness, damping, etc.)."""
        ...

    @property
    def attrs(self) -> RigidBodyAttributesCfg:
        """Rigid-body physics attributes."""
        ...

    # -- required methods -----------------------------------------------------

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu")
    ) -> Dict[str, object]:
        """Build pytorch-kinematics serial chains for each control part.

        Args:
            device: Torch device to place chains on.

        Returns:
            Dictionary mapping part name to serial chain objects.
        """
        ...

    def build_cfg(self, **overrides: object) -> RobotCfg:
        """Convert this robot definition into a :class:`RobotCfg`.

        The method creates a new :class:`RobotCfg`, copies protocol
        properties and any extra fields that exist on both *self* and
        the config, optionally overrides ``build_pk_serial_chain``, and
        finally applies any remaining keyword overrides via
        :func:`~embodichain.lab.sim.utility.cfg_utils.merge_robot_cfg`.

        Args:
            **overrides: Optional overrides applied on top of the defaults.
                ``uid`` is consumed first to set the config uid.

        Returns:
            A fully-populated :class:`RobotCfg` ready for
            :meth:`~embodichain.lab.sim.SimulationManager.add_robot`.
        """
        ...


# ---------------------------------------------------------------------------
# Mixin-style default implementation so concrete classes can simply inherit
# and call ``super().build_cfg(**overrides)``.
# ---------------------------------------------------------------------------


class _RobotDefMixin:
    """Mixin that provides the default ``build_cfg`` implementation.

    This is *not* part of the public API.  Concrete robot classes inherit
    from this mixin (and satisfy the ``RobotDef`` protocol via duck-typing)
    so they only need to declare the required properties.
    """

    def build_cfg(self, **overrides: object) -> RobotCfg:
        """Build a :class:`RobotCfg` from this robot definition.

        See :meth:`RobotDef.build_cfg` for full documentation.
        """
        cfg = RobotCfg()

        # (a) Set uid from overrides or fall back to the protocol name.
        cfg.uid = overrides.pop("uid", self.name)

        # (b) Copy core protocol properties.
        cfg.urdf_cfg = self.urdf_cfg
        cfg.control_parts = self.control_parts
        cfg.solver_cfg = self.solver_cfg
        cfg.drive_pros = self.drive_pros
        cfg.attrs = self.attrs

        # (c) Copy extra scalar/flag fields that exist on both self and
        #     RobotCfg.
        for field_name in _EXTRA_CFG_FIELDS:
            if hasattr(self, field_name) and hasattr(cfg, field_name):
                setattr(cfg, field_name, getattr(self, field_name))

        # (d) Override build_pk_serial_chain on the cfg instance if self
        #     provides one.
        if hasattr(self, "build_pk_serial_chain"):
            cfg.build_pk_serial_chain = self.build_pk_serial_chain  # type: ignore[assignment]

        # (e) Merge remaining overrides.
        if overrides:
            cfg = merge_robot_cfg(cfg, overrides)

        return cfg


# Patch the mixin onto the protocol so callers can do RobotDef.build_cfg(self, ...)
# or simply inherit from the mixin for the default behaviour.
RobotDef.build_cfg = _RobotDefMixin.build_cfg  # type: ignore[assignment]
