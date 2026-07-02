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
"""Backend-aware resolution of rigid-body physical attributes.

This module is the EmbodiChain counterpart of dexsim's spawn-descriptor
resolver (``dexsim.spawn.adapters.newton_adapter``). It decouples the flat
:class:`~embodichain.lab.sim.cfg.RigidBodyAttributesCfg` (backend-neutral common
fields + an optional ``newton`` sub-config) from the backend-specific
descriptors dexsim consumes:

- On the **default** backend it returns the legacy
  :class:`dexsim.types.PhysicalAttr` (unchanged behaviour).
- On the **Newton** backend it builds a resolved Newton shape descriptor
  (carrying the backend-neutral ``mu``/``restitution``/``has_shape_collision``
  projected from common fields, plus the Newton-native sub-config fields) and a
  :class:`dexsim.spawn.descs.RigidBodyPhysicsDesc` body descriptor, suitable for
  dexsim's desc-native ``register_mesh_object_to_newton_patch`` entry point.

It also emits data-driven warnings (ported from dexsim) when a user sets contact
fields the active Newton solver ignores, or PhysX-only fields on the Newton
backend.

.. note::
    Newton-native contact/shape params (``ke``/``kd``/``margin``/...) are
    **build-time only**: there is no runtime batch API to mutate them. Runtime
    mutation (``RigidObject.set_attrs``) still applies the supported live subset
    (mass/friction/restitution/contact_offset).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from dexsim.spawn.descs import (
    NEWTON_CONTACT_FIELDS,
    NEWTON_CONTACT_SOLVER_FIELDS,
    NewtonCollisionDesc,
    RigidBodyPhysicsDesc,
)

from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.types import ActorType, PhysicalAttr

__all__ = [
    "NEWTON_CONTACT_FIELDS",
    "NEWTON_CONTACT_SOLVER_FIELDS",
    "ResolvedNewtonShape",
    "resolve_newton_shape",
    "resolve_newton_body",
    "resolve_rigid_body_attributes",
    "warn_ignored_contact_fields",
    "warn_backend_mismatched_fields",
]


# PhysX-only fields (carried on RigidBodyAttributesCfg) that Newton does not
# model per body. Setting them on the Newton backend is a no-op; warn so users
# notice. `static_friction` is folded into Newton's single `mu`; `rest_offset`
# has no Newton per-shape runtime equivalent (only `contact_offset`/`gap`).
_NEWTON_IGNORED_FIELDS: tuple[str, ...] = (
    "angular_damping",
    "linear_damping",
    "sleep_threshold",
    "enable_ccd",
    "max_depenetration_velocity",
    "min_position_iters",
    "min_velocity_iters",
    "max_linear_velocity",
    "max_angular_velocity",
    "rest_offset",
    "static_friction",
)


@dataclass
class ResolvedNewtonShape(NewtonCollisionDesc):
    """Newton shape descriptor after common-field projection.

    Mirrors dexsim's internal ``_ResolvedNewtonCollisionDesc``: a
    :class:`dexsim.spawn.descs.NewtonCollisionDesc` extended with the four
    ``newton.ModelBuilder.ShapeConfig`` knobs whose values are *projected* from
    backend-neutral common fields rather than read from the Newton sub-config.

    Field names mirror ``ShapeConfig`` attributes so dexsim's
    ``_newton_shape_cfg_from_desc`` overlays them by name.
    """

    density: float | None = None
    mu: float | None = None
    restitution: float | None = None
    has_shape_collision: bool | None = None


def resolve_newton_shape(cfg_attrs: RigidBodyAttributesCfg) -> ResolvedNewtonShape:
    """Project a :class:`RigidBodyAttributesCfg` onto a Newton shape descriptor.

    Backend-neutral common fields map to the four projected ``ShapeConfig``
    knobs (``dynamic_friction``→``mu``, ``restitution``, ``enable_collision``→
    ``has_shape_collision``, ``density``); Newton-native sub-config fields are
    copied verbatim. ``density`` is always set (positive) so dexsim can compute
    a positive body mass from shape density even when only ``mass`` (no
    explicit inertia) is given.

    Args:
        cfg_attrs: The rigid-body attribute config (with optional ``newton``).

    Returns:
        The resolved Newton shape descriptor.
    """
    newton_cfg = cfg_attrs.newton
    data: dict[str, Any] = {}
    if newton_cfg is not None:
        for f in fields(NewtonCollisionDesc):
            val = getattr(newton_cfg, f.name)
            if val is not None:
                data[f.name] = val
    return ResolvedNewtonShape(
        **data,
        density=cfg_attrs.density,
        mu=cfg_attrs.dynamic_friction,
        restitution=cfg_attrs.restitution,
        has_shape_collision=cfg_attrs.enable_collision,
    )


def resolve_newton_body(
    cfg_attrs: RigidBodyAttributesCfg, actor_type: "ActorType"
) -> RigidBodyPhysicsDesc:
    """Build a :class:`RigidBodyPhysicsDesc` body descriptor from common fields.

    dexsim reads ``mass``/``inertia``/``com_position``/``com_quaternion``
    duck-typed from the body descriptor (``actor_type`` is passed separately to
    the registration). Inertia is forwarded only if set on the cfg; otherwise
    dexsim derives it from shape density.

    Args:
        cfg_attrs: The rigid-body attribute config.
        actor_type: The dexsim :class:`ActorType` for this body.

    Returns:
        The body descriptor.
    """
    kwargs: dict[str, Any] = {"mass": cfg_attrs.mass}
    if cfg_attrs.density is not None:
        kwargs["density"] = cfg_attrs.density
    # Inertia / COM are not exposed on RigidBodyAttributesCfg today; if a future
    # config extension adds them, forward them here. Kept explicit for clarity.
    return RigidBodyPhysicsDesc(actor_type=actor_type, **kwargs)


def resolve_rigid_body_attributes(
    cfg_attrs: RigidBodyAttributesCfg,
    backend: str,
    solver_type: str | None = None,
) -> "PhysicalAttr | ResolvedNewtonShape":
    """Resolve a config into the backend-specific descriptor.

    For the Newton backend this returns the resolved Newton shape descriptor
    (and emits per-solver / backend-mismatch warnings); the caller builds the
    body descriptor separately via :func:`resolve_newton_body` since it owns the
    ``actor_type``.

    Args:
        cfg_attrs: The rigid-body attribute config.
        backend: ``"default"`` or ``"newton"``.
        solver_type: Active Newton solver type (e.g. ``"mujoco_warp"``); only
            consulted on the Newton backend for contact-field warnings. May be
            ``None`` to skip the per-solver warning.

    Returns:
        A :class:`dexsim.types.PhysicalAttr` for the default backend, or a
        :class:`ResolvedNewtonShape` for the Newton backend.
    """
    if backend == "newton":
        shape = resolve_newton_shape(cfg_attrs)
        if solver_type is not None:
            warn_ignored_contact_fields(shape, solver_type)
        warn_backend_mismatched_fields(cfg_attrs, backend)
        return shape
    return cfg_attrs.attr()


def warn_ignored_contact_fields(
    newton_shape: NewtonCollisionDesc | ResolvedNewtonShape | None,
    solver_type: str,
) -> None:
    """Warn for contact-material fields the active Newton solver does not read.

    Ported from dexsim's ``_warn_ignored_contact_fields``. A field the user set
    (non-None) that is a contact-material field but not in the active solver's
    read set is a harmless no-op; this makes it visible.
    """
    if newton_shape is None:
        return
    read_fields = NEWTON_CONTACT_SOLVER_FIELDS.get(solver_type)
    if read_fields is None:
        return
    ignored = sorted(
        f.name
        for f in fields(newton_shape)
        if getattr(newton_shape, f.name) is not None
        and f.name in NEWTON_CONTACT_FIELDS
        and f.name not in read_fields
    )
    if ignored:
        logger.log_warning(
            f"Newton solver '{solver_type}' ignores contact field(s) {ignored}; "
            "they have no effect for this solver."
        )


def warn_backend_mismatched_fields(
    cfg_attrs: RigidBodyAttributesCfg, backend: str
) -> None:
    """Warn for attribute fields the active backend does not model.

    On the Newton backend, PhysX-only per-body fields (damping, ccd, sleep
    thresholds, solver iters, rest_offset, static_friction) are not modelled;
    setting them is a no-op. The warning fires only when the user deviated from
    the cfg defaults, so it does not spam the common case.
    """
    if backend != "newton":
        return
    defaults = RigidBodyAttributesCfg()
    ignored = sorted(
        name
        for name in _NEWTON_IGNORED_FIELDS
        if getattr(cfg_attrs, name) != getattr(defaults, name)
    )
    if ignored:
        logger.log_warning(
            f"Newton backend does not model PhysX-only field(s) {ignored}; "
            "they have no runtime effect on Newton."
        )
