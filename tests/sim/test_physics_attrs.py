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
"""Headless unit tests for the backend-aware rigid-body attribute resolver.

No GPU / dexsim world required — these exercise the config layer and the
``physics_attrs`` resolver/warning logic in isolation.
"""

from __future__ import annotations

import logging

import pytest

from embodichain.lab.sim.cfg import (
    NewtonCollisionAttributesCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
)
from embodichain.lab.sim.physics_attrs import (
    NEWTON_CONTACT_SOLVER_FIELDS,
    ResolvedNewtonShape,
    resolve_newton_body,
    resolve_newton_shape,
    resolve_rigid_body_attributes,
    warn_backend_mismatched_fields,
    warn_ignored_contact_fields,
)


def test_from_dict_parses_nested_newton() -> None:
    cfg = RigidBodyAttributesCfg.from_dict(
        {"mass": 2.0, "restitution": 0.3, "newton": {"ke": 1e3, "margin": 0.01}}
    )
    assert cfg.mass == 2.0
    assert cfg.restitution == 0.3
    assert isinstance(cfg.newton, NewtonCollisionAttributesCfg)
    assert cfg.newton.ke == 1e3
    assert cfg.newton.margin == 0.01
    # unset newton fields stay None
    assert cfg.newton.kd is None


def test_override_from_dict_parses_nested_newton() -> None:
    ov = RigidBodyAttributesOverrideCfg.from_dict({"newton": {"kd": 50.0}})
    assert isinstance(ov.newton, NewtonCollisionAttributesCfg)
    assert ov.newton.kd == 50.0
    assert ov.newton.ke is None


def test_resolve_newton_shape_projects_common_fields() -> None:
    cfg = RigidBodyAttributesCfg(
        mass=2.0,
        dynamic_friction=0.4,
        restitution=0.2,
        enable_collision=False,
        density=800.0,
        newton=NewtonCollisionAttributesCfg(ke=1e3, margin=0.01),
    )
    shape = resolve_newton_shape(cfg)
    assert isinstance(shape, ResolvedNewtonShape)
    # common fields projected onto Newton ShapeConfig knobs
    assert shape.mu == 0.4  # dynamic_friction -> mu
    assert shape.restitution == 0.2
    assert shape.has_shape_collision is False  # enable_collision -> has_shape_collision
    assert shape.density == 800.0  # positive, so dexsim computes a positive body mass
    # newton-native sub-config fields copied verbatim
    assert shape.ke == 1e3
    assert shape.margin == 0.01
    # unset newton-native fields stay None
    assert shape.kd is None


def test_resolve_newton_shape_without_subconfig() -> None:
    cfg = RigidBodyAttributesCfg(dynamic_friction=0.5, restitution=0.1)
    shape = resolve_newton_shape(cfg)
    assert shape.mu == 0.5
    assert shape.restitution == 0.1
    assert shape.has_shape_collision is True  # default enable_collision
    assert shape.ke is None  # no newton sub-config


def test_resolve_newton_body_carries_mass_and_density() -> None:
    from dexsim.types import ActorType

    cfg = RigidBodyAttributesCfg(mass=2.0, density=800.0)
    body = resolve_newton_body(cfg, ActorType.DYNAMIC)
    assert body.actor_type == ActorType.DYNAMIC
    assert body.mass == 2.0
    assert body.density == 800.0


def test_resolve_rigid_body_attributes_dispatches_by_backend() -> None:
    cfg = RigidBodyAttributesCfg(mass=2.0, newton=NewtonCollisionAttributesCfg(ke=1e3))
    # default backend -> legacy PhysicalAttr
    pa = resolve_rigid_body_attributes(cfg, "default")
    assert pa.mass == 2.0
    # newton backend -> resolved shape
    shape = resolve_rigid_body_attributes(cfg, "newton", solver_type=None)
    assert isinstance(shape, ResolvedNewtonShape)
    assert shape.ke == 1e3


def test_merge_with_propagates_newton_via_merged_cfg() -> None:
    base = RigidBodyAttributesCfg(
        mass=1.0, newton=NewtonCollisionAttributesCfg(ke=1e3, margin=0.01)
    )
    override = RigidBodyAttributesOverrideCfg(
        mass=3.0, newton=NewtonCollisionAttributesCfg(kd=50.0)
    )
    merged = override.merged_cfg(base)
    # override wins for mass
    assert merged.mass == 3.0
    # newton sub-config: override non-None wins, else base
    assert merged.newton.ke == 1e3  # from base (override None)
    assert merged.newton.kd == 50.0  # from override
    assert merged.newton.margin == 0.01  # from base
    # legacy merge_with still returns a PhysicalAttr (drops newton)
    pa = override.merge_with(base)
    assert pa.mass == 3.0


def test_warn_ignored_contact_fields_xpbd(caplog) -> None:
    shape = ResolvedNewtonShape(ke=1e3, kd=50.0, mu=0.5, restitution=0.2)
    with caplog.at_level(logging.WARNING):
        warn_ignored_contact_fields(shape, "xpbd")
    # xpbd reads {mu, restitution, mu_torsional, mu_rolling}; ke/kd ignored
    msg = caplog.text
    assert "xpbd" in msg
    assert "ke" in msg and "kd" in msg


def test_warn_ignored_contact_fields_mujoco_warp_no_ke_kd_warning(
    caplog,
) -> None:
    shape = ResolvedNewtonShape(ke=1e3, kd=50.0, mu=0.5)
    with caplog.at_level(logging.WARNING):
        warn_ignored_contact_fields(shape, "mujoco_warp")
    # mujoco_warp reads {ke, kd, mu, kh, mu_torsional, mu_rolling}; ke/kd NOT ignored
    assert "ke" not in caplog.text or "ignores" not in caplog.text


def test_warn_ignored_contact_fields_restitution_on_mujoco_warp(
    caplog,
) -> None:
    # mujoco_warp does NOT read restitution -> should warn
    shape = ResolvedNewtonShape(restitution=0.3, mu=0.5)
    with caplog.at_level(logging.WARNING):
        warn_ignored_contact_fields(shape, "mujoco_warp")
    assert "restitution" in caplog.text


def test_warn_backend_mismatched_fields_newton(caplog) -> None:
    # PhysX-only fields deviating from defaults on Newton -> warn
    cfg = RigidBodyAttributesCfg(enable_ccd=True, linear_damping=0.9)
    with caplog.at_level(logging.WARNING):
        warn_backend_mismatched_fields(cfg, "newton")
    msg = caplog.text
    assert "enable_ccd" in msg
    assert "linear_damping" in msg


def test_warn_backend_mismatched_fields_no_warn_for_defaults(caplog) -> None:
    # all defaults -> no warning
    cfg = RigidBodyAttributesCfg()
    with caplog.at_level(logging.WARNING):
        warn_backend_mismatched_fields(cfg, "newton")
    assert caplog.text == ""


def test_warn_backend_mismatched_fields_no_warn_on_default(caplog) -> None:
    cfg = RigidBodyAttributesCfg(enable_ccd=True)
    with caplog.at_level(logging.WARNING):
        warn_backend_mismatched_fields(cfg, "default")
    assert caplog.text == ""


def test_newton_contact_solver_fields_table_sanity() -> None:
    # union of per-solver read sets == NEWTON_CONTACT_FIELDS
    from embodichain.lab.sim.physics_attrs import NEWTON_CONTACT_FIELDS

    union = set()
    for fields_set in NEWTON_CONTACT_SOLVER_FIELDS.values():
        union |= set(fields_set)
    assert union == set(NEWTON_CONTACT_FIELDS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
