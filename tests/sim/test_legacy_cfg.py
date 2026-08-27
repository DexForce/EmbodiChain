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
"""Tests for the isolated, Default-backend-only physics compatibility layer."""

from __future__ import annotations

import numpy as np
import pytest

import embodichain.lab.sim.cfg as sim_cfg
from embodichain.lab.sim import _legacy_cfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)


def test_legacy_classes_are_reexported_from_public_cfg_module() -> None:
    assert RigidBodyAttributesCfg is _legacy_cfg.RigidBodyAttributesCfg
    assert RigidBodyAttributesOverrideCfg is _legacy_cfg.RigidBodyAttributesOverrideCfg
    assert RigidBodyAttributesCfg.__module__ == "embodichain.lab.sim._legacy_cfg"


def test_legacy_cfg_exposes_no_newton_compatibility_surface() -> None:
    assert not hasattr(sim_cfg, "NewtonCollisionAttributesCfg")
    assert not hasattr(RigidBodyAttributesCfg(), "newton")
    assert not hasattr(RigidBodyAttributesOverrideCfg(), "newton")


def test_legacy_cfg_projects_default_backend_physical_attr() -> None:
    cfg = RigidBodyAttributesCfg(
        mass=2.0,
        dynamic_friction=0.4,
        inertia=[1.0, 2.0, 3.0],
        com_position=[0.1, 0.2, 0.3],
    )

    attr = cfg.attr()

    assert attr.mass == 2.0
    assert attr.dynamic_friction == pytest.approx(0.4)
    np.testing.assert_array_equal(attr.inertia, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(attr.com_position, [0.1, 0.2, 0.3])


def test_legacy_override_merges_only_configured_values() -> None:
    base = RigidBodyAttributesCfg(mass=1.0, dynamic_friction=0.4)
    override = RigidBodyAttributesOverrideCfg(mass=3.0)

    merged = override.merged_cfg(base)

    assert merged.mass == 3.0
    assert merged.dynamic_friction == 0.4
    assert override.merge_with(base).mass == 3.0


@pytest.mark.parametrize(
    "config_type",
    [RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg],
)
def test_legacy_cfg_rejects_removed_newton_subconfig(config_type: type) -> None:
    with pytest.raises(ValueError, match="newton"):
        config_type.from_dict({"newton": {"margin": 0.01}})


def test_asset_cfg_parsers_distinguish_grouped_and_legacy_attrs() -> None:
    grouped = RigidObjectCfg.from_dict({"attrs": {"mass_props": {"mass": 2.0}}})
    legacy = ArticulationCfg.from_dict({"attrs": {"mass": 2.0}})

    assert isinstance(grouped.attrs, RigidBodyPhysicsCfg)
    assert grouped.attrs.mass_props.mass == 2.0
    assert isinstance(legacy.attrs, RigidBodyAttributesCfg)
    assert legacy.attrs.mass == 2.0


def test_asset_cfg_parser_rejects_mixed_physics_schemas() -> None:
    with pytest.raises(ValueError, match="Do not mix"):
        RigidObjectCfg.from_dict(
            {"attrs": {"mass_props": {"mass": 2.0}, "density": 500.0}}
        )
