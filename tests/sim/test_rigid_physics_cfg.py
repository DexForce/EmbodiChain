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
"""Tests for the grouped rigid-body physics configuration boundary."""

from __future__ import annotations

import numpy as np
import pytest

import embodichain.lab.sim as sim
import embodichain.lab.sim.cfg as sim_cfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)


def test_public_cfg_facade_no_longer_exports_flat_rigid_attribute_types() -> None:
    for facade in (sim, sim_cfg):
        assert not hasattr(facade, "RigidBodyAttributesCfg")
        assert not hasattr(facade, "RigidBodyAttributesOverrideCfg")


def test_grouped_cfg_converts_com_quaternion_only_at_dexsim_boundary() -> None:
    input_quaternion_xyzw = [1.0, 2.0, 3.0, 4.0]
    cfg = RigidBodyPhysicsCfg.from_dict(
        {
            "mass_props": {
                "mass": 2.0,
                "inertia": [1.0, 2.0, 3.0],
                "com_position": [0.1, 0.2, 0.3],
                "com_quaternion": input_quaternion_xyzw,
            },
            "material_props": {"dynamic_friction": 0.4},
        }
    )

    native = cfg.to_dexsim_physical_attr()
    restored = RigidBodyPhysicsCfg.from_dexsim_physical_attr(native)

    np.testing.assert_allclose(native.com_quaternion, [4.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        restored.mass_props.com_quaternion,
        input_quaternion_xyzw,
    )
    assert restored.mass_props.mass == pytest.approx(2.0)
    assert restored.material_props.dynamic_friction == pytest.approx(0.4)


@pytest.mark.parametrize("config_type", [RigidObjectCfg, ArticulationCfg])
def test_asset_config_rejects_removed_flat_rigid_attributes(config_type: type) -> None:
    with pytest.raises(ValueError, match="Removed flat rigid-body attrs fields"):
        config_type.from_dict({"attrs": {"mass": 2.0}})


def test_grouped_attrs_parse_for_rigid_and_articulation_configs() -> None:
    rigid = RigidObjectCfg.from_dict({"attrs": {"mass_props": {"mass": 2.0}}})
    articulation = ArticulationCfg.from_dict(
        {"attrs": {"material_props": {"static_friction": 0.8}}}
    )

    assert rigid.attrs.mass_props.mass == pytest.approx(2.0)
    assert articulation.attrs.material_props.static_friction == pytest.approx(0.8)
