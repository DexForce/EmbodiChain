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

"""Typed deformable configuration and descriptor contracts."""

from __future__ import annotations

import pickle
from dataclasses import fields

import pytest

from embodichain.lab.sim import cfg as sim_cfg
from embodichain.lab.sim.cfg import (
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.spawn.descriptors import (
    surface_deformable_desc_from_cfg,
    volume_deformable_desc_from_cfg,
)

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("topology", ["volume", "surface"])
def test_new_schema_preserves_defaults_and_round_trips(topology):
    cls = (
        VolumeDeformableObjectCfg
        if topology == "volume"
        else SurfaceDeformableObjectCfg
    )
    cfg = cls.from_dict(
        {"uid": "body", "shape": {"shape_type": "Mesh", "fpath": "mesh.obj"}}
    )
    data = cfg.to_dict()
    assert "attrs" in data and "physical_attr" not in data
    assert "surface_props" in data["attrs"]
    assert {f.name for f in fields(cfg.attrs.surface_props)} == {
        "tri_ke",
        "tri_ka",
        "tri_kd",
        "tri_drag",
        "tri_lift",
        "edge_ke",
        "edge_kd",
    }
    expected = 0.0 if topology == "volume" else None
    assert all(v == expected for v in data["attrs"]["surface_props"].values())
    restored = cls.from_dict(data)
    assert restored.attrs.to_dict() == cfg.attrs.to_dict()
    assert restored.shape.fpath == "mesh.obj"
    copied = cfg.copy()
    copied.attrs.surface_props.tri_ke = 12.0
    assert cfg.attrs.surface_props.tri_ke == expected
    assert pickle.loads(pickle.dumps(cfg)).attrs.to_dict() == cfg.attrs.to_dict()


def test_volume_partial_surface_group_keeps_zero_descriptor_defaults():
    cfg = VolumeDeformableObjectCfg.from_dict(
        {
            "uid": "body",
            "shape": {"shape_type": "Mesh", "fpath": "mesh.obj"},
            "attrs": {"surface_props": {"tri_ke": 12.0}},
        }
    )
    desc, _ = volume_deformable_desc_from_cfg(cfg)
    assert desc.physics.surface_tri_ke == 12.0
    assert desc.physics.surface_edge_ke == 0.0


def test_unknown_nested_fields_are_rejected():
    with pytest.raises(TypeError, match="tri_kee"):
        SurfaceDeformableObjectCfg.from_dict(
            {"attrs": {"surface_props": {"tri_kee": 1.0}}}
        )


@pytest.mark.parametrize("topology", ["volume", "surface"])
def test_surface_coefficients_compile_without_changing_values(topology: str) -> None:
    cls = (
        VolumeDeformableObjectCfg
        if topology == "volume"
        else SurfaceDeformableObjectCfg
    )
    compile_cfg = (
        volume_deformable_desc_from_cfg
        if topology == "volume"
        else surface_deformable_desc_from_cfg
    )
    values = {
        "tri_ke": 12.0,
        "tri_ka": 13.0,
        "tri_kd": 14.0,
        "tri_drag": 15.0,
        "tri_lift": 16.0,
        "edge_ke": 17.0,
        "edge_kd": 18.0,
    }
    cfg = cls.from_dict(
        {
            "uid": "body",
            "shape": {"fpath": "mesh.obj"},
            "attrs": {"surface_props": values},
        }
    )
    desc, _ = compile_cfg(cfg)
    for field, value in values.items():
        native_field = f"surface_{field}" if topology == "volume" else field
        assert getattr(desc.physics, native_field) == value


def test_volume_elasticity_uses_short_names() -> None:
    cfg = VolumeDeformableObjectCfg.from_dict(
        {
            "uid": "body",
            "shape": {"fpath": "mesh.obj"},
            "attrs": {"youngs": 1000.0, "poissons": 0.3},
        }
    )
    desc, _ = volume_deformable_desc_from_cfg(cfg)
    assert desc.physics.k_mu == pytest.approx(1000.0 / (2.0 * 1.3))
    assert desc.physics.k_lambda == pytest.approx(1000.0 * 0.3 / (1.3 * 0.4))
    assert "youngs" in cfg.attrs.to_dict() and "poissons" in cfg.attrs.to_dict()


@pytest.mark.parametrize(
    "cfg_type, values",
    [
        (VolumeDeformableObjectCfg, {"physical_attr": {}}),
        (VolumeDeformableObjectCfg, {"voxel_attr": {}}),
        (sim_cfg.VolumeDeformablePhysicsCfg, {"youngs_modulus": 1.0}),
        (sim_cfg.VolumeDeformablePhysicsCfg, {"poissons_ratio": 0.3}),
        (sim_cfg.VolumeDeformablePhysicsCfg, {"surface_tri_ke": 1.0}),
        (sim_cfg.SurfaceDeformablePhysicsCfg, {"tri_ke": 1.0}),
    ],
)
def test_only_current_config_keywords_are_accepted(cfg_type, values) -> None:
    with pytest.raises(TypeError):
        cfg_type(**values)
