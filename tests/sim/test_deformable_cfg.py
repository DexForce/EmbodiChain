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
from embodichain.lab.sim.shapes import MeshCfg

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


@pytest.mark.parametrize(
    "mode", ["isothermal", "adiabatic", "target_volume", "prescribed_gauge_pressure"]
)
def test_cloth_pneumatic_round_trip_and_descriptor(mode: str) -> None:
    """Keep cavity law and physical parameters through config serialization."""
    gas = {
        "mode": mode,
        "reference_absolute_pressure": 117325.0,
        "ambient_pressure": 101325.0,
        "target_volume": 2.4e-4,
        "volume_stiffness": 2000.0,
        "bulk_damping": 3e6,
        "max_absolute_pressure": 5e5,
        "heat_capacity_ratio": 1.4,
        "min_volume_ratio": 0.05,
        "prescribed_gauge_pressure": 16000.0,
    }
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {"uid": "pouch", "shape": {"fpath": "pouch.obj"}, "attrs": {"pneumatic": gas}}
    )
    restored = SurfaceDeformableObjectCfg.from_dict(cfg.to_dict())
    descriptor, _ = surface_deformable_desc_from_cfg(restored)
    assert restored.attrs.pneumatic == gas
    for name, value in gas.items():
        assert getattr(descriptor.physics.pneumatic, name) == value


def test_ordinary_cloth_does_not_require_pneumatic_support(monkeypatch) -> None:
    """Omitting a cavity remains compatible with wheels lacking its descriptor."""
    import dexsim.spawn

    monkeypatch.delattr(dexsim.spawn, "PneumaticPhysicsDesc")
    cfg = SurfaceDeformableObjectCfg(uid="cloth", shape=MeshCfg(fpath="cloth.obj"))
    descriptor, _ = surface_deformable_desc_from_cfg(cfg)
    assert descriptor.physics.pneumatic is None


def test_explicit_cavity_rejects_unsupported_wheel(monkeypatch) -> None:
    """Never silently discard requested gas physics on an older SDK."""
    import dexsim.spawn

    monkeypatch.delattr(dexsim.spawn, "PneumaticPhysicsDesc")
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {"uid": "pouch", "shape": {"fpath": "pouch.obj"}, "attrs": {"pneumatic": {}}}
    )
    with pytest.raises(NotImplementedError, match="PneumaticPhysicsDesc"):
        surface_deformable_desc_from_cfg(cfg)


@pytest.mark.parametrize(
    "gas, error",
    [
        ({"reference_absolute_pressure": float("nan")}, ValueError),
        ({"bulk_damping": -1.0}, ValueError),
        ({"mode": "unknown"}, ValueError),
        ({"reference_pressure_typo": 1.0}, TypeError),
        (True, TypeError),
    ],
)
def test_cloth_cavity_rejects_invalid_parameters(gas: object, error: type) -> None:
    """Validate gas parameters before committing native scene resources."""
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {"uid": "pouch", "shape": {"fpath": "pouch.obj"}, "attrs": {"pneumatic": gas}}
    )
    with pytest.raises(error):
        surface_deformable_desc_from_cfg(cfg)


@pytest.mark.parametrize("angle", [None, 0.0, 0.35, -0.4])
def test_cloth_rest_angle_round_trip_and_descriptor(angle: float | None) -> None:
    """Preserve inferred or authored signed rest angles through config lowering."""
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {
            "uid": "shirt",
            "shape": {"fpath": "shirt.obj"},
            "attrs": {"edge_rest_angle": angle},
        }
    )
    restored = SurfaceDeformableObjectCfg.from_dict(cfg.to_dict())
    assert restored.attrs.edge_rest_angle == angle
    descriptor, _ = surface_deformable_desc_from_cfg(restored)
    assert descriptor.physics.edge_rest_angle == angle


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), -float("inf")])
def test_cloth_rest_angle_rejects_nonfinite_values(angle: float) -> None:
    """Reject invalid rest geometry before creating native scene resources."""
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {
            "uid": "shirt",
            "shape": {"fpath": "shirt.obj"},
            "attrs": {"edge_rest_angle": angle},
        }
    )
    with pytest.raises(ValueError, match="edge_rest_angle.*finite"):
        surface_deformable_desc_from_cfg(cfg)


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


@pytest.mark.parametrize("topology", ["volume", "surface"])
@pytest.mark.usefixtures("requires_particle_contact_material")
def test_particle_contact_material_round_trip_and_lowering(topology) -> None:
    values = {"stiffness": 5e4, "damping": 0.05, "dynamic_friction": 0.7}
    cls = (
        VolumeDeformableObjectCfg
        if topology == "volume"
        else SurfaceDeformableObjectCfg
    )
    cfg = cls.from_dict(
        {
            "uid": "particles",
            "shape": {
                "shape_type": "Mesh",
                "fpath": "particles.obj",
            },
            "attrs": {"contact_material": values},
        }
    )
    restored = cls.from_dict(cfg.to_dict())
    assert restored.attrs.contact_material == values
    lower = (
        volume_deformable_desc_from_cfg
        if topology == "volume"
        else surface_deformable_desc_from_cfg
    )
    desc, _ = lower(restored)
    assert desc.physics.contact_material.stiffness == values["stiffness"]
    assert desc.physics.contact_material.damping == values["damping"]
    assert desc.physics.contact_material.dynamic_friction == values["dynamic_friction"]


@pytest.mark.parametrize("property_name", ["stiffness", "damping", "dynamic_friction"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
@pytest.mark.usefixtures("requires_particle_contact_material")
def test_invalid_particle_contact_material_values_fail_at_lowering(
    property_name, value
) -> None:
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {
            "uid": "cloth",
            "shape": {"shape_type": "Mesh", "fpath": "cloth.obj"},
            "attrs": {"contact_material": {property_name: value}},
        }
    )
    with pytest.raises(ValueError, match=property_name):
        surface_deformable_desc_from_cfg(cfg)


def test_explicit_particle_material_rejects_unsupported_engine(monkeypatch) -> None:
    import dexsim.spawn

    monkeypatch.delattr(
        dexsim.spawn, "NewtonParticleContactMaterialDesc", raising=False
    )
    cfg = SurfaceDeformableObjectCfg.from_dict(
        {
            "uid": "cloth",
            "shape": {"shape_type": "Mesh", "fpath": "cloth.obj"},
            "attrs": {"contact_material": {"stiffness": 1e3}},
        }
    )
    with pytest.raises(RuntimeError, match="per-object particle materials"):
        surface_deformable_desc_from_cfg(cfg)


def _two_tet_mesh():
    import numpy as np

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.array([[0, 1, 2, 5], [1, 2, 3, 4]], dtype=np.int32)
    return vertices, tetrahedra


def test_volume_explicit_tetrahedra_keep_particle_order() -> None:
    import numpy as np
    from dexsim.spawn.soft_body_meshing import extract_tet_surface_triangles

    vertices, tetrahedra = _two_tet_mesh()
    cfg = VolumeDeformableObjectCfg.from_dict(
        {
            "uid": "block",
            "shape": {"vertices": vertices, "triangles": None},
            "tetrahedra": tetrahedra.tolist(),
        }
    )
    desc, _ = volume_deformable_desc_from_cfg(cfg)

    np.testing.assert_array_equal(desc.positions, vertices)
    np.testing.assert_array_equal(desc.tetrahedra, tetrahedra)
    np.testing.assert_array_equal(
        desc.surface_triangles, extract_tet_surface_triangles(tetrahedra)
    )
    assert desc.meshing is None
    assert desc.meshing_stats.source_kind == "tet_mesh"
    assert desc.meshing_stats.particle_count == len(vertices)
    surface_ids = np.unique(desc.surface_triangles)
    np.testing.assert_array_equal(desc.mesh.vertices, vertices[surface_ids])


def test_volume_explicit_tetrahedra_accept_authored_surface() -> None:
    import numpy as np

    vertices, tetrahedra = _two_tet_mesh()
    surface = np.array([[0, 2, 1], [1, 2, 3]], dtype=np.int32)
    cfg = VolumeDeformableObjectCfg(
        uid="block",
        shape={"vertices": vertices, "triangles": surface},
        tetrahedra=tetrahedra,
    )
    desc, _ = volume_deformable_desc_from_cfg(cfg)
    np.testing.assert_array_equal(desc.surface_triangles, surface)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"shape": {"fpath": "mesh.obj"}}, ValueError, "array-backed"),
        ({"tetrahedra": [[0, 1, 2, 9]]}, ValueError, "outside the vertices"),
        ({"tetrahedra": [[0.0, 1.0, 2.0, 3.0]]}, TypeError, "integer"),
        ({"tetrahedra": [[0, 1, 2]]}, ValueError, r"\(M, 4\)"),
    ],
)
def test_volume_explicit_tetrahedra_reject_invalid_input(
    overrides, error, message
) -> None:
    vertices, tetrahedra = _two_tet_mesh()
    data = {
        "uid": "block",
        "shape": {"vertices": vertices, "triangles": None},
        "tetrahedra": tetrahedra,
        **overrides,
    }
    with pytest.raises(error, match=message):
        volume_deformable_desc_from_cfg(VolumeDeformableObjectCfg.from_dict(data))
