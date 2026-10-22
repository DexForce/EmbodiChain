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

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import (
    CableAttachmentCfg,
    CableObjectCfg,
    CablePhysicsCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    DefaultCollisionPropertiesCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.spawn.descriptors import cable_desc_from_cfg

pytestmark = pytest.mark.no_sim


def _cfg(**kwargs) -> CableObjectCfg:
    return CableObjectCfg(
        uid="wire", points=[[0, 0, 0], [0, 0, 0.2]], radius=0.01, **kwargs
    )


def test_missing_engine_cable_descriptors_are_rejected_at_creation(monkeypatch):
    """Optional cable support must not prevent ordinary core/config imports."""
    import dexsim.spawn

    monkeypatch.delattr(dexsim.spawn, "CableDesc")
    cfg = _cfg()
    with pytest.raises(NotImplementedError, match="DexSim build exposing CableDesc"):
        cable_desc_from_cfg(cfg)


def test_nested_config_and_contact_schema_round_trip() -> None:
    cfg = CableObjectCfg.from_dict(
        {
            "uid": "wire",
            "points": [[0, 0, 0], [0, 0, 0.2]],
            "radius": 0.01,
            "init_pos": [0.4, 0.2, 1.0],
            "fixed_segments": [0],
            "attrs": {
                "bend_stiffness": 0.0,
                "twist_damping": 0.0,
                "collision_props": {"backend": "newton", "margin": 0.002, "gap": 0.003},
                "material_props": {"backend": "newton", "dynamic_friction": 0.6},
            },
            "visual_material": {"uid": "wire_mat", "base_color": [1, 0, 0, 1]},
        }
    )
    assert isinstance(cfg.attrs, CablePhysicsCfg)
    assert isinstance(cfg.attrs.collision_props, NewtonCollisionPropertiesCfg)
    assert isinstance(cfg.attrs.material_props, NewtonRigidBodyMaterialCfg)
    restored = CableObjectCfg.from_dict(pickle.loads(pickle.dumps(cfg.to_dict())))
    desc, materials = cable_desc_from_cfg(restored)
    assert desc.fixed_segments == (0,)
    assert desc.physics.bend_stiffness == 0.0 and desc.physics.stretch_stiffness is None
    assert desc.physics.twist_damping == 0.0
    assert desc.collision.margin == 0.002 and desc.collision.gap == 0.003
    assert desc.collision.mu == 0.6
    assert materials["wire_mat"] is desc.render_material
    np.testing.assert_allclose(desc.pose[:3, 3], [0.4, 0.2, 1.0])


def test_optional_material_defaults_are_not_filled_by_embodichain() -> None:
    desc, materials = cable_desc_from_cfg(_cfg())
    assert desc.collision is None and materials == {}
    assert desc.physics.shear_stiffness is None
    assert desc.physics.twist_stiffness is None


def test_material_frames_convert_xyzw_once_without_mutating_input() -> None:
    # An axial rotation about +Z distinguishes quaternion layouts while
    # preserving the required segment tangent.
    quaternions = Rotation.from_euler("z", [37], degrees=True).as_quat().reshape(1, 4)
    saved = quaternions.copy()
    cfg = _cfg(segment_quaternions=quaternions)
    desc, _ = cable_desc_from_cfg(cfg)
    np.testing.assert_allclose(
        desc.segment_quaternions, saved[:, [3, 0, 1, 2]], atol=1e-7
    )
    np.testing.assert_array_equal(quaternions, saved)


def test_descriptor_owns_geometry_and_attachment_input() -> None:
    points = np.array([[0, 0, 0], [0, 0, 0.2]], dtype=np.float32)
    anchor = np.array([0, 0, 0.2], dtype=np.float32)
    cfg = CableObjectCfg.from_dict(
        {
            "uid": "wire",
            "points": points,
            "radius": 0.01,
            "end_attachment": {"target": "connector", "position": anchor},
        }
    )
    assert isinstance(cfg.end_attachment, CableAttachmentCfg)
    desc, _ = cable_desc_from_cfg(cfg)
    points[:] = 10
    anchor[:] = 10
    np.testing.assert_allclose(desc.points[-1], [0, 0, 0.2])
    np.testing.assert_allclose(desc.end_attachment.position, [0, 0, 0.2])


@pytest.mark.parametrize(
    "overrides",
    [
        {"radius": 0},
        {"points": [[0, 0, 0], [0, 0, 0]]},
        {"fixed_segments": [True]},
        {"closed": True},
        {"segment_quaternions": [[0, 0, 0]]},
        {"attrs": {"bend_stiffness": -1}},
    ],
)
def test_invalid_geometry_and_material_rejected_before_spawn(overrides) -> None:
    data = {"uid": "wire", "points": [[0, 0, 0], [0, 0, 0.2]], "radius": 0.01}
    data.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        cable_desc_from_cfg(CableObjectCfg.from_dict(data))


def test_required_geometry_is_validated() -> None:
    with pytest.raises(TypeError, match="Missing values"):
        cable_desc_from_cfg(CableObjectCfg(uid="wire"))


def test_custom_visual_uses_existing_mesh_and_material_compiler() -> None:
    shape = MeshCfg(
        vertices=np.array([[0, 0, 0], [0.01, 0, 0], [0, 0, 0.2]]),
        triangles=np.array([[0, 1, 2]]),
        visual_material=VisualMaterialCfg(uid="custom_mat"),
    )
    desc, materials = cable_desc_from_cfg(_cfg(visual_shape=shape))
    assert len(desc.renders) == 1
    assert desc.renders[0].material_ref == "custom_mat"
    assert "custom_mat" in materials and desc.render_material is None
    with pytest.raises(ValueError, match="owns its visual_material"):
        cable_desc_from_cfg(
            _cfg(visual_shape=shape, visual_material=VisualMaterialCfg())
        )


def test_unsupported_collision_toggle_is_not_silently_dropped() -> None:
    cfg = _cfg(
        attrs=CablePhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(collision_enabled=False)
        )
    )
    with pytest.raises(NotImplementedError, match="collision_enabled"):
        cable_desc_from_cfg(cfg)


def test_default_native_collision_override_is_rejected() -> None:
    cfg = _cfg(attrs=CablePhysicsCfg(collision_props=DefaultCollisionPropertiesCfg()))
    with pytest.raises(NotImplementedError, match="common or Newton"):
        cable_desc_from_cfg(cfg)


def test_custom_visual_does_not_silently_ignore_projective_uv_request() -> None:
    with pytest.raises(NotImplementedError, match="authored UVs"):
        cable_desc_from_cfg(
            _cfg(visual_shape=MeshCfg(fpath="unused.obj", compute_uv=True))
        )
