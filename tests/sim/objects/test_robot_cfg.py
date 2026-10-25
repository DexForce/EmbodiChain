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

import enum
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    JointDrivePropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RobotCfg,
)
from embodichain.lab.sim.motion.workspace import RobotWorkspaceCfg
from embodichain_robots.dexforce_w1 import DexforceW1Cfg
from embodichain_robots.dexforce_w1.params import W1ArmKineParams
from embodichain_robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
    DexforceW1Type,
    DexforceW1Version,
)
from embodichain_robots.dexforce_w1.hand_specs import (
    get_default_w1_hand_version,
    get_w1_hand_spec,
)
from embodichain_robots.dexforce_w1.specs import get_w1_version_spec
from embodichain_robots.dexforce_w1.utils import (
    build_dexforce_w1_assembly_urdf_cfg,
    build_dexforce_w1_control_parts,
)
from embodichain.lab.sim.motion.solvers import SRSSolverCfg
from embodichain.utils import configclass
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg


@pytest.mark.no_sim
def test_arx5_source_chain_commands_and_ik_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X5A retains two independent fingers and a six-joint flange/TCP chain."""
    from embodichain_robots import arx5

    source = Path(__file__).parents[2] / "fixtures/kinematics/arx5_x5a.urdf"
    monkeypatch.setattr(arx5, "get_data_path", lambda _: str(source))
    cfg = arx5.ARX5Cfg.from_dict({})
    restored = arx5.ARX5Cfg.from_dict(cfg.to_dict())
    assert restored.control_parts == {
        "arm": [f"joint{i}" for i in range(1, 7)],
        "eef": ["joint7", "joint8"],
    }
    assert not ET.parse(source).findall(".//mimic")
    assert restored.fpath == str(source)
    assert restored.init_qpos == cfg.init_qpos
    assert restored.root_props.self_collision_enabled is True
    chain = restored.build_pk_serial_chain()["arm"]
    assert chain.get_joint_parameter_names() == restored.control_parts["arm"]
    solver_cfg = restored.solver_cfg["arm"]
    solver_cfg.urdf_path = str(source)
    solver_cfg.joint_names = restored.control_parts["arm"]
    solver = solver_cfg.init_solver("cpu")
    q = torch.tensor([restored.init_qpos[:6], [0.2, -0.5, 1.2, -0.7, 0.3, -0.2]])
    reference = chain.forward_kinematics(q).get_matrix() @ torch.tensor(
        solver_cfg.tcp, dtype=q.dtype
    )
    torch.testing.assert_close(solver.get_fk(q), reference)
    success, solution = solver.get_ik(reference, q + 0.02)
    assert bool(success.all())
    actual = solver.get_fk(solution[:, 0])
    torch.testing.assert_close(actual[:, :3, 3], reference[:, :3, 3], atol=5e-4, rtol=0)
    torch.testing.assert_close(
        actual[:, :3, :3], reference[:, :3, :3], atol=1e-3, rtol=0
    )

    # Local source and end-frame overrides must reach kinematics as well as
    # simulation; a standalone local asset must not require a catalog download.
    def unexpected_download(_: str) -> str:
        raise AssertionError("A local fpath must not download the default asset")

    monkeypatch.setattr(arx5, "get_data_path", unexpected_download)
    custom = arx5.ARX5Cfg.from_dict(
        {
            "fpath": str(source),
            "solver_cfg": {"arm": {"end_link_name": "link6"}},
            "root_props": {"self_collision_enabled": False},
        }
    )
    assert custom.root_props.self_collision_enabled is False
    import pytorch_kinematics as pk

    reference_chain = pk.build_serial_chain_from_urdf(
        source.read_bytes(), "link6", "base_link"
    )
    torch.testing.assert_close(
        custom.build_pk_serial_chain()["arm"].forward_kinematics(q).get_matrix(),
        reference_chain.forward_kinematics(q).get_matrix(),
    )


def test_mounted_sensor_config_roundtrip_and_instance_isolation() -> None:
    from embodichain.lab.sim.sensors import CameraCfg, StereoCameraCfg

    cfg = RobotCfg.from_dict(
        {
            "enable_sensor": True,
            "sensors": [
                {
                    "sensor_type": "StereoCamera",
                    "uid": "wrist",
                    "width": 32,
                    "extrinsics": {"parent": "tool", "pos": [0.1, 0.0, 0.0]},
                }
            ],
        }
    )
    restored = RobotCfg.from_dict(cfg.to_dict())
    assert restored.enable_sensor is True
    assert isinstance(restored.sensors[0], StereoCameraCfg)
    assert restored.sensors[0].extrinsics.parent == "tool"
    assert restored.sensors[0].width == 32
    restored.sensors[0].extrinsics.parent = "changed"
    assert cfg.sensors[0].extrinsics.parent == "tool"

    first, second = RobotCfg(), RobotCfg()
    first.sensors.append(CameraCfg(uid="camera"))
    assert second.sensors == []
    assert second.enable_sensor is False


def test_cobotmagic_mounted_defaults_replace_and_clear(monkeypatch) -> None:
    from embodichain_robots import CobotMagicCfg

    monkeypatch.setattr(
        "embodichain_robots.cobotmagic.get_data_path", lambda value: value
    )
    cfg = CobotMagicCfg.from_dict({})
    assert cfg.enable_sensor is False
    assert [camera.uid for camera in cfg.sensors] == [
        "cam_right_wrist",
        "cam_left_wrist",
    ]
    assert [camera.extrinsics.parent for camera in cfg.sensors] == [
        "right_link6",
        "left_link6",
    ]
    assert all(camera.enable_mask for camera in cfg.sensors)
    assert CobotMagicCfg.from_dict({"sensors": []}).sensors == []
    restored = CobotMagicCfg.from_dict(cfg.to_dict())
    assert restored.to_dict()["sensors"] == cfg.to_dict()["sensors"]
    replacement = CobotMagicCfg.from_dict(
        {
            "sensors": [
                {
                    "sensor_type": "Camera",
                    "uid": "custom",
                    "extrinsics": {"parent": "left_link6"},
                }
            ]
        }
    )
    assert [camera.uid for camera in replacement.sensors] == ["custom"]
    replacement.sensors[0].extrinsics.parent = "changed"
    assert CobotMagicCfg.from_dict({}).sensors[0].extrinsics.parent == "right_link6"


def _mock_w1_asset_paths(monkeypatch, tmp_path):
    import embodichain.lab.sim.cfg as sim_cfg
    from embodichain_robots.dexforce_w1 import utils as w1_utils

    def resolve(path):
        resolved = tmp_path / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text("<robot name='w1' />")
        return str(resolved)

    monkeypatch.setattr(w1_utils, "get_data_path", resolve)
    monkeypatch.setattr(sim_cfg, "get_data_path", resolve)


def test_dexforce_w1_roundtrip():
    cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1", "version": "v021"})
    assert type(cfg.root_props) is ArticulationRootPropertiesCfg
    assert cfg.root_props.min_position_iters == 32
    assert cfg.root_props.min_velocity_iters == 8
    d = cfg.to_dict()
    assert d["uid"] == "dexforce_w1"
    cfg2 = DexforceW1Cfg.from_dict(d)
    assert cfg2.uid == "dexforce_w1"
    assert cfg2.version == DexforceW1Version.V021
    assert type(cfg2.root_props) is ArticulationRootPropertiesCfg


def test_dexforce_w1_solver_cfg_is_srs_and_set_once():
    cfg = DexforceW1Cfg.from_dict({})
    assert isinstance(cfg.solver_cfg["left_arm"], SRSSolverCfg)
    assert isinstance(cfg.solver_cfg["right_arm"], SRSSolverCfg)


def test_dexforce_w1_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown DexforceW1 configuration fields"):
        DexforceW1Cfg.from_dict({"unsupported_variant": "value"})


@pytest.mark.parametrize(
    "field_name",
    ["hand_types", "hand_versions", "hand_attach_xposes"],
)
def test_w1_optional_hand_mapping_accepts_none(field_name):
    cfg = DexforceW1Cfg.from_dict({field_name: None})

    assert cfg.hand_types == {}
    assert cfg.hand_versions == {
        DexforceW1ArmSide.LEFT: DexforceW1HandVersion.V021,
        DexforceW1ArmSide.RIGHT: DexforceW1HandVersion.V021,
    }
    assert cfg.hand_attach_xposes == {}


@pytest.mark.parametrize(
    "field_name",
    ["hand_types", "hand_versions", "hand_attach_xposes"],
)
def test_w1_optional_hand_mapping_rejects_non_mapping(field_name):
    with pytest.raises(TypeError, match=f"{field_name} must be a mapping or None"):
        DexforceW1Cfg.from_dict({field_name: []})


def test_w1_builders_normalize_string_hand_mappings(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    attach_xpos = np.eye(4)
    attach_xpos[0, 3] = 0.123
    hand_types = {"left": "DH_PGC_GRIPPER"}
    hand_versions = {"left": "v021"}

    urdf_cfg = build_dexforce_w1_assembly_urdf_cfg(
        version="v025",
        hand_types=hand_types,
        hand_versions=hand_versions,
        hand_attach_xposes={"left": attach_xpos},
    )
    control_parts = build_dexforce_w1_control_parts(
        version="v025",
        hand_types=hand_types,
        hand_versions=hand_versions,
        include_hand=True,
    )

    left_hand = urdf_cfg.components["left_hand"]
    assert "DH_PGC_140_50" in left_hand["urdf_path"]
    np.testing.assert_allclose(
        left_hand["transform"],
        get_w1_version_spec("v025").compose_eef_attach_xpos(
            DexforceW1ArmSide.LEFT, attach_xpos
        ),
    )
    assert control_parts["left_eef"] == [
        "LEFT_FINGER1_JOINT",
        "LEFT_FINGER2_JOINT",
    ]


@pytest.mark.parametrize(
    "removed_field",
    [
        "arm_sides",
        "include_chassis",
        "include_torso",
        "include_head",
        "include_eyes",
        "include_wrist_cameras",
        "component_versions",
    ],
)
def test_dexforce_w1_rejects_removed_builder_fields(removed_field):
    with pytest.raises(ValueError, match="Unknown DexforceW1 configuration fields"):
        DexforceW1Cfg.from_dict({removed_field: None})


def test_w1_v025_eef_offset_applies_to_attach_and_tcp():
    v021 = get_w1_version_spec(DexforceW1Version.V021)
    v025 = get_w1_version_spec(DexforceW1Version.V025)
    expected_offset = np.eye(4)
    expected_offset[2, 3] = 0.012
    hand_spec = get_w1_hand_spec(
        DexforceW1HandBrand.BRAINCO_HAND, DexforceW1HandVersion.V021
    )

    for arm_side in DexforceW1ArmSide:
        np.testing.assert_allclose(v021.eef_attach_xpos(arm_side), np.eye(4))
        np.testing.assert_allclose(v025.eef_attach_xpos(arm_side), expected_offset)
        np.testing.assert_allclose(
            v025.tcp(arm_side),
            expected_offset @ v021.tcp(arm_side),
        )
        np.testing.assert_allclose(
            v025.compose_eef_attach_xpos(
                arm_side,
                hand_spec.for_side(arm_side).attach_xpos,
            ),
            expected_offset @ np.asarray(hand_spec.for_side(arm_side).attach_xpos),
        )


def test_w1_v025_eef_offset_composes_with_custom_attach():
    spec = get_w1_version_spec(DexforceW1Version.V025)
    custom_attach = np.eye(4)
    custom_attach[:3, 3] = [0.01, -0.02, 0.03]

    result = spec.compose_eef_attach_xpos(DexforceW1ArmSide.RIGHT, custom_attach)

    np.testing.assert_allclose(result[:3, 3], [0.01, -0.02, 0.042])
    np.testing.assert_allclose(result[:3, :3], np.eye(3))


def test_w1_v022_version_spec_is_registered():
    spec = get_w1_version_spec("v022")

    assert spec.version == DexforceW1Version.V022
    assert spec.assembly_name == "DexforceW1V022"
    assert spec.full_robot_urdf() == "DexforceW1V022/w1/robot.urdf"
    assert (
        spec.component_urdf(DexforceW1Type.LEFT_ARM)
        == "DexforceW1V022/w1/left_arm.urdf"
    )
    assert (
        spec.component_urdf(DexforceW1Type.RIGHT_ARM)
        == "DexforceW1V022/w1/right_arm.urdf"
    )


def test_w1_v022_provisional_eef_baseline_is_composed_consistently():
    v021 = get_w1_version_spec(DexforceW1Version.V021)
    v022 = get_w1_version_spec(DexforceW1Version.V022)

    for arm_side in DexforceW1ArmSide:
        np.testing.assert_allclose(v022.eef_attach_xpos(arm_side), np.eye(4))
        np.testing.assert_allclose(v022.tcp(arm_side), v021.tcp(arm_side))


def test_w1_v022_cfg_uses_registered_asset_paths(tmp_path, monkeypatch):
    import embodichain.lab.sim.cfg as sim_cfg
    from embodichain_robots.dexforce_w1 import utils as w1_utils

    registered_paths = []

    def resolve_registered_path(path):
        registered_paths.append(path)
        resolved = tmp_path / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text("<robot name='w1' />")
        return str(resolved)

    monkeypatch.setattr(w1_utils, "get_data_path", resolve_registered_path)
    monkeypatch.setattr(sim_cfg, "get_data_path", resolve_registered_path)
    cfg = DexforceW1Cfg.from_dict(
        {
            "uid": "dexforce_w1_v022",
            "version": "v022",
            "with_default_eef": False,
        }
    )

    assert cfg.version == DexforceW1Version.V022
    assert cfg.uid == "dexforce_w1_v022"
    assert cfg.urdf_cfg.fname == "DexforceW1V022"
    assert "DexforceW1V022/w1/left_arm.urdf" in registered_paths
    assert "DexforceW1V022/w1/right_arm.urdf" in registered_paths


def test_w1_cfg_builds_complete_dual_arm_robot(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    cfg = DexforceW1Cfg.from_dict({"version": "v021", "with_default_eef": False})

    assert set(cfg.urdf_cfg.components) == {
        "chassis",
        "torso",
        "head",
        "left_arm",
        "right_arm",
    }
    assert set(cfg.control_parts) == {
        "torso",
        "head",
        "left_arm",
        "right_arm",
        "dual_arm",
        "full_body",
    }
    assert set(cfg.solver_cfg) == {"left_arm", "right_arm"}


def test_w1_hand_version_is_independent_of_robot_version(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    from embodichain_robots.dexforce_w1 import utils as w1_utils

    selected_versions = []
    original_get_urdf = w1_utils.hand_manager.get_urdf

    def capture_version(brand, side, version):
        selected_versions.append((side, version))
        return original_get_urdf(brand, side, version)

    monkeypatch.setattr(w1_utils.hand_manager, "get_urdf", capture_version)
    DexforceW1Cfg.from_dict(
        {
            "version": "v025",
        }
    )

    assert selected_versions
    assert all(
        version == DexforceW1HandVersion.V021 for _, version in selected_versions
    )


@pytest.mark.parametrize("brand", list(DexforceW1HandBrand))
def test_w1_hand_brand_defaults_are_explicitly_registered(brand):
    version = get_default_w1_hand_version(brand)

    assert version == DexforceW1HandVersion.V021
    assert get_w1_hand_spec(brand, version).brand == brand


def test_w1_hand_version_roundtrip(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    cfg = DexforceW1Cfg.from_dict(
        {"version": "v025", "hand_versions": {"left": "v021"}}
    )

    data = cfg.to_dict()
    restored = DexforceW1Cfg.from_dict(data)

    assert data["hand_versions"] == {"left": "v021", "right": "v021"}
    assert restored.hand_versions == {
        DexforceW1ArmSide.LEFT: DexforceW1HandVersion.V021,
        DexforceW1ArmSide.RIGHT: DexforceW1HandVersion.V021,
    }


def test_w1_rejects_unregistered_hand_version():
    with pytest.raises(ValueError, match="Invalid Dexforce W1 hand version"):
        DexforceW1Cfg.from_dict({"hand_versions": {"left": "v025"}})


def test_w1_rejects_robot_version_as_hand_version():
    with pytest.raises(ValueError, match="Invalid Dexforce W1 hand version"):
        DexforceW1Cfg.from_dict({"hand_versions": {"left": DexforceW1Version.V021}})


@pytest.mark.parametrize("version", ["v025", "V025", DexforceW1Version.V025])
def test_w1_kine_params_accept_consistent_version_forms(version):
    params = W1ArmKineParams.from_dict({"arm_side": "left", "version": version})

    assert params.arm_side == DexforceW1ArmSide.LEFT
    assert params.version == DexforceW1Version.V025


def test_w1_v030_standard_defaults_and_json_roundtrip(monkeypatch, tmp_path):
    import json
    from embodichain.lab.sim.motion.solvers import FEPSolverCfg

    _mock_w1_asset_paths(monkeypatch, tmp_path)
    cfg = DexforceW1Cfg.from_dict({"version": "v030"})
    assert cfg.with_default_eef is False
    assert set(cfg.urdf_cfg.components) == {
        "chassis",
        "torso",
        "head",
        "left_arm",
        "right_arm",
    }
    assert len(set(j for names in cfg.control_parts.values() for j in names)) == 20
    assert "left_eef" not in cfg.control_parts and "right_eef" not in cfg.control_parts
    assert cfg.urdf_cfg.sensors == []
    for side, solver in cfg.solver_cfg.items():
        assert isinstance(solver, FEPSolverCfg)
        assert solver.redundancy_search
        assert solver.arm_angle_weight == 0.0
        assert solver.user_qpos_limits is None
        np.testing.assert_array_equal(solver.tcp, np.eye(4))
    restored = DexforceW1Cfg.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.with_default_eef is False
    assert restored.control_parts == cfg.control_parts
    assert restored.urdf_cfg.component_prefix == cfg.urdf_cfg.component_prefix
    for side in cfg.solver_cfg:
        np.testing.assert_allclose(
            restored.solver_cfg[side].reference_qpos,
            cfg.solver_cfg[side].reference_qpos,
        )


def test_w1_v030_opt_in_hands_keep_independent_versions(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    cfg = DexforceW1Cfg.from_dict({"version": "v030", "with_default_eef": True})
    assert "left_hand" in cfg.urdf_cfg.components
    assert "right_hand" in cfg.urdf_cfg.components
    assert all(v == DexforceW1HandVersion.V021 for v in cfg.hand_versions.values())
    assert "left_eef" in cfg.control_parts and "right_eef" in cfg.control_parts


def test_w1_v030_custom_tool_tcp_roundtrip_preserves_fep(monkeypatch, tmp_path):
    import json
    from embodichain.lab.sim.motion.solvers import FEPSolverCfg

    _mock_w1_asset_paths(monkeypatch, tmp_path)
    tcp = np.array(
        [
            [0.0, -1.0, 0.0, 0.02],
            [1.0, 0.0, 0.0, -0.03],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cfg = DexforceW1Cfg.from_dict(
        {
            "version": "v030",
            "solver_cfg": {"left_arm": {"tcp": tcp.tolist()}},
        }
    )
    restored = DexforceW1Cfg.from_dict(json.loads(json.dumps(cfg.to_dict())))
    for instance in (cfg, restored):
        solver = instance.solver_cfg["left_arm"]
        assert isinstance(solver, FEPSolverCfg)
        assert solver.redundancy_search
        np.testing.assert_array_equal(solver.tcp, tcp)
        np.testing.assert_allclose(
            solver.reference_qpos, [0, 0, -np.pi / 2, 0, 0, -np.pi / 2, 0]
        )


def test_w1_v030_rejects_srs_geometry_and_uncalibrated_wrist_camera():
    with pytest.raises(ValueError, match="no SRS geometry.*FEP"):
        W1ArmKineParams(arm_side=DexforceW1ArmSide.LEFT, version=DexforceW1Version.V030)
    with pytest.raises(ValueError, match="no wrist camera calibration"):
        get_w1_version_spec("v030").wrist_camera_xpos(DexforceW1ArmSide.LEFT)


def test_w1_v030_solver_specs_are_immutable_and_isolated(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    spec = get_w1_version_spec("v030")
    with pytest.raises(TypeError):
        spec.arm_solver_configs[DexforceW1ArmSide.LEFT]["reference_qpos"] = ()
    cfg = DexforceW1Cfg.from_dict({"version": "v030"})
    cfg.solver_cfg["left_arm"].reference_qpos = [0.0] * 7
    other = DexforceW1Cfg.from_dict({"version": "v030"})
    assert other.solver_cfg["left_arm"].reference_qpos[2] == -np.pi / 2


def test_w1_version_spec_mappings_are_immutable():
    spec = get_w1_version_spec("v025")

    with pytest.raises(TypeError):
        spec.solver_tcp[DexforceW1ArmSide.LEFT] = np.eye(4)
    with pytest.raises(TypeError):
        spec.component_urdfs[DexforceW1Type.LEFT_ARM] = "other.urdf"


def test_w1_v025_custom_eef_and_tcp_roundtrip(monkeypatch, tmp_path):
    _mock_w1_asset_paths(monkeypatch, tmp_path)
    baseline = DexforceW1Cfg.from_dict({"version": "v025"})
    raw_attach = np.eye(4)
    raw_attach[:3, 3] = [0.01, -0.02, 0.03]
    raw_tcp = np.eye(4)
    raw_tcp[:3, 3] = [0.04, 0.05, 0.06]
    cfg = DexforceW1Cfg.from_dict(
        {
            "version": "v025",
            "urdf_cfg": {
                "components": {
                    "left_hand": {
                        "urdf_path": baseline.urdf_cfg.components["left_hand"][
                            "urdf_path"
                        ],
                        "transform": raw_attach.tolist(),
                    }
                }
            },
            "solver_cfg": {"left_arm": {"tcp": raw_tcp.tolist()}},
        }
    )

    expected_attach = get_w1_version_spec("v025").compose_eef_attach_xpos(
        DexforceW1ArmSide.LEFT, raw_attach
    )
    expected_tcp = get_w1_version_spec("v025").compose_eef_attach_xpos(
        DexforceW1ArmSide.LEFT, raw_tcp
    )
    restored = DexforceW1Cfg.from_dict(cfg.to_dict())

    np.testing.assert_allclose(
        cfg.urdf_cfg.components["left_hand"]["transform"], expected_attach
    )
    np.testing.assert_allclose(cfg.solver_cfg["left_arm"].tcp, expected_tcp)
    np.testing.assert_allclose(
        restored.urdf_cfg.components["left_hand"]["transform"], expected_attach
    )
    np.testing.assert_allclose(restored.solver_cfg["left_arm"].tcp, expected_tcp)


class _RoundTripVariant(enum.Enum):
    A = "a"
    B = "b"


@configclass
class _RoundTripCfg(RobotCfg):
    """Synthetic cfg to exercise the base serialization + _build_defaults hook."""

    variant: _RoundTripVariant = _RoundTripVariant.A

    @classmethod
    def from_dict(cls, init_dict):
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict=None):
        init_dict = init_dict or {}
        self.uid = "roundtrip"
        self.variant = _RoundTripVariant(init_dict.get("variant", "a"))
        self.control_parts = {"arm": ["J1", "J2"]}
        self.joint_drive_props = JointDrivePropertiesCfg(
            stiffness={"J[1-2]": 1e4}, damping={"J[1-2]": 1e3}
        )


def test_robotcfg_to_dict_roundtrip():
    cfg = _RoundTripCfg.from_dict({"variant": "b"})
    assert cfg.variant == _RoundTripVariant.B

    d = cfg.to_dict()
    assert d["uid"] == "roundtrip"
    assert d["variant"] == "b"

    cfg2 = _RoundTripCfg.from_dict(d)
    assert cfg2.uid == "roundtrip"
    assert cfg2.variant == _RoundTripVariant.B
    assert cfg2.control_parts == {"arm": ["J1", "J2"]}
    assert cfg2.joint_drive_props.stiffness == {"J[1-2]": 1e4}


from embodichain_robots.cobotmagic import CobotMagicCfg
from embodichain_robots import AlohaMiniCfg, TianjiMarvinCfg
from embodichain_robots.franka_panda import FrankaPandaCfg, FrankaPandaPgiCfg
from embodichain_robots.ur_robot import URRobotCfg
from embodichain.lab.sim.motion.solvers import OPWSolverCfg


@pytest.mark.parametrize("part_name", ["eef", "hand", "custom_claw"])
def test_control_part_naming_convention_does_not_rewrite_user_names(
    part_name: str,
) -> None:
    """Control-group names remain user-defined through config round trips."""
    parts = {"motion_group": ["joint1"], part_name: ["finger_joint"]}
    cfg = RobotCfg.from_dict({"control_parts": parts})

    assert cfg.control_parts == parts
    restored = RobotCfg.from_dict(cfg.to_dict())
    assert restored.control_parts == parts


@pytest.mark.parametrize(
    ("robot_class", "module_path", "expected_eef_parts"),
    [
        (FrankaPandaCfg, "franka_panda", {"eef"}),
        (AlohaMiniCfg, "aloha_mini", {"left_eef", "right_eef"}),
        (TianjiMarvinCfg, "tianji_marvin", {"left_eef", "right_eef"}),
    ],
)
def test_specified_robot_defaults_use_eef_control_part_names(
    robot_class: type[RobotCfg],
    module_path: str,
    expected_eef_parts: set[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renaming the default groups preserves their source joint identifiers."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    monkeypatch.setattr(
        f"embodichain_robots.{module_path}.get_data_path", lambda value: value
    )
    cfg = robot_class.from_dict({})

    assert expected_eef_parts.issubset(cfg.control_parts)
    assert {"hand", "left_hand", "right_hand"}.isdisjoint(cfg.control_parts)
    if robot_class is FrankaPandaCfg:
        assert cfg.control_parts["eef"] == ["fr3_finger_joint1", "fr3_finger_joint2"]
    elif robot_class is AlohaMiniCfg:
        assert cfg.control_parts["left_eef"] == ["left_gripper"]
    else:
        assert cfg.control_parts["left_eef"] == [
            "LEFT_HAND_FINGER_1",
            "LEFT_HAND_FINGER_2",
        ]
    if robot_class in (AlohaMiniCfg, TianjiMarvinCfg):
        # Existing gripper gains and effort budgets must survive the rename.
        for part in expected_eef_parts:
            assert cfg.joint_drive_props.stiffness[part] == 3e2
            assert cfg.joint_drive_props.damping[part] == 3e1
            assert cfg.joint_drive_props.max_effort[part] == 3e3


def test_aloha_mini_control_parts_match_source_joints():
    cfg = AlohaMiniCfg.from_dict({})
    assert cfg.uid == "AlohaMini"
    assert cfg.urdf_cfg is None
    assert cfg.fpath.endswith("AlohaMini/alohamini2pro.urdf")
    assert {part: len(joints) for part, joints in cfg.control_parts.items()} == {
        "left_arm": 6,
        "right_arm": 6,
        "left_eef": 1,
        "right_eef": 1,
        "torso": 1,
    }
    assert cfg.control_parts["torso"] == ["vertical_move"]
    assert cfg.control_parts["left_eef"] == ["left_gripper"]
    assert cfg.control_parts["right_eef"] == ["right_gripper"]
    source = ET.parse(cfg.fpath).getroot()
    movable_joints = {
        joint.get("name")
        for joint in source.findall("joint")
        if joint.get("type") != "fixed"
    }
    controlled = [joint for joints in cfg.control_parts.values() for joint in joints]
    assert len(controlled) == len(set(controlled))
    assert set(controlled) <= movable_joints
    assert movable_joints - set(controlled) == {
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
        "wheel1_joint",
        "wheel2_joint",
        "wheel3_joint",
    }


def test_aloha_mini_overrides_and_roundtrip():
    cfg = AlohaMiniCfg.from_dict(
        {
            "uid": "aloha_mini",
            "init_pos": [0.0, 0.0, 0.2],
            "joint_drive_props": {"stiffness": {"torso": 2e4}},
            "solver_cfg": {"left_arm": {"num_samples": 8}},
        }
    )
    assert cfg.uid == "aloha_mini"
    assert cfg.joint_drive_props.stiffness["torso"] == 2e4
    assert cfg.joint_drive_props.stiffness["left_arm"] == 7e4
    assert cfg.solver_cfg["left_arm"].num_samples == 8
    assert cfg.solver_cfg["right_arm"].num_samples == 128
    assert AlohaMiniCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_aloha_mini_serial_chains_match_control_joint_order():
    cfg = AlohaMiniCfg.from_dict({})
    chains = cfg.build_pk_serial_chain()
    assert set(chains) == set(cfg.control_parts)
    for part, chain in chains.items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]

    # Torso motion is local +Z and must not contain the planar base joints.
    torso = chains["torso"]
    poses = torso.forward_kinematics(torch.tensor([[0.0], [0.1]])).get_matrix()
    torch.testing.assert_close(
        poses[1, :3, 3] - poses[0, :3, 3], torch.tensor([0.0, 0.0, 0.1])
    )


def test_aloha_mini_solvers_use_local_frames_and_source_tcp():
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg, SSIKSolverCfg

    cfg = AlohaMiniCfg.from_dict({})
    chains = cfg.build_pk_serial_chain()
    assert set(cfg.solver_cfg) == {"left_arm", "right_arm", "torso"}
    for part, solver_cfg in cfg.solver_cfg.items():
        if part == "torso":
            assert isinstance(solver_cfg, PytorchSolverCfg)
            assert solver_cfg.is_only_position_constraint
        else:
            assert isinstance(solver_cfg, SSIKSolverCfg)
            assert solver_cfg.backend == "warp"
        if part.endswith("_arm"):
            side = part.removesuffix("_arm")
            assert solver_cfg.root_link_name == f"{side}_Base"
            assert solver_cfg.end_link_name == f"{side}_tcp"
        solver_cfg.urdf_path = cfg.fpath
        solver_cfg.joint_names = cfg.control_parts[part]
        solver = solver_cfg.init_solver(device=torch.device("cpu"))
        assert solver.dof == len(cfg.control_parts[part])
        qpos = torch.zeros((1, solver.dof))
        torch.testing.assert_close(
            solver.get_fk(qpos), chains[part].forward_kinematics(qpos).get_matrix()
        )


@pytest.mark.parametrize("typed", [False, True])
def test_aloha_mini_explicit_pytorch_solver_override_roundtrips(typed):
    """The analytic default does not replace an explicitly selected old solver."""
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg, SSIKSolverCfg

    explicit = PytorchSolverCfg(
        num_samples=8,
        root_link_name="left_Base",
        end_link_name="left_tcp",
        ik_nearest_weight=[1.0] * 6,
    )
    provided = explicit if typed else explicit.to_dict()
    cfg = AlohaMiniCfg.from_dict({"solver_cfg": {"left_arm": provided}})
    assert isinstance(cfg.solver_cfg["left_arm"], PytorchSolverCfg)
    assert cfg.solver_cfg["left_arm"].num_samples == 8
    assert cfg.solver_cfg["left_arm"].root_link_name == "left_Base"
    assert cfg.solver_cfg["left_arm"].end_link_name == "left_tcp"
    assert isinstance(cfg.solver_cfg["right_arm"], SSIKSolverCfg)
    assert isinstance(cfg.solver_cfg["torso"], PytorchSolverCfg)
    restored = AlohaMiniCfg.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
    assert isinstance(restored.solver_cfg["left_arm"], PytorchSolverCfg)
    assert cfg.solver_cfg["left_arm"] is not explicit
    cfg.solver_cfg["left_arm"].num_samples = 12
    cfg.solver_cfg["left_arm"].ik_nearest_weight[0] = 3.0
    assert explicit.num_samples == 8 and explicit.ik_nearest_weight == [1.0] * 6
    if not typed:
        assert provided["num_samples"] == 8
        assert provided["ik_nearest_weight"] == [1.0] * 6


def test_aloha_mini_pk_chains_follow_overridden_asset(tmp_path):
    cfg = AlohaMiniCfg.from_dict({})
    source = ET.parse(cfg.fpath)
    torso_joint = source.getroot().find("joint[@name='vertical_move']")
    torso_joint.find("origin").set("xyz", "0 0 0.25")
    overridden_path = tmp_path / "aloha_mini.urdf"
    source.write(overridden_path)
    cfg = AlohaMiniCfg.from_dict({"fpath": str(overridden_path)})
    chain = cfg.build_pk_serial_chain()["torso"]
    pose = chain.forward_kinematics(torch.zeros((1, 1))).get_matrix()
    assert pose[0, 2, 3].item() == pytest.approx(0.25)


@pytest.mark.no_sim
@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_preserves_source_inertials_without_overlays(with_gripper):
    """Selecting the robot alone does not assert unverified source units."""
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    assert cfg.link_attrs == {}
    assert cfg.attrs.mass_props is None


@pytest.mark.no_sim
def test_tianji_marvin_custom_asset_does_not_inherit_demo_inertials(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "embodichain_robots.tianji_marvin.get_data_path", lambda value: value
    )
    asset = tmp_path / "custom.urdf"
    asset.write_text('<robot name="custom"/>')
    cfg = TianjiMarvinCfg.from_dict({"fpath": str(asset)})
    assert cfg.fpath == str(asset)
    assert cfg.link_attrs == {}
    custom = TianjiMarvinCfg.from_dict(
        {
            "fpath": str(asset),
            "link_attrs": {
                "custom_base": {
                    "link_names_expr": ["custom_base"],
                    "attrs": {"mass_props": {"mass": 2.0}},
                }
            },
        }
    )
    assert set(custom.link_attrs) == {"custom_base"}
    assert custom.link_attrs["custom_base"].attrs.mass_props.mass == 2.0
    assert TianjiMarvinCfg.from_dict(custom.to_dict()).to_dict() == custom.to_dict()


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_control_parts_match_source_joints(with_gripper):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    filename = "robot_with_ee_acd.urdf" if with_gripper else "robot_acd.urdf"
    assert cfg.uid == "TianjiMarvin"
    assert cfg.fpath.endswith(f"TianjiMarvin/{filename}")
    assert cfg.urdf_cfg is None
    expected_dofs = {"left_arm": 7, "right_arm": 7}
    if with_gripper:
        expected_dofs.update(left_eef=2, right_eef=2)
    assert {
        part: len(joints) for part, joints in cfg.control_parts.items()
    } == expected_dofs
    source = ET.parse(cfg.fpath).getroot()
    movable_joints = {
        joint.get("name")
        for joint in source.findall("joint")
        if joint.get("type") != "fixed"
    }
    controlled = [joint for joints in cfg.control_parts.values() for joint in joints]
    assert len(controlled) == len(set(controlled))
    assert set(controlled) == movable_joints
    for side in ("left", "right"):
        if with_gripper:
            leader, follower = cfg.control_parts[f"{side}_eef"]
            mimic = source.find(f"joint[@name='{follower}']/mimic")
            assert mimic.get("joint") == leader
            assert float(mimic.get("multiplier")) == 1.0
    for prop in ("stiffness", "damping", "max_effort"):
        assert set(getattr(cfg.joint_drive_props, prop)) == set(expected_dofs)


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_overrides_and_roundtrip(with_gripper):
    cfg = TianjiMarvinCfg.from_dict(
        {
            "with_gripper": with_gripper,
            "uid": "marvin",
            "joint_drive_props": {"stiffness": {"left_arm": 2e4}},
            "solver_cfg": {"left_arm": {"num_samples": 8}},
        }
    )
    cfg.validate()
    assert cfg.with_gripper is with_gripper
    assert cfg.uid == "marvin"
    assert cfg.joint_drive_props.stiffness["left_arm"] == 2e4
    assert cfg.joint_drive_props.stiffness["right_arm"] == 7e4
    assert cfg.solver_cfg["left_arm"].num_samples == 8
    assert cfg.solver_cfg["right_arm"].num_samples == 30
    assert cfg.root_props.fixed_base
    assert TianjiMarvinCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_ready_pose_has_lowered_elbows_and_joint_margin(with_gripper):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    assert len(cfg.init_qpos) == (18 if with_gripper else 14)
    assert cfg.init_qpos[14:] == ([0.0] * 4 if with_gripper else [])
    chains = cfg.build_pk_serial_chain()
    root_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for index, (side, label, sign) in enumerate((("left", "l", 1), ("right", "r", -1))):
        chain = chains[f"{side}_arm"]
        qpos = np.asarray(cfg.init_qpos[:14][index::2])
        lower, upper = np.asarray(chain.get_joint_limits())
        assert np.min(np.minimum(qpos - lower, upper - qpos)) >= 0.19
        frames = chain.forward_kinematics(
            torch.tensor(qpos, dtype=torch.float32)[None], end_only=False
        )
        elbow = (
            root_rotation
            @ frames[f"elbow_pitch_{label}_j3_link"].get_matrix()[0, :3, 3].numpy()
        )
        np.testing.assert_allclose(elbow, [0.0045, sign * 0.3188, 0.8510], atol=0.002)


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_ready_pose_can_be_overridden_without_shared_state(with_gripper):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    original = cfg.init_qpos.copy()
    cfg.init_qpos[0] = 0.0
    assert (
        TianjiMarvinCfg.from_dict({"with_gripper": with_gripper}).init_qpos == original
    )
    custom = [0.0] * len(original)
    replaced = TianjiMarvinCfg.from_dict(
        {"with_gripper": with_gripper, "init_qpos": custom}
    )
    assert replaced.init_qpos == custom
    assert TianjiMarvinCfg.from_dict(replaced.to_dict()).init_qpos == custom


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_wrist_cameras_match_variant_and_mounts(with_gripper):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    assert cfg.enable_sensor is False
    if not with_gripper:
        assert cfg.sensors == []
        return
    links = {link.get("name") for link in ET.parse(cfg.fpath).getroot().findall("link")}
    assert [camera.uid for camera in cfg.sensors] == [
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    for side, camera in zip(("left", "right"), cfg.sensors, strict=True):
        assert camera.extrinsics.parent == f"{side}_hand_base_link"
        assert camera.extrinsics.parent in links
        assert camera.enable_color and camera.enable_depth
        assert not any(
            (camera.enable_mask, camera.enable_position, camera.enable_normal)
        )
        assert (camera.width, camera.height) == (640, 360)
        fx, fy, cx, cy = camera.intrinsics
        # Validate the optics against the nominal hardware FOV, not copied K values.
        assert np.rad2deg(2 * np.arctan(camera.width / (2 * fx))) == pytest.approx(87)
        assert np.rad2deg(2 * np.arctan(camera.height / (2 * fy))) == pytest.approx(58)
        assert (cx, cy) == (camera.width / 2, camera.height / 2)
        pose = camera.extrinsics.transformation.numpy()
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(
            -pose[:3, 2],
            [0, -np.sin(np.deg2rad(12)), np.cos(np.deg2rad(12))],
            atol=1e-6,
        )
        # The mounting estimate clears the model's front lens plane by 1 mm.
        lens_plane_distance = 0.0473065
        assert pose[:3, 3] @ -pose[:3, 2] - lens_plane_distance == pytest.approx(
            0.001, abs=2e-6
        )


def test_tianji_marvin_camera_overrides_are_isolated_and_roundtrip():
    cfg = TianjiMarvinCfg.from_dict({"enable_sensor": True})
    assert cfg.enable_sensor
    assert TianjiMarvinCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    cfg.sensors[0].extrinsics.parent = "changed"
    assert (
        TianjiMarvinCfg.from_dict({}).sensors[0].extrinsics.parent
        == "left_hand_base_link"
    )
    assert TianjiMarvinCfg.from_dict({"sensors": []}).sensors == []
    custom = TianjiMarvinCfg.from_dict(
        {
            "sensors": [
                {
                    "sensor_type": "Camera",
                    "uid": "custom",
                    "extrinsics": {"parent": "base_link"},
                }
            ]
        }
    )
    assert [camera.uid for camera in custom.sensors] == ["custom"]
    assert custom.sensors[0].extrinsics.parent == "base_link"
    assert TianjiMarvinCfg.from_dict(custom.to_dict()).to_dict() == custom.to_dict()


@pytest.mark.parametrize("with_gripper", [True, False])
@pytest.mark.parametrize("use_link_frame", [False, True])
def test_tianji_marvin_serial_chains_and_solvers_match_source(
    with_gripper, use_link_frame
):
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

    init_dict = {"with_gripper": with_gripper}
    if use_link_frame:
        init_dict["solver_cfg"] = {
            part: {"tcp": np.eye(4).tolist()} for part in ("left_arm", "right_arm")
        }
    cfg = TianjiMarvinCfg.from_dict(init_dict)
    # Documented end-link-to-TCP transform for both variants.
    expected_tcp = (
        torch.eye(4)
        if use_link_frame
        else torch.tensor([[0, 0, 1, 0.13], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    )
    chains = cfg.build_pk_serial_chain()
    assert set(chains) == set(cfg.solver_cfg) == {"left_arm", "right_arm"}
    for part, chain in chains.items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]
        side = part.removesuffix("_arm")
        solver_cfg = cfg.solver_cfg[part]
        assert isinstance(solver_cfg, PytorchSolverCfg)
        assert solver_cfg.root_link_name == f"{side}_arm_base"
        assert solver_cfg.end_link_name == (
            f"{side}_hand_tool_link" if with_gripper else f"{side}_ee"
        )
        solver_cfg.urdf_path = cfg.fpath
        solver_cfg.joint_names = cfg.control_parts[part]
        solver = solver_cfg.init_solver(device=torch.device("cpu"))
        assert solver.dof == 7
        qpos = torch.tensor([[0.1, 0.2, -0.1, -0.3, 0.2, 0.1, -0.2]])
        tcp = torch.as_tensor(solver_cfg.tcp, dtype=qpos.dtype)
        torch.testing.assert_close(tcp, expected_tcp)
        expected_pose = chain.forward_kinematics(qpos).get_matrix() @ expected_tcp
        torch.testing.assert_close(solver.get_fk(qpos), expected_pose)
        success, solved_qpos = solver.get_ik(expected_pose, qpos_seed=qpos)
        assert bool(success.all())
        torch.testing.assert_close(
            solver.get_fk(solved_qpos.reshape_as(qpos)),
            expected_pose,
            atol=1e-3,
            rtol=0,
        )


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_pk_chains_follow_overridden_asset(with_gripper, tmp_path):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    qpos = torch.zeros((1, 7))
    original_pose = cfg.build_pk_serial_chain()["left_arm"].forward_kinematics(qpos)
    source = ET.parse(cfg.fpath)
    shoulder = source.getroot().find("joint[@name='SHOULDER_PITCH_L_J1']/origin")
    xyz = [float(value) for value in shoulder.get("xyz").split()]
    xyz[2] += 0.25
    shoulder.set("xyz", " ".join(str(value) for value in xyz))
    overridden_path = tmp_path / "marvin.urdf"
    source.write(overridden_path)
    cfg = TianjiMarvinCfg.from_dict(
        {"with_gripper": with_gripper, "fpath": str(overridden_path)}
    )
    pose = cfg.build_pk_serial_chain()["left_arm"].forward_kinematics(qpos)
    torch.testing.assert_close(
        pose.get_matrix()[0, :3, 3] - original_pose.get_matrix()[0, :3, 3],
        torch.tensor([0.0, 0.0, 0.25]),
    )


@pytest.mark.parametrize("with_gripper", ["false", "true", 0, 1, None])
def test_tianji_marvin_rejects_non_boolean_variant(with_gripper):
    with pytest.raises(TypeError, match="with_gripper must be a boolean"):
        TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})


def test_tianji_marvin_grippers_follow_position_targets():
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    sim = SimulationManager(
        SimulationManagerCfg(headless=True, device="cpu", num_envs=1)
    )
    try:
        robot = sim.add_robot(cfg=TianjiMarvinCfg.from_dict({}))
        sim.prepare()
        assert robot.dof == 18
        assert len(robot.mimic_ids) == 2
        # Move both ways within the URDF's [-0.05, 0] metre finger range.
        for position in (-0.025, -0.01):
            target = torch.full((1, 2), position)
            for part in ("left_eef", "right_eef"):
                robot.set_qpos(target, name=part, target=True)
            sim.update(step=100)
            for part in ("left_eef", "right_eef"):
                torch.testing.assert_close(
                    robot.get_qpos(name=part), target, atol=1e-3, rtol=0
                )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def test_cobotmagic_from_dict_and_roundtrip():
    cfg = CobotMagicCfg.from_dict({})
    assert cfg.uid == "CobotMagic"
    assert set(cfg.control_parts.keys()) == {
        "left_arm",
        "left_eef",
        "right_arm",
        "right_eef",
    }
    assert isinstance(cfg.solver_cfg["left_arm"], OPWSolverCfg)
    assert isinstance(cfg.solver_cfg["right_arm"], OPWSolverCfg)
    assert isinstance(cfg.attrs, RigidBodyPhysicsCfg)
    assert type(cfg.attrs.collision_props) is CollisionPropertiesCfg
    assert cfg.attrs.collision_props.contact_offset == pytest.approx(0.001)
    assert cfg.attrs.collision_props.rest_offset == pytest.approx(0.0)
    assert type(cfg.root_props) is ArticulationRootPropertiesCfg
    assert cfg.root_props.min_position_iters == 8
    assert cfg.root_props.min_velocity_iters == 2

    d = cfg.to_dict()
    assert d["uid"] == "CobotMagic"
    cfg2 = CobotMagicCfg.from_dict(d)
    assert cfg2.uid == "CobotMagic"
    assert cfg2.control_parts == cfg.control_parts
    assert isinstance(cfg2.solver_cfg["left_arm"], OPWSolverCfg)


def test_cobotmagic_arm_effort_and_conservative_joint_velocity_limits() -> None:
    """Retain source torque bounds and cap speed by the stricter reference."""
    cfg = CobotMagicCfg.from_dict({})
    source_path = cfg.urdf_cfg.components["left_arm"]["urdf_path"]
    source_joints = {
        joint.get("name"): joint
        for joint in ET.parse(source_path).getroot().findall("joint")
    }
    from embodichain.utils.string import resolve_matching_names_values

    for side in ("left", "right"):
        names = [f"{side}_joint{index}" for index in range(1, 9)]
        for field, source_field in (
            ("max_effort", "effort"),
            ("max_velocity", "velocity"),
        ):
            rules = {
                key: value
                for key, value in getattr(cfg.joint_drive_props, field).items()
                if key.startswith(f"{side}_")
            }
            _, matched_names, values = resolve_matching_names_values(rules, names)
            limits = dict(zip(matched_names, values))
            assert set(limits) == set(names)
            for index, name in enumerate(names, 1):
                if field == "max_effort" and index >= 7:
                    continue  # The coupled gripper has a calibrated drive budget.
                expected = float(
                    source_joints[f"joint{index}"].find("limit").get(source_field)
                )
                if field == "max_velocity":
                    reference = (
                        np.deg2rad([180, 195, 180, 225, 225, 225])[index - 1]
                        if index <= 6
                        else 0.25
                    )
                    expected = min(expected, reference)
                assert limits[name] == pytest.approx(expected)


def test_cobotmagic_coupled_gripper_uses_calibrated_simulation_budget() -> None:
    """Keep the tested grasp budget separate from authored joint limits."""
    cfg = CobotMagicCfg.from_dict({})
    for side in ("left", "right"):
        pattern = f"{side}_joint[7-8]"
        assert cfg.joint_drive_props.max_effort[pattern] == 40.0
        assert cfg.joint_drive_props.stiffness[pattern] == 2000.0
        assert cfg.joint_drive_props.damping[pattern] == 70.0
        assert cfg.joint_drive_props.max_velocity[pattern] == 0.25
    restored = CobotMagicCfg.from_dict(cfg.to_dict())
    assert restored.joint_drive_props.max_effort == cfg.joint_drive_props.max_effort


@pytest.mark.parametrize(
    ("field", "arm_gains", "gripper_gain"),
    [
        ("stiffness", [30000, 30000, 20000, 12000, 8000, 4000], 2000),
        ("damping", [600, 600, 400, 240, 160, 80], 70),
    ],
)
def test_cobotmagic_per_joint_gains_resolve_after_roundtrip(
    field: str, arm_gains: list[float], gripper_gain: float
) -> None:
    """Bind each axis once in state order while retaining both gripper gains."""
    from embodichain.utils.string import resolve_matching_names_values

    cfg = CobotMagicCfg.from_dict({})
    restored = CobotMagicCfg.from_dict(cfg.to_dict())
    names = [
        f"{side}_joint{index}"
        for index in range(8, 0, -1)
        for side in ("right", "left")
    ]
    _, matched, values = resolve_matching_names_values(
        getattr(restored.joint_drive_props, field), names
    )
    assert matched == names
    expected = [
        arm_gains[index - 1] if index <= 6 else gripper_gain
        for index in range(8, 0, -1)
        for _ in range(2)
    ]
    assert values == expected


def test_cobotmagic_custom_effort_preserves_velocity_limits_on_roundtrip() -> None:
    cfg = CobotMagicCfg.from_dict(
        {"joint_drive_props": {"max_effort": {"right_joint[1-6]": 70.0}}}
    )
    assert cfg.joint_drive_props.max_effort["right_joint[1-6]"] == 70.0
    assert cfg.joint_drive_props.max_effort["left_joint[1-6]"] == 100.0
    restored = CobotMagicCfg.from_dict(cfg.to_dict())
    assert restored.joint_drive_props.max_effort == cfg.joint_drive_props.max_effort
    assert restored.joint_drive_props.max_velocity == cfg.joint_drive_props.max_velocity


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("stiffness", 12000.0), ("damping", 300.0), ("max_velocity", 2.0)],
)
def test_cobotmagic_legacy_arm_overrides_replace_defaults_without_overlap(
    side: str, field: str, value: float
) -> None:
    """Existing whole-arm patterns keep working after per-axis default tuning."""
    from embodichain.utils.string import resolve_matching_names_values

    names = [
        f"{arm}_joint{joint}" for arm in ("left", "right") for joint in range(1, 9)
    ]
    baseline = CobotMagicCfg.from_dict({})
    _, base_names, base_values = resolve_matching_names_values(
        getattr(baseline.joint_drive_props, field), names
    )
    expected = dict(zip(base_names, base_values))
    expected.update({f"{side}_joint{joint}": value for joint in range(1, 7)})
    overrides = {"joint_drive_props": {field: {f"{side}_joint[1-6]": value}}}
    cfg = CobotMagicCfg.from_dict(overrides)
    restored = CobotMagicCfg.from_dict(cfg.to_dict())
    for candidate in (cfg, restored):
        _, matched, values = resolve_matching_names_values(
            getattr(candidate.joint_drive_props, field), names
        )
        assert dict(zip(matched, values)) == expected
    assert overrides == {"joint_drive_props": {field: {f"{side}_joint[1-6]": value}}}


def test_cobotmagic_partial_drive_override_preserves_other_group_members() -> None:
    from embodichain.utils.string import resolve_matching_names_values

    cfg = CobotMagicCfg.from_dict(
        {"joint_drive_props": {"stiffness": {"left_joint1": 9000.0}}}
    )
    names = [
        f"{arm}_joint{joint}" for arm in ("left", "right") for joint in range(1, 9)
    ]
    _, matched, values = resolve_matching_names_values(
        cfg.joint_drive_props.stiffness, names
    )
    resolved = dict(zip(matched, values))
    assert resolved["left_joint1"] == 9000.0
    assert resolved["left_joint2"] == 30000.0
    assert resolved["right_joint1"] == 30000.0
    assert resolved["left_joint7"] == 2000.0


def test_cobotmagic_rejects_ambiguous_caller_drive_patterns() -> None:
    from embodichain.utils.string import resolve_matching_names_values

    cfg = CobotMagicCfg.from_dict(
        {
            "joint_drive_props": {
                "stiffness": {"left_joint[1-6]": 9000.0, "left_joint1": 8000.0}
            }
        }
    )
    names = [
        f"{arm}_joint{joint}" for arm in ("left", "right") for joint in range(1, 9)
    ]
    with pytest.raises(ValueError, match="Multiple matches"):
        resolve_matching_names_values(cfg.joint_drive_props.stiffness, names)


def test_cobotmagic_control_part_speed_override_retains_direct_rule_priority() -> None:
    from embodichain.lab.sim.spawn.descriptors import _joint_property_matches

    cfg = CobotMagicCfg.from_dict(
        {"joint_drive_props": {"max_velocity": {"left_arm": 2.0, "left_joint6": 1.5}}}
    )
    names = [
        f"{side}_joint{joint}" for side in ("left", "right") for joint in range(1, 9)
    ]
    for candidate in (cfg, CobotMagicCfg.from_dict(cfg.to_dict())):
        resolved = dict(
            _joint_property_matches(
                candidate.joint_drive_props.max_velocity,
                names,
                property_name="max_velocity",
                control_parts=candidate.control_parts,
            )
        )
        assert all(resolved[f"left_joint{joint}"] == 2.0 for joint in range(1, 6))
        assert resolved["left_joint6"] == 1.5
        assert resolved["right_joint1"] == pytest.approx(np.pi)
        assert resolved["left_joint7"] == 0.25


@pytest.mark.parametrize(
    ("cfg_type", "init_dict"),
    [
        (CobotMagicCfg, {}),
        (FrankaPandaCfg, {}),
        (FrankaPandaPgiCfg, {}),
        (URRobotCfg, {}),
        (DexforceW1Cfg, {}),
    ],
)
def test_specified_robots_use_portable_joint_drive_semantics(
    cfg_type: type[RobotCfg],
    init_dict: dict,
) -> None:
    cfg = cfg_type.from_dict(init_dict)

    assert type(cfg.joint_drive_props) is JointDrivePropertiesCfg
    assert cfg.joint_drive_props.drive_type == "force"
    assert cfg.joint_drive_props.target_mode is None
    assert cfg.joint_drive_props._resolve_modes() == ("position_velocity", "force")


_PGI_FINGER_LINKS = {"gripper_finger1_link_1", "gripper_finger2_link_1"}


@pytest.mark.parametrize(
    ("cfg_type", "expected_links"),
    [
        (FrankaPandaCfg, {"fr3_leftfinger|fr3_rightfinger"}),
        (FrankaPandaPgiCfg, _PGI_FINGER_LINKS),
    ],
)
def test_franka_variants_own_newton_gripper_contact_defaults(
    cfg_type: type[RobotCfg], expected_links: set[str]
) -> None:
    cfg = cfg_type.from_dict({})

    group = cfg.link_attrs["newton_gripper_contacts"]
    assert set(group.link_names_expr) == expected_links
    assert isinstance(group.attrs.collision_props, NewtonCollisionPropertiesCfg)
    assert group.attrs.collision_props.condim == 4
    assert isinstance(group.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert group.attrs.material_props.ke == pytest.approx(40000.0)
    assert group.attrs.material_props.kd == pytest.approx(400.0)
    assert group.attrs.material_props.torsional_friction == pytest.approx(0.1)
    assert group.attrs.material_props.rolling_friction == pytest.approx(0.01)


def _assembled_link_and_joint_names(cfg: RobotCfg) -> tuple[list[str], list[str]]:
    tree = ET.parse(cfg.urdf_cfg.assemble_urdf())
    links = [link.get("name") for link in tree.findall("link")]
    joints = [
        joint.get("name")
        for joint in tree.findall("joint")
        if joint.get("type") != "fixed"
    ]
    return links, joints


def test_franka_panda_pgi_selectors_resolve_on_the_assembled_robot() -> None:
    from embodichain.utils.string import resolve_matching_names

    cfg = FrankaPandaPgiCfg.from_dict({})
    links, joints = _assembled_link_and_joint_names(cfg)

    # Spawn rejects any selector that matches nothing; every key must resolve.
    for group in cfg.link_attrs.values():
        resolve_matching_names(group.link_names_expr, links)
    _, contact_links = resolve_matching_names(
        cfg.link_attrs["newton_gripper_contacts"].link_names_expr, links
    )
    assert set(contact_links) == _PGI_FINGER_LINKS
    assert resolve_matching_names(cfg.control_parts["arm"], joints)[1] == [
        f"fr3_joint{i}" for i in range(1, 8)
    ]
    assert resolve_matching_names(cfg.control_parts["eef"], joints)[1] == [
        "gripper_finger1_joint_1"
    ]
    drives = cfg.joint_drive_props
    for rules in (drives.stiffness, drives.damping, drives.max_effort):
        resolve_matching_names(list(rules), joints)
    solver = cfg.solver_cfg["arm"]
    assert {solver.root_link_name, solver.end_link_name} <= set(links)
    assert len(cfg.init_qpos) == len(joints)
    assert cfg.init_qpos[-2:] == [0.0, 0.0]


def test_franka_panda_pgi_roundtrip_keeps_the_assembly_and_other_groups() -> None:
    wrist = {
        "link_names_expr": ["fr3_link7"],
        "attrs": {"material_props": {"static_friction": 0.5}},
    }
    cfg = FrankaPandaPgiCfg.from_dict(
        {
            "uid": "franka_pgi",
            "init_rot": [0.0, 0.0, 180.0],
            "solver_cfg": {"arm": {"tcp": np.eye(4).tolist()}},
            "link_attrs": {"wrist": wrist},
        }
    )

    restored = FrankaPandaPgiCfg.from_dict(cfg.to_dict())

    assert restored.to_dict() == cfg.to_dict()
    assert set(restored.link_attrs) == {"newton_gripper_contacts", "wrist"}
    assert restored.link_attrs["wrist"].link_names_expr == ["fr3_link7"]
    assert list(restored.init_rot) == [0.0, 0.0, 180.0]
    assert restored.solver_cfg["arm"].end_link_name == "fr3_link8"
    assert [
        Path(component["urdf_path"]).name
        for component in restored.urdf_cfg.components.values()
    ] == ["Panda.urdf", "DH_PGI_140_80.urdf"]

    # The merge contract removes one group without touching the others.
    without_contacts = FrankaPandaPgiCfg.from_dict(
        {"link_attrs": {"wrist": wrist, "newton_gripper_contacts": None}}
    )
    assert set(without_contacts.link_attrs) == {"wrist"}


def test_franka_panda_pgi_leaves_the_stock_panda_defaults_intact() -> None:
    before = FrankaPandaCfg.from_dict({}).to_dict()

    FrankaPandaPgiCfg.from_dict({})
    stock = FrankaPandaCfg.from_dict({})

    assert stock.to_dict() == before
    assert stock.control_parts["eef"] == ["fr3_finger_joint1", "fr3_finger_joint2"]
    assert stock.solver_cfg["arm"].end_link_name == "fr3_hand_tcp"
    assert stock.link_attrs["newton_gripper_contacts"].link_names_expr == [
        "fr3_leftfinger|fr3_rightfinger"
    ]
    assert stock._pk_urdf_path.endswith("PandaWithHand.urdf")


def test_franka_panda_pgi_kinematic_chain_matches_the_assembled_solver() -> None:
    pytest.importorskip("pytorch_kinematics")
    cfg = FrankaPandaPgiCfg.from_dict({})
    solver_cfg = cfg.solver_cfg["arm"]
    assert cfg._pk_urdf_path == cfg.urdf_cfg.components["arm"]["urdf_path"]

    chain = cfg.build_pk_serial_chain()["arm"]
    assert chain.get_joint_parameter_names() == cfg.control_parts["arm"]

    # The solver plans on the assembled robot; the chain must end at the same
    # flange so that composing it with the solver TCP gives the same pose.
    solver_cfg.urdf_path = cfg.urdf_cfg.assemble_urdf()
    solver_cfg.joint_names = cfg.control_parts["arm"]
    solver = solver_cfg.init_solver(device=torch.device("cpu"))
    qpos = torch.tensor(
        [cfg.init_qpos[:7], [0.3, -0.2, 0.1, -2.0, 0.2, 2.5, 0.4]],
        dtype=torch.float32,
    )
    tcp = torch.tensor(solver_cfg.tcp, dtype=torch.float32)
    torch.testing.assert_close(
        solver.get_fk(qpos), chain.forward_kinematics(qpos).get_matrix() @ tcp
    )


def test_robotcfg_save_to_file(tmp_path):
    cfg = _RoundTripCfg.from_dict({"variant": "b"})
    fp = tmp_path / "cfg.json"
    cfg.save_to_file(str(fp))
    import json

    loaded = json.loads(fp.read_text())
    assert loaded["variant"] == "b"
    assert loaded["uid"] == "roundtrip"


def test_robot_workspace_cfg_from_dict():
    """RobotCfg deserializes per-control-part workspace settings."""
    cfg = RobotCfg.from_dict(
        {
            "workspace_cfg": {
                "arm": {
                    "cache_path": "/tmp/workspace/results.npz",
                    "strategy": "point_uniform",
                    "voxel_size": 0.05,
                }
            }
        }
    )

    assert isinstance(cfg.workspace_cfg["arm"], RobotWorkspaceCfg)
    assert cfg.workspace_cfg["arm"].strategy == "point_uniform"
    assert cfg.workspace_cfg["arm"].voxel_size == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# PK drift-guard tests -- ensure build_pk_serial_chain DOF matches control_parts
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# URRobotCfg -- UR family (ur3 / ur3e / ur5 / ur5e / ur10 / ur10e)
# --------------------------------------------------------------------------- #

from embodichain.lab.sim.motion.solvers import URSolverCfg

UR_TYPES = ["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e"]


@pytest.mark.parametrize("robot_type", UR_TYPES)
def test_ur_robot_from_dict(robot_type):
    cfg = URRobotCfg.from_dict({"robot_type": robot_type})
    assert cfg.robot_type == robot_type
    assert isinstance(cfg.solver_cfg["arm"], URSolverCfg)
    assert cfg.solver_cfg["arm"].ur_type == robot_type
    assert cfg.solver_cfg["arm"].end_link_name == "ee_link"
    assert cfg.solver_cfg["arm"].root_link_name == "base_link"
    # one arm control part with 6 joints
    assert list(cfg.control_parts.keys()) == ["arm"]
    assert len(cfg.control_parts["arm"]) == 6

    snapshot = cfg.to_dict()
    assert snapshot["robot_type"] == robot_type
    restored = URRobotCfg.from_dict(snapshot)
    assert restored.robot_type == robot_type
    assert isinstance(restored.solver_cfg["arm"], URSolverCfg)


def test_ur_robot_default_type_is_ur10():
    cfg = URRobotCfg.from_dict({})
    assert cfg.robot_type == "ur10"


def test_ur_robot_max_effort_scales_with_size():
    """Larger UR variants have larger max_effort defaults."""
    ur3 = URRobotCfg.from_dict({"robot_type": "ur3"})
    ur5 = URRobotCfg.from_dict({"robot_type": "ur5"})
    ur10 = URRobotCfg.from_dict({"robot_type": "ur10"})
    eff = lambda c: c.joint_drive_props.max_effort["arm"]  # noqa: E731
    assert eff(ur3) < eff(ur5) < eff(ur10)


@pytest.mark.parametrize(
    ("cfg_type", "init_dict", "parts"),
    [
        pytest.param(DexforceW1Cfg, {}, ("left_arm", "right_arm"), id="w1"),
        pytest.param(CobotMagicCfg, {}, ("left_arm", "right_arm"), id="cobotmagic"),
        *[
            pytest.param(URRobotCfg, {"robot_type": kind}, ("arm",), id=kind)
            for kind in UR_TYPES
        ],
    ],
)
def test_pk_chain_dof_matches_control_parts(cfg_type, init_dict, parts) -> None:
    pytest.importorskip("pytorch_kinematics")
    cfg = cfg_type.from_dict(init_dict)
    try:
        chains = cfg.build_pk_serial_chain()
    except FileNotFoundError as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for part in parts:
        assert len(chains[part].get_joint_parameter_names()) == len(
            cfg.control_parts[part]
        ), f"{part}: PK chain DOF drifted from control_parts"


def test_ur_robot_unknown_type_raises():
    with pytest.raises(ValueError):
        URRobotCfg.from_dict({"robot_type": "ur99"})
