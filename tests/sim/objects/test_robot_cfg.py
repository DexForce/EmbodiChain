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

import numpy as np
import pytest

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
)
from embodichain.lab.sim.workspace import RobotWorkspaceCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.robots.dexforce_w1.params import W1ArmKineParams
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
    DexforceW1Type,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.hand_specs import (
    get_default_w1_hand_version,
    get_w1_hand_spec,
)
from embodichain.lab.sim.robots.dexforce_w1.specs import get_w1_version_spec
from embodichain.lab.sim.robots.dexforce_w1.utils import (
    build_dexforce_w1_assembly_urdf_cfg,
    build_dexforce_w1_control_parts,
)
from embodichain.lab.sim.solvers import SRSSolverCfg
from embodichain.utils import configclass
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg


def _mock_w1_asset_paths(monkeypatch, tmp_path):
    import embodichain.lab.sim.cfg as sim_cfg
    from embodichain.lab.sim.robots.dexforce_w1 import utils as w1_utils

    def resolve(path):
        resolved = tmp_path / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text("<robot name='w1' />")
        return str(resolved)

    monkeypatch.setattr(w1_utils, "get_data_path", resolve)
    monkeypatch.setattr(sim_cfg, "get_data_path", resolve)


def test_dexforce_w1_roundtrip():
    cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1", "version": "v021"})
    d = cfg.to_dict()
    assert d["uid"] == "dexforce_w1"
    cfg2 = DexforceW1Cfg.from_dict(d)
    assert cfg2.uid == "dexforce_w1"
    assert cfg2.version == DexforceW1Version.V021


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
    from embodichain.lab.sim.robots.dexforce_w1 import utils as w1_utils

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
    from embodichain.lab.sim.robots.dexforce_w1 import utils as w1_utils

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
        self.drive_pros = JointDrivePropertiesCfg(
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
    assert cfg2.drive_pros.stiffness == {"J[1-2]": 1e4}


from embodichain.lab.sim.robots.cobotmagic import CobotMagicCfg
from embodichain.lab.sim.solvers import OPWSolverCfg


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

    d = cfg.to_dict()
    assert d["uid"] == "CobotMagic"
    cfg2 = CobotMagicCfg.from_dict(d)
    assert cfg2.uid == "CobotMagic"
    assert cfg2.control_parts == cfg.control_parts
    assert isinstance(cfg2.solver_cfg["left_arm"], OPWSolverCfg)


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


def _dof_of_pk_chain(chain) -> int:
    """Number of actuated joints in a pk.SerialChain."""
    return len(chain.get_joint_parameter_names())


def test_dexforce_w1_pk_dof_matches_control_parts():
    pytest.importorskip("pytorch_kinematics")
    cfg = DexforceW1Cfg.from_dict({})
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for arm in ("left_arm", "right_arm"):
        assert _dof_of_pk_chain(chains[arm]) == len(
            cfg.control_parts[arm]
        ), f"{arm}: PK chain DOF drifted from control_parts"


def test_cobotmagic_pk_dof_matches_control_parts():
    pytest.importorskip("pytorch_kinematics")
    cfg = CobotMagicCfg.from_dict({})
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for arm in ("left_arm", "right_arm"):
        assert _dof_of_pk_chain(chains[arm]) == len(
            cfg.control_parts[arm]
        ), f"{arm}: PK chain DOF drifted from control_parts"


# --------------------------------------------------------------------------- #
# URRobotCfg -- UR family (ur3 / ur3e / ur5 / ur5e / ur10 / ur10e)
# --------------------------------------------------------------------------- #

from embodichain.lab.sim.robots.ur_robot import URRobotCfg
from embodichain.lab.sim.solvers import URSolverCfg

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


def test_ur_robot_default_type_is_ur10():
    cfg = URRobotCfg.from_dict({})
    assert cfg.robot_type == "ur10"


@pytest.mark.parametrize("robot_type", UR_TYPES)
def test_ur_robot_roundtrip(robot_type):
    cfg = URRobotCfg.from_dict({"robot_type": robot_type})
    d = cfg.to_dict()
    assert d["robot_type"] == robot_type
    cfg2 = URRobotCfg.from_dict(d)
    assert cfg2.robot_type == robot_type
    assert isinstance(cfg2.solver_cfg["arm"], URSolverCfg)


def test_ur_robot_max_effort_scales_with_size():
    """Larger UR variants have larger max_effort defaults."""
    ur3 = URRobotCfg.from_dict({"robot_type": "ur3"})
    ur5 = URRobotCfg.from_dict({"robot_type": "ur5"})
    ur10 = URRobotCfg.from_dict({"robot_type": "ur10"})
    eff = lambda c: c.drive_pros.max_effort["arm"]  # noqa: E731
    assert eff(ur3) < eff(ur5) < eff(ur10)


@pytest.mark.parametrize("robot_type", UR_TYPES)
def test_ur_robot_pk_dof_matches_control_parts(robot_type):
    pytest.importorskip("pytorch_kinematics")
    cfg = URRobotCfg.from_dict({"robot_type": robot_type})
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    assert _dof_of_pk_chain(chains["arm"]) == len(
        cfg.control_parts["arm"]
    ), "arm: PK chain DOF drifted from control_parts"


def test_ur_robot_unknown_type_raises():
    with pytest.raises(ValueError):
        URRobotCfg.from_dict({"robot_type": "ur99"})
