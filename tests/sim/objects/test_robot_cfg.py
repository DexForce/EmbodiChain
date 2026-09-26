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
from embodichain.lab.sim.motion.solvers import SRSSolverCfg
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


from embodichain.lab.sim.robots.cobotmagic import CobotMagicCfg
from embodichain.lab.sim.robots import AlohaMiniCfg, TianjiMarvinCfg
from embodichain.lab.sim.robots.franka_panda import FrankaPandaCfg
from embodichain.lab.sim.robots.ur_robot import URRobotCfg
from embodichain.lab.sim.motion.solvers import OPWSolverCfg


def test_aloha_mini_control_parts_match_source_joints():
    cfg = AlohaMiniCfg.from_dict({})
    assert cfg.uid == "AlohaMini"
    assert cfg.urdf_cfg is None
    assert cfg.fpath.endswith("AlohaMini/alohamini2pro.urdf")
    assert {part: len(joints) for part, joints in cfg.control_parts.items()} == {
        "left_arm": 6,
        "right_arm": 6,
        "left_hand": 1,
        "right_hand": 1,
        "torso": 1,
    }
    assert cfg.control_parts["torso"] == ["vertical_move"]
    assert cfg.control_parts["left_hand"] == ["left_gripper"]
    assert cfg.control_parts["right_hand"] == ["right_gripper"]
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
    assert cfg.solver_cfg["right_arm"].num_samples == 30
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
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

    cfg = AlohaMiniCfg.from_dict({})
    chains = cfg.build_pk_serial_chain()
    assert set(cfg.solver_cfg) == {"left_arm", "right_arm", "torso"}
    for part, solver_cfg in cfg.solver_cfg.items():
        assert isinstance(solver_cfg, PytorchSolverCfg)
        assert solver_cfg.is_only_position_constraint == (part == "torso")
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


@pytest.mark.parametrize("with_gripper", [True, False])
def test_tianji_marvin_control_parts_match_source_joints(with_gripper):
    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
    filename = "robot_with_ee_acd.urdf" if with_gripper else "robot_acd.urdf"
    assert cfg.uid == "TianjiMarvin"
    assert cfg.fpath.endswith(f"TianjiMarvin/{filename}")
    assert cfg.urdf_cfg is None
    expected_dofs = {"left_arm": 7, "right_arm": 7}
    if with_gripper:
        expected_dofs.update(left_hand=2, right_hand=2)
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
            leader, follower = cfg.control_parts[f"{side}_hand"]
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
def test_tianji_marvin_serial_chains_and_solvers_match_source(with_gripper):
    from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

    cfg = TianjiMarvinCfg.from_dict({"with_gripper": with_gripper})
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
        torch.testing.assert_close(
            solver.get_fk(qpos), chain.forward_kinematics(qpos).get_matrix()
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
            for part in ("left_hand", "right_hand"):
                robot.set_qpos(target, name=part, target=True)
            sim.update(step=100)
            for part in ("left_hand", "right_hand"):
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


@pytest.mark.parametrize(
    ("cfg_type", "init_dict"),
    [
        (CobotMagicCfg, {}),
        (FrankaPandaCfg, {}),
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


def test_franka_panda_owns_newton_fingertip_contact_defaults() -> None:
    cfg = FrankaPandaCfg.from_dict({})

    group = cfg.link_attrs["newton_gripper_contacts"]
    assert group.link_names_expr == ["fr3_leftfinger|fr3_rightfinger"]
    assert isinstance(group.attrs.collision_props, NewtonCollisionPropertiesCfg)
    assert group.attrs.collision_props.condim == 4
    assert isinstance(group.attrs.material_props, NewtonRigidBodyMaterialCfg)
    assert group.attrs.material_props.ke == pytest.approx(40000.0)
    assert group.attrs.material_props.kd == pytest.approx(400.0)
    assert group.attrs.material_props.torsional_friction == pytest.approx(0.1)
    assert group.attrs.material_props.rolling_friction == pytest.approx(0.01)


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
    eff = lambda c: c.joint_drive_props.max_effort["arm"]  # noqa: E731
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
