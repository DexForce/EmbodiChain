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

"""Verify RoboParty catalog defaults against the pinned upstream models."""

from __future__ import annotations

from math import degrees
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from embodichain.robot_registry import resolve_robot_cfg
from embodichain.extensions.robots._catalog import ROBOT_CONFIGS

# Source finger12 and finger13 limits intersect through the thumb mimic relation.
_THUMB_SOURCE_LIMIT = 1.4
_THUMB_FOLLOWER_LIMIT = 1.13
_THUMB_COMMAND_RANGE = (0.01623466, 1.23259634)


@pytest.mark.no_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "robot_id,variant,root_link,independent_dof,mimic_count",
    [
        ("RoboPartyRPO", None, "base_link", 23, 0),
        ("RoboPartyRP1", None, "base_link", 28, 0),
        ("RoboPartyRP1", "24dof", "base_link", 24, 0),
        ("RoboPartyRP1", "hand", "base_link", 40, 10),
        ("RoboPartyRP1", "hand_fixed", "base_link", 28, 0),
        ("RoboPartyRPHand", "left", "lh_base_link", 6, 5),
        ("RoboPartyRPHand", "right", "rh_base_link", 6, 5),
    ],
)
def test_pinned_description_matches_robot_preset(
    robot_id: str,
    variant: str | None,
    root_link: str,
    independent_dof: int,
    mimic_count: int,
) -> None:
    """Check the public path through the real asset cache and config loader."""
    cfg = resolve_robot_cfg(robot_id, variant=variant)
    cfg.validate()
    urdf = Path(cfg.fpath)
    source = ET.parse(urdf)
    joints = source.findall("joint")
    moving = [joint for joint in joints if joint.attrib["type"] != "fixed"]
    mimics = {
        joint.attrib["name"]: mimic
        for joint in moving
        if (mimic := joint.find("mimic")) is not None
    }
    independent = {joint.attrib["name"] for joint in moving} - mimics.keys()
    controlled = {joint for joints in cfg.control_parts.values() for joint in joints}
    assert controlled == independent
    assert len(independent) == independent_dof
    assert len(mimics) == mimic_count

    spec = ROBOT_CONFIGS[robot_id]
    if variant is not None:
        spec = spec["variants"][variant]
    assert spec["dof"] == independent_dof
    assert spec["control_parts"] == {
        part: len(names) for part, names in cfg.control_parts.items()
    }
    assert spec.get("mimic_joints", {}) == {
        joint: mimic.attrib["joint"] for joint, mimic in mimics.items()
    }

    children = {joint.find("child").attrib["link"] for joint in joints}
    assert {link.attrib["name"] for link in source.findall("link")} - children == {
        root_link
    }
    for mesh in source.findall(".//mesh"):
        filename = (urdf.parent / mesh.attrib["filename"]).resolve()
        assert filename.is_relative_to(urdf.parent)
        assert filename.is_file()
    assert cfg.root_props.fixed_base == (robot_id == "RoboPartyRPHand")
    assert not cfg.build_pk_chain
    assert cfg.build_pk_serial_chain() == {}

    assert len(cfg.init_qpos) == len(moving)
    assert cfg.init_qpos_joint_names == [joint.attrib["name"] for joint in moving]
    qpos = dict(zip(cfg.init_qpos_joint_names, cfg.init_qpos, strict=True))
    for joint in moving:
        limit = joint.find("limit")
        assert (
            float(limit.attrib["lower"])
            <= qpos[joint.attrib["name"]]
            <= float(limit.attrib["upper"])
        )
    for joint, mimic in mimics.items():
        assert qpos[joint] == pytest.approx(
            float(mimic.attrib["multiplier"]) * qpos[mimic.attrib["joint"]]
            + float(mimic.attrib["offset"])
        )
    thumb_joints = qpos.keys() & {"lh_finger12", "rh_finger12"}
    if thumb_joints:
        assert cfg.qpos_limits.keys() == thumb_joints
        for thumb in thumb_joints:
            assert cfg.qpos_limits[thumb] == pytest.approx(_THUMB_COMMAND_RANGE)

    body_joint_names = cfg.control_parts.get("locomotion", ())
    for joint in moving:
        name = joint.attrib["name"]
        if name in body_joint_names:
            limit = joint.find("limit")
            assert cfg.joint_drive_props.max_effort[name] == float(
                limit.attrib["effort"]
            )
            assert cfg.joint_drive_props.max_velocity[name] == float(
                limit.attrib["velocity"]
            )

    # A serialized preset preserves the selected body/side and works without
    # re-resolving the default source.
    override = cfg.to_dict()
    assert type(cfg).from_dict(override).to_dict() == override


@pytest.mark.no_sim
def test_local_urdf_limits_survive_a_partial_drive_override(tmp_path: Path) -> None:
    """Apply a regex override while retaining unaffected source limits."""
    urdf = tmp_path / "rpo.urdf"
    urdf.write_text(
        """<robot name="local_rpo">
  <link name="base"/>
  <link name="left"/>
  <link name="right"/>
  <joint name="left_joint" type="revolute">
    <parent link="base"/><child link="left"/>
    <limit lower="-1" upper="1" effort="60" velocity="10"/>
  </joint>
  <joint name="right_joint" type="revolute">
    <parent link="base"/><child link="right"/>
    <limit lower="-1" upper="1" effort="70" velocity="12"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    cfg = resolve_robot_cfg(
        "RoboPartyRPO",
        {"fpath": str(urdf), "joint_drive_props": {"max_velocity": {"left_.*": 5.0}}},
    )
    assert cfg.joint_drive_props.max_velocity == {"left_.*": 5.0, "right_joint": 12.0}
    assert cfg.joint_drive_props.max_effort == {"left_joint": 60.0, "right_joint": 70.0}


_RIGHT_HAND_PARENTS = {
    "11": None,
    "12": "11",
    "13": "12",
    "21": None,
    "22": "21",
    "31": None,
    "32": "31",
    "41": None,
    "42": "41",
    "51": None,
    "52": "51",
}
_RIGHT_HAND_MIMICS = {
    "13": (0.929, -0.015082),
    "22": (0.8847, 0.038265),
    "32": (0.8485, 0.035946),
    "42": (0.8321, 0.043170),
    "52": (0.7349, 0.035218),
}
_HAND_UPPER_LIMITS = {"12": _THUMB_SOURCE_LIMIT, "13": _THUMB_FOLLOWER_LIMIT}


@pytest.fixture
def right_hand_urdf(tmp_path: Path) -> Path:
    """Author the RP Hand's movable tree with child joints declared first."""
    tree = ET.Element("robot", name="local_right_hand")
    names = ["rh_base_link", *(f"rh_finger{i}_link" for i in _RIGHT_HAND_PARENTS)]
    for name in names:
        link = ET.SubElement(tree, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="0.1")
        ET.SubElement(
            inertial,
            "inertia",
            ixx="0.001",
            ixy="0",
            ixz="0",
            iyy="0.001",
            iyz="0",
            izz="0.001",
        )
    for finger, parent in reversed(_RIGHT_HAND_PARENTS.items()):
        joint = ET.SubElement(tree, "joint", name=f"rh_finger{finger}", type="revolute")
        parent_link = "rh_base_link" if parent is None else f"rh_finger{parent}_link"
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=f"rh_finger{finger}_link")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(
            joint,
            "limit",
            lower="0",
            upper=str(_HAND_UPPER_LIMITS.get(finger, 1.75)),
            effort="0",
            velocity="0",
        )
        if finger in _RIGHT_HAND_MIMICS:
            multiplier, offset = _RIGHT_HAND_MIMICS[finger]
            ET.SubElement(
                joint,
                "mimic",
                joint=f"rh_finger{parent}",
                multiplier=str(multiplier),
                offset=str(offset),
            )
    path = tmp_path / "right_hand.urdf"
    ET.ElementTree(tree).write(path, encoding="utf-8")
    return path


@pytest.fixture
def right_hand_usd(tmp_path: Path) -> Path:
    """Author the same hand joint topology as a local USD articulation."""
    from physx_usd_schemas import PhysxSchema
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "right_hand.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    assembly = UsdGeom.Xform.Define(stage, "/Hand")
    stage.SetDefaultPrim(assembly.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(assembly.GetPrim())
    names = ["rh_base_link", *(f"rh_finger{i}_link" for i in _RIGHT_HAND_PARENTS)]
    for name in names:
        body = UsdGeom.Xform.Define(stage, f"/Hand/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.1)
        cube = UsdGeom.Cube.Define(stage, f"/Hand/{name}/mesh")
        cube.CreateSizeAttr(0.02)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    for finger, parent in _RIGHT_HAND_PARENTS.items():
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Hand/rh_finger{finger}")
        parent_link = "rh_base_link" if parent is None else f"rh_finger{parent}_link"
        joint.CreateBody0Rel().SetTargets([f"/Hand/{parent_link}"])
        joint.CreateBody1Rel().SetTargets([f"/Hand/rh_finger{finger}_link"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(degrees(_HAND_UPPER_LIMITS.get(finger, 1.75)))
        if finger in _RIGHT_HAND_MIMICS:
            multiplier, offset = _RIGHT_HAND_MIMICS[finger]
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(joint.GetPrim(), "rotX")
            mimic.CreateReferenceJointRel().SetTargets([f"/Hand/rh_finger{parent}"])
            mimic.CreateReferenceJointAxisAttr("rotX")
            mimic.CreateGearingAttr(-multiplier)
            mimic.CreateOffsetAttr(-degrees(offset))
    stage.GetRootLayer().Save()
    return path


@pytest.mark.requires_sim
@pytest.mark.parametrize("file_field", ("fpath", "source"))
def test_rp_hand_loads_a_local_usd_source(
    right_hand_usd: Path, file_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both file selectors use the local USD articulation and its mimic metadata."""
    from embodichain.data import constants, dataset
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    def unexpected_download(asset: str) -> None:
        pytest.fail(f"Local USD unexpectedly requested a dataset: {asset!r}.")

    override = (
        {"fpath": str(right_hand_usd)}
        if file_field == "fpath"
        else {"source": {"file_path": right_hand_usd.name, "source_prim_path": "/Hand"}}
    )
    with monkeypatch.context() as paths:
        paths.setattr(
            constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(right_hand_usd.parent)
        )
        paths.setattr(dataset, "get_data_class", unexpected_download)
        cfg = config_to_cfg(
            {
                "id": "EmbodiedEnv-v1",
                "physics": "default",
                "env": {},
                "robot": {"class_type": "RoboPartyRPHand", **override},
            }
        ).robot
        if file_field == "source":
            assert cfg.fpath is None
            assert cfg.source.file_path == str(right_hand_usd)
            assert cfg.source.source_prim_path == "/Hand"
        else:
            assert cfg.source is None
            assert cfg.fpath == str(right_hand_usd)
        cfg = type(cfg).from_dict(cfg.to_dict())
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            startup_summary="off",
        )
    )
    try:
        robot = sim.add_robot(cfg)
        sim.prepare()
        assert set(robot.joint_names) == {f"rh_finger{i}" for i in _RIGHT_HAND_PARENTS}
        assert len(robot.get_joint_ids("eef")) == 6
        assert len(robot.mimic_ids) == 5
        thumb = robot.joint_names.index("rh_finger12")
        assert robot.get_qpos()[0, thumb].item() == pytest.approx(0.02)
        assert robot.get_qpos_limits(joint_ids=[thumb])[0, 0].tolist() == pytest.approx(
            _THUMB_COMMAND_RANGE
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.no_sim
def test_rp_hand_accepts_an_explicit_thumb_limit_rule(right_hand_urdf: Path) -> None:
    """A regex selecting the thumb replaces its exact preset rule."""
    limits = {"rh_finger1[2]": [0.1, 0.5]}
    cfg = resolve_robot_cfg(
        "RoboPartyRPHand", {"fpath": str(right_hand_urdf), "qpos_limits": limits}
    )
    assert cfg.qpos_limits == limits


@pytest.mark.requires_sim
@pytest.mark.parametrize(
    "physics,explicit_override_backend",
    [("default", False), ("newton", False), ("newton", True)],
)
def test_rp_hand_reset_by_name_preserves_partial_joint_limits(
    right_hand_urdf: Path, physics: str, explicit_override_backend: bool
) -> None:
    """Keep untouched joint limits and restore the named thumb pose."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import physics_cfg_for_backend

    drive = {"max_velocity": {"rh_finger12": 2.0}}
    if explicit_override_backend:
        drive["backend"] = "newton"
    cfg = resolve_robot_cfg(
        "RoboPartyRPHand",
        {
            "fpath": str(right_hand_urdf),
            "joint_drive_props": drive,
            "qpos_limits": {"rh_finger11": [0.1, 0.4]},
        },
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            physics_cfg=physics_cfg_for_backend(physics),
            startup_summary="off",
        )
    )
    try:
        robot = sim.add_robot(cfg)
        sim.prepare()
        names = robot.joint_names
        speeds = dict(zip(names, robot.get_qvel_limits()[0].tolist(), strict=True))
        assert speeds == {name: 2.0 if name == "rh_finger12" else 5.0 for name in names}
        robot.set_qpos(torch.full_like(robot.get_qpos(), 0.2), target=False)
        robot.reset()
        qpos = dict(zip(names, robot.get_qpos()[0].tolist(), strict=True))
        assert qpos["rh_finger12"] == pytest.approx(0.02)
        assert qpos["rh_finger21"] == pytest.approx(0.0)
        for follower, (multiplier, offset) in _RIGHT_HAND_MIMICS.items():
            leader = _RIGHT_HAND_PARENTS[follower]
            expected = multiplier * qpos[f"rh_finger{leader}"] + offset
            assert qpos[f"rh_finger{follower}"] == pytest.approx(expected, abs=1e-6)

        other_thumb_joint, thumb = (
            names.index(name) for name in ("rh_finger11", "rh_finger12")
        )
        actual_limits = robot.get_qpos_limits(joint_ids=[other_thumb_joint, thumb])[0]
        assert actual_limits[0].tolist() == pytest.approx([0.1, 0.4])
        limits = actual_limits[1].tolist()
        assert limits == pytest.approx(_THUMB_COMMAND_RANGE)
        for command, limit in zip((0.0, _THUMB_SOURCE_LIMIT), limits, strict=True):
            robot.set_qpos(torch.tensor([[command]]), joint_ids=[thumb])
            assert robot.get_qpos(target=True)[0, thumb].item() == pytest.approx(limit)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.no_sim
@pytest.mark.parametrize("solver", (None, "dexuni"))
def test_rp1_thumb_group_override_preserves_other_finger_gains(
    right_hand_urdf: Path, solver: str | None
) -> None:
    """Compile a custom thumb group alongside the preset finger drives."""
    from dexsim.engine import JointType
    from dexsim.spawn import ArticulationDesc, JointDesc

    from embodichain.lab.sim.spawn.descriptors import _compile_joint_properties

    cfg = resolve_robot_cfg(
        "RoboPartyRP1",
        {
            "fpath": str(right_hand_urdf),
            "control_parts": {"thumb": ["rh_finger12"]},
            "joint_drive_props": {"stiffness": {"thumb": 7.0}},
        },
        variant="hand",
    )
    desc = ArticulationDesc(
        name="rp1",
        joints=[
            JointDesc(
                name=joint.attrib["name"],
                parent_link_name=joint.find("parent").attrib["link"],
                child_link_name=joint.find("child").attrib["link"],
                joint_type=JointType.REVOLUTE,
            )
            for joint in ET.parse(right_hand_urdf).findall("joint")
        ],
    )
    drives, _, _, _ = _compile_joint_properties(desc, cfg, newton_solver_type=solver)
    for name in cfg.control_parts["right_eef"]:
        default, newton = drives[name]
        expected = 7.0 if name == "rh_finger12" else 3.0
        assert default.stiffness == newton.target_ke == expected


@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_rp1_newton_index_followers_track_from_first_step(device: str) -> None:
    """The source-coupled fingers stay inside their limits during a command step."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg

    cfg = resolve_robot_cfg("RoboPartyRP1", variant="hand")
    physics_cfg = NewtonPhysicsCfg(
        solver_cfg={
            "solver_type": "dexuni",
            "mujoco_options": {"njmax": 512, "nconmax": 256},
        }
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=device,
            physics_cfg=physics_cfg,
            startup_summary="off",
        )
    )
    try:
        robot = sim.add_robot(cfg)
        sim.prepare()
        names = robot.joint_names
        leaders = [names.index(f"{side}_finger21") for side in ("lh", "rh")]
        followers = [names.index(f"{side}_finger22") for side in ("lh", "rh")]
        robot.set_qpos(torch.full((1, 2), 0.25, device=device), joint_ids=leaders)
        for _ in range(8):
            sim.update(step=1)
            qpos = robot.get_qpos()
            measured = qpos[:, followers]
            assert torch.all((0 <= measured) & (measured <= 1.75))
            torch.testing.assert_close(
                measured, 0.8847 * qpos[:, leaders] + 0.038265, atol=0.02, rtol=0
            )
        assert torch.all(qpos[:, leaders] > 0.0)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "slow or not slow"]))
