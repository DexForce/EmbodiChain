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

"""Standalone and mounted Revo1 hands share source names and mimic behavior."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from embodichain.robot_registry import list_robot_cfgs, resolve_robot_cfg

_ACTIVE_SUFFIXES = ("THUMB1", "THUMB2", "INDEX", "MIDDLE", "RING", "PINKY")
# The shipped source has a nonzero rest offset at each four-finger PIP.
_PIP_OFFSETS = {
    "INDEX": 0.19373154697137057,
    "MIDDLE": 0.1832595714594046,
    "RING": 0.16580627893946132,
    "PINKY": 0.20245819323134223,
}


@pytest.fixture
def local_hands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Use original synthetic URDFs with the shipped active/mimic naming contract."""
    from embodichain_robots.brainco.revo1 import cfg as revo1

    paths = {}
    for side in ("left", "right"):
        prefix = side.upper()
        root = ET.Element("robot", name=f"synthetic_{side}_hand")
        base = f"{side}_hand_base"
        ET.SubElement(root, "link", name=base)
        for suffix in _ACTIVE_SUFFIXES:
            child = f"{side}_{suffix.lower()}"
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(
                root, "joint", name=f"{prefix}_HAND_{suffix}", type="revolute"
            )
            ET.SubElement(joint, "parent", link=base)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(
                joint, "limit", lower="0", upper="1", effort="0", velocity="0"
            )
            if suffix in _PIP_OFFSETS:
                distal = f"{child}_dist"
                ET.SubElement(root, "link", name=distal)
                pip = ET.SubElement(
                    root, "joint", name=f"{prefix}_{suffix}_PIP", type="revolute"
                )
                ET.SubElement(pip, "parent", link=child)
                ET.SubElement(pip, "child", link=distal)
                ET.SubElement(pip, "axis", xyz="0 0 1")
                ET.SubElement(
                    pip, "limit", lower="0", upper="2", effort="0", velocity="0"
                )
                ET.SubElement(
                    pip,
                    "mimic",
                    joint=f"{prefix}_HAND_{suffix}",
                    multiplier="1.2",
                    offset=str(_PIP_OFFSETS[suffix]),
                )
        path = tmp_path / f"{side}.urdf"
        ET.ElementTree(root).write(path)
        paths[side] = path

    def local_path(reference: str) -> str:
        assert reference.startswith("BrainCoHandRevo1/BrainCo")
        side = "left" if "LeftHand" in reference else "right"
        return str(paths[side])

    monkeypatch.setattr(revo1.data, "get_data_path", local_path)
    return paths


@pytest.mark.no_sim
@pytest.mark.parametrize("side", ("left", "right"))
def test_revo1_standalone_preserves_native_control_contract(
    local_hands: dict[str, Path], side: str
) -> None:
    """Six native actuator names survive config decoding and JSON serialization."""
    from embodichain_robots import BrainCoRevo1Cfg

    original = resolve_robot_cfg("BrainCoRevo1", variant=side)
    cfg = BrainCoRevo1Cfg.from_dict(json.loads(json.dumps(original.to_dict())))
    expected_names = [f"{side.upper()}_HAND_{suffix}" for suffix in _ACTIVE_SUFFIXES]
    assert cfg.side == side
    assert cfg.fpath == str(local_hands[side])
    assert cfg.control_parts == {"eef": expected_names}
    assert cfg.root_props.fixed_base is True
    # Native hand meshes straddle base Z=0; the standalone preset clears ground.
    assert cfg.init_pos == [0.0, 0.0, 0.3]
    assert cfg.solver_cfg == {}
    assert cfg.build_pk_serial_chain() == {}
    assert cfg.joint_drive_props.stiffness == {"eef": 100.0}
    assert cfg.joint_drive_props.damping == {"eef": 10.0}
    assert cfg.joint_drive_props.max_effort == {"eef": 1000.0}
    assert cfg.joint_drive_props.max_velocity == {"eef": 1.0}

    source = ET.parse(cfg.fpath)
    active = [
        j.get("name")
        for j in source.findall("joint")
        if j.get("type") != "fixed" and j.find("mimic") is None
    ]
    assert active == expected_names
    assert len(source.findall(".//mimic")) == 4


@pytest.mark.no_sim
def test_revo1_defaults_overrides_and_instances_are_isolated(
    local_hands: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller overrides neither download the catalog asset nor mutate other configs."""
    from embodichain_robots.brainco.revo1 import cfg as revo1
    from embodichain_robots._catalog import ROBOT_CONFIGS

    assert "BrainCoRevo1" in list_robot_cfgs()
    metadata = ROBOT_CONFIGS["BrainCoRevo1"]
    assert metadata["vendor"] == "BrainCo"
    assert metadata["morphology"] == "dexterous_hand"
    assert metadata["arm_count"] == 0
    assert metadata["supports_dual_arm"] is False
    assert metadata["default_variant"] == "right"
    for side in ("left", "right"):
        profile = metadata["variants"][side]
        assert profile["dof"] == 6
        assert profile["control_parts"] == {"eef": 6}
        assert profile["defaults"] == {"side": side}
        assert profile["mimic_joints"] == {
            f"{side.upper()}_{finger}_PIP": f"{side.upper()}_HAND_{finger}"
            for finger in _PIP_OFFSETS
        }
    default = resolve_robot_cfg("BrainCoRevo1")
    assert default.side == "right"
    default.control_parts["eef"].append("LOCAL_ONLY")
    assert "LOCAL_ONLY" not in resolve_robot_cfg("BrainCoRevo1").control_parts["eef"]

    def unexpected_download(reference: str) -> str:
        raise AssertionError(f"Local fpath must bypass downloads: {reference}")

    monkeypatch.setattr(revo1.data, "get_data_path", unexpected_download)
    custom = resolve_robot_cfg(
        "BrainCoRevo1",
        {
            "side": "left",
            "fpath": str(local_hands["left"]),
            "uid": "custom_hand",
            "control_parts": {"hand": ["LEFT_HAND_INDEX"]},
            "joint_drive_props": {"max_velocity": {"hand": 0.5}},
        },
        variant="right",
    )
    restored = revo1.BrainCoRevo1Cfg.from_dict(custom.to_dict())
    assert restored.side == "left"
    assert restored.uid == "custom_hand"
    assert restored.fpath == str(local_hands["left"])
    assert restored.control_parts["hand"] == ["LEFT_HAND_INDEX"]
    assert restored.joint_drive_props.max_velocity["hand"] == 0.5


@pytest.mark.no_sim
@pytest.mark.parametrize("side", ("LEFT", "both", "", None))
def test_revo1_rejects_unknown_side(side: object) -> None:
    from embodichain_robots import BrainCoRevo1Cfg

    with pytest.raises((ValueError, TypeError), match="side"):
        BrainCoRevo1Cfg.from_dict({"side": side})


@pytest.mark.no_sim
@pytest.mark.parametrize("side", ("left", "right"))
def test_w1_revo1_adapter_retains_names_frames_and_mount(side: str) -> None:
    """The shared model extraction preserves the established W1 assembly ABI."""
    from embodichain_robots.dexforce.w1.hand_specs import get_w1_hand_spec

    spec = get_w1_hand_spec("BRAINCO_HAND", "v021").for_side(side)
    title = side.title()
    assert spec.urdf_path == (
        f"BrainCoHandRevo1/BrainCo{title}Hand/BrainCo{title}Hand.urdf"
    )
    assert spec.joint_names == tuple(
        f"{side.upper()}_HAND_{suffix}" for suffix in _ACTIVE_SUFFIXES
    )
    assert spec.end_link_name == f"{side}_hand_base"
    assert spec.root_link_name == f"{side}_thumb_dist"
    # Independent expected rotation for the existing Rx(90), Rz(left:180) mount.
    expected = (
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        if side == "left"
        else [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    np.testing.assert_allclose(spec.attach_xpos, expected, atol=1e-15)


@pytest.mark.requires_sim
class TestRevo1Simulation:
    """Default-backend creation, joint commands, mimic tracking, and reset."""

    def setup_method(self) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

        self.sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=1)
        )

    def teardown_method(self) -> None:
        from embodichain.lab.sim import SimulationManager

        if hasattr(self, "sim"):
            self.sim.destroy()
            del self.sim
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    @pytest.mark.parametrize("selection", ("left", "right", "w1"))
    def test_standalone_and_mounted_hand_control(self, selection: str) -> None:
        if selection == "w1":
            cfg = resolve_robot_cfg("DexforceW1", variant="v021/BrainCo")
            part = "right_eef"
        else:
            cfg = resolve_robot_cfg("BrainCoRevo1", variant=selection)
            part = "eef"
            source_joints = {
                joint.get("name")
                for joint in ET.parse(cfg.fpath).getroot().findall("joint")
            }
            assert set(cfg.control_parts[part]) <= source_joints, cfg.fpath
        robot = self.sim.add_robot(cfg=cfg)
        self.sim.prepare()
        initial_qpos = robot.get_qpos().clone()
        initial_targets = robot.get_qpos(target=True).clone()
        initial_part = robot.get_qpos(name=part).clone()
        if selection != "w1":
            torch.testing.assert_close(initial_part, torch.zeros_like(initial_part))
        assert len(robot.get_joint_ids(part, remove_mimic=True)) == 6
        assert len(robot.mimic_ids) == (8 if selection == "w1" else 4)

        # Move index/middle/ring/pinky only, keeping the existing thumb pose.
        target = initial_part.clone()
        target[:, 2:] += 0.15
        selected_ids = robot.get_joint_ids(part)
        untouched = [
            index
            for index in range(robot.dof)
            if index not in selected_ids and index not in robot.mimic_ids
        ]
        robot.set_qpos(target, name=part)
        torch.testing.assert_close(
            robot.get_qpos(target=True)[:, untouched], initial_targets[:, untouched]
        )
        initial_error = torch.linalg.vector_norm(initial_part - target)
        self.sim.update(step=100)
        measured = robot.get_qpos(name=part)
        assert bool(torch.isfinite(robot.get_qpos()).all())
        assert torch.linalg.vector_norm(measured - target) < initial_error

        qpos = robot.get_qpos()
        for follower, leader, multiplier, offset in zip(
            robot.mimic_ids,
            robot.mimic_parents,
            robot.mimic_multipliers,
            robot.mimic_offsets,
            strict=True,
        ):
            assert (
                torch.max(
                    torch.abs(
                        qpos[:, follower] - (qpos[:, leader] * multiplier + offset)
                    )
                )
                < 0.02
            )
        robot.reset()
        torch.testing.assert_close(robot.get_qpos(), initial_qpos)
