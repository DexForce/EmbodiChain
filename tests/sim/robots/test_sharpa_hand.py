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

"""Standalone SharpaWave configuration, source preservation and native control."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from embodichain.robot_registry import list_robot_cfgs, resolve_robot_cfg
from embodichain_robots.sharpa.wave import cfg as hand_cfg


@pytest.fixture
def local_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Small source files exercise native names, limits and local-path overrides."""
    paths = {}
    for side in ("left", "right"):
        names = [
            f"{side}_{suffix}"
            for suffix in (
                "thumb_CMC_FE",
                "thumb_CMC_AA",
                "thumb_MCP_FE",
                "thumb_MCP_AA",
                "thumb_IP",
                "index_MCP_FE",
                "index_MCP_AA",
                "index_PIP",
                "index_DIP",
                "middle_MCP_FE",
                "middle_MCP_AA",
                "middle_PIP",
                "middle_DIP",
                "ring_MCP_FE",
                "ring_MCP_AA",
                "ring_PIP",
                "ring_DIP",
                "pinky_CMC",
                "pinky_MCP_FE",
                "pinky_MCP_AA",
                "pinky_PIP",
                "pinky_DIP",
            )
        ]
        root = ET.Element("robot", name=side)
        ET.SubElement(root, "link", name="palm")
        for index, name in enumerate(names):
            ET.SubElement(root, "link", name=f"link_{index}")
            joint = ET.SubElement(root, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link="palm")
            ET.SubElement(joint, "child", link=f"link_{index}")
            ET.SubElement(
                joint, "limit", lower="-1", upper="2", effort="2", velocity="4"
            )
        path = tmp_path / f"{side}.urdf"
        ET.ElementTree(root).write(path)
        paths[side] = path
    monkeypatch.setattr(
        hand_cfg,
        "get_data_path",
        lambda reference: str(paths["left" if "left" in reference else "right"]),
    )
    return paths


@pytest.mark.no_sim
@pytest.mark.parametrize("side", ("left", "right"))
def test_native_side_control_and_round_trip(
    local_asset: dict[str, Path], side: str
) -> None:
    original = resolve_robot_cfg("SharpaWave", variant=side)
    cfg = hand_cfg.SharpaWaveCfg.from_dict(json.loads(json.dumps(original.to_dict())))
    assert cfg.to_dict() == original.to_dict()
    assert cfg.side == side
    assert cfg.fpath == str(local_asset[side])
    assert cfg.control_parts == {
        "eef": [
            f"{side}_{suffix}"
            for suffix in (
                "thumb_CMC_FE",
                "thumb_CMC_AA",
                "thumb_MCP_FE",
                "thumb_MCP_AA",
                "thumb_IP",
                "index_MCP_FE",
                "index_MCP_AA",
                "index_PIP",
                "index_DIP",
                "middle_MCP_FE",
                "middle_MCP_AA",
                "middle_PIP",
                "middle_DIP",
                "ring_MCP_FE",
                "ring_MCP_AA",
                "ring_PIP",
                "ring_DIP",
                "pinky_CMC",
                "pinky_MCP_FE",
                "pinky_MCP_AA",
                "pinky_PIP",
                "pinky_DIP",
            )
        ]
    }
    assert cfg.init_qpos == [0.0] * 22
    assert cfg.root_props.fixed_base is True
    assert cfg.solver_cfg == {}
    assert cfg.build_pk_serial_chain() == {}
    # Source limits are carried through portable fields without retuning.
    assert cfg.joint_drive_props.max_effort == {
        name: 2.0 for name in cfg.control_parts["eef"]
    }
    assert cfg.joint_drive_props.max_velocity == {
        name: 4.0 for name in cfg.control_parts["eef"]
    }
    assert cfg.attrs.mass_props is None


@pytest.mark.no_sim
def test_catalog_defaults_overrides_and_isolation(
    local_asset: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain_robots._catalog import ROBOT_CONFIGS

    assert "SharpaWave" in list_robot_cfgs()
    metadata = ROBOT_CONFIGS["SharpaWave"]
    assert metadata["vendor"] == "sharpa"
    assert metadata["morphology"] == "dexterous_hand"
    assert metadata["arm_count"] == 0
    assert metadata["supports_dual_arm"] is False
    assert metadata["default_variant"] == "right"
    for side in ("left", "right"):
        assert metadata["variants"][side]["defaults"] == {"side": side}
        assert metadata["variants"][side]["dof"] == 22
        assert metadata["variants"][side]["control_parts"] == {"eef": 22}
    default = resolve_robot_cfg("SharpaWave")
    assert default.side == "right"
    default.control_parts["eef"].append("LOCAL_ONLY")
    default.init_qpos[0] = 1.0
    fresh = resolve_robot_cfg("SharpaWave")
    assert "LOCAL_ONLY" not in fresh.control_parts["eef"]
    assert fresh.init_qpos == [0.0] * 22

    def unexpected_download(reference: str) -> str:
        raise AssertionError(f"Must not download: {reference}")

    monkeypatch.setattr(hand_cfg, "get_data_path", unexpected_download)
    custom = resolve_robot_cfg(
        "SharpaWave",
        {
            "side": "left",
            "fpath": str(local_asset["left"]),
            "uid": "custom_hand",
            "control_parts": {"fingers": ["custom_joint"]},
            "joint_drive_props": {"stiffness": 1.25},
            "init_pos": [0.1, 0.2, 0.4],
        },
        variant="right",
    )
    restored = hand_cfg.SharpaWaveCfg.from_dict(custom.to_dict())
    assert restored.side == "left"
    assert restored.uid == "custom_hand"
    assert restored.fpath == str(local_asset["left"])
    assert restored.control_parts["fingers"] == ["custom_joint"]
    assert restored.joint_drive_props.stiffness == 1.25
    assert restored.init_pos == [0.1, 0.2, 0.4]


@pytest.mark.no_sim
@pytest.mark.parametrize("side", ("LEFT", "both", "", None))
def test_invalid_side_fails_before_download(
    side: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_download(reference: str) -> str:
        pytest.fail("Invalid side reached asset lookup")

    monkeypatch.setattr(hand_cfg, "get_data_path", unexpected_download)
    with pytest.raises(ValueError, match="side"):
        hand_cfg.SharpaWaveCfg.from_dict({"side": side})
    with pytest.raises(ValueError, match="side"):
        hand_cfg.SharpaWaveCfg(side=side)


@pytest.mark.requires_sim
@pytest.mark.parametrize("side", ("left", "right"))
def test_published_model_spawn_command_and_reset(side: str) -> None:
    """Real source names, limits and inertials survive native Default spawning."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    cfg = resolve_robot_cfg("SharpaWave", variant=side)
    serialized = cfg.to_dict()
    cfg = type(cfg).from_dict(json.loads(json.dumps(serialized)))
    assert cfg.to_dict() == serialized
    source = ET.parse(cfg.fpath)
    joints = [j for j in source.findall("joint") if j.get("type") != "fixed"]
    assert [j.get("name") for j in joints] == cfg.control_parts["eef"]
    assert len(joints) == 22
    assert not source.findall(".//mimic")
    limits = np.array(
        [[float(j.find("limit").get(k)) for k in ("lower", "upper")] for j in joints]
    )
    initial = np.asarray(cfg.init_qpos)
    assert np.all(initial >= limits[:, 0]) and np.all(initial <= limits[:, 1])
    sim = None
    try:
        sim = SimulationManager(
            SimulationManagerCfg(headless=True, device="cpu", num_envs=1)
        )
        robot = sim.add_robot(cfg=cfg)
        sim.prepare()
        assert robot.dof == 22
        assert not robot.mimic_ids
        expected_names = [j.get("name") for j in joints]
        assert [
            robot.joint_names[i] for i in robot.get_joint_ids("eef")
        ] == expected_names
        measured_initial = robot.get_qpos().clone()
        torch.testing.assert_close(
            robot.get_qpos(name="eef"), torch.tensor(initial[None], dtype=torch.float32)
        )
        for field, key in (("get_qpos_limits", None), ("get_qvel_limits", "velocity")):
            actual = getattr(robot, field)(name="eef").cpu().numpy()[0]
            expected = (
                limits
                if key is None
                else np.array([float(j.find("limit").get(key)) for j in joints])
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        source_links = {link.get("name"): link for link in source.findall("link")}
        assert set(robot.link_names) == set(source_links)
        for name, link in source_links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            assert robot.get_mass(name).item() == pytest.approx(
                float(inertial.find("mass").get("value")), rel=1e-5
            )
            inertia = inertial.find("inertia")
            matrix = np.array(
                [
                    [
                        float(inertia.get("i" + a + b) or inertia.get("i" + b + a))
                        for b in "xyz"
                    ]
                    for a in "xyz"
                ]
            )
            np.testing.assert_allclose(
                np.sort(robot.get_inertia(name).cpu().numpy()[0, 0]),
                np.linalg.eigvalsh(matrix),
                rtol=1e-4,
                atol=1e-9,
            )
        np.testing.assert_allclose(
            robot.get_qf_limits(name="eef").cpu().numpy()[0],
            [float(j.find("limit").get("effort")) for j in joints],
            rtol=1e-5,
        )
        target = robot.get_qpos(name="eef").clone() + 0.08
        robot.set_qpos(target, name="eef")
        error_before = torch.linalg.vector_norm(robot.get_qpos(name="eef") - target)
        sim.update(step=100)
        assert bool(torch.isfinite(robot.get_qpos()).all())
        assert (
            torch.linalg.vector_norm(robot.get_qpos(name="eef") - target)
            < error_before * 0.5
        )
        robot.reset()
        torch.testing.assert_close(robot.get_qpos(), measured_initial)
        torch.testing.assert_close(robot.get_qpos(target=True), measured_initial)
    finally:
        if sim is not None:
            sim.destroy()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
