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

"""G2 control topology, source frames and portable configuration contracts."""

from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from embodichain.robot_registry import resolve_robot_cfg
from embodichain_robots import AgibotG2Cfg
from embodichain_robots.agibot.g2 import cfg as g2

_URDF = Path(g2.__file__).with_name("g2.urdf")


@pytest.mark.no_sim
def test_g2_native_joint_and_mimic_contract() -> None:
    """Catalog counts describe the actual source tree, excluding mimic followers."""
    from embodichain_robots._catalog import ROBOT_CONFIGS

    cfg = resolve_robot_cfg("AgibotG2", overrides={"fpath": str(_URDF)})
    robot = ET.parse(_URDF)
    moving = {
        j.get("name"): j for j in robot.findall("joint") if j.get("type") != "fixed"
    }
    mimics = {
        name: j.find("mimic").get("joint")
        for name, j in moving.items()
        if j.find("mimic") is not None
    }
    controlled = {name for joints in cfg.control_parts.values() for name in joints}
    assert controlled == set(moving)
    assert len(controlled) == 46
    assert len(controlled - set(mimics)) == 32
    assert ROBOT_CONFIGS["AgibotG2"]["mimic_joints"] == mimics
    for side, label, offset in (("left", "l", 30), ("right", "r", 70)):
        independent = [j for j in cfg.control_parts[f"{side}_eef"] if j not in mimics]
        assert independent == [f"idx{offset+1}_gripper_{label}_inner_joint1"]
        assert float(
            moving[independent[0]].find("limit").get("lower")
        ) == pytest.approx(-np.pi / 4)
        assert float(moving[independent[0]].find("limit").get("upper")) == 0.0
        for name in cfg.control_parts[f"{side}_eef"]:
            if name in mimics:
                assert cfg.joint_drive_props.target_mode[f"{side}_eef"] == "none"
    for part, names in cfg.control_parts.items():
        for name in names:
            effort = cfg.joint_drive_props.max_effort.get(
                name, cfg.joint_drive_props.max_effort.get(part)
            )
            assert effort == float(moving[name].find("limit").get("effort"))
    assert cfg.joint_drive_props.max_velocity is None
    # Native wheel centers: 0.16675 - 0.13675 m; cylinder radius: 0.07 m.
    # A root at Z=0 penetrates the ground and destabilizes the fixed chassis.
    assert cfg.init_pos[2] + 0.03 - 0.07 == pytest.approx(0.001)
    lock = json.loads(_URDF.with_name("source.lock.json").read_text())
    for mesh in robot.findall(".//mesh"):
        assert (
            f"{lock['source_directory']}/robots/genie/g2/{mesh.get('filename')}"
            in lock["files"]
        )
    assert robot.findall("loop_joint") == []
    for side in ("l", "r"):
        assert (
            robot.find(f"link[@name='gripper_{side}_inner_link4']/collision")
            is not None
        )


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "effort_rules",
    [
        {"left_arm": 30.0},
        {"idx2[1-7]_arm_l_joint[1-7]": 30.0},
        {"left_arm": 30.0, "idx21_arm_l_joint1": 20.0},
    ],
    ids=["control_part", "regex", "control_part_and_joint"],
)
def test_g2_roundtrip_and_local_overrides_do_not_download(
    monkeypatch: pytest.MonkeyPatch,
    effort_rules: dict[str, float],
) -> None:
    """A caller-owned URDF, TCP and drive overrides survive JSON round-tripping."""
    monkeypatch.setattr(
        g2, "get_data_path", lambda _: pytest.fail("Local URDF must avoid downloads")
    )
    tcp = np.eye(4).tolist()
    tcp[2][3] = 0.15
    overrides = {
        "fpath": str(_URDF),
        "uid": "local_g2",
        "root_props": {"fixed_base": False},
        "solver_cfg": {"left_arm": {"tcp": tcp}},
        "joint_drive_props": {"max_effort": effort_rules},
        "sensors": [],
    }
    original = deepcopy(overrides)
    cfg = AgibotG2Cfg.from_dict(overrides)
    restored = AgibotG2Cfg.from_dict(json.loads(cfg.to_string()))
    assert restored.to_dict() == cfg.to_dict()
    assert restored._pk_urdf_path == str(_URDF)
    assert restored.root_props.fixed_base is False
    assert restored.uid == "local_g2"
    np.testing.assert_array_equal(restored.solver_cfg["left_arm"].tcp, tcp)
    from embodichain.lab.sim.spawn.descriptors import _joint_property_matches

    expected = {
        joint.get("name"): float(joint.find("limit").get("effort"))
        for joint in ET.parse(_URDF).findall("joint")
        if joint.get("type") != "fixed"
    }
    expected.update({name: 30.0 for name in cfg.control_parts["left_arm"]})
    if "idx21_arm_l_joint1" in effort_rules:
        expected["idx21_arm_l_joint1"] = 20.0
    for candidate in (cfg, restored):
        actual = dict(
            _joint_property_matches(
                candidate.joint_drive_props.max_effort,
                list(expected),
                property_name="max_effort",
                control_parts=candidate.control_parts,
            )
        )
        assert actual == expected
    assert overrides == original


@pytest.mark.no_sim
def test_g2_link_inertias_are_physically_admissible() -> None:
    """Source-invalid tensors must not trigger the importer's tiny fallback."""
    lock = json.loads(_URDF.with_name("source.lock.json").read_text())
    assert set(lock["inertia_repairs"]) == {"body_link4", "head_link1"}
    for link in ET.parse(_URDF).findall("link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            continue
        values = {name: float(value) for name, value in inertia.attrib.items()}
        tensor = np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ]
        )
        moments = np.linalg.eigvalsh(tensor.astype(np.float32))
        assert moments[0] > 0, link.get("name")
        assert moments[2] <= moments[0] + moments[1], link.get("name")


def _source_fk(end: str, positions: dict[str, float]) -> np.ndarray:
    """Multiply source joint transforms independently of PyTorch Kinematics."""
    joints = {j.find("child").get("link"): j for j in ET.parse(_URDF).findall("joint")}
    path = []
    while end != "arm_base_link":
        joint = joints[end]
        path.append(joint)
        end = joint.find("parent").get("link")
    pose = np.eye(4)
    for joint in reversed(path):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        transform[:3, 3] = xyz
        pose = pose @ transform
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(
                axis * positions[joint.get("name")]
            ).as_matrix()
            pose = pose @ rotation
    return pose


@pytest.mark.parametrize("q", ([0.0] * 7, [-0.2, 0.3, -0.1, -0.8, 0.2, -0.2, 0.15]))
@pytest.mark.no_sim
def test_g2_arm_chains_match_source_frames(q: list[float]) -> None:
    """Each chain has exactly seven native joints and agrees with source FK."""
    cfg = AgibotG2Cfg.from_dict({"fpath": str(_URDF)})
    for part, chain in cfg.build_pk_serial_chain().items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]
        assert len(chain.get_joint_parameter_names()) == 7
        actual = chain.forward_kinematics(torch.tensor([q])).get_matrix()[0].numpy()
        expected = _source_fk(
            cfg.solver_cfg[part].end_link_name, dict(zip(cfg.control_parts[part], q))
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.no_sim
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_g2_default_ik_preserves_offset_limits_tcp_and_failure_masks(
    device: str,
) -> None:
    """Solve the original offset arms, including boundary and unreachable queries."""
    from embodichain.lab.sim.motion.solvers import SSIKSolverCfg

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = AgibotG2Cfg.from_dict({"fpath": str(_URDF)})
    for part, chain in cfg.build_pk_serial_chain(torch.device(device)).items():
        options = cfg.solver_cfg[part].copy()
        assert isinstance(options, SSIKSolverCfg)
        assert options.backend == "warp" and options.srs_search == "seed_local"
        # A caller-configured rigid tool transform must remain in the IK query.
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
        tcp[:3, 3] = [0.02, -0.01, 0.12]
        options.tcp = tcp
        solver = options.init_solver(device=device, pk_serial_chain=chain)
        assert solver.family == "srs"
        low, high = (torch.tensor(v, device=device) for v in chain.get_joint_limits())
        fractions = 0.05 + 0.9 * torch.rand(
            32, 7, generator=torch.Generator().manual_seed(42)
        )
        q = low + fractions.to(device) * (high - low)
        q = torch.cat((q, torch.zeros_like(q[:1]), low[None], high[None]))
        tool = torch.as_tensor(tcp, device=device, dtype=q.dtype)
        targets = chain.forward_kinematics(q).get_matrix() @ tool
        targets[-1, :3, 3] = 100.0
        seed = ((low + high) / 2).expand_as(q).clone()
        seed[-1] = high + 10.0
        valid, result = solver.get_ik_batch(targets, qpos_seed=seed)
        assert valid.tolist() == [True] * (len(q) - 1) + [False]
        assert torch.isfinite(result).all()
        assert (result >= low).all() and (result <= high).all()
        torch.testing.assert_close(result[-1], high, atol=0, rtol=0)
        actual = chain.forward_kinematics(result[valid]).get_matrix() @ tool
        torch.testing.assert_close(actual, targets[valid], atol=1e-5, rtol=0)
        # Candidate enumeration must also return valid original-model poses.
        mask, candidates = solver.get_ik(
            targets[:1], seed[:1], return_all_solutions=True
        )
        assert mask.any() and candidates.shape[-1] == 7
        actual = chain.forward_kinematics(candidates[mask]).get_matrix() @ tool
        torch.testing.assert_close(
            actual, targets[:1].expand_as(actual), atol=1e-5, rtol=0
        )


class _AgibotG2Simulation:
    """Default ground placement, native FK, stable stepping and gripper actuation."""

    def setup_method(self) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import RenderCfg

        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                num_envs=1,
                render_cfg=RenderCfg(renderer="no-render"),
                robot_ik_gizmo=None,
            )
        )

    def teardown_method(self) -> None:
        from embodichain.lab.sim import SimulationManager

        if hasattr(self, "sim"):
            self.sim.destroy(exit_process=False)
            del self.sim
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    def test_native_fk_and_mimic_commands_remain_finite(self) -> None:
        cfg = resolve_robot_cfg(
            "AgibotG2",
            overrides={"joint_drive_props": {"max_effort": {"left_arm": 30.0}}},
        )
        robot = self.sim.add_robot(cfg)
        self.sim.prepare()
        assert robot.dof == 46
        assert len(robot.active_joint_ids) == 32
        torch.testing.assert_close(
            robot.get_joint_drive(joint_ids=robot.get_joint_ids(name="left_arm"))[2],
            torch.full((1, 7), 30.0),
        )
        torch.testing.assert_close(
            robot.get_joint_drive(joint_ids=robot.get_joint_ids(name="right_arm"))[2],
            torch.tensor([[108.0, 108.0, 35.0, 35.0, 35.0, 18.0, 18.0]]),
        )
        chains = cfg.build_pk_serial_chain()
        for part in chains:
            q = torch.tensor([[-0.2, 0.3, -0.1, -0.8, 0.2, -0.2, 0.15]])
            robot.set_qpos(q, name=part, target=False)
            robot.set_qpos(q, name=part)
        self.sim.update(step=1)
        base = robot.get_link_pose("arm_base_link", to_matrix=True)
        for part, chain in chains.items():
            expected = (
                base @ chain.forward_kinematics(robot.get_qpos(name=part)).get_matrix()
            )
            actual = robot.get_link_pose(
                cfg.solver_cfg[part].end_link_name, to_matrix=True
            )
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0.0)
            success, solved = robot.compute_ik(
                pose=actual, name=part, joint_seed=robot.get_qpos(name=part) + 0.02
            )
            assert success.all()
            reconstructed = robot.compute_fk(solved, name=part, to_matrix=True)
            torch.testing.assert_close(reconstructed, actual, atol=1e-5, rtol=0)
        for part in ("left_eef", "right_eef"):
            leader_ids = robot.get_joint_ids(name=part, remove_mimic=True)
            assert robot.get_joint_drive(joint_ids=leader_ids)[2].item() == 10.0
            robot.set_qpos(
                torch.tensor([[-0.6]]),
                joint_ids=leader_ids,
            )
        # At Z=0 the wheel cylinders penetrate the ground, producing nonfinite
        # chassis states within eight steps. Exercise a full second plus motion.
        self.sim.update(step=100)
        assert torch.isfinite(robot.get_qpos()).all()
        source = ET.parse(cfg.fpath)
        for part in ("left_eef", "right_eef"):
            positions = robot.get_qpos(name=part)[0]
            assert positions[0].item() == pytest.approx(-0.6, abs=0.02)
            for index, name in enumerate(cfg.control_parts[part]):
                mimic = source.find(f"joint[@name='{name}']/mimic")
                if mimic is not None:
                    expected = positions[0].item() * float(mimic.get("multiplier"))
                    assert positions[index].item() == pytest.approx(expected, abs=0.03)


@pytest.mark.subprocess_sim
def test_native_fk_and_mimic_commands_remain_finite() -> None:
    """Isolate NoRender from renderers initialized by earlier simulation tests."""
    code = """
import runpy
import sys

case = runpy.run_path(sys.argv[1])["_AgibotG2Simulation"]()
try:
    case.setup_method()
    case.test_native_fk_and_mimic_commands_remain_finite()
finally:
    case.teardown_method()
print("G2 native FK and mimic checks completed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "G2 native FK and mimic checks completed" in result.stdout, (
        result.stdout + result.stderr
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
