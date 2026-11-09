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

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.motion.solvers import FEPSolver, FEPSolverCfg
from embodichain_robots.dexforce.w1 import DexforceW1Cfg
from embodichain_robots.dexforce.w1.utils import build_dexforce_w1_assembly_urdf_cfg


@pytest.fixture
def standard_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DexforceW1Cfg:
    """Split the independent full-body reference into local kinematic components."""
    from embodichain_robots.dexforce.w1 import utils

    fixture = Path(__file__).parents[2] / "fixtures/kinematics/w1_v030_body.urdf"
    reference = ET.parse(fixture).getroot()
    roots = {
        "chassis": "chassis_base_link",
        "torso": "torso_base",
        "head": "head_base",
        "left_arm": "left_arm_base",
        "right_arm": "right_arm_base",
    }
    assets = tmp_path / "assets"
    assets.mkdir()
    for part, root in roots.items():
        members = {root}
        while True:
            children = {
                j.find("child").get("link")
                for j in reference.findall("joint")
                if j.find("parent").get("link") in members
                and j.find("child").get("link") not in roots.values()
            }
            if children <= members:
                break
            members |= children
        component = ET.Element("robot", name=part)
        for link in reference.findall("link"):
            if link.get("name") in members:
                component.append(deepcopy(link))
        for joint in reference.findall("joint"):
            if all(
                joint.find(tag).get("link") in members for tag in ("parent", "child")
            ):
                component.append(deepcopy(joint))
        ET.ElementTree(component).write(assets / f"{part}.urdf")

    def local_asset(path: str) -> str:
        assert path.startswith("DexforceW1V030/w1/")
        return str(assets / Path(path).name)

    monkeypatch.setattr(utils, "get_data_path", local_asset)
    cfg = DexforceW1Cfg.from_dict({"version": "v030"})
    cfg.urdf_cfg.fpath = str(tmp_path / "assembled.urdf")
    return cfg


def _assert_poses(
    expected: torch.Tensor, actual: torch.Tensor, solver: FEPSolver
) -> None:
    expected = expected.detach().cpu().numpy().astype(float)
    actual = actual.detach().cpu().numpy().astype(float)
    position = np.linalg.norm(expected[:, :3, 3] - actual[:, :3, 3], axis=-1)
    rotation = (
        Rotation.from_matrix(expected[:, :3, :3]).inv()
        * Rotation.from_matrix(actual[:, :3, :3])
    ).magnitude()
    assert np.all(position <= solver.cfg.position_tolerance)
    assert np.all(rotation <= solver.cfg.rotation_tolerance)


def test_standard_assembly_and_json_roundtrip_preserve_source_body(
    standard_cfg: DexforceW1Cfg,
) -> None:
    import pytorch_kinematics as pk

    original = Path(__file__).parents[2] / "fixtures/kinematics/w1_v030_body.urdf"
    reference = pk.build_chain_from_urdf(original.read_bytes())
    restored = DexforceW1Cfg.from_dict(json.loads(json.dumps(standard_cfg.to_dict())))
    path = restored.urdf_cfg.assemble_urdf()
    tree = ET.parse(path)
    assert len(tree.findall("link")) == 30
    assert len(tree.findall("joint")) == 29
    assert len([j for j in tree.findall("joint") if j.get("type") == "revolute"]) == 20
    assert not tree.findall(".//mimic")
    assert restored.urdf_cfg.sensors == []
    assembled = pk.build_chain_from_urdf(Path(path).read_bytes())
    names = reference.get_joint_parameter_names()
    assert len(names) == len(assembled.get_joint_parameter_names()) == 20
    qpos = 0.5 * torch.randn(32, 20, generator=torch.Generator().manual_seed(30))
    actual_qpos = qpos[
        :, [names.index(n) for n in assembled.get_joint_parameter_names()]
    ]
    expected, actual = reference.forward_kinematics(qpos), assembled.forward_kinematics(
        actual_qpos
    )
    assert set(expected) == set(actual)
    for link in expected:
        torch.testing.assert_close(
            expected[link].get_matrix(), actual[link].get_matrix(), atol=1e-6, rtol=0
        )
    # Direct builder callers get the same revision-owned bare default.
    direct = build_dexforce_w1_assembly_urdf_cfg(version="v030")
    assert set(direct.components) == set(standard_cfg.urdf_cfg.components)
    assert direct.sensors == []


@pytest.mark.parametrize("part", ["left_arm", "right_arm"])
@pytest.mark.parametrize("with_tcp", [False, True])
def test_standard_cfg_fep_matches_assembled_chain_and_all_valid_branches(
    standard_cfg: DexforceW1Cfg, part: str, with_tcp: bool
) -> None:
    paths = standard_cfg.urdf_cfg.assemble_urdf()
    config = standard_cfg.solver_cfg[part]
    assert isinstance(config, FEPSolverCfg)
    assert config.user_qpos_limits is None
    # Exercise exactly Robot.init_solver's full assembled-URDF binding.
    config.urdf_path = paths
    config.joint_names = standard_cfg.control_parts[part]
    config.redundancy_search = False
    solver = config.init_solver("cpu")
    source = standard_cfg.build_pk_serial_chain()[part]
    assert (
        solver.pk_serial_chain.get_joint_parameter_names()
        == source.get_joint_parameter_names()
    )
    assert len(source.get_joint_parameter_names()) == 7
    np.testing.assert_allclose(solver.upper_qpos_limits[5], np.deg2rad(95), atol=1e-7)
    if with_tcp:
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
        tcp[:3, 3] = [0.05, -0.03, 0.1]
        solver.set_tcp(tcp)
    generator = torch.Generator().manual_seed(30)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    qpos = lower + (upper - lower) * (
        0.1 + 0.8 * torch.rand(128, 7, generator=generator)
    )
    target = source.forward_kinematics(qpos).get_matrix() @ torch.as_tensor(
        solver.tcp_xpos, dtype=qpos.dtype
    )
    _assert_poses(target, solver.get_fk(qpos), solver)
    seed = lower + (upper - lower) * torch.rand(128, 7, generator=generator)
    seed[:, 6] = qpos[:, 6]
    valid, candidates = solver.get_ik(target, seed, return_all_solutions=True)
    assert bool(valid.any(dim=-1).all())
    accepted = candidates[valid]
    assert bool(((accepted >= lower) & (accepted <= upper)).all())
    _assert_poses(
        target[:, None].expand(-1, 8, -1, -1)[valid], solver.get_fk(accepted), solver
    )
    # A stale SRS/Franka limit must not clip valid enlarged wrist positions.
    assert bool((qpos[:, 5].abs() > np.pi / 4).any())


@pytest.mark.parametrize("part", ["left_arm", "right_arm"])
def test_standard_cfg_search_and_partial_failures(
    standard_cfg: DexforceW1Cfg, part: str
) -> None:
    config = standard_cfg.solver_cfg[part]
    solver = config.init_solver(
        "cpu", pk_serial_chain=standard_cfg.build_pk_serial_chain()[part]
    )
    assert solver.cfg.redundancy_search
    generator = torch.Generator().manual_seed(30)
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    qpos = lower + (upper - lower) * (
        0.2 + 0.6 * torch.rand(32, 7, generator=generator)
    )
    seed = qpos + 0.03 * torch.sin(torch.arange(224).reshape(32, 7))
    target = solver.get_fk(qpos)
    target[-1, 0, 3] += 10.0
    valid, result = solver.get_ik(target, seed)
    assert bool(valid[:-1].all()) and not bool(valid[-1])
    assert bool(((result >= lower) & (result <= upper)).all())
    _assert_poses(target[valid], solver.get_fk(result[valid]), solver)
    torch.testing.assert_close(result[-1], seed[-1], atol=0, rtol=0)


def test_standard_component_frame_metadata_resolves_source_links(
    standard_cfg: DexforceW1Cfg,
) -> None:
    from embodichain_robots.dexforce.w1.utils import (
        chassis_manager,
        torso_manager,
        head_manager,
    )

    for manager in (chassis_manager, torso_manager, head_manager):
        info = manager.get_config("v030")
        links = {l.get("name") for l in ET.parse(info["urdf_path"]).findall("link")}
        assert info["root_link_name"] in links
        assert info["end_link_name"] in links


@pytest.mark.parametrize("part", ["left_arm", "right_arm"])
def test_standard_home_pose_and_joint_boundaries(
    standard_cfg: DexforceW1Cfg, part: str
) -> None:
    solver = standard_cfg.solver_cfg[part].init_solver(
        "cpu", pk_serial_chain=standard_cfg.build_pk_serial_chain()[part]
    )
    lower, upper = solver.lower_qpos_limits, solver.upper_qpos_limits
    qpos = torch.stack((torch.zeros_like(lower), lower, upper, (lower + upper) / 2))
    target = solver.get_fk(qpos)
    valid, result = solver.get_ik(target, qpos)
    assert bool(valid.all())
    assert bool(((result >= lower) & (result <= upper)).all())
    _assert_poses(target, solver.get_fk(result), solver)
    # The source home posture must also be reachable from the default seed.
    valid, result = solver.get_ik(target[:1])
    assert bool(valid.all())
    _assert_poses(target[:1], solver.get_fk(result), solver)
