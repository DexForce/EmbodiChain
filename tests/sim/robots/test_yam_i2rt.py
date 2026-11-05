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

"""YAM source assembly, serialization and per-arm FK/IK contracts."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from embodichain_robots import YamI2rtCfg
from embodichain_robots import yam_i2rt

pytestmark = pytest.mark.no_sim


@pytest.fixture
def arm_asset(tmp_path: Path) -> Path:
    robot = ET.Element("robot", name="yam_fixture")
    ET.SubElement(robot, "link", name="base_link")
    for side in ("right", "left"):
        parent = "base_link"
        for name, kind, axis in [
            (f"{side}_base", "fixed", None),
            *[(f"{side}_link{i}", "revolute", "0 0 1") for i in range(1, 7)],
            (f"{side}_tcp", "fixed", None),
        ]:
            ET.SubElement(robot, "link", name=name)
            number = name[-1] if kind == "revolute" else None
            joint = ET.SubElement(
                robot,
                "joint",
                name=f"{side}_joint{number}" if number else f"{name}_fixed",
                type=kind,
            )
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "origin", xyz="0.1 0 0", rpy="0 0 0")
            if axis:
                ET.SubElement(joint, "axis", xyz=axis)
                ET.SubElement(
                    joint, "limit", lower="-3", upper="3", effort="27", velocity="3"
                )
            parent = name
    path = tmp_path / "yam_i2rt.urdf"
    ET.ElementTree(robot).write(path, encoding="unicode")
    return path


def test_default_asset_is_resolved_once_and_shared_with_fk(
    monkeypatch: pytest.MonkeyPatch, arm_asset: Path
) -> None:
    calls = []

    def resolve(name):
        calls.append(name)
        return str(arm_asset)

    monkeypatch.setattr(yam_i2rt, "get_data_path", resolve)
    cfg = YamI2rtCfg.from_dict({})
    assert calls == ["YamI2rt/yam_i2rt.urdf"]
    assert cfg.urdf_cfg.components["arm"]["urdf_path"] == str(arm_asset)
    assert cfg._pk_urdf_path == str(arm_asset)
    assert cfg.fpath is None
    restored = YamI2rtCfg.from_dict(cfg.to_dict())
    assert restored.urdf_cfg.components["arm"]["urdf_path"] == str(arm_asset)
    np.testing.assert_array_equal(
        restored.urdf_cfg.components["arm"]["transform"], np.eye(4)
    )
    assert restored.urdf_cfg.assemble_urdf() == str(arm_asset)
    assert restored.control_parts == cfg.control_parts
    assert restored._pk_urdf_path == cfg._pk_urdf_path


def test_local_override_roundtrips_without_network(
    monkeypatch: pytest.MonkeyPatch, arm_asset: Path
) -> None:
    def no_download(*args):
        raise AssertionError("Explicit offline assets must not download")

    monkeypatch.setattr(yam_i2rt, "get_data_path", no_download)
    cfg = YamI2rtCfg.from_dict({"fpath": str(arm_asset)})
    restored = YamI2rtCfg.from_dict(cfg.to_dict())
    assert restored.fpath == str(arm_asset)
    assert restored._pk_urdf_path == str(arm_asset)
    assert restored.control_parts == cfg.control_parts


def test_per_arm_chain_joint_order_and_solver_fk_match(arm_asset: Path) -> None:
    pytest.importorskip("pytorch_kinematics")
    cfg = YamI2rtCfg.from_dict({"fpath": str(arm_asset)})
    chains = cfg.build_pk_serial_chain()
    assert set(chains) == {"right_arm", "left_arm"}
    qpos = torch.tensor([[0.0] * 6, [0.2, -0.3, 0.1, 0.4, -0.1, 0.2]])
    for part, chain in chains.items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]
        assert len(chain.get_joint_parameter_names()) == 6
        solver_cfg = cfg.solver_cfg[part]
        solver_cfg.urdf_path = str(arm_asset)
        solver_cfg.joint_names = cfg.control_parts[part]
        solver = solver_cfg.init_solver(device=torch.device("cpu"))
        torch.testing.assert_close(
            solver.get_fk(qpos), chain.forward_kinematics(qpos).get_matrix()
        )
