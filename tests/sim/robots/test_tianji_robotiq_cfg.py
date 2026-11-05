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

"""Model-owned fixed wrist assembly, source geometry and explicit calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.robot_registry import resolve_robot_cfg
from embodichain_robots import TianjiMarvinRobotiq140Cfg


@pytest.fixture(autouse=True)
def _no_runtime_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.solvers import BaseSolver

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Static configuration/URDF assembly created a runtime instance.")

    for kind in (SimulationManager, Robot, BaseSolver):
        monkeypatch.setattr(kind, "__init__", forbidden)


def test_robotiq_marvin_catalog_and_overrides_are_owned(tmp_path):
    source = tmp_path / "explicit.urdf"
    source.write_text('<robot name="explicit"/>')
    assert (
        type(resolve_robot_cfg("TianjiMarvinRobotiq140", {"fpath": str(source)}))
        is TianjiMarvinRobotiq140Cfg
    )
    cfg = TianjiMarvinRobotiq140Cfg.from_dict(
        {"fpath": str(source), "solver_cfg": {"left_arm": {"pos_eps": 0.005}}}
    )
    assert cfg.sensors == []
    assert cfg.link_attrs is None
    assert len(cfg.control_parts["dual_arm"]) == 14
    assert (
        len(cfg.control_parts["left_eef"]) == len(cfg.control_parts["right_eef"]) == 6
    )
    assert cfg.joint_drive_props.drive_type == "force"
    assert cfg.joint_drive_props.friction == 0.0
    assert cfg.joint_drive_props.armature == 0.0
    assert cfg.solver_cfg["left_arm"].end_link_name == "left_ee"
    assert cfg.solver_cfg["left_arm"].tcp[2][3] == 0.2
    assert TianjiMarvinRobotiq140Cfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


@pytest.mark.slow
@pytest.mark.no_sim
def test_cached_pinned_marvin_resources_and_arm_chains_match_b0(tmp_path, monkeypatch):
    """Use only existing pinned inputs; this does not download replacement assets."""
    from embodichain_robots.tianji.marvin import robotiq

    data_root = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "extract"
    paths = {
        name: data_root / name
        for name in (
            "TianjiMarvin/robot_acd.urdf",
            "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf",
        )
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip(
            "Pinned model inputs are absent; physical qualification remains unavailable."
        )
    monkeypatch.setattr(robotiq, "get_data_path", lambda name: str(paths[name]))
    cfg = TianjiMarvinRobotiq140Cfg.from_dict({"assembly_directory": str(tmp_path)})
    path = Path(cfg.fpath)
    frozen = json.loads(
        (
            Path(__file__).parents[2]
            / "gym/envs/task_program/data/robotiq_embodiments_b0.json"
        ).read_text()
    )["dual_tianji"]
    for relative, expected in frozen["assembly_resources_sha256"].items():
        assert (
            hashlib.sha256((path.parent / relative).read_bytes()).hexdigest()
            == expected
        ), relative
    root = ET.parse(path).getroot()
    for side in ("left", "right"):
        joint = next(
            j
            for j in root.findall("joint")
            if j.find("child").attrib["link"] == f"{side}_robotiq_arg2f_base_link"
        )
        assert joint.attrib["type"] == "fixed"
        assert joint.find("parent").attrib["link"] == f"{side}_ee"
    chains = cfg.build_pk_serial_chain()
    assert set(chains) == {"left_arm", "right_arm"}
    for part, chain in chains.items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]
    # A second reader sees the fully published model and its relative mesh resources.
    again = TianjiMarvinRobotiq140Cfg.from_dict({"assembly_directory": str(tmp_path)})
    assert again.fpath == cfg.fpath
    assert all(
        (path.parent / mesh.attrib["filename"]).is_file() for mesh in root.iter("mesh")
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-m", "not gpu"]))
