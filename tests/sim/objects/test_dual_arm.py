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

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from embodichain.lab.sim.robots.dual_arm import (
    DualArmRobotCfg,
    _transform_from_xyz_rpy,
    build_dual_arm_cfg,
    resolve_mounts,
)
from embodichain.lab.sim.robots.ur_robot import URRobotCfg
from embodichain.lab.sim.solvers import URSolverCfg

# --------------------------------------------------------------------------- #
# resolve_mounts
# --------------------------------------------------------------------------- #


def test_resolve_mounts_side_by_side_is_symmetric():
    m = resolve_mounts({"preset": "side_by_side", "separation": 0.6})
    # Left at +separation/2 in Y, right at -separation/2, identity orientation.
    assert np.allclose(m["left"][:3, 3], [0.0, 0.3, 0.0])
    assert np.allclose(m["right"][:3, 3], [0.0, -0.3, 0.0])
    assert np.allclose(m["left"][:3, :3], np.eye(3))
    assert np.allclose(m["right"][:3, :3], np.eye(3))


def test_resolve_mounts_side_by_side_default_preset():
    # No preset -> side_by_side.
    m = resolve_mounts({"separation": 0.6})
    assert np.allclose(m["left"][:3, 3], [0.0, 0.3, 0.0])


def test_resolve_mounts_facing_inward_is_mirrored():
    m = resolve_mounts({"preset": "facing_inward", "separation": 0.6})
    # Both translated in Y, orientations are mirrored (transpose of each other)
    # and non-identity.
    assert np.allclose(m["left"][:3, 3], [0.0, 0.3, 0.0])
    assert np.allclose(m["right"][:3, 3], [0.0, -0.3, 0.0])
    assert not np.allclose(m["left"][:3, :3], np.eye(3))
    assert np.allclose(m["left"][:3, :3], m["right"][:3, :3].T)


def test_resolve_mounts_mirrored_rz_uses_signed_yaw():
    rz = np.pi / 4
    m = resolve_mounts({"preset": "mirrored_rz", "separation": 0.6, "rz": rz})
    assert np.allclose(m["left"][:3, 3], [0.0, 0.3, 0.0])
    assert np.allclose(m["right"][:3, 3], [0.0, -0.3, 0.0])
    expected_left = _transform_from_xyz_rpy([0.0, 0.3, 0.0], [0.0, 0.0, rz])
    expected_right = _transform_from_xyz_rpy([0.0, -0.3, 0.0], [0.0, 0.0, -rz])
    assert np.allclose(m["left"], expected_left)
    assert np.allclose(m["right"], expected_right)


def test_resolve_mounts_mirrored_rz_defaults_to_zero_yaw():
    m = resolve_mounts({"preset": "mirrored_rz", "separation": 0.6})
    assert np.allclose(m["left"][:3, 3], [0.0, 0.3, 0.0])
    assert np.allclose(m["right"][:3, 3], [0.0, -0.3, 0.0])
    assert np.allclose(m["left"][:3, :3], np.eye(3))
    assert np.allclose(m["right"][:3, :3], np.eye(3))


def test_resolve_mounts_per_arm_override():
    m = resolve_mounts(
        {
            "preset": "side_by_side",
            "separation": 0.6,
            "left": {"xyz": [0.1, 0.35, 0.0], "rpy": [0, 0, 0]},
            "right": {"xyz": [0.1, -0.35, 0.0], "rpy": [0, 0, 0]},
        }
    )
    assert np.allclose(m["left"][:3, 3], [0.1, 0.35, 0.0])
    assert np.allclose(m["right"][:3, 3], [0.1, -0.35, 0.0])


def test_resolve_mounts_one_sided_override_raises():
    with pytest.raises(ValueError):
        resolve_mounts(
            {
                "preset": "side_by_side",
                "separation": 0.6,
                "left": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            }
        )


def test_resolve_mounts_unknown_preset_raises():
    with pytest.raises(ValueError):
        resolve_mounts({"preset": "telepathic", "separation": 0.6})


# --------------------------------------------------------------------------- #
# build_dual_arm_cfg on a URRobotCfg (generic engine)
# --------------------------------------------------------------------------- #


def _ur5_dual():
    base = URRobotCfg.from_dict({"robot_type": "ur5"})
    mounts = resolve_mounts({"preset": "side_by_side", "separation": 0.6})
    return build_dual_arm_cfg(base, mounts)


def test_build_dual_arm_ur_control_parts():
    cfg = _ur5_dual()
    assert cfg.control_parts["left_arm"] == [f"left_joint{i}" for i in range(1, 7)]
    assert cfg.control_parts["right_arm"] == [f"right_joint{i}" for i in range(1, 7)]
    # dual_arm composite part concatenates both arms (12 joints).
    assert cfg.control_parts["dual_arm"] == (
        [f"left_joint{i}" for i in range(1, 7)]
        + [f"right_joint{i}" for i in range(1, 7)]
    )


def test_build_dual_arm_ur_solver_is_per_arm_and_arm_local():
    cfg = _ur5_dual()
    for side in ("left_arm", "right_arm"):
        solver = cfg.solver_cfg[side]
        assert isinstance(solver, URSolverCfg)
        assert solver.ur_type == "ur5"
        # URSolverCfg pins urdf_path to the single-arm URDF in __post_init__, so
        # the engine keeps link names UNPREFIXED to match that URDF (arm-local).
        assert solver.root_link_name == "base_link"
        assert solver.end_link_name == "ee_link"


def test_build_dual_arm_ur_urdf_components():
    cfg = _ur5_dual()
    assert set(cfg.urdf_cfg.components.keys()) == {"left_arm", "right_arm"}
    # Both arms use the same base arm URDF.
    base = URRobotCfg.from_dict({"robot_type": "ur5"})
    arm_urdf = base.urdf_cfg.components["arm"]["urdf_path"]
    assert cfg.urdf_cfg.components["left_arm"]["urdf_path"] == arm_urdf
    assert cfg.urdf_cfg.components["right_arm"]["urdf_path"] == arm_urdf


def test_build_dual_arm_dual_part_toggle():
    base = URRobotCfg.from_dict({"robot_type": "ur5"})
    mounts = resolve_mounts({"preset": "side_by_side", "separation": 0.6})
    cfg = build_dual_arm_cfg(base, mounts, dual_part=False)
    assert "dual_arm" not in cfg.control_parts


# --------------------------------------------------------------------------- #
# DualArmRobotCfg from_dict + round-trip
# --------------------------------------------------------------------------- #


def test_dual_arm_from_dict_one_liner():
    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    assert set(["left_arm", "right_arm", "dual_arm"]).issubset(cfg.control_parts.keys())
    assert isinstance(cfg.solver_cfg["left_arm"], URSolverCfg)
    assert cfg.solver_cfg["left_arm"].ur_type == "ur5"


def test_dual_arm_from_dict_explicit_base_robot():
    cfg = DualArmRobotCfg.from_dict(
        {
            "base_robot": {"type": "ur5", "init": {"robot_type": "ur5"}},
            "mount": {"preset": "side_by_side", "separation": 0.6},
        }
    )
    assert cfg.solver_cfg["left_arm"].ur_type == "ur5"


def test_dual_arm_from_dict_franka_base_robot():
    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "franka", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    assert set(["left_arm", "right_arm", "dual_arm"]).issubset(cfg.control_parts.keys())
    assert set(["left_hand", "right_hand"]).issubset(cfg.control_parts.keys())


def test_dual_arm_unknown_base_robot_raises():
    with pytest.raises(ValueError):
        DualArmRobotCfg.from_dict({"base_robot": "telepathic"})


def test_dual_arm_roundtrip():
    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    d = cfg.to_dict()
    assert d["base_robot"] == "ur5"
    cfg2 = DualArmRobotCfg.from_dict(d)
    assert cfg2.control_parts == cfg.control_parts
    assert isinstance(cfg2.solver_cfg["left_arm"], URSolverCfg)
    assert cfg2.solver_cfg["left_arm"].ur_type == "ur5"


# --------------------------------------------------------------------------- #
# PK drift-guard: build_pk_serial_chain DOF matches control_parts
# --------------------------------------------------------------------------- #


def _dof_of_pk_chain(chain) -> int:
    """Number of actuated joints in a pk.SerialChain."""
    return len(chain.get_joint_parameter_names())


def test_dual_arm_pk_dof_matches_control_parts():
    pytest.importorskip("pytorch_kinematics")
    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for arm in ("left_arm", "right_arm"):
        assert _dof_of_pk_chain(chains[arm]) == len(
            cfg.control_parts[arm]
        ), f"{arm}: PK chain DOF drifted from control_parts"


# --------------------------------------------------------------------------- #
# Integration: assemble a real dual UR5 URDF and verify predicted names match
# --------------------------------------------------------------------------- #


def test_dual_arm_ur5_assembled_joint_names_match_prediction():
    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": "ur5", "mount": {"preset": "side_by_side", "separation": 0.6}}
    )
    fpath = cfg.urdf_cfg.assemble_urdf()
    tree = ET.parse(fpath)
    joint_names = {j.get("name") for j in tree.findall("joint")}
    for i in range(1, 7):
        assert f"left_joint{i}" in joint_names, (
            f"predicted left_joint{i} not in assembled URDF; "
            f"prefix/case convention drifted from URDFAssemblyManager"
        )
        assert f"right_joint{i}" in joint_names
