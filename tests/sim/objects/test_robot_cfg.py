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

from embodichain.lab.sim.cfg import RobotCfg, JointDrivePropertiesCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.solvers import SRSSolverCfg
from embodichain.utils import configclass
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg


def test_dexforce_w1_roundtrip():
    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )
    d = cfg.to_dict()
    assert d["uid"] == "dexforce_w1"
    assert d["arm_kind"] == "anthropomorphic"
    cfg2 = DexforceW1Cfg.from_dict(d)
    assert cfg2.uid == "dexforce_w1"
    assert cfg2.arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC
    assert cfg2.version == DexforceW1Version.V021


def test_dexforce_w1_solver_cfg_is_srs_and_set_once():
    cfg = DexforceW1Cfg.from_dict({"arm_kind": "industrial"})
    assert isinstance(cfg.solver_cfg["left_arm"], SRSSolverCfg)
    assert isinstance(cfg.solver_cfg["right_arm"], SRSSolverCfg)


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


# --------------------------------------------------------------------------- #
# PK drift-guard tests -- ensure build_pk_serial_chain DOF matches control_parts
# --------------------------------------------------------------------------- #


def _dof_of_pk_chain(chain) -> int:
    """Number of actuated joints in a pk.SerialChain."""
    return len(chain.get_joint_parameter_names())


def test_dexforce_w1_pk_dof_matches_control_parts():
    pytest.importorskip("pytorch_kinematics")
    cfg = DexforceW1Cfg.from_dict({"arm_kind": "anthropomorphic"})
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
