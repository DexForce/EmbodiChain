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

"""Passive E6 drawers must compose with calibrated E9 joints in one asset."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dexsim.types import DriveType
import pytest

from embodichain.gen_sim.task_engine.task_program_bundle import (
    _merge_passive_prismatic_drive,
)
from embodichain.lab.sim.cfg import ArticulationCfg, JointDrivePropertiesCfg
from embodichain.lab.sim.spawn.descriptors import _compile_joint_properties


def test_button_remains_driven_while_drawer_is_passive() -> None:
    drive = {
        "target_mode": {"button_slide": "position_velocity"},
        "stiffness": {"button_slide": 640.0},
        "damping": {"button_slide": 5.8},
        "max_effort": {"button_slide": 6.4},
    }
    original = deepcopy(drive)
    _merge_passive_prismatic_drive(drive, ["drawer_slide"])
    cfg = ArticulationCfg(
        uid="cabinet",
        fpath="assembly.usdc",
        joint_drive_props=JointDrivePropertiesCfg.from_dict(drive),
    )
    desc = SimpleNamespace(
        joints=[SimpleNamespace(name=name) for name in ("drawer_slide", "button_slide")]
    )
    properties, common, _, modes = _compile_joint_properties(
        desc, cfg, newton_solver_type=None
    )
    assert modes == {"button_slide": 3, "drawer_slide": 0}
    drawer, button = properties["drawer_slide"][0], properties["button_slide"][0]
    assert drawer.drive_mode == DriveType.NONE
    assert drawer.stiffness == drawer.damping == 0.0
    assert button.drive_mode == DriveType.FORCE
    assert button.stiffness == original["stiffness"]["button_slide"]
    assert button.damping == original["damping"]["button_slide"]
    assert common["button_slide"]["effort_limit"] == 6.4
    assert drive["friction"] == {"drawer_slide": 0.01}


def test_plain_drawer_retains_legacy_configuration() -> None:
    drive = {}
    _merge_passive_prismatic_drive(drive, ["drawer_slide"])
    assert drive == {"drive_type": "none", "friction": {"drawer_slide": 0.01}}


@pytest.mark.parametrize(
    "drive",
    [
        {"drive_type": "force", "stiffness": 15.0},
        {"drive_type": "acceleration", "damping": 8.0},
        {"drive_type": None, "target_mode": "position"},
        {"drive_type": "none", "friction": 0.3},
    ],
)
def test_authored_drive_and_friction_values_are_preserved(drive: dict) -> None:
    original = deepcopy(drive)
    _merge_passive_prismatic_drive(drive, ["drawer_slide"])
    assert drive == original


def test_authored_joint_rule_is_not_shadowed() -> None:
    drive = {"target_mode": {"drawer_.*": "effort", "button_slide": "position"}}
    _merge_passive_prismatic_drive(drive, ["drawer_slide"])
    assert drive["target_mode"] == {
        "drawer_.*": "effort",
        "button_slide": "position",
    }
    assert "drive_type" not in drive


def test_scalar_target_mode_is_not_overridden() -> None:
    drive = {"target_mode": "position"}
    _merge_passive_prismatic_drive(drive, ["drawer_slide"])
    assert drive == {"target_mode": "position", "friction": {"drawer_slide": 0.01}}
