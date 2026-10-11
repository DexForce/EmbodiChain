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

"""Robot catalogs are shared by Gym, workspace analysis, and composition."""

from __future__ import annotations

from copy import deepcopy
from types import ModuleType, SimpleNamespace
import sys

import pytest

from embodichain.lab.sim.cfg import RobotCfg, URDFCfg
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg, SolverCfg
from embodichain import robot_registry as registry
from embodichain.utils import configclass

pytestmark = pytest.mark.no_sim


@pytest.fixture
def catalogs(monkeypatch):
    entries = []
    registry._robot_specs.cache_clear()
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: entries)
    yield entries
    registry._robot_specs.cache_clear()


def _entry(name, catalog):
    return SimpleNamespace(name=name, load=lambda: catalog)


@pytest.fixture
def external_robot(catalogs, monkeypatch, tmp_path):
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(
        '<robot name="arm"><link name="base"/><link name="tip"/>'
        '<joint name="joint" type="revolute"><parent link="base"/>'
        '<child link="tip"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" effort="10" velocity="1"/>'
        "</joint></robot>"
    )

    @configclass
    class ExternalRobotCfg(RobotCfg):
        uid: str = "external"
        control_parts: dict[str, list[str]] = {"arm": ["joint"]}
        urdf_cfg: URDFCfg = URDFCfg(
            components=[{"component_type": "arm", "urdf_path": str(urdf)}]
        )
        solver_cfg: dict[str, SolverCfg] = {
            "arm": PytorchSolverCfg(root_link_name="base", end_link_name="tip")
        }

    module = ModuleType("external_test_robots")
    module.ExternalRobotCfg = ExternalRobotCfg
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = {
        "class_path": "external_test_robots:ExternalRobotCfg",
        "defaults": {"uid": "catalog_default", "init_qpos": [0.2]},
    }
    catalogs.append(_entry("external", {"vendor.arm": spec}))
    return ExternalRobotCfg, spec


def test_catalog_listing_defers_robot_imports(catalogs):
    catalogs.append(
        _entry(
            "external",
            {
                "vendor.arm": {
                    "class_path": "robot_module_that_is_not_installed:RobotCfg"
                }
            },
        )
    )

    assert registry.list_robot_cfgs() == ("DualArmRobot", "vendor.arm")
    assert "robot_module_that_is_not_installed" not in sys.modules


def test_resolver_merges_defaults_without_mutating_inputs(external_robot):
    cfg_cls, spec = external_robot
    original = deepcopy(spec)
    overrides = {"uid": "override", "init_qpos": [0.4]}

    cfg = registry.resolve_robot_cfg("vendor.arm", overrides)
    cfg.init_qpos.append(0.8)

    assert isinstance(cfg, cfg_cls)
    assert cfg.uid == "override"
    assert spec == original
    assert overrides == {"uid": "override", "init_qpos": [0.4]}
    assert registry.resolve_robot_cfg("vendor.arm").init_qpos == [0.2]


def test_resolver_unknown_robot_has_actionable_error(catalogs):
    with pytest.raises(ValueError, match="Unknown robot.*missing"):
        registry.resolve_robot_cfg("missing")


def test_resolver_accepts_configuration_class_spelling(external_robot):
    assert isinstance(registry.resolve_robot_cfg("vendor.armCfg"), external_robot[0])


@pytest.mark.parametrize("name", [None, ""])
def test_resolver_rejects_missing_id(catalogs, name):
    with pytest.raises(ValueError, match="nonempty string"):
        registry.resolve_robot_cfg(name)


@pytest.mark.parametrize("duplicate", ["vendor.arm", "DualArmRobot"])
def test_catalog_rejects_duplicate_ids(catalogs, duplicate):
    spec = {"class_path": "some_module:RobotCfg"}
    catalogs.append(_entry("first", {"vendor.arm": spec}))
    catalogs.append(_entry("second", {duplicate: spec}))

    with pytest.raises(ValueError, match="Duplicate robot ID"):
        registry.list_robot_cfgs()


@pytest.mark.parametrize(
    "catalog",
    [[], {"bad": {}}, {"bad": {"class_path": "module:Class", "defaults": []}}],
)
def test_invalid_catalog_is_rejected(catalogs, catalog):
    catalogs.append(_entry("invalid", catalog))

    with pytest.raises(TypeError):
        registry.list_robot_cfgs()


def test_selected_class_must_be_a_robot_config(catalogs):
    catalogs.append(_entry("invalid", {"bad": {"class_path": "builtins:object"}}))

    with pytest.raises(TypeError, match="RobotCfg subclass"):
        registry.resolve_robot_cfg("bad")


def test_selected_class_path_must_be_explicit(catalogs):
    catalogs.append(_entry("invalid", {"bad": {"class_path": "builtins.object"}}))

    with pytest.raises(ValueError, match="module:Class"):
        registry.resolve_robot_cfg("bad")


def test_gym_decodes_external_robot_through_catalog(external_robot):
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "default",
        "env": {},
        "robot": {"class_type": "vendor.arm", "uid": "gym_robot"},
    }
    original = deepcopy(config)

    cfg = config_to_cfg(config)

    assert isinstance(cfg.robot, external_robot[0])
    assert cfg.robot.uid == "gym_robot"
    assert config == original


def test_workspace_cli_accepts_and_builds_external_robot(external_robot):
    from embodichain.lab.scripts import analyze_workspace

    args = analyze_workspace.parse_args(
        [
            "--robot",
            "vendor.arm",
            "--robot-params",
            '{"uid":"workspace_robot"}',
            "--control-part",
            "arm",
        ]
    )
    cfg, control_part, urdf_path = analyze_workspace.build_preset_robot_cfg(args)

    assert isinstance(cfg, external_robot[0])
    assert cfg.uid == "workspace_robot"
    assert control_part == "arm"
    assert urdf_path is None


def test_dual_arm_composes_external_robot_without_builtin_mapping(external_robot):
    from embodichain.lab.sim.robots.dual_arm import DualArmRobotCfg

    cfg = DualArmRobotCfg.from_dict(
        {"base_robot": {"type": "vendor.arm", "init": {"uid": "single_arm"}}}
    )
    restored = DualArmRobotCfg.from_dict(cfg.to_dict())

    assert cfg.uid == "DualArm_single_arm"
    assert cfg.control_parts["dual_arm"] == ["left_joint", "right_joint"]
    assert restored.control_parts == cfg.control_parts
    assert restored.base_robot == cfg.base_robot
