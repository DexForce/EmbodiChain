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
    original_entry_points = registry.metadata.entry_points

    def entry_points(**kwargs):
        if kwargs.get("group") == "embodichain.robots":
            return entries
        return original_entry_points(**kwargs)

    monkeypatch.setattr(registry.metadata, "entry_points", entry_points)
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


def test_explicit_variant_preserves_defaults_and_isolates_overrides(external_robot):
    cfg_cls, spec = external_robot
    spec["variants"] = {
        "alternate": {"defaults": {"uid": "variant", "init_qpos": [0.6]}}
    }
    spec["default_variant"] = "alternate"
    original = deepcopy(spec)
    overrides = {"uid": "caller"}

    selected = registry.resolve_robot_cfg("vendor.arm", overrides, variant="alternate")
    selected.init_qpos.append(0.9)

    assert isinstance(selected, cfg_cls)
    assert selected.uid == "caller"
    assert selected.init_qpos == [0.6, 0.9]
    assert registry.resolve_robot_cfg("vendor.arm").init_qpos == [0.2]
    assert registry.resolve_robot_cfg("vendor.arm", variant="alternate").init_qpos == [
        0.6
    ]
    assert spec == original
    assert overrides == {"uid": "caller"}


@pytest.mark.parametrize("variant", ["missing", "", 1])
def test_variant_is_validated_before_loading_robot_implementation(catalogs, variant):
    catalogs.append(
        _entry(
            "external",
            {
                "vendor.arm": {
                    "class_path": "uninstalled_robot:RobotCfg",
                    "variants": {"standard": {"defaults": {}}},
                }
            },
        )
    )

    with pytest.raises(ValueError, match="Unknown variant.*Available variants"):
        registry.resolve_robot_cfg("vendor.arm", variant=variant)
    assert "uninstalled_robot" not in sys.modules


def test_variant_defaults_require_a_mapping(catalogs):
    catalogs.append(
        _entry(
            "external",
            {
                "vendor.arm": {
                    "class_path": "uninstalled_robot:RobotCfg",
                    "variants": {"standard": {"defaults": []}},
                }
            },
        )
    )
    with pytest.raises(TypeError, match="defaults must be a mapping"):
        registry.list_robot_cfgs()


@pytest.mark.parametrize(
    "name, variant, field",
    [
        ("URRobot", "ur5e", "robot_type"),
        ("DualArmRobot", "ur5", "base_robot"),
    ],
)
def test_named_official_variant_selects_its_model(
    catalogs, local_robot_assets, name, variant, field
):
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))

    cfg = registry.resolve_robot_cfg(name, variant=variant)

    assert getattr(cfg, field) == variant


def test_resolver_rejects_unregistered_configuration_class_spelling(external_robot):
    with pytest.raises(ValueError, match="Unknown robot.*vendor.armCfg"):
        registry.resolve_robot_cfg("vendor.armCfg")


@pytest.mark.parametrize(
    "name",
    [
        "aloha_mini",
        "tianji_marvin",
        "dexforce_w1",
        "cobotmagic",
        "franka",
        "franka_panda",
        "ur",
        "AlohaMiniCfg",
        "TianjiMarvinCfg",
        "DexforceW1Cfg",
        "CobotMagicCfg",
        "FrankaPandaCfg",
        "URRobotCfg",
        "DualArmRobotCfg",
    ],
)
def test_official_catalog_rejects_retired_selectors(catalogs, name):
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))

    with pytest.raises(ValueError, match="Unknown robot"):
        registry.resolve_robot_cfg(name)


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


@pytest.mark.parametrize(
    "fields",
    [
        {"dof": -1},
        {"vendor": 1},
        {"morphology": ""},
        {"arm_count": -1},
        {"arm_count": True},
        {"variants": {"alternate": {"arm_count": 1.5}}},
        {"dof": True},
        {"dof": 1.5},
        {"control_parts": ["arm"]},
        {"control_parts": {"arm": -1}},
        {"control_parts": {"arm": True}},
        {"control_parts": {"": 6}},
        {"mimic_joints": {"finger": None}},
        {"control_part_mimics": {"arm": "finger"}},
        {"mimic_joints": {}, "control_part_mimics": {"arm": ("finger",)}},
        {"supports_dual_arm": "yes"},
        {"variants": {"bad": {"dof": -1}}},
    ],
)
def test_invalid_control_metadata_is_rejected(catalogs, fields):
    spec = {"class_path": "absent_robots:RobotCfg", **fields}
    catalogs.append(_entry("invalid", {"vendor.arm": spec}))
    with pytest.raises(
        TypeError,
        match="dof|control_parts|mimic|supports_dual_arm|vendor|morphology|arm_count",
    ):
        registry.list_robot_cfgs()


@pytest.fixture
def local_robot_assets(monkeypatch, tmp_path):
    """Build metadata-only configs from temporary files without downloads."""
    from importlib import import_module

    def local_asset(path: str) -> str:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '<robot name="test"><link name="base_link"/><link name="ee_link"/></robot>'
        )
        return str(target)

    for module_name in (
        "embodichain.data",
        "embodichain.lab.sim.cfg",
        "embodichain_robots.arx.arx5.cfg",
        "embodichain_robots.alohamini.aloha_mini.cfg",
        "embodichain_robots.tianji.marvin.cfg",
        "embodichain_robots.agilex.cobotmagic.cfg",
        "embodichain_robots.franka.panda",
        "embodichain_robots.flexiv.rizon4s.cfg",
        "embodichain_robots.the_robot_studio.so101.cfg",
        "embodichain_robots.universal_robots.ur_robot",
        "embodichain_robots.dexforce.w1.utils",
    ):
        module = import_module(module_name)
        if hasattr(module, "get_data_path"):
            monkeypatch.setattr(module, "get_data_path", local_asset)


@pytest.mark.parametrize(
    "name",
    [
        "UnitreeGo1",
        "UnitreeGo2",
        "UnitreeG1",
        "UnitreeH1_2",
        "ANYmalC",
        "MicroDuck",
        "AlohaMini",
        "ARX5",
        "TianjiMarvin",
        "CobotMagic",
        "DexforceW1",
        "FrankaPanda",
        "FlexivRizon4S",
        "SO101Follower",
        "URRobot",
        "DualArmRobot",
        "ur3",
        "ur3e",
        "ur5",
        "ur5e",
        "ur10",
        "ur10e",
    ],
)
def test_default_control_metadata_matches_robot_configuration(
    catalogs, local_robot_assets, name
):
    """Catalog summaries track real defaults without downloading robot assets."""
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))

    cfg = registry.resolve_robot_cfg(name)
    spec = registry._robot_specs()[name]

    assert spec["control_parts"] == {
        part: len(set(joints)) for part, joints in cfg.control_parts.items()
    }
    controlled = {joint for joints in cfg.control_parts.values() for joint in joints}
    assert spec["dof"] == len(controlled - set(spec.get("mimic_joints", {})))
    assert spec["control_part_mimics"] == {
        part: tuple(joint for joint in joints if joint in spec["mimic_joints"])
        for part, joints in cfg.control_parts.items()
        if any(joint in spec["mimic_joints"] for joint in joints)
    }


@pytest.mark.parametrize(
    "name,joint,stiffness,effort,height",
    [
        ("UnitreeGo1", "FR_hip_joint", 15.895242654143557, 23.7, 0.278),
        ("UnitreeGo2", "FL_hip_joint", 20.0, 23.5, 0.32),
        ("UnitreeG1", "left_knee_joint", 99.09842777666111, 139.0, 0.8),
        ("UnitreeH1_2", "left_knee_joint", 157.7, 300.0, 1.02),
        ("ANYmalC", "LF_HAA", 40.0, 80.0, 0.6),
        ("MicroDuck", "left_hip_yaw", 0.55, 0.96, 0.12),
    ],
)
def test_locomotion_presets_preserve_standalone_drive_defaults(
    catalogs, local_robot_assets, name, joint, stiffness, effort, height
) -> None:
    """Standalone presets retain the previous official task's effective drives."""
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))
    cfg = registry.resolve_robot_cfg(name)
    drives = cfg.joint_drive_props
    actual_stiffness = (
        drives.stiffness[joint]
        if isinstance(drives.stiffness, dict)
        else drives.stiffness
    )
    actual_effort = (
        drives.max_effort[joint]
        if isinstance(drives.max_effort, dict)
        else drives.max_effort
    )
    assert actual_stiffness == pytest.approx(stiffness)
    assert actual_effort == pytest.approx(effort)
    assert cfg.init_pos[2] == pytest.approx(height)
    assert cfg.root_props.fixed_base is False
    assert cfg.build_pk_chain is False


@pytest.mark.parametrize(
    "name,rules,parts,selected,default",
    [
        ("UnitreeGo2", {"FL_hip_joint": 12.5}, {}, {"FL_hip_joint": 12.5}, 20.0),
        (
            "UnitreeGo2",
            {".*hip_joint": 12.5},
            {},
            {
                "FL_hip_joint": 12.5,
                "FR_hip_joint": 12.5,
                "RL_hip_joint": 12.5,
                "RR_hip_joint": 12.5,
            },
            20.0,
        ),
        ("UnitreeGo2", {"locomotion": 12.5}, {}, None, 20.0),
        (
            "UnitreeGo2",
            {"front_legs": 12.5, "FL_hip_joint": 8.0},
            {"front_legs": ["F[LR]_.*"]},
            {
                "FL_hip_joint": 8.0,
                "FL_thigh_joint": 12.5,
                "FL_calf_joint": 12.5,
                "FR_hip_joint": 12.5,
                "FR_thigh_joint": 12.5,
                "FR_calf_joint": 12.5,
            },
            20.0,
        ),
        (
            "ANYmalC",
            {"LF_.*": 12.5},
            {},
            {"LF_HAA": 12.5, "LF_HFE": 12.5, "LF_KFE": 12.5},
            40.0,
        ),
    ],
)
def test_locomotion_overrides_are_isolated_and_roundtrip(
    catalogs, local_robot_assets, name, rules, parts, selected, default
) -> None:
    """Caller rules lower without default overlaps and survive serialization."""
    from copy import deepcopy
    from embodichain.lab.sim.spawn.descriptors import _joint_property_matches
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))
    overrides = {"joint_drive_props": {"stiffness": rules}, "control_parts": parts}
    original = deepcopy(overrides)
    cfg = registry.resolve_robot_cfg(name, overrides)
    snapshot = cfg.to_dict()
    restored = type(cfg).from_dict(snapshot)
    assert restored.to_dict() == snapshot
    names = cfg.control_parts["locomotion"]

    def resolved(candidate):
        return dict(
            _joint_property_matches(
                candidate.joint_drive_props.stiffness,
                names,
                property_name="stiffness",
                control_parts=candidate.control_parts,
            )
        )

    baseline = {
        joint: (
            40.0
            if name == "UnitreeGo2"
            and joint
            in {"FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"}
            else default
        )
        for joint in names
    }
    expected = (
        {joint: 12.5 for joint in names}
        if selected is None
        else {joint: selected.get(joint, baseline[joint]) for joint in names}
    )
    assert resolved(cfg) == expected
    assert resolved(restored) == expected
    assert resolved(registry.resolve_robot_cfg(name)) == baseline
    assert overrides == original


@pytest.mark.parametrize(
    "name,vendor,morphology,arm_count",
    [
        ("UnitreeGo1", "unitree", "quadruped", 0),
        ("UnitreeGo2", "unitree", "quadruped", 0),
        ("UnitreeG1", "unitree", "bipedal_humanoid", 2),
        ("UnitreeH1_2", "unitree", "bipedal_humanoid", 2),
        ("ANYmalC", "anybotics", "quadruped", 0),
        ("MicroDuck", "pollen_robotics", "biped", 0),
    ],
)
def test_locomotion_catalog_classifies_body_independently_of_arms(
    catalogs, name, vendor, morphology, arm_count
) -> None:
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))
    spec = registry._robot_specs()[name]
    assert (spec["vendor"], spec["morphology"], spec["arm_count"]) == (
        vendor,
        morphology,
        arm_count,
    )


def test_w1_variant_catalog_covers_body_and_independent_hand_registries():
    from embodichain_robots._catalog import ROBOT_CONFIGS, _W1_HANDS, _W1_VERSIONS
    from embodichain_robots.dexforce.w1.types import (
        DexforceW1Version,
        DexforceW1HandBrand,
    )

    assert set(_W1_VERSIONS) == {version.value for version in DexforceW1Version}
    assert set(_W1_HANDS) == {brand.value for brand in DexforceW1HandBrand}
    variants = ROBOT_CONFIGS["DexforceW1"]["variants"]
    expected_count = (len(DexforceW1Version) - 1) * (
        1 + (len(DexforceW1HandBrand) - 1) ** 2 + 1
    ) + 1
    assert len(variants) == expected_count
    assert {name for name in variants if name.startswith("v030/")} == {"v030/bare"}
    assert ROBOT_CONFIGS["DexforceW1"]["default_variant"] == "v021/BrainCo"


@pytest.mark.parametrize(
    "variant",
    [
        f"{version}/{hands}"
        for version in ("v021", "v022", "v025")
        for hands in (
            "BrainCo",
            "DH_PGC",
            "DH_PGC_M",
            "bare",
            "L=DH_PGC,R=DH_PGC_M",
            "L=DH_PGC_M,R=DH_PGC",
        )
    ]
    + ["v030/bare"],
)
def test_w1_variant_metadata_matches_config(catalogs, local_robot_assets, variant):
    from embodichain_robots._catalog import ROBOT_CONFIGS

    catalogs.append(_entry("official", ROBOT_CONFIGS))
    profile = ROBOT_CONFIGS["DexforceW1"]["variants"][variant]
    cfg = registry.resolve_robot_cfg("DexforceW1", variant=variant)
    joints = {joint for names in cfg.control_parts.values() for joint in names}

    assert cfg.version.value == variant.split("/")[0]
    assert profile["control_parts"] == {
        part: len(set(names)) for part, names in cfg.control_parts.items()
    }
    assert profile["dof"] == len(joints - set(profile["mimic_joints"]))
    assert profile["control_part_mimics"] == {
        part: tuple(joint for joint in names if joint in profile["mimic_joints"])
        for part, names in cfg.control_parts.items()
        if any(joint in profile["mimic_joints"] for joint in names)
    }
    if "hand_versions" in profile["defaults"]:
        assert set(profile["defaults"]["hand_versions"].values()) == {"v021"}


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


def test_external_robot_mounted_defaults_reach_gym_and_online_buffers(external_robot):
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        init_rollout_buffer_from_config,
    )
    from embodichain.lab.sim.sensors import CameraCfg

    cfg_cls, spec = external_robot
    spec["defaults"]["sensors"] = [
        {
            "sensor_type": "Camera",
            "uid": "vendor_wrist",
            "width": 16,
            "height": 12,
            "extrinsics": {"parent": "tip"},
        }
    ]
    original = deepcopy(spec)
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "default",
        "env": {},
        "robot": {"class_type": "vendor.arm"},
    }
    cfg = config_to_cfg(config)
    buffer = init_rollout_buffer_from_config(
        config, max_episode_steps=1, batch_size=1, state_dim=1
    )

    assert isinstance(cfg.robot, cfg_cls)
    assert cfg.robot.enable_sensor is True
    assert isinstance(cfg.robot.sensors[0], CameraCfg)
    assert cfg.robot.sensors[0].extrinsics.parent == "tip"
    assert set(buffer["obs", "sensor"].keys()) == {"vendor_wrist"}
    assert buffer["obs", "sensor", "vendor_wrist", "color"].shape == (1, 1, 12, 16, 4)
    assert spec == original


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


def test_declared_dual_arm_summary_matches_external_configuration(external_robot):
    """Compatible provider bases appear with their own composed joint counts."""
    _cfg_cls, spec = external_robot
    spec.update(
        dof=1,
        control_parts={"arm": 1},
        mimic_joints={},
        control_part_mimics={},
        supports_dual_arm=True,
    )
    profile = registry._robot_specs()["DualArmRobot"]["variants"]["vendor.arm"]
    cfg = registry.resolve_robot_cfg("DualArmRobot", {"base_robot": "vendor.arm"})

    assert profile["dof"] == 2
    assert profile["control_parts"] == {
        part: len(joints) for part, joints in cfg.control_parts.items()
    }


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
