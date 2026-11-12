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

"""Native hand defaults and production embodiment binding contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.robot_registry import resolve_gripper_defaults


@pytest.mark.parametrize(
    "name,variant,width,opened,closed",
    [
        ("FrankaPandaHand", None, 0.08, 0.04, 0.0),
        ("DHPGI14080", None, 0.1, 0.0, 0.024),
        ("CobotMagicV100Gripper", None, 0.1, 0.05, 0.0),
        ("PikaGripper", None, 0.094, 0.05, 0.0),
        ("TianjiMarvinGripper", "left", 0.095, 0.0, -0.05),
        ("TianjiMarvinGripper", "right", 0.095, 0.0, -0.05),
        ("RobotiqArg2F85", None, 0.085, 0.0, 0.8),
        ("RobotiqArg2F140", None, 0.15, 0.0, 0.7),
    ],
)
def test_native_defaults_preserve_qualified_width_and_commands(
    name, variant, width, opened, closed
):
    defaults = resolve_gripper_defaults(name, variant=variant)
    model = defaults
    assert model["max_opening_width"] == width
    leader = defaults["control_parts"]["eef"][0]
    assert defaults["joint_commands"]["open"][leader] == opened
    assert defaults["joint_commands"]["grasp"][leader] == closed
    json.dumps(defaults, allow_nan=False)
    defaults["joint_commands"]["open"][leader] = 99.0
    assert (
        resolve_gripper_defaults(name, variant=variant)["joint_commands"]["open"][
            leader
        ]
        == opened
    )


def test_static_descriptions_do_not_import_runtime_or_access_assets():
    code = """
import sys
class Fence:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("embodichain.lab", "dexsim", "open3d", "scipy", "trimesh", "torch", "embodichain.toolkits.graspkit")):
            raise AssertionError("Unexpected runtime import: " + fullname)
sys.meta_path.insert(0, Fence())
from embodichain.robot_registry import resolve_gripper_defaults
for name in ("FrankaPandaHand", "DHPGI14080", "CobotMagicV100Gripper", "PikaGripper", "RobotiqArg2F85", "RobotiqArg2F140"):
    resolve_gripper_defaults(name)
for side in ("left", "right"):
    resolve_gripper_defaults("TianjiMarvinGripper", variant=side)
assert "embodichain.data" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def _compose(
    tmp_path,
    *,
    logical=True,
    model="DHPGI14080",
    generator_kind=None,
    generator_policy=None,
    simulation=None,
    resources=None,
    gripper_overrides=None,
):
    from embodichain.lab.gym.utils.composition import resolve_gym_components

    declaration = {
        "embodiment_id": "test_hand",
        "simulation": {"class_type": "URRobot", "robot_type": "ur5"},
        "sensor": [],
        "grippers": {
            "tool": {
                "class_type": model,
                "mode": "mounted",
                "component_type": "hand",
                "motion_part": "arm",
                "control_part": "custom_fingers",
            }
        },
    }
    if logical:
        declaration["skill_profile"] = {
            "contract_id": "single_arm_parallel_gripper",
            "profile_id": "test_hand",
            "resources": [
                {
                    "resource_id": "primary",
                    "endpoints": [
                        {
                            "endpoint_id": "grasp",
                            "control_part": "custom_fingers",
                            "capabilities": ["interaction.grasp"],
                            "gripper": "tool",
                        }
                    ],
                }
            ],
        }
    if logical and generator_kind is not None:
        declaration["skill_profile"]["runtime_services"] = {
            "grasp_pose_generators": {"custom_fingers": {"kind": generator_kind}}
        }
    if logical and generator_policy is not None:
        declaration["skill_profile"]["runtime_services"] = {
            "grasp_pose_generators": {"custom_fingers": generator_policy}
        }
    declaration["simulation"].update(simulation or {})
    # Prefix variants must not replace each other's assembled files under xdist.
    declaration["simulation"].setdefault("urdf_cfg", {}).setdefault(
        "fpath_prefix", str(tmp_path / "assembled")
    )
    declaration["grippers"]["tool"].update(gripper_overrides or {})
    if resources is not None:
        declaration["skill_profile"]["resources"] = resources
    path = tmp_path / "hand.yaml"
    return resolve_gym_components(
        {"embodiment": {"component": "hand.yaml"}},
        base_dir=tmp_path,
        component_sources={str(path): declaration},
    )


def test_production_composition_fills_only_explicit_endpoints_and_owns_commands(
    tmp_path,
):
    physical = _compose(tmp_path)
    profile = physical.embodiment_skill_profile
    endpoint = profile["resources"][0]["endpoints"][0]
    assert endpoint["command_preset"] == "gripper.tool"
    assert profile["command_presets"][0]["commands"] == {
        "open": [0.0, 0.0],
        "grasp": [0.024, 0.024],
    }
    generators = profile["runtime_services"]["grasp_pose_generators"]
    assert set(generators) == {"custom_fingers"}
    assert generators["custom_fingers"]["gripper"]["catalog_id"] == "DHPGI14080"
    assert "tcp" not in physical.config["robot"].get("solver_cfg", {}).get("arm", {})
    assert _compose(tmp_path, logical=False).embodiment_skill_profile is None
    profile["command_presets"][0]["commands"]["open"][0] = 9.0
    assert _compose(tmp_path).embodiment_skill_profile["command_presets"][0][
        "commands"
    ]["open"] == [0.0, 0.0]


def test_runtime_service_binding_rejects_changed_physical_commands(tmp_path):
    from types import SimpleNamespace
    from embodichain.lab.task_program.integrations.configured import (
        _decode_grasp_generator,
    )

    physical = _compose(tmp_path)
    factory = _decode_grasp_generator(
        physical.embodiment_skill_profile["runtime_services"]["grasp_pose_generators"][
            "custom_fingers"
        ],
        path="test",
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(
            gripper_bindings=deepcopy(physical.config["robot"]["gripper_bindings"])
        )
    )
    factory.validate_binding(robot, "custom_fingers")
    robot.cfg.gripper_bindings["tool"]["defaults"]["joint_commands"]["grasp"][
        "gripper_finger1_joint_1"
    ] = 0.02
    with pytest.raises(ValueError, match="joint_commands"):
        factory.validate_binding(robot, "custom_fingers")


def test_configured_geometry_honors_corrected_limits_for_nonzero_opening(tmp_path):
    import hashlib
    from types import SimpleNamespace
    import torch
    from embodichain.data import get_data_path
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers
    from embodichain.lab.task_program.integrations.configured import (
        _decode_grasp_generator,
    )

    defaults = resolve_gripper_defaults("RobotiqArg2F85")
    opening = {
        name: value * 0.25
        for name, value in defaults["joint_commands"]["grasp"].items()
    }
    source = Path(get_data_path(defaults["fpath"]))
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    physical = _compose(
        tmp_path,
        model="RobotiqArg2F85",
        gripper_overrides={"commands": {"open": opening}},
    )
    values = deepcopy(physical.config["robot"])
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)
    factory = _decode_grasp_generator(
        physical.embodiment_skill_profile["runtime_services"]["grasp_pose_generators"][
            "custom_fingers"
        ],
        path="test",
    )
    factory.validate_binding(SimpleNamespace(cfg=cfg), "custom_fingers")
    geometry = factory().collision_geometry
    assert geometry.provenance["joint_positions"] == opening
    assert geometry.provenance["joint_positions"]["left_inner_finger_joint"] == -0.2
    assert (
        geometry.provenance["files_sha256"][str(source)]
        == original_hash
        == defaults["source_sha256"]
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert torch.isfinite(geometry.points).all()


@pytest.mark.parametrize(
    "name,variant",
    [
        ("FrankaPandaHand", None),
        ("DHPGI14080", None),
        ("CobotMagicV100Gripper", None),
        ("PikaGripper", None),
        ("TianjiMarvinGripper", "left"),
        ("TianjiMarvinGripper", "right"),
        ("RobotiqArg2F85", None),
        ("RobotiqArg2F140", None),
    ],
)
def test_standalone_gripper_cfg_roundtrip_keeps_native_commands(name, variant):
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.cfg import ParallelJawGripperCfg

    cfg = resolve_robot_cfg(name, {"variant": variant} if variant else {})
    assert isinstance(cfg, ParallelJawGripperCfg)
    assert cfg.solver_cfg == {}
    serialized = cfg.to_dict()
    restored = type(cfg).from_dict(serialized)
    assert restored.to_dict() == serialized
    assert cfg.build_pk_serial_chain() == {}
    cfg.joint_commands["open"][next(iter(cfg.joint_commands["open"]))] = 99.0
    assert restored.joint_commands != cfg.joint_commands


@pytest.mark.parametrize(
    "change,error",
    [
        ({"mode": "guessed"}, "mode"),
        ({"root_from_tcp": [[0.0] * 4 for _ in range(4)]}, "transform"),
        ({"commands": {"open": {"gripper_finger1_joint_1": 0.0}}}, "cover"),
        (
            {
                "commands": {
                    "open": {
                        "gripper_finger1_joint_1": 99.0,
                        "gripper_finger2_joint_1": 99.0,
                    }
                }
            },
            "limits",
        ),
    ],
)
def test_production_binding_rejects_invalid_physical_declarations(
    tmp_path, change, error
):
    from embodichain.lab.gym.utils.composition import resolve_gym_components

    declaration = {
        "embodiment_id": "invalid",
        "simulation": {"class_type": "URRobot", "robot_type": "ur5"},
        "sensor": [],
        "grippers": {
            "tool": {
                "class_type": "DHPGI14080",
                "mode": "mounted",
                "component_type": "hand",
                "motion_part": "arm",
                "control_part": "eef",
                **change,
            }
        },
    }
    with pytest.raises(ValueError, match=error):
        resolve_gym_components(
            {"embodiment": {"component": "hand.yaml"}},
            base_dir=tmp_path,
            component_sources={str(tmp_path / "hand.yaml"): declaration},
        )


def test_mounted_drive_defaults_and_explicit_overrides_share_production_path(tmp_path):
    from embodichain.robot_registry import resolve_robot_cfg

    physical = _compose(tmp_path, logical=False)
    values = deepcopy(physical.config["robot"])
    name = values.pop("class_type")
    cfg = resolve_robot_cfg(name, values)
    assert cfg.joint_drive_props.stiffness["gripper_finger1_joint_1"] == 1000.0
    assert cfg.joint_drive_props.damping["gripper_finger2_joint_1"] == 100.0
    assert cfg.joint_drive_props.max_effort["gripper_finger1_joint_1"] == 40.0
    values["joint_drive_props"]["stiffness"]["gripper_finger1_joint_1"] = 250.0
    assert (
        resolve_robot_cfg(name, values).joint_drive_props.stiffness[
            "gripper_finger1_joint_1"
        ]
        == 250.0
    )


def test_static_decode_rejects_missing_physical_geometry(tmp_path):
    from embodichain.lab.task_program.integrations.configured import (
        _decode_grasp_generator,
    )

    description = resolve_gripper_defaults("DHPGI14080")
    description.pop("root_link")
    description["collision_links"] = []
    with pytest.raises(ValueError, match="root_link|collision_links"):
        _decode_grasp_generator(
            {"kind": "antipodal_parallel_jaw", "gripper": description}, path="test"
        )


def test_runtime_service_binding_rejects_changed_contact_frame(tmp_path):
    from types import SimpleNamespace
    from embodichain.lab.task_program.integrations.configured import (
        _decode_grasp_generator,
    )

    physical = _compose(tmp_path)
    factory = _decode_grasp_generator(
        physical.embodiment_skill_profile["runtime_services"]["grasp_pose_generators"][
            "custom_fingers"
        ],
        path="test",
    )
    actual = deepcopy(physical.config["robot"]["gripper_bindings"])
    actual["tool"]["defaults"]["root_from_contact_frame"][2][3] += 0.01
    with pytest.raises(ValueError, match="calibration"):
        factory.validate_binding(
            SimpleNamespace(cfg=SimpleNamespace(gripper_bindings=actual)),
            "custom_fingers",
        )


@pytest.mark.parametrize("mutation", ["scale", "mount", "mount_without_solver"])
def test_runtime_binding_rejects_changed_collision_geometry_or_installation(
    tmp_path, mutation
):
    import xml.etree.ElementTree as ET
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(_compose(tmp_path, logical=False).config["robot"])
    if mutation == "mount_without_solver":
        values["solver_cfg"] = {}
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    cfg.urdf_cfg.fpath_prefix = str(tmp_path / "assembled")
    cfg.urdf_cfg.fpath = str(tmp_path / "unused.urdf")
    prepare_robot_grippers(cfg)
    original = Path(cfg.fpath)
    xml = ET.parse(original)
    for mesh in xml.findall(".//mesh"):
        mesh.set("filename", str((original.parent / mesh.get("filename")).resolve()))
    if mutation == "scale":
        root = cfg.gripper_bindings["tool"]["root_link"]
        xml.find(f"./link[@name='{root}']/collision/geometry/mesh").set(
            "scale", "2 2 2"
        )
    else:
        root = cfg.gripper_bindings["tool"]["root_link"]
        joint = next(
            j for j in xml.findall("joint") if j.find("child").get("link") == root
        )
        joint.find("origin").set("xyz", "0 0 0.1")
    cfg.fpath = str(tmp_path / "modified.urdf")
    xml.write(cfg.fpath)
    with pytest.raises(ValueError, match="scale|installation"):
        prepare_robot_grippers(cfg)


def test_joint_only_binding_needs_no_solver_and_accepts_source_order_limit_arrays(
    tmp_path,
):
    import numpy as np
    import xml.etree.ElementTree as ET
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(_compose(tmp_path, logical=False).config["robot"])
    values["solver_cfg"] = {}
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)
    count = sum(j.get("type") != "fixed" for j in ET.parse(cfg.fpath).findall("joint"))
    cfg.qpos_limits = np.tile([-10.0, 10.0], (count, 1))
    prepare_robot_grippers(cfg)
    cfg.qpos_limits = {"gripper_finger[12]_joint_1": [0.0, 0.01]}
    with pytest.raises(ValueError, match="limits"):
        prepare_robot_grippers(cfg)


def test_native_drive_mapping_and_optional_resource_corrections(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from embodichain.lab.task_program.integrations.configured import (
        _decode_grasp_generator,
    )
    from embodichain.robot_registry import resolve_robot_cfg
    import embodichain_robots.dh_robotics.pgi_140_80._gripper_defaults as provider

    original = provider.get_gripper_defaults()
    original["joint_drive_props"]["stiffness"] = {
        "gripper_finger1_joint_1": 1000.0,
        "gripper_finger2_joint_1": 900.0,
    }
    original.pop("mesh_paths")
    monkeypatch.setattr(
        provider, "get_gripper_defaults", lambda variant=None: deepcopy(original)
    )
    physical = _compose(tmp_path)
    values = deepcopy(physical.config["robot"])
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    assert cfg.joint_drive_props.stiffness["gripper_finger1_joint_1"] == 1000.0
    assert cfg.joint_drive_props.stiffness["gripper_finger2_joint_1"] == 900.0
    factory = _decode_grasp_generator(
        physical.embodiment_skill_profile["runtime_services"]["grasp_pose_generators"][
            "custom_fingers"
        ],
        path="test",
    )
    factory.validate_binding(SimpleNamespace(cfg=cfg), "custom_fingers")


def test_generic_dual_arm_composition_retains_independent_bound_hands(tmp_path):
    import numpy as np
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.robots.dual_arm import build_dual_arm_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(_compose(tmp_path, logical=False).config["robot"])
    base = resolve_robot_cfg(values.pop("class_type"), values)
    left, right = np.eye(4), np.eye(4)
    left[1, 3] = 0.3
    right[1, 3] = -0.3
    dual = build_dual_arm_cfg(base, {"left": left, "right": right})
    assert set(dual.gripper_bindings) == {"left_tool", "right_tool"}
    assert {"left_hand", "right_hand"} <= dual.urdf_cfg.components.keys()
    prepare_robot_grippers(dual)
    assert dual.solver_cfg["left_arm"].tcp[2][3] == 0.15
    assert dual.solver_cfg["right_arm"].tcp[2][3] == 0.15
    dual.gripper_bindings["left_tool"]["defaults"]["default_root_from_tcp"][2][
        3
    ] += 0.01
    assert (
        dual.gripper_bindings["right_tool"]["defaults"]["default_root_from_tcp"][2][3]
        == 0.15
    )
    assert (
        base.gripper_bindings["tool"]["defaults"]["default_root_from_tcp"][2][3] == 0.15
    )


@pytest.mark.parametrize("form", ["array", "tensor", "list", "regex"])
def test_generic_dual_bound_hands_preserve_source_order_limit_overrides(tmp_path, form):
    import numpy as np
    import torch
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.robots.dual_arm import build_dual_arm_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(
        _compose(tmp_path, logical=False, model="RobotiqArg2F85").config["robot"]
    )
    base = resolve_robot_cfg(values.pop("class_type"), values)
    rows = np.tile(
        [-10.0, 10.0], (12, 1)
    )  # Six UR joints and all six authored hand joints.
    base.qpos_limits = (
        rows
        if form == "array"
        else torch.from_numpy(rows) if form == "tensor" else rows.tolist()
    )
    if form == "regex":
        base.qpos_limits = {
            "left_inner_finger_joint|right_inner_finger_joint": [-10.0, 10.0]
        }
    dual = build_dual_arm_cfg(base, {"left": np.eye(4), "right": np.eye(4)})
    assert len(dual.qpos_limits) == (4 if form == "regex" else 24)
    assert all(value == [-10.0, 10.0] for value in dual.qpos_limits.values())
    prepare_robot_grippers(dual)


@pytest.mark.parametrize("supported", [[], ["antipodal_parallel_jaw", "other_kind"]])
def test_generator_capabilities_are_explicit_and_do_not_grant_undeclared_endpoints(
    tmp_path, monkeypatch, supported
):
    import embodichain_robots.dh_robotics.pgi_140_80._gripper_defaults as provider

    values = provider.get_gripper_defaults()
    values["grasp_generators"] = supported
    monkeypatch.setattr(
        provider, "get_gripper_defaults", lambda variant=None: deepcopy(values)
    )
    assert _compose(tmp_path, logical=False).embodiment_skill_profile is None
    if supported:
        with pytest.raises(ValueError, match="generator"):
            _compose(tmp_path)
        selected = _compose(tmp_path, generator_kind="antipodal_parallel_jaw")
        assert (
            selected.embodiment_skill_profile["runtime_services"][
                "grasp_pose_generators"
            ]["custom_fingers"]["kind"]
            == "antipodal_parallel_jaw"
        )
    else:
        profile = _compose(tmp_path).embodiment_skill_profile
        assert profile["command_presets"][0]["commands"]["grasp"] == [0.024, 0.024]
        assert profile["runtime_services"]["grasp_pose_generators"] == {}
        assert (
            _compose(tmp_path, generator_policy={}).embodiment_skill_profile[
                "runtime_services"
            ]["grasp_pose_generators"]
            == {}
        )
        with pytest.raises(ValueError, match="does not declare generator"):
            _compose(tmp_path, generator_kind="antipodal_parallel_jaw")


def test_bound_endpoint_cannot_select_an_undeclared_generator(tmp_path):
    with pytest.raises(ValueError, match="does not declare generator"):
        _compose(tmp_path, generator_kind="other_kind")


def test_physical_fields_and_capability_declaration_live_on_the_concrete_cfg():
    from embodichain.robot_registry import resolve_robot_cfg

    cfg = resolve_robot_cfg("DHPGI14080")
    assert not hasattr(cfg, "grasp_cfg")
    assert cfg.root_link == "gripper_base_link_1"
    assert cfg.min_opening_width == 0.005 and cfg.max_opening_width == 0.1
    assert cfg.grasp_generators == ["antipodal_parallel_jaw"]


def test_concrete_cfg_overrides_export_owned_numeric_inputs_without_a_model_cfg():
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_geometry import grasp_generator_parameters

    source = resolve_gripper_defaults("DHPGI14080")
    commands = deepcopy(source["joint_commands"])
    commands["open"] = {name: 0.001 for name in commands["open"]}
    cfg = resolve_robot_cfg(
        "DHPGI14080",
        {"max_opening_width": 0.08, "grasp_generators": [], "joint_commands": commands},
    )
    assert cfg.grasp_generators == [] and cfg.init_qpos == [0.001, 0.001]
    inputs = grasp_generator_parameters(cfg.to_dict())
    assert inputs == {
        "model_id": "dh_pgi_140_80",
        "min_opening_width": 0.005,
        "max_opening_width": 0.08,
    }
    inputs["max_opening_width"] = 99.0
    assert cfg.max_opening_width == 0.08
    assert resolve_gripper_defaults("DHPGI14080")["max_opening_width"] == 0.1


def test_w1_pika_partial_prefixes_match_assembled_hand_names():
    """A bare-body arm-prefix patch retains the assembler's hand prefixes."""
    import yaml
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers
    from embodichain.robot_registry import resolve_robot_cfg

    source = (
        Path(__file__).resolve().parents[1]
        / "extensions/embodichain_tasks/configs/components/embodiments/dexforce_w1_030_pika.yaml"
    )
    declaration = yaml.safe_load(source.read_text())
    declaration["simulation"]["urdf_cfg"]["component_prefix"] = [
        ("left_arm", None),
        ("right_arm", None),
    ]
    physical = resolve_gym_components(
        {"embodiment": {"component": str(source)}},
        base_dir=source.parent,
        component_sources={str(source): declaration},
    )
    values = deepcopy(physical.config["robot"])
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)

    assert cfg.control_parts["left_eef"] == ["LEFT_FINGER1_JOINT", "LEFT_FINGER2_JOINT"]
    assert cfg.control_parts["right_eef"] == [
        "RIGHT_FINGER1_JOINT",
        "RIGHT_FINGER2_JOINT",
    ]
    assert cfg.gripper_bindings["left_eef"]["root_link"] == "left_gripper_root_link"
    assert cfg.gripper_bindings["right_eef"]["root_link"] == "right_gripper_root_link"


@pytest.mark.parametrize("second_preset", [None, "alternate"])
def test_shared_hand_endpoints_reuse_generated_presets_and_service(
    tmp_path, second_preset
):
    endpoint = {
        "endpoint_id": "grasp",
        "control_part": "custom_fingers",
        "capabilities": ["interaction.grasp"],
        "gripper": "tool",
    }
    alias = deepcopy(endpoint)
    if second_preset is not None:
        alias["command_preset"] = second_preset
    physical = _compose(
        tmp_path,
        generator_kind="antipodal_parallel_jaw",
        resources=[
            {"resource_id": "primary", "endpoints": [endpoint]},
            {"resource_id": "alias", "endpoints": [alias]},
        ],
    )
    profile = physical.embodiment_skill_profile
    assert len(profile["command_presets"]) == (1 if second_preset is None else 2)
    generators = profile["runtime_services"]["grasp_pose_generators"]
    assert list(generators) == ["custom_fingers"]
    assert generators["custom_fingers"]["kind"] == "antipodal_parallel_jaw"
    for resource in profile["resources"]:
        assert "gripper" not in resource["endpoints"][0]
    # Authored duplicates still violate binding ownership.
    from embodichain.lab.gym.utils._gripper_composition import _expand_grippers

    with pytest.raises(ValueError, match="duplicates explicit command preset"):
        _expand_grippers(
            {},
            {
                "command_presets": [{"preset_id": "gripper.tool"}],
                "resources": [{"endpoints": [endpoint]}],
            },
            {
                "tool": {
                    "class_type": "DHPGI14080",
                    "mode": "mounted",
                    "component_type": "hand",
                    "motion_part": "arm",
                    "control_part": "custom_fingers",
                }
            },
        )


def test_mounted_explicit_nulls_preserve_source_ownership(tmp_path):
    physical = _compose(
        tmp_path,
        logical=False,
        simulation={"joint_drive_props": {"stiffness": None}, "solver_cfg": None},
    )
    robot = physical.config["robot"]
    assert robot["joint_drive_props"]["stiffness"] is None
    assert robot["solver_cfg"] is None
    assert robot["joint_drive_props"]["damping"]["gripper_finger1_joint_1"] == 100.0


def test_authored_limit_regex_prevents_duplicate_hand_defaults(tmp_path):
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    rule = "(?:left|right)_inner_finger_joint"
    physical = _compose(
        tmp_path,
        logical=False,
        model="RobotiqArg2F85",
        simulation={"qpos_limits": {rule: [-0.8, 0.0]}},
    )
    values = deepcopy(physical.config["robot"])
    assert values["qpos_limits"] == {rule: [-0.8, 0.0]}
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)


def test_reordered_open_command_preserves_native_startup_order():
    from embodichain.robot_registry import resolve_robot_cfg

    defaults = resolve_gripper_defaults("RobotiqArg2F85")
    commands = deepcopy(defaults["joint_commands"])
    commands["open"] = dict(reversed(list(commands["grasp"].items())))
    cfg = resolve_robot_cfg("RobotiqArg2F85", {"joint_commands": commands})
    assert cfg.init_qpos == [
        commands["grasp"][name] for name in defaults["control_parts"]["eef"]
    ]
    assert cfg.init_qpos != list(commands["open"].values())
    explicit = [0.0] * len(cfg.init_qpos)
    assert (
        resolve_robot_cfg(
            "RobotiqArg2F85", {"joint_commands": commands, "init_qpos": explicit}
        ).init_qpos
        == explicit
    )


@pytest.mark.parametrize("end_at_root", [True, False])
def test_mounted_tcp_uses_actual_fixed_chain_and_survives_roundtrip(
    tmp_path, end_at_root
):
    import numpy as np
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    mount, tcp = np.eye(4), np.eye(4)
    mount[2, 3], tcp[2, 3] = 0.10, 0.05
    end = "gripper_base_link_1" if end_at_root else "ee_link"
    expected = tcp.copy()
    if not end_at_root:
        expected = mount @ tcp
    values = deepcopy(
        _compose(
            tmp_path,
            logical=False,
            simulation={
                "solver_cfg": {"arm": {"end_link_name": end, "tcp": expected.tolist()}}
            },
            gripper_overrides={
                "transform": mount.tolist(),
                "root_from_tcp": tcp.tolist(),
            },
        ).config["robot"]
    )
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)
    np.testing.assert_allclose(cfg.solver_cfg["arm"].tcp, expected, atol=1e-6)
    serialized = cfg.to_dict()
    assert serialized["solver_cfg"]["arm"]["tcp"] == cfg.solver_cfg["arm"].tcp
    restored = type(cfg).from_dict(serialized)
    prepare_robot_grippers(restored)
    np.testing.assert_allclose(restored.solver_cfg["arm"].tcp, expected, atol=1e-6)
    # A post-selection explicit override must still reach authoritative validation.
    from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg

    wrong = expected.copy()
    wrong[2, 3] += 0.02
    with pytest.raises(ValueError, match="TCP disagrees"):
        prepare_robot_grippers(
            merge_robot_cfg(cfg, {"solver_cfg": {"arm": {"tcp": wrong.tolist()}}})
        )


@pytest.mark.parametrize("prepared", [False, True])
def test_mounted_implicit_tcp_survives_preassembly_roundtrip(tmp_path, prepared):
    import numpy as np
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    mount, hand_tcp = np.eye(4), np.eye(4)
    mount[2, 3], hand_tcp[2, 3] = 0.10, 0.05
    values = deepcopy(
        _compose(
            tmp_path,
            logical=False,
            simulation={"solver_cfg": {"arm": {"end_link_name": "ee_link"}}},
            gripper_overrides={
                "transform": mount.tolist(),
                "root_from_tcp": hand_tcp.tolist(),
            },
        ).config["robot"]
    )
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    if prepared:
        prepare_robot_grippers(cfg)
    serialized = cfg.to_dict()
    restored = type(cfg).from_dict(json.loads(json.dumps(serialized)))
    prepare_robot_grippers(restored)
    np.testing.assert_allclose(
        restored.solver_cfg["arm"].tcp, mount @ hand_tcp, rtol=0, atol=1e-6
    )
    assert "tcp" not in serialized["solver_cfg"]["arm"]
    assert "explicit_solver_tcp" not in restored.gripper_bindings["tool"]
    prepare_robot_grippers(cfg)
    np.testing.assert_allclose(
        cfg.solver_cfg["arm"].tcp, restored.solver_cfg["arm"].tcp, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "mutation", ["remove", "add", "parent", "multiplier", "offset"]
)
def test_actual_mimic_topology_must_match_source(tmp_path, mutation):
    import xml.etree.ElementTree as ET
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(_compose(tmp_path, logical=False).config["robot"])
    values["solver_cfg"] = {}
    cfg = resolve_robot_cfg(values.pop("class_type"), values)
    prepare_robot_grippers(cfg)
    original = Path(cfg.fpath)
    document = ET.parse(original)
    for mesh in document.findall(".//mesh"):
        mesh.set("filename", str((original.parent / mesh.get("filename")).resolve()))
    follower = document.find("./joint[@name='gripper_finger2_joint_1']")
    mimic = follower.find("mimic")
    if mutation == "remove":
        follower.remove(mimic)
    elif mutation == "add":
        document.find("./joint[@name='gripper_finger1_joint_1']").append(
            deepcopy(mimic)
        )
    else:
        mimic.set(
            {"parent": "joint"}.get(mutation, mutation),
            (
                "gripper_finger2_joint_1"
                if mutation == "parent"
                else "0.001" if mutation == "offset" else "0.5"
            ),
        )
        if mutation in {"multiplier", "offset"}:
            # Match the altered relation so command consistency alone cannot
            # hide a mismatch against the pinned source topology.
            for command in cfg.gripper_bindings["tool"]["defaults"][
                "joint_commands"
            ].values():
                leader = command["gripper_finger1_joint_1"]
                command["gripper_finger2_joint_1"] = leader * (
                    0.5 if mutation == "multiplier" else 1.0
                ) + (0.001 if mutation == "offset" else 0.0)
    cfg.fpath = str(tmp_path / "tampered.urdf")
    document.write(cfg.fpath)
    with pytest.raises(ValueError, match="mimic"):
        prepare_robot_grippers(cfg)


@pytest.mark.parametrize("prefix", ["tool_", "gripper_"])
def test_dual_hand_names_controls_limits_and_drives_follow_assembler(tmp_path, prefix):
    import numpy as np
    import xml.etree.ElementTree as ET
    from embodichain.lab.sim.cfg import MimicComplianceCfg
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim.robots.dual_arm import build_dual_arm_cfg
    from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

    values = deepcopy(
        _compose(
            tmp_path,
            logical=False,
            simulation={
                "urdf_cfg": {"component_prefix": [("hand", prefix)]},
                "joint_drive_props": {"stiffness": {f"{prefix}.*joint_1": 321.0}},
                "qpos_limits": {f"{prefix}.*joint_1": [0.0, 0.04]},
            },
        ).config["robot"]
    )
    base = resolve_robot_cfg(values.pop("class_type"), values)
    master = base.gripper_bindings["tool"]["joint_names"]["gripper_finger1_joint_1"]
    base.mimic_compliance = {master: MimicComplianceCfg(0.0, 0.0)}

    dual = build_dual_arm_cfg(base, {"left": np.eye(4), "right": np.eye(4)})
    prepare_robot_grippers(dual)
    joints = ET.parse(dual.fpath).findall("joint")
    names = {joint.get("name") for joint in joints}
    mimic_masters = {
        mimic.get("joint") for joint in joints for mimic in joint.findall("mimic")
    }
    assert dual.mimic_compliance == {
        "left_gripper_finger1_joint_1": MimicComplianceCfg(0.0, 0.0),
        "right_gripper_finger1_joint_1": MimicComplianceCfg(0.0, 0.0),
    }
    assert dual.mimic_compliance.keys() <= mimic_masters
    dual.mimic_compliance["left_gripper_finger1_joint_1"].damping_ratio = 1.0
    assert dual.mimic_compliance["right_gripper_finger1_joint_1"].damping_ratio == 0.0
    assert base.mimic_compliance[master].damping_ratio == 0.0
    for side in ("left", "right"):
        expected = [
            f"{side}_gripper_finger1_joint_1",
            f"{side}_gripper_finger2_joint_1",
        ]
        assert dual.control_parts[f"{side}_custom_fingers"] == expected
        assert set(expected) <= names
        assert (
            list(dual.gripper_bindings[f"{side}_tool"]["joint_names"].values())
            == expected
        )
        for name in expected:
            assert dual.qpos_limits[name] == [0.0, 0.04]
            assert dual.joint_drive_props.stiffness[name] == 321.0
