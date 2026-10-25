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

"""Production configuration contracts for the three Tianji Marvin demos."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import xml.etree.ElementTree as ET

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils._component_composition import (
    _resolve_gym_components,
    _validate_scene_binding_targets,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.sim.cfg import DefaultPhysicsCfg
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.task_program.integrations._configured_services import (
    _JointPositionConstraintObserver,
)
from embodichain.lab.task_program.semantics import (
    BinaryEffectClause,
    BinaryEffectEvidenceQuery,
    BinaryEvidenceKind,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    EffectEvidenceCollectionContext,
    EffectEvidenceSourceRef,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)
from embodichain.utils.utility import load_config
from embodichain_robots import TianjiMarvinCfg

_CONFIGS = (
    Path(__file__).parents[4]
    / "extensions/embodichain_tasks/configs/tasks/manipulation"
)
_DEPLOYMENTS = (
    ("repeated_pick_place", "TaskProgramRepeatedPickPlace-TianjiMarvin-v1", 6),
    ("hand_over", "HandOver-TianjiMarvin-v1", 1),
    ("tableware/pour_water", "PourWater-TianjiMarvin-v1", 4),
)


def test_marvin_embodiment_explicitly_opts_into_pinned_simulation_inertials() -> None:
    """Check recorded demo overlays, without certifying the asset's units."""
    source = TianjiMarvinCfg.from_dict({})
    asset = Path(source.fpath)
    # The calibration belongs to this model snapshot, not arbitrary robot files.
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == (
        "691a5f5a08d3335a1e1a50487b0f886f17eb30da4ea7703bf320a0bf9eac874f"
    )
    path = _CONFIGS / "tableware/pour_water/task.tianji_marvin.yaml"
    deployed = config_to_cfg(load_config(path), source_path=path).robot
    inertials = {
        link.get("name"): link.find("inertial")
        for link in ET.parse(asset).getroot().findall("link")
        if link.find("inertial") is not None
    }
    assert source.link_attrs == {}
    assert set(deployed.link_attrs) == set(inertials)
    for name, inertial in inertials.items():
        assert deployed.link_attrs[name].link_names_expr == [name]
        props = deployed.link_attrs[name].attrs.mass_props
        assert props.mass == pytest.approx(
            float(inertial.find("mass").get("value")) * 1000
        )
        values = inertial.find("inertia").attrib
        xx, xy, xz, yy, yz, zz = (
            float(values[k]) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        )
        expected = (
            torch.tensor(
                [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=torch.float64
            )
            * 1000
        )
        actual = torch.tensor(props.inertia, dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        assert torch.linalg.eigvalsh(actual).min() > 0
        assert props.com_position is None
        assert props.com_quaternion is None
        assert deployed.link_attrs[name].attrs.collision_props is None


def _deployment(task: str):
    """Resolve the actual installed-resource composition without a simulator."""
    path = _CONFIGS / task / "task.tianji_marvin.yaml"
    payload = load_config(path)
    physical = _resolve_gym_components(payload, base_dir=path.parent)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding, simulation=physical.scene_config
    )
    return path, payload, physical, deployment


@pytest.mark.parametrize("task,env_id,segments", _DEPLOYMENTS)
def test_marvin_deployments_preflight_and_register(
    task: str, env_id: str, segments: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each real task selects compatible contracts and registers EmbodiedEnv."""
    monkeypatch.setattr(
        "embodichain_robots.tianji_marvin.get_data_path", lambda value: value
    )
    path, payload, physical, deployment = _deployment(task)
    registration = deployment.integration.registration
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=registration.catalog,
    )
    compiled = registration.catalog.preflight(program)
    cfg = config_to_cfg(payload, source_path=path)

    assert payload["id"] == env_id
    assert REGISTERED_ENVS[env_id].cls is EmbodiedEnv
    assert compiled.segment_count == segments
    assert registration.robot_profile_binding.profile_id == "tianji_marvin"
    assert isinstance(cfg.sim_cfg.physics_config, DefaultPhysicsCfg)
    assert [camera.uid for camera in cfg.sensor] == ["cam_high"]
    assert cfg.sensor[0].enable_color
    assert "init_qpos" not in payload["embodiment"].get("overrides", {})
    assert cfg.robot.init_qpos == TianjiMarvinCfg.from_dict({}).init_qpos
    assert physical.config["robot"]["init_qpos"] == cfg.robot.init_qpos
    assert [camera.uid for camera in cfg.robot.sensors] == [
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert all(
        camera.enable_color and camera.enable_depth for camera in cfg.robot.sensors
    )
    assert set(cfg.robot.control_parts) == {
        "left_arm",
        "right_arm",
        "left_eef",
        "right_eef",
    }
    evidence = physical.embodiment_skill_profile["runtime_services"][
        "control_part_evidence"
    ]
    assert evidence["open_qpos"] == [0.0, 0.0]
    assert "object_ids" not in evidence


def test_marvin_task_resource_roles_and_effect_authority() -> None:
    """Task defaults bind existing logical hands and keep assurance explicit."""
    _, _, _, pick = _deployment("repeated_pick_place")
    _, _, _, handover = _deployment("hand_over")
    _, _, _, pour = _deployment("tableware/pour_water")
    pick_profile = pick.integration.registration.robot_profile_binding
    handover_profile = handover.integration.registration.robot_profile_binding
    pour_profile = pour.integration.registration.robot_profile_binding

    assert dict(pick_profile.defaults["pick_up"]) == {"primary": "left"}
    assert dict(pick_profile.defaults["place"]) == {"primary": "left"}
    assert dict(handover_profile.defaults["hand_over"]) == {
        "source": "left",
        "destination": "right",
    }
    assert all(
        dict(slots) == {"primary": "right"} for slots in pour_profile.defaults.values()
    )
    assert pick_profile.presets[0].effect_assurance.value == "verified"
    assert handover_profile.presets[0].effect_assurance.value == "verified"
    assert pour_profile.presets[0].effect_assurance.value == "projected"


@pytest.mark.parametrize("side", ["left", "right"])
def test_marvin_grasp_tcp_aligns_with_physical_hand_axes(
    side: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task-facing frame opens on hand X and approaches on hand Z."""
    monkeypatch.setattr(
        "embodichain_robots.tianji_marvin.get_data_path", lambda value: value
    )
    path, payload, _, _ = _deployment("repeated_pick_place")
    cfg = config_to_cfg(payload, source_path=path)
    # Both URDF hand-tool fixed joints rotate by Rz(-pi/2) @ Ry(-pi/2).
    hand_to_tool = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    hand_to_tcp = hand_to_tool @ torch.tensor(cfg.robot.solver_cfg[f"{side}_arm"].tcp)
    torch.testing.assert_close(hand_to_tcp[:3, :3], torch.eye(3))
    torch.testing.assert_close(hand_to_tcp[:3, 3], torch.tensor([0.0, 0.0, 0.1125]))


def test_marvin_stack_open_fingers_clear_the_cube_support() -> None:
    """The inclined grasp must not drive an open finger through the table."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    import trimesh

    path, payload, physical, deployment = _deployment("tableware/stack_blocks_two")
    cfg = config_to_cfg(payload, source_path=path)
    asset = Path(cfg.robot.fpath)
    robot_xml = ET.parse(asset).getroot()
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    grasp = preset.skill_option_templates["pick"].fixed_object_to_eef.numpy()

    def origin_pose(origin: ET.Element | None) -> np.ndarray:
        result = np.eye(4)
        if origin is not None:
            result[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        return result

    tool_joint = robot_xml.find("joint[@name='LEFT_HAND_TOOL']")
    assert tool_joint is not None
    hand_to_tcp = origin_pose(tool_joint.find("origin")) @ np.asarray(
        cfg.robot.solver_cfg["left_arm"].tcp
    )
    block = next(
        item
        for item in physical.scene_config["rigid_object"]
        if item["uid"] == "block_2"
    )
    support_z = -0.5 * block["shape"]["size"][2] * block["body_scale"][2]
    for index in (1, 2):
        joint = robot_xml.find(f"joint[@name='LEFT_HAND_FINGER_{index}']")
        collision = robot_xml.find(
            f"link[@name='left_hand_finger_{index}_link']/collision"
        )
        assert joint is not None and collision is not None
        mesh = collision.find("geometry/mesh")
        assert mesh is not None
        fingers = trimesh.load(
            asset.parent / mesh.attrib["filename"], force="mesh", process=False
        )
        fingers.apply_scale(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
        fingers.apply_transform(
            grasp
            @ np.linalg.inv(hand_to_tcp)
            @ origin_pose(joint.find("origin"))
            @ origin_pose(collision.find("origin"))
        )
        # Open jaws at the nominal grip must clear both scene contact offsets.
        assert fingers.bounds[0, 2] - support_z > 0.004


@pytest.mark.parametrize("side", ["left", "right"])
def test_marvin_constraint_evidence_reads_negative_two_finger_displacement(
    side: str,
) -> None:
    """A closing Marvin hand crosses the aperture threshold in either hand."""
    part = f"{side}_eef"
    measured = torch.tensor(
        [[0.0, 0.0], [-0.0039, -0.0039], [-0.0041, -0.0041], [-0.02, -0.02]]
    )
    robot = SimpleNamespace(get_qpos=lambda *, name: measured if name == part else None)
    expectation = HeldObjectStateExpectation(
        expectation_id="holder",
        relation=HeldObjectRelation.ATTACHED,
        object_id="cube",
        slot_id="primary",
        resource_id=side,
        task_state_key=side,
    )
    clause = BinaryEffectClause(
        clause_id="holder.constraint",
        expectation_id="holder",
        source=EffectEvidenceSourceRef(
            CONTROL_PART_EVIDENCE_PROVIDER_ID,
            CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
            ControlPartEvidenceAddress(part, CONSTRAINT_EFFECT_CHANNEL),
        ),
        evidence_kind=BinaryEvidenceKind.CONSTRAINT,
        expected=True,
    )
    observer = _JointPositionConstraintObserver(
        robot,
        control_parts=("left_eef", "right_eef"),
        object_ids=("cube",),
        open_qpos=(0.0, 0.0),
        minimum_displacement=0.004,
    )
    observation = observer(
        BinaryEffectEvidenceQuery(clause, expectation),
        EffectEvidenceCollectionContext(
            timestamp=0.1,
            observation_revision=1,
            env_ids=torch.tensor([2, 0, 3, 1], dtype=torch.long),
        ),
    )
    assert observation.values.tolist() == [True, False, True, False]
    assert observation.valid.tolist() == [True, True, True, True]


def test_marvin_pour_lowers_scene_and_targets_without_changing_program_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 750 mm table variant shifts assets and targets together by 125 mm."""
    monkeypatch.setattr(
        "embodichain_robots.tianji_marvin.get_data_path", lambda value: value
    )
    path, payload, _, deployment = _deployment("tableware/pour_water")
    reference = load_config(path.parent / "task.cobotmagic.yaml")
    scene = load_config(path.parent / payload["environment"]["component"])
    original_scene = load_config(path.parent / reference["environment"]["component"])
    assert scene.pop("environment_id") != original_scene.pop("environment_id")
    assert (
        scene["env"]["dataset"]["lerobot"]["params"]["robot_meta"].pop("robot_type")
        == "TianjiMarvin"
    )
    assert (
        original_scene["env"]["dataset"]["lerobot"]["params"]["robot_meta"].pop(
            "robot_type"
        )
        == "CobotMagic"
    )
    for category in ("background", "rigid_object"):
        for entity, original in zip(
            scene["simulation"][category],
            original_scene["simulation"][category],
            strict=True,
        ):
            position, original_position = entity.pop("init_pos"), original.pop(
                "init_pos"
            )
            assert position[:2] == original_position[:2]
            assert position[2] == pytest.approx(original_position[2] - 0.125)
    assert scene == original_scene
    program = load_config(path.parent / payload["task_program"]["program"])
    original_program = load_config(path.parent / reference["task_program"]["program"])
    targets, original_targets = program.pop("targets"), original_program.pop("targets")
    assert targets.keys() == original_targets.keys()
    for name, target in targets.items():
        position = target["values"][0].pop("position")
        original_position = original_targets[name]["values"][0].pop("position")
        assert position[:2] == original_position[:2]
        assert position[2] == pytest.approx(original_position[2] - 0.125)
        assert target == original_targets[name]
    assert program == original_program
    cfg = config_to_cfg(payload, source_path=path)
    assert cfg.target_control_frequency == 100
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    assert preset.motion_policy.sample_count == 160
    assert preset.skill_option_templates[
        "simulation.pour"
    ].rotate_angle == pytest.approx(-torch.pi / 3)
