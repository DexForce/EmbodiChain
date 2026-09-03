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

"""Tests for config-created Task Program environment integrations."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

from gymnasium.envs.registration import registry as gym_registry
import pytest
import torch

from embodichain.lab.task_program import load_task_program
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.task_program.integrations._configured_services import (
    _AxisAlignLowerer,
    _CoordinatedTransportLowerer,
    _CoordinatedTransportRoute,
    _MoveHeldObjectLowerer,
    _ParkLowerer,
    _PourLowerer,
    _RelativePlaceLowerer,
    _RelativePlaceRoute,
)
from embodichain.lab.task_program.integrations.configured import (
    _decode_action_options,
    _decode_configured_task_program_integration,
    _decode_grasp_generator,
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _load_configured_task_program_deployment,
    _resolve_task_program_components,
)
from embodichain.lab.gym.utils._component_composition import (
    _ResolvedGymComponents,
    _resolve_gym_components,
)
from embodichain.lab.gym.envs.task_program.registration import (
    _register_configured_task_program_integration,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    CoordinatedPickGoal,
    CoordinatedPickmentOptions,
    EntityState,
    HandOverOptions,
    HeldObjectState,
    HeldObjectPoseGoal,
    JointPositionGoal,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    PARK_COMMAND,
    PickUpOptions,
    PlanningContext,
    PlaceOptions,
    PlaceGoal,
    PourGoal,
    PourOptions,
    RobotObservation,
    SceneEntityPose,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.goals import (
    collect_scene_dependencies,
    resolve_pose_goal,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.task_program.semantics import (
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticEffectKind,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.utils.utility import load_config

_REPOSITORY_ROOT = Path(__file__).parents[4]
_CONFIG_DIRECTORY = _REPOSITORY_ROOT / "embodichain_tasks/configs/tasks/manipulation"
_TABLEWARE_CONFIG_DIRECTORY = _CONFIG_DIRECTORY / "tableware"
_TASKS = {
    "repeated_pick_place": (
        "task_program_repeated_pick_place",
        "ur5_dh_pgi_140_80",
        frozenset({"pick", "place", "hand_over"}),
    ),
    "open_drawer": (
        "task_program_open_drawer",
        "ur5_dh_pgi_140_80",
        frozenset(
            {
                "pick",
                "place",
                "hand_over",
                "simulation.articulation_link_slide",
            }
        ),
    ),
    "hand_over": (
        "dual_ur5_handover_v1",
        "dual_ur5_dh_pgi_140_80",
        frozenset({"pick", "place", "hand_over"}),
    ),
}
_TEST_ENV_ID = "ConfiguredTaskProgramIntegrationTest-v1"
_TABLEWARE_TASKS = {
    "pour_water": (
        "task_program_pour_water",
        "cobotmagic",
        frozenset(
            {
                "pick",
                "place",
                "hand_over",
                "simulation.move_held_object",
                "simulation.pour",
            }
        ),
    ),
}


def _config_path(task_name: str) -> Path:
    """Return one official task's Gym config path."""
    filename = (
        "task.dual_ur5_dh_pgi_140_80.yaml"
        if task_name == "hand_over"
        else "task.ur5.yaml"
    )
    return _CONFIG_DIRECTORY / task_name / filename


def _gym_config(task_name: str) -> dict[str, object]:
    """Return an independently owned production Gym configuration."""
    payload = load_config(_config_path(task_name))
    assert type(payload) is dict
    return payload


def _physical_components(
    path: Path,
    config: dict[str, object],
) -> _ResolvedGymComponents:
    """Resolve one deployment's reusable physical components."""
    return _resolve_gym_components(config, base_dir=path.parent)


def _integration_payload(task_name: str) -> dict[str, object]:
    """Compose one production integration into its strict decoder payload."""
    path = _config_path(task_name)
    config = _gym_config(task_name)
    physical = _physical_components(path, config)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    assert physical.embodiment_skill_profile is not None
    scene_binding = task["scene_binding"]
    assert type(scene_binding) is dict
    return _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=scene_binding,
    )


def _deployment_from_path(path: Path):
    """Return one fully composed production deployment."""
    config = load_config(path)
    physical = _physical_components(path, config)
    assert physical.embodiment_skill_profile is not None
    return _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )


def _tableware_config_path(task_name: str) -> Path:
    """Return one config-defined tableware task's Gym config path."""
    return _TABLEWARE_CONFIG_DIRECTORY / task_name / "task.cobotmagic.yaml"


def _tableware_gym_config(task_name: str) -> dict[str, object]:
    """Return an independently owned tableware Gym configuration."""
    payload = load_config(_tableware_config_path(task_name))
    assert type(payload) is dict
    return payload


def _tableware_integration_payload(task_name: str) -> dict[str, object]:
    """Return one tableware task's independently owned integration."""
    path = _tableware_config_path(task_name)
    config = _tableware_gym_config(task_name)
    physical = _physical_components(path, config)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    assert physical.embodiment_skill_profile is not None
    scene_binding = task["scene_binding"]
    assert type(scene_binding) is dict
    return _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=scene_binding,
    )


def _tableware_deployment(task_name: str):
    """Return one fully composed tableware deployment."""
    path = _tableware_config_path(task_name)
    config = _tableware_gym_config(task_name)
    physical = _physical_components(path, config)
    assert physical.embodiment_skill_profile is not None
    return _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )


@pytest.fixture
def registered_test_ids() -> Iterator[list[str]]:
    """Track and remove exact test-owned integration registrations."""
    identifiers: list[str] = []
    yield identifiers
    for env_id in identifiers:
        REGISTERED_ENVS.pop(env_id, None)
        gym_registry.pop(env_id, None)


@pytest.mark.parametrize(
    ("task_name", "expected_scene", "expected_profile", "expected_calls"),
    tuple(
        (task_name, scene_id, profile_id, calls)
        for task_name, (scene_id, profile_id, calls) in _TASKS.items()
    ),
)
def test_all_examples_decode_through_one_composable_integration_schema(
    task_name: str,
    expected_scene: str,
    expected_profile: str,
    expected_calls: frozenset[str],
) -> None:
    """Each official example uses the same scene/profile/services decoder."""
    integration = _decode_configured_task_program_integration(
        _integration_payload(task_name)
    )
    registration = integration.registration

    assert registration.scene_binding.registry_id == expected_scene
    assert registration.robot_profile_binding.profile_id == expected_profile
    assert frozenset(registration.call_catalog.descriptors) == expected_calls
    assert integration.adapter_factory.registration is registration


@pytest.mark.parametrize("task_name", ("repeated_pick_place", "open_drawer"))
def test_single_task_program_composes_with_ur5_and_franka(
    task_name: str,
) -> None:
    """Changing only the embodiment selects a second valid deployment."""
    task_dir = _CONFIG_DIRECTORY / task_name
    ur5_config = load_config(task_dir / "task.ur5.yaml")
    franka_config = load_config(task_dir / "task.franka.yaml")
    ur5 = _deployment_from_path(task_dir / "task.ur5.yaml")
    franka = _deployment_from_path(task_dir / "task.franka.yaml")

    ur5_program = load_task_program(
        ur5.program_path,
        integration=ur5.selection,
        validation_context=ur5.integration.registration.catalog,
    )
    franka_program = load_task_program(
        franka.program_path,
        integration=franka.selection,
        validation_context=franka.integration.registration.catalog,
    )

    assert (
        ur5_config["environment"]
        == franka_config["environment"]
        == {"component": "env.yaml"}
    )
    assert ur5_config["task_program"] == franka_config["task_program"]
    assert ur5.program_path == franka.program_path
    assert ur5.program_id == franka.program_id == ur5_program.program_id
    assert franka_program.program_id == ur5_program.program_id
    assert ur5.selection.scene_registry == franka.selection.scene_registry
    assert ur5.selection.robot_profile == "ur5_dh_pgi_140_80"
    assert franka.selection.robot_profile == "franka_panda"


def test_shared_embodiment_keeps_task_grasp_override_local() -> None:
    """Drawer-specific clearance does not mutate the shared embodiment."""
    repeated_config = _gym_config("repeated_pick_place")
    drawer_config = _gym_config("open_drawer")
    repeated = _integration_payload("repeated_pick_place")
    drawer = _integration_payload("open_drawer")

    repeated_generator = repeated["runtime_services"]["grasp_pose_generators"]["hand"]
    drawer_generator = drawer["runtime_services"]["grasp_pose_generators"]["hand"]

    repeated_embodiment = repeated_config["embodiment"]
    drawer_embodiment = drawer_config["embodiment"]
    assert type(repeated_embodiment) is dict
    assert type(drawer_embodiment) is dict
    assert repeated_embodiment["component"] == drawer_embodiment["component"]
    repeated_environment = load_config(
        _config_path("repeated_pick_place").parent / "env.yaml"
    )
    drawer_environment = load_config(_config_path("open_drawer").parent / "env.yaml")
    assert (
        repeated_environment["environment_id"] != drawer_environment["environment_id"]
    )
    assert "task_program" not in repeated_environment
    assert "task_program" not in drawer_environment
    assert repeated_generator["opening_margin"] == pytest.approx(0.002)
    assert drawer_generator["opening_margin"] == pytest.approx(0.03)


def test_task_rejects_embodiment_with_an_incompatible_contract() -> None:
    """Composition fails before program decoding when capabilities differ."""
    path = _config_path("repeated_pick_place")
    config = _gym_config("repeated_pick_place")
    physical = _physical_components(path, config)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    assert physical.embodiment_skill_profile is not None
    scene_binding = task["scene_binding"]
    assert type(scene_binding) is dict
    incompatible_skill_profile = dict(physical.embodiment_skill_profile)
    incompatible_skill_profile["contract_id"] = "dual_arm_only_v1"

    with pytest.raises(ValueError, match="Embodiment contract.*does not satisfy"):
        _compose_integration_payload(
            task=task,
            policy=policy,
            skill_profile=incompatible_skill_profile,
            scene=scene_binding,
        )


def test_embodiment_owns_the_deployed_sensor_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensors are selected with the physical embodiment, not the task scene."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    path = _tableware_config_path("pour_water")
    config = _tableware_gym_config("pour_water")
    physical = _physical_components(path, config)
    sensor_config = physical.config["sensor"]
    assert type(sensor_config) is list
    assert type(sensor_config[0]) is dict
    cfg = config_to_cfg(config, source_path=path)

    assert [sensor["sensor_type"] for sensor in sensor_config] == [
        "Camera",
        "Camera",
        "Camera",
    ]
    assert [sensor["uid"] for sensor in sensor_config] == [
        "cam_high",
        "cam_right_wrist",
        "cam_left_wrist",
    ]
    assert [sensor.uid for sensor in cfg.sensor] == [
        "cam_high",
        "cam_right_wrist",
        "cam_left_wrist",
    ]
    assert all(type(sensor) is CameraCfg for sensor in cfg.sensor)


@pytest.mark.parametrize(
    ("task_name", "expected_scene", "expected_profile", "expected_calls"),
    tuple(
        (task_name, scene_id, profile_id, calls)
        for task_name, (scene_id, profile_id, calls) in _TABLEWARE_TASKS.items()
    ),
)
def test_tableware_programs_decode_and_preflight_without_task_environment_code(
    task_name: str,
    expected_scene: str,
    expected_profile: str,
    expected_calls: frozenset[str],
) -> None:
    """Migrated tableware tasks use the common configured integration end to end."""
    deployment = _tableware_deployment(task_name)
    integration = deployment.integration
    registration = integration.registration
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=registration.catalog,
    )
    registration.catalog.preflight(program)

    assert registration.scene_binding.registry_id == expected_scene
    assert registration.robot_profile_binding.profile_id == expected_profile
    assert frozenset(registration.call_catalog.descriptors) == expected_calls


def test_pour_water_integration_declares_live_transport_and_axis_aware_pour() -> None:
    """Pouring policy owns its calibrated grasp, cup offset, axis, and tilt."""
    integration_payload = _tableware_integration_payload("pour_water")
    integration = _decode_configured_task_program_integration(integration_payload)
    registration = integration.registration
    grasp = registration.scene_binding.antipodal_grasps[0]
    preset = registration.robot_profile_binding.presets[0]
    options = preset.action_option_templates
    factories = registration.registered_semantic_lowerer_factories

    assert grasp.internal_axis == pytest.approx((1.0, 0.0, 0.0))
    assert type(options["pick"]) is PickUpOptions
    assert options["pick"].hand_interp_steps == 11
    assert options["pick"].grasp_settle_steps == 25
    assert options["pick"].rotate_upright is None
    assert options["pick"].fixed_object_to_eef is not None
    assert options["pick"].fixed_object_to_eef == pytest.approx(
        torch.tensor(
            (
                (-0.0530918874, 0.4963395894, 0.8665033579, 0.0358702540),
                (-0.0525476672, -0.8679134846, 0.4939277470, 0.0204655528),
                (0.9972059727, -0.0193091929, 0.0721606836, 0.0321167707),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    assert type(options["simulation.move_held_object"]) is MoveHeldObjectOptions
    assert type(options["simulation.pour"]) is PourOptions
    assert options["simulation.pour"].rotate_angle == pytest.approx(-torch.pi / 3)
    assert type(options["place"]) is PlaceOptions
    assert options["place"].hand_interp_steps == 11
    assert options["place"].release_settle_steps == 15
    assert options["place"].preserve_current_object_orientation is True
    assert {factory.call_id for factory in factories} == {
        "simulation.move_held_object",
        "simulation.pour",
    }
    assert "grasp_pose_generators" not in integration_payload["runtime_services"]


def test_pour_water_registered_lowerers_build_only_typed_goals() -> None:
    """Task arguments select configured values while presets own motion policy."""
    relative_pose = (
        1.0,
        0.0,
        0.0,
        0.05,
        0.0,
        1.0,
        0.0,
        -0.1,
        0.0,
        0.0,
        1.0,
        0.125,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    transport = _MoveHeldObjectLowerer(
        "cup_pour_pose",
        "cup",
        relative_pose,
    ).lower(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"target": "cup_pour_pose"},
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=MoveHeldObjectOptions(),
    )
    pour = _PourLowerer("bottle").lower(
        RegisteredSemanticCall(
            call_id="simulation.pour",
            arguments={"object": "bottle"},
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=PourOptions(),
    )

    assert type(transport.goal) is HeldObjectPoseGoal
    assert transport.goal.object_target_pose.entity_id == "cup"
    assert torch.equal(
        transport.goal.object_target_pose.relative_pose,
        torch.tensor(relative_pose, dtype=torch.float32).reshape(4, 4),
    )
    assert type(pour.goal) is PourGoal

    transport_targets = _MoveHeldObjectLowerer(
        "cup_pour_pose",
        "cup",
        relative_pose,
    ).pick_lookahead_targets(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"target": "cup_pour_pose"},
        ),
        picked_object=SceneObjectRef("bottle"),
        bound=None,  # type: ignore[arg-type]
        previous_target=None,
    )
    assert transport_targets is not None
    pour_targets = _PourLowerer("bottle").pick_lookahead_targets(
        RegisteredSemanticCall(
            call_id="simulation.pour",
            arguments={"object": "bottle"},
        ),
        picked_object=SceneObjectRef("bottle"),
        bound=SimpleNamespace(
            preset=SimpleNamespace(
                action_option_template=lambda semantic_id: PourOptions(
                    rotate_angle=-torch.pi / 3
                )
            )
        ),  # type: ignore[arg-type]
        previous_target=transport_targets[0],
    )

    assert len(transport_targets) == 1
    assert transport_targets[0].pose.entity_id == "cup"
    assert pour_targets is not None
    assert len(pour_targets) == 2
    tilted = pour_targets[0].pose.relative_pose
    assert tilted is not None
    assert tilted[:3, :3] == pytest.approx(
        torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.5, 0.8660254),
                (0.0, -0.8660254, 0.5),
            )
        )
    )


def test_move_held_object_lowerer_rejects_non_se3_relative_pose() -> None:
    """Configured live-relative transport fails closed on malformed transforms."""
    payload = deepcopy(_tableware_integration_payload("pour_water"))
    services = payload["runtime_services"]
    lowerer = services["registered_semantic_lowerers"][0]
    lowerer["relative_pose"][-1] = 2.0

    with pytest.raises(ValueError, match="bottom row"):
        _decode_configured_task_program_integration(payload)


def test_coordinated_transport_lowerer_builds_one_releasing_atomic_goal() -> None:
    """A registered transport owns no arm names, goals, or trajectory data."""
    relative_pose = (
        1.0,
        0.0,
        0.0,
        0.2,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="tray",
        entity_id="tray",
    )
    lowerer = _CoordinatedTransportLowerer(
        (("tray", "tray_forward", "tray", relative_pose),),
        (semantics,),
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.coordinated_transport",
            arguments={"object": "tray", "target": "tray_forward"},
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=CoordinatedPickmentOptions(release=True),
    )

    assert type(lowering.goal) is CoordinatedPickGoal
    assert lowering.goal.object_initial_pose is None
    assert lowering.goal.object_target_pose.entity_id == "tray"
    assert lowering.goal.object_target_pose.relative_pose == pytest.approx(
        torch.tensor(relative_pose, dtype=torch.float32).reshape(4, 4)
    )
    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.RELEASE
    assert tuple(
        (effect.slot_id, effect.relation, effect.object_id)
        for effect in lowering.registered_effect.held_objects
    ) == (
        ("left", HeldObjectRelation.DETACHED, "tray"),
        ("right", HeldObjectRelation.DETACHED, "tray"),
    )
    with pytest.raises(ValueError, match="enable coordinated release"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id="simulation.coordinated_transport",
                arguments={"object": "tray", "target": "tray_forward"},
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=CoordinatedPickmentOptions(),
        )


def test_axis_align_registered_call_builds_typed_goal_and_attach_effect() -> None:
    """The semantic call exposes AxisAlign without duplicating its planner."""
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([0.0, 0.0, 1.0])),
        geometry={},
        label="can",
        entity_id="can",
    )
    lowerer = _AxisAlignLowerer((semantics,))

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.axis_align",
            arguments={"object": "can"},
            resources={"primary": "right"},
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=AxisAlignOptions(),
    )

    assert type(lowering.goal) is AxisAlignGoal
    assert lowering.goal.semantics is semantics
    assert lowering.skill_options is None
    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert tuple(
        (effect.slot_id, effect.relation, effect.object_id)
        for effect in lowering.registered_effect.held_objects
    ) == (
        ("primary", HeldObjectRelation.ATTACHED, "can"),
    )
    with pytest.raises(ValueError, match="contain only 'object'"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id="simulation.axis_align",
                arguments={"object": "can", "target_axis": [0.0, 0.0, 1.0]},
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=AxisAlignOptions(),
        )


def test_relative_place_uses_fresh_reference_pose_and_verified_grasp() -> None:
    """Relative placement is grounded from the latest scene and TaskState."""
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="can",
        entity_id="can",
    )
    object_to_eef = torch.eye(4).unsqueeze(0)
    object_to_eef[:, 2, 3] = 0.2
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=object_to_eef,
        grasp_xpos=torch.eye(4).unsqueeze(0),
    )
    object_pose = torch.eye(4).unsqueeze(0)
    object_pose[:, 1, 3] = -0.3
    reference_pose = torch.eye(4).unsqueeze(0)
    reference_pose[:, :3, 3] = torch.tensor(((0.4, 0.1, 1.05),))
    qpos = torch.zeros((1, 1))
    context = PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState(
            batch_size=1,
            device="cpu",
            held_objects={"right_arm": held},
        ),
        scene=SceneSnapshot(
            timestamp=1.0,
            version=1,
            entities={
                "can": EntityState(object_pose),
                "notebook": EntityState(reference_pose),
            },
        ),
        env_ids=torch.tensor((0,), dtype=torch.long),
    )
    options = PlaceOptions(preserve_current_object_orientation=True)
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="right_arm")}
                )
            }
        ),
        preset=SimpleNamespace(action_option_template=lambda _semantic_id: options),
    )
    lowerer = _RelativePlaceLowerer(
        (
            _RelativePlaceRoute(
                object_id="can",
                reference_entity_id="notebook",
                relation="behind",
                world_displacement=(0.18, 0.0, 0.02),
            ),
        )
    )

    call = RegisteredSemanticCall(
        call_id="simulation.place_relative",
        arguments={
            "object": "can",
            "reference": "notebook",
            "relation": "behind",
        },
        resources={"primary": "right"},
    )
    lowering = lowerer.lower(
        call,
        context=context,
        bound=bound,  # type: ignore[arg-type]
        option_template=options,
    )

    assert type(lowering.goal) is PlaceGoal
    assert type(lowering.goal.xpos) is SceneEntityPose
    assert lowering.goal.xpos.entity_id == "notebook"
    assert collect_scene_dependencies(lowering.goal) == ("notebook",)
    resolved = resolve_pose_goal(lowering.goal.xpos, context, name="xpos")
    torch.testing.assert_close(
        resolved[:, :3, 3],
        torch.tensor(((0.58, 0.1, 1.27),)),
    )
    moved_reference_pose = reference_pose.clone()
    moved_reference_pose[:, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved_reference_pose[:, 1, 3] = 0.4
    moved_context = PlanningContext(
        robot=context.robot,
        task=context.task,
        scene=SceneSnapshot(
            timestamp=2.0,
            version=2,
            entities={
                "can": EntityState(object_pose),
                "notebook": EntityState(moved_reference_pose),
            },
        ),
        env_ids=context.env_ids,
    )
    moved_resolved = resolve_pose_goal(
        lowering.goal.xpos,
        moved_context,
        name="xpos",
    )
    torch.testing.assert_close(
        moved_resolved[:, :3, 3],
        torch.tensor(((0.58, 0.4, 1.27),)),
    )
    torch.testing.assert_close(moved_resolved[:, :3, :3], torch.eye(3).unsqueeze(0))
    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.RELEASE
    assert lowering.registered_effect.held_objects[0].relation is (
        HeldObjectRelation.DETACHED
    )
    lookahead = lowerer.pick_lookahead_targets(
        call,
        picked_object=SceneObjectRef("can"),
        bound=bound,  # type: ignore[arg-type]
        previous_target=None,
    )
    assert lookahead is not None and len(lookahead) == 1
    assert type(lookahead[0].pose) is SceneEntityPose
    assert lookahead[0].pose.entity_id == "notebook"
    torch.testing.assert_close(
        lookahead[0].pose.world_displacement,
        torch.tensor((0.18, 0.0, 0.02)),
    )
    assert lookahead[0].preserve_current_object_orientation


def test_axis_align_and_relative_place_configs_use_closed_decoders() -> None:
    """Both generated extensions contribute typed options and descriptors."""
    axis_factory = _decode_registered_lowerer(
        {"kind": "axis_align", "object_ids": ["can"]},
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    place_factory = _decode_registered_lowerer(
        {
            "kind": "place_relative",
            "routes": [
                {
                    "object_id": "can",
                    "reference_entity_id": "notebook",
                    "relation": "on",
                    "world_displacement": [0.0, 0.0, 0.04],
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[1]",
    )
    axis_options = _decode_action_options(
        {"kind": "axis_align", "target_axis": [0.0, 0.0, 1.0]},
        path="policy.action_options.simulation.axis_align",
    )

    assert axis_factory.call_id == "simulation.axis_align"
    assert axis_factory.object_ids == ("can",)
    assert place_factory.call_id == "simulation.place_relative"
    assert place_factory.routes[0].relation == "on"
    assert type(axis_options) is AxisAlignOptions
    torch.testing.assert_close(axis_options.target_axis, torch.tensor((0.0, 0.0, 1.0)))


def test_coordinated_transport_config_decodes_closed_routes_and_options() -> None:
    """Configured transport data round-trips through the strict allowlist."""
    identity = [
        1.0,
        0.0,
        0.0,
        0.2,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    factory = _decode_registered_lowerer(
        {
            "kind": "coordinated_transport",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "tray_forward",
                    "reference_entity_id": "tray",
                    "relative_pose": identity,
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    options = _decode_action_options(
        {
            "kind": "coordinated_pickment",
            "release": True,
            "release_steps": 6,
            "retreat_steps": 8,
            "grasp_seed": 17393,
        },
        path="policy.action_options.simulation.coordinated_transport",
    )

    assert factory.call_id == "simulation.coordinated_transport"
    assert type(options) is CoordinatedPickmentOptions
    assert options.release is True
    assert options.release_steps == 6
    assert options.retreat_steps == 8
    assert options.grasp_seed == 17393


def test_park_lowerer_and_config_keep_joint_values_in_the_profile() -> None:
    """Semantic Park reuses MoveJoints while profile data continues to own qpos."""
    lowerer = _ParkLowerer()
    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.park",
            arguments={},
            resources={"primary": "left"},
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=MoveJointsOptions(),
    )
    factory = _decode_registered_lowerer(
        {"kind": "park"},
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    options = _decode_action_options(
        {"kind": "move_joints"},
        path="policy.action_options.simulation.park",
    )

    assert type(lowering.goal) is JointPositionGoal
    assert lowering.goal.target == PARK_COMMAND
    assert factory.call_id == "simulation.park"
    assert factory.target_descriptor == MoveJoints.descriptor()
    assert type(options) is MoveJointsOptions
    with pytest.raises(ValueError, match="arguments must be empty"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id="simulation.park",
                arguments={"qpos": [0.0]},
                resources={"primary": "left"},
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=MoveJointsOptions(),
        )


def test_configured_handover_decodes_source_retreat_clearance() -> None:
    """The semantic profile may tune physical source-hand clearance."""
    options = _decode_action_options(
        {
            "kind": "hand_over",
            "retreat_distance": 0.12,
            "retreat_steps": 28,
        },
        path="policy.action_options.hand_over",
    )

    assert type(options) is HandOverOptions
    assert options.retreat_distance == pytest.approx(0.12)
    assert options.retreat_steps == 28


def test_coordinated_transport_world_displacement_uses_fresh_object_pose() -> None:
    """A relative task motion keeps live orientation and robot-frame direction."""
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="tray",
        entity_id="tray",
    )
    lowerer = _CoordinatedTransportLowerer(
        (
            _CoordinatedTransportRoute(
                object_id="tray",
                target_id="tray_forward",
                world_displacement=(-0.16, 0.0, 0.0),
            ),
        ),
        (semantics,),
    )
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, :3] = torch.tensor(
        (((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)
    )
    pose[:, :3, 3] = torch.tensor(((0.03, -0.02, 0.68),))
    qpos = torch.zeros((1, 1))
    context = PlanningContext(
        robot=RobotObservation(timestamp=1.0, qpos=qpos, qvel=torch.zeros_like(qpos)),
        task=TaskState(batch_size=1, device="cpu"),
        scene=SceneSnapshot(
            timestamp=1.0,
            version=1,
            entities={"tray": EntityState(pose)},
        ),
        env_ids=torch.tensor((0,), dtype=torch.long),
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.coordinated_transport",
            arguments={"object": "tray", "target": "tray_forward"},
        ),
        context=context,
        bound=None,  # type: ignore[arg-type]
        option_template=CoordinatedPickmentOptions(release=True),
    )

    assert type(lowering.goal) is CoordinatedPickGoal
    assert isinstance(lowering.goal.object_target_pose, torch.Tensor)
    torch.testing.assert_close(
        lowering.goal.object_target_pose[:, :3, :3],
        pose[:, :3, :3],
    )
    torch.testing.assert_close(
        lowering.goal.object_target_pose[:, :3, 3],
        torch.tensor(((-0.13, -0.02, 0.68),)),
    )


def test_coordinated_transport_config_decodes_world_displacement() -> None:
    factory = _decode_registered_lowerer(
        {
            "kind": "coordinated_transport",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "tray_forward",
                    "world_displacement": [-0.16, 0.0, 0.0],
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )

    assert factory.routes[0].world_displacement == (-0.16, 0.0, 0.0)


def test_pick_option_rejects_malformed_fixed_object_to_eef() -> None:
    """Configured fixed grasps must contain exactly one SE(3) transform."""
    payload = deepcopy(_tableware_integration_payload("pour_water"))
    pick_options = payload["robot_profile"]["presets"][0]["action_options"]["pick"]
    pick_options["fixed_object_to_eef"] = [1.0] * 15

    with pytest.raises(ValueError, match="exactly 16 values"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize("task_name", tuple(_TASKS))
def test_all_examples_register_plain_embodied_env_under_config_selected_ids(
    task_name: str,
    registered_test_ids: list[str],
) -> None:
    """Config loading creates a runnable ID without a task environment class."""
    env_id = f"Configured-{task_name.replace('_', '-')}-v1"
    registered_test_ids.append(env_id)
    config = _gym_config(task_name)
    config["id"] = env_id

    cfg = config_to_cfg(config, source_path=_config_path(task_name))
    spec = REGISTERED_ENVS[env_id]

    assert spec.cls is EmbodiedEnv
    assert spec.max_episode_steps == cfg.max_episode_steps
    assert spec.task_program_adapter_factory is not None
    assert spec.task_program_registration is not None
    assert spec.task_program_adapter_factory.registration is (
        spec.task_program_registration
    )
    assert cfg.task_program is not None
    assert cfg.task_program.integration.scene_registry == (
        spec.task_program_registration.scene_binding.registry_id
    )
    assert cfg.task_program.integration.robot_profile == (
        spec.task_program_registration.robot_profile_binding.profile_id
    )
    assert env_id in gym_registry


def test_trajectory_examples_disable_validation_and_recovery_layers() -> None:
    """The two showcase profiles contain only open-loop trajectory execution."""
    for task_name in ("repeated_pick_place", "open_drawer"):
        integration = _decode_configured_task_program_integration(
            _integration_payload(task_name)
        )
        preset = integration.registration.robot_profile_binding.presets[0]

        assert preset.motion_policy.sample_count == 40
        assert dict(preset.effect_monitors) == {}
        assert preset.recovery_policy.max_replans == 0
        assert preset.recovery_policy.max_action_retries == 0
        assert preset.workflow_recovery_policy.max_recovery_attempts == 0
        assert preset.runner_cfg.minimum_cycle_time == 0.0


def test_grasp_generator_resolves_named_model_and_library_defaults() -> None:
    """Serialized services name reusable geometry and omit policy defaults."""
    factory = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "dh_pgi_140_80",
        },
        path="generator",
    )

    assert factory.model_id == "dh_pgi_140_80"
    assert factory.min_opening_width == pytest.approx(0.005)
    assert factory.palm_depth == pytest.approx(0.096)
    assert factory.sample_count is None
    assert factory.approach_deviation_angle is None
    assert factory.approach_direction_samples is None
    assert factory.max_candidates is None
    assert factory.opening_margin is None
    assert factory.point_sample_density is None
    assert factory.filter_ground_collision is None
    assert factory.force_refresh is None
    assert not hasattr(factory, "viser_port")


def test_grasp_generator_factory_defers_to_toolkit_policy_defaults() -> None:
    """Omitted integration policy fields remain owned by the grasp toolkit."""
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "dh_pgi_140_80",
        },
        path="generator",
    )()

    assert generator.algorithm_cfg.sample_count == 20_000
    assert generator.algorithm_cfg.approach_deviation_angle == pytest.approx(
        math.pi / 6
    )
    assert generator.algorithm_cfg.approach_direction_samples == 4
    assert generator.algorithm_cfg.max_candidates == 50
    assert generator.collision_cfg.opening_margin == pytest.approx(0.01)
    assert generator.collision_cfg.point_sample_density == pytest.approx(0.01)
    assert generator.collision_cfg.filter_ground_collision is True
    assert generator.annotation_cfg.selection_mode == "whole_mesh"
    assert generator.annotation_cfg.force_refresh is False


def test_grasp_generator_decodes_candidate_search_policy() -> None:
    """Robot profiles may widen and constrain the canonical grasp search."""
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "dh_pgi_140_80",
            "approach_deviation_angle": math.pi / 9,
            "max_candidates": 500,
        },
        path="generator",
    )()

    assert generator.algorithm_cfg.approach_deviation_angle == pytest.approx(
        math.pi / 9
    )
    assert generator.algorithm_cfg.max_candidates == 500


def test_grasp_generator_inline_model_uses_geometry_defaults() -> None:
    """Custom inline models only override geometry that differs from defaults."""
    factory = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": {
                "model_id": "custom_parallel_jaw",
                "palm_depth": 0.09,
            },
        },
        path="generator",
    )

    assert factory.model_id == "custom_parallel_jaw"
    assert factory.min_opening_width == pytest.approx(0.001)
    assert factory.max_opening_width == pytest.approx(0.1)
    assert factory.finger_length == pytest.approx(0.08)
    assert factory.finger_width == pytest.approx(0.03)
    assert factory.finger_thickness == pytest.approx(0.01)
    assert factory.palm_depth == pytest.approx(0.09)


def test_grasp_generator_rejects_unknown_models_and_removed_viser_port() -> None:
    """The integration keeps a closed model catalog and no interactive port."""
    with pytest.raises(ValueError, match="available.*dh_pgi_140_80"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "model": "missing_gripper",
            },
            path="generator",
        )

    with pytest.raises(ValueError, match="unsupported fields.*viser_port"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "model": "dh_pgi_140_80",
                "viser_port": 11_801,
            },
            path="generator",
        )


def test_scene_shorthand_derives_native_ids_and_affordance_metadata() -> None:
    """Common scene identities need not be repeated in serialized bindings."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    grasp = rigid_object["affordances"][0]
    for field in ("simulation_uid", "default_grasp_affordance"):
        rigid_object.pop(field, None)
    for field in ("native_name", "revision"):
        grasp.pop(field, None)

    binding = _decode_configured_task_program_integration(
        payload
    ).registration.scene_binding

    assert binding.rigid_objects[0].simulation_uid == "cube"
    assert binding.rigid_objects[0].default_grasp_affordance is None
    assert binding.antipodal_grasps[0].native_name == "cube_grasp"
    assert binding.antipodal_grasps[0].revision == "1"
    assert binding.antipodal_grasps[0].object_id == "cube"


def test_scene_entity_nesting_derives_all_affordance_parents() -> None:
    """Affordance ownership comes only from the containing scene entity."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    affordances = rigid_object["affordances"]
    affordances.extend(
        (
            {
                "entity_id": "cube_top",
                "kind": "support_surface",
                "native_name": "top",
            },
            {
                "entity_id": "cube_inside",
                "kind": "container",
                "native_name": "inside",
                "release_clearance": 0.12,
            },
        )
    )

    binding = _decode_configured_task_program_integration(
        payload
    ).registration.scene_binding

    assert binding.antipodal_grasps[0].object_id == "cube"
    assert binding.support_surfaces[0].parent_id == "cube"
    assert binding.containers[0].parent_id == "cube"
    assert binding.containers[0].release_clearance == pytest.approx(0.12)


def test_placement_affordances_can_belong_to_articulations_and_links() -> None:
    """Placement ownership supports every scene parent accepted by the registry."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    scene["articulations"] = [
        {
            "entity_id": "fixture",
            "affordances": [
                {
                    "entity_id": "fixture_container",
                    "kind": "container",
                    "native_name": "interior",
                }
            ],
        }
    ]
    scene["links"] = [
        {
            "entity_id": "fixture_shelf",
            "articulation_id": "fixture",
            "native_link_name": "shelf",
            "affordances": [
                {
                    "entity_id": "shelf_surface",
                    "kind": "support_surface",
                    "native_name": "top",
                }
            ],
        }
    ]

    binding = _decode_configured_task_program_integration(
        payload
    ).registration.scene_binding

    assert binding.containers[0].parent_id == "fixture"
    assert binding.support_surfaces[0].parent_id == "fixture_shelf"


@pytest.mark.parametrize(
    "field_name",
    ("antipodal_grasps", "support_surfaces", "containers"),
)
def test_scene_rejects_flat_affordance_collections(field_name: str) -> None:
    """Configured scenes expose affordances only beneath their owner entity."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    scene[field_name] = []

    with pytest.raises(ValueError, match=f"unsupported fields.*{field_name}"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize("field_name", ("object_id", "parent_id"))
def test_nested_affordance_rejects_repeated_parent_fields(field_name: str) -> None:
    """A nested affordance cannot restate or redirect its structural parent."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    grasp = rigid_object["affordances"][0]
    grasp[field_name] = "cube"

    with pytest.raises(ValueError, match=f"unsupported fields.*{field_name}"):
        _decode_configured_task_program_integration(payload)


def test_nested_affordance_rejects_unknown_kind() -> None:
    """The configured affordance discriminator is a closed allowlist."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    rigid_object["affordances"][0]["kind"] = "suction_grasp"

    with pytest.raises(ValueError, match="kind must be one of"):
        _decode_configured_task_program_integration(payload)


def test_antipodal_grasp_rejects_non_object_owner() -> None:
    """Mesh-backed antipodal grasps belong only to rigid objects."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    rigid_object = scene["rigid_objects"][0]
    affordances = rigid_object.pop("affordances")
    scene["articulations"] = [
        {
            "entity_id": "drawer",
            "affordances": affordances,
        }
    ]

    with pytest.raises(ValueError, match="only under a rigid object"):
        _decode_configured_task_program_integration(payload)


def test_official_generators_use_named_models_without_default_fields() -> None:
    """Reference configs expose only task-specific grasp-generator tuning."""
    forbidden = {"force_refresh", "viser_port"}
    for task_name in _TASKS:
        services = _integration_payload(task_name)["runtime_services"]
        generators = services["grasp_pose_generators"]
        for generator in generators.values():
            assert generator["model"] == "dh_pgi_140_80"
            assert forbidden.isdisjoint(generator)
            if task_name == "hand_over":
                assert generator["approach_direction_samples"] == 1
            else:
                assert "approach_direction_samples" not in generator


def test_open_drawer_trajectory_integration_selects_a_snapshot_target() -> None:
    """The showcase does not install dynamic-target recovery monitoring."""
    integration = _decode_configured_task_program_integration(
        _integration_payload("open_drawer")
    )
    factories = integration.registration.registered_semantic_lowerer_factories

    assert len(factories) == 1
    assert factories[0].target_pose_mode == "snapshot"


def test_integration_config_rejects_unknown_slide_target_pose_mode() -> None:
    """Configured lowerers accept only explicit live or snapshot targets."""
    payload = deepcopy(_integration_payload("open_drawer"))
    services = payload["runtime_services"]
    assert type(services) is dict
    lowerers = services["registered_semantic_lowerers"]
    assert type(lowerers) is list
    lowerer = lowerers[0]
    assert type(lowerer) is dict
    lowerer["target_pose_mode"] = "moving"

    with pytest.raises(ValueError, match="target_pose_mode"):
        _decode_configured_task_program_integration(payload)


def test_integration_config_rejects_unknown_fields_before_live_construction() -> None:
    """Misspelled or unsupported integration fields fail closed."""
    payload = _integration_payload("repeated_pick_place")
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields.*unexpected"):
        _decode_configured_task_program_integration(payload)


def test_integration_config_rejects_removed_schema_version() -> None:
    """The integration declaration has no development-history version field."""
    payload = _integration_payload("repeated_pick_place")
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="unsupported fields.*schema_version"):
        _decode_configured_task_program_integration(payload)


def test_integration_registration_is_idempotent_for_the_same_config(
    registered_test_ids: list[str],
) -> None:
    """Loading the same config twice preserves its exact integration registration."""
    registered_test_ids.append(_TEST_ENV_ID)
    config = _gym_config("repeated_pick_place")
    config["id"] = _TEST_ENV_ID

    first_cfg = config_to_cfg(config, source_path=_config_path("repeated_pick_place"))
    first = REGISTERED_ENVS[_TEST_ENV_ID]
    second_cfg = config_to_cfg(
        deepcopy(config),
        source_path=_config_path("repeated_pick_place"),
    )

    assert REGISTERED_ENVS[_TEST_ENV_ID] is first
    assert first_cfg.task_program is not None
    assert second_cfg.task_program is not None


def test_invalid_config_does_not_leave_an_integration_registration(
    registered_test_ids: list[str],
) -> None:
    """Registration occurs only after program and environment parsing succeeds."""
    registered_test_ids.append(_TEST_ENV_ID)
    config = _gym_config("repeated_pick_place")
    config["id"] = _TEST_ENV_ID
    task_program = config["task_program"]
    assert type(task_program) is dict
    task_program["execution_policy"] = "missing-execution-policy.yaml"

    with pytest.raises(FileNotFoundError):
        config_to_cfg(config, source_path=_config_path("repeated_pick_place"))

    assert _TEST_ENV_ID not in REGISTERED_ENVS
    assert _TEST_ENV_ID not in gym_registry


def test_integration_registration_rejects_reusing_an_id_for_changed_config(
    registered_test_ids: list[str],
) -> None:
    """Changing integration data requires a distinct ID instead of silent override."""
    registered_test_ids.append(_TEST_ENV_ID)
    integration = _decode_configured_task_program_integration(
        _integration_payload("repeated_pick_place")
    )
    _register_configured_task_program_integration(
        _TEST_ENV_ID,
        integration,
        max_episode_steps=321,
    )
    changed_payload = deepcopy(_integration_payload("repeated_pick_place"))
    profile = changed_payload["robot_profile"]
    assert type(profile) is dict
    presets = profile["presets"]
    assert type(presets) is list
    preset = presets[0]
    assert type(preset) is dict
    motion = preset["motion"]
    assert type(motion) is dict
    motion["sample_count"] = 41
    changed_integration = _decode_configured_task_program_integration(changed_payload)

    with pytest.raises(ValueError, match="different environment or integration"):
        _register_configured_task_program_integration(
            _TEST_ENV_ID,
            changed_integration,
            max_episode_steps=321,
        )


def test_examples_have_no_importable_task_environment_modules() -> None:
    """All three environment implementations are now serialized configuration."""
    for task_name in _TASKS:
        module_name = f"embodichain_tasks.manipulation.{task_name}"
        assert importlib.util.find_spec(module_name) is None


__all__: list[str] = []
