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

from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
    build_gripper_collision_geometry,
)

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
from embodichain.lab.gym.envs.task_program.bridge import RuntimeCommandFrameEncoder
from embodichain.lab.task_program.integrations import (
    ConfiguredTaskProgramDeployment,
    ConfiguredTaskProgramIntegration,
    TaskProgramAdapterFactory,
    load_configured_task_program_deployment,
)
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
    _decode_motion_policy,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _decode_planner_config,
    _load_configured_task_program_deployment,
    _resolve_task_program_components,
)
from embodichain.lab.gym.utils.composition import (
    ResolvedGymComponents,
    resolve_gym_components,
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
    MoveEndEffectorOptions,
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
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
)
from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
    TrapezoidalPlanOptions,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain_robots import TianjiMarvinCfg
from embodichain.lab.task_program.semantics import (
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticEffectKind,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.utils.utility import load_config
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGeneratorCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]
_CONFIG_DIRECTORY = (
    _REPOSITORY_ROOT / "extensions/embodichain_tasks/configs/tasks/manipulation"
)
_TABLEWARE_CONFIG_DIRECTORY = _CONFIG_DIRECTORY / "tableware"
_TASKS = {
    "rubiks_cube_pick_place": (
        "task_program_rubiks_cube_pick_place",
        "ur5_dh_pgi_140_80",
        frozenset({"pick", "place", "hand_over"}),
    ),
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
    "stack_blocks_two": (
        "task_program_stack_blocks_two",
        "cobotmagic",
        frozenset({"pick", "place", "hand_over"}),
    ),
    "blocks_ranking_rgb": (
        "task_program_blocks_ranking_rgb",
        "cobotmagic",
        frozenset({"pick", "place", "hand_over"}),
    ),
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
) -> ResolvedGymComponents:
    """Resolve one deployment's reusable physical components."""
    return resolve_gym_components(config, base_dir=path.parent)


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


def _deployment_from_path(path: Path) -> ConfiguredTaskProgramDeployment:
    """Return one fully composed production deployment."""
    config = load_config(path)
    physical = _physical_components(path, config)
    assert physical.embodiment_skill_profile is not None
    return load_configured_task_program_deployment(
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


def _tableware_deployment(task_name: str) -> ConfiguredTaskProgramDeployment:
    """Return one fully composed tableware deployment."""
    path = _tableware_config_path(task_name)
    config = _tableware_gym_config(task_name)
    physical = _physical_components(path, config)
    assert physical.embodiment_skill_profile is not None
    return load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )


def test_public_composition_owns_declarations_and_defers_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramAdapterFactory,
    )

    def unexpected_live_creation(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Composition must not create simulation or a live adapter."
        )

    monkeypatch.setattr(SimulationManager, "__init__", unexpected_live_creation)
    monkeypatch.setattr(
        SimulationTaskProgramAdapterFactory, "create_adapter", unexpected_live_creation
    )
    registered_before = dict(REGISTERED_ENVS)
    gym_before = dict(gym_registry)
    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    input_snapshot = deepcopy(config)
    physical = _physical_components(path, config)
    other = _physical_components(path, config)
    assert isinstance(physical, ResolvedGymComponents)
    assert physical.config == other.config
    other.config["robot"]["uid"] = "changed_physical_copy"
    assert physical.config["robot"]["uid"] != "changed_physical_copy"
    assert config == input_snapshot
    assert physical.embodiment_skill_profile is not None
    selectors = physical.config["task_program"]
    profile = physical.embodiment_skill_profile
    original = deepcopy((selectors, profile))
    first = load_configured_task_program_deployment(
        task_program=selectors, skill_profile=profile, base_dir=path.parent
    )
    second = load_configured_task_program_deployment(
        task_program=selectors, skill_profile=profile, base_dir=path.parent
    )
    assert isinstance(first, ConfiguredTaskProgramDeployment)
    assert isinstance(first.integration, ConfiguredTaskProgramIntegration)
    assert isinstance(first.integration.adapter_factory, TaskProgramAdapterFactory)
    assert first.program_path.is_file()
    assert (
        first.integration.integration_fingerprint
        == second.integration.integration_fingerprint
    )
    assert first.scene_binding == second.scene_binding
    assert (
        first.planner_config
        == second.planner_config
        == {"type": "toppra", "config": {}}
    )
    first.scene_binding["registry_id"] = "changed_by_caller"
    first.planner_config["config"]["grid_size"] = 127
    assert second.scene_binding["registry_id"] != "changed_by_caller"
    assert second.planner_config == {"type": "toppra", "config": {}}
    assert (selectors, profile) == original
    assert REGISTERED_ENVS == registered_before
    assert gym_registry == gym_before


@pytest.mark.parametrize("owner", ["task_program", "skill_profile"])
def test_public_composition_retains_closed_declaration_fields(owner: str) -> None:
    path = _config_path("hand_over")
    physical = _physical_components(path, _gym_config("hand_over"))
    assert physical.embodiment_skill_profile is not None
    values = {
        "task_program": deepcopy(physical.config["task_program"]),
        "skill_profile": deepcopy(physical.embodiment_skill_profile),
    }
    values[owner]["unexpected_override"] = True
    with pytest.raises(ValueError, match="unsupported fields.*unexpected_override"):
        load_configured_task_program_deployment(**values, base_dir=path.parent)


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


def test_execution_policy_planner_config_reaches_configured_adapter() -> None:
    """The policy-owned planner selection stays outside semantic payloads."""
    deployment = _deployment_from_path(
        _CONFIG_DIRECTORY / "hand_over" / "task.dual_ur5_dh_pgi_140_80.yaml"
    )

    assert deployment.planner_config == {"type": "toppra", "config": {}}
    delegate = deployment.integration.adapter_factory.delegate
    assert delegate._planner_config == deployment.planner_config
    assert (
        deployment.integration.registration.robot_profile_binding.presets[
            0
        ].required_planner
        == "toppra"
    )


def test_execution_policy_overrides_reach_runtime_without_changing_the_base(
    tmp_path: Path,
) -> None:
    """A relative selector patches tuning while retaining the policy contract."""
    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    physical = _physical_components(path, config)
    program, task, base_policy = _resolve_task_program_components(
        config["task_program"], base_dir=path.parent
    )
    from embodichain.utils.utility import save_config

    save_config(tmp_path / "policy.yaml", base_policy)
    save_config(tmp_path / "integration.yaml", task)
    original = (tmp_path / "policy.yaml").read_bytes()
    selection = {
        "program": str(program),
        "integration": "integration.yaml",
        "execution_policy": {
            "component": "policy.yaml",
            "overrides": {
                "motion": {
                    "sample_count": 160,
                    "velocity_limit": 0.35,
                    "acceleration_limit": 0.4,
                    "dynamic_collision_mode": "off",
                },
                "recovery": {
                    "action_timeout": 90.0,
                    "goal_rotation_threshold": 0.75,
                },
            },
        },
    }
    deployment = _load_configured_task_program_deployment(
        task_program=selection,
        skill_profile=physical.embodiment_skill_profile,
        base_dir=tmp_path,
    )
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    assert preset.motion_policy.sample_count == 160
    assert preset.motion_policy.velocity_limit == 0.35
    assert preset.motion_policy.acceleration_limit == 0.4
    assert preset.motion_policy.dynamic_collision_mode.value == "off"
    assert preset.recovery_policy.action_timeout == 90.0
    assert preset.recovery_policy.goal_rotation_threshold == 0.75
    assert preset.recovery_policy.max_replans == 3
    assert preset.effect_assurance.value == "verified"
    assert deployment.planner_config == {"type": "toppra", "config": {}}
    assert (tmp_path / "policy.yaml").read_bytes() == original
    _, _, unmodified = _resolve_task_program_components(
        config["task_program"], base_dir=path.parent
    )
    assert unmodified == base_policy


@pytest.mark.parametrize(
    "overrides,error,message",
    (
        (
            {"effect_assurance": "projected"},
            ValueError,
            "unsupported fields.*effect_assurance",
        ),
        (
            {"motion": {"strategy": "ik_interp"}},
            ValueError,
            "unsupported fields.*strategy",
        ),
        ({"motion": {"velocity_limit": float("nan")}}, ValueError, "velocity_limit"),
        ({"motion": {"sample_count": True}}, ValueError, "sample_count"),
        (
            {"motion": {"dynamic_collision_mode": "disabled"}},
            ValueError,
            "dynamic_collision_mode",
        ),
        ({"recovery": {"action_timeout": 0.0}}, ValueError, "action_timeout"),
        (None, TypeError, "overrides must be a mapping"),
    ),
)
def test_execution_policy_overrides_are_closed_and_use_existing_value_validation(
    overrides: object, error: type[Exception], message: str
) -> None:
    """Deployment tuning cannot replace semantics or bypass typed decoders."""
    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    config["task_program"]["execution_policy"] = {
        "component": "../../../components/execution_policies/motion_gen_verified.yaml",
        "overrides": overrides,
    }
    with pytest.raises(error, match=message):
        _resolve_task_program_components(config["task_program"], base_dir=path.parent)


def test_planner_policy_decoder_validates_backend_and_options() -> None:
    """Planner backends remain closed while typed options stay configurable."""
    assert _decode_planner_config(
        {"type": "toppra", "config": {"grid_size": 127}},
        path="execution policy.planner",
    ) == {"type": "toppra", "config": {"grid_size": 127}}

    with pytest.raises(ValueError, match="execution policy.planner.type"):
        _decode_planner_config(
            {"type": "custom_callable", "config": {}},
            path="execution policy.planner",
        )

    for field_name in ("world", "auto_gen"):
        with pytest.raises(TypeError, match="must be a mapping"):
            _decode_planner_config(
                {"type": "curobo", "config": {field_name: None}},
                path="execution policy.planner",
            )


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
        == {
            "default": "envs/default.yaml",
            "newton": "envs/newton.yaml",
        }
    )
    assert ur5_config["task_program"] == franka_config["task_program"]
    assert ur5.program_path == franka.program_path
    assert ur5.program_id == franka.program_id == ur5_program.program_id
    assert franka_program.program_id == ur5_program.program_id
    assert ur5.selection.scene_registry == franka.selection.scene_registry
    assert ur5.selection.robot_profile == "ur5_dh_pgi_140_80"
    assert franka.selection.robot_profile == "franka_panda"


def test_shared_embodiment_uses_actual_hand_geometry_without_box_overrides() -> None:
    """Tasks share actual hand geometry instead of obsolete scalar box margins."""
    repeated_config = _gym_config("repeated_pick_place")
    drawer_config = _gym_config("open_drawer")
    repeated = _integration_payload("repeated_pick_place")
    drawer = _integration_payload("open_drawer")

    repeated_generator = repeated["runtime_services"]["grasp_pose_generators"]["eef"]
    drawer_generator = drawer["runtime_services"]["grasp_pose_generators"]["eef"]

    repeated_embodiment = repeated_config["embodiment"]
    drawer_embodiment = drawer_config["embodiment"]
    assert type(repeated_embodiment) is dict
    assert type(drawer_embodiment) is dict
    assert repeated_embodiment["component"] == drawer_embodiment["component"]
    repeated_environment = load_config(
        _config_path("repeated_pick_place").parent / "envs/default.yaml"
    )
    drawer_environment = load_config(
        _config_path("open_drawer").parent / "envs/default.yaml"
    )
    assert (
        repeated_environment["environment_id"] != drawer_environment["environment_id"]
    )
    assert "task_program" not in repeated_environment
    assert "task_program" not in drawer_environment
    assert repeated_generator["point_spacing"] == pytest.approx(0.012)
    assert drawer_generator == repeated_generator
    assert "opening_margin" not in drawer_generator


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


def test_embodiment_and_robot_compose_the_deployed_sensor_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent and robot-mounted cameras retain the deployed sensor suite."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    path = _tableware_config_path("pour_water")
    config = _tableware_gym_config("pour_water")
    physical = _physical_components(path, config)
    sensor_config = physical.config["sensor"]
    assert type(sensor_config) is list
    assert type(sensor_config[0]) is dict
    cfg = config_to_cfg(config, source_path=path)

    assert [sensor["sensor_type"] for sensor in sensor_config] == ["Camera"]
    assert [sensor["uid"] for sensor in sensor_config] == ["cam_high"]
    assert cfg.robot.enable_sensor is True
    assert [sensor.uid for sensor in [*cfg.sensor, *cfg.robot.sensors]] == [
        "cam_high",
        "cam_right_wrist",
        "cam_left_wrist",
    ]
    assert all(
        type(sensor) is CameraCfg for sensor in [*cfg.sensor, *cfg.robot.sensors]
    )


def test_pour_water_commands_preserve_idle_left_arm_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh measured holds for full actions must not retarget the idle arm."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    path = _tableware_config_path("pour_water")
    cfg = config_to_cfg(_tableware_gym_config("pour_water"), source_path=path)
    parts = {
        "left_arm": [0, 2, 4, 6, 8, 10],
        "left_eef": [12],
        "right_arm": [1, 3, 5, 7, 9, 11],
        "right_eef": [13],
    }
    measured = torch.zeros(1, 16)
    targets = measured.clone()

    def set_qpos(*, qpos: torch.Tensor, joint_ids: list[int]) -> None:
        targets[:, joint_ids] = qpos

    env = object.__new__(EmbodiedEnv)
    env.cfg = cfg
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.robot = SimpleNamespace(
        control_parts=parts,
        get_joint_ids=lambda *, name, remove_mimic: parts[name],
        build_pk_serial_chain=lambda: None,
        body_data=SimpleNamespace(qpos_limits=torch.tensor([[[-3.0, 3.0]] * 16])),
        get_qpos=lambda: measured.clone(),
        set_qpos=set_qpos,
    )
    env._setup_robot()
    assert len(env.active_joint_ids) == 14
    encoder = RuntimeCommandFrameEncoder(
        SimpleNamespace(current_qpos=lambda env_ids: measured.clone())
    )

    for step in range(6):
        # Reproduce gravity deflection entering the bridge's observed-qpos base.
        if step:
            measured[:, parts["left_arm"]] -= 0.001
        frame = RuntimeCommandFrame(
            commands=(
                EndpointCommand(
                    target=JointPositionTarget(
                        control_part="right_arm", joint_ids=tuple(parts["right_arm"])
                    ),
                    payload=JointPositionPayload(
                        positions=torch.full((1, 6), step * 0.1)
                    ),
                ),
            ),
            active_mask=torch.tensor([True]),
            env_ids=torch.tensor([0]),
            hold_duration=torch.tensor([0.04]),
        )
        env._apply_controller_action(encoder.encode(frame))

    assert torch.equal(targets[:, parts["left_arm"]], torch.zeros(1, 6))
    assert torch.equal(targets[:, parts["left_eef"]], torch.zeros(1, 1))
    assert torch.equal(targets[:, parts["right_arm"]], torch.full((1, 6), 0.5))


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

    grasp_rotation = options["pick"].fixed_object_to_eef[:3, :3]
    wrist_roll_axis = grasp_rotation[:, 2] / torch.linalg.norm(grasp_rotation[:, 2])
    assert grasp.internal_axis == pytest.approx(wrist_roll_axis.tolist())
    assert type(options["pick"]) is PickUpOptions
    assert options["pick"].hand_interp_steps == 44
    assert options["pick"].grasp_settle_steps == 100
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
    assert options["place"].hand_interp_steps == 44
    assert options["place"].release_settle_steps == 60
    assert options["place"].preserve_current_object_orientation is True
    assert {factory.call_id for factory in factories} == {
        "simulation.move_held_object",
        "simulation.pour",
    }
    services = integration_payload["runtime_services"]["grasp_pose_generators"]
    assert set(services) == {"left_eef", "right_eef"}
    for generator in services.values():
        assert generator["kind"] == "antipodal_parallel_jaw"
        assert (
            generator["gripper"]["grasp_cfg"]["model_id"] == "cobotmagic_v100_gripper"
        )


def test_pour_water_mouth_alignment_and_measured_checkpoints() -> None:
    """The tilted mouth clears the cup and a missed grasp stops before pouring."""
    deployment = _tableware_deployment("pour_water")
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    picking, positioning, pouring, returning = program.program.items
    checkpoint = positioning.validators[0]
    assert checkpoint.object == "bottle"
    assert checkpoint.position_tolerance <= 0.01
    assert picking.steps.call.kind == "pick"
    assert positioning.steps.call.call_id == "simulation.move_held_object"
    assert pouring.steps.call.call_id == "simulation.pour"
    placement = returning.steps.call
    release_position = program.targets[placement.at.target].values[0].position
    rest_position = program.targets["bottle_return_pose"].values[0].position
    assert release_position[:2] == rest_position[:2]
    assert 0.006 <= release_position[2] - rest_position[2] <= 0.01
    cup_check = next(check for check in returning.validators if check.object == "cup")
    assert cup_check.position_tolerance <= 0.002

    integration = _tableware_integration_payload("pour_water")
    lowerer = integration["runtime_services"]["registered_semantic_lowerers"][0]
    relative = torch.tensor(lowerer["relative_pose"], dtype=torch.float64).reshape(4, 4)
    registration = deployment.integration.registration
    axis = torch.tensor(
        registration.scene_binding.antipodal_grasps[0].internal_axis,
        dtype=torch.float64,
    )
    angle = (
        registration.robot_profile_binding.presets[0]
        .action_option_templates["simulation.pour"]
        .rotate_angle
    )
    mouth = torch.tensor([0.0, 0.0, 0.069724626839], dtype=torch.float64)
    theta = mouth.new_tensor(angle)
    rotated = (
        mouth * torch.cos(theta)
        + torch.linalg.cross(axis, mouth) * torch.sin(theta)
        + axis * torch.dot(axis, mouth) * (1 - torch.cos(theta))
    )
    torch.testing.assert_close(
        (relative[:3, 3] + rotated)[:2],
        torch.zeros(2, dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    assert float(relative[2, 3] + rotated[2]) >= 0.21
    cup_position = torch.tensor(program.targets[cup_check.target].values[0].position)
    bottle_position = torch.tensor(
        program.targets[checkpoint.target].values[0].position
    )
    torch.testing.assert_close(bottle_position, cup_position + relative[:3, 3].float())


def test_pour_water_uses_shared_calibrated_embodiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pouring reuses the base robot with its qualified force and control cadence."""
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)
    path = _tableware_config_path("pour_water")
    config = _tableware_gym_config("pour_water")
    cfg = config_to_cfg(config, source_path=path)
    assert Path(config["embodiment"]["component"]).name == "cobotmagic.yaml"
    assert cfg.robot.joint_drive_props.max_effort["right_joint[7-8]"] == 40.0
    assert cfg.robot.joint_drive_props.max_effort["left_joint[7-8]"] == 40.0
    assert cfg.robot.joint_drive_props.max_velocity["right_joint[7-8]"] == 0.25
    assert cfg.robot.joint_drive_props.stiffness["right_joint6"] == 4000.0
    assert cfg.target_control_frequency == 100
    assert cfg.expert_trajectory.joint_command_mode == "position"
    preset = _tableware_deployment(
        "pour_water"
    ).integration.registration.robot_profile_binding.presets[0]
    assert preset.motion_policy.sample_count == 160
    assert preset.effect_assurance.value == "projected"


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

    assert (
        lowering.registered_effect.held_objects[0].release_before_segment == "retract"
    )
    assert lowering.registered_effect.held_objects[0].retain_during_segments == (
        "approach",
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


def test_relative_place_rejects_unmigrated_local_service_kind() -> None:
    """Do not retain a second factory contract for the same semantic call."""
    with pytest.raises(ValueError, match="Unsupported.*relative_place"):
        _decode_registered_lowerer({"kind": "relative_place"}, path="lowerer")


def test_relative_place_rejects_call_owned_world_offset() -> None:
    """Only the trusted integration may declare relative placement geometry."""
    lowerer = _RelativePlaceLowerer(
        (
            _RelativePlaceRoute(
                object_id="cube",
                reference_entity_id="table",
                relation="on",
                world_displacement=(0.0, 0.0, 0.1),
            ),
        )
    )
    with pytest.raises(ValueError, match="arguments must contain only"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id="simulation.place_relative",
                arguments={
                    "object": "cube",
                    "reference": "table",
                    "relation": "on",
                    "world_offset": [0.0, 0.0, 0.1],
                },
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=PlaceOptions(),
        )


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
        {
            "kind": "axis_align",
            "target_axis": [0.0, 0.0, 1.0],
            "pre_grasp_distance": 0.12,
        },
        path="policy.action_options.simulation.axis_align",
    )

    assert axis_factory.call_id == "simulation.axis_align"
    assert axis_factory.object_ids == ("can",)
    assert place_factory.call_id == "simulation.place_relative"
    assert place_factory.routes[0].relation == "on"
    assert type(axis_options) is AxisAlignOptions
    assert axis_options.pre_grasp_distance == pytest.approx(0.12)
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


def test_slide_clearance_options_decode() -> None:
    options = _decode_action_options(
        {
            "kind": "slide",
            "preshape_fraction": 0.85,
            "release_retreat_distance": 0.04,
            "approach_along_grasp_axis": True,
        },
        path="options",
    )
    assert options.preshape_fraction == 0.85
    assert options.release_retreat_distance == 0.04
    assert options.approach_along_grasp_axis is True
    default = _decode_action_options({"kind": "slide"}, path="options")
    assert default.preshape_fraction == 0.0
    assert default.release_retreat_distance == 0.0
    assert default.approach_along_grasp_axis is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("preshape_fraction", True),
        ("preshape_fraction", 1.1),
        ("release_retreat_distance", -0.1),
        ("approach_along_grasp_axis", "true"),
    ],
)
def test_slide_clearance_options_reject_invalid_values(
    field: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _decode_action_options({"kind": "slide", field: value}, path="options")


def test_move_end_effector_options_decode_without_task_specific_fields() -> None:
    options = _decode_action_options({"kind": "move_end_effector"}, path="options")
    assert type(options) is MoveEndEffectorOptions
    with pytest.raises(ValueError):
        _decode_action_options(
            {"kind": "move_end_effector", "withdraw_distance": 0.1}, path="options"
        )


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


@pytest.mark.parametrize(
    "kind,key",
    [
        ("hand_over", "source_hold_mode"),
        ("move_held_object", "hand_source"),
        ("place", "hand_source"),
        ("pour", "hand_source"),
    ],
)
def test_shared_action_decoders_reject_retired_hand_policy_fields(
    kind: str,
    key: str,
) -> None:
    with pytest.raises(ValueError, match=key):
        _decode_action_options(
            {"kind": kind, key: "grasp_command"},
            path="policy.action_options",
        )


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config loading creates a runnable ID without a task environment class."""
    env_id = f"Configured-{task_name.replace('_', '-')}-v1"
    registered_test_ids.append(env_id)
    config = _gym_config(task_name)
    config["id"] = env_id
    if task_name == "rubiks_cube_pick_place":
        monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)

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


def test_rubiks_cube_example_registers_rigidized_articulation_scene(
    registered_test_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _config_path("rubiks_cube_pick_place")
    config = _gym_config("rubiks_cube_pick_place")
    env_id = config["id"]
    assert type(env_id) is str
    registered_test_ids.append(env_id)
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)

    config_to_cfg(config, source_path=path)
    spec = REGISTERED_ENVS[env_id]
    registration = spec.task_program_registration

    assert gym_registry[env_id].id == "TaskProgramRubiksCubePickPlace-v1"
    assert registration is not None
    assert (
        registration.scene_binding.rigidized_articulations[0].simulation_uid
        == "rubiks_cube"
    )
    assert (
        registration.scene_binding.rigidized_articulation_grasps[0].grasp_link
        == "lower_two_layers"
    )


def test_rubiks_cube_physical_config_applies_declared_joint_lock(
    registered_test_ids: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime applies the coincident limits required by scene binding."""
    path = _config_path("rubiks_cube_pick_place")
    config = _gym_config("rubiks_cube_pick_place")
    env_id = config["id"]
    assert type(env_id) is str
    registered_test_ids.append(env_id)
    monkeypatch.setattr("embodichain.data.get_data_path", lambda value: value)

    cfg = config_to_cfg(config, source_path=path)
    cube_cfg = next(item for item in cfg.articulation if item.uid == "rubiks_cube")

    assert cube_cfg.asset_physics_mode == "overlay"
    assert cube_cfg.qpos_limits == {"top_turn": [0.0, 0.0]}


def test_grasp_generator_resolves_named_model_and_library_defaults() -> None:
    """Serialized services name reusable geometry and omit policy defaults."""
    factory = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "gripper": {"class_type": "DHPGI14080"},
        },
        path="generator",
    )

    assert factory.model_id == "dh_pgi_140_80"
    assert factory.min_opening_width == pytest.approx(0.005)
    assert factory.max_opening_width == pytest.approx(0.1)
    assert factory.sample_count is None
    assert factory.approach_direction_samples is None
    assert factory.point_spacing == pytest.approx(0.012)
    assert factory.root_from_tcp is None
    assert factory.filter_ground_collision is None
    assert factory.force_refresh is None
    assert not hasattr(factory, "viser_port")


def test_grasp_generator_factory_defers_to_toolkit_policy_defaults() -> None:
    """Omitted integration policy fields remain owned by the grasp toolkit."""
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "gripper": {"class_type": "DHPGI14080"},
        },
        path="generator",
    )()

    assert generator.algorithm_cfg.sample_count == 20_000
    assert generator.algorithm_cfg.approach_direction_samples == 4
    assert generator.collision_geometry.provenance["point_spacing_m"] == pytest.approx(
        0.012
    )
    assert generator.collision_geometry.provenance["model_id"] == "dh_pgi_140_80"
    assert generator.collision_cfg.filter_ground_collision is True
    assert generator.annotation_cfg.selection_mode == "whole_mesh"
    assert generator.annotation_cfg.force_refresh is False


def test_declared_grasp_search_options_reach_the_existing_generator(monkeypatch):
    from types import SimpleNamespace
    import embodichain.toolkits.graspkit.pg_grasp as grasp
    import embodichain.toolkits.graspkit.pg_grasp.model_geometry as geometry

    marker = object()
    geometry_values = {}
    contact = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0.21], [0, 0, 0, 1]]
    monkeypatch.setattr(
        geometry,
        "build_gripper_collision_geometry",
        lambda defaults, **kw: (geometry_values.update(defaults), marker)[1],
    )
    monkeypatch.setattr(
        grasp,
        "AntipodalGraspPoseGenerator",
        lambda model, **kw: SimpleNamespace(model=model, **kw),
    )
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "gripper": {"class_type": "RobotiqArg2F140"},
            "approach_deviation_angle": 0.3490658503988659,
            "max_candidates": 500,
            "center_mode": "centroid",
            "root_from_contact_frame": contact,
        },
        path="generator",
    )()
    assert generator.collision_geometry is marker
    assert generator.algorithm_cfg.approach_deviation_angle == 0.3490658503988659
    assert generator.algorithm_cfg.max_candidates == 500
    assert generator.algorithm_cfg.center_mode == "centroid"

    assert geometry_values["grasp_cfg"]["root_from_contact_frame"] == contact


@pytest.mark.parametrize(
    "value",
    [
        {"approach_deviation_angle": -0.1},
        {"approach_deviation_angle": float("nan")},
        {"max_candidates": 0},
        {"max_candidates": True},
        {"center_mode": "guessed"},
        {"root_from_contact_frame": [[0.0] * 4 for _ in range(4)]},
    ],
)
def test_grasp_search_declarations_reject_invalid_values_before_construction(value):
    with pytest.raises((TypeError, ValueError)):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "gripper": {"class_type": "RobotiqArg2F140"},
                **value,
            },
            path="generator",
        )


@pytest.mark.parametrize(
    "profile,component",
    [
        ("dual_franka", "dual_franka_robotiq_arg2f_140"),
        ("dual_ur10", "dual_ur10_robotiq_arg2f_140"),
    ],
)
def test_migrated_robotiq_embodiments_preserve_frozen_physical_declarations(
    profile, component
):
    from copy import deepcopy
    import json

    frozen = json.loads(
        (Path(__file__).parent / "data/robotiq_embodiments_b0.json").read_text()
    )[profile]["embodiment"]
    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    config["embodiment"] = {
        "component": f"../../../components/embodiments/{component}.yaml"
    }
    physical = _physical_components(path, config)
    expected = deepcopy(frozen["simulation"])
    actual = deepcopy(physical.config["robot"])
    # Scene placement is an explicit deployment input, separate from the component's reference pose.
    actual["init_pos"] = expected["init_pos"]
    bindings = actual.pop("gripper_bindings")
    for side in ("left", "right"):
        from embodichain.lab.sim.cfg import RobotCfg
        from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers

        runtime = RobotCfg.from_dict(deepcopy(physical.config["robot"]))
        prepare_robot_grippers(runtime)
        import numpy as np

        np.testing.assert_allclose(
            runtime.solver_cfg[f"{side}_arm"].tcp,
            expected["solver_cfg"][f"{side}_arm"]["tcp"],
            rtol=0,
            atol=1e-6,
        )
        expected["solver_cfg"][f"{side}_arm"].pop("tcp")

    def components(values):
        result = deepcopy(values)
        result["urdf_cfg"]["components"] = sorted(
            [
                {
                    k: v
                    for k, v in c.items()
                    if k not in {"source_root_link", "mesh_paths"}
                }
                for c in result["urdf_cfg"]["components"]
            ],
            key=lambda c: c["component_type"],
        )
        for item in result["urdf_cfg"]["components"]:
            item.setdefault(
                "transform",
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        return json.loads(json.dumps(result))

    expected["joint_drive_props"].setdefault("drive_type", "force")
    assert components(actual) == components(expected)
    declaration = physical.embodiment_skill_profile
    assert declaration is not None
    services = declaration["runtime_services"]
    for part, generator in services["grasp_pose_generators"].items():
        previous = deepcopy(
            frozen["skill_profile"]["runtime_services"]["grasp_pose_generators"][part]
        )
        previous.pop("model")
        previous.pop("root_from_tcp", None)
        previous.pop("root_from_contact_frame", None)
        assert {
            key: value
            for key, value in generator.items()
            if key not in {"gripper", "min_opening_width", "max_opening_width"}
        } == previous
        old_presets = frozen["skill_profile"]["command_presets"]
        old_commands = next(
            x["commands"] for x in old_presets if x["control_part"] == part
        )
        assert (
            next(
                x["commands"]
                for x in declaration["command_presets"]
                if x["control_part"] == part
            )
            == old_commands
        )
    for side in ("left", "right"):
        hand = services["hand_geometries"][f"{side}_eef"]
        assert (
            hand["sample_count"],
            hand["body_max_edge"],
            hand["pad_max_edge"],
            hand["max_subdivisions"],
        ) == (65, 0.01, 0.004, 10)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=declaration,
        base_dir=path.parent,
    )
    assert (
        deployment.integration.registration.robot_profile_binding.profile_id
        == component
    )


@pytest.mark.parametrize("field", ["opening_margin", "point_sample_density"])
def test_grasp_generator_rejects_removed_box_collision_fields(field: str) -> None:
    with pytest.raises(ValueError, match=f"unsupported fields.*{field}"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "gripper": {"class_type": "DHPGI14080"},
                field: 0.01,
            },
            path="generator",
        )


def test_inline_builtin_aperture_matches_registered_model_and_rejects_unknown() -> None:
    payload = {
        "kind": "antipodal_parallel_jaw",
        "gripper": {"class_type": "FrankaPandaHand"},
    }
    factory = _decode_grasp_generator(payload, path="generator")
    assert factory.min_opening_width == 0.001
    assert factory.max_opening_width == 0.08
    payload["gripper"] = {"class_type": "custom"}
    with pytest.raises(
        ValueError,
        match="no parallel-jaw description",
    ):
        _decode_grasp_generator(payload, path="generator")


@pytest.mark.parametrize(
    ("relative_path", "control_part", "model_id"),
    [
        (
            "repeated_pick_place/task.franka.yaml",
            "eef",
            "franka_panda_hand",
        ),
        ("open_drawer/task.franka.yaml", "eef", "franka_panda_hand"),
        (
            "tableware/pour_water/task.cobotmagic.yaml",
            "right_eef",
            "cobotmagic_v100_gripper",
        ),
    ],
)
def test_embodiment_grasp_defaults_reach_composed_deployments(
    relative_path: str, control_part: str, model_id: str
) -> None:
    """Embodiments select geometry; tasks can still override collision policy."""
    path = _CONFIG_DIRECTORY / relative_path
    physical = _physical_components(path, load_config(path))
    profile = physical.embodiment_skill_profile
    assert profile is not None
    declaration = profile["runtime_services"]["grasp_pose_generators"][control_part]
    assert declaration["kind"] == "antipodal_parallel_jaw"
    assert declaration["gripper"]["grasp_cfg"]["model_id"] == model_id

    deployment = _deployment_from_path(path)
    delegate = deployment.integration.adapter_factory.delegate
    factory = delegate._grasp_pose_generator_factories[control_part]
    generator = factory()

    assert generator.gripper_model.model_id == model_id
    assert generator.algorithm_cfg == AntipodalGraspPoseGeneratorCfg()
    assert generator.collision_cfg == ParallelJawGraspCollisionCfg()
    assert generator.collision_geometry.provenance["model_id"] == model_id
    assert generator.annotation_cfg == GraspAnnotationCfg()
    assert factory() is not generator


@pytest.mark.parametrize("side", ["left", "right"])
def test_marvin_embodiment_grasp_defaults_compose_with_bimanual_task(
    side: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Marvin hands use source joint commands and toolkit policy defaults."""
    monkeypatch.setattr(
        "embodichain_robots.tianji.marvin.cfg.get_data_path", lambda value: value
    )
    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    config["embodiment"] = {
        "component": "../../../components/embodiments/tianji_marvin.yaml"
    }
    physical = _physical_components(path, config)
    robot_values = dict(physical.config["robot"])
    assert robot_values.pop("class_type") == "TianjiMarvin"
    robot = TianjiMarvinCfg.from_dict(robot_values)
    assert robot.with_gripper is True
    assert len(robot.control_parts[f"{side}_arm"]) == 7
    assert len(robot.control_parts[f"{side}_eef"]) == 2
    assert [sensor["uid"] for sensor in physical.config["sensor"]] == ["cam_high"]

    profile = physical.embodiment_skill_profile
    assert profile is not None
    assert profile["profile_id"] == "tianji_marvin"
    resource = next(
        resource for resource in profile["resources"] if resource["resource_id"] == side
    )
    assert {
        endpoint["endpoint_id"]: endpoint["control_part"]
        for endpoint in resource["endpoints"]
    } == {"motion": f"{side}_arm", "grasp": f"{side}_eef"}
    commands = next(
        preset["commands"]
        for preset in profile["command_presets"]
        if preset["control_part"] == f"{side}_eef"
    )
    # Both source prismatic joints span [-0.05, 0]; decreasing qpos closes.
    assert commands == {"open": [0.0, 0.0], "grasp": [-0.05, -0.05]}
    services = profile["runtime_services"]["grasp_pose_generators"]
    assert set(services) == {"left_eef", "right_eef"}
    declaration = services[f"{side}_eef"]
    assert declaration["kind"] == "antipodal_parallel_jaw"
    assert declaration["gripper"]["grasp_cfg"]["model_id"] == "tianji_marvin_gripper"
    assert declaration["gripper"]["binding"]["variant"] == side
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=profile,
        base_dir=path.parent,
    )
    registration = deployment.integration.registration
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=registration.catalog,
    )
    registration.catalog.preflight(program)
    delegate = deployment.integration.adapter_factory.delegate
    factory = delegate._grasp_pose_generator_factories[f"{side}_eef"]
    generator = factory()
    assert generator.gripper_model.model_id == "tianji_marvin_gripper"
    assert (
        generator.collision_geometry.provenance["root_link"] == f"{side}_hand_base_link"
    )
    # Resolve the real rounded URDF tool joint and this deployment's solver
    # override independently; the model's default TCP differs from this one.
    import xml.etree.ElementTree as ET
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.data import get_data_path

    origin = ET.parse(get_data_path("TianjiMarvin/robot_with_ee_acd.urdf")).find(
        f"./joint[@name='{side.upper()}_HAND_TOOL']/origin"
    )
    root_from_tool = np.eye(4)
    root_from_tool[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
    ).as_matrix()
    root_from_tool[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
    # This embodiment selects its calibrated root TCP through the hand binding.
    root_from_tcp = np.asarray(declaration["gripper"]["default_root_from_tcp"])
    np.testing.assert_allclose(
        root_from_tcp[:3, 3], [0.0, -3.0144258464370744e-09, 0.1125], rtol=0, atol=1e-12
    )
    geometry = generator.collision_geometry
    default = build_gripper_collision_geometry(
        resolve_gripper_defaults("TianjiMarvinGripper", variant=side)
    )
    torch.testing.assert_close(geometry.points, default.points, rtol=0, atol=1e-8)
    canonical = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    actual_pose = geometry.canonical_to_tcp_poses(canonical)[0].numpy()
    np.testing.assert_allclose(
        np.asarray(geometry.provenance["root_from_contact_frame"]) @ actual_pose,
        root_from_tcp,
        rtol=0,
        atol=1e-12,
    )
    assert not np.allclose(
        actual_pose, default.canonical_to_tcp_poses(canonical)[0].numpy()
    )
    assert generator.algorithm_cfg == AntipodalGraspPoseGeneratorCfg()
    assert generator.collision_cfg == ParallelJawGraspCollisionCfg()
    assert generator.annotation_cfg == GraspAnnotationCfg()


def test_grasp_generator_rejects_unknown_models_and_removed_viser_port() -> None:
    """The integration keeps a closed model catalog and no interactive port."""
    with pytest.raises(ValueError, match="no parallel-jaw description"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "gripper": {"class_type": "missing_gripper"},
            },
            path="generator",
        )

    with pytest.raises(ValueError, match="unsupported fields.*viser_port"):
        _decode_grasp_generator(
            {
                "kind": "antipodal_parallel_jaw",
                "gripper": {"class_type": "DHPGI14080"},
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


def _rigidized_articulation_payload() -> dict[str, object]:
    """Return an integration whose object is a locked native articulation."""
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    scene = payload["scene"]
    assert type(scene) is dict
    scene["rigidized_articulations"] = [
        {
            "entity_id": "rubiks_cube",
            "simulation_uid": "cube_articulation",
            "locked_qpos": {"top_turn": 0.0},
            "dynamics": "dynamic",
            "semantic_type": "rubiks_cube",
            "affordances": [
                {
                    "kind": "antipodal_grasp",
                    "entity_id": "rubiks_cube_grasp",
                    "grasp_link": "lower_two_layers",
                }
            ],
        }
    ]
    return payload


def test_rigidized_articulation_scene_decodes_link_backed_grasp() -> None:
    binding = _decode_configured_task_program_integration(
        _rigidized_articulation_payload()
    ).registration.scene_binding

    assert dict(binding.rigidized_articulations[0].locked_qpos) == {"top_turn": 0.0}
    assert binding.rigidized_articulation_grasps[0].grasp_link == "lower_two_layers"


@pytest.mark.parametrize("collision_role", ("static", "dynamic"))
def test_rigidized_articulation_config_rejects_unavailable_collision_geometry(
    collision_role: str,
) -> None:
    """Configured articulation objects cannot claim a planner collision role."""
    payload = _rigidized_articulation_payload()
    payload["scene"]["rigidized_articulations"][0]["collision_role"] = collision_role

    with pytest.raises(ValueError, match="rigidized.*collision_role.*none"):
        _decode_configured_task_program_integration(payload)


def test_rigidized_articulation_requires_locked_qpos() -> None:
    payload = _rigidized_articulation_payload()
    root = payload["scene"]["rigidized_articulations"][0]
    root.pop("locked_qpos")

    with pytest.raises(ValueError, match="missing required fields.*locked_qpos"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize(
    ("locked_qpos", "exception", "message"),
    [
        ({}, ValueError, "must not be empty"),
        ([], TypeError, "must be a mapping"),
        ({"top_turn": True}, TypeError, "finite number"),
        ({"top_turn": float("inf")}, ValueError, "must be finite"),
    ],
)
def test_rigidized_articulation_rejects_invalid_locked_qpos(
    locked_qpos: object,
    exception: type[Exception],
    message: str,
) -> None:
    payload = _rigidized_articulation_payload()
    payload["scene"]["rigidized_articulations"][0]["locked_qpos"] = locked_qpos

    with pytest.raises(exception, match=message):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize(
    "field_name",
    ("joint_position_tolerance", "link_transform_tolerance"),
)
def test_rigidized_articulation_rejects_unsafe_large_tolerance(
    field_name: str,
) -> None:
    payload = _rigidized_articulation_payload()
    root = payload["scene"]["rigidized_articulations"][0]
    root[field_name] = 10.0

    with pytest.raises(ValueError, match=f"{field_name}.*at most"):
        _decode_configured_task_program_integration(payload)


def test_rigidized_articulation_grasp_requires_grasp_link() -> None:
    payload = _rigidized_articulation_payload()
    grasp = payload["scene"]["rigidized_articulations"][0]["affordances"][0]
    grasp.pop("grasp_link")

    with pytest.raises(ValueError, match="missing required fields.*grasp_link"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize("field_name", ("mesh_env_id", "internal_axis"))
def test_rigidized_articulation_grasp_rejects_rigid_object_fields(
    field_name: str,
) -> None:
    payload = _rigidized_articulation_payload()
    grasp = payload["scene"]["rigidized_articulations"][0]["affordances"][0]
    grasp[field_name] = 0 if field_name == "mesh_env_id" else [1.0, 0.0, 0.0]

    with pytest.raises(ValueError, match=f"unsupported fields.*{field_name}"):
        _decode_configured_task_program_integration(payload)


def test_rigid_object_grasp_rejects_grasp_link() -> None:
    payload = deepcopy(_integration_payload("repeated_pick_place"))
    grasp = payload["scene"]["rigid_objects"][0]["affordances"][0]
    grasp["grasp_link"] = "lower_two_layers"

    with pytest.raises(ValueError, match="unsupported fields.*grasp_link"):
        _decode_configured_task_program_integration(payload)


def test_rigidized_articulation_accepts_nested_placement_affordance() -> None:
    payload = _rigidized_articulation_payload()
    root = payload["scene"]["rigidized_articulations"][0]
    root["affordances"].append(
        {
            "kind": "support_surface",
            "entity_id": "rubiks_cube_top",
            "native_name": "top",
        }
    )

    binding = _decode_configured_task_program_integration(
        payload
    ).registration.scene_binding

    assert binding.support_surfaces[0].parent_id == "rubiks_cube"


def test_scene_rejects_flat_rigidized_articulation_grasps() -> None:
    payload = _rigidized_articulation_payload()
    payload["scene"]["rigidized_articulation_grasps"] = []

    with pytest.raises(
        ValueError,
        match="unsupported fields.*rigidized_articulation_grasps",
    ):
        _decode_configured_task_program_integration(payload)


def test_rigidized_articulation_rejects_unknown_root_field() -> None:
    payload = _rigidized_articulation_payload()
    payload["scene"]["rigidized_articulations"][0]["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields.*unexpected"):
        _decode_configured_task_program_integration(payload)


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
            assert generator["gripper"]["grasp_cfg"]["model_id"] == "dh_pgi_140_80"
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
    """Pure configuration-defined examples do not need Python task modules.

    ``repeated_pick_place`` also ships import-registered RLinf variants, so
    its task module is intentional even though the original Task Program
    deployments remain configuration-defined.
    """
    for task_name in ("open_drawer", "hand_over"):
        module_name = f"embodichain_tasks.manipulation.{task_name}"
        assert importlib.util.find_spec(module_name) is None


__all__: list[str] = []


@pytest.mark.parametrize("kind", ["release_safe_pick", "move_held_object_upright"])
def test_configured_services_reject_task_specific_e2_kinds(kind: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _decode_registered_lowerer({"kind": kind, "routes": []}, path="lowerer")


@pytest.mark.parametrize("mode", ["auto", "zero"])
def test_motion_velocity_target_policy_decodes(mode: str) -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    policy = _decode_motion_policy(
        {"sample_count": 40, "velocity_targets": mode}, path="motion"
    )
    assert policy.velocity_targets == mode
    assert policy.strategy == "ik_interp"


@pytest.mark.parametrize("mode", ["keep_old", "required"])
def test_motion_velocity_target_policy_rejects_unknown_mode(mode: str) -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    with pytest.raises(ValueError, match="velocity_targets"):
        _decode_motion_policy(
            {"sample_count": 40, "velocity_targets": mode}, path="motion"
        )


def test_motion_limit_policy_decodes() -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    policy = _decode_motion_policy(
        {
            "sample_count": 40,
            "velocity_limit": 0.2,
            "acceleration_limit": 0.5,
        },
        path="motion",
    )

    assert policy.velocity_limit == pytest.approx(0.2)
    assert policy.acceleration_limit == pytest.approx(0.5)


def test_motion_limit_policy_preserves_omitted_velocity_limit() -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    policy = _decode_motion_policy(
        {"sample_count": 40, "acceleration_limit": 0.5},
        path="motion",
    )

    assert policy.velocity_limit is None


@pytest.mark.parametrize("field", ["velocity_limit", "acceleration_limit"])
@pytest.mark.parametrize("value", [0.0, -0.1, float("inf")])
def test_motion_limit_policy_rejects_nonpositive_or_nonfinite_values(
    field: str,
    value: float,
) -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    with pytest.raises(ValueError, match=field):
        _decode_motion_policy(
            {"sample_count": 40, field: value},
            path="motion",
        )


def test_pour_dwell_duration_decodes() -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
    )

    options = _decode_action_options(
        {"kind": "pour", "dwell_duration": 0.75},
        path="options",
    )

    assert isinstance(options, PourOptions)
    assert options.dwell_duration == pytest.approx(0.75)


def test_configured_pour_target_lowers_live_geometry_and_is_closed() -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_registered_lowerer,
    )

    source = torch.eye(4)
    source[2, 3] = 0.07
    receiver = torch.eye(4)
    receiver[2, 3] = 0.04
    payload = {
        "kind": "pour",
        "object_id": "bottle",
        "target": {
            "receiver_id": "cup",
            "source_mouth_pose": source.flatten().tolist(),
            "receiver_mouth_pose": receiver.flatten().tolist(),
            "receiver_radius": 0.03,
            "mouth_radius": 0.01,
        },
    }
    factory = _decode_registered_lowerer(payload, path="lowerer")
    result = _PourLowerer("bottle", target=factory.target).lower(
        RegisteredSemanticCall(
            call_id="simulation.pour", arguments={"object": "bottle"}
        ),
        context=None,
        bound=None,
        option_template=PourOptions(),
    )
    assert result.goal.target.receiver_pose.entity_id == "cup"
    torch.testing.assert_close(result.goal.target.receiver_pose.relative_pose, receiver)
    torch.testing.assert_close(result.goal.target.source_mouth_pose, source)
    invalid = deepcopy(payload)
    invalid["target"]["world_bottle_pose"] = source.flatten().tolist()
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_registered_lowerer(invalid, path="lowerer")
    invalid = deepcopy(payload)
    invalid["target"]["receiver_id"] = "bottle"
    with pytest.raises(ValueError, match="distinct"):
        _decode_registered_lowerer(invalid, path="lowerer")


@pytest.mark.parametrize("value", [-0.1, float("inf")])
def test_pour_dwell_duration_rejects_negative_or_nonfinite_values(
    value: float,
) -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
    )

    with pytest.raises(ValueError, match="dwell_duration"):
        _decode_action_options(
            {"kind": "pour", "dwell_duration": value},
            path="options",
        )


@pytest.mark.parametrize(
    ("variant", "arm_dof", "velocity"),
    [("dual_ur5", 6, 0.5), ("tianji_marvin", 7, 0.2)],
)
def test_configured_motion_budgets_reach_planner_options_and_metadata(
    variant: str, arm_dof: int, velocity: float
) -> None:
    """The qualified speed budget survives deployment decoding and reporting."""
    from embodichain.lab.task_program.runtime.results import _motion_policy_to_metadata

    deployment = _deployment_from_path(
        _CONFIG_DIRECTORY / "hand_over" / f"task.{variant}.yaml"
    )
    policy = deployment.integration.registration.robot_profile_binding.presets[
        0
    ].motion_policy
    options = policy.to_motion_gen_options(
        start_qpos=torch.zeros(1, arm_dof), control_part="left_arm"
    )
    assert options.velocity_limit == velocity
    assert options.acceleration_limit == 0.5
    metadata = _motion_policy_to_metadata(policy)
    assert metadata["velocity_limit"] == velocity
    assert metadata["acceleration_limit"] == 0.5


@pytest.mark.parametrize("field", ["velocity_limit", "acceleration_limit"])
@pytest.mark.parametrize("value", [0, -0.1, True, float("nan"), float("inf"), "fast"])
def test_configured_motion_budgets_reject_invalid_values(
    field: str, value: object
) -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )

    with pytest.raises((TypeError, ValueError), match=field):
        _decode_motion_policy(
            {"strategy": "motion_gen", "sample_count": 140, field: value},
            path="motion",
        )


@pytest.mark.parametrize(
    ("task_name", "backend"),
    [
        ("rubiks_cube_pick_place", "default"),
        ("repeated_pick_place", "default"),
        ("repeated_pick_place", "newton"),
        ("open_drawer", "default"),
        ("open_drawer", "newton"),
        ("hand_over", "default"),
        ("pour_water", "default"),
    ],
)
def test_segmented_examples_preserve_overall_and_segment_instructions(
    task_name: str,
    backend: str,
    registered_test_ids: list[str],
) -> None:
    """Production composition retains both language levels for every backend."""
    path = (
        _tableware_config_path(task_name)
        if task_name in _TABLEWARE_TASKS
        else _config_path(task_name)
    )
    config = load_config(path)
    if backend == "newton":
        config["environment"] = {"component": config["environment"][backend]}
    config["id"] = _TEST_ENV_ID
    registered_test_ids.append(_TEST_ENV_ID)
    cfg = config_to_cfg(config, source_path=path)
    assert cfg.task_program is not None
    overall = cfg.task_program.instruction
    assert "instruction" not in cfg.dataset.lerobot.params
    assert isinstance(overall, str) and overall.strip()
    assert overall != "unknown_task"

    physical = _physical_components(path, config)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    catalog = deployment.integration.registration.catalog
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=catalog,
    )
    compiled = catalog.preflight(program)
    assert compiled.instruction == overall
    segments = tuple(compiled.iter_segments())
    expected_names = {
        "rubiks_cube_pick_place": ["pick_rubiks_cube", "place_rubiks_cube"] * 3,
        "repeated_pick_place": ["pick_cube", "place_cube"] * 3,
        "open_drawer": ["open_drawer"],
        "hand_over": ["hand_over_can"],
        "pour_water": [
            "pick_bottle",
            "move_bottle_above_cup",
            "pour_water",
            "return_bottle",
        ],
    }
    assert [segment.name for segment in segments] == expected_names[task_name]
    if task_name == "pour_water":
        assert [len(segment.post_policies) for segment in segments] == [0, 0, 0, 2]
        assert [len(segment.validators) for segment in segments] == [0, 1, 0, 2]
    for segment in segments:
        assert isinstance(segment.instruction, str) and segment.instruction.strip()
        assert segment.instruction != overall


def test_trapezoidal_motion_options_reach_the_runtime_policy() -> None:
    payload = _integration_payload("repeated_pick_place")
    preset = payload["robot_profile"]["presets"][0]
    preset.pop("required_planner", None)
    preset["motion"] = {
        "strategy": "motion_gen",
        "sample_count": 120,
        "plan_opts": {
            "kind": "trapezoidal",
            "stop_at_waypoints": False,
            "blend_tolerance": 0.001,
            "collinearity_tolerance": 0.0001,
        },
    }
    integration = _decode_configured_task_program_integration(payload)
    decoded = integration.registration.robot_profile_binding.presets[0]
    options = decoded.motion_policy.to_motion_gen_options(
        start_qpos=torch.zeros(1, 6), control_part="arm", interpolation_dt=0.01
    )
    assert decoded.required_planner == "trapezoidal"
    assert isinstance(options.plan_opts, TrapezoidalPlanOptions)
    assert not options.plan_opts.stop_at_waypoints
    assert options.plan_opts.blend_tolerance == 0.001
    assert options.plan_opts.collinearity_tolerance == 0.0001
    assert options.plan_opts.sample_interval == options.sample_count == 120
    # Resolve robot speed caps and the planner acceleration default at planning time.
    assert options.plan_opts.constraints["velocity"] is None
    assert options.plan_opts.constraints["acceleration"] is None


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"kind": "unknown"}, ValueError, "kind"),
        ({"kind": "trapezoidal", "unknown": 1}, ValueError, "unsupported"),
        ({"kind": "trapezoidal", "stop_at_waypoints": "false"}, TypeError, "bool"),
        ({"kind": "trapezoidal", "blend_tolerance": True}, TypeError, "number"),
        ({"kind": "trapezoidal", "blend_tolerance": -0.1}, ValueError, "finite"),
        (
            {"kind": "trapezoidal", "blend_tolerance": float("nan")},
            ValueError,
            "finite",
        ),
        (
            {"kind": "trapezoidal", "blend_tolerance": float("inf")},
            ValueError,
            "finite",
        ),
        (
            {"kind": "trapezoidal", "blend_tolerance": 0.001},
            ValueError,
            "stop_at_waypoints",
        ),
        (
            {"kind": "trapezoidal", "collinearity_tolerance": 1.0},
            ValueError,
            "collinearity",
        ),
    ],
)
def test_trapezoidal_motion_options_reject_invalid_configuration(
    options: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _decode_motion_policy(
            {"strategy": "motion_gen", "sample_count": 120, "plan_opts": options},
            path="motion",
        )


def test_trapezoidal_motion_options_require_backend_planning() -> None:
    with pytest.raises(ValueError, match="requires strategy='motion_gen'"):
        _decode_motion_policy(
            {"sample_count": 120, "plan_opts": {"kind": "trapezoidal"}}, path="motion"
        )
    payload = _integration_payload("repeated_pick_place")
    preset = payload["robot_profile"]["presets"][0]
    preset["required_planner"] = "toppra"
    preset["motion"] = {
        "strategy": "motion_gen",
        "sample_count": 120,
        "plan_opts": {"kind": "trapezoidal"},
    }
    with pytest.raises(ValueError, match="requires the trapezoidal planner"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize(
    "limits",
    [
        {"velocity_limit": 0.05},
        {"acceleration_limit": 0.1},
        {"velocity_limit": 0.05, "acceleration_limit": 0.1},
    ],
)
def test_motion_limits_reach_explicit_blended_planner_options(
    limits: dict[str, float],
) -> None:
    """Configured scalar budgets still constrain explicitly selected blending."""
    policy = _decode_motion_policy(
        {
            "strategy": "motion_gen",
            "sample_count": 40,
            **limits,
            "plan_opts": {
                "kind": "trapezoidal",
                "stop_at_waypoints": False,
                "blend_tolerance": 0.001,
            },
        },
        path="motion",
    )
    options = policy.to_motion_gen_options(
        start_qpos=torch.zeros(1, 6), control_part="arm", interpolation_dt=0.01
    )
    assert isinstance(options.plan_opts, TrapezoidalPlanOptions)
    assert options.plan_opts.constraints["velocity"] == limits.get("velocity_limit")
    assert options.plan_opts.constraints["acceleration"] == limits.get(
        "acceleration_limit"
    )
    assert TrapezoidalPlanOptions().constraints["velocity"] is None
    assert TrapezoidalPlanOptions().constraints["acceleration"] is None


def test_configured_nominal_source_preflights_content_and_revision(tmp_path):
    import hashlib
    import numpy as np

    from embodichain.data_pipeline.real2sim import (
        JointSpec,
        PhaseAnnotation,
        Real2SimTrajectory,
        prepare_prototype,
    )
    from embodichain.lab.task_program.integrations.configured import (
        _decode_runtime_services,
    )

    source = Real2SimTrajectory(
        source_id="synthetic",
        source_revision="1",
        unit_id="pick",
        skill_id="pick_up",
        origin="synthetic_reference",
        joint_names=("arm",),
        joint_units=("rad",),
        qpos=np.zeros((6, 1)),
        timestamps=np.arange(6) * 0.02,
        time_unit="s",
        phases=(
            PhaseAnnotation("approach", 0, 2, "free", "object"),
            PhaseAnnotation("close", 2, 4, "contact", "object"),
            PhaseAnnotation("lift", 4, 6, "hold", "gravity"),
        ),
        object_pose=np.eye(4),
    )
    manifest = prepare_prototype(
        source,
        joints=(JointSpec("arm", "revolute"),),
        control_dt=0.02,
        directory=tmp_path / "prototype",
    )
    declaration = {
        "call_index": 0,
        "program_fingerprint": "a" * 64,
        "provider_id": "real2sim.prototype",
        "revision": "1",
        "prototype_manifest": str(manifest),
        "prototype_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "object_entity_id": "cube",
    }
    services = _decode_runtime_services({"nominal_motion_sources": [declaration]})
    assert services.nominal_motion_sources[0].call_index == 0
    refined = _decode_runtime_services(
        {
            "nominal_motion_sources": [
                {
                    **declaration,
                    "authority": "eef_pose",
                    "refinement": {
                        "max_translation_m": 0.015,
                        "project_pick_lift": True,
                    },
                }
            ]
        }
    )
    assert refined.nominal_motion_sources[0].refinement.max_translation_m == 0.015
    assert refined.nominal_motion_sources[0].refinement.project_pick_lift is True
    with pytest.raises(ValueError, match="eef_pose authority"):
        _decode_runtime_services(
            {"nominal_motion_sources": [{**declaration, "refinement": {}}]}
        )
    with pytest.raises(ValueError, match="unknown|unexpected|unsupported"):
        _decode_runtime_services(
            {
                "nominal_motion_sources": [
                    {
                        **declaration,
                        "authority": "eef_pose",
                        "refinement": {"silently_change_time": True},
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="unknown provider or revision"):
        _decode_runtime_services(
            {"nominal_motion_sources": [{**declaration, "revision": "2"}]}
        )
    with pytest.raises(ValueError, match="changed"):
        _decode_runtime_services(
            {"nominal_motion_sources": [{**declaration, "prototype_sha256": "b" * 64}]}
        )


@pytest.mark.parametrize("asset_backed", [False, True])
def test_real2sim_collection_profile_rejects_ambiguous_limits_and_recipes(
    tmp_path, monkeypatch, asset_backed
):
    import yaml
    from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
        load_real2sim_tabletop_profile,
    )

    payload = {
        "kind": "tabletop_box_v1",
        "scene_revision": "scene-v1",
        "object_uid": "cube",
        "target_uid": "platform",
        "object_entity_id": "cube",
        "target_entity_id": "platform",
        "base_link": "base",
        "finger_links": ["finger"],
        "finger_joints": ["finger_joint"],
        "closing_directions": [1],
        "velocity_limits": [1.0, 2.0],
        "velocity_limit_origin": "simulated",
        "acceleration_limits": [1.0, 2.0],
        "acceleration_limit_origin": "simulated",
        "layouts": [
            {"id": "baseline", "object_xyyaw": [0, 0, 0], "target_xyyaw": [0, 0, 0]}
        ],
        "recipes": ["nominal", "time_130"],
    }
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(payload))
    profile = load_real2sim_tabletop_profile(path)
    assert profile.recipes == ("nominal", "time_130")
    assert profile.layouts[0]["object_xyyaw"] == (0.0, 0.0, 0.0)
    assert profile.motion_headroom_fraction is None

    artifacts = {
        "original_recording": {"path": "original.h5", "sha256": "a" * 64},
        "source_evidence": {"path": "source.npz", "sha256": "b" * 64},
    }
    if asset_backed:
        from types import SimpleNamespace

        for item in artifacts.values():
            (tmp_path / item["path"]).write_text("fixture")
            item["path"] = {"asset": "DemoInputs", "path": item["path"]}
        monkeypatch.setattr(
            "embodichain.data.get_data_class",
            lambda _: lambda: SimpleNamespace(extract_dir=str(tmp_path)),
        )
    json_path = tmp_path / "profile.json"
    import json

    json_path.write_text(
        json.dumps(
            {
                **payload,
                "source_artifacts": {0: artifacts},
                "retime_phase_allowlist": {"place": ["approach"]},
            }
        )
    )
    decoded = load_real2sim_tabletop_profile(json_path)
    assert list(decoded.source_artifacts) == [0]
    assert decoded.source_artifacts[0]["original_recording"]["path"] == str(
        tmp_path / "original.h5"
    )
    assert decoded.retime_phase_allowlist == {"place": ("approach",)}
    for keys in ({"00": artifacts}, {0: artifacts, "0": artifacts}, {True: artifacts}):
        path.write_text(yaml.safe_dump({**payload, "source_artifacts": keys}))
        with pytest.raises(ValueError, match="call indices"):
            load_real2sim_tabletop_profile(path)

    path.write_text(yaml.safe_dump({**payload, "motion_headroom_fraction": 0.95}))
    assert load_real2sim_tabletop_profile(path).motion_headroom_fraction == 0.95
    for invalid in (True, 0, 1):
        path.write_text(
            yaml.safe_dump({**payload, "motion_headroom_fraction": invalid})
        )
        with pytest.raises(ValueError, match="motion_headroom_fraction"):
            load_real2sim_tabletop_profile(path)

    path.write_text(yaml.safe_dump({**payload, "acceleration_limits": [True, 2.0]}))
    with pytest.raises(ValueError, match="cannot be booleans"):
        load_real2sim_tabletop_profile(path)
    path.write_text(yaml.safe_dump({**payload, "recipes": ["nominal", "nominal"]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_real2sim_tabletop_profile(path)
    path.write_text(yaml.safe_dump({**payload, "unexpected": 1}))
    with pytest.raises(ValueError, match="unknown Real2Sim profile field"):
        load_real2sim_tabletop_profile(path)


@pytest.mark.parametrize("relative", [False, True])
def test_routed_transport_is_declared_owned_and_lowered_without_live_reads(relative):
    from types import SimpleNamespace
    from embodichain.lab.task_program.semantics import (
        SceneEntityRegistration,
        SceneRegistry,
        SemanticEffectKind,
    )
    from embodichain.lab.task_program.integrations.simulation.transport import (
        RoutedMoveHeldObjectLowererFactory,
    )

    pose = (
        {
            "kind": "scene_entity",
            "entity_id": "cup",
            "relative_pose": torch.eye(4).reshape(-1).tolist(),
            "world_orientation": torch.eye(3).reshape(-1).tolist(),
        }
        if relative
        else {
            "kind": "pose",
            "position": [0.1, 0.2, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 2.0],
        }
    )
    raw = {
        "kind": "move_held_object",
        "routes": [{"object_id": "bottle", "target_id": "destination", "pose": pose}],
    }
    payload = _tableware_integration_payload("pour_water")
    lowerers = payload["runtime_services"]["registered_semantic_lowerers"]
    index = next(
        i for i, item in enumerate(lowerers) if item["kind"] == "move_held_object"
    )
    lowerers[index] = raw
    configured = _decode_configured_task_program_integration(payload)
    factory = next(
        f
        for f in configured.registration.registered_semantic_lowerer_factories
        if f.call_id == "simulation.move_held_object"
    )
    assert type(factory) is RoutedMoveHeldObjectLowererFactory
    fingerprint = configured.registration.fingerprint_payload()
    raw["routes"][0]["object_id"] = "mutated"
    if relative:
        pose["relative_pose"][3] = 99.0
    else:
        pose["position"][0] = 99.0
    assert configured.registration.fingerprint_payload() == fingerprint
    robot = object()
    registry = SceneRegistry(
        [
            SceneEntityRegistration(
                SceneObjectRef(name),
                state_provider=SimpleNamespace(
                    observe=lambda **kwargs: pytest.fail("lowering read live state")
                ),
            )
            for name in ("bottle", "cup")
        ]
    )
    lowerer = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    other = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    assert other is not lowerer
    arguments = {"target": "destination", "object": "bottle"}
    if relative:
        arguments["reference"] = "cup"
    call = RegisteredSemanticCall(call_id=factory.call_id, arguments=arguments)
    lowered = lowerer.lower(
        call, context=None, bound=None, option_template=MoveHeldObjectOptions()
    )
    assert lowered.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert lowered.registered_effect.held_objects[0].object_id == "bottle"
    target = lowered.goal.object_target_pose
    if relative:
        assert target.entity_id == "cup"
        torch.testing.assert_close(target.relative_pose, torch.eye(4))
        torch.testing.assert_close(target.world_orientation, torch.eye(3))
    else:
        torch.testing.assert_close(target[:3, 3], torch.tensor([0.1, 0.2, 0.3]))
        torch.testing.assert_close(target[:3, :3], torch.eye(3))
    lookahead = lowerer.pick_lookahead_targets(
        call, picked_object=SceneObjectRef("bottle"), bound=None, previous_target=None
    )
    assert len(lookahead) == 1
    assert (
        lowerer.pick_lookahead_targets(
            call, picked_object=SceneObjectRef("cup"), bound=None, previous_target=None
        )
        is None
    )
    target_matrix = target.relative_pose if relative else target
    target_matrix.fill_(99)
    again = lowerer.lower(
        call, context=None, bound=None, option_template=MoveHeldObjectOptions()
    ).goal.object_target_pose
    assert not torch.equal(again.relative_pose if relative else again, target_matrix)
    with pytest.raises(ValueError, match="object-target route"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=factory.call_id,
                arguments={"target": "destination", "object": "cup"},
            ),
            context=None,
            bound=None,
            option_template=MoveHeldObjectOptions(),
        )


@pytest.mark.parametrize(
    "damage", ["mixed", "duplicate", "unknown_pose_field", "nonrigid"]
)
def test_routed_transport_rejects_ambiguous_or_invalid_declarations(damage):
    route = {
        "object_id": "bottle",
        "target_id": "destination",
        "pose": {
            "kind": "scene_entity",
            "entity_id": "cup",
            "relative_pose": torch.eye(4).reshape(-1).tolist(),
        },
    }
    raw = {"kind": "move_held_object", "routes": [route]}
    if damage == "mixed":
        raw["target_id"] = "legacy"
    elif damage == "duplicate":
        raw["routes"].append(deepcopy(route))
    elif damage == "unknown_pose_field":
        route["pose"]["planner"] = "arbitrary"
    else:
        route["pose"]["relative_pose"][0] = 2.0
    with pytest.raises((TypeError, ValueError)):
        _decode_registered_lowerer(raw, path="lowerer")


@pytest.mark.parametrize("missing", ["object", "reference"])
def test_routed_transport_missing_entities_reject_before_runtime_assembly(missing):
    payload = _tableware_integration_payload("pour_water")
    routes = [
        {
            "object_id": "missing" if missing == "object" else "bottle",
            "target_id": "destination",
            "pose": {
                "kind": "scene_entity",
                "entity_id": "missing" if missing == "reference" else "cup",
            },
        }
    ]
    lowerers = payload["runtime_services"]["registered_semantic_lowerers"]
    index = next(
        i for i, item in enumerate(lowerers) if item["kind"] == "move_held_object"
    )
    lowerers[index] = {"kind": "move_held_object", "routes": routes}
    with pytest.raises(ValueError, match="missing"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize("value", [(True,) * 16, ("invalid",) * 16])
def test_public_scene_target_rejects_non_numeric_matrix_with_type_error(value):
    from embodichain.lab.task_program.integrations.simulation.targets import (
        SceneEntityTarget,
    )

    with pytest.raises(TypeError, match="relative_pose.*sixteen|relative_pose.*16"):
        SceneEntityTarget("cup", relative_pose=value)


def test_configured_settle_table_is_closed_owned_and_uses_the_standard_registration():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
    )
    from embodichain.lab.task_program.integrations.simulation.policies import (
        SimulationSettleMonitorCfg,
    )
    from embodichain.lab.task_program.integrations.catalog import (
        IntegrationFingerprintMismatch,
    )
    from embodichain.lab.task_program import WaitStablePostCfg

    payload = _integration_payload("repeated_pick_place")
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "position_check": {
            "kind": "placement",
            "entity": "cube",
            "target_position": [0.5, 0.0, 0.1],
            "duration": 0.08,
            "timeout": 0.4,
        },
        "pose_speed": {"measurement": "pose_delta", "min_steps": 1},
    }
    value = _decode_configured_task_program_integration(payload)
    presets = value.registration.settle_presets
    assert type(presets["position_check"]) is SimulationStabilityConditionCfg
    assert type(presets["pose_speed"]) is SimulationSettleMonitorCfg
    assert set(value.registration.catalog.settle_preset_ids) == set(presets)
    value.registration.catalog.validate_post_policy(
        WaitStablePostCfg(entity="cube", preset="position_check"), path=("post",)
    )
    payload["runtime_services"]["settle_presets"]["position_check"]["target_position"][
        0
    ] = 2.0
    assert presets["position_check"].target_position == (0.5, 0.0, 0.1)
    presets["position_check"].duration = 0.16
    with pytest.raises(IntegrationFingerprintMismatch):
        value.registration.assert_unchanged()


@pytest.mark.parametrize(
    "change",
    ["unknown_field", "missing_sensor", "unbound_reference", "unknown_measurement"],
)
def test_configured_stability_rejects_unsupported_combinations_before_runtime(change):
    payload = _integration_payload("repeated_pick_place")
    cfg = {"kind": "placement", "entity": "cube", "target_position": [0.0, 0.0, 0.0]}
    if change == "unknown_field":
        cfg["constraints"] = {}
    elif change == "missing_sensor":
        cfg = {"kind": "supported_region", "entity": "cube", "reference": "cube"}
    elif change == "unbound_reference":
        cfg = {
            "kind": "supported_region",
            "entity": "cube",
            "reference": "missing_link",
            "support_sensor": {"uid": "contacts"},
        }
    else:
        cfg = {"measurement": "auto_by_task_name"}
    payload.setdefault("runtime_services", {})["settle_presets"] = {"check": cfg}
    with pytest.raises((TypeError, ValueError)):
        _decode_configured_task_program_integration(payload)


def test_settle_component_ownership_is_disjoint_without_a_second_constraint_table():
    from embodichain.lab.task_program.integrations._configured_composition import (
        _merge_runtime_services,
    )

    monitor = {"measurement": "pose_delta"}
    condition = {"kind": "upright", "entity": "cube", "local_axis": [0.0, 0.0, 1.0]}
    value = _merge_runtime_services(
        {"settle_presets": {"slow": monitor}},
        {"settle_presets": {"upright": condition}},
    )
    assert set(value["settle_presets"]) == {"slow", "upright"}
    condition["local_axis"][2] = 5.0
    assert value["settle_presets"]["upright"]["local_axis"] == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="Duplicate settle-preset"):
        _merge_runtime_services(
            {"settle_presets": {"same": monitor}}, {"settle_presets": {"same": monitor}}
        )
    with pytest.raises(ValueError):
        _merge_runtime_services({}, {"constraints": {}})


def test_condition_snapshot_is_idempotent_and_plain_matching_data_has_no_sensor_object():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
        SupportSensorDeclarationCfg,
    )

    cfg = SimulationStabilityConditionCfg(
        entity="cube",
        kind="stack",
        reference="table",
        local_axis=(1.0, 1.0, 1.0),
        reference_axis=(0.0, 0.0, 1.0),
        reference_half_extents=(0.1, 0.1),
        support_sensor=SupportSensorDeclarationCfg(uid="support"),
    )
    owned = cfg.snapshot()
    assert owned.to_dict() == owned.snapshot().to_dict()
    cfg.support_sensor.uid = "changed"
    assert owned.support_sensor.uid == "support"
    plain = owned.to_condition_dict()
    assert "support_sensor" not in plain
    assert plain["kind"] == "stack"
    owned.local_axis = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="zero"):
        owned.snapshot()


def test_generic_planning_budget_decodes_without_source_recovery_factories():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.sim.atomic_actions import PlanningBudget

    config = dict(
        max_replans=0,
        max_action_retries=0,
        planning_budget={"wall_seconds": 0.25},
        retry_all_failed_plans=False,
    )
    policy = _decode_recovery_policy(config, path="recovery")
    assert policy.planning_budget == PlanningBudget(0.25)
    assert policy.retry_all_failed_plans is False
    with pytest.raises(ValueError, match="planning_recovery"):
        _decode_recovery_policy(
            {**config, "planning_recovery": {"kind": "not_installed"}}, path="recovery"
        )


def test_configured_joint_stability_owns_target_and_requires_articulation():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationJointStabilityConditionCfg,
    )
    from embodichain.lab.task_program.integrations.catalog import (
        IntegrationFingerprintMismatch,
    )
    from embodichain.lab.task_program import WaitStablePostCfg

    payload = _integration_payload("open_drawer")
    declaration = {
        "kind": "joint_position_and_stability",
        "entity": "drawer",
        "joint": "cabinet_to_drawer",
        "target": 0.1,
        "tolerance": 0.03,
    }
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "retained": declaration,
    }
    value = _decode_configured_task_program_integration(payload)
    cfg = value.registration.settle_presets["retained"]
    assert type(cfg) is SimulationJointStabilityConditionCfg
    assert (cfg.duration, cfg.timeout, cfg.position_drift) == (1.0, 5.0, 0.001)
    value.registration.catalog.validate_post_policy(
        WaitStablePostCfg(entity="drawer", preset="retained"), path=("post",)
    )
    declaration["target"] = 0.2
    assert cfg.target == 0.1
    cfg.target = 0.3
    with pytest.raises(IntegrationFingerprintMismatch):
        value.registration.assert_unchanged()
    declaration["entity"] = "drawer_handle"
    with pytest.raises(ValueError, match="explicit articulation binding"):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize("change", ["missing_tolerance", "unknown_field", "bad_window"])
def test_configured_joint_stability_rejects_incomplete_policy(change):
    payload = _integration_payload("open_drawer")
    declaration = {
        "kind": "joint_position_and_stability",
        "entity": "drawer",
        "joint": "cabinet_to_drawer",
        "target": 0.1,
        "tolerance": 0.03,
    }
    if change == "missing_tolerance":
        del declaration["tolerance"]
    elif change == "unknown_field":
        declaration["infer_target"] = True
    else:
        declaration["timeout"] = 0.5
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "retained": declaration,
    }
    with pytest.raises((TypeError, ValueError)):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"invalid_observation_policy": "ignore"},
        {"max_observation_samples": 1},
        {"max_observation_samples": True},
    ],
)
def test_joint_window_configuration_rejects_unknown_runtime_modes(changes):
    payload = _integration_payload("open_drawer")
    declaration = dict(
        kind="joint_position_and_stability",
        entity="drawer",
        joint="cabinet_to_drawer",
        target=0.1,
        tolerance=0.03,
        **changes,
    )
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "retained": declaration
    }
    with pytest.raises((ValueError, TypeError)):
        _decode_configured_task_program_integration(payload)


def test_joint_window_runtime_settings_do_not_redefine_normative_condition_data():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationJointStabilityConditionCfg,
    )

    baseline = SimulationJointStabilityConditionCfg(
        entity="drawer", joint="cabinet_to_drawer", target=0.1, tolerance=0.03
    )
    changed = baseline.snapshot()
    changed.invalid_observation_policy = "isolate_rows"
    changed.max_observation_samples = 128
    assert baseline.to_condition_dict() == changed.to_condition_dict()
    payload = _integration_payload("open_drawer")
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "retained": dict(kind="joint_position_and_stability", **baseline.to_dict())
    }
    first = _decode_configured_task_program_integration(payload).registration
    payload["runtime_services"]["settle_presets"]["retained"] = dict(
        kind="joint_position_and_stability", **changed.to_dict()
    )
    second = _decode_configured_task_program_integration(payload).registration
    assert first.fingerprint != second.fingerprint


def test_segment_motion_policy_is_closed_forwarded_and_reported_separately():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )
    from embodichain.lab.task_program.runtime.results import _motion_policy_to_metadata

    declaration = {
        "sample_count": 140,
        "interpolation_mode": "fixed_grid",
        "ik_seed_mode": "sequential",
        "acceleration_limit": None,
        "sample_count_retries": [260, 320],
        "segment_sample_count_retries": [260, 320],
        "segment_continuous_ik_recovery": True,
    }
    policy = _decode_motion_policy(declaration, path="motion")
    options = policy.to_motion_gen_options(
        start_qpos=torch.zeros(1, 3),
        control_part="arm",
        interpolation_dt=0.04,
        cartesian_keyframes=True,
    )
    assert (
        options.segment_sample_count_retries == (260, 320)
        and options.segment_continuous_ik_recovery
    )
    report = _motion_policy_to_metadata(policy)
    assert report["sample_count_retries"] == [260, 320] and report[
        "segment_sample_count_retries"
    ] == [260, 320]
    assert report["segment_continuous_ik_recovery"] is True
    for invalid in (
        {"segment_sample_count_retries": [320, 260]},
        {"segment_continuous_ik_recovery": 1},
        {"strategy": "motion_gen"},
    ):
        with pytest.raises((TypeError, ValueError)):
            _decode_motion_policy({**declaration, **invalid}, path="motion")


def test_final_velocity_rejection_representation_is_explicit_and_default_identity_stable():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
    )
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import MotionPolicy

    assert "final_velocity_rejection" not in _canonical_value(MotionPolicy())["fields"]
    selected = _decode_motion_policy(
        {
            "sample_count": 50,
            "require_final_velocity": True,
            "final_velocity_rejection": "mask_only",
        },
        path="motion",
    )
    assert (
        selected.final_velocity_rejection == "mask_only"
        and selected.require_final_velocity
    )
    assert (
        _canonical_value(selected)["fields"]["final_velocity_rejection"] == "mask_only"
    )
    with pytest.raises((TypeError, ValueError)):
        _decode_motion_policy(
            {"sample_count": 50, "final_velocity_rejection": "unknown"}, path="motion"
        )


def test_physical_component_snapshot_matches_files_without_component_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.gym.utils import _component_composition as composition
    from embodichain.utils.config_paths import resolve_config_path

    path = _config_path("hand_over")
    config = _gym_config("hand_over")
    expected = resolve_gym_components(config, base_dir=path.parent)
    binding = _deployment_from_path(path).scene_binding
    selected = deepcopy(config)
    sources = {}
    for owner in ("environment", "embodiment", "scene"):
        if owner not in selected:
            continue
        declaration = selected[owner]
        reference = declaration.get("component", declaration.get("default"))
        if isinstance(reference, dict):
            reference = reference["component"]
        component = resolve_config_path(reference, base_dir=path.parent).resolve()
        sources[str(component)] = load_config(component)
        selected[owner] = {"component": str(component)}
        if "overrides" in declaration:
            selected[owner]["overrides"] = deepcopy(declaration["overrides"])
    original = deepcopy(sources)
    physical_only = deepcopy(selected)
    physical_only.pop("task_program")
    expected_cfg = config_to_cfg(physical_only, source_path=path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot parsing reopened a component file")

    monkeypatch.setattr(composition, "load_config", forbidden)
    monkeypatch.setattr(composition, "resolve_config_path", forbidden)
    actual = resolve_gym_components(
        selected, base_dir=path.parent, component_sources=sources
    )
    assert actual == expected
    actual_cfg = config_to_cfg(
        physical_only, source_path=path, component_sources=sources
    )
    import numpy as np

    np.testing.assert_equal(actual_cfg.to_dict(), expected_cfg.to_dict())
    actual.validate_scene_bindings(binding)
    with pytest.raises(ValueError, match="not declared"):
        actual.validate_scene_bindings(
            {"rigid_objects": [{"entity_id": "missing_native_root"}]}
        )
    assert sources == original
    sources[next(iter(sources))].clear()
    assert actual == expected
    missing = dict(original)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="absent from"):
        resolve_gym_components(
            selected, base_dir=path.parent, component_sources=missing
        )
    duplicate = deepcopy(selected)
    duplicate["robot"] = {"uid": "duplicate_owner"}
    with pytest.raises(ValueError, match="owns robot"):
        resolve_gym_components(
            duplicate, base_dir=path.parent, component_sources=original
        )


@pytest.mark.parametrize("inline", [False, True])
def test_demo_skill_registered_calls_and_occurrence_sources_are_content_pinned(
    tmp_path,
    inline,
):
    import numpy as np
    from embodichain.data_pipeline.real2sim import (
        RobotDemonstration,
        DemoSkillSpec,
        DemoSkillPhase,
        DemoSkillConstraint,
        JointSpec,
        build_demo_skill,
        save_demo_skill,
    )
    from embodichain.lab.task_program.integrations.configured import (
        _decode_runtime_services,
    )
    from embodichain.lab.sim.atomic_actions import ExecuteDemoSkill, ActionOptions

    artifact = build_demo_skill(
        RobotDemonstration(
            "teleop",
            "1",
            "specialized",
            ("a",),
            ("rad",),
            np.zeros((6, 1)),
            np.arange(6) * 0.02,
            "s",
            "a" * 64,
            "b" * 64,
        ),
        DemoSkillSpec(
            asset_id="task.contour",
            instruction="Trace contour",
            phases=(DemoSkillPhase("continuous", 0, 6),),
            constraints=(DemoSkillConstraint("done", "metric", "terminal"),),
        ),
        joints=(JointSpec("a", "revolute"),),
        control_dt=0.02,
    )
    manifest = save_demo_skill(artifact, tmp_path / "skill")
    declaration = {
        "kind": "demo_skill",
        "call_id": "task.contour",
        "asset_manifest": str(manifest),
        "asset_sha256": artifact.content_sha256,
    }
    source = {
        "provider_id": "demo_skill",
        "revision": "1",
        "call_index": 1,
        "program_fingerprint": "c" * 64,
        "call_id": "task.contour",
        "asset_manifest": str(manifest),
        "asset_sha256": artifact.content_sha256,
    }
    # File integrations resolve assets from the component directory; inline
    # integrations resolve them from their owning deployment directory.
    config_path = _config_path("repeated_pick_place")
    program, task, policy = _resolve_task_program_components(
        _gym_config("repeated_pick_place")["task_program"], base_dir=config_path.parent
    )
    from embodichain.utils.utility import save_config

    save_config(tmp_path / "policy.yaml", policy)
    relative_manifest = (
        "skill/demo_skill.json" if inline else "../skill/demo_skill.json"
    )
    task["runtime_services"] = {
        "registered_semantic_lowerers": [
            {**declaration, "asset_manifest": relative_manifest}
        ],
        "nominal_motion_sources": [{**source, "asset_manifest": relative_manifest}],
    }
    original_task = deepcopy(task)
    if inline:
        selected = {"inline": task}
    else:
        components = tmp_path / "components"
        components.mkdir()
        save_config(components / "integration.yaml", task)
        selected = "components/integration.yaml"
    _, resolved_task, _ = _resolve_task_program_components(
        {
            "program": str(program),
            "integration": selected,
            "execution_policy": "policy.yaml",
        },
        base_dir=tmp_path,
    )
    assert task == original_task
    resolved = resolved_task["runtime_services"]
    assert resolved["registered_semantic_lowerers"][0]["asset_manifest"] == str(
        manifest
    )
    assert resolved["nominal_motion_sources"][0]["asset_manifest"] == str(manifest)
    services = _decode_runtime_services(resolved)
    factory = services.registered_semantic_lowerers[0]
    assert factory.call_id == "task.contour"
    assert factory.target_descriptor == ExecuteDemoSkill.descriptor()
    assert (
        type(_decode_action_options({"kind": "execute_demo_skill"}, path="options"))
        is ActionOptions
    )
    payload = _integration_payload("repeated_pick_place")
    payload.setdefault("runtime_services", {})["registered_semantic_lowerers"] = [
        declaration
    ]
    integration = _decode_configured_task_program_integration(payload)
    assert "task.contour" in integration.registration.catalog.call_catalog.descriptors
    assert services.nominal_motion_sources[0].call_index == 1
    with pytest.raises(ValueError, match="unsupported fields|unknown|requires kind"):
        _decode_runtime_services(
            {
                "registered_semantic_lowerers": [
                    {**declaration, "class_type": "arbitrary"}
                ]
            }
        )
    with pytest.raises(ValueError, match="digest|changed"):
        _decode_runtime_services(
            {"nominal_motion_sources": [{**source, "asset_sha256": "e" * 64}]}
        )
