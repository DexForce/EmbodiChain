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
from dataclasses import replace
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
    _decode_recovery_policy,
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
    assert integration_payload["runtime_services"]["grasp_pose_generators"] == {
        f"{side}_eef": {
            "kind": "antipodal_parallel_jaw",
            "model": "cobotmagic_v100_gripper",
        }
        for side in ("right", "left")
    }


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
            "model": "dh_pgi_140_80",
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
            "model": "dh_pgi_140_80",
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
        "get_parallel_jaw_collision_geometry",
        lambda *a, **kw: (geometry_values.update(kw), marker)[1],
    )
    monkeypatch.setattr(
        grasp,
        "AntipodalGraspPoseGenerator",
        lambda model, **kw: SimpleNamespace(model=model, **kw),
    )
    generator = _decode_grasp_generator(
        {
            "kind": "antipodal_parallel_jaw",
            "model": "robotiq_arg2f_140",
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

    assert geometry_values["root_from_contact_frame"].tolist() == contact


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
            {"kind": "antipodal_parallel_jaw", "model": "robotiq_arg2f_140", **value},
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
    assert actual == expected
    declaration = physical.embodiment_skill_profile
    assert declaration is not None
    services = declaration["runtime_services"]
    assert (
        services["grasp_pose_generators"]
        == frozen["skill_profile"]["runtime_services"]["grasp_pose_generators"]
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
            {"kind": "antipodal_parallel_jaw", "model": "dh_pgi_140_80", field: 0.01},
            path="generator",
        )


def test_inline_builtin_aperture_matches_registered_model_and_rejects_unknown() -> None:
    payload = {
        "kind": "antipodal_parallel_jaw",
        "model": {"model_id": "franka_panda_hand"},
    }
    factory = _decode_grasp_generator(payload, path="generator")
    assert factory.min_opening_width == 0.001
    assert factory.max_opening_width == 0.08
    payload["model"] = {"model_id": "custom"}
    with pytest.raises(
        ValueError,
        match="unknown parallel-jaw gripper model",
    ):
        _decode_grasp_generator(payload, path="generator")


@pytest.mark.parametrize(
    ("relative_path", "model_id"),
    [
        ("repeated_pick_place/task.franka.yaml", "franka_panda_hand"),
        (
            "repeated_pick_place/task.dual_ur5_dh_pgi_140_80.yaml",
            "dh_pgi_140_80",
        ),
        ("tableware/pour_water/task.cobotmagic.yaml", "cobotmagic_v100_gripper"),
    ],
)
def test_configured_asset_inventory_uses_model_owner_without_runtime_construction(
    relative_path: str, model_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.task_program.integrations import (
        DeclaredAssetFileProvider,
        SimulationTaskProgramAdapterFactory,
    )
    from embodichain.lab.task_program.integrations._configured_services import (
        _AntipodalGraspPoseGeneratorFactory,
    )
    from embodichain.toolkits.graspkit.pg_grasp import model_geometry
    from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
        AntipodalSampler,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Asset declarations cannot construct or sample runtime state."
        )

    monkeypatch.setattr(SimulationManager, "__new__", staticmethod(unexpected))
    monkeypatch.setattr(Robot, "__init__", unexpected)
    monkeypatch.setattr(AntipodalSampler, "__init__", unexpected)
    monkeypatch.setattr(
        SimulationTaskProgramAdapterFactory, "create_adapter", unexpected
    )
    monkeypatch.setattr(_AntipodalGraspPoseGeneratorFactory, "__call__", unexpected)
    monkeypatch.setattr(
        model_geometry, "get_parallel_jaw_collision_geometry", unexpected
    )
    expected = model_geometry.get_parallel_jaw_model_asset_files(model_id)

    deployment = _deployment_from_path(_CONFIG_DIRECTORY / relative_path)
    provider = deployment.integration.adapter_factory
    original_fingerprint = deployment.integration.integration_fingerprint
    assert isinstance(provider, DeclaredAssetFileProvider)
    assert expected
    assert provider.declared_asset_files() == expected
    assert provider.declared_asset_files() == expected
    assert deployment.integration.integration_fingerprint == original_fingerprint


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
    assert declaration == {"kind": "antipodal_parallel_jaw", "model": model_id}

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
    assert {
        key: value for key, value in declaration.items() if key != "root_from_tcp"
    } == {
        "kind": "antipodal_parallel_jaw",
        "model": "tianji_marvin_gripper",
        "side": side,
    }
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
    assert generator.collision_geometry.provenance["side"] == side
    assert (
        generator.collision_geometry.provenance["root_link"] == f"{side}_hand_base_link"
    )
    # Resolve the real rounded URDF tool joint and this deployment's solver
    # override independently; the model's default TCP differs from this one.
    import xml.etree.ElementTree as ET
    import numpy as np
    from scipy.spatial.transform import Rotation
    from embodichain.data import get_data_path
    from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
        get_parallel_jaw_collision_geometry,
    )

    origin = ET.parse(get_data_path("TianjiMarvin/robot_with_ee_acd.urdf")).find(
        f"./joint[@name='{side.upper()}_HAND_TOOL']/origin"
    )
    root_from_tool = np.eye(4)
    root_from_tool[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
    ).as_matrix()
    root_from_tool[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
    root_from_tcp = root_from_tool @ np.asarray(robot.solver_cfg[f"{side}_arm"].tcp)
    np.testing.assert_allclose(
        declaration["root_from_tcp"], root_from_tcp, rtol=0, atol=1e-12
    )
    geometry = generator.collision_geometry
    default = get_parallel_jaw_collision_geometry("tianji_marvin_gripper", side=side)
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
        _decode_recovery_policy,
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
        _decode_recovery_policy,
    )

    with pytest.raises(ValueError, match="velocity_targets"):
        _decode_motion_policy(
            {"sample_count": 40, "velocity_targets": mode}, path="motion"
        )


def test_motion_limit_policy_decodes() -> None:
    from embodichain.lab.task_program.integrations.configured import (
        _decode_motion_policy,
        _decode_recovery_policy,
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
        _decode_recovery_policy,
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
        _decode_recovery_policy,
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
        _decode_recovery_policy,
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
        _decode_recovery_policy,
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
        _decode_recovery_policy,
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


def test_planning_services_are_closed_owned_sealed_and_deferred(monkeypatch):
    from embodichain.lab.sim.motion import mesh_collision
    from embodichain.lab.task_program.integrations.catalog import (
        IntegrationFingerprintMismatch,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Static composition constructed a live geometry query.")

    monkeypatch.setattr(mesh_collision, "SimulationFreeMotionQuery", forbidden)
    payload = _integration_payload("repeated_pick_place")
    declaration = {
        "kind": "simulation_geometry",
        "free_motion_query": "discrete_mesh",
        "capture_root_pose": True,
        "hold_qpos_source": "observed",
    }
    payload.setdefault("runtime_services", {})["planning_services"] = declaration
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.planning_services.free_motion_query == "discrete_mesh"
    assert registration.planning_services.capture_root_pose
    declaration["capture_root_pose"] = False
    registration.assert_unchanged()
    assert registration.planning_services.capture_root_pose
    registration.planning_services.capture_root_pose = False
    with pytest.raises(IntegrationFingerprintMismatch):
        registration.assert_unchanged()


@pytest.mark.parametrize(
    "invalid",
    [
        {"free_motion_query": "auto_by_object"},
        {"revision": 2},
        {"capture_root_pose": 1},
        {"class_type": "arbitrary.Factory"},
    ],
)
def test_planning_services_reject_unknown_explicit_capabilities(invalid):
    payload = _integration_payload("repeated_pick_place")
    payload.setdefault("runtime_services", {})["planning_services"] = {
        "kind": "simulation_geometry",
        **invalid,
    }
    with pytest.raises((TypeError, ValueError)):
        _decode_configured_task_program_integration(payload)


def test_configured_occurrence_presets_preflight_and_seal_without_runtime(tmp_path):
    from embodichain.lab.task_program import load_task_program
    from embodichain.lab.task_program.integrations import task_program_fingerprint
    from embodichain.utils.utility import save_config

    path = _config_path("repeated_pick_place")
    physical = _physical_components(path, _gym_config("repeated_pick_place"))
    original_program, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    task, policy = deepcopy(task), deepcopy(policy)
    policy["presets"] = {"careful": {"motion": {"sample_count": 141}}}
    pick = deepcopy(task["profile"]["action_options"]["pick"])
    pick["lift_height"] = 0.17
    task["profile"]["preset_options"] = {"careful": {"action_options": {"pick": pick}}}
    save_config(tmp_path / "program.yaml", load_config(original_program))
    save_config(tmp_path / "integration.yaml", task)
    save_config(tmp_path / "policy.yaml", policy)
    selection = {
        "program": "program.yaml",
        "integration": "integration.yaml",
        "execution_policy": "policy.yaml",
    }

    def load():
        return load_configured_task_program_deployment(
            task_program=selection,
            skill_profile=physical.embodiment_skill_profile,
            base_dir=tmp_path,
        )

    initial = load()
    compiled = initial.integration.registration.catalog.preflight(
        load_task_program(initial.program_path, integration=initial.selection)
    )
    segments = tuple(compiled.iter_segments())
    picks = tuple(
        segment for segment in segments if segment.calls[0].call.semantic_id == "pick"
    )
    assert len(picks) >= 2
    selected = picks[0]
    binding = {
        "program_fingerprint": task_program_fingerprint(compiled),
        "invocation_id": f"{compiled.program_id}/{selected.segment_id}:0",
        "preset_id": "careful",
    }
    other_binding = {
        **binding,
        "invocation_id": f"{compiled.program_id}/{picks[1].segment_id}:0",
        "preset_id": policy["preset_id"],
    }
    task.setdefault("runtime_services", {})["invocation_policy_bindings"] = [
        binding,
        other_binding,
    ]
    save_config(tmp_path / "integration.yaml", task)
    configured = load()
    registered = configured.integration.registration
    assert len(registered.invocation_policy_bindings) == 2
    assert (
        registered.invocation_policy_bindings[0].invocation_id
        != registered.invocation_policy_bindings[1].invocation_id
    )
    assert registered.invocation_policy_bindings[1].preset_id == policy["preset_id"]
    assert (
        registered.invocation_policy_bindings[0].invocation_id
        == binding["invocation_id"]
    )
    assert (
        configured.integration.integration_fingerprint
        != initial.integration.integration_fingerprint
    )
    checked = registered.catalog.preflight(
        load_task_program(configured.program_path, integration=configured.selection)
    )
    assert task_program_fingerprint(checked) == task_program_fingerprint(compiled)
    preset = next(
        item
        for item in registered.robot_profile_binding.presets
        if item.preset_id == "careful"
    )
    assert preset.motion_policy.sample_count == 141
    assert preset.action_option_template("pick").lift_height == 0.17
    for key, value in (
        ("program_fingerprint", "0" * 64),
        ("invocation_id", "foreign/occurrence:0"),
        ("preset_id", "missing"),
    ):
        broken = deepcopy(task)
        broken["runtime_services"]["invocation_policy_bindings"][0][key] = value
        save_config(tmp_path / "integration.yaml", broken)
        with pytest.raises(ValueError):
            load()
    broken = deepcopy(task)
    broken["runtime_services"]["invocation_policy_bindings"].append(deepcopy(binding))
    save_config(tmp_path / "integration.yaml", broken)
    with pytest.raises(ValueError, match="Duplicate"):
        load()


def test_native_handover_service_is_explicit_closed_and_preflighted():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        CollisionSceneRosterCfg,
        SimulationPlanningServicesCfg,
    )

    ordinary = SimulationPlanningServicesCfg()
    assert "native_handover_scene" not in _canonical_value(ordinary)["fields"]
    for values in ({"native_handover_scene": True}, {"native_handover_scene": 1}):
        with pytest.raises((ValueError, TypeError)):
            SimulationPlanningServicesCfg(**values)
    services = SimulationPlanningServicesCfg(
        native_handover_scene=True,
        capture_root_pose=True,
        capture_native_revision=True,
        collision_roster=CollisionSceneRosterCfg(
            rigid_objects=(("object", "native-object"),)
        ),
    )
    assert _canonical_value(services)["fields"]["native_handover_scene"] is True
    services.validate_num_envs(1)
    with pytest.raises(ValueError, match="B=1"):
        services.validate_num_envs(2)
    ordinary.validate_num_envs(2)
    ordinary.native_handover_scene = None
    assert _canonical_value(ordinary)["fields"]["native_handover_scene"] is None

    payload = _integration_payload("hand_over")
    preset = payload["robot_profile"]["presets"][0]
    preset["action_options"]["hand_over"] = {
        "kind": "hand_over",
        "adaptive_receive": True,
        "require_receiver_scene": True,
        "release_at_target": False,
        "existing_hold_hand_source": "grasp_command",
        "arm_selection": "bound",
    }
    preset["recovery"]["planning_recovery"] = {
        "kind": "handover_bounded",
        "revision": 1,
    }
    with pytest.raises(ValueError, match="native_handover_scene"):
        _decode_configured_task_program_integration(payload)

    payload.setdefault("runtime_services", {})["planning_services"] = {
        "kind": "simulation_geometry",
        "native_handover_scene": True,
        "capture_root_pose": True,
        "capture_native_revision": True,
        "collision_roster": {
            "rigid_objects": [
                [item["entity_id"], item.get("simulation_uid", item["entity_id"])]
                for item in payload["scene"]["rigid_objects"]
            ],
            "articulation_links": [],
        },
    }
    result = _decode_configured_task_program_integration(payload)
    assert result.registration.planning_services.native_handover_scene is True
    result.registration.validate_num_envs(1)
    result.registration.validate_physics("default")
    with pytest.raises(ValueError, match="B=1"):
        result.registration.validate_num_envs(2)
    with pytest.raises(ValueError, match="Default physics"):
        result.registration.validate_physics("newton")
    payload["runtime_services"]["planning_services"]["collision_roster"][
        "rigid_objects"
    ] = []
    with pytest.raises(ValueError, match="cover declared rigid owners"):
        _decode_configured_task_program_integration(payload)


def test_configured_geometry_physical_scope_fails_before_environment_registration(
    tmp_path: Path, registered_test_ids: list[str]
) -> None:
    """The runnable config rejects an unsupported batch without Sim."""
    from embodichain.utils.utility import save_config

    original_path = _config_path("repeated_pick_place")
    physical = _physical_components(original_path, _gym_config("repeated_pick_place"))
    program_path, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=original_path.parent
    )
    task = deepcopy(task)
    task.setdefault("runtime_services", {})["planning_services"] = {
        "kind": "simulation_geometry",
        "free_motion_query": "discrete_mesh",
        "capture_native_revision": True,
    }
    save_config(tmp_path / "integration.yaml", task)
    save_config(tmp_path / "policy.yaml", policy)
    config = _gym_config("repeated_pick_place")
    config["id"] = _TEST_ENV_ID
    config["task_program"] = {
        "program": str(program_path),
        "integration": str(tmp_path / "integration.yaml"),
        "execution_policy": str(tmp_path / "policy.yaml"),
    }
    config["num_envs"] = 2
    registered_test_ids.append(_TEST_ENV_ID)
    with pytest.raises(ValueError, match="B=1"):
        config_to_cfg(config, source_path=original_path)
    assert _TEST_ENV_ID not in REGISTERED_ENVS


def test_idle_declaration_keeps_paired_qualification_and_closed_route_contract():
    from dataclasses import replace
    from embodichain.lab.task_program.integrations.simulation.idle_clearance import (
        IdleClearanceFactory,
    )
    from embodichain.lab.task_program.language.decoder import decode_task_program

    payload = _integration_payload("repeated_pick_place")
    resource = payload["robot_profile"]["resources"][0]["resource_id"]
    route = {
        "route_id": "clearance",
        "clear_segment": "clear",
        "restore_segment": "restore",
        "resource": resource,
        "joint_name": "joint_from_model",
        "offset_rad": 0.2,
    }
    lowerers = payload["runtime_services"].setdefault(
        "registered_semantic_lowerers", []
    )
    lowerers.extend(
        [{"kind": "idle_arm_clearance", "routes": [route]}, {"kind": "park"}]
    )
    profile = payload["robot_profile"]
    profile["defaults"]["move_joints"] = {"primary": resource}
    profile["resources"][0]["endpoints"][0]["capabilities"].append(
        "motion.joint_position"
    )
    options = profile["presets"][0]["action_options"]
    for call in ("gen_sim.clear_idle_arm", "simulation.park"):
        options[call] = {"kind": "move_joints", "require_free_motion": True}
    with pytest.raises(ValueError, match="discrete_mesh"):
        _decode_configured_task_program_integration(payload)
    payload["runtime_services"]["planning_services"] = {
        "kind": "simulation_geometry",
        "free_motion_query": "discrete_mesh",
    }
    registration = _decode_configured_task_program_integration(payload).registration
    factory = next(
        f
        for f in registration.registered_semantic_lowerer_factories
        if type(f) is IdleClearanceFactory
    )
    assert factory.routes[0].joint_name == "joint_from_model"
    program = decode_task_program(
        {
            "program_id": "idle",
            "targets": {},
            "integration": {
                "scene_registry": registration.scene_binding.registry_id,
                "robot_profile": registration.robot_profile_binding.profile_id,
                "runtime_preset": profile["presets"][0]["preset_id"],
            },
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": name,
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "registered",
                                "call_id": call,
                                "arguments": args,
                                "resources": {"primary": resource},
                            },
                        },
                    }
                    for name, call, args in (
                        ("clear", "gen_sim.clear_idle_arm", {"route": "clearance"}),
                        ("restore", "simulation.park", {}),
                    )
                ],
            },
        }
    )
    registration.catalog.preflight(program)
    broken = deepcopy(payload)
    broken["robot_profile"]["presets"][0]["action_options"]["simulation.park"][
        "require_free_motion"
    ] = False
    with pytest.raises(ValueError, match="clear/restore"):
        _decode_configured_task_program_integration(
            broken
        ).registration.catalog.preflight(program)
    bad = deepcopy(route)
    bad["search"] = True
    with pytest.raises(ValueError, match="protected"):
        _decode_registered_lowerer(
            {"kind": "idle_arm_clearance", "routes": [bad]}, path="services"
        )


def test_configured_alignment_options_preflight_exact_policy_and_batch_scope() -> None:
    """Publishing typed flags also publishes their inert capability checks."""
    from embodichain.lab.task_program import TaskProgramIntegrationCfg

    payload = _integration_payload("repeated_pick_place")
    profile = payload["robot_profile"]
    resource = profile["resources"][0]["resource_id"]
    profile["defaults"]["move_held_object"] = {"primary": resource}
    payload["runtime_services"].setdefault("registered_semantic_lowerers", []).append(
        dict(
            kind="align_held",
            routes=[
                dict(
                    object_id="cube",
                    target_id="current_object_pose",
                    preserve_yaw=True,
                    axis_direction_free=False,
                    local_axis=[0.0, 0.0, 1.0],
                )
            ],
        )
    )
    preset = profile["presets"][0]
    preset["action_options"]["gen_sim.align_held"] = dict(
        kind="move_held_object",
        linear_segments=True,
        lift_clearance=None,
        hand_hold="observed",
        report_retained_attachment=True,
    )
    preset["motion"].update(
        strategy="ik_interp",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        dynamic_collision_mode="off",
        velocity_limit=None,
        acceleration_limit=None,
    )
    registration = _decode_configured_task_program_integration(payload).registration
    selection = TaskProgramIntegrationCfg(
        scene_registry=registration.scene_binding.registry_id,
        robot_profile=registration.robot_profile_binding.profile_id,
        runtime_preset=preset["preset_id"],
    )
    declaration = dict(
        program_id="alignment",
        targets={},
        program=dict(
            kind="invoke",
            call=dict(
                kind="registered",
                call_id="gen_sim.align_held",
                arguments=dict(
                    object="cube", target="current_object_pose", preserve_yaw=True
                ),
                resources={"primary": resource},
            ),
        ),
    )
    # Use the public bound program parser for the same catalog checks as installation.
    from embodichain.lab.task_program import decode_task_program

    declared = {
        **declaration,
        "integration": dict(
            scene_registry=selection.scene_registry,
            robot_profile=selection.robot_profile,
            runtime_preset=selection.runtime_preset,
        ),
    }
    compiled = registration.catalog.preflight(decode_task_program(declared))
    registration.validate_num_envs(1)
    with pytest.raises(ValueError, match="B=1"):
        registration.validate_num_envs(2)
    options = registration.robot_profile_binding.presets[0].action_option_template(
        "gen_sim.align_held"
    )
    assert (
        options.linear_segments
        and options.hand_hold == "observed"
        and options.report_retained_attachment
    )
    assert next(compiled.iter_segments()).calls[0].call.call_id == "gen_sim.align_held"
    broken = deepcopy(payload)
    broken["robot_profile"]["presets"][0]["motion"]["dynamic_collision_mode"] = "auto"
    with pytest.raises(ValueError, match="fixed-grid sequential"):
        _decode_configured_task_program_integration(
            broken
        ).registration.catalog.preflight(decode_task_program(declared))
    # Bounded staging and compiler-bound release sessions remain closed until their providers land.
    with pytest.raises(ValueError, match="planning_recovery.kind"):
        _decode_recovery_policy(
            dict(
                max_replans=0,
                max_action_retries=0,
                planning_recovery=dict(kind="uninstalled_staging", revision=1),
            ),
            path="recovery",
        )


@pytest.mark.parametrize("kind", ["press_control", "twist_control"])
def test_control_conditions_require_configured_operation_declarations_before_runtime(
    kind,
):
    payload = _integration_payload("open_drawer")
    payload.setdefault("runtime_services", {})["settle_presets"] = {
        "control": {
            "kind": kind,
            "entity": "drawer",
            "operation_id": "operation",
            "phase": "ready",
        }
    }
    with pytest.raises(ValueError, match="control-operation declarations"):
        _decode_configured_task_program_integration(payload)


def _press_service_payload():
    from dataclasses import asdict
    from embodichain.lab.sim.scene.press_inputs import PressBinding

    payload = _integration_payload("open_drawer")
    payload["scene"]["articulations"].append(
        {"entity_id": "button", "simulation_uid": "button"}
    )
    payload["scene"]["links"].append(
        {
            "entity_id": "button_cap",
            "articulation_id": "button",
            "native_link_name": "cap",
        }
    )
    binding = asdict(
        PressBinding(
            "button",
            "joint",
            "cap",
            "base",
            "",
            "a" * 64,
            "/mesh",
            1.0,
            (0.0, 0.0, -1.0),
            (0.0, 0.01),
            (0.0, 0.0, 0.0),
            0.0,
            0.01,
        )
    )
    binding.pop("source_path")
    services = payload.setdefault("runtime_services", {})
    services["hand_geometries"] = {
        "eef": {
            "kind": "sampled_parallel_jaw",
            "motion_part": "arm",
            "hand_part": "eef",
            "command_preset": "parallel_jaw_commands",
            "hand_root_link": "hand",
            "pad_link_names": ["finger_a", "finger_b"],
            "sample_count": 65,
            "body_max_edge": 0.01,
            "pad_max_edge": 0.004,
            "max_subdivisions": 10,
        }
    }
    services["control_operations"] = [
        {
            "kind": "press",
            "operation_id": "press0",
            "entity_id": "button_cap",
            "binding": binding,
            "sensor_uid": "contact",
            "grasp_target_id": "eef",
            "finger_link_names": ["finger_a", "finger_b"],
            "arm_label": "right",
            "moving_mass": 0.1,
        }
    ]
    services["registered_semantic_lowerers"].extend(
        [
            {
                "kind": "press_prepare",
                "operation_id": "press0",
                "operation_suffix": "press0",
            },
            {"kind": "press", "operation_id": "press0", "operation_suffix": "press0"},
        ]
    )
    services["settle_presets"] = {
        phase: {
            "kind": "press_control",
            "entity": "button_cap",
            "operation_id": "press0",
            "phase": phase,
        }
        for phase in ("ready", "released")
    }
    return payload


def test_configured_control_services_are_typed_closed_and_fingerprinted():
    payload = _press_service_payload()
    integration = _decode_configured_task_program_integration(payload)
    registration = integration.registration
    declaration = registration.control_operations[0]
    assert declaration.call.operation_id == "press0"
    assert declaration.source_route_payload()["binding"]["source_sha256"] == "a" * 64
    assert "control_operations" in registration.fingerprint_payload()
    assert "gen_sim.press.press0" in registration.call_catalog.descriptors
    changed = deepcopy(payload)
    changed["runtime_services"]["control_operations"][0]["moving_mass"] = 0.2
    assert (
        _decode_configured_task_program_integration(changed).registration.fingerprint
        != registration.fingerprint
    )
    assert (
        "control_operations"
        not in _decode_configured_task_program_integration(
            _integration_payload("open_drawer")
        ).registration.fingerprint_payload()
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_operation",
        "wrong_entity",
        "wrong_family",
        "unknown_field",
        "missing_phase",
    ],
)
def test_configured_controls_reject_incomplete_or_crosswired_services(change):
    payload = _press_service_payload()
    services = payload["runtime_services"]
    if change == "missing_operation":
        services["control_operations"] = []
    elif change == "wrong_entity":
        services["settle_presets"]["ready"]["entity"] = "drawer"
    elif change == "wrong_family":
        services["registered_semantic_lowerers"][-1]["kind"] = "twist"
    elif change == "unknown_field":
        services["control_operations"][0]["planner_override"] = {}
    else:
        del services["settle_presets"]["released"]
    with pytest.raises((TypeError, ValueError)):
        _decode_configured_task_program_integration(payload)


def test_control_preflight_requires_exact_occurrences_and_policies():
    from embodichain.lab.task_program.language.decoder import decode_task_program

    payload = _press_service_payload()
    from embodichain.lab.sim.atomic_actions.requirements import (
        JOINT_POSITION_CAPABILITY,
    )

    payload["robot_profile"]["resources"][0]["endpoints"][0]["capabilities"].append(
        JOINT_POSITION_CAPABILITY
    )
    payload["robot_profile"]["defaults"].update(
        move_joints={"primary": "primary_manipulator"},
        press={"primary": "primary_manipulator"},
    )
    payload["robot_profile"]["presets"][0]["action_options"].update(
        {
            "gen_sim.press_prepare.press0": {"kind": "move_joints"},
            "gen_sim.press.press0": {"kind": "press", "press_distance": 0.02},
        }
    )
    registration = _decode_configured_task_program_integration(payload).registration
    program = {
        "program_id": "press_program",
        "targets": {},
        "integration": {
            "robot_profile": registration.robot_profile_binding.profile_id,
            "scene_registry": registration.scene_binding.registry_id,
            "runtime_preset": "trajectory",
        },
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": phase,
                    "steps": {
                        "kind": "invoke",
                        "call": {
                            "kind": "registered",
                            "call_id": call_id,
                            "arguments": {"object": "button"},
                        },
                    },
                    "post": [
                        {"kind": "wait_stable", "entity": "button_cap", "preset": phase}
                    ],
                }
                for phase, call_id in (
                    ("ready", "gen_sim.press_prepare.press0"),
                    ("released", "gen_sim.press.press0"),
                )
            ],
        },
    }
    registration.catalog.preflight(decode_task_program(program))
    missing = deepcopy(program)
    missing["program"]["items"].pop()
    with pytest.raises(ValueError, match="operation budget"):
        registration.catalog.preflight(decode_task_program(missing))
    wrong_policy = deepcopy(program)
    wrong_policy["program"]["items"][0]["post"][0]["preset"] = "released"
    with pytest.raises(ValueError, match="matching physical post-policy"):
        registration.catalog.preflight(decode_task_program(wrong_policy))


def test_generated_pick_and_place_declarations_remain_closed_and_typed():
    from embodichain.lab.sim.atomic_actions import PickPlanningRecovery
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )

    pick = _decode_action_options(
        {
            "kind": "pick_up",
            "adaptive_grasp": "ordinary",
            "full_opening_clearance": True,
            "require_qualified_approach": True,
        },
        path="options",
    )
    place = _decode_action_options(
        {
            "kind": "place",
            "linear_segments": True,
            "qualify_required_waypoints": True,
            "monitor_scene_until_approach": True,
        },
        path="options",
    )
    assert (
        pick.adaptive_grasp == "ordinary"
        and pick.full_opening_clearance
        and pick.require_qualified_approach
    )
    assert (
        place.linear_segments
        and place.qualify_required_waypoints
        and place.monitor_scene_until_approach
    )
    rule = _decode_recovery_policy(
        {
            "max_replans": 0,
            "max_action_retries": 0,
            "planning_recovery": {"kind": "pick_bounded", "revision": 1},
        },
        path="recovery",
    ).planning_recovery
    assert type(rule) is PickPlanningRecovery and rule.max_candidate_paths == 4
    for kind, field in (
        ("place", "adaptive_grasp"),
        ("pick_up", "qualify_required_waypoints"),
    ):
        with pytest.raises(ValueError, match="unsupported fields"):
            _decode_action_options({"kind": kind, field: True}, path="options")
    with pytest.raises((TypeError, ValueError), match="bool"):
        _decode_action_options({"kind": "place", "linear_segments": 1}, path="options")


def test_place_recovery_declaration_is_closed_and_rejects_other_skill_variants():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.sim.atomic_actions import PlacePlanningRecovery

    base = {"max_replans": 0, "max_action_retries": 0}
    rule = _decode_recovery_policy(
        {
            **base,
            "planning_recovery": {
                "kind": "place_bounded",
                "revision": 1,
                "variant": "waypoint",
            },
        },
        path="recovery",
    ).planning_recovery
    assert type(rule) is PlacePlanningRecovery and rule.max_complete_plans == 10
    assert rule.inside_world_yaw is False
    active = _decode_recovery_policy(
        {
            **base,
            "planning_recovery": {
                "kind": "place_bounded",
                "revision": 1,
                "inside_world_yaw": True,
            },
        },
        path="recovery",
    ).planning_recovery
    assert active.inside_world_yaw is True
    for declaration in (
        {"kind": "place_bounded", "revision": 1, "variant": "support"},
        {"kind": "pick_bounded", "revision": 1, "variant": "waypoint"},
        {"kind": "place_bounded", "revision": 1, "max_complete_plans": 100},
        {"kind": "pick_bounded", "revision": 1, "inside_world_yaw": True},
        {
            "kind": "place_bounded",
            "revision": 1,
            "inside_world_yaw": True,
            "variant": "lateral",
        },
        {"kind": "place_bounded", "revision": 1, "inside_world_yaw": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            _decode_recovery_policy(
                {**base, "planning_recovery": declaration}, path="recovery"
            )


def test_place_scene_permissions_require_program_bound_occurrences():
    payload = _integration_payload("repeated_pick_place")
    payload.setdefault("runtime_services", {})["planning_services"] = {
        "kind": "simulation_geometry",
        "collision_roster": {
            "rigid_objects": [["cube", "cube_native"]],
            "articulation_links": [],
        },
        "place_scene_bindings": [
            {"invocation_id": "unbound/segment:0", "floor_entity_id": "cube"}
        ],
    }
    with pytest.raises(ValueError, match="program-bound"):
        _decode_configured_task_program_integration(payload)


def test_planning_budget_declaration_is_closed_fingerprinted_and_reported():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.task_program.runtime.results import (
        _recovery_policy_to_metadata,
    )

    base = {"max_replans": 0, "max_action_retries": 0}
    default = _decode_recovery_policy(base, path="recovery")
    disabled = _decode_recovery_policy(
        {**base, "planning_budget": None}, path="recovery"
    )
    measured = _decode_recovery_policy({**base, "planning_budget": {}}, path="recovery")
    bounded = _decode_recovery_policy(
        {**base, "planning_budget": {"wall_seconds": 90}}, path="recovery"
    )
    assert default.planning_budget is None
    assert _canonical_value(default) == _canonical_value(disabled)
    assert "planning_budget" not in _canonical_value(default)["fields"]
    assert "planning_budget" not in _recovery_policy_to_metadata(default)
    assert measured.planning_budget.wall_seconds is None
    assert bounded.planning_budget.wall_seconds == 90.0
    assert _canonical_value(default) != _canonical_value(measured)
    assert _canonical_value(measured) != _canonical_value(bounded)
    assert _recovery_policy_to_metadata(bounded)["planning_budget"] == {
        "wall_seconds": 90.0
    }
    for declaration in (
        {"wall_seconds": 0},
        {"wall_seconds": -1},
        {"wall_seconds": True},
        {"wall_seconds": float("inf")},
        {"wall_seconds": 90, "retries": 3},
    ):
        with pytest.raises((ValueError, TypeError)):
            _decode_recovery_policy(
                {**base, "planning_budget": declaration}, path="recovery"
            )


def test_explicit_planning_recovery_receipt_retains_declared_variant():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.task_program.runtime.results import (
        _recovery_policy_to_metadata,
    )

    policy = _decode_recovery_policy(
        {
            "max_replans": 0,
            "max_action_retries": 0,
            "planning_recovery": {
                "kind": "place_bounded",
                "revision": 1,
                "variant": "lateral",
            },
        },
        path="recovery",
    )
    declaration = _recovery_policy_to_metadata(policy)["planning_recovery"]
    assert declaration["variant"] == "lateral"
    assert declaration["revision"] == 1


def test_public_inline_deployment_preflights_without_component_io(
    monkeypatch: pytest.MonkeyPatch, registered_test_ids: list[str]
) -> None:
    from embodichain.lab.task_program.integrations import (
        _configured_composition as composition,
    )

    path = _config_path("hand_over")
    physical = _physical_components(path, _gym_config("hand_over"))
    selectors = physical.config["task_program"]
    file_deployment = load_configured_task_program_deployment(
        task_program=selectors,
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    program_path, integration, policy = composition._resolve_task_program_components(
        selectors, base_dir=path.parent
    )
    inline = {
        "program": {"inline": load_config(program_path)},
        "integration": {"inline": integration},
        "execution_policy": {"inline": policy},
    }
    original = deepcopy(inline)

    def no_component_io(*args, **kwargs):
        raise AssertionError("Inline preflight attempted component filesystem access")

    monkeypatch.setattr(composition, "_component_path", no_component_io)
    monkeypatch.setattr(composition, "_load_yaml_component", no_component_io)
    actual = load_configured_task_program_deployment(
        task_program=inline,
        skill_profile=physical.embodiment_skill_profile,
        base_dir="/nonexistent/preflight-only",
    )
    assert actual.program_path is None
    assert actual.load_program() == file_deployment.load_program()
    assert actual.program_source == original["program"]["inline"]
    assert (
        actual.integration.integration_fingerprint
        == file_deployment.integration.integration_fingerprint
    )
    assert inline == original
    inline["program"]["inline"]["program_id"] = "changed_source"
    assert actual.program_source == original["program"]["inline"]
    exported = actual.program_source
    exported["program_id"] = "changed_copy"
    assert actual.program_source == original["program"]["inline"]
    config = _gym_config("hand_over")
    config["id"] = _TEST_ENV_ID
    registered_test_ids.append(_TEST_ENV_ID)
    config["task_program"] = deepcopy(original)
    cfg = config_to_cfg(config, source_path=path)
    assert cfg.task_program == actual.load_program()

    invalid = deepcopy(original)
    invalid["program"]["inline"]["program_id"] = "foreign_program"
    with pytest.raises(ValueError, match="program_id"):
        load_configured_task_program_deployment(
            task_program=invalid,
            skill_profile=physical.embodiment_skill_profile,
            base_dir="/nonexistent/preflight-only",
        )
    ambiguous = deepcopy(original)
    ambiguous["execution_policy"]["component"] = "policy.yaml"
    with pytest.raises(ValueError, match="unsupported fields"):
        load_configured_task_program_deployment(
            task_program=ambiguous,
            skill_profile=physical.embodiment_skill_profile,
            base_dir="/nonexistent/preflight-only",
        )


def test_handover_adaptive_options_and_bounded_recovery_decode_before_runtime():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_policy_preset,
    )
    from embodichain.lab.sim.atomic_actions import HandOverPlanningRecovery

    preset = deepcopy(_integration_payload("hand_over")["robot_profile"]["presets"][0])
    declaration = {
        "kind": "hand_over",
        "adaptive_receive": True,
        "receive_region": "supported_end",
        "existing_hold_hand_source": "grasp_command",
        "source_retreat_mode": "root_separation",
        "release_at_target": False,
        "arm_selection": "bound",
    }
    preset["action_options"] = {"hand_over": declaration}
    preset["recovery"]["planning_recovery"] = {
        "kind": "handover_bounded",
        "revision": 1,
    }
    decoded = _decode_policy_preset(preset, path="preset")
    assert isinstance(
        decoded.recovery_policy.planning_recovery, HandOverPlanningRecovery
    )
    assert decoded.recovery_policy.planning_recovery.max_direction_attempts == 2
    options = decoded.action_option_template("hand_over")
    assert options.adaptive_receive and options.receive_region == "supported_end"
    assert options.source_retreat_mode == "root_separation"
    assert options.existing_hold_hand_source == "grasp_command"
    preset["action_options"]["hand_over"]["require_receiver_scene"] = True
    native = _decode_policy_preset(preset, path="preset").action_option_template(
        "hand_over"
    )
    assert native.require_receiver_scene is True
    preset["recovery"].pop("planning_recovery")
    with pytest.raises(ValueError, match="declared together"):
        _decode_policy_preset(preset, path="preset")
    preset["action_options"]["hand_over"]["receiver_native_scene"] = True
    with pytest.raises(ValueError, match="unsupported fields.*receiver_native_scene"):
        _decode_policy_preset(preset, path="preset")


def test_handover_default_decoder_retains_original_options():
    from dataclasses import asdict

    assert asdict(
        _decode_action_options({"kind": "hand_over"}, path="options")
    ) == asdict(HandOverOptions())
    with pytest.raises((TypeError, ValueError), match="adaptive_receive"):
        _decode_action_options(
            {"kind": "hand_over", "adaptive_receive": "yes"}, path="options"
        )


def test_twist_options_and_recovery_decode_through_closed_shared_contracts():
    from embodichain.lab.sim.atomic_actions import (
        Twist,
        TwistOptions,
        TwistPlanningRecovery,
    )
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
        _decode_recovery_policy,
    )

    options = _decode_action_options(
        {
            "kind": "twist",
            "hand_interp_steps": 12,
            "contact_bite_depth": 0.005,
            "contact_roll": None,
            "contact_settle_steps": 5,
            "max_twist_angle": 0.35,
            "rotation_enabled": False,
        },
        path="options",
    )
    policy = _decode_recovery_policy(
        {
            "max_replans": 0,
            "max_action_retries": 0,
            "planning_recovery": {"kind": "twist_bounded", "revision": 1},
        },
        path="recovery",
    )
    assert not options.rotation_enabled
    assert type(options) is TwistOptions
    assert options.contact_settle_steps == 5 and options.max_twist_angle == 0.35
    assert type(policy.planning_recovery) is TwistPlanningRecovery
    Twist.descriptor().validate_planning_recovery(policy)
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_action_options(
            {"kind": "twist", "operation_id": "hidden"}, path="options"
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_recovery_policy(
            {
                "max_replans": 0,
                "max_action_retries": 0,
                "planning_recovery": {
                    "kind": "twist_bounded",
                    "revision": 1,
                    "max_candidates": 100,
                },
            },
            path="recovery",
        )

    with pytest.raises((TypeError, ValueError), match="bool"):
        _decode_action_options({"kind": "twist", "rotation_enabled": 0}, path="options")


@pytest.mark.parametrize(
    "option", ["continue_verified_holds", "prefer_cargo_clearance"]
)
def test_coordinated_verified_hold_option_decodes_without_changing_shared_default(
    option,
):
    assert (
        getattr(
            _decode_action_options({"kind": "coordinated_pickment"}, path="options"),
            option,
        )
        is False
    )
    enabled = _decode_action_options(
        {"kind": "coordinated_pickment", option: True},
        path="options",
    )
    assert getattr(enabled, option) is True
    with pytest.raises((TypeError, ValueError), match=option):
        _decode_action_options(
            {"kind": "coordinated_pickment", option: 1},
            path="options",
        )


def test_held_alignment_recovery_decodes_as_exact_closed_policy():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.sim.atomic_actions import MoveHeldObjectPlanningRecovery

    base = {"max_replans": 0, "max_action_retries": 0}
    declaration = {"kind": "held_alignment_bounded", "revision": 1}
    decoded = _decode_recovery_policy(
        {**base, "planning_recovery": declaration}, path="recovery"
    )
    assert type(decoded.planning_recovery) is MoveHeldObjectPlanningRecovery
    assert decoded.planning_recovery.staging is False
    staged = _decode_recovery_policy(
        {**base, "planning_recovery": {**declaration, "staging": True}}, path="recovery"
    )
    assert staged.planning_recovery.staging is True
    from embodichain.lab.task_program.integrations.catalog import _canonical_value

    assert (
        _canonical_value(decoded.planning_recovery)["fields"].get(
            "place_continuation", False
        )
        is False
    )
    prospective = _decode_recovery_policy(
        {
            **base,
            "planning_recovery": {
                **declaration,
                "staging": True,
                "place_continuation": True,
            },
        },
        path="recovery",
    )
    assert prospective.planning_recovery.place_continuation
    with pytest.raises(ValueError, match="requires declared staging"):
        _decode_recovery_policy(
            {**base, "planning_recovery": {**declaration, "place_continuation": True}},
            path="recovery",
        )

    with pytest.raises(TypeError, match="staging"):
        _decode_recovery_policy(
            {**base, "planning_recovery": {**declaration, "staging": 1}},
            path="recovery",
        )
    with pytest.raises(ValueError, match="does not support"):
        _decode_recovery_policy(
            {**base, "planning_recovery": {**declaration, "variant": "scene"}},
            path="recovery",
        )


def test_inside_landing_services_decode_preflight_and_preserve_default_identity():
    from embodichain.lab.task_program.integrations import task_program_fingerprint
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.place_scene import (
        PlaceSceneBinding,
    )
    from embodichain.lab.task_program.language.decoder import decode_task_program

    fields = _canonical_value(PlaceSceneBinding("p/s", "floor"))["fields"]
    assert fields == {
        "invocation_id": "p/s",
        "floor_entity_id": "floor",
        "support_owner_id": None,
    }
    payload = _integration_payload("repeated_pick_place")
    payload["scene"]["rigid_objects"].append(
        {"entity_id": "container_body", "simulation_uid": "native_container"}
    )
    payload["scene"]["rigid_objects"][-1]["affordances"] = [
        {
            "kind": "container",
            "entity_id": "inside_frame",
            "native_name": "inside",
            "release_clearance": 0.12,
        }
    ]
    initial = _decode_configured_task_program_integration(payload).registration
    preset = payload["robot_profile"]["presets"][0]
    program = decode_task_program(
        {
            "program_id": "inside_service",
            "targets": {},
            "integration": {
                "robot_profile": initial.robot_profile_binding.profile_id,
                "scene_registry": initial.scene_binding.registry_id,
                "runtime_preset": preset["preset_id"],
            },
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": "place",
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "place",
                                "object": "cube",
                                "inside": "inside_frame",
                            },
                        },
                    }
                ],
            },
        }
    )
    compiled = initial.catalog.preflight(program)
    segment = next(iter(compiled.iter_segments()))
    occurrence = f"{compiled.program_id}/{segment.segment_id}:0"
    preset["recovery"]["planning_recovery"] = {
        "kind": "place_bounded",
        "revision": 1,
        "inside_world_yaw": True,
    }
    preset["action_options"]["place"]["qualify_required_waypoints"] = True
    services = payload.setdefault("runtime_services", {})
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": task_program_fingerprint(compiled),
            "invocation_id": occurrence,
            "preset_id": preset["preset_id"],
        }
    ]
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "collision_roster": {
            "rigid_objects": [
                [item["entity_id"], item.get("simulation_uid", item["entity_id"])]
                for item in payload["scene"]["rigid_objects"]
            ],
            "articulation_links": [],
        },
        "place_scene_bindings": [
            {
                "invocation_id": occurrence,
                "floor_entity_id": "container_body",
                "support_owner_id": "container_body",
                "inside_subject_id": "cube",
                "inside_container_id": "inside_frame",
            }
        ],
    }
    registered = _decode_configured_task_program_integration(payload).registration
    assert task_program_fingerprint(
        registered.catalog.preflight(program)
    ) == task_program_fingerprint(compiled)
    assert registered.fingerprint != initial.fingerprint
    with pytest.raises(ValueError, match="Default physics"):
        registered.planning_services.validate_physics("newton")
    missing = deepcopy(payload)
    binding = missing["runtime_services"]["planning_services"]["place_scene_bindings"][
        0
    ]
    del binding["inside_subject_id"]
    del binding["inside_container_id"]
    decoded = _decode_configured_task_program_integration(missing).registration
    with pytest.raises(ValueError, match="exact occurrence-bound"):
        decoded.catalog.preflight(program)
    crossed = deepcopy(payload)
    crossed["scene"]["rigid_objects"][-1]["affordances"].append(
        {
            "kind": "container",
            "entity_id": "other_inside",
            "native_name": "other_inside",
            "release_clearance": 0.12,
        }
    )
    crossed["runtime_services"]["planning_services"]["place_scene_bindings"][0][
        "inside_container_id"
    ] = "other_inside"
    crossed_registration = _decode_configured_task_program_integration(
        crossed
    ).registration
    with pytest.raises(ValueError, match="match the selected inside Place"):
        crossed_registration.catalog.preflight(program)
    wrong = deepcopy(payload)
    wrong["scene"]["rigid_objects"][-1]["affordances"][0]["release_clearance"] = 0.1
    with pytest.raises(ValueError, match="0.12"):
        _decode_configured_task_program_integration(wrong)


def test_cargo_service_is_explicit_and_required_by_configured_preference():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    assert (
        _canonical_value(SimulationPlanningServicesCfg())["fields"].get(
            "capture_coordinated_cargo", False
        )
        is False
    )
    with pytest.raises(TypeError, match="booleans"):
        SimulationPlanningServicesCfg(capture_coordinated_cargo=1)
    payload = _integration_payload("hand_over")
    resources = payload["robot_profile"]["resources"]
    payload["robot_profile"]["defaults"]["coordinated_pickment"] = {
        "left": resources[0]["resource_id"],
        "right": resources[1]["resource_id"],
    }
    payload["robot_profile"]["presets"][0]["action_options"][
        "simulation.coordinated_transport"
    ] = {
        "kind": "coordinated_pickment",
        "release": True,
        "prefer_cargo_clearance": True,
    }
    carrier = next(
        item["entity_id"]
        for item in payload["scene"]["rigid_objects"]
        if any(a["kind"] == "antipodal_grasp" for a in item.get("affordances", ()))
    )
    services = payload.setdefault("runtime_services", {})
    services.setdefault("registered_semantic_lowerers", []).append(
        {
            "kind": "coordinated_transport",
            "routes": [
                {
                    "object_id": carrier,
                    "target_id": "cargo_destination",
                    "world_displacement": [0.14, 0.0, 0.0],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="capture_coordinated_cargo"):
        _decode_configured_task_program_integration(payload)
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_coordinated_cargo": True,
    }
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.planning_services.capture_coordinated_cargo is True
    assert (
        registration.fingerprint_payload()["planning_services"][
            "capture_coordinated_cargo"
        ]
        is True
    )
    with pytest.raises(ValueError, match="B=1"):
        registration.planning_services.validate_num_envs(2)
    with pytest.raises(ValueError, match="Default physics"):
        registration.planning_services.validate_physics("newton")


def test_held_staging_declarations_are_closed_and_program_bound():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    assert (
        "held_staging_bindings"
        not in _canonical_value(SimulationPlanningServicesCfg())["fields"]
    )
    payload = _integration_payload("repeated_pick_place")
    preset = payload["robot_profile"]["presets"][0]["preset_id"]
    services = payload.setdefault("runtime_services", {})
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": "a" * 64,
            "invocation_id": "p/stage:0",
            "preset_id": preset,
        }
    ]
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_root_pose": True,
        "capture_native_revision": True,
        "held_staging_bindings": [
            {"invocation_id": "p/stage:0", "rigid_entity_id": "cube"}
        ],
    }
    registration = _decode_configured_task_program_integration(payload).registration
    assert (
        registration.planning_services.held_staging_bindings[0].rigid_entity_id
        == "cube"
    )
    with pytest.raises(ValueError, match="B=1"):
        registration.planning_services.validate_num_envs(2)
    bad = deepcopy(payload)
    bad["runtime_services"]["planning_services"]["capture_root_pose"] = False
    with pytest.raises(ValueError, match="root and native-source"):
        _decode_configured_task_program_integration(bad)
    bad = deepcopy(payload)
    bad["runtime_services"]["planning_services"]["held_staging_bindings"][0].update(
        predecessor_invocation_id="p/source:0",
        predecessor_call_id="pick",
        resource_id="arm",
    )
    with pytest.raises(ValueError, match="program-bound current/predecessor"):
        _decode_configured_task_program_integration(bad)


def test_held_transport_options_decode_exact_cartesian_and_retention_intent():
    options = _decode_action_options(
        {
            "kind": "move_held_object",
            "linear_segments": True,
            "lift_clearance": 0.06,
            "hand_hold": "observed",
            "report_retained_attachment": True,
        },
        path="options",
    )
    assert options.linear_segments is True
    assert options.lift_clearance == 0.06
    assert options.hand_hold == "observed"
    assert options.report_retained_attachment is True
    with pytest.raises(ValueError):
        _decode_action_options(
            {"kind": "place", "lift_clearance": 0.06}, path="options"
        )


def test_public_declaration_compiler_matches_full_deployment_occurrences():
    from embodichain.lab.task_program.integrations import (
        compile_configured_task_program,
        task_program_fingerprint,
    )
    from embodichain.lab.task_program import TaskProgramIntegrationCfg

    path = _config_path("repeated_pick_place")
    physical = _physical_components(path, _gym_config("repeated_pick_place"))
    program_path, task, _ = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    deployment = _deployment_from_path(path)
    declaration = load_config(program_path)
    scene = deepcopy(task["scene_binding"])
    before = deepcopy((declaration, scene))
    compiled = compile_configured_task_program(
        declaration, scene_binding=scene, selection=deployment.selection
    )
    complete = deployment.integration.registration.catalog.preflight(
        deployment.load_program()
    )
    assert task_program_fingerprint(compiled) == task_program_fingerprint(complete)
    assert [segment.segment_id for segment in compiled.iter_segments()] == [
        segment.segment_id for segment in complete.iter_segments()
    ]
    assert (declaration, scene) == before
    bad = {**scene, "runtime_callback": "not_a_declaration"}
    with pytest.raises(ValueError, match="unsupported"):
        compile_configured_task_program(
            declaration, scene_binding=bad, selection=deployment.selection
        )
    wrong = TaskProgramIntegrationCfg(
        robot_profile=deployment.selection.robot_profile,
        scene_registry="foreign",
        runtime_preset=deployment.selection.runtime_preset,
    )
    with pytest.raises(ValueError, match="scene registry"):
        compile_configured_task_program(
            declaration, scene_binding=scene, selection=wrong
        )


def test_retained_pour_factory_preserves_canonical_goal_and_phase_effect():
    from embodichain.lab.task_program.integrations.simulation.retained_pour import (
        RetainedPourFactory,
        _RetainedPourLowerer,
    )
    from embodichain.lab.task_program.semantics import SemanticEffectKind

    factory = _decode_registered_lowerer(
        {"kind": "retained_pour", "object_id": "bottle"}, path="lowerer"
    )
    assert type(factory) is RetainedPourFactory
    assert (
        factory.call_id == "simulation.pour" and factory.revision == "retained-pour-1"
    )
    call = RegisteredSemanticCall(
        call_id="simulation.pour", arguments={"object": "bottle"}
    )
    lowerer = _RetainedPourLowerer("bottle")
    ordinary = _PourLowerer("bottle").lower(
        call, context=None, bound=None, option_template=PourOptions()
    )
    actual = lowerer.lower(
        call, context=None, bound=None, option_template=PourOptions()
    )
    assert (
        type(actual.goal) is type(ordinary.goal)
        and actual.goal.target is ordinary.goal.target is None
    )
    effect = actual.registered_effect
    assert effect.effect_kind is SemanticEffectKind.ATTACH
    assert len(effect.held_objects) == 1
    held = effect.held_objects[0]
    assert (
        held.object_id,
        held.slot_id,
        held.acquire_before_segment,
        held.retain_during_segments,
    ) == ("bottle", "primary", "pour", ("pour",))
    assert (
        lowerer.pick_lookahead_targets(
            call,
            picked_object=SceneObjectRef("bottle"),
            bound=None,
            previous_target=None,
        )
        == ()
    )
    assert (
        lowerer.pick_lookahead_targets(
            call,
            picked_object=SceneObjectRef("other"),
            bound=None,
            previous_target=None,
        )
        is None
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_registered_lowerer(
            {"kind": "retained_pour", "object_id": "bottle", "target": {}},
            path="lowerer",
        )
    with pytest.raises(ValueError, match="Default/B1"):
        factory.create(
            simulation=SimpleNamespace(physics_backend="newton", num_envs=1),
            robot=None,
            scene_registry=None,
            engine=None,
        )


@pytest.mark.parametrize("release", [False, True])
def test_coordinated_hold_and_transport_keep_distinct_effects(release):
    from embodichain.lab.task_program.integrations.simulation.coordinated import (
        CoordinatedHoldLowererFactory,
    )
    from embodichain.lab.task_program.semantics import SemanticEffectKind

    factory = _decode_registered_lowerer(
        {
            "kind": "coordinated_transport" if release else "coordinated_hold",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "forward",
                    "world_displacement": [0.14, 0.0, 0.0],
                }
            ],
        },
        path="lowerer",
    )
    assert isinstance(factory, CoordinatedHoldLowererFactory) is (not release)
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, label="tray", entity_id="tray"
    )
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 0.75
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        scene=SimpleNamespace(
            entities={"tray": SimpleNamespace(pose=pose, confidence=1.0)}
        ),
    )
    lowerer = factory.lowerer_type(factory.routes, (semantics,))
    call = RegisteredSemanticCall(
        call_id=factory.call_id,
        arguments={
            "object": "tray",
            "target": "forward",
            "world_displacement": [0.14, 0.0, 0.0],
        },
    )
    result = lowerer.lower(
        call,
        context=context,
        bound=None,
        option_template=CoordinatedPickmentOptions(release=release),
    )
    torch.testing.assert_close(
        result.goal.object_target_pose[:, :3, 3], torch.tensor([[0.14, 0.0, 0.75]])
    )
    assert result.registered_effect.effect_kind is (
        SemanticEffectKind.RELEASE if release else SemanticEffectKind.ATTACH
    )
    assert tuple(item.slot_id for item in result.registered_effect.held_objects) == (
        "left",
        "right",
    )
    assert all(
        item.allow_missing_detached_baseline is release
        for item in result.registered_effect.held_objects
    )
    with pytest.raises(ValueError, match="release|retain"):
        lowerer.lower(
            call,
            context=context,
            bound=None,
            option_template=CoordinatedPickmentOptions(release=not release),
        )
    with pytest.raises(ValueError, match="differs"):
        lowerer.lower(
            replace(
                call,
                arguments={
                    "object": "tray",
                    "target": "forward",
                    "world_displacement": [0.15, 0.0, 0.0],
                },
            ),
            context=context,
            bound=None,
            option_template=CoordinatedPickmentOptions(release=release),
        )


def test_coordinated_relative_support_preserves_observed_world_orientation():
    from embodichain.lab.task_program.integrations.simulation.coordinated import (
        CoordinatedPlacementLowererFactory,
    )

    factory = _decode_registered_lowerer(
        {
            "kind": "coordinated_transport",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "landing",
                    "reference_entity_id": "support",
                    "world_displacement": [0.2, -0.1, 0.04],
                    "relation": "on",
                }
            ],
        },
        path="lowerer",
    )
    assert type(factory) is CoordinatedPlacementLowererFactory
    pose = torch.eye(4)[None]
    pose[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original = pose.clone()
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        scene=SimpleNamespace(
            entities={"tray": SimpleNamespace(pose=pose, confidence=1.0)}
        ),
    )
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, label="tray", entity_id="tray"
    )
    lowerer = factory.lowerer_type(factory.routes, (semantics,))
    call = RegisteredSemanticCall(
        call_id=factory.call_id,
        arguments={
            "object": "tray",
            "target": "landing",
            "reference": "support",
            "relation": "on",
        },
    )
    result = lowerer.lower(
        call,
        context=context,
        bound=None,
        option_template=CoordinatedPickmentOptions(release=True),
    )
    target = result.goal.object_target_pose
    assert type(target) is SceneEntityPose and target.entity_id == "support"
    torch.testing.assert_close(
        target.world_displacement, torch.tensor([0.2, -0.1, 0.04])
    )
    torch.testing.assert_close(target.world_orientation, pose[:, :3, :3])
    torch.testing.assert_close(pose, original)
    with pytest.raises(ValueError, match="support relation"):
        lowerer.lower(
            replace(call, arguments={"object": "tray", "target": "landing"}),
            context=context,
            bound=None,
            option_template=CoordinatedPickmentOptions(release=True),
        )


@pytest.mark.parametrize("kind", ["coordinated_hold", "coordinated_transport"])
def test_coordinated_zero_displacement_keeps_an_explicit_same_pose_request(kind):
    factory = _decode_registered_lowerer(
        {
            "kind": kind,
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "same",
                    "world_displacement": [0.0, 0.0, 0.0],
                }
            ],
        },
        path="lowerer",
    )
    assert factory.routes[0].world_displacement == (0.0, 0.0, 0.0)
    assert factory.call_id == "simulation." + kind


def test_drawer_transport_alias_retains_effectless_contract_and_requires_owner_service():
    from embodichain.lab.task_program.integrations.simulation.transport import (
        DrawerTransportLowererFactory,
        _DrawerTransportLowerer,
    )
    from embodichain.lab.task_program.semantics import RegisteredSemanticCall

    route = {
        "object_id": "cube",
        "target_id": "inside",
        "pose": {"kind": "scene_entity", "entity_id": "cube"},
    }
    factory = _decode_registered_lowerer(
        {"kind": "drawer_transport", "routes": [route]}, path="service"
    )
    assert type(factory) is DrawerTransportLowererFactory
    lowerer = _DrawerTransportLowerer(factory.routes)
    call = RegisteredSemanticCall(
        call_id="gen_sim.drawer_transport",
        arguments={"object": "cube", "target": "inside"},
    )
    result = lowerer.lower(
        call, context=None, bound=None, option_template=MoveHeldObjectOptions()
    )
    assert result.registered_effect is None and result.skill_options is None
    assert result.goal.object_target_pose.entity_id == "cube"
    assert result.goal.object_id is None
    assert lowerer.preserves_symbolic_state
    payload = _integration_payload("repeated_pick_place")
    payload["runtime_services"].setdefault("registered_semantic_lowerers", []).append(
        {"kind": "drawer_transport", "routes": [route]}
    )
    payload["robot_profile"]["presets"][0]["action_options"][
        "gen_sim.drawer_transport"
    ] = {"kind": "move_held_object"}
    with pytest.raises(ValueError, match="drawer-specific planning service"):
        _decode_configured_task_program_integration(payload)


def test_native_pick_services_are_closed_bound_and_preflight_subjects():
    from embodichain.lab.task_program.integrations import (
        compile_configured_task_program,
        task_program_fingerprint,
    )
    from embodichain.lab.task_program.language import TaskProgramIntegrationCfg
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    assert (
        "pick_closing_bindings"
        not in _canonical_value(SimulationPlanningServicesCfg())["fields"]
    )
    payload = _integration_payload("repeated_pick_place")
    initial = _decode_configured_task_program_integration(payload).registration
    preset_id = payload["robot_profile"]["presets"][0]["preset_id"]
    declaration = {
        "program_id": "native_pick",
        "targets": {},
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": "acquire",
                    "steps": {
                        "kind": "invoke",
                        "call": {"kind": "pick", "object": "cube"},
                    },
                }
            ],
        },
    }
    selection = TaskProgramIntegrationCfg(
        robot_profile=initial.robot_profile_binding.profile_id,
        scene_registry=initial.scene_binding.registry_id,
        runtime_preset=preset_id,
    )
    compiled = compile_configured_task_program(
        declaration, scene_binding=payload["scene"], selection=selection
    )
    occurrence = (
        f"{compiled.program_id}/{next(iter(compiled.iter_segments())).segment_id}:0"
    )
    payload["robot_profile"]["presets"][0]["action_options"]["pick"][
        "native_closing_revision"
    ] = 1
    services = payload.setdefault("runtime_services", {})
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": task_program_fingerprint(compiled),
            "invocation_id": occurrence,
            "preset_id": preset_id,
        }
    ]
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_root_pose": True,
        "capture_native_revision": True,
        "pick_closing_bindings": [{"invocation_id": occurrence, "object_id": "cube"}],
    }
    with pytest.raises(ValueError, match="65-sample hand geometry"):
        _decode_configured_task_program_integration(payload)
    resource = payload["robot_profile"]["resources"][0]
    motion = next(
        value["control_part"]
        for value in resource["endpoints"]
        if value["endpoint_id"] == "motion"
    )
    hand = next(
        value["control_part"]
        for value in resource["endpoints"]
        if value["endpoint_id"] == "grasp"
    )
    commands = next(
        value["preset_id"]
        for value in payload["robot_profile"]["command_presets"]
        if value["control_part"] == hand
    )
    services["hand_geometries"] = {
        hand: {
            "kind": "sampled_parallel_jaw",
            "motion_part": motion,
            "hand_part": hand,
            "command_preset": commands,
            "hand_root_link": "hand",
            "pad_link_names": ["pad_a", "pad_b"],
            "sample_count": 65,
            "body_max_edge": 0.01,
            "pad_max_edge": 0.004,
            "max_subdivisions": 10,
        }
    }
    registration = _decode_configured_task_program_integration(payload).registration
    program = compile_configured_task_program(
        declaration, scene_binding=payload["scene"], selection=selection
    )
    from embodichain.lab.task_program.language.decoder import decode_task_program

    registration.catalog.preflight(
        decode_task_program(
            {
                **declaration,
                "integration": {
                    "robot_profile": selection.robot_profile,
                    "scene_registry": selection.scene_registry,
                    "runtime_preset": selection.runtime_preset,
                },
            }
        )
    )
    assert registration.planning_services.pick_closing_bindings[0].object_id == "cube"
    assert registration.fingerprint != initial.fingerprint
    with pytest.raises(ValueError, match="B=1"):
        registration.planning_services.validate_num_envs(2)
    with pytest.raises(ValueError, match="Default physics"):
        registration.planning_services.validate_physics("newton")
    missing = deepcopy(payload)
    missing["runtime_services"]["planning_services"]["pick_closing_bindings"] = []
    rejected = _decode_configured_task_program_integration(missing).registration
    with pytest.raises(ValueError, match="exact occurrence-bound"):
        rejected.catalog.preflight(
            decode_task_program(
                {
                    **declaration,
                    "integration": {
                        "robot_profile": selection.robot_profile,
                        "scene_registry": selection.scene_registry,
                        "runtime_preset": selection.runtime_preset,
                    },
                }
            )
        )
    wrong = deepcopy(payload)
    wrong["scene"]["rigid_objects"].append(
        {"entity_id": "wrong_object", "simulation_uid": "different_native"}
    )
    wrong["runtime_services"]["planning_services"]["pick_closing_bindings"][0][
        "object_id"
    ] = "wrong_object"
    crossed = _decode_configured_task_program_integration(wrong).registration
    with pytest.raises(ValueError, match="subject differs"):
        crossed.catalog.preflight(
            decode_task_program(
                {
                    **declaration,
                    "integration": {
                        "robot_profile": selection.robot_profile,
                        "scene_registry": selection.scene_registry,
                        "runtime_preset": selection.runtime_preset,
                    },
                }
            )
        )
    precontact = deepcopy(payload)
    precontact["robot_profile"]["presets"][0]["action_options"]["pick"].update(
        native_closing_revision=8, adaptive_grasp="ordinary"
    )
    precontact_registration = _decode_configured_task_program_integration(
        precontact
    ).registration
    with pytest.raises(ValueError, match="discrete_mesh"):
        precontact_registration.catalog.preflight(
            decode_task_program(
                {
                    **declaration,
                    "integration": {
                        "robot_profile": selection.robot_profile,
                        "scene_registry": selection.scene_registry,
                        "runtime_preset": selection.runtime_preset,
                    },
                }
            )
        )
    opening = deepcopy(payload)
    opening_binding = opening["runtime_services"]["planning_services"][
        "pick_closing_bindings"
    ][0]
    opening_binding.update(
        opening_reference_id="cube",
        opening_object_target={
            "kind": "pose",
            "position": [0.1, 0.2, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    opening_registration = _decode_configured_task_program_integration(
        opening
    ).registration
    assert opening_registration.planning_services.pick_closing_bindings[
        0
    ].opening_object_target.position == (0.1, 0.2, 0.3)
    assert opening_registration.fingerprint != registration.fingerprint
    bad_grid = deepcopy(payload)
    bad_grid["runtime_services"]["hand_geometries"][hand]["sample_count"] = 64
    with pytest.raises(ValueError, match="65-sample"):
        _decode_configured_task_program_integration(bad_grid)
    wrong = deepcopy(payload)
    wrong["runtime_services"]["planning_services"]["pick_closing_bindings"][0][
        "unknown_capability"
    ] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_configured_task_program_integration(wrong)


@pytest.mark.parametrize("damage", ["missing_stamp", "duplicate", "half_opening"])
def test_native_pick_service_declaration_rejects_incomplete_bindings(damage):
    payload = _integration_payload("repeated_pick_place")
    services = payload.setdefault("runtime_services", {})
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_root_pose": True,
        "capture_native_revision": True,
        "pick_closing_bindings": [{"invocation_id": "p/s:0", "object_id": "cube"}],
    }
    cfg = services["planning_services"]
    if damage == "missing_stamp":
        cfg["capture_native_revision"] = False
    elif damage == "duplicate":
        cfg["pick_closing_bindings"] *= 2
    else:
        cfg["pick_closing_bindings"][0]["opening_reference_id"] = "cube"
    with pytest.raises((ValueError, TypeError)):
        _decode_configured_task_program_integration(payload)


@pytest.mark.parametrize(
    "call_id", ["simulation.pick", "gen_sim.stack_pick", "gen_sim.pick.upper_half"]
)
def test_registered_pick_alias_uses_typed_goal_constraints_and_explicit_options(
    call_id,
):
    from unittest.mock import Mock
    from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
    from embodichain.lab.task_program.integrations.extensions import (
        declare_registered_semantic_lowerer_factory,
    )
    from embodichain.toolkits.graspkit.pg_grasp.constraints import (
        GraspGeometryConstraint,
    )
    from embodichain.lab.sim.atomic_actions import AxisAlignAffordance

    cfg = {
        "kind": "pick",
        "call_id": call_id,
        "verify_phases": True,
        "routes": [
            {"object_id": "item", "target_id": "upper", "grasp_region": "upper_half"}
        ],
    }
    factory = _decode_registered_lowerer(cfg, path="lowerer")
    assert declare_registered_semantic_lowerer_factory(factory).call_id == call_id
    assert type(_decode_registered_lowerer(cfg, path="lowerer")) is type(factory)
    vertices = torch.tensor([[-0.02, -0.02, -0.1], [0.02, 0.02, 0.2]])
    affordance = AxisAlignAffordance(
        mesh_vertices=vertices,
        mesh_triangles=torch.tensor([[0, 1, 0]]),
        internal_axis=torch.tensor([0.0, 0.0, 1.0]),
    )
    semantics = ObjectSemantics(
        affordance=affordance, geometry={}, label="item", entity_id="item"
    )
    registry = SimpleNamespace(
        resolve=lambda name, **kw: SceneObjectRef(name),
        resolve_affordance=lambda *a, **kw: object(),
        object_semantics=lambda *a, **kw: semantics,
    )
    generator = Mock(spec=ParallelJawGraspPoseGenerator)
    generator.gripper_model = SimpleNamespace(
        model_id="declared", max_opening_width=0.14
    )
    robot = object()
    lowerer = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot, grasp_pose_generators={"hand": generator}),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "grasp": SimpleNamespace(
                            runtime_target=SimpleNamespace(target_id="hand")
                        )
                    }
                )
            }
        )
    )
    call = RegisteredSemanticCall(
        call_id=call_id, arguments={"object": "item", "target": "upper"}
    )
    with pytest.raises(ValueError, match="explicit unrotated"):
        lowerer.lower(call, context=None, bound=bound, option_template=PickUpOptions())
    result = lowerer.lower(
        call,
        context=None,
        bound=bound,
        option_template=PickUpOptions(constrained_grasp=True),
    )
    assert type(result.goal.grasp_constraint) is GraspGeometryConstraint
    assert result.goal.grasp_constraint.midpoint == pytest.approx(0.05)
    assert result.goal.grasp_constraint.upper_half
    assert result.goal.semantics is semantics
    held = result.registered_effect.held_objects[0]
    assert held.acquire_before_segment == "lift" and held.retain_during_segments == (
        "lift",
    )
    with pytest.raises(ValueError, match="no matching"):
        lowerer.lower(
            replace(call, arguments={"object": "item", "target": "absent"}),
            context=None,
            bound=bound,
            option_template=PickUpOptions(constrained_grasp=True),
        )


def test_constrained_pick_options_and_recovery_are_closed_and_default_identity_is_stable():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import PickPlanningRecovery

    assert (
        _canonical_value(PickUpOptions())["fields"].get("constrained_grasp", False)
        is False
    )
    assert (
        _canonical_value(PickUpOptions())["fields"].get("native_scene_trial", False)
        is False
    )
    assert _decode_action_options(
        {"kind": "pick_up", "native_scene_trial": True}, path="options"
    ).native_scene_trial

    assert (
        _canonical_value(PickUpOptions())["fields"].get(
            "qualify_downstream_targets", True
        )
        is True
    )
    assert (
        _decode_action_options(
            {"kind": "pick_up", "qualify_downstream_targets": False}, path="options"
        ).qualify_downstream_targets
        is False
    )

    assert (
        _canonical_value(PickPlanningRecovery())["fields"].get(
            "constrained_roll", False
        )
        is False
    )
    assert (
        _canonical_value(PickPlanningRecovery())["fields"].get(
            "place_continuation", False
        )
        is False
    )
    prospective = _decode_recovery_policy(
        {
            "max_replans": 0,
            "max_action_retries": 0,
            "planning_recovery": {
                "kind": "pick_bounded",
                "revision": 1,
                "place_continuation": True,
            },
        },
        path="recovery",
    ).planning_recovery
    assert prospective.place_continuation
    with pytest.raises(ValueError, match="distinct modes"):
        PickPlanningRecovery(constrained_roll=True, place_continuation=True)

    options = _decode_action_options(
        {"kind": "pick_up", "constrained_grasp": True}, path="options"
    )
    assert options.constrained_grasp
    recovery = _decode_recovery_policy(
        {
            "max_replans": 0,
            "max_action_retries": 0,
            "planning_recovery": {
                "kind": "pick_bounded",
                "revision": 1,
                "constrained_roll": True,
            },
        },
        path="recovery",
    )
    assert recovery.planning_recovery.constrained_roll
    with pytest.raises(ValueError, match="only belongs to Pick"):
        _decode_recovery_policy(
            {
                "max_replans": 0,
                "max_action_retries": 0,
                "planning_recovery": {
                    "kind": "place_bounded",
                    "revision": 1,
                    "constrained_roll": True,
                },
            },
            path="recovery",
        )
    with pytest.raises(ValueError):
        _decode_action_options(
            {"kind": "axis_align", "constrained_grasp": True}, path="options"
        )


def test_stack_place_alias_preserves_verified_grasp_and_exact_descent_targets():
    from embodichain.lab.task_program.integrations.simulation.stack_placement import (
        StackPlaceLowererFactory,
    )
    from embodichain.lab.sim.atomic_actions import StateDelta

    factory = _decode_registered_lowerer(
        {
            "kind": "stack_place",
            "routes": [
                {
                    "object_id": "payload",
                    "reference_entity_id": "support",
                    "relation": "on",
                    "world_displacement": [0.1, 0.2, 0.3],
                }
            ],
        },
        path="lowerer",
    )
    assert type(factory) is StackPlaceLowererFactory
    held_pose = torch.eye(4)[None]
    held_pose[:, 2, 3] = 0.05
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="payload",
        entity_id="payload",
    )
    held = HeldObjectState(
        semantics=semantics, object_to_eef=held_pose, grasp_xpos=torch.eye(4)[None]
    )
    task = StateDelta(held_object_updates={"custom": held}).apply(
        TaskState.empty(1, "cpu"), torch.tensor([True])
    )
    reference = torch.eye(4)[None]
    reference[:, :3, 3] = torch.tensor([[0.4, -0.2, 0.6]])
    object_pose = torch.eye(4)[None]
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=0.0, qpos=torch.zeros(1, 1), qvel=torch.zeros(1, 1)
        ),
        task=task,
        scene=SceneSnapshot(
            timestamp=0.0,
            version=0,
            entities={
                "payload": EntityState(object_pose),
                "support": EntityState(reference),
            },
        ),
        env_ids=torch.tensor([0]),
    )
    options = PlaceOptions(preserve_current_object_orientation=True)
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="custom")}
                )
            }
        ),
        preset=SimpleNamespace(action_option_template=lambda _: options),
    )
    scene = SimpleNamespace(resolve=lambda name, **kwargs: SceneObjectRef(name))
    robot = object()
    lowerer = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=scene,
        engine=SimpleNamespace(robot=robot),
    )
    call = RegisteredSemanticCall(
        call_id="gen_sim.stack_place",
        arguments={"object": "payload", "reference": "support", "relation": "on"},
    )
    result = lowerer.lower(call, context=context, bound=bound, option_template=options)
    assert result.goal.tcp_symmetry == "z_roll_180"
    torch.testing.assert_close(
        result.goal.xpos[:, :, :3, 3],
        torch.tensor([[[0.5, 0.0, 0.96], [0.5, 0.0, 0.943]]]),
    )
    assert (
        result.registered_effect.held_objects[0].relation is HeldObjectRelation.DETACHED
    )
    assert result.registered_effect.held_objects[0].release_before_segment is None
    assert result.registered_effect.held_objects[0].retain_during_segments == ()
    lookahead = lowerer.pick_lookahead_targets(
        call, picked_object=SceneObjectRef("payload"), bound=bound, previous_target=None
    )
    torch.testing.assert_close(
        lookahead[0].pose.world_displacement, torch.tensor([0.1, 0.2, 0.293])
    )
    torch.testing.assert_close(task.get_held_object("custom").object_to_eef, held_pose)
    with pytest.raises(ValueError, match="above"):
        replace(factory, approach_clearance=0.001)


def test_slide_preshape_requires_exact_hand_support_and_selected_occurrence():
    from embodichain.lab.task_program.language.decoder import decode_task_program
    from embodichain.lab.task_program.compiler import TaskProgramCompiler
    from embodichain.lab.task_program.integrations import task_program_fingerprint
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import SlideOptions
    from embodichain.lab.sim.atomic_actions.articulation_continuation import (
        SlidePlanningRecovery,
    )

    assert (
        _canonical_value(SlideOptions())["fields"].get("qualify_hand_clearance", False)
        is False
    )
    assert (
        _canonical_value(SlidePlanningRecovery())["fields"].get(
            "preshape_aperture_recovery", False
        )
        is False
    )
    payload = _prismatic_service_payload()
    payload["scene"].setdefault("rigid_objects", []).append(
        {"entity_id": "declared_support", "simulation_uid": "table_for_preshape"}
    )
    initial = _decode_configured_task_program_integration(payload).registration
    program = decode_task_program(
        {
            "program_id": "preshape_program",
            "targets": {},
            "integration": {
                "scene_registry": initial.scene_binding.registry_id,
                "robot_profile": initial.robot_profile_binding.profile_id,
                "runtime_preset": "slide",
            },
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": "slide",
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "registered",
                                "call_id": "gen_sim.articulation_slide",
                                "arguments": {
                                    "object": "cabinet_source",
                                    "state": "open",
                                },
                            },
                        },
                    }
                ],
            },
        }
    )
    compiled = TaskProgramCompiler(initial.catalog.scene).compile(program)
    segment = next(iter(compiled.iter_segments()))
    occurrence = f"{compiled.program_id}/{segment.segment_id}:0"
    resource = payload["robot_profile"]["resources"][0]
    hand = next(
        endpoint
        for endpoint in resource["endpoints"]
        if endpoint["endpoint_id"] == "grasp"
    )
    motion = next(
        endpoint
        for endpoint in resource["endpoints"]
        if endpoint["endpoint_id"] == "motion"
    )
    target = hand["control_part"]
    services = payload["runtime_services"]
    services["hand_geometries"] = {
        target: {
            "kind": "sampled_parallel_jaw",
            "motion_part": motion["control_part"],
            "hand_part": target,
            "command_preset": hand["command_preset"],
            "hand_root_link": "declared_hand_root",
            "pad_link_names": ["first_pad", "second_pad"],
            "sample_count": 65,
            "body_max_edge": 0.01,
            "pad_max_edge": 0.004,
            "max_subdivisions": 10,
        }
    }
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": task_program_fingerprint(compiled),
            "invocation_id": occurrence,
            "preset_id": "slide",
        }
    ]
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_native_revision": True,
        "slide_preshape_bindings": [
            {
                "invocation_id": occurrence,
                "grasp_target_id": target,
                "table_uid": "table_for_preshape",
            }
        ],
    }
    preset = next(
        p for p in payload["robot_profile"]["presets"] if p["preset_id"] == "slide"
    )
    preset["action_options"]["gen_sim.articulation_slide"].update(
        qualify_hand_clearance=True, preshape_fraction=0.85
    )
    preset["recovery"]["planning_recovery"]["preshape_aperture_recovery"] = True
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.catalog.preflight(program) == compiled
    assert registration.fingerprint != initial.fingerprint
    missing = deepcopy(payload)
    missing["runtime_services"]["hand_geometries"] = {}
    with pytest.raises(ValueError, match="hand geometry calibration"):
        _decode_configured_task_program_integration(missing)
    broken = deepcopy(payload)
    broken["runtime_services"]["planning_services"]["slide_preshape_bindings"] = []
    with pytest.raises(ValueError, match="exact serial occurrence"):
        _decode_configured_task_program_integration(
            broken
        ).registration.catalog.preflight(program)
    unknown = deepcopy(payload)
    unknown["runtime_services"]["planning_services"]["slide_preshape_bindings"][0][
        "table_uid"
    ] = "not_registered"
    with pytest.raises(ValueError, match="declared native support"):
        _decode_configured_task_program_integration(unknown)
    bad = deepcopy(payload)
    next(p for p in bad["robot_profile"]["presets"] if p["preset_id"] == "slide")[
        "action_options"
    ]["gen_sim.articulation_slide"]["preshape_fraction"] = 0.8
    with pytest.raises(ValueError, match="original85%"):
        _decode_configured_task_program_integration(bad).registration.catalog.preflight(
            program
        )


def _pour_projection_payload(region=False):
    """Build one exact five-role recipe through the public compile-only entry."""
    from embodichain.lab.task_program import TaskProgramIntegrationCfg
    from embodichain.lab.task_program.integrations import (
        compile_configured_task_program,
        task_program_fingerprint,
    )

    payload = _tableware_integration_payload("pour_water")
    services = payload["runtime_services"]
    services["registered_semantic_lowerers"] = [
        {
            "kind": "pour_region_transport" if region else "pour_transport",
            "routes": [
                {
                    "object_id": "bottle",
                    "target_id": "stage",
                    "pose": {"kind": "scene_entity", "entity_id": "cup"},
                },
                {
                    "object_id": "bottle",
                    "target_id": "home",
                    "pose": {
                        "kind": "pose",
                        "position": [0.1, -0.2, 0.3],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    },
                },
            ],
            "staging_routes": [
                {
                    "object_id": "bottle",
                    "target_id": "stage",
                    "receiver_id": "cup",
                    "clearance_m": 0.05,
                }
            ],
            **({"proxy_position_tolerance_m": 0.03} if region else {}),
        },
        {"kind": "retained_pour", "object_id": "bottle"},
    ]
    preset = payload["robot_profile"]["presets"][0]
    preset["action_options"]["pick"].pop("fixed_object_to_eef")
    preset["action_options"]["pick"]["adaptive_grasp"] = "pour"
    future = deepcopy(preset)
    future["preset_id"] = "future_pour"
    future["motion"]["sample_count"] = 101
    future["action_options"]["simulation.pour"]["rotate_angle"] = 1.2
    payload["robot_profile"]["presets"].append(future)
    calls = [
        {"kind": "pick", "object": "bottle", "grasp": "bottle_grasp"},
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {"object": "bottle", "target": "stage", "reference": "cup"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.pour",
            "arguments": {"object": "bottle"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {"object": "bottle", "target": "home"},
        },
        {
            "kind": "place",
            "object": "bottle",
            "at": {"kind": "target_ref", "target": "home"},
        },
    ]
    for call in calls:
        call["resources"] = {"primary": "primary_manipulator"}
    declaration = {
        "program_id": "pour_roles",
        "targets": {
            "home": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.1, -0.2, 0.3],
                        "quaternion_xyzw": [0, 0, 0, 1],
                    }
                ],
            },
        },
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": name,
                    "steps": {"kind": "invoke", "call": call},
                }
                for name, call in zip(
                    ("pick", "stage", "pour", "return", "place"), calls
                )
            ],
        },
    }
    selection = TaskProgramIntegrationCfg(
        scene_registry=payload["scene"]["registry_id"],
        robot_profile=payload["robot_profile"]["profile_id"],
        runtime_preset="trajectory",
    )
    compiled = compile_configured_task_program(
        declaration,
        scene_binding=payload["scene"],
        selection=selection,
    )
    ids = [
        f"{compiled.program_id}/{segment.segment_id}:0"
        for segment in compiled.iter_segments()
    ]
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": task_program_fingerprint(compiled),
            "invocation_id": identity,
            "preset_id": "future_pour" if i == 2 else "trajectory",
        }
        for i, identity in enumerate(ids)
    ]
    services["hand_geometries"] = {
        "right_eef": {
            "kind": "sampled_parallel_jaw",
            "motion_part": "right_arm",
            "hand_part": "right_eef",
            "command_preset": "right_parallel_jaw_commands",
            "hand_root_link": "declared_root",
            "pad_link_names": ["pad1", "pad2"],
            "sample_count": 65,
            "body_max_edge": 0.01,
            "pad_max_edge": 0.004,
            "max_subdivisions": 10,
        },
    }
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_root_pose": True,
        "capture_native_revision": True,
        "pour_projection_bindings": [
            dict(
                zip(
                    (
                        "pick_invocation_id",
                        "stage_invocation_id",
                        "pour_invocation_id",
                        "return_invocation_id",
                        "place_invocation_id",
                    ),
                    ids,
                )
            )
        ],
    }
    from embodichain.lab.task_program import decode_task_program

    declaration["integration"] = {
        "scene_registry": selection.scene_registry,
        "robot_profile": selection.robot_profile,
        "runtime_preset": selection.runtime_preset,
    }
    return payload, decode_task_program(declaration)


@pytest.mark.parametrize("region", [False, True])
def test_pour_projection_resolves_future_occurrence_policy_without_simulation(region):
    payload, program = _pour_projection_payload(region)
    registration = _decode_configured_task_program_integration(payload).registration
    compiled = registration.catalog.preflight(program)
    (route,) = registration.catalog.resolve_pour_projection_routes(compiled)
    assert (route.angle_rad, route.sample_count, route.clearance_m) == (1.2, 101, 0.05)
    assert route.proxy_position_tolerance_m == (0.03 if region else None)
    assert route.object_id == "bottle" and route.receiver_id == "cup"
    assert torch.equal(route.return_pose[:3, 3], torch.tensor([0.1, -0.2, 0.3]))


@pytest.mark.parametrize(
    "defect",
    [
        "unknown",
        "order",
        "missing_hand",
        "missing_stamp",
        "policy",
        "return",
        "shared_occurrence",
        "extra_field",
    ],
)
def test_pour_projection_rejects_unclosed_roles_before_runtime(defect):
    payload, program = _pour_projection_payload()
    services = payload["runtime_services"]
    binding = services["planning_services"]["pour_projection_bindings"][0]
    if defect == "unknown":
        binding["pour_invocation_id"] = "unknown/call:0"
    elif defect == "order":
        binding["stage_invocation_id"], binding["return_invocation_id"] = (
            binding["return_invocation_id"],
            binding["stage_invocation_id"],
        )
    elif defect == "missing_hand":
        services["hand_geometries"] = {}
    elif defect == "missing_stamp":
        services["planning_services"]["capture_native_revision"] = False
    elif defect == "policy":
        payload["robot_profile"]["presets"][0]["action_options"]["pick"][
            "adaptive_grasp"
        ] = "none"
    elif defect == "return":
        services["registered_semantic_lowerers"][0]["routes"][1]["pose"]["position"][
            0
        ] += 0.1
    elif defect == "shared_occurrence":
        services["planning_services"]["pour_projection_bindings"].append(
            deepcopy(binding)
        )
    elif defect == "extra_field":
        binding["angle_rad"] = 2.0
    with pytest.raises((ValueError, TypeError)):
        _decode_configured_task_program_integration(
            payload
        ).registration.catalog.preflight(program)


def test_coordinated_contact_services_require_explicit_owned_generators_and_geometry():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    assert (
        "coordinated_grasps"
        not in _canonical_value(SimulationPlanningServicesCfg())["fields"]
    )
    payload = _integration_payload("hand_over")
    services = payload.setdefault("runtime_services", {})
    resources = payload["robot_profile"]["resources"]
    hands = []
    geometry = {}
    for resource in resources:
        endpoints = {item["endpoint_id"]: item for item in resource["endpoints"]}
        target = endpoints["grasp"]["control_part"]
        hands.append(target)
        services["grasp_pose_generators"][target]["filter_ground_collision"] = False
        geometry[target] = dict(
            kind="sampled_parallel_jaw",
            motion_part=endpoints["motion"]["control_part"],
            hand_part=target,
            command_preset=endpoints["grasp"]["command_preset"],
            hand_root_link=target + "_root",
            pad_link_names=[target + "_a", target + "_b"],
            sample_count=65,
            body_max_edge=0.01,
            pad_max_edge=0.004,
            max_subdivisions=10,
        )
    services["hand_geometries"] = geometry
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "coordinated_grasps": [
            dict(
                left_grasp_target_id=hands[0],
                right_grasp_target_id=hands[1],
                contact=dict(
                    contact_tolerance=0.0005,
                    minimum_normal_cosine=0.5,
                    normal_probe_distance=0.00001,
                    max_candidates=32,
                    candidate_chunk_size=32,
                    cargo_clearance_reserve=0.008,
                ),
            )
        ],
    }
    carrier = next(
        item["entity_id"]
        for item in payload["scene"]["rigid_objects"]
        if any(
            value["kind"] == "antipodal_grasp" for value in item.get("affordances", ())
        )
    )
    services.setdefault("registered_semantic_lowerers", []).append(
        dict(
            kind="coordinated_transport",
            routes=[
                dict(
                    object_id=carrier,
                    target_id="paired_target",
                    world_displacement=[0.14, 0.0, 0.0],
                )
            ],
        )
    )
    payload["robot_profile"]["defaults"]["coordinated_pickment"] = {
        "left": resources[0]["resource_id"],
        "right": resources[1]["resource_id"],
    }
    payload["robot_profile"]["presets"][0]["action_options"][
        "simulation.coordinated_transport"
    ] = {"kind": "coordinated_pickment", "release": True}
    for resource in resources:
        endpoint = next(
            value for value in resource["endpoints"] if value["endpoint_id"] == "motion"
        )
        endpoint["capabilities"] = sorted(
            set(endpoint["capabilities"])
            | {"kinematics.inverse", "kinematics.batch_inverse"}
        )
    registered = _decode_configured_task_program_integration(payload).registration
    assert (
        registered.planning_services.coordinated_grasps[0].contact.max_candidates == 32
    )
    from embodichain.lab.task_program import decode_task_program

    program = dict(
        program_id="paired",
        targets={},
        integration=dict(
            robot_profile=registered.robot_profile_binding.profile_id,
            scene_registry=registered.scene_binding.registry_id,
            runtime_preset=payload["robot_profile"]["presets"][0]["preset_id"],
        ),
        program=dict(
            kind="sequence",
            items=[
                dict(
                    kind="segment",
                    name="carry",
                    steps=dict(
                        kind="invoke",
                        call=dict(
                            kind="registered",
                            call_id="simulation.coordinated_transport",
                            arguments=dict(object=carrier, target="paired_target"),
                        ),
                    ),
                )
            ],
        ),
    )
    registered.catalog.preflight(decode_task_program(program))
    reversed_roles = deepcopy(program)
    reversed_roles["program"]["items"][0]["steps"]["call"]["resources"] = {
        "left": resources[1]["resource_id"],
        "right": resources[0]["resource_id"],
    }
    with pytest.raises(ValueError, match="grasp slots differ"):
        registered.catalog.preflight(decode_task_program(reversed_roles))
    with pytest.raises(ValueError, match="B=1"):
        registered.planning_services.validate_num_envs(2)
    for change in ("geometry", "ground", "pair", "unknown"):
        bad = deepcopy(payload)
        values = bad["runtime_services"]
        if change == "geometry":
            del values["hand_geometries"][hands[0]]
        elif change == "ground":
            values["grasp_pose_generators"][hands[0]]["filter_ground_collision"] = True
        elif change == "pair":
            values["planning_services"]["coordinated_grasps"] *= 2
        else:
            values["planning_services"]["coordinated_grasps"][0]["contact"][
                "hidden_retry"
            ] = 1
        with pytest.raises((ValueError, TypeError)):
            _decode_configured_task_program_integration(bad)


@pytest.mark.parametrize("revision", [1, 2])
def test_native_grip_evidence_config_is_closed_bound_and_provider_free(
    monkeypatch, revision
):
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensor
    from embodichain.lab.task_program.integrations.simulation.grip_evidence import (
        NativeGripEvidenceFactory,
    )

    monkeypatch.setattr(
        ContactSensor,
        "__init__",
        lambda *a, **k: pytest.fail("Decoder constructed a contact sensor"),
    )
    payload = _integration_payload("hand_over")
    resources = payload["robot_profile"]["resources"]
    objects = [
        (item["entity_id"], item["simulation_uid"])
        for item in payload["scene"]["rigid_objects"]
        if any(
            value["kind"] == "antipodal_grasp" for value in item.get("affordances", ())
        )
    ]
    pads = []
    openings = []
    for resource in resources:
        endpoint = next(
            value for value in resource["endpoints"] if value["endpoint_id"] == "grasp"
        )
        part = endpoint["control_part"]
        if any(existing == part for existing, _ in pads):
            continue
        pads.append([part, [part + "_pad_a", part + "_pad_b"]])
        openings.append(
            dict(
                control_part=part,
                joint_names=[part + "_driver"],
                driver_indices=[0],
                open_qpos=[0.0],
                tolerance=0.05,
            )
        )
    config = dict(
        kind="native_grip",
        sensor_uid="explicit_grip",
        object_bindings=[list(value) for value in objects],
        pad_bindings=pads,
        algorithm_revision=revision,
    )
    if revision == 2:
        config["opening_bindings"] = openings
    payload.setdefault("runtime_services", {})["control_part_evidence"] = config
    registration = _decode_configured_task_program_integration(payload).registration
    factory = registration.control_part_evidence_factory
    assert (
        type(factory) is NativeGripEvidenceFactory
        and factory.algorithm_revision == revision
    )
    sensor = factory.contact_sensor_cfg("physical_robot")
    assert sensor.rigid_uid_list == [uid for _, uid in objects]
    assert sensor.filter_need_both_actor is True and sensor.max_contacts_per_env == 1024
    assert sensor.articulation_cfg_list[0].link_name_list == [
        name for _, names in pads for name in names
    ]
    before = registration.fingerprint
    config["pad_bindings"][0][1][0] = "mutated"
    assert factory.pad_bindings[0][1][0] != "mutated"
    assert registration.fingerprint == before
    registration.validate_physics("default")
    # A read-only contact provider does not select native planning algorithms.
    registration.validate_physics("newton")
    unknown = deepcopy(payload)
    unknown["runtime_services"]["control_part_evidence"][
        "hidden_callback"
    ] = "untrusted"
    with pytest.raises(ValueError, match="unsupported"):
        _decode_configured_task_program_integration(unknown)
    wrong = deepcopy(payload)
    wrong["runtime_services"]["control_part_evidence"]["object_bindings"][0][
        1
    ] = "foreign_native"
    with pytest.raises(ValueError, match="registered rigid owners"):
        _decode_configured_task_program_integration(wrong)
    wrong = deepcopy(payload)
    wrong["runtime_services"]["control_part_evidence"]["pad_bindings"][0][
        0
    ] = "foreign_hand"
    if revision == 2:
        wrong["runtime_services"]["control_part_evidence"]["opening_bindings"][0][
            "control_part"
        ] = "foreign_hand"
    # Physical observation hands need not be selected as semantic action resources.
    observed = _decode_configured_task_program_integration(wrong).registration
    assert observed.control_part_evidence_factory.pad_bindings[0][0] == "foreign_hand"
    if revision == 2:
        wrong = deepcopy(payload)
        wrong["runtime_services"]["control_part_evidence"]["opening_bindings"] = []
        with pytest.raises(ValueError, match="opening bindings"):
            _decode_configured_task_program_integration(wrong)


@pytest.mark.parametrize("reference_kind", ["native_uid", "explicit_alias"])
def test_native_grip_preflight_preserves_scene_aliases_and_exact_native_owner(
    monkeypatch, reference_kind
):
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensor

    monkeypatch.setattr(
        ContactSensor,
        "__init__",
        lambda *a, **k: pytest.fail("Preflight constructed a contact sensor"),
    )
    payload = _integration_payload("hand_over")
    body = payload["scene"]["rigid_objects"][0]
    canonical = body["entity_id"]
    uid = body["simulation_uid"] = "observed_native_" + canonical
    alias = "observed_alias_" + canonical
    body.setdefault("aliases", []).append(alias)
    reference = uid if reference_kind == "native_uid" else alias
    evidence = payload.setdefault("runtime_services", {})["control_part_evidence"] = {
        "kind": "native_grip",
        "sensor_uid": "observed_contacts",
        "object_bindings": [[reference, uid]],
        "pad_bindings": [["observation_hand", ["pad_a", "pad_b"]]],
    }
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.control_part_evidence_factory.object_bindings == (
        (reference, uid),
    )
    wrong = deepcopy(payload)
    wrong["runtime_services"]["control_part_evidence"]["object_bindings"][0][
        1
    ] = "another_native_owner"
    with pytest.raises(ValueError, match="registered rigid owners"):
        _decode_configured_task_program_integration(wrong)
    evidence["object_bindings"].append([canonical, uid])
    with pytest.raises(ValueError, match="distinct canonical"):
        _decode_configured_task_program_integration(payload)


def test_pick_cartesian_and_fixed_flags_remain_explicit_default_inert():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value

    original = _canonical_value(PickUpOptions())["fields"]
    assert original.get("linear_segments", False) is False
    assert original.get("qualify_fixed_grasp", False) is False
    selected = _decode_action_options(
        {
            "kind": "pick_up",
            "linear_segments": True,
            "qualify_fixed_grasp": True,
            "fixed_object_to_eef": torch.eye(4).flatten().tolist(),
        },
        path="options",
    )
    assert selected.linear_segments and selected.qualify_fixed_grasp
    for key in ("linear_segments", "qualify_fixed_grasp"):
        with pytest.raises((ValueError, TypeError)):
            _decode_action_options({"kind": "pick_up", key: 1}, path="options")


def test_place_continuation_service_schema_is_closed_and_identity_bound():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    payload = _integration_payload("repeated_pick_place")
    declaration = {
        "program_fingerprint": "1" * 64,
        "source_invocation_id": "program/acquire:0",
        "place_invocation_id": "program/release:0",
    }
    payload["runtime_services"]["planning_services"] = {
        "kind": "simulation_geometry",
        "place_continuation_bindings": [declaration],
    }
    decoded = _decode_configured_task_program_integration(payload)
    binding = decoded.registration.planning_services.place_continuation_bindings[0]
    assert (
        binding.program_fingerprint == "1" * 64
        and binding.transport_invocation_id is None
    )
    assert (
        "place_continuation_bindings"
        not in _canonical_value(SimulationPlanningServicesCfg())["fields"]
    )
    from embodichain.lab.task_program.integrations.simulation.place_continuation import (
        HeldAlignmentContinuationBinding,
    )

    held_payload = deepcopy(payload)
    held_payload["runtime_services"]["planning_services"][
        "place_continuation_bindings"
    ][0]["kind"] = "held_alignment"
    held_decoded = _decode_configured_task_program_integration(held_payload)
    assert (
        type(held_decoded.registration.planning_services.place_continuation_bindings[0])
        is HeldAlignmentContinuationBinding
    )
    for change in (
        {"requires_full_preview": False},
        {"program_fingerprint": "old"},
        {"place_invocation_id": declaration["source_invocation_id"]},
    ):
        invalid = deepcopy(payload)
        invalid["runtime_services"]["planning_services"][
            "place_continuation_bindings"
        ] = [{**declaration, **change}]
        with pytest.raises((TypeError, ValueError)):
            _decode_configured_task_program_integration(invalid)


def _jaw_pour_payload():
    payload, program = _pour_projection_payload()
    services = payload["runtime_services"]
    (binding,) = services["planning_services"].pop("pour_projection_bindings")
    transport, _ = services["registered_semantic_lowerers"]
    transport["kind"] = "move_held_object"
    transport.pop("staging_routes")
    services["registered_semantic_lowerers"][1] = dict(
        kind="jaw_pour",
        object_id="bottle",
        receivers=[[binding["pour_invocation_id"], "cup"]],
    )
    return payload, program


def test_jaw_pour_factory_requires_exact_occurrences_and_preserves_goal_semantics():
    from types import SimpleNamespace
    from embodichain.lab.task_program.integrations.simulation.jaw_pour import (
        JawPourFactory,
        _JawPourLowerer,
    )
    from embodichain.lab.task_program.compiler.lowering import SemanticLowering
    from embodichain.lab.sim.atomic_actions.primitives.pour import PourGoal, JawPourGoal

    payload, program = _jaw_pour_payload()
    registration = _decode_configured_task_program_integration(payload).registration
    registration.catalog.preflight(program)
    factory = next(
        value
        for value in registration.registered_semantic_lowerer_factories
        if type(value) is JawPourFactory
    )
    occurrence, receiver = factory.receivers[0]
    observed = []

    def lower(call, **kwargs):
        observed.append(call)
        return SemanticLowering(goal=PourGoal())

    lowerer = _JawPourLowerer(SimpleNamespace(lower=lower), factory.receivers)
    goal = lowerer.lower_occurrence(
        "sentinel",
        context=None,
        bound=None,
        option_template=None,
        invocation_id=occurrence,
        revision=0,
    ).goal
    assert type(goal) is JawPourGoal and goal.receiver_entity_id == receiver
    assert observed == ["sentinel"]
    with pytest.raises(ValueError, match="no declared receiver"):
        lowerer.lower_occurrence(
            "sentinel",
            context=None,
            bound=None,
            option_template=None,
            invocation_id=occurrence + "stale",
            revision=0,
        )
    with pytest.raises(ValueError, match="exact execution occurrence"):
        lowerer.lower("sentinel", context=None, bound=None, option_template=None)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_occurrence",
        "unknown_receiver",
        "duplicate",
        "missing_geometry",
        "extra_field",
    ],
)
def test_jaw_pour_preflight_rejects_unbound_or_unsupported_declarations(defect):
    payload, program = _jaw_pour_payload()
    services = payload["runtime_services"]
    declaration = services["registered_semantic_lowerers"][1]
    if defect == "unknown_occurrence":
        declaration["receivers"][0][0] += "stale"
    elif defect == "unknown_receiver":
        declaration["receivers"][0][1] = "unknown"
    elif defect == "duplicate":
        declaration["receivers"].append(deepcopy(declaration["receivers"][0]))
    elif defect == "missing_geometry":
        services.pop("hand_geometries")
    else:
        declaration["fallback"] = True
    with pytest.raises((ValueError, KeyError)):
        _decode_configured_task_program_integration(
            payload
        ).registration.catalog.preflight(program)


def test_free_end_effector_cartesian_option_is_explicit_and_closed():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_end_effector import (
        MoveEndEffectorOptions,
    )

    assert (
        _decode_action_options(
            {"kind": "move_end_effector"}, path="options"
        ).linear_segments
        is False
    )
    assert (
        _decode_action_options(
            {"kind": "move_end_effector", "linear_segments": True}, path="options"
        ).linear_segments
        is True
    )
    with pytest.raises((TypeError, ValueError)):
        _decode_action_options(
            {"kind": "move_end_effector", "linear_segments": 1}, path="options"
        )


def test_handover_cartesian_is_explicit_and_preflight_checks_whole_policy():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value

    assert (
        _canonical_value(HandOverOptions())["fields"].get("linear_segments", False)
        is False
    )
    selected = _decode_action_options(
        {"kind": "hand_over", "linear_segments": True}, path="options"
    )
    assert selected.linear_segments
    with pytest.raises((TypeError, ValueError)):
        _decode_action_options(
            {"kind": "hand_over", "linear_segments": 1}, path="options"
        )
    payload = _integration_payload("hand_over")
    preset = payload["robot_profile"]["presets"][0]
    preset["action_options"]["hand_over"]["linear_segments"] = True
    program = _deployment_from_path(_config_path("hand_over")).load_program()
    with pytest.raises(ValueError, match="Cartesian HandOver"):
        _decode_configured_task_program_integration(
            payload
        ).registration.catalog.preflight(program)
    preset["motion"].update(
        strategy="ik_interp",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        dynamic_collision_mode="off",
        segment_sample_count_retries=[260, 320],
        velocity_limit=None,
        acceleration_limit=None,
        segment_continuous_ik_recovery=False,
    )
    compiled = _decode_configured_task_program_integration(
        payload
    ).registration.catalog.preflight(program)
    assert compiled.program_id == program.program_id
    invalid = deepcopy(payload)
    invalid["robot_profile"]["presets"][0]["motion"][
        "segment_continuous_ik_recovery"
    ] = True
    with pytest.raises(ValueError, match="Cartesian HandOver"):
        _decode_configured_task_program_integration(
            invalid
        ).registration.catalog.preflight(program)


def _drawer_configured_payload():
    from embodichain.lab.task_program.language.decoder import decode_task_program
    from embodichain.lab.task_program.compiler import TaskProgramCompiler
    from embodichain.lab.task_program.integrations import task_program_fingerprint
    from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding

    payload = _integration_payload("repeated_pick_place")
    payload["scene"]["articulations"] = [
        {"entity_id": "cabinet", "simulation_uid": "cabinet"}
    ]
    payload["scene"]["links"] = [
        {
            "entity_id": "drawer_frame",
            "articulation_id": "cabinet",
            "native_link_name": "drawer",
            "affordances": [
                {
                    "entity_id": "inside_drawer",
                    "kind": "container",
                    "native_name": "interior",
                }
            ],
        }
    ]
    initial = _decode_configured_task_program_integration(payload).registration
    program = decode_task_program(
        {
            "program_id": "drawer_program",
            "targets": {},
            "integration": {
                "scene_registry": initial.scene_binding.registry_id,
                "robot_profile": initial.robot_profile_binding.profile_id,
                "runtime_preset": "trajectory",
            },
            "program": {
                "kind": "sequence",
                "items": [
                    {
                        "kind": "segment",
                        "name": "pick",
                        "steps": {
                            "kind": "invoke",
                            "call": {"kind": "pick", "object": "cube"},
                        },
                    },
                    {
                        "kind": "segment",
                        "name": "transport",
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "registered",
                                "call_id": "gen_sim.drawer_transport",
                                "arguments": {
                                    "object": "cube",
                                    "target": "inside_drawer",
                                },
                            },
                        },
                    },
                    {
                        "kind": "segment",
                        "name": "place",
                        "steps": {
                            "kind": "invoke",
                            "call": {
                                "kind": "place",
                                "object": "cube",
                                "inside": "inside_drawer",
                            },
                        },
                    },
                ],
            },
        }
    )
    compiled = TaskProgramCompiler(initial.catalog.scene).compile(program)
    occurrences = {
        role: f"{compiled.program_id}/{segment.segment_id}:0"
        for role, segment in zip(
            ("pick", "transport", "place"), compiled.iter_segments(), strict=True
        )
    }
    profile = payload["robot_profile"]
    profile["defaults"]["move_held_object"] = {"primary": "primary_manipulator"}
    profile["presets"][0]["action_options"]["gen_sim.drawer_transport"] = {
        "kind": "move_held_object"
    }
    base = deepcopy(profile["presets"][0])
    transport = deepcopy(base)
    transport["preset_id"] = "drawer_transport"
    transport["motion"]["strategy"] = "motion_gen"
    transport["action_options"]["gen_sim.drawer_transport"] = {
        "kind": "move_held_object",
        "drawer_transport": True,
    }
    place = deepcopy(base)
    place["preset_id"] = "drawer_place"
    place["action_options"]["place"].update(drawer_placement=True)
    pick = deepcopy(base)
    pick["preset_id"] = "drawer_pick"
    pick["action_options"]["pick"]["qualify_downstream_targets"] = False
    profile["presets"].extend([pick, transport, place])
    source = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/grip",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.4),
    )
    services = payload["runtime_services"]
    services.setdefault("registered_semantic_lowerers", []).append(
        {
            "kind": "drawer_transport",
            "routes": [
                {
                    "object_id": "cube",
                    "target_id": "inside_drawer",
                    "pose": {"kind": "scene_entity", "entity_id": "inside_drawer"},
                }
            ],
        }
    )
    services["invocation_policy_bindings"] = [
        {
            "program_fingerprint": task_program_fingerprint(compiled),
            "invocation_id": occurrences[role],
            "preset_id": f"drawer_{role}",
        }
        for role in ("pick", "transport", "place")
    ]
    services["planning_services"] = {
        "kind": "simulation_geometry",
        "capture_root_pose": True,
        "capture_native_revision": True,
        "collision_roster": {
            "rigid_objects": [["cube", "cube"]],
            "articulation_links": [["drawer_frame", "cabinet", "drawer"]],
        },
        "observed_curobo": {},
        "drawer_bindings": [
            {
                "invocation_id": occurrences[role],
                "operation": role,
                "target_id": "inside_drawer",
                "link_entity_id": "drawer_frame",
                "object_id": "cube",
                "source": source.payload(),
                "lower": [-0.2, -0.2, 0.0],
                "upper": [0.2, 0.2, 0.2],
            }
            for role in ("transport", "place")
        ],
    }
    return payload, program, compiled


def test_configured_drawer_routes_require_exact_options_identity_and_native_scope():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import (
        PlaceOptions,
        MoveHeldObjectOptions,
        SlideOptions,
    )

    for value, field in [
        (PlaceOptions(), "drawer_placement"),
        (MoveHeldObjectOptions(), "drawer_transport"),
        (SlideOptions(), "drawer_containment"),
    ]:
        assert _canonical_value(value)["fields"].get(field, False) is False
    payload, program, compiled = _drawer_configured_payload()
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.catalog.preflight(program) == compiled
    assert len(registration.planning_services.drawer_bindings) == 2
    wrong_parent = deepcopy(payload)
    wrong_parent["scene"]["rigid_objects"][0]["affordances"].append(
        wrong_parent["scene"]["links"][0]["affordances"].pop()
    )
    with pytest.raises(ValueError, match="declared physical drawer link"):
        _decode_configured_task_program_integration(wrong_parent)
    bad = deepcopy(payload)
    bad["runtime_services"]["planning_services"]["drawer_bindings"][0]["unknown"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_configured_task_program_integration(bad)
    bad = deepcopy(payload)
    bad["runtime_services"]["planning_services"]["drawer_bindings"][0][
        "link_entity_id"
    ] = "undeclared"
    with pytest.raises(ValueError, match="exact native entity owners"):
        _decode_configured_task_program_integration(bad)
    bad = deepcopy(payload)
    next(
        value
        for value in bad["robot_profile"]["presets"]
        if value["preset_id"] == "drawer_transport"
    )["action_options"]["gen_sim.drawer_transport"]["drawer_transport"] = False
    with pytest.raises(ValueError, match="generic MoveHeldObject"):
        _decode_configured_task_program_integration(bad).registration.catalog.preflight(
            program
        )


def test_all_failed_retry_declaration_is_closed_and_default_identity_stays_unchanged():
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
    )
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import RecoveryPolicy

    assert "retry_all_failed_plans" not in _canonical_value(RecoveryPolicy())["fields"]
    selected = _decode_recovery_policy(
        {"max_replans": 3, "max_action_retries": 5, "retry_all_failed_plans": False},
        path="recovery",
    )
    assert selected.retry_all_failed_plans is False
    assert selected.max_action_retries == 5 and selected.max_replans == 3
    assert _canonical_value(selected)["fields"]["retry_all_failed_plans"] is False
    with pytest.raises((ValueError, TypeError)):
        _decode_recovery_policy(
            {"max_replans": 3, "max_action_retries": 5, "retry_all_failed_plans": 0},
            path="recovery",
        )


def _grounded_opening_payload():
    from embodichain.lab.task_program import (
        TaskProgramIntegrationCfg,
        decode_task_program,
    )
    from embodichain.lab.task_program.integrations import (
        compile_configured_task_program,
        task_program_fingerprint,
    )

    payload = _integration_payload("repeated_pick_place")
    initial = _decode_configured_task_program_integration(payload).registration
    payload["scene"]["rigid_objects"].append(
        dict(
            entity_id="support",
            simulation_uid="support",
            dynamics="static",
            semantic_type="support_surface",
        )
    )
    resource = payload["robot_profile"]["resources"][0]
    arm = next(
        value["control_part"]
        for value in resource["endpoints"]
        if value["endpoint_id"] == "motion"
    )
    hand = next(
        value["control_part"]
        for value in resource["endpoints"]
        if value["endpoint_id"] == "grasp"
    )
    command = next(
        value["preset_id"]
        for value in payload["robot_profile"]["command_presets"]
        if value["control_part"] == hand
    )
    acquire = deepcopy(payload["robot_profile"]["presets"][0])
    release = deepcopy(acquire)
    acquire["preset_id"] = "native_acquire"
    release["preset_id"] = "native_release"
    acquire["effect_assurance"] = release["effect_assurance"] = payload[
        "robot_profile"
    ]["presets"][0]["effect_assurance"] = "verified"
    acquire["action_options"]["pick"].pop("fixed_object_to_eef", None)
    acquire["action_options"]["pick"].update(
        adaptive_grasp="ordinary",
        native_closing_revision=3,
        full_opening_clearance=True,
        require_qualified_approach=True,
    )
    acquire["recovery"] = {
        "max_replans": 0,
        "max_action_retries": 0,
        "planning_recovery": {
            "kind": "pick_bounded",
            "revision": 1,
            "place_continuation": True,
        },
    }
    acquire["motion"].update(
        strategy="ik_interp",
        dynamic_collision_mode="off",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        velocity_limit=None,
        acceleration_limit=None,
    )
    release["action_options"]["simulation.place_relative"] = deepcopy(
        release["action_options"]["place"]
    )
    for preset, identity in [(acquire, "pick"), (release, "simulation.place_relative")]:
        preset.setdefault("effect_monitors", {})[identity] = {
            "monitor_id": "builtin.composite_effect",
            "revision": "1",
            "params": {},
        }
    payload["robot_profile"]["presets"] += [acquire, release]
    services = payload.setdefault("runtime_services", {})
    services["registered_semantic_lowerers"] = [
        dict(
            kind="place_relative",
            routes=[
                dict(
                    object_id="cube",
                    reference_entity_id="support",
                    relation="on",
                    world_displacement=[0.0, 0.0, 0.1],
                )
            ],
        )
    ]
    selection = TaskProgramIntegrationCfg(
        robot_profile=initial.robot_profile_binding.profile_id,
        scene_registry=initial.scene_binding.registry_id,
        runtime_preset=payload["robot_profile"]["presets"][0]["preset_id"],
    )
    declaration = dict(
        program_id="grounded_opening",
        targets={},
        program={
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": "acquire",
                    "steps": {
                        "kind": "invoke",
                        "call": {
                            "kind": "pick",
                            "object": "cube",
                            "resources": {"primary": resource["resource_id"]},
                        },
                    },
                },
                {
                    "kind": "segment",
                    "name": "release",
                    "steps": {
                        "kind": "invoke",
                        "call": {
                            "kind": "registered",
                            "call_id": "simulation.place_relative",
                            "arguments": {
                                "object": "cube",
                                "reference": "support",
                                "relation": "on",
                            },
                            "resources": {"primary": resource["resource_id"]},
                        },
                    },
                },
            ],
        },
    )
    compiled = compile_configured_task_program(
        declaration, scene_binding=payload["scene"], selection=selection
    )
    ids = [
        f"{compiled.program_id}/{segment.segment_id}:0"
        for segment in compiled.iter_segments()
    ]
    fingerprint = task_program_fingerprint(compiled)
    services["invocation_policy_bindings"] = [
        dict(program_fingerprint=fingerprint, invocation_id=identity, preset_id=preset)
        for identity, preset in zip(ids, ("native_acquire", "native_release"))
    ]
    services["hand_geometries"] = {
        hand: dict(
            kind="sampled_parallel_jaw",
            motion_part=arm,
            hand_part=hand,
            command_preset=command,
            hand_root_link="hand",
            pad_link_names=["pad_a", "pad_b"],
            sample_count=65,
            body_max_edge=0.01,
            pad_max_edge=0.004,
            max_subdivisions=10,
        )
    }
    services["planning_services"] = dict(
        kind="simulation_geometry",
        capture_root_pose=True,
        capture_native_revision=True,
        free_motion_query="discrete_mesh",
        place_continuation_bindings=[
            dict(
                program_fingerprint=fingerprint,
                source_invocation_id=ids[0],
                place_invocation_id=ids[1],
            )
        ],
        pick_closing_bindings=[
            dict(
                invocation_id=ids[0],
                object_id="cube",
                opening_reference_id="support",
                opening_place_invocation_id=ids[1],
            )
        ],
    )
    declaration["integration"] = dict(
        robot_profile=selection.robot_profile,
        scene_registry=selection.scene_registry,
        runtime_preset=selection.runtime_preset,
    )
    return payload, decode_task_program(declaration)


def test_native_opening_links_exact_declared_place_without_grounding():
    payload, program = _grounded_opening_payload()
    result = _decode_configured_task_program_integration(payload)
    result.registration.catalog.preflight(program)
    (binding,) = result.registration.planning_services.pick_closing_bindings
    assert binding.opening_object_target is None
    assert binding.opening_place_invocation_id.endswith(":0")
    disabled = deepcopy(payload)
    selected_preset = disabled["runtime_services"]["invocation_policy_bindings"][0][
        "preset_id"
    ]
    preset = next(
        value
        for value in disabled["robot_profile"]["presets"]
        if value["preset_id"] == selected_preset
    )
    preset["action_options"]["pick"]["qualify_downstream_targets"] = False
    with pytest.raises(ValueError, match="opening"):
        _decode_configured_task_program_integration(
            disabled
        ).registration.catalog.preflight(program)
    for field, value in [
        ("opening_place_invocation_id", "unknown"),
        ("opening_reference_id", "cube"),
    ]:
        damaged = deepcopy(payload)
        damaged["runtime_services"]["planning_services"]["pick_closing_bindings"][0][
            field
        ] = value
        with pytest.raises(ValueError, match="opening"):
            _decode_configured_task_program_integration(
                damaged
            ).registration.catalog.preflight(program)


def _released_continuation_payload():
    from embodichain.lab.task_program import (
        TaskProgramIntegrationCfg,
        decode_task_program,
    )
    from embodichain.lab.task_program.integrations import (
        compile_configured_task_program,
        task_program_fingerprint,
    )

    payload = _integration_payload("repeated_pick_place")
    profile = payload["robot_profile"]
    endpoint = profile["resources"][0]["endpoints"][0]
    endpoint["capabilities"].append("motion.joint_position")
    endpoint["command_preset"] = "ready_commands"
    profile["command_presets"].append(
        dict(
            preset_id="ready_commands", control_part="arm", commands={"park": [0.0] * 6}
        )
    )
    services = payload["runtime_services"]
    services["registered_semantic_lowerers"] = [
        dict(
            kind="clear_released",
            routes=[dict(object_id="cube", target_id="exit", minimum_height=0.8)],
            hands=[
                dict(
                    grasp_target_id="eef",
                    model_id="dh_pgi_140_80",
                    envelope=dict(
                        finger_length=0.13,
                        finger_width=0.03,
                        finger_thickness=0.01,
                        palm_depth=0.08,
                        opening_margin=0.003,
                    ),
                )
            ],
        ),
        dict(kind="park"),
    ]
    base = profile["presets"][0]
    base["effect_assurance"] = "verified"
    base["effect_monitors"] = {
        "place": dict(
            monitor_id="builtin.composite_effect",
            revision="1",
            params=dict(consecutive_samples=3),
        )
    }
    base["action_options"].update(
        {
            "gen_sim.clear_released": {
                "kind": "move_end_effector",
                "linear_segments": True,
            },
            "simulation.park": {"kind": "move_joints"},
        }
    )
    base["motion"].update(
        strategy="ik_interp",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        dynamic_collision_mode="off",
        velocity_limit=None,
        acceleration_limit=None,
    )
    roles = []
    for name, rule in (
        ("release", None),
        ("clearance", "released_clearance"),
        ("park", "released_park"),
    ):
        preset = deepcopy(base)
        preset["preset_id"] = name
        preset["recovery"].pop("planning_recovery", None)
        if rule:
            preset["recovery"]["planning_recovery"] = {"kind": rule, "revision": 1}
        preset["motion"]["segment_sample_count_retries"] = (
            [260, 320] if name == "clearance" else []
        )
        roles.append(preset)
    profile["presets"] = roles
    calls = [
        dict(kind="place", object="cube", at=dict(kind="target_ref", target="home")),
        dict(
            kind="registered",
            call_id="gen_sim.clear_released",
            arguments=dict(object="cube", target="exit"),
        ),
        dict(kind="registered", call_id="simulation.park", arguments={}),
    ]
    for call in calls:
        call["resources"] = {"primary": "primary_manipulator"}
    declaration = dict(
        program_id="released",
        targets={
            "home": dict(
                kind="cyclic_pose",
                values=[dict(position=[0, 0, 0.8], quaternion_xyzw=[0, 0, 0, 1])],
            )
        },
        program=dict(
            kind="sequence",
            items=[
                dict(kind="segment", name=name, steps=dict(kind="invoke", call=call))
                for name, call in zip(("release", "clearance", "park"), calls)
            ],
        ),
    )
    selection = TaskProgramIntegrationCfg(
        scene_registry=payload["scene"]["registry_id"],
        robot_profile=profile["profile_id"],
        runtime_preset="release",
    )
    compiled = compile_configured_task_program(
        declaration, scene_binding=payload["scene"], selection=selection
    )
    ids = [
        f"{compiled.program_id}/{segment.segment_id}:0"
        for segment in compiled.iter_segments()
    ]
    fingerprint = task_program_fingerprint(compiled)
    services["invocation_policy_bindings"] = [
        dict(program_fingerprint=fingerprint, invocation_id=identity, preset_id=name)
        for identity, name in zip(ids, ("release", "clearance", "park"))
    ]
    services["planning_services"] = dict(
        kind="simulation_geometry",
        free_motion_query="discrete_mesh",
        capture_root_pose=True,
        capture_native_revision=True,
        released_hand_continuations=[
            dict(
                program_fingerprint=fingerprint,
                release_invocation_id=ids[0],
                clearance_invocation_id=ids[1],
                park_invocation_ids=[ids[2]],
            )
        ],
    )
    declaration["integration"] = dict(
        scene_registry=selection.scene_registry,
        robot_profile=selection.robot_profile,
        runtime_preset=selection.runtime_preset,
    )
    return payload, decode_task_program(declaration)


def test_released_continuation_requires_closed_exact_program_roles():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.simulation.planning_services import (
        SimulationPlanningServicesCfg,
    )

    assert (
        "released_hand_continuations"
        not in _canonical_value(SimulationPlanningServicesCfg())["fields"]
    )
    payload, program = _released_continuation_payload()
    registration = _decode_configured_task_program_integration(payload).registration
    registration.catalog.preflight(program)
    (declaration,) = registration.planning_services.released_hand_continuations
    assert declaration.park_invocation_ids[0].endswith(":0")
    for defect in (
        "fingerprint",
        "unknown",
        "release_monitor",
        "missing_binding",
        "collision_service",
    ):
        bad = deepcopy(payload)
        service = bad["runtime_services"]["planning_services"]
        if defect == "fingerprint":
            service["released_hand_continuations"][0]["program_fingerprint"] = "0" * 64
        elif defect == "unknown":
            service["released_hand_continuations"][0][
                "clearance_invocation_id"
            ] = "unknown:0"
        elif defect == "release_monitor":
            bad["robot_profile"]["presets"][0]["effect_assurance"] = "projected"
        elif defect == "missing_binding":
            service["released_hand_continuations"] = []
        elif defect == "collision_service":
            service["free_motion_query"] = "none"
        with pytest.raises((ValueError, TypeError)):
            _decode_configured_task_program_integration(
                bad
            ).registration.catalog.preflight(program)


def test_measured_rigid_configuration_keeps_exact_typed_source_permissions():
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.sim.atomic_actions import PlaceOptions

    assert (
        _canonical_value(PickUpOptions())["fields"].get("measured_preshape", False)
        is False
    )
    assert (
        _canonical_value(PlaceOptions())["fields"].get("measured_inside", False)
        is False
    )
    payload, program = _grounded_opening_payload()
    acquire = next(
        value
        for value in payload["robot_profile"]["presets"]
        if value["preset_id"] == "native_acquire"
    )
    acquire["recovery"]["planning_budget"] = {"wall_seconds": 90.0}
    acquire["action_options"]["pick"].update(
        measured_preshape=True, native_closing_revision=4
    )
    services = payload["runtime_services"]
    identity = services["invocation_policy_bindings"][0]["invocation_id"]
    services["planning_services"]["measured_rigid_bindings"] = [
        dict(invocation_id=identity, object_id="cube")
    ]
    actual = _decode_configured_task_program_integration(payload).registration
    actual.catalog.preflight(program)
    assert actual.planning_services.measured_rigid_bindings[0].invocation_id == identity
    for defect in (
        "unknown",
        "subject",
        "native8",
        "no_stamp",
        "missing_flag",
        "wrong_kind",
    ):
        damaged = deepcopy(payload)
        binding = damaged["runtime_services"]["planning_services"][
            "measured_rigid_bindings"
        ][0]
        options = next(
            value
            for value in damaged["robot_profile"]["presets"]
            if value["preset_id"] == "native_acquire"
        )["action_options"]["pick"]
        if defect == "unknown":
            binding["invocation_id"] = "unknown/missing:0"
        elif defect == "subject":
            binding["object_id"] = "support"
        elif defect == "native8":
            options["native_closing_revision"] = 8
        elif defect == "no_stamp":
            damaged["runtime_services"]["planning_services"][
                "capture_native_revision"
            ] = False
        elif defect == "missing_flag":
            options["measured_preshape"] = False
        else:
            binding["container_id"] = "not_a_container"
        with pytest.raises((TypeError, ValueError)):
            _decode_configured_task_program_integration(
                damaged
            ).registration.catalog.preflight(program)


def _effect_transport_payload(contract=None):
    payload = _tableware_integration_payload("pour_water")
    values = payload["runtime_services"]["registered_semantic_lowerers"]
    index = next(
        i for i, value in enumerate(values) if value["kind"] == "move_held_object"
    )
    values[index] = dict(
        kind="move_held_object",
        routes=[
            dict(
                object_id="bottle",
                target_id="destination",
                pose=dict(
                    kind="pose", position=[0.1, 0.2, 0.3], quaternion_xyzw=[0, 0, 0, 1]
                ),
            )
        ],
    )
    if contract is not None:
        values[index]["effect_contract"] = contract
    return payload


def test_source_terminal_attachment_keeps_full_effect_without_phase_requalification():
    from embodichain.lab.task_program.integrations.simulation.transport import (
        TerminalAttachmentTransportLowererFactory,
    )

    route = dict(
        object_id="cube",
        target_id="destination",
        pose=dict(kind="pose", position=[0.1, 0.2, 0.3], quaternion_xyzw=[0, 0, 0, 1]),
    )
    factory = _decode_registered_lowerer(
        dict(
            kind="move_held_object",
            routes=[route],
            effect_contract="terminal_attachment",
        ),
        path="service",
    )
    assert type(factory) is TerminalAttachmentTransportLowererFactory
    from embodichain.lab.task_program.integrations.simulation.transport import (
        _TerminalAttachmentTransportLowerer,
    )

    lowerer = _TerminalAttachmentTransportLowerer(factory.routes)
    call = RegisteredSemanticCall(
        call_id=factory.call_id, arguments=dict(object="cube", target="destination")
    )
    options = MoveHeldObjectOptions(report_retained_attachment=True)
    result = lowerer.lower(call, context=None, bound=None, option_template=options)
    assert result.goal.object_id is None
    assert result.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert result.registered_effect.held_objects[0].object_id == "cube"
    assert result.registered_effect.held_objects[0].retain_during_segments == ()
    assert result.registered_effect.held_objects[0].release_before_segment is None
    with pytest.raises(ValueError, match="report_retained_attachment"):
        lowerer.lower(
            call, context=None, bound=None, option_template=MoveHeldObjectOptions()
        )


@pytest.mark.parametrize(
    "kind,mode",
    [
        ("move_held_object", "terminal_release"),
        ("place_relative", "terminal_attachment"),
        ("place_relative", "guessed"),
    ],
)
def test_effect_contract_modes_reject_other_families_and_unknown_values(kind, mode):
    value = (
        dict(
            kind=kind,
            routes=[
                dict(
                    object_id="cube",
                    target_id="destination",
                    pose=dict(
                        kind="pose", position=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]
                    ),
                )
            ],
        )
        if kind == "move_held_object"
        else dict(kind=kind, routes=[])
    )
    value["effect_contract"] = mode
    with pytest.raises(ValueError, match="effect_contract"):
        _decode_registered_lowerer(value, path="service")


@pytest.mark.parametrize("damage", [None, "no_report", "projected", "no_monitor"])
def test_source_terminal_transport_preflight_requires_the_selected_full_verification(
    damage,
):
    from embodichain.lab.task_program import decode_task_program

    payload = _effect_transport_payload("terminal_attachment")
    preset = payload["robot_profile"]["presets"][0]
    preset["action_options"]["simulation.move_held_object"][
        "report_retained_attachment"
    ] = (damage != "no_report")
    preset["effect_assurance"] = "projected" if damage == "projected" else "verified"
    preset["effect_monitors"] = (
        {}
        if damage in {"projected", "no_monitor"}
        else {
            "simulation.move_held_object": dict(
                monitor_id="builtin.composite_effect", revision="1"
            )
        }
    )
    registration = _decode_configured_task_program_integration(payload).registration
    program = decode_task_program(
        dict(
            program_id="explicit_terminal_contract",
            targets={},
            integration=dict(
                scene_registry=registration.scene_binding.registry_id,
                robot_profile=registration.robot_profile_binding.profile_id,
                runtime_preset=preset["preset_id"],
            ),
            program=dict(
                kind="invoke",
                call=dict(
                    kind="registered",
                    call_id="simulation.move_held_object",
                    arguments=dict(object="bottle", target="destination"),
                ),
            ),
        )
    )
    if damage is None:
        registration.catalog.preflight(program)
    else:
        with pytest.raises(
            ValueError,
            match=(
                "report_retained_attachment"
                if damage == "no_report"
                else "verified terminal monitoring"
            ),
        ):
            registration.catalog.preflight(program)


def _prismatic_service_payload():
    from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding

    payload = _integration_payload("open_drawer")
    payload["scene"]["articulations"].append(
        {"entity_id": "cabinet_source", "simulation_uid": "cabinet_source"}
    )
    payload["scene"]["links"].append(
        {
            "entity_id": "source_handle",
            "articulation_id": "cabinet_source",
            "native_link_name": "slider",
        }
    )
    source = PrismaticBinding(
        "cabinet_source",
        "rail",
        "slider",
        "base",
        "/slider/grip",
        "a" * 64,
        1.0,
        (0.0, -1.0, 0.0),
        (0.0, 0.2),
    )
    bindings = [{"entity_id": "source_handle", "source": source.payload()}]
    services = payload["runtime_services"]
    services["registered_semantic_lowerers"] = [
        {"kind": kind, "bindings": bindings, "table_uid": "table"}
        for kind in ("prismatic_slide", "prismatic_withdraw")
    ] + [{"kind": "articulation_park"}]
    profile = payload["robot_profile"]
    profile["defaults"].update(
        move_end_effector={"primary": "primary_manipulator"},
        move_joints={"primary": "primary_manipulator"},
    )
    profile["resources"][0]["endpoints"][0]["capabilities"].append(
        "motion.joint_position"
    )
    base = deepcopy(profile["presets"][0])
    base["action_options"] = {
        "gen_sim.articulation_slide": {
            "kind": "slide",
            "release_retreat_distance": 0.04,
        },
        "gen_sim.articulation_withdraw": {"kind": "move_end_effector"},
        "gen_sim.articulation_park": {"kind": "move_joints"},
    }
    profile["presets"] = [base]
    for role, kind, count in (
        ("slide", "slide_bounded", 260),
        ("withdraw", "articulation_withdrawal", 40),
        ("park", "articulation_park", 120),
    ):
        preset = deepcopy(base)
        preset["preset_id"] = role
        preset["motion"].update(
            sample_count=count,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=[] if role == "park" else [260, 320],
        )
        preset["recovery"]["planning_recovery"] = {"kind": kind, "revision": 1}
        profile["presets"].append(preset)
    return payload


@pytest.mark.parametrize("defect", [None, "retries", "recovery", "state"])
def test_prismatic_source_contract_is_rejected_during_static_preflight(
    defect: str | None,
) -> None:
    from embodichain.lab.task_program import decode_task_program

    payload = _prismatic_service_payload()
    slide = next(
        preset
        for preset in payload["robot_profile"]["presets"]
        if preset["preset_id"] == "slide"
    )
    if defect == "retries":
        slide["motion"]["segment_sample_count_retries"] = []
    elif defect == "recovery":
        slide["recovery"]["planning_recovery"] = None
    registration = _decode_configured_task_program_integration(payload).registration
    program = decode_task_program(
        {
            "program_id": "source_preflight",
            "targets": {},
            "integration": {
                "scene_registry": registration.scene_binding.registry_id,
                "robot_profile": registration.robot_profile_binding.profile_id,
                "runtime_preset": "slide",
            },
            "program": {
                "kind": "segment",
                "name": "slide",
                "steps": {
                    "kind": "invoke",
                    "call": {
                        "kind": "registered",
                        "call_id": "gen_sim.articulation_slide",
                        "arguments": {
                            "object": "cabinet_source",
                            "state": "ajar" if defect == "state" else "open",
                        },
                    },
                },
            },
        }
    )
    if defect is None:
        fingerprint = registration.fingerprint
        compiled = registration.catalog.preflight(program)
        assert compiled.program_id == program.program_id
        registration.assert_unchanged()
        assert registration.fingerprint == fingerprint
    else:
        expected = {
            "retries": "exact fixed-grid segment recovery contract",
            "recovery": "explicit typed planning recovery",
            "state": "target state must be open or closed",
        }
        with pytest.raises(ValueError, match=expected[defect]):
            registration.catalog.preflight(program)


@pytest.mark.parametrize(
    "field", [None, "require_qualified_approach", "measured_preshape"]
)
def test_declared_nominal_planning_rejects_only_selected_source_modes(field) -> None:
    from embodichain.lab.task_program import (
        TaskProgramIntegrationCfg,
        decode_task_program,
    )
    from embodichain.lab.task_program.integrations._configured_composition import (
        _validate_configured_nominal_planning_modes,
    )
    from embodichain.lab.task_program.compiler import TaskProgramCompiler

    payload = _integration_payload("repeated_pick_place")
    preset = payload["robot_profile"]["presets"][0]
    if field is not None:
        preset["action_options"]["pick"][field] = True
    registration = _decode_configured_task_program_integration(payload).registration
    config = decode_task_program(
        dict(
            program_id="nominal_modes",
            targets={},
            integration=dict(
                scene_registry=registration.scene_binding.registry_id,
                robot_profile=registration.robot_profile_binding.profile_id,
                runtime_preset=preset["preset_id"],
            ),
            program=dict(kind="invoke", call=dict(kind="pick", object="cube")),
        )
    )
    compiled = TaskProgramCompiler(registration.catalog.scene).compile(config)
    declaration = (SimpleNamespace(call_index=0),)
    if field is None:
        _validate_configured_nominal_planning_modes(compiled, registration, declaration)
    else:
        with pytest.raises(ValueError, match="external nominal source cannot bypass"):
            _validate_configured_nominal_planning_modes(
                compiled, registration, declaration
            )
        _validate_configured_nominal_planning_modes(compiled, registration, ())


def test_cargo_service_declaration_is_closed_owned_and_fingerprinted():
    from embodichain.lab.task_program.integrations import IntegrationFingerprintMismatch

    payload = _integration_payload("repeated_pick_place")
    baseline = _decode_configured_task_program_integration(payload).registration
    actor = payload["scene"]["rigid_objects"][0]
    uid = actor.get("simulation_uid", actor["entity_id"])
    payload["runtime_services"]["cargo_envelope"] = {
        "carrier_uids": [uid],
        "excluded_uids": [],
    }
    registered = _decode_configured_task_program_integration(payload).registration
    assert registered.cargo_envelope_cfg.carrier_uids == (uid,)
    assert registered.fingerprint != baseline.fingerprint
    for declaration in (
        {"carrier_uids": ["missing"]},
        {"carrier_uids": [uid, uid]},
        {"carrier_uids": [uid], "scale": 2.0},
    ):
        invalid = deepcopy(payload)
        invalid["runtime_services"]["cargo_envelope"] = declaration
        with pytest.raises((ValueError, TypeError)):
            _decode_configured_task_program_integration(invalid)
    registered.cargo_envelope_cfg.carrier_uids = ("changed",)
    with pytest.raises(IntegrationFingerprintMismatch):
        registered.assert_unchanged()


def test_public_component_composition_preserves_task_owned_cargo_selection():
    path = _config_path("repeated_pick_place")
    config = _gym_config("repeated_pick_place")
    physical = _physical_components(path, config)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    actor = task["scene_binding"]["rigid_objects"][0]
    uid = actor.get("simulation_uid", actor["entity_id"])
    task.setdefault("runtime_services", {})["cargo_envelope"] = {"carrier_uids": [uid]}
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.cargo_envelope_cfg.carrier_uids == (uid,)


def test_recorded_contacts_are_closed_fingerprinted_and_load_without_simulator(
    monkeypatch,
):
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.task_program.integrations import IntegrationFingerprintMismatch

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration constructed simulation")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    payload = _integration_payload("repeated_pick_place")
    baseline = _decode_configured_task_program_integration(payload).registration
    declaration = dict(
        sensor_uid="gen_sim_task_spec_contacts",
        retention=dict(step_dt=0.04),
        pad_bindings=[["eef", ["pad_a", "pad_b"]]],
        opening_bindings=[
            dict(
                control_part="eef",
                joint_names=["drive"],
                driver_indices=[0],
                open_qpos=[0.8],
                tolerance=0.008,
            )
        ],
    )
    payload["runtime_services"]["recorded_contacts"] = declaration
    registration = _decode_configured_task_program_integration(payload).registration
    cfg = registration.recorded_contacts_cfg
    assert cfg.pad_bindings == (("eef", ("pad_a", "pad_b")),)
    assert cfg.opening_bindings[0].driver_indices == (0,)
    assert (
        cfg.retention.max_control_steps == 30000
        and registration.fingerprint != baseline.fingerprint
    )
    declaration["retention"]["step_dt"] = 0.2
    assert cfg.retention.step_dt == 0.04
    for error in (
        dict(retention=dict(step_dt=0.0)),
        dict(pad_bindings=[["unknown", ["a", "b"]]]),
        dict(extra="unregistered"),
    ):
        invalid = deepcopy(payload)
        invalid["runtime_services"]["recorded_contacts"].update(error)
        with pytest.raises((ValueError, TypeError)):
            _decode_configured_task_program_integration(invalid)
    with pytest.raises(ValueError, match="Default"):
        registration.validate_physics("newton")
    cfg.retention.max_contact_rows = 1
    with pytest.raises(IntegrationFingerprintMismatch):
        registration.assert_unchanged()


def test_recorded_contact_selection_composes_only_from_task_component():
    path = _config_path("repeated_pick_place")
    physical = _physical_components(path, _gym_config("repeated_pick_place"))
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    task.setdefault("runtime_services", {})["recorded_contacts"] = dict(
        sensor_uid="history",
        retention=dict(step_dt=0.04),
        pad_bindings=[["eef", ["a", "b"]]],
    )
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )
    registration = _decode_configured_task_program_integration(payload).registration
    assert registration.recorded_contacts_cfg.sensor_uid == "history"


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
