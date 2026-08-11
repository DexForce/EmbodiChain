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

"""Tests for the declarative dual-UR5 hand-over task."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.expert_program import (
    ConfiguredHandOverPoseProvider,
    ExpertProgramEnvironmentMixin,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.atomic_actions import HandOverOptions, PickUpOptions
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.skills import HandOver, Pick

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.tableware.hand_over import (  # noqa: E402
    CAN_SIMULATION_UID,
    CAN_UID,
    CAN_MASS,
    GRIPPER_MASTER_DRIVE_DAMPING,
    GRIPPER_MASTER_DRIVE_MAX_EFFORT,
    GRIPPER_MASTER_DRIVE_STIFFNESS,
    GRIPPER_GRASP_QPOS,
    GRIPPER_OPEN_QPOS,
    HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
    HAND_OVER_POSE_PROVIDER,
    HAND_OVER_ROBOT_PROFILE_ID,
    HAND_OVER_SCENE_REGISTRY_ID,
    HAND_OVER_SAMPLE_COUNT,
    SUPPORT_SURFACE_UID,
    HandOverEnv,
    _create_default_env_cfg,
    create_hand_over_robot_profile_binding,
    create_hand_over_scene_binding,
)

EXPECTED_GRIPPER_GRASP_QPOS = 0.011


def _gym_config_path() -> Path:
    """Return the installed-source dual-UR5 Gym config path."""
    return (
        Path(__file__).parents[4]
        / "embodichain_tasks/configs/gym/hand_over/dual_ur5.json"
    )


def _gym_payload() -> dict[str, object]:
    """Load the runnable HandOver Gym config as inert JSON data."""
    payload = json.loads(_gym_config_path().read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def test_registered_hand_over_task_uses_shared_expert_program_mixin() -> None:
    """The task registers one semantic environment without local demo code."""
    from embodichain_tasks.tableware import __all__

    assert "HandOverEnv" in __all__
    spec = REGISTERED_ENVS["HandOver-v1"]
    assert spec.cls is HandOverEnv
    assert spec.max_episode_steps == 1200
    assert spec.expert_program_registration is HAND_OVER_EXPERT_PROGRAM_REGISTRATION
    assert "expert_program_registration" not in spec.default_kwargs
    assert issubclass(HandOverEnv, ExpertProgramEnvironmentMixin)
    assert issubclass(HandOverEnv, EmbodiedEnv)
    assert "create_demo_action_list" not in HandOverEnv.__dict__


def test_hand_over_gym_config_selects_packaged_program_without_contact_sensor() -> None:
    """Normal startup selects the semantic program and needs no contact sensor."""
    payload = _gym_payload()

    assert payload["id"] == "HandOver-v1"
    assert payload["expert_program_path"] == (
        "../../expert_program/tableware/hand_over.yaml"
    )
    assert payload["sensor"] == []
    assert payload["env"]["extensions"] == {}
    settle = payload["env"]["events"]["settle_can_on_reset"]
    assert settle["func"] == "wait_for_dynamic_objects_to_settle"
    assert settle["params"]["entity_cfgs"] == [{"uid": CAN_SIMULATION_UID}]


def test_hand_over_gym_config_builds_dual_ur5_pgi_scene() -> None:
    """Config parsing preserves the tutorial robot, can, and support geometry."""
    path = _gym_config_path()
    cfg = config_to_cfg(_gym_payload(), source_path=path)

    assert type(cfg.robot) is RobotCfg
    assert cfg.robot.uid == "DualUR5HandOver"
    assert cfg.robot.control_parts["left_arm"] == ["left_joint[0-9]"]
    assert cfg.robot.control_parts["right_arm"] == ["right_joint[0-9]"]
    assert cfg.robot.control_parts["left_hand"] == ["left_gripper_finger1_joint_1"]
    assert cfg.robot.control_parts["right_hand"] == ["right_gripper_finger1_joint_1"]
    assert set(cfg.robot.urdf_cfg.components) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    assert cfg.robot.solver_cfg["left_arm"].ik_nearest_weight == [
        1.0,
        4.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert cfg.robot.solver_cfg["left_arm"].root_link_name == "left_base_link"
    assert cfg.robot.solver_cfg["left_arm"].end_link_name == "left_ee_link"
    assert cfg.robot.solver_cfg["right_arm"].root_link_name == "right_base_link"
    assert cfg.robot.solver_cfg["right_arm"].end_link_name == "right_ee_link"
    assert cfg.robot.solver_cfg["right_arm"].tcp[2][3] == pytest.approx(0.155)
    assert list(cfg.robot.init_qpos) == pytest.approx(
        [
            0.0,
            0.0,
            -1.57,
            -1.57,
            1.57,
            1.57,
            -1.57,
            -1.57,
            -1.57,
            -1.57,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert [item.uid for item in cfg.background] == [SUPPORT_SURFACE_UID]
    assert [item.uid for item in cfg.rigid_object] == [CAN_SIMULATION_UID]
    assert cfg.rigid_object[0].max_convex_hull_num == 1
    assert cfg.expert_program is not None
    assert cfg.expert_program.program_id == "dual_ur5_hand_over"


def test_hand_over_physics_configs_match_tuned_can_and_pgi_parameters() -> None:
    """Python and JSON configs share real can and master-only PGI dynamics."""
    direct_cfg = _create_default_env_cfg()
    json_cfg = config_to_cfg(_gym_payload(), source_path=_gym_config_path())

    expected_values = {
        "stiffness": GRIPPER_MASTER_DRIVE_STIFFNESS,
        "damping": GRIPPER_MASTER_DRIVE_DAMPING,
        "max_effort": GRIPPER_MASTER_DRIVE_MAX_EFFORT,
    }
    for cfg in (direct_cfg, json_cfg):
        assert cfg.rigid_object[0].attrs.mass == pytest.approx(CAN_MASS)
        drive = cfg.robot.drive_pros
        for property_name, master_value in expected_values.items():
            values = getattr(drive, property_name)
            for side in ("left", "right"):
                assert values[f"{side}_gripper_finger1_joint_1"] == pytest.approx(
                    master_value
                )
                assert values[f"{side}_gripper_finger2_joint_1"] == pytest.approx(0.0)


def test_hand_over_registration_owns_scene_and_pose_provider() -> None:
    """Static registration fingerprints the exact grasp and pose declarations."""
    scene = create_hand_over_scene_binding()
    grasp = scene.antipodal_grasps[0]

    assert scene.registry_id == HAND_OVER_SCENE_REGISTRY_ID
    assert [item.entity_id for item in scene.rigid_objects] == [
        CAN_UID,
        SUPPORT_SURFACE_UID,
    ]
    assert scene.rigid_objects[0].simulation_uid == CAN_SIMULATION_UID
    assert grasp.object_id == CAN_UID
    assert grasp.generator_cfg.antipodal_sampler_cfg.n_sample == 10000
    assert grasp.force_reannotate is False
    assert HAND_OVER_EXPERT_PROGRAM_REGISTRATION.scene_binding == scene
    assert HAND_OVER_EXPERT_PROGRAM_REGISTRATION.handover_pose_providers == (
        HAND_OVER_POSE_PROVIDER,
    )
    assert (
        ConfiguredHandOverPoseProvider.provider_id
        == "simulation.configured_handover_pose"
    )
    assert HAND_OVER_POSE_PROVIDER.middle_position == pytest.approx((0.0, 0.0, 0.7))
    assert HAND_OVER_POSE_PROVIDER.final_position == pytest.approx((0.0, -0.2, 0.7))


def test_hand_over_profile_binds_left_pick_and_left_to_right_transfer() -> None:
    """The profile selects both participants and its tuned motion policy."""
    binding = create_hand_over_robot_profile_binding()

    assert binding.profile_id == HAND_OVER_ROBOT_PROFILE_ID
    assert [resource.resource_id for resource in binding.resources] == [
        "left",
        "right",
    ]
    assert [
        endpoint.control_part
        for resource in binding.resources
        for endpoint in resource.endpoints
    ] == ["left_arm", "left_hand", "right_arm", "right_hand"]
    assert dict(binding.defaults["pick_up"]) == {"primary": "left"}
    assert dict(binding.defaults["hand_over"]) == {
        "source": "left",
        "destination": "right",
    }
    assert binding.presets[0].preset_id == "safe"
    assert binding.presets[0].motion_policy.sample_count == HAND_OVER_SAMPLE_COUNT
    assert binding.presets[0].workflow_recovery_policy.max_recovery_attempts == 2
    assert binding.presets[0].runner_cfg.hold_during_effect_verification is False
    assert binding.presets[0].runner_cfg.hold_on_completion is False
    templates = binding.presets[0].action_option_templates
    pick_options = templates["pick"]
    assert type(pick_options) is PickUpOptions
    assert pick_options.pick_object_part == "top"
    assert pick_options.pre_grasp_distance == pytest.approx(0.08)
    assert pick_options.lift_height == pytest.approx(0.10)
    assert pick_options.hand_interp_steps == 5
    torch.testing.assert_close(
        pick_options.approach_direction,
        torch.tensor([0.0, -0.7071067812, -0.7071067812]),
    )
    hand_over_options = templates["hand_over"]
    assert type(hand_over_options) is HandOverOptions
    assert hand_over_options.receive_pick_object_part == "bottom"
    assert hand_over_options.pre_grasp_distance == pytest.approx(0.08)
    assert hand_over_options.lift_height == pytest.approx(0.08)
    assert hand_over_options.hand_interp_steps == 10
    assert hand_over_options.hold_steps == 4
    assert hand_over_options.retreat_steps == 28
    torch.testing.assert_close(
        hand_over_options.receive_approach_direction,
        torch.tensor([0.0, 0.7071067812, -0.7071067812]),
    )
    assert dict(binding.grounding_providers) == {
        "hand_over": ConfiguredHandOverPoseProvider.provider_id,
    }
    for side, preset in zip(("left", "right"), binding.command_presets):
        assert preset.control_part == f"{side}_hand"
        assert tuple(preset.commands["open"]) == (GRIPPER_OPEN_QPOS,)
        assert tuple(preset.commands["grasp"]) == (GRIPPER_GRASP_QPOS,)
        assert tuple(preset.commands["grasp"]) == pytest.approx(
            (EXPECTED_GRIPPER_GRASP_QPOS,)
        )


def test_direct_default_cfg_loads_the_registered_semantic_program() -> None:
    """Direct construction and JSON startup select the same registration IDs."""
    cfg = _create_default_env_cfg()

    assert type(cfg.robot) is RobotCfg
    assert cfg.sensor == []
    assert cfg.expert_program is not None
    assert cfg.expert_program.integration.scene_registry == HAND_OVER_SCENE_REGISTRY_ID
    assert cfg.expert_program.integration.robot_profile == HAND_OVER_ROBOT_PROFILE_ID
    assert cfg.expert_program.integration.runtime_preset == "safe"
    settle = cfg.events["settle_can_on_reset"]
    assert settle.params["entity_cfgs"][0].uid == CAN_SIMULATION_UID


def test_task_initialization_passes_only_registration_to_shared_factory(
    monkeypatch,
) -> None:
    """Task setup has no provider side channel or local motion generator."""
    adapter = object()
    captured: dict[str, object] = {}

    def fake_base_init(self, cfg, **kwargs) -> None:
        del self, cfg, kwargs

    def fake_create_adapter(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    task_module = importlib.import_module(HandOverEnv.__module__)
    monkeypatch.setattr(
        task_module,
        "create_simulation_expert_program_adapter",
        fake_create_adapter,
    )

    env = HandOverEnv(cfg=object())

    assert env.expert_program_adapter is adapter
    assert captured == {
        "environment": env,
        "registration": HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
    }


def test_task_config_compiles_through_real_simulation_factory(monkeypatch) -> None:
    """The packaged program reaches the real adapter through explicit mocks."""

    class FakeRobot:
        uid = "DualUR5HandOver"

        @staticmethod
        def get_qpos(*, target: bool = False) -> torch.Tensor:
            del target
            return torch.zeros((1, 16), dtype=torch.float32)

    class FakeRigidObject:
        def __init__(self, *, is_non_dynamic: bool) -> None:
            self.is_non_dynamic = is_non_dynamic

    robot = FakeRobot()
    can = FakeRigidObject(is_non_dynamic=False)
    support = FakeRigidObject(is_non_dynamic=True)

    class FakeSimulation:
        @staticmethod
        def get_robot(uid: str):
            return robot if uid == robot.uid else None

        @staticmethod
        def get_rigid_object(uid: str):
            return {
                CAN_SIMULATION_UID: can,
                SUPPORT_SURFACE_UID: support,
            }.get(uid)

    def fake_base_init(self, cfg, **kwargs) -> None:
        del kwargs
        self.cfg = cfg
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.sim = FakeSimulation()
        self.robot = robot

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    cfg = config_to_cfg(_gym_payload(), source_path=_gym_config_path())

    env = HandOverEnv(cfg=cfg)
    segments = tuple(env.compile_expert_program(cfg.expert_program))

    assert len(segments) == 1
    assert segments[0].name == "hand_over_can"
    assert [type(call.call) for call in segments[0].calls] == [Pick, HandOver]
    assert len(segments[0].post_policies) == 1
    assert len(segments[0].validators) == 1
    assert env.expert_program_adapter.scene_registry_id == (HAND_OVER_SCENE_REGISTRY_ID)
    assert env.expert_program_adapter.robot_profile_id == (HAND_OVER_ROBOT_PROFILE_ID)


@pytest.mark.requires_sim
@pytest.mark.slow
def test_real_sim_expert_episode_transfers_can_with_effect_and_validation_trace() -> (
    None
):
    """The full semantic episode proves transfer effects, settling, and validation."""
    import gc

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    cfg = config_to_cfg(_gym_payload(), source_path=_gym_config_path())
    cfg.num_envs = 1
    cfg.sim_cfg = SimulationManagerCfg(
        headless=True,
        sim_device="cpu",
        num_envs=1,
    )
    cfg.sensor = []
    cfg.observations = None
    cfg.dataset = None
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    cfg.filter_dataset_saving = True

    env: HandOverEnv | None = None
    try:
        env = HandOverEnv(cfg=cfg)
        env.reset(seed=0)
        can = env.sim.get_rigid_object(CAN_SIMULATION_UID)
        assert can is not None
        initial_can_pose = can.get_local_pose(to_matrix=True).tolist()
        initial_left_eef = env.robot.compute_fk(
            env.robot.get_qpos(name="left_arm"),
            name="left_arm",
            to_matrix=True,
        ).tolist()
        initial_qpos = env.robot.get_qpos().tolist()

        result = execute_demo_episode(env)

        if not result.completed:
            runtime = result.segments[0].metadata["runtime"]
            failed_call = runtime["calls"][-1]
            last_effect = (
                None if not failed_call["effects"] else failed_call["effects"][-1]
            )
            pytest.fail(
                json.dumps(
                    {
                        "terminal_reason": result.terminal_reason,
                        "initial_can_pose": initial_can_pose,
                        "initial_left_eef": initial_left_eef,
                        "initial_qpos": initial_qpos,
                        "final_can_pose": can.get_local_pose(to_matrix=True).tolist(),
                        "final_left_eef": env.robot.compute_fk(
                            env.robot.get_qpos(name="left_arm"),
                            name="left_arm",
                            to_matrix=True,
                        ).tolist(),
                        "final_left_hand_qpos": env.robot.get_qpos(
                            name="left_hand"
                        ).tolist(),
                        "events": [
                            {
                                "kind": event["kind"],
                                "timestamp": event["timestamp"],
                                "message": event["message"],
                            }
                            for event in runtime["events"]
                        ],
                        "plan_success_masks": [
                            attempt["plan_success_mask"]
                            for attempt in failed_call["plan_attempts"]
                        ],
                        "last_effect": last_effect,
                        "post_policies": result.segments[0].metadata["post_policies"],
                        "validation": result.segments[0].metadata["validation"],
                    },
                    sort_keys=True,
                ),
                pytrace=False,
            )
        assert result.all_success
        assert result.terminal_reason == "success"
        assert len(result.segments) == 1
        segment = result.segments[0]
        assert segment.name == "hand_over_can"
        assert segment.success

        metadata = segment.metadata
        runtime = metadata["runtime"]
        assert runtime["kind"] == "skill_result"
        assert runtime["status"] == "completed"
        assert runtime["masks"]["success"] == [True]
        assert [call["semantic_id"] for call in runtime["calls"]] == [
            "pick",
            "hand_over",
        ]
        for call in runtime["calls"]:
            assert call["status"] == "completed"
            assert call["masks"] == {
                "entered": [True],
                "completed": [True],
                "failed": [False],
            }
            assert call["plan_attempts"]
            assert call["plan_attempts"][-1]["plan_success_mask"] == [True]
            assert call["effects"]
            assert call["effects"][-1]["decision"] == {
                "success_mask": [True],
                "failure_mask": [False],
            }

        pick_effect = runtime["calls"][0]["effects"][-1]
        assert pick_effect["effect_spec"]["semantic_id"] == "pick"
        assert set(pick_effect["evidence"]) == {
            "destination.pose",
            "destination.constraint",
        }
        assert pick_effect["evidence"]["destination.constraint"]["values"] == [True]

        transfer_effect = runtime["calls"][1]["effects"][-1]
        assert transfer_effect["effect_spec"]["semantic_id"] == "hand_over"
        assert set(transfer_effect["evidence"]) == {
            "source.pose",
            "source.constraint",
            "destination.pose",
            "destination.constraint",
        }
        for evidence in transfer_effect["evidence"].values():
            assert evidence["valid_mask"] == [True]
            assert evidence["acquisition_errors"] == [None]
            assert evidence["env_ids"] == [0]
        assert transfer_effect["evidence"]["source.constraint"]["values"] == [False]
        assert transfer_effect["evidence"]["destination.constraint"]["values"] == [True]

        post_policies = metadata["post_policies"]
        assert len(post_policies) == 1
        assert post_policies[0]["kind"] == "wait_stable"
        assert post_policies[0]["result_mask"] == [True]
        assert post_policies[0]["result"]["status"] == "settled"
        assert post_policies[0]["result"]["state"]["settled_mask"] == [True]
        assert post_policies[0]["result"]["state"]["timeout_mask"] == [False]

        validation = metadata["validation"]
        assert validation["runtime_success_mask"] == [True]
        assert validation["eligible_mask_before_validation"] == [True]
        assert validation["post_policy_success_mask"] == [True]
        assert validation["accepted_mask"] == [True]
        assert len(validation["validators"]) == 1
        validator = validation["validators"][0]
        assert validator["kind"] == "object_near_target"
        assert validator["result_mask"] == [True]
        assert validator["result"]["accepted_mask"] == [True]
        assert validator["result"]["position_tolerance"] == pytest.approx(0.12)
        assert validator["result"]["position_error"][0] <= 0.12
    finally:
        if env is not None:
            env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


__all__: list[str] = []
