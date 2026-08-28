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

"""Tests for the configuration-defined dual-UR5 hand-over task."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.expert_program import (
    ConfiguredHandOverPoseProvider,
    ObjectNearTargetValidatorCfg,
    SegmentCfg,
)
from embodichain.lab.gym.envs.expert_program._configured_runtime_services import (
    _JointPositionConstraintObserver,
)
from embodichain.lab.gym.envs.expert_program.configured_runtime import (
    _decode_configured_expert_program_runtime,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.sim.atomic_actions import HandOverOptions
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.skills import (
    BinaryEffectClause,
    BinaryEffectEvidenceQuery,
    BinaryEvidenceKind,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    EffectEvidenceCollectionContext,
    EffectEvidenceSourceRef,
    HandOver,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)

_ENV_ID = "HandOver-v1"
_CAN_ID = "can"
_CAN_SIMULATION_UID = "handover_object"
_SUPPORT_SURFACE_UID = "support_surface"
_SCENE_ID = "dual_ur5_handover_v1"
_PROFILE_ID = "dual_ur5_handover_v1"
_OPEN_QPOS = 0.0
_GRASP_QPOS = 0.04
_CONSTRAINT_QPOS_THRESHOLD = 0.004
_PRODUCTION_POSITION_TOLERANCE = 0.12
_REPOSITORY_ROOT = Path(__file__).parents[4]
_SUBPROCESS_TIMEOUT_SECONDS = 180
_RUN_REAL_SIM_EPISODE = (
    "import json, os, runpy, sys; "
    "from pathlib import Path; "
    "module = runpy.run_path(sys.argv[1]); "
    "metadata = module['_run_real_sim_expert_episode'](); "
    "Path(sys.argv[2]).write_text(json.dumps(metadata), encoding='utf-8'); "
    "sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
)


def _gym_config_path() -> Path:
    """Return the installed-source dual-UR5 Gym config path."""
    return (
        Path(__file__).parents[4]
        / "embodichain_tasks/configs/tasks/manipulation/hand_over/env.json"
    )


def _gym_payload() -> dict[str, object]:
    """Load the runnable HandOver Gym config as inert JSON data."""
    payload = json.loads(_gym_config_path().read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def _runtime():
    """Decode a fresh HandOver runtime from the packaged Gym config."""
    payload = _gym_payload()["expert_program_runtime"]
    assert type(payload) is dict
    return _decode_configured_expert_program_runtime(payload)


def _configured_env_cfg():
    """Parse the production config and ensure its runtime ID is registered."""
    return config_to_cfg(_gym_payload(), source_path=_gym_config_path())


def test_hand_over_registers_plain_embodied_env_from_config() -> None:
    """The runnable ID needs no task module or environment subclass."""
    cfg = _configured_env_cfg()
    spec = REGISTERED_ENVS[_ENV_ID]

    from embodichain_tasks.manipulation import __all__

    assert __all__ == []
    assert importlib.util.find_spec("embodichain_tasks.manipulation.hand_over") is None
    assert spec.cls is EmbodiedEnv
    assert spec.max_episode_steps == 1200
    assert spec.expert_program_registration is not None
    assert spec.expert_program_adapter_factory is not None
    assert spec.expert_program_adapter_factory.registration is (
        spec.expert_program_registration
    )
    assert cfg.expert_program is not None


def test_hand_over_gym_config_selects_packaged_program_without_contact_sensor() -> None:
    """Normal startup selects the semantic program and needs no contact sensor."""
    payload = _gym_payload()

    assert payload["id"] == _ENV_ID
    assert payload["expert_program_path"] == "expert/program.yaml"
    assert payload["sensor"] == []
    assert payload["env"]["extensions"] == {}
    settle = payload["env"]["events"]["settle_can_on_reset"]
    assert settle["func"] == "wait_for_dynamic_objects_to_settle"
    assert settle["params"]["entity_cfgs"] == [{"uid": _CAN_SIMULATION_UID}]


def test_hand_over_gym_config_builds_dual_ur5_pgi_scene() -> None:
    """Config parsing preserves the tutorial robot, can, and support geometry."""
    cfg = _configured_env_cfg()

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
    assert [item.uid for item in cfg.background] == [_SUPPORT_SURFACE_UID]
    assert [item.uid for item in cfg.rigid_object] == [_CAN_SIMULATION_UID]
    assert cfg.rigid_object[0].max_convex_hull_num == 16
    assert cfg.expert_program is not None
    assert cfg.expert_program.program_id == "dual_ur5_hand_over"


def test_hand_over_config_owns_tuned_can_and_pgi_physics() -> None:
    """The sole config source retains the tuned object and gripper dynamics."""
    cfg = _configured_env_cfg()

    assert cfg.rigid_object[0].attrs.mass == pytest.approx(0.33)
    drive = cfg.robot.drive_pros
    expected_values = {
        "stiffness": 1e3,
        "damping": 1e2,
        "max_effort": 1e4,
    }
    for property_name, master_value in expected_values.items():
        values = getattr(drive, property_name)
        for side in ("left", "right"):
            assert values[f"{side}_gripper_finger1_joint_1"] == pytest.approx(
                master_value
            )
            assert values[f"{side}_gripper_finger2_joint_1"] == pytest.approx(0.0)
    finger_attrs = cfg.robot.link_attrs["gripper_fingers"].attrs
    assert finger_attrs.dynamic_friction == pytest.approx(2.0)
    assert finger_attrs.static_friction == pytest.approx(2.0)


def test_hand_over_runtime_owns_scene_pose_and_evidence_services() -> None:
    """The generic runtime composes all HandOver-specific declarations."""
    registration = _runtime().registration
    scene = registration.scene_binding
    grasp = scene.antipodal_grasps[0]

    assert scene.registry_id == _SCENE_ID
    assert [item.entity_id for item in scene.rigid_objects] == [
        _CAN_ID,
        _SUPPORT_SURFACE_UID,
    ]
    assert scene.rigid_objects[0].simulation_uid == _CAN_SIMULATION_UID
    assert grasp.object_id == _CAN_ID
    assert grasp.native_name == "can_antipodal_grasp"
    assert grasp.revision == "1"
    assert len(registration.handover_pose_providers) == 1
    provider = registration.handover_pose_providers[0]
    assert type(provider) is ConfiguredHandOverPoseProvider
    assert provider.final_position == pytest.approx((0.0, -0.2, 0.6))

    declaration = registration.catalog.control_part_evidence_declaration
    assert declaration is not None
    assert declaration.provider_id == CONTROL_PART_EVIDENCE_PROVIDER_ID
    assert declaration.revision == CONTROL_PART_EVIDENCE_PROVIDER_REVISION


def test_hand_over_joint_position_evidence_uses_measured_aperture() -> None:
    """Configured constraint evidence reads only the requested environment rows."""

    class FakeRobot:
        @staticmethod
        def get_qpos(*, name: str) -> torch.Tensor:
            assert name == "left_hand"
            return torch.tensor(
                [
                    [_OPEN_QPOS],
                    [_CONSTRAINT_QPOS_THRESHOLD - 1.0e-4],
                    [_CONSTRAINT_QPOS_THRESHOLD + 1.0e-4],
                ],
                dtype=torch.float32,
            )

    expectation = HeldObjectStateExpectation(
        expectation_id="source",
        relation=HeldObjectRelation.ATTACHED,
        object_id=_CAN_ID,
        slot_id="source",
        resource_id="left",
        task_state_key="left",
    )
    query = BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            clause_id="source.constraint",
            expectation_id="source",
            source=EffectEvidenceSourceRef(
                CONTROL_PART_EVIDENCE_PROVIDER_ID,
                CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
                ControlPartEvidenceAddress(
                    "left_hand",
                    CONSTRAINT_EFFECT_CHANNEL,
                ),
            ),
            evidence_kind=BinaryEvidenceKind.CONSTRAINT,
            expected=True,
        ),
        expectation,
    )
    context = EffectEvidenceCollectionContext(
        timestamp=0.1,
        observation_revision=2,
        env_ids=torch.tensor([2, 0], dtype=torch.long),
    )
    observer = _JointPositionConstraintObserver(
        FakeRobot(),
        control_parts=("left_hand", "right_hand"),
        object_ids=(_CAN_ID,),
        open_qpos=(_OPEN_QPOS,),
        minimum_displacement=_CONSTRAINT_QPOS_THRESHOLD,
    )

    observation = observer(query, context)

    assert observation.values.tolist() == [True, False]
    assert observation.valid is not None
    assert observation.valid.tolist() == [True, True]


def test_hand_over_profile_binds_unified_left_to_right_transfer() -> None:
    """The decoded profile retains both participants and its tuned policy."""
    binding = _runtime().registration.robot_profile_binding

    assert binding.profile_id == _PROFILE_ID
    assert [resource.resource_id for resource in binding.resources] == [
        "left",
        "right",
    ]
    assert [
        endpoint.control_part
        for resource in binding.resources
        for endpoint in resource.endpoints
    ] == ["left_arm", "left_hand", "right_arm", "right_hand"]
    assert dict(binding.defaults["hand_over"]) == {
        "source": "left",
        "destination": "right",
    }
    preset = binding.presets[0]
    assert preset.preset_id == "safe"
    assert preset.motion_policy.strategy == "motion_gen"
    assert preset.motion_policy.sample_count == 140
    assert preset.tracking_policy.in_flight is not None
    assert preset.tracking_policy.in_flight.metrics[0].tolerance == pytest.approx(1.0)
    assert preset.tracking_policy.terminal.metrics[0].tolerance == pytest.approx(1.0)
    assert preset.recovery_policy.max_action_retries == 0
    assert preset.recovery_policy.goal_rotation_threshold == pytest.approx(0.5)
    assert preset.workflow_recovery_policy.max_recovery_attempts == 2
    assert preset.runner_cfg.minimum_cycle_time == pytest.approx(0.04)
    assert preset.runner_cfg.hold_during_effect_verification is False
    assert preset.runner_cfg.hold_on_completion is False
    assert preset.effect_monitors["hand_over"].params["consecutive_samples"] == 10
    hand_over_options = preset.action_option_templates["hand_over"]
    assert type(hand_over_options) is HandOverOptions
    assert hand_over_options.pre_grasp_distance == pytest.approx(0.08)
    assert hand_over_options.lift_height == pytest.approx(0.08)
    assert hand_over_options.hand_interp_steps == 10
    assert dict(binding.grounding_providers) == {
        "hand_over": ConfiguredHandOverPoseProvider.provider_id,
    }
    for side, command_preset in zip(("left", "right"), binding.command_presets):
        assert command_preset.control_part == f"{side}_hand"
        assert tuple(command_preset.commands["open"]) == (_OPEN_QPOS,)
        assert tuple(command_preset.commands["grasp"]) == (_GRASP_QPOS,)


def test_hand_over_program_preflights_against_configured_registration() -> None:
    """The packaged semantic program statically links through the generic runtime."""
    cfg = _configured_env_cfg()
    assert cfg.expert_program is not None
    registration = _runtime().registration

    compiled = registration.catalog.preflight(cfg.expert_program)
    segments = tuple(compiled)

    assert len(segments) == 1
    assert segments[0].name == "hand_over_can"
    assert [type(call.call) for call in segments[0].calls] == [HandOver]
    hand_over = segments[0].calls[0].call
    assert hand_over.final_target is not None
    pose_provider = registration.handover_pose_providers[0]
    assert hand_over.final_target.position.tolist() == pytest.approx(
        pose_provider.final_position
    )
    assert len(segments[0].post_policies) == 1
    assert len(segments[0].validators) == 1


def _initialize_child_sim_engine() -> None:
    """Match the real-simulation pytest fixture's explicit engine setup."""
    from embodichain.lab.sim import cfg as sim_cfg

    import dexsim
    import dexsim.types

    sim_cfg.DEFAULT_RENDERER = "hybrid"
    if dexsim.get_world_num() != 0:
        return

    engine_cfg = dexsim.WorldConfig()
    engine_cfg.renderer = dexsim.types.Renderer.HYBRID
    engine_cfg.backend = dexsim.types.Backend.VULKAN
    engine_cfg.open_windows = False
    dexsim.init_sim_engine(engine_cfg)


def _run_real_sim_expert_episode() -> dict[str, object]:
    """Run one production hand-over episode and return JSON-safe metadata.

    This helper is intentionally invoked by the child process below. DexSim can
    abort during interpreter teardown after a successful in-process episode.
    """
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    _initialize_child_sim_engine()
    cfg = _configured_env_cfg()
    cfg.seed = 0
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
    assert cfg.expert_program is not None
    program = cfg.expert_program.program
    assert type(program) is SegmentCfg
    assert len(program.validators) == 1
    validator = program.validators[0]
    assert type(validator) is ObjectNearTargetValidatorCfg
    assert validator.position_tolerance == pytest.approx(_PRODUCTION_POSITION_TOLERANCE)

    env: EmbodiedEnv | None = None
    try:
        env = REGISTERED_ENVS[_ENV_ID].make(cfg=cfg)
        env.reset(seed=0)
        return execute_demo_episode(env).to_metadata()
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.slow
def test_real_sim_expert_episode_reports_configured_runtime_and_validation(
    tmp_path: Path,
) -> None:
    """Validate the production hand-over runtime and final validation outcome.

    The physical transfer can land on either side of the configured position
    tolerance across supported GPU/physics backends. The semantic runtime and
    the validator's accounting must remain consistent in both outcomes.
    """
    metadata_path = tmp_path / "hand_over_episode.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_REAL_SIM_EPISODE,
            str(Path(__file__).resolve()),
            str(metadata_path),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    episode = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert type(episode) is dict
    segments = episode["segments"]
    assert type(segments) is list
    assert len(segments) == 1
    segment = segments[0]
    assert segment["name"] == "hand_over_can"
    metadata = segment["metadata"]
    runtime = metadata["runtime"]
    assert runtime["kind"] == "skill_result"
    assert runtime["status"] == "completed"
    assert [call["semantic_id"] for call in runtime["calls"]] == ["hand_over"]
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
        decision = call["effects"][-1]["decision"]
        assert decision["success_mask"] == [True]
        assert decision["failure_mask"] == [False]
        assert {
            expectation["expectation_id"]
            for expectation in decision["expectations"]
            if expectation["satisfied_mask"] == [True]
        } == {"source", "destination"}

    transfer_effect = runtime["calls"][0]["effects"][-1]
    assert transfer_effect["effect_spec"]["semantic_id"] == "hand_over"
    assert set(transfer_effect["evidence"]) == {
        "source.constraint",
        "destination.constraint",
    }
    for evidence in transfer_effect["evidence"].values():
        assert evidence["valid_mask"] == [True]
        assert evidence["acquisition_errors"] == [None]
        assert evidence["env_ids"] == [0]
    assert transfer_effect["evidence"]["source.constraint"]["values"] == [False]
    assert transfer_effect["evidence"]["destination.constraint"]["values"] == [False]

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
    assert len(validation["validators"]) == 1
    validator = validation["validators"][0]
    assert validator["kind"] == "object_near_target"
    result = validator["result"]
    tolerance = result["position_tolerance"]
    assert result["accepted_mask"] in ([True], [False])
    accepted = result["accepted_mask"] == [True]
    assert tolerance == pytest.approx(_PRODUCTION_POSITION_TOLERANCE)
    assert validator["result_mask"] == [accepted]
    assert validation["accepted_mask"] == [accepted]
    assert episode["completed"] is accepted
    assert episode["success"] == [accepted]
    assert segment["success"] is accepted
    assert episode["terminal_reason"] == (
        "success" if accepted else "segment_validation_failed"
    )
    if accepted:
        assert result["position_error"][0] <= tolerance
    else:
        assert result["position_error"][0] > tolerance


__all__: list[str] = []
