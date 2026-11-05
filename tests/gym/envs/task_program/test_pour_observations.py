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

"""Tests for recorded Pour geometry from canonical observation history."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensordict import TensorDict
from embodichain.lab.gym.envs.managers import ObservationManager
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationFrame,
    ObservationSample,
)
from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout, _write_raw_rollout
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    EnvironmentStepClock,
    BufferedGymCommandSink,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations import (
    SimulationTaskProgramRegistration,
    SimulationRobotSkillProfileBinding,
    ControlPartResourceBinding,
    ControlPartEndpointBinding,
)
from embodichain.lab.task_program.integrations.simulation.bindings import (
    SimulationSceneBinding,
    SimulationRigidObjectBinding,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
)
from embodichain.lab.task_program.integrations.simulation.observations import (
    SimulationStateObservationCollector,
)
from embodichain.lab.task_program.semantics import SkillPolicyPreset, EffectAssurance
from embodichain.lab.sim.atomic_actions import CARTESIAN_POSE_CAPABILITY

_SOURCE = json.loads(
    (Path(__file__).parent / "data/pour_observation_source_reference.json").read_text()
)


def _collector_case(
    name="nominal", *, count=1, robot_qpos=None, joint_valid=True, joint_timestamp=True
):
    case = next(item for item in _SOURCE["cases"] if item["name"] == name)
    common = _SOURCE["common"]
    bodies = {}

    def no_live_pose(*args, **kwargs):
        raise AssertionError("Historical Pour geometry cannot read live poses.")

    for entity, mesh in common["meshes"].items():
        vertices = torch.tensor(mesh["vertices"], dtype=torch.float32).repeat(
            count, 1, 1
        )
        triangles = torch.tensor(mesh["triangles"], dtype=torch.long).repeat(
            count, 1, 1
        )
        bodies[mesh["native_uid"]] = SimpleNamespace(
            uid=mesh["native_uid"],
            get_vertices=lambda scale=True, vertices=vertices: vertices,
            get_triangles=lambda triangles=triangles: triangles,
            get_local_pose=no_live_pose,
        )
    sim = SimpleNamespace(
        num_envs=count,
        simulation_time=0.1,
        get_rigid_object=bodies.get,
        get_articulation=lambda uid: None,
    )
    robot = SimpleNamespace(get_qpos=lambda target=False: torch.zeros(count, 1))
    if robot_qpos is not None:
        robot.uid = "robot"
        robot.device = robot_qpos.device
        robot.cfg = SimpleNamespace(root_props=SimpleNamespace(fixed_base=True))
    scene = SimulationSceneBinding(
        registry_id="recorded_scene",
        rigid_objects=tuple(
            SimulationRigidObjectBinding(
                entity_id=entity, simulation_uid=mesh["native_uid"]
            )
            for entity, mesh in common["meshes"].items()
        ),
    )
    profile = SimulationRobotSkillProfileBinding(
        "robot",
        resources=(
            ControlPartResourceBinding(
                resource_id="primary",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                ),
            ),
        ),
        presets=(
            SkillPolicyPreset(
                "default",
                action_option_templates={},
                effect_assurance=EffectAssurance.PROJECTED,
            ),
        ),
        default_preset="default",
    )
    registration = SimulationTaskProgramRegistration(scene, profile)
    compiled = TaskProgramCompiler(scene.declare()).compile(
        decode_task_program(
            dict(
                program_id=common["program"]["program_id"],
                targets={},
                integration=dict(
                    robot_profile="robot",
                    scene_registry="recorded_scene",
                    runtime_preset="default",
                ),
                program=dict(
                    kind="segment",
                    name="observe",
                    steps=dict(kind="invoke", call=dict(kind="pick", object="cube")),
                ),
            )
        )
    )
    clock = EnvironmentStepClock(0.04)
    template = SimulationSegmentPolicyPort(
        sim, robot, scene, settle_presets=registration.settle_presets, step_dt=0.04
    )
    encoder = RuntimeCommandFrameEncoder(
        SimpleNamespace(current_qpos=lambda ids: torch.zeros(len(ids), 1))
    )
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        BufferedGymCommandSink(encoder, clock),
        clock,
        post_policy_port=template,
        isolate_policy_sessions=True,
    )
    episodes = EpisodeContext(count)
    episodes._begin_episode(torch.arange(count))
    episodes._finish_initialization(torch.arange(count))
    poses = deepcopy(common["poses"])
    patch = case["source_pose_window"]
    start = patch["start_frame"]
    poses[patch["native_uid"]][0][start : start + len(patch["values"])] = patch[
        "values"
    ]
    states = {
        uid: {"pose": torch.tensor(values, dtype=torch.float32).repeat(count, 1, 1)}
        for uid, values in poses.items()
    }
    env = SimpleNamespace(
        sim=sim,
        robot=robot,
        num_envs=count,
        episode_context=episodes,
        _closed=False,
    )
    recording = ObservationManager({}, env)
    storage = TensorDict({}, batch_size=[count])
    horizon = max(value["pose"].shape[1] for value in states.values()) - 1
    _init_raw_rollout(storage, episodes, {}, torch.zeros(count, horizon, 1))
    recording.bind_rollout_storage(storage)
    for slot in range(common["written_steps"][0] + 1):
        fields = {
            uid: ObservationSample(
                ObservationDescriptor(
                    uid,
                    "pose",
                    "arena",
                    "xyz_xyzw",
                    ("m", "unitless"),
                    (7,),
                    torch.float32,
                ),
                value["pose"][:, slot],
                torch.ones(count, dtype=torch.bool),
                (None,) * count,
            )
            for uid, value in states.items()
        }
        if robot_qpos is not None:
            fields["robot_qpos"] = ObservationSample(
                ObservationDescriptor(
                    robot.uid,
                    "qpos",
                    "joint",
                    "joint_order",
                    ("joint_units",),
                    (robot_qpos.shape[2],),
                    robot_qpos.dtype,
                ),
                robot_qpos[:, slot],
                torch.full((count,), joint_valid, dtype=torch.bool),
                (
                    (None,) * count
                    if joint_valid
                    else ("joint acquisition unavailable",) * count
                ),
            )
        frame = ObservationFrame(
            None,
            fields,
            episodes.episode_ids,
            slot,
            timestamp=slot * clock.step_dt if joint_timestamp or slot != 1 else None,
            clock_domain="simulation_time",
            episode_owner=episodes,
        )
        recording._bind_schema(frame)
        _write_raw_rollout(
            storage,
            episodes,
            frame,
            **(
                {
                    "field_ids": tuple(recording.field_ids.values()),
                    "env_ids": list(range(count)),
                }
                if slot == 0
                else {"action": torch.zeros(count, 1)}
            ),
        )
    env.observation_manager = recording
    env._test_raw_storage = storage
    collector = SimulationStateObservationCollector(
        env, bridge, registration, observations=recording
    )
    for _ in range(common["written_steps"][0]):
        clock.advance_after_env_step()
    sim.simulation_time = 0.1 + clock.now()
    return collector, env, case, bodies, compiled


@pytest.mark.parametrize("name", [item["name"] for item in _SOURCE["cases"]])
def test_recorded_pour_geometry_reproduces_frozen_b0_measurements_without_live_poses(
    name,
):
    collector, env, case, _, _compiled = _collector_case(name)
    expected = case["record"]["measurements"]
    observed = collector.capture_recorded_pour_geometry(
        "cube",
        "region",
        start_frames=expected["start_frames"],
        end_frames=expected["end_frames"],
    )
    frame = observed["geometry_frame"]
    values = frame["batches"]["pour.motion_geometry"]["values"][0]
    row = expected["rows"][0]
    assert values == [
        row["maximum_tilt_rad"],
        row["restore_error_rad"],
        *row["anchor_world_m"],
        row["receiver_projection_distance_m"],
        row["receiver_top_clearance_m"],
    ]
    assert observed["peak_frames"] == expected["peak_frames"]
    assert observed["sample_counts"] == [
        expected["end_frames"][0] - expected["start_frames"][0] + 1
    ]
    assert frame["timestamp"] == env.sim.simulation_time
    assert frame["clock_domain"] == "simulation_time"
    assert observed["recording_clock"] == "environment_step"
    assert "accepted" not in observed and "released" not in observed
    assert all(
        batch["valid"] == [True] and batch["acquisition_errors"] == [None]
        for batch in frame["batches"].values()
    )


def test_pour_capture_keeps_native_rows_and_rejects_missing_windows_or_rebound_owners():
    collector, env, case, bodies, _compiled = _collector_case(count=2)
    metrics = case["record"]["measurements"]
    starts = metrics["start_frames"] * 2
    ends = metrics["end_frames"] * 2
    observed = collector.capture_recorded_pour_geometry(
        "cube", "region", start_frames=starts, end_frames=ends
    )
    assert observed["geometry_frame"]["env_ids"] == [0, 1]
    assert observed["episode_ids"] == [0, 0]
    assert (
        observed["geometry_frame"]["batches"]["pour.motion_geometry"]["values"][0]
        == observed["geometry_frame"]["batches"]["pour.motion_geometry"]["values"][1]
    )
    env._test_raw_storage["raw_steps"][1] = ends[1] - 1
    with pytest.raises(ValueError, match="not fully recorded"):
        collector.capture_recorded_pour_geometry(
            "cube", "region", start_frames=starts, end_frames=ends
        )
    env._test_raw_storage["raw_steps"][1] = ends[1]
    bodies["cube"] = deepcopy(bodies["cube"])
    with pytest.raises(ValueError, match="another native object"):
        collector.capture_recorded_pour_geometry(
            "cube", "region", start_frames=starts, end_frames=ends
        )


@pytest.mark.parametrize("name", [item["name"] for item in _SOURCE["cases"]])
def test_lab_geometry_and_declared_occurrences_feed_independent_taskspec_reports(name):
    from embodichain.gen_sim.task_engine.evidence import project_pour_evidence_binding
    from embodichain.lab.task_program.integrations import task_program_fingerprint
    from embodichain.gen_sim.task_spec.evidence_report import (
        evaluate_physical_evidence_report,
    )

    collector, _, case, _, compiled = _collector_case(name)
    common = _SOURCE["common"]
    measured = case["record"]["measurements"]
    packet = collector.capture_recorded_pour_geometry(
        "cube",
        "region",
        start_frames=measured["start_frames"],
        end_frames=measured["end_frames"],
    )
    packet.update(
        segments=common["runtime"]["segments"],
        boundary_frames=measured["boundary_frames"],
    )
    occurrences = {
        segment["name"]: dict(
            segment_id=segment["metadata"]["program_segment_id"],
            segment_name=segment["name"],
            source_path=segment["metadata"]["program_segment_source_path"],
            invocation_id=segment["metadata"]["runtime"]["calls"][0]["invocation_id"],
        )
        for segment in common["runtime"]["segments"]
    }
    operation = dict(
        object_id="cube",
        reference_id="region",
        resource_id="left",
        object_native_uid="cube",
        reference_native_uid="region",
        program_fingerprint=task_program_fingerprint(compiled),
    )
    bindings = [
        project_pour_evidence_binding(
            common["spec"],
            common["graph"],
            common["validation"],
            common["program"],
            occurrences,
            operation,
            predicate_id=predicate_id,
            stage_node_id="step_01__call_02",
            pour_node_id="step_01__call_03",
            release_node_id="step_01__call_05",
            boundary_node_id="step_01__call_06",
            step_dt=0.04,
            episode_ids=collector.episode_ids,
        )
        for predicate_id in ("goal_001", "milestone:step_01:0")
    ]
    report = evaluate_physical_evidence_report(
        common["spec"],
        bindings,
        {binding.predicate_id: packet for binding in bindings},
        program=common["program"],
        num_envs=1,
    )[0]
    assert report.certificate_ready is case["record"]["accepted"][0]
    expected = "satisfied" if case["record"]["accepted"][0] else "failed"
    assert report.goal[0].status.value == expected
    assert report.milestones[0].status.value == expected


@pytest.mark.parametrize("missing", ["pose", "timestamp", "interior_slot"])
def test_pour_capture_rejects_unavailable_raw_acquisition(missing):
    collector, env, case, _, _ = _collector_case()
    measured = case["record"]["measurements"]
    slot = measured["start_frames"][0] + 1
    storage = env._test_raw_storage
    if missing == "pose":
        field = env.observation_manager.field_ids["cube"]
        storage["raw_valid"][field][0, slot] = False
        storage.get_non_tensor("raw_acquisition_errors")[field][
            0, slot
        ] = "source unavailable"
        error, message = RuntimeError, "source unavailable"
    elif missing == "timestamp":
        storage["raw_timestamp"][0, slot] = float("nan")
        error, message = RuntimeError, "acquisition time is unavailable"
    else:
        storage["raw_revision"][0, slot] = -1
        error, message = ValueError, "not fully recorded"
    with pytest.raises(error, match=message):
        collector.capture_recorded_pour_geometry(
            "cube",
            "region",
            start_frames=measured["start_frames"],
            end_frames=measured["end_frames"],
        )


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_held_poses_preserve_source_fk_precision_and_exact_historical_boundary(
    final, dtype
):
    # The original held observer case uses this exactly representable joint
    # coordinate and an identity object pose. Slot 1 in the frozen source is I.
    coordinate = 0.125 if dtype == torch.float32 else 0.125 + 2.0**-40
    qpos = torch.full((1, 31, 1), coordinate, dtype=dtype)
    collector, env, _, bodies, _ = _collector_case(robot_qpos=qpos)
    env.step_dt = 0.04
    collector._bridge.post_policy_port.bind_clock(collector._bridge.clock)
    robot = env.robot
    robot.get_joint_ids = lambda name: [0]
    seen = []

    def fk(qpos, name, to_matrix):
        seen.append((name, qpos.clone()))
        result = torch.eye(4, dtype=qpos.dtype).repeat(len(qpos), 1, 1)
        result[:, 0, 3] = qpos[:, 0]
        return result

    robot.compute_fk = fk
    robot.get_qpos = lambda: qpos[:, 1].clone()
    bodies["cube"].get_local_pose = lambda to_matrix: torch.eye(4).unsqueeze(0)
    packet = collector.capture_held_poses(
        "cube", ["custom_arm"], [1], observation_kind="final" if final else "recorded"
    )
    assert len(seen) == (2 if final else 1)
    assert all(
        name == "custom_arm" and value.dtype == dtype and torch.equal(value, qpos[:, 1])
        for name, value in seen
    )
    assert packet["object_pose"] == torch.eye(4).unsqueeze(0).tolist()
    expected_tcp = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    expected_tcp[:, 0, 3] = coordinate
    assert packet["tcp_poses"] == {"custom_arm": expected_tcp.tolist()}
    assert packet["pose_frame"]["observation_revision"] == 1
    assert (
        packet["pose_frame"]["batches"]["robot.tcp_pose:custom_arm"]["value_dtype"]
        == "torch.float64"
    )
    assert packet["recorded_acquisition"]["timestamps"] == [0.04]
    assert packet["recorded_acquisition"]["observation_revisions"] == [1]
    if final:
        robot.get_qpos = lambda: qpos[:, 1] + 0.01
        with pytest.raises(ValueError, match="recorded boundary"):
            collector.capture_held_poses(
                "cube", ["custom_arm"], [1], observation_kind="final"
            )
    else:
        robot.get_qpos = lambda: pytest.fail(
            "Historical FK cannot read live robot positions"
        )
        bodies["cube"].get_local_pose = lambda **kwargs: pytest.fail(
            "Historical FK cannot read a live body pose"
        )
        collector.capture_held_poses(
            "cube", ["custom_arm"], [1], observation_kind="recorded"
        )


def test_held_poses_reject_missing_robot_history_before_fk():
    collector, env, _, _, _ = _collector_case()
    env.robot.uid = "robot"
    env.robot.device = torch.device("cpu")
    env.robot.cfg = SimpleNamespace(root_props=SimpleNamespace(fixed_base=True))
    env.robot.get_joint_ids = lambda name: pytest.fail(
        "Missing raw joints cannot reach FK"
    )
    with pytest.raises(RuntimeError, match="qpos"):
        collector.capture_held_poses("cube", ["arm"], [1], observation_kind="recorded")


@pytest.mark.parametrize(
    "damage", ["joint_validity", "timestamp", "episode_during_fk", "close_during_fk"]
)
def test_held_poses_keep_missing_acquisition_and_lifetime_failures_unavailable(damage):
    qpos = torch.full((1, 31, 1), 0.125, dtype=torch.float32)
    collector, env, _, _, _ = _collector_case(
        robot_qpos=qpos,
        joint_valid=damage != "joint_validity",
        joint_timestamp=damage != "timestamp",
    )
    env.step_dt = 0.04
    robot = env.robot
    robot.get_joint_ids = lambda name: [0]

    def fk(qpos, **kwargs):
        if damage in {"joint_validity", "timestamp"}:
            pytest.fail("Unavailable history cannot reach FK")
        if damage == "episode_during_fk":
            env.episode_context._begin_episode([0])
        else:
            env._closed = True
        return torch.eye(4, dtype=qpos.dtype).unsqueeze(0)

    robot.compute_fk = fk
    with pytest.raises(
        (ValueError, RuntimeError),
        match={
            "joint_validity": "joint acquisition unavailable",
            "timestamp": "acquisition time",
            "episode_during_fk": "another episode",
            "close_during_fk": "after close",
        }[damage],
    ):
        collector.capture_held_poses("cube", ["arm"], [1], observation_kind="recorded")


def test_held_poses_read_only_selected_slots_once_and_return_owned_values(monkeypatch):
    qpos = torch.full((1, 31, 1), 0.125, dtype=torch.float32)
    collector, env, _, _, _ = _collector_case(robot_qpos=qpos)
    env.step_dt = 0.04
    env.robot.get_joint_ids = lambda name: [0]
    env.robot.compute_fk = lambda **kwargs: torch.eye(4).unsqueeze(0)
    reads = []
    original = env.observation_manager.read_history

    def read_history(*args, **kwargs):
        reads.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(env.observation_manager, "read_history", read_history)
    monkeypatch.setattr(
        env.observation_manager,
        "snapshot_history",
        lambda **kwargs: pytest.fail(
            "Selected held slots must not copy complete history"
        ),
    )
    packet = collector.capture_held_poses(
        "cube", ["arm"], [1], observation_kind="recorded"
    )
    assert len(reads) == 1
    assert reads[0][0] == ((1,),)
    assert reads[0][1]["fields"] == ("cube", "robot_qpos")
    assert packet["recorded_acquisition"]["frame_indices"] == [1]
    packet["object_pose"][0][0][0] = 2
    again = collector.capture_held_poses(
        "cube", ["arm"], [1], observation_kind="recorded"
    )
    assert again["object_pose"][0][0][0] == 1


@pytest.mark.parametrize("fixed_base", [False, None])
def test_held_poses_reject_unsupported_robot_bases_before_reading_history(
    monkeypatch, fixed_base
):
    qpos = torch.full((1, 31, 1), 0.125, dtype=torch.float32)
    collector, env, _, _, _ = _collector_case(robot_qpos=qpos)
    env.robot.cfg.root_props.fixed_base = fixed_base
    monkeypatch.setattr(
        env.observation_manager,
        "read_history",
        lambda *a, **k: pytest.fail(
            "Unsupported robot base cannot acquire historical data"
        ),
    )
    with pytest.raises(ValueError, match="fixed robot base"):
        collector.capture_held_poses("cube", ["arm"], [1], observation_kind="recorded")
