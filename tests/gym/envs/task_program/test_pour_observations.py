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

"""Tests for shared trajectory state capture and restore helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.utils.trajectory_state import RecordedTrajectoryReader
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


def _collector_case(name="nominal", *, count=1):
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
        _traj_buffer={"states": {"rigid_objects": states}},
        _traj_steps=torch.zeros(count, dtype=torch.long),
        _closed=False,
    )
    recording = RecordedTrajectoryReader(
        source=lambda: (env._traj_buffer["states"], env._traj_steps),
        episode_context=episodes,
        is_closed=lambda: env._closed,
    )
    collector = SimulationStateObservationCollector(
        env, bridge, registration, recording=recording
    )
    env._traj_steps[:] = common["written_steps"][0]
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
    env._traj_steps[1] = ends[1] - 1
    with pytest.raises(ValueError, match="not fully recorded"):
        collector.capture_recorded_pour_geometry(
            "cube", "region", start_frames=starts, end_frames=ends
        )
    env._traj_steps[1] = ends[1]
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
