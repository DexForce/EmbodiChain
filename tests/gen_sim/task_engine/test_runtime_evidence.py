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

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
import yaml

from embodichain.gen_sim.task_engine.bundle_deployment import (
    prepare_configured_bundle,
    load_configured_bundle,
)
from embodichain.gen_sim.task_engine.bundle_io import write_bundle_atomic
from embodichain.gen_sim.task_engine.runtime_evidence import TaskEngineEvidenceSession
from embodichain.gen_sim.task_spec import (
    TaskSpec,
    Predicate,
    MilestoneSpec,
    canonical_hash,
)
from embodichain.gen_sim.task_spec.condition_evidence import ConditionEvidenceBinding
from embodichain.lab.gym.envs.demo import DemoEpisodeResult, DemoSegmentResult
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    EnvironmentStepClock,
    BufferedGymCommandSink,
    RuntimeCommandFrameEncoder,
)


class _StateAcquisition:
    """Spy the public acquisition boundary; TaskSpec itself remains real."""

    def __init__(self, environment, bridge, registration, *, observations):
        assert observations is environment.observation_manager
        self.env, self.bridge = environment, bridge
        self.episode_ids = tuple(
            row["episode"] for row in environment.episode_context.snapshot()
        )

    def capture_current_rigid_pose(self, entity_id):
        self.env.reads.append((entity_id, self.bridge.clock.step_index))
        if self.env.missing:
            raise ValueError("native pose channel unavailable")
        return dict(
            schema_version="simulation.state_observation/v1",
            entity_id=entity_id,
            native_uid=entity_id,
            observation_kind="final",
            episode_ids=list(self.episode_ids),
            state_acquisition=dict(
                evidence_id="rigid.pose",
                value_dtype="torch.float32",
                values=self.env.pose.tolist(),
                valid=[True],
                acquisition_errors=[None],
                timestamp=self.bridge.clock.now(),
                env_ids=[0],
                observation_revision=1,
                clock_domain="simulation_time",
                revision_kind="collector_acquisition",
            ),
        )

    def capture_recorded_rigid_pose(self, entity_id, frames):
        result = self.capture_current_rigid_pose(entity_id)
        result.update(observation_kind="recorded", boundary_frames=list(frames))
        return result


class _SupportAcquisition:
    def __init__(self, environment, *args, observations):
        assert observations is environment.observation_manager


class _Timeline:
    def __init__(self, environment, *args, observations):
        assert observations is environment.observation_manager
        self.env = environment
        self.identity = {"explicit": "timeline"}

    def close(self):
        self.env.timeline_closes += 1


def _session(
    tmp_path,
    monkeypatch,
    *,
    recorded=False,
    owned_timeline=False,
    absent_spec=False,
    cargo=False,
):
    from embodichain.lab.task_program.integrations.simulation import (
        observations,
        recorded_contacts,
    )

    from .test_bundle_deployment import _files

    files = _files()
    source = TaskSpec.from_dict(json.loads(files["task_spec.json"]))
    predicate = Predicate(
        "object_upright",
        predicate_id="now",
        arguments=dict(subject="role_001", task_type="E2"),
    )
    past = replace(predicate, predicate_id="past")
    spec = TaskSpec(
        roles=source.roles,
        goal=(predicate,),
        milestones=(MilestoneSpec("observed", (past,)),) if recorded else (),
    )
    from embodichain.gen_sim.task_engine.bundle_spec import bundle_task_spec_payload

    files["task_spec.json"] = json.dumps(
        bundle_task_spec_payload(None if absent_spec else spec)
    ).encode()
    if cargo:
        from embodichain.lab.gym.utils.composition import resolve_gym_components
        from embodichain.lab.task_program.integrations import (
            load_configured_task_program_deployment,
        )

        parsed = {name: yaml.safe_load(data) for name, data in files.items()}
        parsed["task_program/integration.yaml"]["runtime_services"][
            "cargo_envelope"
        ] = {"carrier_uids": ["cylinder"]}
        physical = resolve_gym_components(
            parsed["task_program_deployment.yaml"],
            base_dir=tmp_path,
            component_sources={
                name: value for name, value in parsed.items() if name.endswith(".yaml")
            },
        )
        configured = load_configured_task_program_deployment(
            task_program={
                key: {"inline": parsed[name]}
                for key, name in parsed["task_program_deployment.yaml"][
                    "task_program"
                ].items()
            },
            skill_profile=physical.embodiment_skill_profile,
            base_dir=tmp_path,
        )
        parsed["semantic_task_graph.json"][
            "integration_fingerprint"
        ] = configured.integration.integration_fingerprint
        for name in ("task_program/integration.yaml", "semantic_task_graph.json"):
            files[name] = (
                yaml.safe_dump(parsed[name], sort_keys=False).encode()
                if name.endswith(".yaml")
                else json.dumps(parsed[name]).encode()
            )
    snapshot = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "test-revision"},
    )
    root = write_bundle_atomic(tmp_path / "bundle", snapshot)
    loaded = load_configured_bundle(
        root,
        external_input_digests={"fixture-assets": "a" * 64},
        implementation_revisions={"embodichain": "test-revision"},
    )
    clock = EnvironmentStepClock(0.04)
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(
            SimpleNamespace(current_qpos=lambda ids: torch.zeros(1, 1))
        ),
        clock,
    )
    bridge = TaskProgramDemoBridge(loaded.program, runtime, sink, clock)
    epochs = EpisodeContext(1)
    epochs._begin_episode(torch.tensor([0]))
    epochs._finish_initialization(torch.tensor([0]))
    env = SimpleNamespace(
        num_envs=1,
        active_task_program_bridge=bridge,
        episode_context=epochs,
        pose=torch.eye(4)[None],
        reads=[],
        missing=False,
        timeline_closes=0,
        observation_manager=object(),
    )
    from embodichain.lab.task_program.integrations.simulation import (
        observation_services,
    )

    class Services:
        def __init__(self, environment, bridge, registration, *, observations):
            assert observations is environment.observation_manager
            self.state = _StateAcquisition(
                environment, bridge, registration, observations=observations
            )
            self.support = _SupportAcquisition(
                environment, bridge, registration, observations=observations
            )
            self.controls = {}
            self.recorded_contacts = (
                _Timeline(environment, observations=observations)
                if owned_timeline
                else None
            )
            self.cargo = None
            self.closed = False
            if registration.cargo_envelope_cfg is not None:
                from embodichain.lab.task_program.integrations.simulation import (
                    cargo_observation,
                )

                self.cargo = cargo_observation.SimulationCargoEnvelopeCollector(
                    environment,
                    bridge,
                    registration.cargo_envelope_cfg,
                    observations=observations,
                )

        def close(self):
            if not self.closed:
                if self.cargo is not None:
                    self.cargo.close()
                if self.recorded_contacts is not None:
                    self.recorded_contacts.close()
                self.closed = True

    monkeypatch.setattr(observation_services, "SimulationObservationServices", Services)
    session = TaskEngineEvidenceSession(loaded, env, bridge)
    program = yaml.safe_load(files["task_program/program.yaml"])
    final = ConditionEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(program),
        "now",
        "cylinder",
        "cylinder",
        "orientation",
        "final",
        dict(
            entity="cylinder",
            kind="upright",
            local_axis=[0.0, 0.0, 1.0],
            minimum_alignment=0.86,
            axis_direction_free=False,
        ),
        episode_ids=session.episode_ids,
    )
    return session, loaded, env, bridge, final


def _boundary(bridge, index=0, *, success=False):
    segment = tuple(bridge.program.iter_segments())[index]
    return DemoSegmentResult(
        index,
        segment.name,
        index,
        index + 1,
        success,
        active=(True,),
        start_steps=(index,),
        end_steps=(index + 1,),
        successes=(success,),
        metadata=dict(
            task_program_id=bridge.program.program_id,
            program_segment_id=segment.segment_id,
            program_segment_source_path=list(segment.source_path),
            runtime=dict(
                env_ids=[0],
                calls=[
                    dict(
                        invocation_id="original-invocation",
                        invocation_revision=0,
                        status="failed",
                        effects=[dict(decision=dict(success_mask=[False]))],
                    )
                ],
            ),
            validation=dict(accepted_mask=[success]),
        ),
    )


def _episode(boundaries, *, success=False):
    length = boundaries[-1].end_steps[0] if boundaries else 0
    return DemoEpisodeResult(
        0,
        length,
        success,
        (success,),
        (False,),
        (False,),
        "success" if success else "failed",
        segments=tuple(boundaries),
        lengths=(length,),
        completed_by_env=(success,),
        terminal_reasons=("success" if success else "failed",),
    )


def test_actual_bundle_session_keeps_program_effect_and_task_outcomes_separate(
    tmp_path, monkeypatch
):
    session, loaded, env, bridge, final = _session(
        tmp_path, monkeypatch, owned_timeline=True
    )
    session.bind(())
    bridge.clock.advance_after_env_step()
    boundary = _boundary(bridge)
    session.capture_boundary(boundary)
    assert (
        session.final_boundary["segment_id"] == boundary.metadata["program_segment_id"]
    )
    report = session.finish(_episode([boundary]), final_bindings=[final])
    assert report.program_success == (False,)
    assert report.action_physical_effects[0]["effects"][0]["decision"][
        "success_mask"
    ] == [False]
    assert report.strict_task_spec[0]["certificate_ready"]
    assert report.bundle_digest == loaded.snapshot.identity.digest
    assert env.timeline_closes == 1
    session.close()
    assert env.timeline_closes == 1
    assert env.reads == [("cylinder", 1)]
    exported = report.to_dict()
    exported["strict_task_spec"][0]["certificate_ready"] = False
    assert report.to_dict()["strict_task_spec"][0]["certificate_ready"]


def test_missing_acquisition_is_unavailable_even_when_program_succeeded(
    tmp_path, monkeypatch
):
    session, _, env, bridge, final = _session(tmp_path, monkeypatch)
    session.bind([final])
    env.missing = True
    bridge.clock.advance_after_env_step()
    boundary = _boundary(bridge, success=True)
    session.capture_boundary(boundary)
    report = session.finish(_episode([boundary], success=True))
    assert report.program_success == (True,)
    assert not report.strict_task_spec[0]["certificate_ready"]
    assert "native pose channel unavailable" in report.acquisition_errors["now"]
    assert "unavailable" in session.snapshots["now"]


def test_recorded_predicate_is_not_filled_by_late_final_pose(tmp_path, monkeypatch):
    session, _, env, bridge, final = _session(tmp_path, monkeypatch, recorded=True)
    boundary = tuple(bridge.program.iter_segments())[0]
    past = replace(
        final,
        predicate_id="past",
        observation_kind="recorded",
        boundary_segment_id=boundary.name,
        boundary_source_path=boundary.source_path,
        policy_segment_id=boundary.name,
        policy_source_path=boundary.source_path,
        stability_preset="declared",
        required_execution=(
            {"segment_id": boundary.name, "source_path": list(boundary.source_path)},
        ),
        binding_hash="",
    )
    session.bind([past, final])
    bridge.clock.advance_after_env_step()
    result = _boundary(bridge, success=True)
    # Intentionally omit the boundary callback: no late reconstruction is allowed.
    report = session.finish(_episode([result], success=True))
    assert "not observed" in session.snapshots["past"]["unavailable"]
    assert env.reads == [("cylinder", 1)]
    assert not report.strict_task_spec[0]["certificate_ready"]


@pytest.mark.parametrize("case", ["observed", "missing", "unreached", "cancelled"])
def test_raw_archive_keeps_effective_bindings_and_replays_original_report(
    tmp_path, monkeypatch, case
):
    from embodichain.gen_sim.task_engine.execution import TaskProgramBundleEpisode
    from embodichain.gen_sim.task_spec.evidence_report import (
        evaluate_physical_evidence_report,
    )

    session, bundle, env, bridge, final = _session(tmp_path, monkeypatch, recorded=True)
    selected = tuple(bridge.program.iter_segments())[0]
    past = replace(
        final,
        predicate_id="past",
        observation_kind="recorded",
        boundary_segment_id=selected.name,
        boundary_source_path=selected.source_path,
        policy_segment_id=selected.name,
        policy_source_path=selected.source_path,
        stability_preset="declared",
        required_execution=(
            {"segment_id": selected.name, "source_path": list(selected.source_path)},
        ),
        binding_hash="",
    )
    original_spec = bundle.task_spec.to_dict()
    original_spec_hash = canonical_hash(original_spec)
    session.bind((past, final))
    expected = (final.to_dict(), past.to_dict())
    boundaries = []
    env.missing = case == "missing"
    if case in {"observed", "missing"}:
        bridge.clock.advance_after_env_step()
        boundaries.append(_boundary(bridge, success=True))
        session.capture_boundary(boundaries[-1])
    execution = _episode(boundaries, success=case in {"observed", "missing"})
    if case == "cancelled":
        execution = replace(
            execution, terminal_reason="cancelled", terminal_reasons=("cancelled",)
        )
    report = session.finish(execution, final_bindings=(final,))
    reads = list(env.reads)
    assert session.binding_declarations == expected
    result = TaskProgramBundleEpisode(
        execution, report, session.snapshots, session.binding_declarations
    )
    archive_path = tmp_path / "raw_evidence.json"
    archive_path.write_text(json.dumps(result.to_raw_evidence()))
    archived = json.loads(archive_path.read_text())
    assert archived["schema_version"] == "task_engine.raw_evidence/v2"
    assert archived["bundle_digest"] == bundle.snapshot.identity.digest
    assert archived["run_id"] == report.run_id
    assert archived["episode_ids"] == list(session.episode_ids)
    assert archived["runtime"] == execution.to_metadata()
    assert archived["bindings"] == list(expected)
    assert archived["acquisition_errors"] == report.acquisition_errors
    if case != "observed":
        assert "unavailable" in archived["measurements"]["past"]
        assert not report.strict_task_spec[0]["certificate_ready"]
    if case in {"unreached", "cancelled"}:
        assert archived["runtime"]["segments"] == []
        assert "not observed" in archived["measurements"]["past"]["unavailable"]
        assert report.program_success == (False,)
    if case == "cancelled":
        assert archived["runtime"]["terminal_reason"] == "cancelled"
        assert report.terminal_reasons == ("cancelled",)
    replay = evaluate_physical_evidence_report(
        original_spec,
        archived["bindings"],
        archived["measurements"],
        program=yaml.safe_load(bundle.snapshot.files["task_program/program.yaml"]),
        num_envs=len(archived["env_ids"]),
    )
    assert [value.to_dict() for value in replay] == list(report.strict_task_spec)
    assert bundle.task_spec.to_dict() == original_spec
    assert canonical_hash(bundle.task_spec.to_dict()) == original_spec_hash
    archived["bindings"][0]["binding_hash"] = "changed export"
    archived["measurements"]["now"].clear()
    assert result.to_raw_evidence()["bindings"] == list(expected)
    assert result.to_raw_evidence()["measurements"] == session.snapshots
    assert env.reads == reads


def test_session_rejects_foreign_episode_and_never_resigns_mutated_binding(
    tmp_path, monkeypatch
):
    session, _, env, bridge, final = _session(tmp_path, monkeypatch)
    final.condition["minimum_alignment"] = 0.1
    with pytest.raises(ValueError, match="binding_hash"):
        session.bind([final])
    env.episode_context._begin_episode(torch.tensor([0]))
    env.episode_context._finish_initialization(torch.tensor([0]))
    with pytest.raises(ValueError, match="another initialized episode"):
        session.bind(())
    session.close()


def test_finish_rejects_foreign_executor_result_and_closes_only_own_observer(
    tmp_path, monkeypatch
):
    session, _, env, bridge, final = _session(
        tmp_path, monkeypatch, owned_timeline=True
    )
    session.bind([final])
    bridge.clock.advance_after_env_step()
    observed = _boundary(bridge)
    session.capture_boundary(observed)
    foreign = replace(observed, name="other")
    with pytest.raises(ValueError, match="differs from observed"):
        session.finish(_episode([foreign]))
    assert env.timeline_closes == 1 and env.reads == []


def test_boundary_snapshot_is_not_overwritten_after_later_motion(tmp_path, monkeypatch):
    session, _, env, bridge, final = _session(tmp_path, monkeypatch, recorded=True)
    selected = tuple(bridge.program.iter_segments())[0]
    past = replace(
        final,
        predicate_id="past",
        observation_kind="recorded",
        boundary_segment_id=selected.name,
        boundary_source_path=selected.source_path,
        policy_segment_id=selected.name,
        policy_source_path=selected.source_path,
        stability_preset="declared",
        required_execution=(
            {"segment_id": selected.name, "source_path": list(selected.source_path)},
        ),
        binding_hash="",
    )
    session.bind([past, final])
    bridge.clock.advance_after_env_step()
    first = _boundary(bridge, 0, success=True)
    session.capture_boundary(first)
    original = session.snapshots["past"]
    env.pose[0, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    bridge.clock.advance_after_env_step()
    second = _boundary(bridge, 1, success=True)
    session.capture_boundary(second)
    session.finish(_episode([first, second], success=True))
    assert session.snapshots["past"] == original
    assert original["boundary_frames"] == [1]
    assert env.reads == [("cylinder", 1), ("cylinder", 2)]


def test_failed_boundary_acquisition_retains_actual_frame_identity(
    tmp_path, monkeypatch
):
    session, _, env, bridge, final = _session(tmp_path, monkeypatch, recorded=True)
    selected = tuple(bridge.program.iter_segments())[0]
    past = replace(
        final,
        predicate_id="past",
        observation_kind="recorded",
        boundary_segment_id=selected.name,
        boundary_source_path=selected.source_path,
        policy_segment_id=selected.name,
        policy_source_path=selected.source_path,
        stability_preset="declared",
        required_execution=(
            {"segment_id": selected.name, "source_path": list(selected.source_path)},
        ),
        binding_hash="",
    )
    session.bind([past])
    env.missing = True
    bridge.clock.advance_after_env_step()
    boundary = _boundary(bridge)
    session.capture_boundary(boundary)
    assert session.snapshots["past"]["boundary_frames"] == [1]
    assert "unavailable" in session.snapshots["past"]
    session.finish(_episode([boundary]), final_bindings=[])


@pytest.mark.parametrize("failure", [None, "binding", "execution", "final_projection"])
@pytest.mark.parametrize("with_initial_observer", [False, True])
def test_bundle_episode_entry_uses_actual_boundary_and_closes_only_owned_resources(
    tmp_path, monkeypatch, failure, with_initial_observer
):
    from embodichain.gen_sim.task_engine.execution import (
        execute_task_program_bundle_episode,
    )
    from embodichain.lab.gym.envs import demo

    original, bundle, env, bridge, final = _session(
        tmp_path, monkeypatch, owned_timeline=True
    )
    original.close()
    env.timeline_closes = 0
    sequence = []

    class Segments:
        def close(self):
            sequence.append("segments_closed")

    segments = Segments()

    def select():
        sequence.append("select")
        return segments

    observer = lambda values: None

    def select_observed(*, initial_plan_observer):
        assert initial_plan_observer is observer
        return select()

    env.create_demo_segments = select_observed if with_initial_observer else select

    def prepare(session):
        assert bridge.clock.step_index == 0
        sequence.append("prepare")
        if failure == "binding":
            raise ValueError("binding failed")
        return ()

    def execute(actual, **kwargs):
        assert actual is env and kwargs["segments"] is segments
        assert kwargs["episode_index"] == 4 and kwargs["attempt_id"] == 2
        sequence.append("execute")
        if failure == "execution":
            raise ValueError("execution failed")
        bridge.clock.advance_after_env_step()
        boundary = _boundary(bridge, success=True)
        kwargs["on_segment_completed"](boundary)
        return _episode((boundary,), success=True)

    def project(session, result):
        sequence.append("final")
        assert (
            session.runtime_metadata["segments"][0]["name"] == result.segments[0].name
        )
        if failure == "final_projection":
            raise ValueError("final projection failed")
        return (final.to_dict(),)

    monkeypatch.setattr(demo, "execute_demo_episode", execute)
    kwargs = dict(
        prepare_evidence=prepare,
        final_evidence=project,
        episode_index=4,
        attempt_id=2,
    )
    if with_initial_observer:
        kwargs["initial_plan_observer"] = observer
    if failure is not None:
        with pytest.raises(ValueError, match="failed"):
            execute_task_program_bundle_episode(bundle, env, **kwargs)
    else:
        result = execute_task_program_bundle_episode(bundle, env, **kwargs)
        assert result.execution.success == (True,)
        assert result.evidence.program_success == (True,)
        assert result.evidence.strict_task_spec[0]["status"] == "satisfied"
        assert result.measurements["now"]["observation_kind"] == "final"
        assert result.measurements["now"]["run_id"] == result.evidence.run_id
        assert result.binding_declarations == (final.to_dict(),)
        assert result.to_raw_evidence()["bindings"] == [final.to_dict()]
        original_pose = deepcopy(
            result.measurements["now"]["state_acquisition"]["values"]
        )
        env.pose.zero_()
        assert (
            result.measurements["now"]["state_acquisition"]["values"] == original_pose
        )
    assert sequence[-1] == "segments_closed"
    assert env.timeline_closes == 1
    assert env.active_task_program_bridge is bridge


@pytest.mark.parametrize("success", [False, True])
def test_absent_spec_executes_without_inventing_normative_success(
    tmp_path, monkeypatch, success
):
    session, loaded, env, bridge, final = _session(
        tmp_path, monkeypatch, absent_spec=True, owned_timeline=True
    )
    assert loaded.task_spec is None
    with pytest.raises(ValueError, match="without TaskSpec"):
        session.bind((final,))
    session.bind(())
    bridge.clock.advance_after_env_step()
    boundary = _boundary(bridge, success=success)
    session.capture_boundary(boundary)
    report = session.finish(_episode((boundary,), success=success), final_bindings=())
    assert report.program_success == (success,)
    assert report.strict_task_spec[0]["status"] == "unavailable"
    assert report.strict_task_spec[0]["certificate_ready"] is False
    assert "No normative TaskSpec" in report.strict_task_spec[0]["reason"]
    assert report.action_physical_effects[0]["effects"][0]["decision"][
        "success_mask"
    ] == [False]
    assert env.timeline_closes == 1 and env.reads == []


def test_cargo_observation_keeps_program_and_taskspec_results_independent(
    tmp_path, monkeypatch
):
    from embodichain.lab.task_program.integrations.simulation import cargo_observation

    made = []

    class Cargo:
        def __init__(self, environment, bridge, cfg, *, observations):
            assert observations is environment.observation_manager
            self.environment = environment
            self.bridge = bridge
            self.closed = 0
            assert cfg.carrier_uids == ("cylinder",)
            made.append(self)

        def capture_report(self, *, invocation_id):
            assert invocation_id == "original-invocation"
            return {
                "schema_version": "lab.cargo_envelope_observation/v1",
                "episode_ids": [
                    row["episode"]
                    for row in self.environment.episode_context.snapshot()
                ],
                "env_ids": [0],
                "scope": "recorded_trajectory_and_terminal",
                "contents": [
                    {
                        "carrier": "cylinder",
                        "observed_mask": [False],
                        "valid_mask": [True],
                        "first_failed_frames": [0],
                    }
                ],
            }

        def close(self):
            self.closed += 1

    monkeypatch.setattr(cargo_observation, "SimulationCargoEnvelopeCollector", Cargo)
    session, loaded, env, bridge, final = _session(tmp_path, monkeypatch, cargo=True)
    boundary = _boundary(bridge, success=True)
    session.capture_boundary(boundary)
    report = session.finish(_episode([boundary], success=True), final_bindings=(final,))
    assert report.program_success == (True,)
    assert report.cargo_observation["contents"][0]["observed_mask"] == [False]
    assert report.to_dict()["cargo_observation"] == report.cargo_observation
    assert made[0].closed == 1
    assert report.strict_task_spec[0]["status"] == "satisfied"
    from embodichain.gen_sim.task_engine.execution import TaskProgramBundleEpisode
    from embodichain.gen_sim.task_engine.reporting import (
        build_bundle_execution_report,
        verify_bundle_execution_report,
    )

    execution = build_bundle_execution_report(
        loaded,
        TaskProgramBundleEpisode(_episode([boundary], success=True), report),
        record_dir=tmp_path / "run",
    )
    assert execution["environments"][0]["success"] is False
    assert execution["environments"][0]["terminal_reason"] == "cargo_envelope_failed"
    missing = deepcopy(execution)
    del missing["runtime_result"]["physical_run_report"]["cargo_observation"]
    with pytest.raises(ValueError, match="cargo evidence is missing"):
        verify_bundle_execution_report(missing, loaded)


def test_public_report_verifies_current_bundle_and_rejects_acceptance_drift(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine.execution import TaskProgramBundleEpisode
    from embodichain.gen_sim.task_engine.reporting import (
        build_bundle_execution_report,
        verify_bundle_execution_report,
    )

    session, bundle, env, bridge, binding = _session(tmp_path, monkeypatch)
    boundary = _boundary(bridge, success=True)
    session.capture_boundary(boundary)
    result = _episode([boundary], success=True)
    evidence = session.finish(result, final_bindings=(binding,))
    report = build_bundle_execution_report(
        bundle, TaskProgramBundleEpisode(result, evidence), record_dir=tmp_path / "run"
    )
    assert report["environments"][0]["success"] is True
    assert report["runtime_result"]["physical_run_report"]["run_id"] == evidence.run_id
    assert session.snapshots["now"]["run_id"] == evidence.run_id
    assert (
        verify_bundle_execution_report(
            report, bundle, expected_record_dir=tmp_path / "run"
        )
        == report
    )
    changed = deepcopy(report)
    changed["environments"][0]["success"] = False
    with pytest.raises(ValueError, match="contradicts"):
        verify_bundle_execution_report(changed, bundle)
    old = deepcopy(report)
    del old["runtime_result"]["bundle_identity"]
    with pytest.raises(ValueError, match="old or different"):
        verify_bundle_execution_report(old, bundle)

    # Renaming a provenance owner changes only the sealed bundle identity; its
    # previous run and normative certificate still belong to the original.
    from embodichain.gen_sim.task_engine.bundle_inputs import (
        SOURCE_INPUT_PROVENANCE_FILE,
    )

    files = dict(bundle.snapshot.files)
    provenance = json.loads(files.pop(SOURCE_INPUT_PROVENANCE_FILE))
    provenance["groups"] = {"source_owner": provenance["groups"]["application"]}
    identity = bundle.snapshot.identity.to_dict()
    current = prepare_configured_bundle(
        files,
        base_dir=tmp_path,
        external_input_digests=identity["external_inputs"],
        implementation_revisions=identity["implementation_revisions"],
        source_input_provenance=provenance,
    )
    current_root = write_bundle_atomic(tmp_path / "new_source", current)
    current_bundle = load_configured_bundle(
        current_root,
        external_input_digests=identity["external_inputs"],
        implementation_revisions=identity["implementation_revisions"],
    )
    assert current.identity.digest != bundle.snapshot.identity.digest
    assert (
        current.identity.to_dict()["task_spec_digest"] == identity["task_spec_digest"]
    )
    with pytest.raises(ValueError, match="old or different"):
        verify_bundle_execution_report(report, current_bundle)


def test_new_evidence_sessions_get_distinct_run_namespaces(tmp_path, monkeypatch):
    ids = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        session, bundle, env, bridge, binding = _session(root, monkeypatch)
        boundary = _boundary(bridge, success=True)
        session.capture_boundary(boundary)
        report = session.finish(
            _episode([boundary], success=True), final_bindings=(binding,)
        )
        ids.append((report.run_id, session.snapshots["now"]["run_id"]))
    assert ids[0][0] != ids[1][0]
    assert all(a == b for a, b in ids)


def test_missing_proof_diagnostic_does_not_supply_a_normative_result(
    tmp_path, monkeypatch
):
    session, bundle, env, bridge, binding = _session(tmp_path, monkeypatch)
    session.record_unavailable("now", "The declared native proof is unavailable.")
    with pytest.raises(ValueError, match="existing normative predicate"):
        session.record_unavailable("foreign", "No proof.")
    boundary = _boundary(bridge, success=True)
    session.capture_boundary(boundary)
    report = session.finish(_episode([boundary], success=True), final_bindings=())
    assert report.program_success == (True,)
    assert report.strict_task_spec[0]["status"] == "unavailable"
    assert report.strict_task_spec[0]["certificate_ready"] is False
    assert (
        report.acquisition_errors["now"] == "The declared native proof is unavailable."
    )


def test_report_retains_original_initial_plan_rejection_classification(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine.execution import TaskProgramBundleEpisode
    from embodichain.gen_sim.task_engine.reporting import (
        build_bundle_execution_report,
        verify_bundle_execution_report,
    )

    session, bundle, env, bridge, binding = _session(tmp_path, monkeypatch)
    boundary = _boundary(bridge, success=False)
    metadata = boundary.to_metadata()["metadata"]
    metadata["runtime"]["calls"][0]["plan_attempts"] = [{"plan_success_mask": [False]}]
    boundary = replace(boundary, metadata=metadata)
    session.capture_boundary(boundary)
    result = _episode([boundary], success=False)
    evidence = session.finish(result, final_bindings=(binding,))
    report = build_bundle_execution_report(
        bundle, TaskProgramBundleEpisode(result, evidence), record_dir=tmp_path / "run"
    )
    assert report["status"] == "rejected"
    assert report["environments"][0]["terminal_reason"] == "initial_plan_rejected"
    assert report["runtime_result"]["physical_run_report"]["terminal_reasons"] == list(
        evidence.terminal_reasons
    )
    report["status"] = "failed"
    with pytest.raises(ValueError, match="status contradicts"):
        verify_bundle_execution_report(report, bundle)


def test_evidence_session_rejects_unsealed_contact_history_configuration(
    tmp_path, monkeypatch
):
    _, loaded, env, bridge, _ = _session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="sealed runtime declaration"):
        TaskEngineEvidenceSession(loaded, env, bridge, timeline_cfg=object())
