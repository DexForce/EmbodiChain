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
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from embodichain.gen_sim.task_engine.bundle_deployment import load_configured_bundle
from embodichain.gen_sim.task_engine.runtime_bindings import BundleEvidenceProjector
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from .test_task_program_bundle import _frozen
from .test_semantic_graph_validation import _plan


def _declared_bundle(tmp_path, *, count=0, family="E2", orientation=None):
    _, physical = _frozen(
        tmp_path, slender=orientation is not None, slender_pad=orientation is not None
    )
    if family == "E2":
        plan = _plan(count=count)
    else:
        from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
        from .test_task_spec_generation import candidate, step, selector

        source = candidate(
            step(
                "E1",
                "items",
                object=selector("items", count=count),
                target=selector("table"),
                relation="on",
                required_arm="left_arm",
                orientation_goal=orientation or "none",
            )
        )
        bindings = dict(
            schema_version="action_engine_role_bindings_v1",
            task_id=source["draft"]["task_id"],
            candidate_id=source["candidate_id"],
            reference_bindings={
                "step_01.object": ["a", "b"][:count],
                "step_01.target": ["table"],
            },
            role_bindings={},
        )
        objects = json.loads(
            json.dumps(physical.scene.planner_objects)
            .replace('"cylinder"', '"a"')
            .replace('"pad"', '"b"')
        )
        plan = SemanticTaskPlanner().plan_spec(source, bindings, objects)

    def native(value):
        return json.loads(
            json.dumps(value).replace('"a"', '"cylinder"').replace('"b"', '"pad"')
        )

    graph, paths = generate_task_program_bundle(
        native(plan.graph),
        physical,
        tmp_path / "bundle",
        robot_profile="dual_franka",
        implementation_revisions={"embodichain": "test"},
        task_spec=plan.task_spec,
        task_candidate=plan.candidate,
        role_bindings=native(plan.role_bindings),
        step_lineage=native(plan.lineage),
    )
    identity = json.loads((paths.root / "integration_fingerprint.json").read_text())
    loaded = load_configured_bundle(
        paths.root,
        external_input_digests={
            key: hashlib.sha256(Path(key).read_bytes()).hexdigest()
            for key in identity["external_inputs"]
        },
        implementation_revisions={"embodichain": "test"},
    )
    errors = {}
    session = SimpleNamespace(
        episode_ids=(4,),
        final_boundary=dict(segment_name=graph["nodes"][-1]["id"]),
        record_unavailable=lambda key, reason: errors.__setitem__(key, reason),
    )
    return loaded, session, errors


def test_configured_evidence_uses_sealed_coverage_and_exact_occurrences(tmp_path):
    loaded, session, errors = _declared_bundle(tmp_path)
    before = dict(loaded.snapshot.files)
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    recorded = projector.prepare(session)
    final = projector.finish(session, object())
    assert len(recorded) == len(final) == 1
    assert recorded[0]["observation_kind"] == "recorded"
    assert final[0]["observation_kind"] == "final"
    assert final[0]["episode_ids"] == [4]
    assert final[0]["native_uid"] == "cylinder"
    assert final[0]["condition"] == recorded[0]["condition"]
    assert final[0]["condition"]["kind"] == "upright"
    assert errors == {} and projector.unavailable == {}
    assert loaded.snapshot.files == before


def test_final_evidence_does_not_substitute_a_future_boundary(tmp_path):
    loaded, session, errors = _declared_bundle(tmp_path)
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    assert projector.prepare(session)
    session.final_boundary = None
    assert projector.finish(session, object()) == []
    assert errors and all("No actual final" in reason for reason in errors.values())


def test_finite_count_keeps_every_bound_member_and_its_recorded_occurrence(tmp_path):
    loaded, session, errors = _declared_bundle(tmp_path, count=2)
    before = loaded.task_spec.to_dict()
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    recorded = projector.prepare(session)
    final = projector.finish(session, object())
    assert len(recorded) == len(final) == 1
    assert [value["entity_id"] for value in final[0]["members"]] == ["cylinder", "pad"]
    assert len({value["policy_segment_id"] for value in recorded[0]["members"]}) == 2
    assert recorded[0]["predicate_id"] != final[0]["predicate_id"]
    assert errors == {} and projector.unavailable == {}
    assert loaded.task_spec.to_dict() == before


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("family", ["E2", "E1"])
def test_count_session_captures_member_boundaries_without_reobserving_prior_members(
    tmp_path, monkeypatch, partial, family
):
    import torch
    import yaml
    from scipy.spatial.transform import Rotation
    from dataclasses import replace
    from embodichain.gen_sim.task_engine.runtime_evidence import (
        TaskEngineEvidenceSession,
    )
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.task_program.bridge import (
        TaskProgramDemoBridge,
        EnvironmentStepClock,
        BufferedGymCommandSink,
        RuntimeCommandFrameEncoder,
    )
    from embodichain.lab.task_program.integrations.simulation import (
        observation_services,
    )
    from .test_runtime_evidence import _boundary, _episode

    loaded, _, _ = _declared_bundle(tmp_path, count=2, family=family)
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
        observation_manager=object(),
        poses={},
        reads=[],
    )

    class State:
        def __init__(self, environment, selected, registration, *, observations):
            assert observations is environment.observation_manager
            self.env, self.bridge = environment, selected
            self.episodes = [
                row["episode"] for row in environment.episode_context.snapshot()
            ]
            self.episode_ids = tuple(self.episodes)

        def packet(self, entity, kind):
            return dict(
                schema_version="simulation.state_observation/v1",
                entity_id=entity,
                native_uid=entity,
                observation_kind=kind,
                episode_ids=self.episodes,
                state_acquisition=dict(
                    evidence_id="rigid.pose",
                    value_dtype="torch.float32",
                    values=self.env.poses[entity].tolist(),
                    valid=[True],
                    acquisition_errors=[None],
                    timestamp=clock.now(),
                    env_ids=[0],
                    observation_revision=clock.step_index,
                    clock_domain="simulation_time",
                    revision_kind="collector_acquisition",
                ),
            )

        def capture_current_rigid_pose(self, entity):
            self.env.reads.append(("final", entity, clock.step_index))
            return self.packet(entity, "final")

        def capture_recorded_rigid_pose(self, entity, frames):
            self.env.reads.append(("recorded", entity, tuple(frames)))
            return {**self.packet(entity, "recorded"), "boundary_frames": list(frames)}

    class Support:
        def __init__(self, environment, selected, registration, *, observations):
            assert observations is environment.observation_manager
            self.env, self.registration = environment, registration

        def capture_recorded(self, preset, *, segment_id, source_path, frame_indices):
            self.env.reads.append(
                (
                    "recorded",
                    self.registration.settle_presets[preset].entity,
                    tuple(frame_indices),
                )
            )
            return {"unavailable": "Software collector has no native support channels."}

        def capture_current(self, preset):
            self.env.reads.append(
                (
                    "final",
                    self.registration.settle_presets[preset].entity,
                    clock.step_index,
                )
            )
            return {"unavailable": "Software collector has no native support channels."}

    def services(environment, selected, registration, *, observations):
        assert observations is environment.observation_manager
        return SimpleNamespace(
            state=State(environment, selected, registration, observations=observations),
            support=Support(
                environment, selected, registration, observations=observations
            ),
            controls={},
            recorded_contacts=None,
            cargo=None,
            close=lambda: None,
        )

    monkeypatch.setattr(observation_services, "SimulationObservationServices", services)
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    session = TaskEngineEvidenceSession(loaded, env, bridge)
    recorded = projector.prepare(session)
    assert len(recorded) == 1
    for value in recorded[0]["members"] if family == "E2" else ():
        pose = torch.eye(4)[None]
        rotation, _ = Rotation.align_vectors(
            [[0.0, 0.0, 1.0]], [value["condition"]["local_axis"]]
        )
        pose[0, :3, :3] = torch.tensor(rotation.as_matrix(), dtype=torch.float32)
        env.poses[value["entity_id"]] = pose
    session.bind(recorded)
    first = recorded[0]["members"][0]
    first_copy = None
    program = yaml.safe_load(loaded.snapshot.files["task_program/program.yaml"])
    boundaries = []
    for index, segment in enumerate(loaded.program.iter_segments()):
        clock.advance_after_env_step()
        boundary = _boundary(bridge, index, success=True)
        declared = program
        for part in segment.source_path:
            declared = declared[part]
        policies = []
        for policy_index, declaration in enumerate(declared.get("post", [])):
            if declaration["kind"] == "wait_stable":
                preset = declaration["preset"]
                policies.append(
                    dict(
                        kind="wait_stable",
                        policy_index=policy_index,
                        source_path=[*segment.source_path, "post", policy_index],
                        result_mask=[True],
                        result=dict(
                            kind="upright",
                            preset=preset,
                            accepted_mask=[True],
                            measurements=dict(alignment=[1.0]),
                        ),
                    )
                )
        boundary = replace(
            boundary, metadata={**boundary.metadata, "post_policies": policies}
        )
        session.capture_boundary(boundary)
        boundaries.append(boundary)
        if segment.name == first["boundary_segment_id"]:
            first_copy = session.snapshots[recorded[0]["predicate_id"]]["members"][
                first["binding_hash"]
            ]
            if partial:
                break
            # A later live negative must not replace the already captured member.
            if family == "E2":
                pose = env.poses[first["entity_id"]]
                world_x = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
                )
                pose[0, :3, :3] = world_x @ pose[0, :3, :3]
    result = _episode(boundaries, success=not partial)
    final = projector.finish(session, result)
    report = session.finish(result, final_bindings=final)
    packet = session.snapshots[recorded[0]["predicate_id"]]
    assert packet["members"][first["binding_hash"]] == first_copy
    assert len(packet["members"]) == (1 if partial else 2)
    assert len([row for row in env.reads if row[0] == "recorded"]) == (
        1 if partial else 2
    )
    assert len([row for row in env.reads if row[0] == "final"]) == 2
    row = report.strict_task_spec[0]
    if family == "E2":
        assert row["goal"][0]["status"] == ("satisfied" if partial else "failed")
        assert row["milestones"][0]["status"] == (
            "unavailable" if partial else "satisfied"
        )
    else:
        assert (
            row["goal"][0]["status"] == row["milestones"][0]["status"] == "unavailable"
        )
        current = session.snapshots[final[0]["predicate_id"]]["members"]
        for index, member in enumerate(final[0]["members"]):
            names = {
                value["name"] for value in current[member["binding_hash"]]["segments"]
            }
            assert member["boundary_segment_id"] in names
            if index == 0 or not partial:
                assert member["release_segment_id"] in names
            else:
                assert member["release_segment_id"] not in names
    assert row["certificate_ready"] is False


def test_explicit_count_one_keeps_the_existing_single_member_binding(tmp_path):
    loaded, session, errors = _declared_bundle(tmp_path, count=1)
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    recorded = projector.prepare(session)
    final = projector.finish(session, object())
    assert len(recorded) == len(final) == 1
    assert final[0]["schema_version"] == "task_spec.condition_evidence_binding/v1"
    assert "members" not in final[0]
    assert final[0]["entity_id"] == "cylinder"
    assert loaded.task_spec.roles[0].quantifier == "count"
    assert loaded.task_spec.roles[0].count == 1
    assert errors == {}


def test_finite_placement_public_projection_keeps_all_release_occurrences(tmp_path):
    loaded, session, errors = _declared_bundle(tmp_path, count=2, family="E1")
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    past = projector.prepare(session)
    final = projector.finish(session, object())
    assert len(past) == len(final) == 1
    assert final[0]["schema_version"] == "task_spec.placement_group_evidence_binding/v1"
    assert [value["object_id"] for value in final[0]["members"]] == ["cylinder", "pad"]
    assert len({value["release_invocation_id"] for value in final[0]["members"]}) == 2
    assert (
        len({tuple(value["boundary_source_path"]) for value in final[0]["members"]})
        == 1
    )
    assert errors == {}


def test_finite_placement_projection_rejects_incomplete_member_and_group_coverage(
    tmp_path,
):
    from embodichain.gen_sim.task_engine.evidence import (
        project_configured_placement_evidence_group_binding,
    )
    from embodichain.gen_sim.task_spec import canonical_hash
    import yaml

    loaded, session, _ = _declared_bundle(tmp_path, count=2, family="E1")
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    projected = projector.finish(session, object())[0]
    graph = json.loads(loaded.snapshot.files["semantic_task_graph.json"])
    audit = json.loads(loaded.snapshot.files["semantic_graph_validation.json"])
    program = yaml.safe_load(loaded.snapshot.files["task_program/program.yaml"])
    validation = dict(
        task_spec_digest=canonical_hash(loaded.task_spec.to_dict()),
        program_digest=canonical_hash(program),
        graph_digest=canonical_hash(graph),
        coverage=audit["coverage"],
    )
    occurrences = {
        value.name: dict(
            segment_name=value.name,
            source_path=list(value.source_path),
        )
        for value in loaded.program.iter_segments()
    }
    declarations = []
    for group_id, member in zip(
        projected["group_ids"], projected["members"], strict=True
    ):
        occurrences[member["release_segment_id"]].update(
            invocation_id=member["release_invocation_id"],
            **{
                key: member[key] for key in ("object_id", "reference_id", "resource_id")
            },
        )
        declarations.append(
            dict(
                group_id=group_id,
                release_node_id=member["release_segment_id"],
                boundary_node_id=member["boundary_segment_id"],
                **{
                    key: member[key]
                    for key in ("condition", "evidence_scope", "stability_preset")
                },
            )
        )

    def project(values, coverage):
        return project_configured_placement_evidence_group_binding(
            loaded.task_spec,
            graph,
            {**validation, "coverage": coverage},
            program,
            occurrences,
            values,
            predicate_id=projected["predicate_id"],
            episode_ids=session.episode_ids,
        )

    assert project(declarations, validation["coverage"]).to_dict() == projected
    with pytest.raises(ValueError, match="every covered group"):
        project(declarations[:1], validation["coverage"])
    incomplete = deepcopy(validation["coverage"])
    role = projected["role_id"]
    incomplete["roles"][role] = incomplete["roles"][role][:1]
    with pytest.raises(ValueError, match="bijection"):
        project(declarations, incomplete)


@pytest.mark.parametrize(
    "count,orientation", [(1, "upright"), (2, "upright"), (2, "vertical")]
)
def test_e1_orientation_projection_preserves_source_current_and_recorded_scope(
    tmp_path, count, orientation
):
    loaded, session, errors = _declared_bundle(
        tmp_path, count=count, family="E1", orientation=orientation
    )
    projector = BundleEvidenceProjector(loaded, step_dt=0.04, robot_uid="robot")
    past = projector.prepare(session)
    final = projector.finish(session, object())
    predicate_name = "object_upright" if orientation == "upright" else "orientation"
    final_id = next(
        value.id for value in loaded.task_spec.goal if value.name == predicate_name
    )
    past_id = next(
        value.id
        for milestone in loaded.task_spec.milestones
        for value in milestone.achieve
        if value.name == predicate_name
    )
    declaration = next(value for value in final if value["predicate_id"] == final_id)
    assert declaration["schema_version"] == (
        "task_spec.condition_group_evidence_binding/v1"
        if count == 2
        else "task_spec.condition_evidence_binding/v1"
    )
    members = declaration["members"] if count == 2 else [declaration]
    assert [value["entity_id"] for value in members] == ["cylinder", "pad"][:count]
    assert all(value["condition"]["kind"] == "upright" for value in members)
    assert len({tuple(value["boundary_source_path"]) for value in members}) == 1
    if orientation == "upright":
        assert past_id in {value["predicate_id"] for value in past}
        assert errors == {}
    else:
        assert past_id not in {value["predicate_id"] for value in past}
        assert set(errors) == {past_id}
        assert "recorded" in errors[past_id] and "upright" in errors[past_id]
