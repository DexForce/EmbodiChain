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
import json
import pytest
from embodichain.gen_sim.task_spec import (
    TaskSpec,
    RoleSpec,
    Predicate,
    MilestoneSpec,
    canonical_hash,
)
from embodichain.lab.task_program.integrations.evidence.transport_evidence import (
    CoordinatedTransportEvidenceBinding,
)
from ._evaluation import evaluate_coordinated_transport_evidence

_SOURCE = json.loads(
    (
        Path(__file__).parents[4]
        / "gen_sim/task_spec/data/transport_placement_source_reference.json"
    ).read_text()
)


def _case(*, raw=False):
    source = deepcopy(_SOURCE["cases"][0])
    segment = source["runtime"]["segments"][0]
    segment["active"] = [True]
    meta = segment["metadata"]
    call = meta["runtime"]["calls"][0]
    descriptor = dict(
        segment_id=meta["program_segment_id"],
        segment_name=segment["name"],
        source_path=meta["program_segment_source_path"],
        invocation_id=call["invocation_id"],
    )
    operation = dict(
        operation_id="g",
        object_id="pan",
        reference_id="pad",
        target_id="target",
        resources=dict(left="left", right="right"),
        transport=descriptor,
    )
    past = Predicate(
        "held_by_both_grippers",
        predicate_id="past",
        arguments=dict(subject="payload", task_type="E5"),
    )
    current = replace(past, predicate_id="current")
    spec = TaskSpec(
        roles=(RoleSpec("payload"),),
        goal=(current,),
        milestones=(MilestoneSpec("transported", (past,)),),
    )
    timeline = source["timeline"]
    identity = None
    if raw:
        identity = dict(
            sensor_uid="contacts",
            program_fingerprint="a" * 64,
            step_dt=0.04,
            object_native_uids={"pan": "native_pan", "pad": "native_pad"},
            object_actors=timeline["object_actors"],
            object_body_types=timeline["object_body_types"],
            pad_actors=timeline["pad_actors"],
            topology_revision=3,
            timeline_episode_generations=[0],
        )
    binding = CoordinatedTransportEvidenceBinding(
        canonical_hash(spec.to_dict()),
        canonical_hash(source["program"]),
        "past",
        operation,
        ({"segment_id": "n", "source_path": ["program", "items", 0]},),
        "n",
        ("program", "items", 0),
        0.04,
        "raw_snapshot" if raw else "historical_record",
        episode_ids=(7,),
        sampled_contact_identity=identity,
    )
    metrics = source["record"]["measurements"]
    proof = {
        key: metrics["transport_scope" if key == "scope" else key]
        for key in (
            "scope",
            "invocation_id",
            "invocation_revision",
            "release_step",
            "acquired_step",
            "contact_timeline_digest",
            "release_receipt_digest",
        )
    }
    evidence = dict(
        segments=source["runtime"]["segments"], boundary_frames=[78], episode_ids=[7]
    )
    if raw:
        samples = []
        revision = 0
        for old in timeline["samples"]:
            subs = []
            for sub in old.get(
                "substeps",
                [
                    dict(
                        offset=0.04,
                        dt=0.04,
                        dropped_contacts=old["dropped_contacts"],
                        rows=old["rows"],
                    )
                ],
            ):
                revision += 1
                subs.append(
                    dict(
                        offset=sub["offset"],
                        dt=sub["dt"],
                        timestamp=(old["step"] - 1) * 0.04 + sub["offset"],
                        observation_revision=revision,
                        native_revision=revision,
                        topology_revision=3,
                        dropped_contacts=sub["dropped_contacts"],
                        available=True,
                        contacts=sub["rows"][0],
                    )
                )
            samples.append(
                dict(
                    step=old["step"],
                    time=old["time"],
                    episode_generation=0,
                    native_revision=revision,
                    topology_revision=3,
                    dropped_contacts=old["dropped_contacts"],
                    available=True,
                    contacts=old["rows"][0],
                    substeps=subs,
                    hand_opening={
                        part: values[0]
                        for part, values in old.get("hand_opening", {}).items()
                    },
                )
            )
        evidence["recorded_contacts"] = dict(
            schema_version="simulation.recorded_contacts/v1",
            **identity,
            object_id="pan",
            reference_id="pad",
            episode_ids=[7],
            env_ids=[0],
            boundary_frames=[78],
            timeline=dict(
                scope="native_substep_samples_not_continuous_contact_proof",
                step_dt=0.04,
                environments=[
                    dict(
                        env_id=0,
                        episode_generation=0,
                        errors=[],
                        incomplete_control_step=None,
                        samples=samples,
                    )
                ],
            ),
        )
    else:
        evidence.update(
            contact_timeline=timeline,
            record=dict(
                group_id="g",
                uid="pan",
                part_id=None,
                accepted=[True],
                measurements=proof,
            ),
        )
    return spec, binding, evidence, source["program"], source["graph"]


@pytest.mark.parametrize("raw", [False, True])
def test_transport_event_retains_source_acquisition_after_release_without_final_hold(
    raw,
):
    from ._evaluation import evaluate_physical_evidence_report

    spec, binding, evidence, program, _ = _case(raw=raw)
    outcome = evaluate_coordinated_transport_evidence(
        spec, binding, evidence, program=program
    )[0]
    assert outcome.status.value == "satisfied", outcome.reason
    report = evaluate_physical_evidence_report(
        spec, [binding], {"past": evidence}, program=program, num_envs=1
    )[0]
    assert report.milestones[0].status.value == "satisfied"
    assert not report.certificate_ready
    with pytest.raises(ValueError, match="not a final held state"):
        replace(binding, observation_kind="final", binding_hash="")
    with pytest.raises(ValueError, match="B=1/env0"):
        replace(binding, env_ids=(0, 1), binding_hash="")


@pytest.mark.parametrize(
    "damage", ["acquisition", "prefix", "episode", "topology", "boundary"]
)
def test_transport_event_rejects_missing_native_or_incomplete_occurrence_evidence(
    damage,
):
    spec, binding, evidence, program, _ = _case(raw=True)
    raw = evidence["recorded_contacts"]
    if damage == "acquisition":
        sample = raw["timeline"]["environments"][0]["samples"][0]
        empty = {"pairs": [], "normals": [], "impulses": []}
        sample["contacts"] = empty
        sample["substeps"][-1]["contacts"] = empty
    elif damage == "prefix":
        evidence["segments"][0]["successes"] = [False]
    elif damage == "episode":
        raw["episode_ids"] = [8]
    elif damage == "topology":
        raw["timeline"]["environments"][0]["samples"][0]["substeps"][0][
            "topology_revision"
        ] = 4
    else:
        evidence["boundary_frames"] = [77]
    assert (
        evaluate_coordinated_transport_evidence(
            spec, binding, evidence, program=program
        )[0].status.value
        == "unavailable"
    )
