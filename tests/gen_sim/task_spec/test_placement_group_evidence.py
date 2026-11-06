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
import json
from pathlib import Path
import pytest

from embodichain.gen_sim.task_spec import canonical_hash, EvaluationStatus
from embodichain.gen_sim.task_spec.evidence import SupportedPlacementEvidenceBinding
from embodichain.gen_sim.task_spec.placement_group_evidence import (
    PlacementEvidenceGroupBinding,
    evaluate_placement_group_evidence,
)
from embodichain.gen_sim.task_spec.evidence_report import (
    evaluate_physical_evidence_report,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "data/e1_count_inside_source_reference.json").read_text()
)


def _binding(*, recorded, raw):
    data = REFERENCE
    spec, program, graph = data["spec"], data["program"], data["graph"]
    predicate = spec["milestones"][0]["achieve"][0] if recorded else spec["goal"][0]
    paths = {
        node["id"]: ("program", "items", index)
        for index, node in enumerate(graph["nodes"])
    }
    by_node = {node["id"]: node for node in graph["nodes"]}
    lineage = {
        v["group_id"]: v
        for values in data["validation"]["coverage"]["lineage"].values()
        for v in values
    }
    members = []
    for group in graph["task_groups"]:
        release = next(
            name
            for name in group["node_ids"]
            if by_node[name]["call"]["kind"] == "place"
        )
        terminal = group["node_ids"][-1]
        boundary = terminal if recorded else graph["nodes"][-1]["id"]
        uid = lineage[group["id"]]["object_uid"]
        members.append(
            SupportedPlacementEvidenceBinding(
                canonical_hash(spec),
                canonical_hash(program),
                predicate["id"],
                uid,
                "region",
                "left",
                release,
                "count/" + release + ":0",
                paths[release],
                terminal,
                "unused-legacy-planar",
                paths[terminal],
                boundary,
                paths[boundary],
                "measured_planar_landing_region",
                "recorded" if recorded else "final",
                evidence_format="raw_snapshot" if raw else "historical_record",
                episode_ids=(4,),
                required_execution=(
                    tuple(
                        dict(segment_id=name, source_path=list(paths[name]))
                        for name in group["node_ids"]
                    )
                    if recorded
                    else ()
                ),
                planar_geometry=dict(
                    exact_silhouette=True,
                    program_fingerprint="a" * 64,
                    object_native_uid=uid,
                    reference_native_uid="region",
                ),
            )
        )
    return PlacementEvidenceGroupBinding(
        canonical_hash(spec),
        canonical_hash(program),
        predicate["id"],
        predicate["subject"],
        tuple(group["id"] for group in graph["task_groups"]),
        tuple(members),
    )


def _packet(binding, case):
    members = {}
    segments = {value["name"]: value for value in case["runtime"]["segments"]}
    for group_id, member in zip(binding.group_ids, binding.members, strict=True):
        if binding.observation_kind == "recorded":
            record = deepcopy(case["recorded_measurements"].get(group_id, {}))
        else:
            record = next(
                (
                    deepcopy(v)
                    for v in case["final_measurements"]
                    .get(binding.predicate_id, {})
                    .get("members", [])
                    if v["uid"] == member.object_id
                ),
                {},
            )
        if "measurements" not in record:
            members[member.binding_hash] = dict(
                unavailable=record.get(
                    "unavailable", "This original member observation is unavailable."
                ),
                episode_ids=[4],
            )
            continue
        frames = segments[member.boundary_segment_id]["end_steps"]
        metrics = record["measurements"]
        packet = dict(
            record=record,
            episode_ids=[4],
            segments=case["runtime"]["segments"],
            boundary_frames=frames,
            boundary_poses={
                member.object_id: metrics["object_pose"],
                "region": metrics["reference_pose"],
            },
        )
        if binding.evidence_format == "raw_snapshot":
            packet.pop("record")
            metrics.pop("support_tolerance_m")
            metrics.pop("penetration_tolerance_m")
            columns = ["support_gap_m", "footprint_area_m2", "outside_floor_area_m2"]
            channels = {
                "planar.geometry": [[metrics[key][0] for key in columns]],
                "planar.full_footprint_inside": metrics["full_footprint_inside"],
                "rigid.object_pose": metrics["object_pose"],
                "rigid.reference_pose": metrics["reference_pose"],
                **{
                    "planar." + key: metrics[key]
                    for key in (
                        "reference_has_raised_geometry",
                        "reference_aabb_overlap",
                    )
                },
            }
            frame = dict(
                timestamp=frames[0] * 0.04,
                observation_revision=1,
                env_ids=[0],
                clock_domain="simulation_time",
                revision_kind="planar_geometry_acquisition",
                batches={
                    key: dict(
                        evidence_id=key,
                        value_dtype=(
                            "torch.float64"
                            if key == "planar.geometry"
                            else (
                                "torch.float32"
                                if key.startswith("rigid.")
                                else "torch.bool"
                            )
                        ),
                        values=value,
                        valid=[True],
                        acquisition_errors=[None],
                        timestamp=frames[0] * 0.04,
                        observation_revision=1,
                        env_ids=[0],
                    )
                    for key, value in channels.items()
                },
            )
            packet.update(
                schema_version="simulation.planar_observation/v1",
                object_id=member.object_id,
                reference_id="region",
                **member.planar_geometry,
                observation_kind=binding.observation_kind,
                native_planar=metrics,
                geometry_columns=columns,
                geometry_frame=frame,
            )
        members[member.binding_hash] = packet
    packet = dict(
        schema_version="task_spec.placement_group_observation/v1",
        binding_hash=binding.binding_hash,
        episode_ids=[4],
        members=members,
    )
    frames = [value.get("boundary_frames") for value in members.values()]
    if (
        binding.observation_kind == "recorded"
        and len(frames) == len(binding.members)
        and all(isinstance(value, list) for value in frames)
    ):
        packet["boundary_frames"] = [max(value[0] for value in frames)]
    return packet


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda value: value["name"])
@pytest.mark.parametrize("raw", [False, True])
def test_finite_inside_matches_original_producer_and_geometry_evaluator(case, raw):
    final, past = _binding(recorded=False, raw=raw), _binding(recorded=True, raw=raw)
    packets = {value.predicate_id: _packet(value, case) for value in (final, past)}
    report = evaluate_physical_evidence_report(
        REFERENCE["spec"],
        (past, final),
        packets,
        program=REFERENCE["program"],
        num_envs=1,
    )[0].to_dict()
    observed = dict(
        goal=[value["status"] for value in report["goal"]],
        milestones=[value["status"] for value in report["milestones"]],
        status=report["status"],
        certificate_ready=report["certificate_ready"],
    )
    assert observed == case["expected"]


def test_one_placement_cannot_prove_a_complete_count_or_borrow_another_reference():
    group = _binding(recorded=False, raw=True)
    incomplete = replace(
        group, members=group.members[:1], group_ids=group.group_ids[:1], binding_hash=""
    )
    with pytest.raises(ValueError, match="finite count"):
        evaluate_placement_group_evidence(
            REFERENCE["spec"], incomplete, {}, program=REFERENCE["program"]
        )
    packet = _packet(group, REFERENCE["cases"][0])
    packet["members"].pop(group.members[-1].binding_hash)
    assert (
        evaluate_placement_group_evidence(
            REFERENCE["spec"], group, packet, program=REFERENCE["program"]
        )[0].status
        is EvaluationStatus.UNAVAILABLE
    )
    duplicate = group.members[0]
    with pytest.raises(ValueError, match="duplicate"):
        replace(group, members=(duplicate, duplicate), binding_hash="")
    foreign = replace(group.members[1], reference_id="foreign", binding_hash="")
    with pytest.raises(ValueError, match="reference"):
        replace(group, members=(group.members[0], foreign), binding_hash="")


def test_group_and_nested_placement_binding_drift_are_rejected():
    group = _binding(recorded=False, raw=True)
    before = group.to_dict()
    assert PlacementEvidenceGroupBinding.from_dict(group).to_dict() == before
    group.members[0].planar_geometry["exact_silhouette"] = False
    with pytest.raises(ValueError, match="binding_hash"):
        PlacementEvidenceGroupBinding.from_dict(group)
