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

"""Independent positive and negative controls for native acceptance evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.task_engine.contracts import canonical_hash
from embodichain.gen_sim.task_engine._task_program.stable_support_evidence import (
    CRITERION,
    stable_support_flags,
)
from embodichain.gen_sim.task_engine._task_program.stability import StabilityConstraint
from embodichain.gen_sim.task_engine._task_program.coordinated_placement_evidence import (
    transport_proof,
)


def _support_case():
    timeline = {
        "errors": [],
        "step_dt": 0.04,
        "object_actors": {"fruit": [[1]], "plate": [[2]]},
        "object_body_types": {"fruit": "dynamic", "plate": "dynamic"},
        "samples": [
            {
                "step": step,
                "time": step * 0.04,
                "dropped_contacts": 0,
                "rows": [
                    {"pairs": [[2, 1]], "normals": [[0, 0, 1]], "impulses": [0.01]}
                ],
            }
            for step in range(1, 4)
        ],
    }
    metrics = {
        "criterion": CRITERION,
        "entity": "fruit",
        "reference": "plate",
        "start_step": 0,
        "end_step": 3,
        "step_dt": 0.04,
        "duration": 0.12,
        "release_timestamp": 0,
        "constraint": asdict(
            StabilityConstraint(
                entity="fruit",
                reference="plate",
                displacement=(0.0, 0.0, 0.0),
                kind="placement",
                duration=0.12,
            )
        ),
        "relative_poses": torch.eye(4).repeat(4, 1, 1).tolist(),
        "timeline_digest": canonical_hash(timeline),
    }
    for sample in timeline["samples"]:
        sample["substeps"] = [
            {"offset": 0.04, "dt": 0.04, "dropped_contacts": 0, "rows": sample["rows"]}
        ]
    metrics["timeline_digest"] = canonical_hash(timeline)
    return metrics, timeline


def test_stable_native_reference_load_is_not_a_com_contact_hull_test():
    metrics, timeline = _support_case()
    assert stable_support_flags(metrics, timeline) == [True]


def test_rooted_cargo_can_rest_on_another_dynamic_cargo_without_external_support():
    metrics, timeline = _support_case()
    timeline["object_actors"]["other"] = [[3]]
    timeline["object_body_types"]["other"] = "dynamic"
    for sample in timeline["samples"]:
        sample["rows"][0].update(
            pairs=[[2, 3], [3, 1]], normals=[[0, 0, 1]] * 2, impulses=[0.01] * 2
        )
    metrics["timeline_digest"] = canonical_hash(timeline)
    assert stable_support_flags(metrics, timeline) == [True]


@pytest.mark.parametrize(
    "damage",
    [
        "unrooted_cycle",
        "external_cargo_contact",
        "cross_substep_path",
        "return_to_origin",
        "static_cargo",
    ],
)
def test_support_graph_and_maximum_drift_do_not_invent_success(damage):
    metrics, timeline = _support_case()
    timeline["object_actors"]["other"] = [[3]]
    timeline["object_body_types"]["other"] = "dynamic"
    for sample in timeline["samples"]:
        row = sample["rows"][0]
        row.update(pairs=[[2, 3], [3, 1]], normals=[[0, 0, 1]] * 2, impulses=[0.01] * 2)
        if damage == "unrooted_cycle":
            row["pairs"] = [[1, 3], [3, 1]]
        elif damage == "external_cargo_contact":
            row["pairs"].append([99, 3])
            row["normals"].append([1, 0, 0])
            row["impulses"].append(0.01)
        elif damage == "cross_substep_path":
            sample["substeps"] = [
                {
                    "dt": 0.02,
                    "offset": offset,
                    "dropped_contacts": 0,
                    "rows": [
                        {"pairs": [pair], "normals": [[0, 0, 1]], "impulses": [0.01]}
                    ],
                }
                for offset, pair in ((0.02, [3, 1]), (0.04, [2, 3]))
            ]
    if damage == "return_to_origin":
        metrics["relative_poses"][2][0][3] = 0.021
    elif damage == "static_cargo":
        timeline["object_body_types"]["other"] = "static"
    metrics["timeline_digest"] = canonical_hash(timeline)
    assert stable_support_flags(metrics, timeline) == [False]


@pytest.mark.parametrize(
    "damage",
    ["wrong_receiver", "another_support", "side_wall_friction", "moving"],
)
def test_stable_support_rejects_observable_negative_controls(damage):
    metrics, timeline = _support_case()
    if damage == "moving":
        for index, pose in enumerate(metrics["relative_poses"]):
            pose[0][3] = index * 0.02
    else:
        for sample in timeline["samples"]:
            row = sample["rows"][0]
            if damage == "wrong_receiver":
                row["pairs"] = [[3, 1]]
            else:
                row["pairs"].append([3, 1])
                row["normals"].append(
                    [1, 0, 0] if damage == "side_wall_friction" else [0, 0, 1]
                )
                row["impulses"].append(0.01)
    metrics["timeline_digest"] = canonical_hash(timeline)
    assert stable_support_flags(metrics, timeline) == [False]


@pytest.mark.parametrize("moving", [False, True])
def test_zero_active_contacts_do_not_distinguish_sleep_from_absent_support(moving):
    metrics, timeline = _support_case()
    for sample in timeline["samples"]:
        sample["rows"][0].update(pairs=[], normals=[], impulses=[])
    metrics["timeline_digest"] = canonical_hash(timeline)
    if moving:
        metrics["relative_poses"][2][0][3] = 0.03
        assert stable_support_flags(metrics, timeline) == [False]
    else:
        with pytest.raises(ValueError, match="zero active contacts"):
            stable_support_flags(metrics, timeline)


@pytest.mark.parametrize(
    "damage", ["missing", "stale", "dropped", "nonfinite", "before_release"]
)
def test_missing_or_invalid_support_source_is_not_guessed_success(damage):
    metrics, timeline = _support_case()
    if damage == "missing":
        timeline["samples"].pop()
    elif damage == "stale":
        timeline["samples"][0]["time"] = 1
    elif damage == "dropped":
        timeline["samples"][0]["dropped_contacts"] = 1
    elif damage == "nonfinite":
        metrics["relative_poses"][0][0][0] = float("nan")
    else:
        metrics["release_timestamp"] = 0.04
    metrics["timeline_digest"] = canonical_hash(timeline)
    with pytest.raises((ValueError, KeyError)):
        stable_support_flags(metrics, timeline)


def _transport_case():
    call_id, invocation = (
        "simulation.coordinated_transport",
        "p/p:segment:$.program.items[0]:0",
    )
    semantic = {
        "kind": "registered",
        "call_id": call_id,
        "arguments": {
            "object": "pan",
            "target": "target",
            "reference": "pad",
            "relation": "on",
        },
        "resources": {"left": "left", "right": "right"},
    }
    node = {"id": "n", "task_type": "E5", "task_instance_id": "g", "call": semantic}
    group = {"id": "g", "task_type": "E5", "node_ids": ["n"]}
    program = {
        "program_id": "p",
        "program": {"items": [{"name": "n", "steps": {"call": semantic}}]},
    }
    expectations, clauses, evidence, endpoints = [], [], {}, []
    for role in ("left", "right"):
        expectations.append(
            {
                "expectation_id": role,
                "slot_id": role,
                "resource_id": role,
                "task_state_key": role,
                "object_id": "pan",
                "relation": "detached",
            }
        )
        clauses.append(
            {
                "expectation_id": role,
                "clause_id": role,
                "evidence_kind": "constraint",
                "expected": False,
                "source": {
                    "address": {"control_part": role + "_eef", "channel": "constraint"}
                },
            }
        )
        evidence[role] = {
            "timestamp": 3.0,
            "observation_revision": 9,
            "env_ids": [0],
            "valid_mask": [True],
            "values": [False],
        }
        endpoints.append(
            {
                "slot_id": role,
                "endpoint_id": "grasp",
                "resource_id": role,
                "task_state_key": role,
                "target_id": role + "_eef",
            }
        )
    terminal = {
        "timestamp": 3.0,
        "observation_revision": 9,
        "boundary": {"kind": "terminal"},
        "monitor": {"monitor_id": "builtin.composite_effect", "revision": "1"},
        "effect_spec": {
            "semantic_id": call_id,
            "skill_id": "coordinated_pickment",
            "invocation_id": invocation,
            "invocation_revision": 0,
            "env_ids": [0],
            "effect_kind": "release",
            "state_expectations": expectations,
            "clauses": clauses,
        },
        "decision": {"success_mask": [True], "failure_mask": [False]},
        "evidence": evidence,
    }
    runtime_call = {
        "semantic_id": call_id,
        "invocation_id": invocation,
        "invocation_revision": 0,
        "skill_id": "coordinated_pickment",
        "command_count": 5,
        "status": "completed",
        "effects": [terminal],
        "resolved_core_policy": {"endpoints": endpoints},
        "events": [
            {
                "kind": "trajectory_segment_entered",
                "segment_name": name,
                "timestamp": time,
                "invocation_id": invocation,
                "invocation_revision": 0,
            }
            for name, time in (("lift", 1.0), ("move", 2.0))
        ],
    }
    segment = {
        "name": "n",
        "successes": [True],
        "metadata": {
            "task_program_id": "p",
            "program_segment_id": "p:segment:$.program.items[0]",
            "program_segment_source_path": ["program", "items", 0],
            "validation": {"accepted_mask": [True]},
            "runtime": {"env_ids": [0], "calls": [runtime_call]},
        },
    }
    opening = {
        "finite": True,
        "measured_qpos": [0.0],
        "open_qpos": [0.0],
        "tolerance": 0.01,
    }
    timeline = {
        "errors": [],
        "step_dt": 0.04,
        "object_actors": {"pan": [[10]]},
        "pad_actors": {"left_eef": [[1, 2]], "right_eef": [[3, 4]]},
        "samples": [
            {
                "step": 30,
                "time": 1.2,
                "dropped_contacts": 0,
                "rows": [
                    {
                        "pairs": [[10, 1], [10, 2], [10, 3], [10, 4]],
                        "normals": [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                        "impulses": [1.0] * 4,
                    }
                ],
            },
            {
                "step": 75,
                "time": 3.0,
                "dropped_contacts": 0,
                "rows": [{"pairs": [], "normals": [], "impulses": []}],
                "hand_opening": {
                    "left_eef": [deepcopy(opening)],
                    "right_eef": [deepcopy(opening)],
                },
            },
        ],
    }
    return group, {"nodes": [node]}, {"segments": [segment]}, program, timeline


def test_coordinated_transport_requires_native_acquisition_and_dual_release():
    *_, proof = transport_proof(*_transport_case())
    assert proof["acquired_step"] == 30
    assert proof["release_step"] == 75


def _ordered_e5_case():
    transport = {
        "invocation_id": "program/segment:0",
        "invocation_revision": 0,
        "acquired_step": 87,
        "release_step": 267,
        "contact_timeline_digest": "same-native-timeline",
        "release_receipt_digest": "same-terminal-effect",
    }
    acquired = {
        "measurements": {"scope": "measured_coordinated_transport/v1", **transport}
    }
    placed = {
        "measurements": {
            "scope": "native_stable_support",
            **transport,
            "start_step": 884,
            "end_step": 959,
            "boundary_frames": [959],
        }
    }
    terminal = {"end_steps": [959]}
    return acquired, placed, terminal


def test_e5_ordered_snapshot_uses_measured_acquisition_before_support():
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _ordered_coordinated_placement,
    )

    acquired, placed, terminal = _ordered_e5_case()

    assert _ordered_coordinated_placement(acquired, placed, terminal, env_id=0)


@pytest.mark.parametrize(
    "damage",
    ["foreign_invocation", "reversed_release", "early_support", "stale_boundary"],
)
def test_e5_ordered_snapshot_rejects_changed_event_identity_or_order(damage):
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _ordered_coordinated_placement,
    )

    acquired, placed, terminal = _ordered_e5_case()
    if damage == "foreign_invocation":
        placed["measurements"]["invocation_id"] = "another/segment:0"
    elif damage == "reversed_release":
        placed["measurements"]["release_step"] = 86
        acquired["measurements"]["release_step"] = 86
    elif damage == "early_support":
        placed["measurements"]["start_step"] = 266
    else:
        terminal["end_steps"] = [958]

    assert not _ordered_coordinated_placement(acquired, placed, terminal, env_id=0)


@pytest.mark.parametrize("mode", ["supported", "overhang", "table_support", "hovering"])
def test_e5_on_requires_native_load_not_a_footprint_or_release_proxy(monkeypatch, mode):
    from embodichain.gen_sim.task_engine._task_program.task_spec_runtime import (
        _placement_measurement_matches,
    )
    from embodichain.gen_sim.task_engine._task_program import (
        coordinated_placement_evidence as observer,
    )

    group, graph, runtime, program, timeline = _transport_case()
    graph["task_groups"] = [group]
    timeline["object_actors"]["pad"] = [[20]]
    timeline["object_body_types"] = {"pan": "dynamic", "pad": "dynamic"}
    for step in range(76, 79):
        row = {
            "pairs": [[20 if mode != "table_support" else 99, 10]],
            "normals": [[0, 0, 1]],
            "impulses": [0.01],
        }
        if mode == "hovering":
            row.update(pairs=[], normals=[], impulses=[])
        timeline["samples"].append(
            {
                "step": step,
                "time": step * 0.04,
                "dropped_contacts": 0,
                "rows": [row],
                "substeps": [
                    {"offset": 0.04, "dt": 0.04, "dropped_contacts": 0, "rows": [row]}
                ],
            }
        )
    boundary = runtime["segments"][0]
    boundary["end_steps"] = [78]
    boundary["metadata"]["runtime"]["task_state"] = {
        "held_objects": [],
        "coordinated_held_objects": [],
    }
    cfg = StabilityConstraint(
        entity="pan",
        reference="pad",
        displacement=(0, 0, 0.003),
        kind="placement",
        duration=0.12,
    )
    bindings = tuple(
        SimpleNamespace(entity_id=name, simulation_uid=name) for name in ("pan", "pad")
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(rigid_objects=bindings),
                registered_semantic_lowerer_factories=(
                    SimpleNamespace(
                        call_id="simulation.coordinated_transport",
                        routes=(
                            SimpleNamespace(
                                object_id="pan",
                                target_id="target",
                                reference_entity_id="pad",
                                relation="on",
                            ),
                        ),
                    ),
                ),
            ),
            adapter_factory=SimpleNamespace(constraints=(("gen_sim.n.stable", cfg),)),
        )
    )
    poses = {name: {"pose": torch.zeros(1, 79, 7)} for name in ("pan", "pad")}
    for value in poses.values():
        value["pose"][..., 6] = 1
    poses["pan"]["pose"][..., 2] = 0.03 if mode == "hovering" else 0.003
    env = SimpleNamespace(
        num_envs=1,
        sim=SimpleNamespace(),
        step_dt=0.04,
        _traj_steps=torch.tensor([78]),
        _traj_buffer={"states": {"rigid_objects": poses}},
        _gen_sim_acceptance_contacts=SimpleNamespace(snapshot=lambda: timeline),
    )
    monkeypatch.setattr(
        observer,
        "measure_planar_placement",
        lambda *args, **kwargs: (
            [mode != "overhang"],
            {
                "scope": "measured_planar_landing_region",
                "full_footprint_inside": [mode != "overhang"],
                "support_gap_m": [0.03 if mode == "hovering" else 0.003],
            },
        ),
    )
    if mode == "hovering":
        with pytest.raises(ValueError, match="zero active contacts"):
            observer.capture_transport_placement(
                env,
                deployment,
                graph,
                runtime,
                program,
                group,
                {"object_part_id": None},
                boundary,
                1,
                recorded=True,
            )
        return
    record = observer.capture_transport_placement(
        env,
        deployment,
        graph,
        runtime,
        program,
        group,
        {"object_part_id": None},
        boundary,
        1,
        recorded=True,
    )
    assert record["accepted"] == [mode != "table_support"]
    assert record["measurements"]["landing_region_diagnostic"][
        "full_footprint_inside"
    ] == [mode != "overhang"]
    assert (
        _placement_measurement_matches(
            record,
            SimpleNamespace(
                arguments={
                    "task_type": "E5",
                    "relation": "on",
                    "subject": "obj",
                    "reference": "region",
                }
            ),
            {
                "obj": [{"uid": "pan", "part_id": None}],
                "region": [{"uid": "pad", "part_id": None}],
            },
            1,
            graph=graph,
            constraints={"presets": {"gen_sim.n.stable": asdict(cfg)}},
            segments={s["name"]: s for s in runtime["segments"]},
            program=program,
            contact_timeline=timeline,
        )
        is True
    )
    record["measurements"]["release_timestamp"] -= 0.04
    from embodichain.gen_sim.task_engine._task_program.coordinated_placement_evidence import (
        transport_record_matches,
    )

    assert (
        transport_record_matches(
            record,
            SimpleNamespace(
                arguments={
                    "task_type": "E5",
                    "relation": "on",
                    "subject": "obj",
                    "reference": "region",
                }
            ),
            {
                "obj": [{"uid": "pan", "part_id": None}],
                "region": [{"uid": "pad", "part_id": None}],
            },
            group,
            graph,
            runtime,
            program,
            timeline,
        )
        is False
    )


def test_unresolved_early_grip_is_not_malformed_or_a_release():
    data = _transport_case()
    early = deepcopy(data[4]["samples"][0])
    early.update(step=29, time=29 * 0.04)
    early["rows"][0].update(pairs=[[10, 1]], normals=[[1, 0, 0]], impulses=[1.0])
    data[4]["samples"].insert(0, early)
    assert transport_proof(*data)[-1]["acquired_step"] == 30
    data[4]["samples"][-1]["rows"] = early["rows"]
    with pytest.raises(ValueError, match="not physically open and released"):
        transport_proof(*data)


@pytest.mark.parametrize("damage", ["nonfinite", "zero_normal", "negative_impulse"])
def test_malformed_native_grip_is_not_skipped_as_unresolved(damage):
    data = _transport_case()
    row = data[4]["samples"][0]["rows"][0]
    if damage == "nonfinite":
        row["normals"][0][0] = float("nan")
    elif damage == "zero_normal":
        row["normals"][0] = [0, 0, 0]
    else:
        row["impulses"][0] = -1
    with pytest.raises(ValueError, match="payload is invalid"):
        transport_proof(*data)


@pytest.mark.parametrize(
    "damage",
    [
        "one_hand",
        "closed_drive",
        "foreign_receipt",
        "lost_contacts",
        "same_pads",
        "stale_acquisition",
        "no_release",
    ],
)
def test_coordinated_completion_does_not_replace_physical_evidence(damage):
    data = _transport_case()
    call = data[2]["segments"][0]["metadata"]["runtime"]["calls"][0]
    if damage == "one_hand":
        data[4]["samples"][0]["rows"][0]["pairs"] = [[10, 1], [10, 2], [11, 3], [11, 4]]
    elif damage == "closed_drive":
        data[4]["samples"][1]["hand_opening"]["left_eef"][0]["measured_qpos"] = [0.5]
    elif damage == "foreign_receipt":
        call["effects"][0]["effect_spec"]["invocation_id"] = "other"
    elif damage == "lost_contacts":
        data[4]["samples"][0]["dropped_contacts"] = 1
    elif damage == "same_pads":
        data[4]["pad_actors"]["right_eef"] = deepcopy(data[4]["pad_actors"]["left_eef"])
    elif damage == "stale_acquisition":
        data[4]["samples"][0]["step"] = 10000
    else:
        call["effects"] = []
    with pytest.raises((ValueError, KeyError)):
        transport_proof(*data)
