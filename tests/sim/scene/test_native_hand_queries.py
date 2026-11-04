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

"""Newton native hand query decisions, scopes and owned query lifetimes."""

from __future__ import annotations
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    ClosingContactPolicy,
    NativeClosingQuery,
    NativeContactPart,
    NativeOpeningQuery,
    NativeQueryScope,
    NativeRigidRestScene,
    NativeRigidObstacle,
)
from embodichain.lab.sim.atomic_actions import GraspOpeningTarget

from ._native_hand_fixture import (
    exercise,
    query_inputs,
    recording_manager,
    stable_report,
    decision_digest,
)


@pytest.fixture
def newton_available():
    pytest.importorskip(
        "newton",
        reason="Actual native geometry is unqualified without Newton runtime.",
    )


def _reference():
    return json.loads(
        Path(__file__).with_name("fixtures").joinpath("native_hand_f1.json").read_text()
    )


def _scope():
    return NativeQueryScope("deterministic-test-scope", 1.25, 7)


@pytest.mark.parametrize(
    "options", [{"contact_halo": "unknown"}, {"opening_check": "enabled"}]
)
def test_unknown_or_untyped_geometric_policy_fails_before_query_construction(options):
    with pytest.raises(ValueError):
        ClosingContactPolicy(**options)


@pytest.mark.parametrize("index", range(15))
def test_closing_results_candidate_axes_and_newton_query_order_match_f1_decisions(
    newton_available, index
):
    case = _reference()["closing"][index]
    manager, trace = recording_manager(_ConvexCollection)
    parts, target, options = query_inputs(case["case"], NativeContactPart)
    options["collision_factory"] = manager
    with NativeClosingQuery(parts, target, 0.0, scope=_scope(), **options) as query:
        actual = exercise(query, case["case"])
        assert actual["inspect"]["valid"] is case["expected_valid"]
        assert actual["inspect"]["reason"] == case["expected_reason"]
        assert decision_digest(actual) == case["result_sha256"]
        assert decision_digest(trace) == case["trace_sha256"]


@pytest.mark.parametrize("index", [0, 1])
def test_opening_prefix_and_contact_halo_report_match_f1(newton_available, index):
    case = _reference()["opening"][index]
    manager, trace = recording_manager(_ConvexCollection)
    parts, target, options = query_inputs({"revision": 4}, NativeContactPart)
    options["collision_factory"] = manager
    closing = NativeClosingQuery(parts, target, 0.0, scope=_scope(), **options)
    pose = np.eye(4)
    source = closing.inspect(pose)
    target_pose = np.eye(4)
    target_pose[2, 3] = case["height"]
    opening = NativeOpeningQuery(
        closing,
        (trimesh.creation.box((0.5, 0.5, 0.05)),),
        np.eye(4),
        target_pose,
        0.0,
        0.02,
        identity={"support": "fixture"},
        scope=closing.scope,
    )
    closing.opening_screen = opening
    assert decision_digest(opening.inspect(pose, source)) == case["result_sha256"]
    assert (
        decision_digest(opening.inspect(pose.copy(), source)) == case["repeated_sha256"]
    )
    assert decision_digest(stable_report(opening.report())) == case["report_sha256"]
    assert decision_digest(trace) == case["trace_sha256"]
    report = opening.report()
    closing.close()
    assert opening.closed and opening.report() == report
    with pytest.raises(RuntimeError, match="closed"):
        opening.inspect(pose, source)


@pytest.mark.parametrize("index", [0, 1])
def test_rigid_scene_prefix_uses_original_pair_rest_and_query_order(
    newton_available, index
):
    case = _reference()["scene"][index]
    manager, trace = recording_manager(_ConvexCollection)
    parts, _, _ = query_inputs({"revision": 4}, NativeContactPart)
    pose = np.eye(4)
    pose[2, 3] = case["height"]
    obstacle = NativeRigidObstacle(
        "obstacle", 0.001, (trimesh.creation.box((0.5, 0.5, 0.05)),), pose
    )
    scene = NativeRigidRestScene(
        parts, [obstacle], scope=_scope(), collision_factory=manager
    )
    assert decision_digest(scene.inspect(np.eye(4), 2)) == case["result_sha256"]
    assert decision_digest(scene.inspect(np.eye(4), 2)) == case["repeated_sha256"]
    assert decision_digest(trace) == case["trace_sha256"]
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.inspect(np.eye(4), 2)


@pytest.mark.parametrize("tensor_poses", [False, True])
def test_opening_scope_and_owned_grounding_are_explicit(newton_available, tensor_poses):
    pose = torch.eye(4, dtype=torch.float64) if tensor_poses else np.eye(4)
    identity = {"place_id": "planned-place", "labels": ["owned"]}
    target = GraspOpeningTarget("support", pose, pose, _scope(), identity)
    pose[0, 3] = 9
    identity["labels"].append("external")
    assert target.reference_pose[0, 3] == 0 and target.object_target_pose[0, 3] == 0
    assert target.identity["labels"] == ["owned"]
    assert not target.reference_pose.flags.writeable
    parts, source, options = query_inputs({"revision": 4}, NativeContactPart)
    with NativeClosingQuery(parts, source, 0.0, scope=_scope(), **options) as closing:
        with pytest.raises(ValueError, match="same query/observation"):
            NativeOpeningQuery(
                closing,
                (trimesh.creation.box(),),
                target.reference_pose,
                target.object_target_pose,
                0.0,
                0.01,
                identity={},
                scope=replace(_scope(), query_id="other-prepare"),
            )
        opening = NativeOpeningQuery(
            closing,
            (trimesh.creation.box(),),
            target.reference_pose,
            target.object_target_pose,
            0.0,
            0.01,
            identity={},
            scope=closing.scope,
        )
        closing.opening_screen = opening
        with pytest.raises(ValueError, match="bound query"):
            opening.inspect(
                np.eye(4),
                {
                    "valid": True,
                    "reason": "nominal_bilateral_pad_rest_first",
                    "checked_samples": 2,
                },
            )


@pytest.mark.parametrize(
    "change",
    [
        {"env_id": 1},
        {"scene_version": True},
        {"observation_timestamp": float("nan")},
        {"query_id": " "},
    ],
)
def test_scope_rejects_unsupported_observation_identity(change):
    with pytest.raises(ValueError):
        replace(_scope(), **change)


def test_returned_report_and_contact_grids_do_not_alias_caller_data(newton_available):
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    part = NativeContactPart("pad", (trimesh.creation.box(),), poses, 0.0, True)
    poses[0, 0, 3] = 5
    assert part.tcp_poses[0, 0, 3] == 0 and not part.tcp_poses.flags.writeable
    parts, source, options = query_inputs(
        {"revision": 4, "body": "halo"}, NativeContactPart
    )
    with NativeClosingQuery(parts, source, 0.0, scope=_scope(), **options) as query:
        options["policy"].contact_halo = "reject"
        result = query.inspect(np.eye(4))
        result["valid"] = False
        assert query.inspect(np.eye(4))["valid"] is True
        before_close = query.report()
        assert before_close["collision_backend"] == "newton.CollisionPipeline"
        assert before_close["collision_backend_version"]
        assert before_close["distance_semantics"] == "capped_conversion_interval/v1"
    assert query.closed and query.report() == before_close
    for operation in (
        lambda: query.inspect(np.eye(4)),
        lambda: query.mask(torch.eye(4)[None], torch.eye(4)[None]),
        query.depth_offsets,
        query.record_deferred_proposal,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_caller_part_lists_cannot_change_prepared_newton_queries(newton_available):
    parts, source, options = query_inputs({"revision": 4}, NativeContactPart)
    closing_parts, scene_parts = list(parts), list(parts)
    obstacle_pose = np.eye(4)
    obstacle_pose[2, 3] = 0.4
    targets = [
        NativeRigidObstacle(
            "obstacle", 0.001, (trimesh.creation.box((0.5, 0.5, 0.05)),), obstacle_pose
        )
    ]
    closing = NativeClosingQuery(closing_parts, source, 0.0, scope=_scope(), **options)
    scene = NativeRigidRestScene(
        scene_parts, targets, scope=_scope(), collision_factory=_ConvexCollection
    )
    try:
        closing_parts.clear()
        scene_parts.reverse()
        scene_parts.pop()
        targets.clear()
        assert closing.inspect(np.eye(4))["valid"] is True
        record = scene.inspect(np.eye(4), 2)
        assert record["valid"] is True and record["sample_count"] == 2
        assert [part.name for part in closing.parts] == [part.name for part in parts]
        assert [part.name for part in scene.parts] == [part.name for part in parts]
    finally:
        closing.close()
        scene.close()
