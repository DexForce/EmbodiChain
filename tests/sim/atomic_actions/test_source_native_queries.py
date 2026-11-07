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

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeClosingQuery,
    NativeContactPart,
    NativeOpeningQuery,
    NativeOpeningTarget,
    NativeQueryScope,
    NativeRigidRestScene,
)
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
from embodichain.lab.sim.scene.feasibility.collision import UnsupportedFeasibilityError
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeRigidObstacle
from ._source_native_grasp_fixture import (
    exercise,
    query_inputs,
    recording_manager,
    stable_report,
    decisions,
)


@pytest.fixture
def newton_available():
    pytest.importorskip(
        "newton",
        reason="Actual native geometry is unqualified without the Newton runtime.",
    )


def _reference():
    return json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("native_hand_f1_queries.json")
        .read_text()
    )


def _scope():
    return NativeQueryScope("deterministic-test-scope", 1.25, 7)


@pytest.mark.parametrize("index", range(15))
def test_closing_results_candidate_axes_and_query_order_match_f1(
    newton_available, monkeypatch, index
):
    case = _reference()["closing"][index]
    manager, trace = recording_manager(_ConvexCollection)
    from embodichain.lab.sim.atomic_actions import source_native_contact

    monkeypatch.setattr(source_native_contact, "_ConvexCollection", manager)
    parts, target, options = query_inputs(case["case"], NativeContactPart)
    with NativeClosingQuery(parts, target, 0.0, scope=_scope(), **options) as query:
        actual = exercise(query, case["case"])
        assert decisions(actual) == decisions(case["result"])
        assert decisions(trace) == decisions(case["trace"])
        before_close = query.report()
    assert query.closed and query.report() == before_close
    for operation in (
        lambda: query.inspect(np.eye(4)),
        lambda: query.mask(torch.eye(4)[None], torch.eye(4)[None]),
        query.depth_offsets,
        query.record_deferred_proposal,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("index", [0, 1])
def test_opening_prefix_and_contact_halo_report_match_f1(
    newton_available, monkeypatch, index
):
    case = _reference()["opening"][index]
    manager, trace = recording_manager(_ConvexCollection)
    from embodichain.lab.sim.atomic_actions import source_native_contact

    monkeypatch.setattr(source_native_contact, "_ConvexCollection", manager)
    parts, target, options = query_inputs({"revision": 4}, NativeContactPart)
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
    assert decisions(stable_report(opening.inspect(pose, source))) == decisions(
        case["result"]
    )
    assert decisions(stable_report(opening.inspect(pose.copy(), source))) == decisions(
        case["repeated"]
    )
    assert decisions(stable_report(opening.report())) == decisions(case["report"])
    assert decisions(trace) == decisions(case["trace"])
    report = opening.report()
    closing.close()
    assert opening.closed and opening.report() == report
    with pytest.raises(RuntimeError, match="closed"):
        opening.inspect(pose, source)


@pytest.mark.parametrize("index", [0, 1])
def test_rigid_scene_prefix_uses_original_pair_rest_and_query_order(
    newton_available, monkeypatch, index
):
    case = _reference()["scene"][index]
    manager, trace = recording_manager(_ConvexCollection)
    from embodichain.lab.sim.atomic_actions import source_native_contact

    monkeypatch.setattr(source_native_contact, "_ConvexCollection", manager)
    parts, _, _ = query_inputs({"revision": 4}, NativeContactPart)
    pose = np.eye(4)
    pose[2, 3] = case["height"]
    obstacle = NativeRigidObstacle(
        "obstacle", 0.001, (trimesh.creation.box((0.5, 0.5, 0.05)),), pose
    )
    scene = NativeRigidRestScene(
        parts, [obstacle], scope=_scope(), collision_factory=manager
    )
    assert decisions(stable_report(scene.inspect(np.eye(4), 2))) == decisions(
        case["result"]
    )
    assert decisions(stable_report(scene.inspect(np.eye(4), 2))) == decisions(
        case["repeated"]
    )
    assert decisions(trace) == decisions(case["trace"])
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.inspect(np.eye(4), 2)


@pytest.mark.parametrize("tensor_poses", [False, True])
def test_opening_scope_and_owned_grounding_are_explicit(newton_available, tensor_poses):
    pose = torch.eye(4, dtype=torch.float64) if tensor_poses else np.eye(4)
    identity = {"place_id": "planned-place", "labels": ["owned"]}
    target = NativeOpeningTarget("support", pose, pose, _scope(), identity)
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
    parts, source, options = query_inputs({"revision": 4}, NativeContactPart)
    with NativeClosingQuery(parts, source, 0.0, scope=_scope(), **options) as query:
        result = query.inspect(np.eye(4))
        result["valid"] = False
        assert query.inspect(np.eye(4))["valid"] is True


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


@pytest.mark.parametrize(
    "reason", ["precision interval crosses threshold", "contact buffer saturated"]
)
def test_unavailable_native_clearance_cannot_qualify(
    newton_available, monkeypatch, reason
):
    parts, meshes, options = query_inputs({"revision": 4}, NativeContactPart)

    def unavailable(*args):
        raise UnsupportedFeasibilityError(reason)

    monkeypatch.setattr(_ConvexCollection, "distance_interval", unavailable)
    with NativeClosingQuery(parts, meshes, 0.0, scope=_scope(), **options) as query:
        with pytest.raises(UnsupportedFeasibilityError, match=reason):
            query.inspect(np.eye(4))


@pytest.mark.parametrize("kind", ["open_surface", "nonconvex"])
def test_source_query_rejects_unsupported_geometry_without_convexifying(
    newton_available, kind
):
    parts, _, options = query_inputs({"revision": 4}, NativeContactPart)
    if kind == "open_surface":
        mesh = trimesh.creation.box()
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[:-1], process=False
        )
    else:
        first = trimesh.creation.box()
        second = trimesh.creation.box()
        second.apply_translation([2.0, 0.0, 0.0])
        mesh = trimesh.util.concatenate([first, second])
    with pytest.raises(UnsupportedFeasibilityError, match="qualified convex solids"):
        NativeClosingQuery(parts, (mesh,), 0.0, scope=_scope(), **options)


def test_exact_contact_at_precision_boundary_remains_unavailable(newton_available):
    parts, target, options = query_inputs({"revision": 4}, NativeContactPart)
    touching = []
    for part, sign in zip(parts, [-1, 1], strict=True):
        poses = part.tcp_poses.copy()
        # The target half-width is .1 m and each finger half-width is .01 m.
        # Exact surface contact cannot establish either side of a zero threshold.
        poses[0, 0, 3] = sign * 0.11
        touching.append(replace(part, tcp_poses=poses))
    with NativeClosingQuery(touching, target, 0.0, scope=_scope(), **options) as query:
        with pytest.raises(
            UnsupportedFeasibilityError, match="precision interval crosses"
        ):
            query.inspect(np.eye(4))


def test_explicit_fcl_zero_rest_contact_retains_source_policy_and_owned_cache():
    parts, target, options = query_inputs({"revision": 4}, NativeContactPart)
    touching = []
    for part, sign in zip(parts, [-1, 1], strict=True):
        poses = part.tcp_poses.copy()
        poses[0, 0, 3] = sign * 0.11
        touching.append(replace(part, tcp_poses=poses))
    pose = np.eye(4)
    with NativeClosingQuery(
        touching, target, 0.0, scope=_scope(), collision_backend="fcl", **options
    ) as query:
        result = query.inspect(pose)
        assert result["valid"] is True
        assert result["reason"] == "nominal_bilateral_pad_rest_first"
        assert result["checked_samples"] == 1
        assert query.report()["collision_backend"].endswith("/python-fcl")
        assert query.collision_backend == "fcl"
        with pytest.raises(AttributeError):
            query.collision_backend = "newton"
        assert query.inspect(pose.copy()) == result and len(query.cache) == 1
        with NativeClosingQuery(
            touching, target, 0.0, scope=_scope(), **options
        ) as default:
            assert default.collision_backend == "newton"
            assert default.cache == {} and default.cache is not query.cache
            opening = NativeOpeningQuery(
                default,
                (trimesh.creation.box((0.5, 0.5, 0.05)),),
                np.eye(4),
                np.eye(4),
                0.0,
                0.02,
                identity={},
                scope=default.scope,
            )
            try:
                with pytest.raises(ValueError, match="bound query"):
                    opening.inspect(pose, result)
            finally:
                opening.close()


def test_explicit_fcl_unavailable_never_falls_back_to_newton(monkeypatch):
    from embodichain.lab.sim.atomic_actions import source_native_contact as owner

    def absent():
        raise ImportError("controlled FCL unavailable")

    def forbidden():
        pytest.fail("An explicitly missing FCL backend fell back to Newton")

    parts, target, options = query_inputs({"revision": 4}, NativeContactPart)
    monkeypatch.setattr(owner, "_require_fcl", absent)
    monkeypatch.setattr(owner, "_require_newton", forbidden)
    with pytest.raises(ImportError, match="controlled FCL unavailable"):
        NativeClosingQuery(
            parts, target, 0.0, scope=_scope(), collision_backend="fcl", **options
        )


def test_newton_precision_interval_keeps_the_original_unavailable_boundary():
    from embodichain.lab.sim.scene._newton_queries import _ClearanceInterval

    with pytest.raises(UnsupportedFeasibilityError, match="precision interval crosses"):
        _ClearanceInterval(0.0, 2.240103185613407e-06).below(0.0)
