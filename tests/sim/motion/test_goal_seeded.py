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

"""Goal-seeded required waypoints retain source solve order and qualification."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import torch
import pytest

from embodichain.lab.sim.motion.goal_seeded import solve_goal_seeded_waypoints
from embodichain.lab.sim.motion.search import search_cost_scope


def fixture(mode="valid"):
    calls = []
    solver = SimpleNamespace(
        get_iteration_params=lambda: {"pos_eps": 1e-4, "rot_eps": 1e-4}
    )

    def fk(*, qpos, name, to_matrix):
        result = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        result[..., 0, 3] = qpos.sum(-1)
        return result

    def ik(*, pose, name, joint_seed):
        calls.append((pose.clone(), joint_seed.clone()))
        result = joint_seed.clone()
        result[..., 0] = pose[..., 0, 3] - result[..., 1]
        valid = torch.ones(result.shape[:-1], dtype=torch.bool)
        if mode == "release_failed" and result.ndim == 2:
            valid[:] = False
        if mode == "partial_mask" and result.ndim == 3:
            valid[0, 1] = False
        if mode == "false_fk" and result.ndim == 3:
            result[0, 1, 0] += 0.02
        if mode == "aggregate_mask" and result.ndim == 3:
            valid = torch.ones(1, dtype=torch.bool)
        return valid, result

    robot = SimpleNamespace(
        get_qpos_limits=lambda name: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
        get_solver=lambda name: solver,
        compute_ik=ik,
        compute_batch_ik=ik,
        compute_fk=fk,
        compute_batch_fk=fk,
    )
    poses = torch.eye(4).repeat(1, 4, 1, 1)
    poses[0, :, 0, 3] = torch.tensor([0.1, 0.3, 0.4, 0.2])
    start = torch.tensor([[0.0, 0.1]])
    return robot, poses, start, calls


def test_goal_seed_then_all_mandatory_poses_use_same_seed():
    robot, poses, start, calls = fixture()
    state = torch.get_rng_state().clone()
    with search_cost_scope() as costs:
        ok, q, receipt = solve_goal_seeded_waypoints(
            robot, "arm", poses, start, goal_index=2
        )
    assert ok.tolist() == [True] and receipt["waypoints"]["valid"]
    assert len(calls) == 2
    assert torch.equal(calls[0][0], poses[:, 2]) and torch.equal(calls[0][1], start)
    release = torch.tensor([[0.3, 0.1]])
    assert torch.equal(calls[1][1], release[:, None].expand(-1, 4, -1))
    assert torch.equal(q[..., 0], poses[..., 0, 3] - 0.1)
    assert torch.equal(state, torch.get_rng_state())
    assert costs.counters["goal_seed_ik"] == 1
    assert costs.counters["goal_seeded_waypoints"] == 4
    # Result and solver inputs cannot alias the caller's observed start.
    q.zero_()
    assert torch.equal(start, torch.tensor([[0.0, 0.1]]))


@pytest.mark.parametrize(
    "mode", ["valid", "release_failed", "partial_mask", "false_fk", "aggregate_mask"]
)
def test_invalid_goal_or_any_mandatory_pose_returns_false_hold(mode):
    robot, poses, start, calls = fixture(mode)
    ok, q, receipt = solve_goal_seeded_waypoints(
        robot, "arm", poses, start, goal_index=2
    )
    trace = {
        "ok": ok.tolist(),
        "q": q.tolist(),
        "receipt": receipt,
        "calls": [[p.tolist(), s.tolist()] for p, s in calls],
    }
    oracle = json.loads(
        (Path(__file__).parent / "data/goal_seeded_source.json").read_text()
    )
    assert (
        hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
        == oracle[mode]
    )
    assert ok.tolist() == [mode == "valid"]
    if mode != "valid":
        assert torch.equal(q, start[:, None].expand(-1, 4, -1))
        assert receipt["reason"] == "required_waypoint_ik_fk_limits_rejected"
    assert len(calls) == (1 if mode == "release_failed" else 2)


def test_existing_branch_is_requalified_and_skips_redundant_goal_ik():
    robot, poses, start, calls = fixture()
    ok, _, _ = solve_goal_seeded_waypoints(
        robot,
        "arm",
        poses,
        start,
        goal_index=2,
        qualified_goal_seed=torch.tensor([[0.2, 0.2]]),
    )
    assert ok.all() and len(calls) == 1
    assert torch.equal(calls[0][1], torch.tensor([[[0.2, 0.2]]]).expand(-1, 4, -1))
    calls.clear()
    ok, _, receipt = solve_goal_seeded_waypoints(
        robot,
        "arm",
        poses,
        start,
        goal_index=2,
        qualified_goal_seed=torch.tensor([[0.1, 0.1]]),
    )
    assert not ok.any() and not calls and not receipt["release_seed"]["fk_valid"]


def test_batch_input_is_rejected_before_solving():
    robot, poses, start, calls = fixture()
    with pytest.raises(ValueError, match="B1"):
        solve_goal_seeded_waypoints(
            robot, "arm", poses.repeat(2, 1, 1, 1), start.repeat(2, 1), goal_index=2
        )
    assert not calls


def test_bounded_endpoint_uses_original_limit_intersection_and_counts_only_ik():
    from embodichain.lab.sim.motion.goal_seeded import solve_bounded_endpoint

    calls = []
    counts = []
    solver = SimpleNamespace(
        lower_qpos_limits=torch.tensor([-1.0, -1.0]),
        upper_qpos_limits=torch.tensor([1.0, 1.0]),
    )
    answer = torch.tensor([[0.4, 0.6]])

    def solve(**kwargs):
        calls.append(kwargs)
        return torch.tensor([True]), answer

    robot = SimpleNamespace(
        get_qpos_limits=lambda **kw: torch.tensor([[[-0.5, 0.5], [-0.5, 0.5]]]),
        get_solver=lambda name: solver,
        compute_ik=solve,
    )
    accepted, q, report = solve_bounded_endpoint(
        robot,
        "arm",
        torch.eye(4)[None],
        torch.zeros(1, 2),
        env_ids=torch.tensor([0]),
        record_operation=lambda name, count: counts.append((name, count)),
    )
    assert accepted is False and report["solver_success"] is True
    assert report["within_runtime_and_solver_limits"] is False
    assert torch.equal(q, answer) and q.data_ptr() != answer.data_ptr()
    assert len(calls) == 1 and counts == [("ik_queries", 1)]
    solver.lower_qpos_limits[0] = float("nan")
    with pytest.raises(ValueError, match="joint_limits_unavailable"):
        solve_bounded_endpoint(
            robot,
            "arm",
            torch.eye(4)[None],
            torch.zeros(1, 2),
            env_ids=torch.tensor([0]),
            record_operation=lambda name, count: counts.append((name, count)),
        )
    assert len(calls) == 1 and counts == [("ik_queries", 1)]
