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

"""CPU checks for the actual PickUp and Slide tutorial orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions import AntipodalAffordance, SlideAffordance
from embodichain.toolkits.graspkit import GraspCandidateBatch
from scripts.tutorials.atomic_action import pickup, slide


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_sim
@pytest.mark.parametrize("tutorial", ["pickup", "slide"])
def test_real_affordance_tutorial_replays_distinct_grasps_on_four_envs(
    tmp_path: Path, tutorial: str
) -> None:
    """Exercise real sampling, IK, physics replay and normal process cleanup."""
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / tutorial
    process = subprocess.run(
        [
            sys.executable,
            f"scripts/tutorials/atomic_action/{tutorial}.py",
            "--headless",
            "--num_envs",
            "1",  # The new flag must set the actual simulator batch to four.
            "--n_affordance_multi_gen",
            "4",
            "--affordance_seed",
            "13",
            "--affordance_output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=360,
    )
    assert process.returncode == 0, (process.stdout + process.stderr)[-12000:]
    names = ("pickup",) if tutorial == "pickup" else ("slide_pull", "slide_push")
    parent_ids = None
    for name in names:
        report = json.loads((output / f"{name}.json").read_text())
        assert report["requested"] == 4
        assert report["summary"]["physical_env_count"] == 4
        assert not report["physical_validation"]
        assert report["expert_episodes_committed"] == 0
        with np.load(output / f"{name}.npz", allow_pickle=False) as saved:
            qpos = saved["qpos"]
            # The initial cube/drawer profile must supply four feasible grasps.
            # Push can filter rows after their different physical pull outcomes.
            if name != "slide_push":
                assert len(qpos) == 4
            assert qpos.shape[0] == report["output_count"] <= 4
            assert qpos.shape[2] == 8 and np.isfinite(qpos).all()
            assert saved["success_mask"].shape == (4,)
            np.testing.assert_array_equal(
                saved["env_ids"], np.flatnonzero(saved["success_mask"])
            )
            assert len(set(saved["grasp_ids"])) == len(qpos)
            if len(qpos):
                assert len(np.unique(qpos.reshape(len(qpos), -1), axis=0)) == len(qpos)
                assert np.all(saved["dt"][:, 0] == 0)
                assert np.all(saved["dt"][~saved["valid_mask"]] == 0)
            if name == "slide_push":
                assert report["metadata"]["parent_grasp_ids"] == parent_ids
                for env_id, identity in zip(saved["env_ids"], saved["grasp_ids"]):
                    assert identity == parent_ids[env_id]
            parent_ids = report["grasp_ids"]


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        approach="top",
        affordance_seed=13,
        affordance_output=None,
        approach_distance=0.1,
        translation_distance=0.18,
    )


def _raw_batch() -> GraspCandidateBatch:
    return GraspCandidateBatch(
        torch.eye(4).repeat(2, 2, 1, 1),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.ones(2, 2, dtype=torch.bool),
        grasp_ids=(("grasp-a", "grasp-b"),) * 2,
    )


def _plan_result(success: list[bool]) -> SimpleNamespace:
    raw_poses = torch.eye(4).repeat(2, 1, 1)
    raw_poses[:, 2, 3] = 0.2
    return SimpleNamespace(
        success_mask=torch.tensor(success),
        trajectory=object(),
        grasp_poses=raw_poses,
        grasp_ids=tuple(
            f"grasp-{'ab'[row]}" if accepted else None
            for row, accepted in enumerate(success)
        ),
    )


@pytest.mark.parametrize("success", [[True, False], [False, False]])
def test_pickup_multi_path_uses_raw_candidates_and_skips_empty_replay(
    monkeypatch: pytest.MonkeyPatch, success: list[bool]
) -> None:
    from embodichain.lab.trajectory_generation.integrations import atomic_affordance
    from scripts.tutorials.atomic_action import affordance_utils

    result, grasps = _plan_result(success), _raw_batch()
    planner = MagicMock(return_value=result)
    sampler = MagicMock(return_value=grasps)
    saver, replay = MagicMock(), MagicMock()
    monkeypatch.setattr(atomic_affordance, "plan_affordance_batch", planner)
    monkeypatch.setattr(affordance_utils, "sample_affordance_grasps", sampler)
    monkeypatch.setattr(affordance_utils, "save_affordance_result", saver)
    monkeypatch.setattr(pickup, "replay_trajectory", replay)
    sim = SimpleNamespace(
        device=torch.device("cpu"), sim_config=SimpleNamespace(physics_dt=0.01)
    )
    robot = SimpleNamespace(num_instances=2)
    obj = SimpleNamespace(get_local_pose=lambda **_: torch.eye(4).repeat(2, 1, 1))
    semantics = SimpleNamespace(
        affordance=AntipodalAffordance(
            mesh_vertices=torch.zeros(3, 3), mesh_triangles=torch.tensor([[0, 1, 2]])
        )
    )
    engine = MagicMock()
    monkeypatch.setattr(pickup, "GraspGoal", lambda value: value)

    pickup._run_affordance_pickup(
        sim, robot, obj, engine, object(), semantics, _args(), False
    )

    sampler.assert_called_once()
    assert planner.call_args.args[3] is grasps
    assert (
        engine.make_invocation.call_args.kwargs["motion_policy"].strategy == "ik_interp"
    )
    assert saver.call_args.kwargs["physical_envs"] == 2
    assert replay.call_count == int(any(success))
    if any(success):
        assert replay.call_args.args[2] is result.trajectory


def test_slide_push_reuses_selected_grasp_at_each_observed_handle_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.trajectory_generation.integrations import atomic_affordance
    from scripts.tutorials.atomic_action import affordance_utils

    initial = torch.eye(4).repeat(2, 1, 1)
    initial[:, 0, 3] = -1.1
    observed = initial.clone()
    pull_result = _plan_result([True, True])
    # The two grasps differ in handle-local Z; world poses belong to each row.
    pull_result.grasp_poses = initial.clone()
    pull_result.grasp_poses[:, 2, 3] = torch.tensor([0.2, 0.3])
    push_result = _plan_result([True, False])
    planner = MagicMock(side_effect=[pull_result, push_result])
    monkeypatch.setattr(atomic_affordance, "plan_affordance_batch", planner)
    monkeypatch.setattr(
        affordance_utils,
        "sample_affordance_grasps",
        MagicMock(return_value=_raw_batch()),
    )
    saver = MagicMock()
    monkeypatch.setattr(affordance_utils, "save_affordance_result", saver)
    invocation_calls = []

    def make_invocation(engine, semantics, target_pose, **kwargs):
        invocation_calls.append((target_pose.clone(), kwargs))
        return object()

    def replay(*args, **kwargs):
        if kwargs["video_prefix"].startswith("slide_pull"):
            # Actual movement differs between physical rows and the command.
            observed[:, 0, 3] += torch.tensor([0.11, 0.16])

    monkeypatch.setattr(slide, "create_invocation", make_invocation)
    monkeypatch.setattr(slide, "replay_trajectory", replay)
    sim = SimpleNamespace(
        device=torch.device("cpu"), sim_config=SimpleNamespace(physics_dt=0.01)
    )
    drawer = SimpleNamespace(get_link_pose=lambda *args, **kwargs: observed)
    semantics = SimpleNamespace(
        affordance=SlideAffordance(
            mesh_vertices=torch.zeros(3, 3), mesh_triangles=torch.tensor([[0, 1, 2]])
        )
    )
    engine = MagicMock()

    slide._run_affordance_slides(
        sim,
        SimpleNamespace(num_instances=2),
        drawer,
        engine,
        object(),
        semantics,
        _args(),
        False,
    )

    assert planner.call_count == 2
    assert engine.initial_context.call_count == 2
    push_grasps = planner.call_args_list[1].args[3]
    expected = observed @ torch.linalg.inv(initial) @ pull_result.grasp_poses
    torch.testing.assert_close(push_grasps.poses[:, 0], expected)
    assert push_grasps.grasp_ids == (("grasp-a",), ("grasp-b",))
    torch.testing.assert_close(
        planner.call_args_list[1].kwargs["active_mask"], pull_result.success_mask
    )
    torch.testing.assert_close(invocation_calls[1][0], observed)
    assert [kwargs["direction"] for _, kwargs in invocation_calls] == ["pull", "push"]
    assert (
        saver.call_args.kwargs["metadata"]["parent_grasp_ids"] == pull_result.grasp_ids
    )


def test_slide_all_failed_pull_does_not_plan_or_replay_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.lab.trajectory_generation.integrations import atomic_affordance
    from scripts.tutorials.atomic_action import affordance_utils

    planner = MagicMock(return_value=_plan_result([False, False]))
    replay = MagicMock()
    monkeypatch.setattr(atomic_affordance, "plan_affordance_batch", planner)
    monkeypatch.setattr(
        affordance_utils,
        "sample_affordance_grasps",
        MagicMock(return_value=_raw_batch()),
    )
    monkeypatch.setattr(affordance_utils, "save_affordance_result", MagicMock())
    monkeypatch.setattr(slide, "create_invocation", MagicMock())
    monkeypatch.setattr(slide, "replay_trajectory", replay)
    sim = SimpleNamespace(
        device=torch.device("cpu"), sim_config=SimpleNamespace(physics_dt=0.01)
    )
    drawer = SimpleNamespace(
        get_link_pose=lambda *args, **kwargs: torch.eye(4).repeat(2, 1, 1)
    )
    semantics = SimpleNamespace(
        affordance=SlideAffordance(
            mesh_vertices=torch.zeros(3, 3), mesh_triangles=torch.tensor([[0, 1, 2]])
        )
    )

    slide._run_affordance_slides(
        sim,
        SimpleNamespace(num_instances=2),
        drawer,
        MagicMock(),
        object(),
        semantics,
        _args(),
        False,
    )

    planner.assert_called_once()
    replay.assert_not_called()
