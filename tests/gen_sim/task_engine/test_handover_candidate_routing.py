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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.lab.sim.atomic_actions import HandOverOptions, MotionPolicy
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
    AffordanceSamplingContext,
)


@pytest.mark.parametrize(
    "mask,expected_id",
    [([True] * 4, 1), ([True, False, True, True], 3), ([False] * 4, -1)],
)
@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("sampling", [None, AffordanceSamplingContext(count=2, seed=0)])
@pytest.mark.parametrize("nonfinite", [False, True])
@pytest.mark.parametrize("station_owned", [False, True])
def test_receiver_filters_the_original_pool_without_resampling(
    monkeypatch, mask, expected_id, configured, sampling, nonfinite, station_owned
):
    from embodichain.gen_sim.task_engine._task_program import handover_exchange

    poses = torch.eye(4).repeat(1, 4, 1, 1)
    poses[0, :, 0, 3] = torch.arange(4)
    costs = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    if nonfinite:
        costs[0, 0] = torch.inf
        poses[0, 2, 0, 0] = torch.nan
    pool = AffordancePoseCandidates(poses, costs, torch.ones(1, 4, dtype=torch.bool))
    frozen = tuple(item.clone() for item in (pool.poses, pool.costs, pool.valid))
    affordance = AntipodalAffordance()
    affordance.get_grasp_candidates = Mock(return_value=pool)
    affordance.sample_candidates = Mock(wraps=affordance.sample_candidates)
    destination = torch.eye(4)[None]
    destination[:, 1, 3] = 1
    robot = SimpleNamespace(compute_fk=Mock(return_value=destination))
    generator = object()
    service = SimpleNamespace(
        robot=robot,
        device=torch.device("cpu"),
        grasp_pose_generator=Mock(return_value=generator),
    )
    action = actions.GenSimHandOver()
    action._planning_services = service
    monkeypatch.setattr(
        actions, "observed_axis", lambda *_: torch.tensor([[0.0, 0.0, 1.0]])
    )
    resources = SimpleNamespace(
        second=SimpleNamespace(arm=SimpleNamespace(joint_ids=(0,), control_part="arm"))
    )
    request = SimpleNamespace(
        motion_policy=MotionPolicy(sample_count=260),
        skill_options=HandOverOptions(
            release_at_target=False,
            pre_grasp_distance=0.15,
            hand_interp_steps=16,
            hold_steps=8,
            retreat_steps=36,
        ),
    )

    def inspect(original, eligible, **kwargs):
        assert original is pool.poses
        assert eligible.tolist() == [[not nonfinite, True, not nonfinite, True]]
        assert kwargs == {
            "resources": resources,
            "approach_samples": 92,
            "pre_grasp_distance": request.skill_options.pre_grasp_distance,
        }
        return eligible & torch.tensor([mask]), {"scope": "nominal_open_hand"}

    screen = SimpleNamespace(candidate_mask=Mock(side_effect=inspect))
    held = SimpleNamespace(object_to_eef=torch.eye(4)[None])
    selection = actions._ReceiveSelection(
        resources, held, torch.tensor([False]), request, screen if configured else None
    )
    if station_owned:
        selection.exchange_pose = torch.eye(4)[None]
        selection.root_poses = (torch.eye(4)[None], torch.eye(4)[None])

    def select_station(*args):
        assert args[9] is pool
        assert not args[10].any()
        station = args[6].clone()
        station[:, 0, 3] += 0.125
        translated = pool.poses.clone()
        translated[:, :, 0, 3] += 0.125
        eligible = (
            pool.valid
            & torch.isfinite(pool.costs)
            & torch.isfinite(pool.poses).all(dim=(-1, -2))
        )
        return station, translated, eligible, {"accepted": True}

    select = Mock(side_effect=select_station)
    monkeypatch.setattr(handover_exchange, "select_exchange", select)
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 1)),
        affordance_sampling=sampling,
        env_ids=torch.tensor([0]),
    )
    state = torch.random.get_rng_state().clone()
    token = actions._RECEIVE_SELECTION.set(selection)
    try:
        result = action._resolve_grasp(
            affordance,
            torch.eye(4)[None],
            torch.tensor([[0.0, 0.0, -1.0]]),
            "hand",
            obj_longest_axis=None,
            is_positive_part=None,
            context=context,
            sample_key="same_call:receive_grasp",
        )
    finally:
        actions._RECEIVE_SELECTION.reset(token)
    screened = configured and sampling is None
    relocated = screened and station_owned and expected_id == -1
    wanted = expected_id if screened else 1
    if relocated:
        wanted = 1
    assert result.metadata["candidate_ids"] == [wanted]
    assert result.success.tolist() == [wanted >= 0]
    if wanted >= 0:
        expected_pose = pool.poses[0, wanted].clone()
        if relocated:
            expected_pose[0, 3] += 0.125
        assert torch.equal(result.poses[0], expected_pose)
    affordance.get_grasp_candidates.assert_called_once()
    affordance.sample_candidates.assert_called_once()
    service.grasp_pose_generator.assert_called_once_with("hand")
    assert screen.candidate_mask.call_count == int(screened)
    assert select.call_count == int(relocated)
    for before, after in zip(frozen, (pool.poses, pool.costs, pool.valid), strict=True):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
    assert torch.equal(state, torch.random.get_rng_state())
    assert selection.sampled
    if station_owned:
        assert result.metadata["reference_poses"] == selection.exchange_pose.tolist()
    evidence = result.metadata.get("gen_sim_handover_candidate_environment")
    if screened:
        assert evidence["raw_valid_count"] == [4]
        assert evidence["sampler_eligible_count"] == [2 if nonfinite else 4]
        expected_indices = [1, 3] if nonfinite else list(range(4))
        expected_costs = [1, 2] if nonfinite else [4, 1, 3, 2]
        assert [
            item["pool_index"] for item in evidence["original_candidates"]
        ] == expected_indices
        assert [
            item["cost"] for item in evidence["original_candidates"]
        ] == expected_costs
    else:
        assert evidence is None
