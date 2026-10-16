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

"""CPU checks for semantic boundaries in handwritten expert tasks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain_tasks.manipulation.tableware import blocks_ranking_rgb as ranking
from embodichain_tasks.manipulation.tableware import stack_blocks_two as stacking
from embodichain_tasks.manipulation.tableware.scoop_ice import ScoopIce
from embodichain_tasks.special.simple_task import SimpleTaskEnv
from embodichain_tasks.special.stay_still_save import StayStillSaveEnv


def _block(pose: torch.Tensor) -> SimpleNamespace:
    """Return a mutable measured pose with stationary body velocities."""
    return SimpleNamespace(
        get_local_pose=lambda **kwargs: pose,
        clear_dynamics=Mock(),
        body_data=SimpleNamespace(
            lin_vel=torch.zeros(pose.shape[0], 3),
            ang_vel=torch.zeros(pose.shape[0], 3),
        ),
    )


def test_stack_segments_preserve_commands_and_settle_only_after_release() -> None:
    """Splitting does not drop, duplicate, or reorder controller commands."""
    env = stacking.StackBlocksTwoEnv.__new__(stacking.StackBlocksTwoEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    source = torch.eye(4).unsqueeze(0)
    measured = source.clone()
    env._stack_block = _block(measured)
    env._base_block = _block(source)
    pick_commands = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    place_commands = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + 10
    env._plan_stack = Mock(
        return_value=(
            torch.tensor([True]),
            torch.tensor([True]),
            pick_commands,
            place_commands,
            source.clone(),
            source.clone(),
        )
    )
    segments = iter(env.create_demo_segments())
    pick = next(segments)
    frames = list(pick.actions)
    measured[:, 2, 3] = stacking.MIN_PICK_LIFT_HEIGHT * 2
    assert pick.validator().tolist() == [True]
    assert len(frames) == pick_commands.shape[1]
    place = next(segments)
    frames.extend(place.actions)
    measured[:, 2, 3] = stacking.BLOCK_HEIGHT
    assert place.validator().tolist() == [True]
    with pytest.raises(StopIteration):
        next(segments)

    expected = torch.cat(
        (
            pick_commands,
            place_commands,
            place_commands[:, -1:].repeat(1, stacking.SETTLE_STEPS, 1),
        ),
        dim=1,
    )
    torch.testing.assert_close(torch.stack(frames, dim=1), expected)
    assert [pick.name, place.name] == ["pick_block_2", "place_block_2_on_block_1"]
    assert pick.metadata["atomic_actions"] == ["pick_up"]
    assert place.metadata["atomic_actions"] == ["place"]
    assert pick.instruction != place.instruction
    env._stack_block.clear_dynamics.assert_called_once()
    env._plan_stack.assert_called_once()


def test_ranking_plans_next_block_only_after_grasp_and_placement() -> None:
    """Each block retains its two original stages and its release-only settling."""
    env = ranking.BlocksRankingRGBEnv.__new__(ranking.BlocksRankingRGBEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    poses = {
        uid: torch.eye(4).unsqueeze(0) for uid in ("block_1", "block_2", "block_3")
    }
    env._blocks = {uid: _block(pose) for uid, pose in poses.items()}
    env._target_position = lambda offset: torch.tensor([[offset, 0.0, 0.0]])
    pick_steps = (
        round((ranking.PICK_SAMPLE_INTERVAL - ranking.HAND_INTERP_STEPS) * 0.6)
        + ranking.HAND_INTERP_STEPS
        + ranking.GRASP_HOLD_STEPS
    )
    place_steps = 4

    def plan_block(*, uid: str, **kwargs: object) -> tuple:
        del kwargs
        value = 1.0 if uid == "block_1" else 3.0
        return (
            torch.tensor([True]),
            torch.tensor([True]),
            torch.full((1, pick_steps, 2), value),
            torch.full((1, place_steps, 2), value + 1),
            poses[uid].clone(),
        )

    env._plan_block_segment = Mock(side_effect=plan_block)
    names = []
    stages = []
    for index, segment in enumerate(env.create_demo_segments()):
        assert env._plan_block_segment.call_count == index // 2 + 1
        names.append(segment.name)
        frames = list(segment.actions)
        stages.append(torch.stack(frames, dim=1))
        uid = segment.target_uid
        if index % 2 == 0:
            poses[uid][:, 2, 3] = ranking.MIN_PICK_LIFT_HEIGHT * 2
        else:
            poses[uid][:, :3, 3] = torch.tensor(segment.metadata["target_position"])
        assert segment.validator().tolist() == [True]
        assert segment.metadata["segment_index"] == index
        assert segment.metadata["segment_count"] == 4

    assert names == [
        "pick_red_block",
        "place_red_block",
        "pick_blue_block",
        "place_blue_block",
    ]
    settling_steps = ranking.SETTLE_MIN_STEPS + ranking.SETTLE_STABLE_STEPS - 1
    for index, commands in enumerate(stages):
        expected_steps = pick_steps if index % 2 == 0 else place_steps + settling_steps
        assert commands.shape[1] == expected_steps
        assert (commands == index + 1).all()
    env._blocks["block_1"].clear_dynamics.assert_called_once()
    env._blocks["block_3"].clear_dynamics.assert_called_once()


@pytest.mark.parametrize("task", ["stacking", "ranking"])
def test_grasp_acceptance_requires_finite_measured_lift_and_successful_plan(
    task: str,
) -> None:
    source = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
    measured = source.clone()
    measured[:, 2, 3] = torch.tensor([0.1, 0.01, float("nan"), 0.1])
    planned = torch.tensor([True, True, True, False])
    if task == "stacking":
        env = stacking.StackBlocksTwoEnv.__new__(stacking.StackBlocksTwoEnv)
        env._stack_block = _block(measured)
        env.sim = SimpleNamespace(device=torch.device("cpu"))
        accepted = env._validate_pick(planned, source)
    else:
        env = ranking.BlocksRankingRGBEnv.__new__(ranking.BlocksRankingRGBEnv)
        env._blocks = {"block_1": _block(measured)}
        env.sim = SimpleNamespace(device=torch.device("cpu"))
        accepted = env._validate_block_pick("block_1", planned, source)
    assert accepted.tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    ("task_class", "segment_name"),
    [
        (SimpleTaskEnv, "oscillate_joints"),
        (StayStillSaveEnv, "hold_still_100_steps"),
        (ScoopIce, "scoop_ice"),
    ],
)
def test_single_goal_tasks_preserve_existing_action_lists(
    task_class: type, segment_name: str
) -> None:
    """Adding explicit labels preserves the old planner and its command objects."""
    env = task_class.__new__(task_class)
    actions = [torch.full((1, 3), float(index)) for index in range(3)]
    env.create_demo_action_list = Mock(return_value=actions)
    (segment,) = tuple(env.create_demo_segments())
    assert segment.actions is actions
    assert segment.name == segment_name
    assert segment.instruction and segment.instruction != "unknown_task"
    assert segment.progress_total_steps == len(actions)
    env.create_demo_action_list.assert_called_once()
