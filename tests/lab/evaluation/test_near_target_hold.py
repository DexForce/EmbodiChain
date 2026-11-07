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

"""Compatibility and replay of bounded holds over the host's raw observations."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.evaluation import (
    EvaluationSession,
    NearTargetHoldSpec,
    EvaluationStatus,
)
from embodichain.lab.evaluation.observations import (
    advance_observation,
    evaluate_recording,
)
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers import (
    ObservationCfg,
    ObservationManager,
    SceneEntityCfg,
)
from embodichain.lab.gym.envs.managers.observations import get_entity_state
from embodichain.lab.gym.envs.objectives import evaluate_near_target
from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout, _write_raw_rollout


def _session():
    return EvaluationSession(
        NearTargetHoldSpec(
            "object", "goal", tolerance=0.5, hold_steps=2, window_steps=4
        ),
        env_ids=(3, 0, 2),
        episode_ids=(0, 0, 0),
    )


def test_live_and_chunked_replay_share_truth_and_original_hold_boundaries():
    session = _session()
    context = EpisodeContext(4)
    context._begin_episode([0, 1, 2, 3])
    context._finish_initialization([0, 1, 2, 3])
    target = torch.zeros(4, 7)
    target[:, 6] = 1
    position = target.clone()
    entities = {
        "object": SimpleNamespace(get_local_pose=lambda **kwargs: position),
        "goal": SimpleNamespace(get_local_pose=lambda **kwargs: target),
    }
    env = SimpleNamespace(
        episode_context=context,
        num_envs=4,
        device=torch.device("cpu"),
        sim=SimpleNamespace(asset_uids=entities, get_asset=entities.get),
    )
    observations = {
        uid: ObservationCfg(
            func=get_entity_state,
            mode="add",
            name=uid,
            params={"entity_cfg": SceneEntityCfg(uid=uid)},
            consumers=("recording",),
        )
        for uid in entities
    }
    manager = ObservationManager(observations, env)
    storage = TensorDict({}, batch_size=[4])
    _init_raw_rollout(storage, context, {}, torch.zeros(4, 3, 1))
    manager.bind_rollout_storage(storage)
    revisions = []
    positions_by_step = [
        [0.0, 0.0, 1.0, 0.5],
        [float("nan"), 0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    ]
    for step, xs in enumerate(positions_by_step):
        position[:, 0] = torch.tensor(xs)
        frame = manager.capture(
            lambda: manager.compute(TensorDict({}, batch_size=[4])),
            episode_context=context,
            timestamp=step * 0.1,
        )
        revisions.append(frame.revision)
        advance_observation(session, frame, step=step, episode_owner=context)
        snapshot = session.snapshot()
        legacy = evaluate_near_target(
            position[[3, 0, 2], :3], target[[3, 0, 2], :3], 0.5
        )
        assert torch.equal(
            snapshot.predicate_status == EvaluationStatus.SATISFIED, legacy.satisfied
        )
        assert torch.equal(
            snapshot.predicate_status != EvaluationStatus.UNAVAILABLE, legacy.valid
        )
        torch.testing.assert_close(
            snapshot.distance[legacy.valid], legacy.metrics["distance"][legacy.valid]
        )
        if step == 0:
            assert snapshot.window_status.tolist() == [EvaluationStatus.UNKNOWN] * 3
            foreign = EpisodeContext(4)
            foreign._begin_episode([0, 1, 2, 3])
            foreign_frame = replace(
                frame, episode_owner=foreign, revision=frame.revision + 1
            )
            with pytest.raises(ValueError, match="evaluation episodes"):
                advance_observation(
                    session, foreign_frame, step=1, episode_owner=foreign
                )
        if step == 1:
            assert snapshot.window_status[0].item() == EvaluationStatus.SATISFIED

        kwargs = (
            {"action": torch.zeros(4, 1)}
            if step
            else {
                "field_ids": tuple(manager.field_ids.values()),
                "env_ids": [0, 1, 2, 3],
            }
        )
        _write_raw_rollout(storage, context, frame, **kwargs)

    live = session.snapshot()
    assert live.predicate_status.tolist() == [EvaluationStatus.FAILED] * 3
    assert live.window_status.tolist() == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
        EvaluationStatus.FAILED,
    ]
    assert live.witness_steps.tolist() == [[0, 1], [-1, -1], [-1, -1]]
    assert live.witness_revisions.tolist() == [revisions[:2], [-1, -1], [-1, -1]]

    def replay_history(end, chunk_steps=64):
        return evaluate_recording(
            session.spec,
            manager,
            env_ids=session.env_ids,
            episode_ids=session.episode_ids,
            recorded_end_step=end,
            chunk_steps=chunk_steps,
        )

    # The recording path has no live acquisition provider.
    del env.sim
    for chunk_steps in (1, 2, 7):
        replay = replay_history(3, chunk_steps)
        assert replay.env_ids == live.env_ids
        assert replay.episode_ids == live.episode_ids
        assert replay.step == live.step
        for field in (
            "distance",
            "predicate_status",
            "window_status",
            "revisions",
            "witness_steps",
            "witness_revisions",
        ):
            torch.testing.assert_close(getattr(replay, field), getattr(live, field))

    prefix = replay_history(0)
    assert prefix.window_status.tolist() == [EvaluationStatus.UNAVAILABLE] * 3
    context._begin_episode([0])
    with pytest.raises(ValueError, match="evaluation episodes"):
        advance_observation(_session(), frame, step=0, episode_owner=context)
    with pytest.raises(ValueError, match="another or uninitialized episode"):
        replay_history(3)


def test_gaps_and_stale_revisions_break_holds_without_advancing_on_snapshot():
    session = _session()
    positions = torch.zeros(3, 3)
    valid = torch.ones(3, dtype=torch.bool)
    assert session.snapshot().window_status.tolist() == [EvaluationStatus.NOT_RUN] * 3
    session.advance(
        0, positions, positions, valid=valid, revisions=torch.tensor([2, 2, 2])
    )
    snapshot = session.snapshot()
    snapshot.window_status.fill_(EvaluationStatus.SATISFIED)
    assert session.snapshot().window_status.tolist() == [EvaluationStatus.UNKNOWN] * 3
    with pytest.raises(ValueError, match="Control slots"):
        session.advance(
            0, positions, positions, valid=valid, revisions=torch.tensor([3, 3, 3])
        )
    session.advance(
        2, positions, positions, valid=valid, revisions=torch.tensor([4, 4, 2])
    )
    assert session.snapshot().predicate_status.tolist() == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
    ]
    session.advance(
        3, positions, positions, valid=valid, revisions=torch.tensor([5, 4, 5])
    )
    terminal = session.snapshot()
    assert terminal.window_status.tolist() == [
        EvaluationStatus.SATISFIED,
        EvaluationStatus.UNAVAILABLE,
        EvaluationStatus.UNAVAILABLE,
    ]
    assert terminal.witness_steps.tolist() == [[2, 3], [-1, -1], [-1, -1]]


def test_evaluation_cold_import_does_not_load_tensor_or_gym_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import embodichain.lab.evaluation.observations; "
            "assert all(name not in sys.modules for name in "
            "('torch', 'gymnasium', 'embodichain.lab.sim', 'embodichain.lab.gym'))",
        ],
        check=True,
    )
