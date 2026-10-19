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

"""Pair actual ODS sampling with official LeRobot persistence and loading.

The producer is replaced by a deterministic CPU buffer. Sampling, provenance,
quota accounting, SDK serialization/readback and learner calls are real.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.engine.data import (
    OnlineDataBudgetExhaustedError,
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
    _populate_provenance,
)
from examples.data_pipeline._ods_pick_place_benchmark import (
    EfficiencyTracker,
    FrozenCorpus,
)
from examples.data_pipeline._ods_pick_place_lerobot import (
    DirectLeRobotWriter,
    load_lerobot_corpus,
)
from examples.data_pipeline._ods_pick_place_policy import ChunkBC, pose_targets
from examples.data_pipeline._ods_pick_place_training import _sample, _schema, _snapshot
from examples.data_pipeline.ods_pick_place_showcase import _geometry_statistics

_ROWS, _STEPS, _HORIZON = 3, 7, 4


def _buffer(engine: OnlineDataEngine) -> TensorDict:
    generator = torch.Generator().manual_seed(1207)
    shape = (_ROWS, _STEPS)
    cube = torch.randn(*shape, 7, generator=generator)
    cube[..., 3:] = torch.nn.functional.normalize(cube[..., 3:], dim=-1)
    tcp = torch.eye(4).expand(*shape, 4, 4).clone()
    tcp[..., :3, 3] = torch.randn(*shape, 3, generator=generator)
    valid = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, False, True, True, False, True, False],
            [False, False, True, False, False, False, False],
        ]
    )
    buffer = TensorDict(
        {
            "obs": {
                "robot": {
                    "qpos": torch.randn(*shape, 9, generator=generator),
                    "qvel": torch.randn(*shape, 9, generator=generator),
                },
                "tcp_pose": tcp,
                "cube_pose": cube,
                "goal_position": torch.randn(*shape, 3, generator=generator),
                "cube_velocity": torch.randn(*shape, 6, generator=generator),
            },
            "actions": torch.rand(*shape, 7, generator=generator) * 2 - 1,
            "valid": valid,
            # Changes across gaps are representable by the current offline format.
            "continuity_id": torch.tensor(
                [[0] * _STEPS, [0, -1, 1, 1, -1, 2, -1], [0] * _STEPS]
            ),
            **{
                name: torch.full(shape, -1, dtype=torch.int64)
                for name in (
                    "ods_run_id",
                    "ods_episode_id",
                    "ods_frame_index",
                    "ods_generation",
                    "ods_accepted_at_ns",
                )
            },
        },
        batch_size=shape,
        device="cpu",
    )
    # Dirty tails must never become real observations or supervised targets.
    for _, value in buffer["obs"].items(include_nested=True, leaves_only=True):
        value[~valid] = float("nan")
    buffer["actions"][~valid] = float("nan")
    _populate_provenance(
        buffer, run_id=engine.run_id, generation=1, episode_start=0, accepted_at_ns=1
    )
    return buffer.share_memory_()


@pytest.fixture
def source(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[OnlineDataEngine]:
    budget = getattr(request, "param", None)
    monkeypatch.setattr(OnlineDataEngine, "_create_buffer", _buffer)
    engine = OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config={"num_envs": 1},
            state_dim=9,
            action_dim=7,
            action_semantics="policy_request",
            action_contracts=("eef_pose.relative@1", "gripper.parallel@1"),
            buffer_size=_ROWS,
            max_episode_steps=_STEPS,
            refill_policy="manual",
            include_provenance=True,
            episode_sample_budget=budget,
            episode_sample_hard_limit=budget,
        )
    )
    # Exercise the consumer without requiring a simulation worker's liveness.
    engine._owner_pid = -1
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
        if engine._sampling_budget is not None:
            engine._sampling_budget.publish(0, _ROWS, 0, 1)
    try:
        yield engine
    finally:
        # No producer was started; return the synthetic fixture to pre-start
        # state so stop() does not diagnose a missing real worker as a failure.
        engine._set_state(OnlineDataEngineState.CREATED)
        engine._owner_pid = os.getpid()
        engine.stop()


def _roundtrip(engine: OnlineDataEngine, destination: Path) -> FrozenCorpus:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    schema = _schema({"action_terms": []})
    snapshot = _snapshot(engine)
    with DirectLeRobotWriter(
        destination, schema=schema, episode_count=_ROWS, max_episode_steps=_STEPS
    ) as writer:
        for row in range(_ROWS):
            writer.submit(
                {
                    key: value[row]
                    for key, value in snapshot.items()
                    if key != "initial_positions"
                }
            )
        writer.finalize()
    corpus = load_lerobot_corpus(destination, expected_schema=schema, read_batch_size=2)
    if engine.cfg.episode_sample_budget is not None:
        corpus.configure_sampling_budget(
            engine.cfg.episode_sample_budget, engine.cfg.episode_sample_hard_limit
        )
    return corpus


def _paired_batch(
    engine: OnlineDataEngine,
    corpus: FrozenCorpus,
    batch_size: int,
    horizon: int = _HORIZON,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    before = torch.get_rng_state()
    online = _sample(engine, batch_size, EfficiencyTracker(20), horizon)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    offline = corpus.sample_context(batch_size, 2, horizon)
    assert torch.equal(torch.get_rng_state(), after)
    assert online.keys() == offline.keys()
    for field in online:
        assert torch.equal(online[field], offline[field]), field
    return online, offline


@pytest.mark.parametrize("source", [None, 6], indirect=True)
@pytest.mark.parametrize("horizon", [1, 4, 64])
def test_sdk_roundtrip_and_context_batches_are_exactly_equal(
    source: OnlineDataEngine, tmp_path: Path, horizon: int
) -> None:
    corpus = _roundtrip(source, tmp_path / "dataset")
    seen: set[tuple[int, int]] = set()
    batch_size = 128 if source.cfg.episode_sample_budget is None else 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(818)
        for _ in range(4):
            online, _ = _paired_batch(source, corpus, batch_size, horizon)
            mask = online["valid"]
            assert torch.isfinite(online["observations"]).all()
            assert (online["observations"][~mask] == 0).all()
            assert (online["actions"][~mask] == 0).all()
            assert (online["frame_ids"][~mask] == -1).all()
            assert not online["loss_mask"][:, 0].any()
            anchors = online["frame_ids"][:, 1, 1:]
            seen.update(map(tuple, anchors.tolist()))
            # Recorded requests belong to the current pre-action observation.
            assert torch.equal(
                online["actions"][:, 1],
                source.shared_buffer["actions"][anchors[:, 0], anchors[:, 1]],
            )
    if source.cfg.episode_sample_budget is None:
        assert seen == set(map(tuple, source.shared_buffer["valid"].nonzero().tolist()))
    else:

        def counts(snapshot: dict[str, Any]) -> list[int]:
            return [row["delivered"] for row in snapshot["rows"]]

        assert counts(source.get_sampling_budget_snapshot()) == counts(
            corpus.get_sampling_budget_snapshot()
        )
        for sample in (
            lambda count: source.sample_batch_with_context(count, 2, _HORIZON),
            lambda count: corpus.sample_context(count, 2, _HORIZON),
        ):
            with pytest.raises(OnlineDataBudgetExhaustedError):
                sample(3)  # Only two of the original eighteen credits remain.
        _paired_batch(source, corpus, 2)
        assert source.get_sampling_budget_snapshot()["unlocked_remaining_chunks"] == 0
        assert corpus.get_sampling_budget_snapshot()["remaining_chunks"] == 0


def test_identical_sdk_batches_produce_identical_optimizer_updates(
    source: OnlineDataEngine, tmp_path: Path
) -> None:
    corpus = _roundtrip(source, tmp_path / "dataset")
    observations = corpus.observations[corpus.valid]
    mean, std = _geometry_statistics(observations)
    targets = pose_targets(observations, corpus.actions[corpus.valid])
    target_mean = targets.mean(0)
    target_std = targets.std(0, correction=0).clamp_min(0.05)
    target_mean[3:], target_std[3:] = 0, 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(504)
        first = ChunkBC(mean, std, target_mean, target_std, _HORIZON, 2)
        second = copy.deepcopy(first)
        initial = copy.deepcopy(first.state_dict())
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
            for model in (first, second)
        ]
        for _ in range(3):
            online, offline = _paired_batch(source, corpus, 8)
            noise_state = torch.get_rng_state()
            for model, optimizer, batch in zip(
                (first, second), optimizers, (online, offline)
            ):
                torch.set_rng_state(noise_state)
                arm, hand = model.loss(
                    batch["observations"],
                    batch["actions"],
                    loss_mask=batch["loss_mask"],
                    valid_mask=batch["valid"],
                )
                optimizer.zero_grad(set_to_none=True)
                ((9 * arm + hand) / 10).backward()
                if model is first:
                    expected_loss = torch.stack((arm, hand)).detach()
                    expected_gradients = [p.grad.clone() for p in model.parameters()]
                else:
                    assert torch.equal(expected_loss, torch.stack((arm, hand)).detach())
                    for expected, parameter in zip(
                        expected_gradients, second.parameters()
                    ):
                        assert torch.equal(expected, parameter.grad)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
            for key in first.state_dict():
                assert torch.equal(first.state_dict()[key], second.state_dict()[key])
            for left, right in zip(
                optimizers[0].state.values(), optimizers[1].state.values()
            ):
                assert left.keys() == right.keys()
                assert all(torch.equal(left[key], right[key]) for key in left)
        assert any(
            not torch.equal(initial[key], first.state_dict()[key]) for key in initial
        )


@pytest.mark.parametrize("kind", ["jump", "negative"])
def test_offline_snapshot_rejects_unrepresentable_continuity(
    source: OnlineDataEngine, kind: str
) -> None:
    if kind == "jump":
        source.shared_buffer["continuity_id"][0, 1:] = 1
    else:
        source.shared_buffer["continuity_id"][0].fill_(-1)
    with pytest.raises(ValueError, match="continuous"):
        _snapshot(source)
