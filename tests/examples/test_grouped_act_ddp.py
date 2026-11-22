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

"""Grouped ACT routing, weighted DDP updates and supply evidence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel

from examples.learning.act._grouped_ddp import (
    BatchTransfer,
    MaskedACTLoss,
    RGB,
    RankFailure,
    Ranks,
    training_batch,
)
from examples.learning.act.audit_grouped import _audit_supply
from examples.learning.act.grouped import _calibration_stats, _scene_shard


class _TinyACT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.config = SimpleNamespace(image_features=(), use_vae=True, kl_weight=0.2)

    def model(self, batch: dict):
        feature = batch["feature"] * self.weight
        return feature[:, None, :].expand(-1, 2, -1), (
            feature,
            torch.zeros_like(feature),
        )


def _peer(rank: int, store: Path) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store.as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        ranks = Ranks(torch.device("cpu"), dist.group.WORLD)
        for layout in ("per-rank", "shared"):
            commits = []
            sample_size = 4 if layout == "shared" else 2
            start = 0 if layout == "shared" else 2 * rank
            transfer = BatchTransfer(ranks, 2, [RGB]) if layout == "shared" else None
            storage = None
            for step in range(3):

                @contextmanager
                def draw():
                    ids = torch.arange(start, start + sample_size)
                    sample = TensorDict(
                        {
                            RGB: (ids + 10 * step).to(torch.uint8).reshape(-1, 1),
                            "actions": torch.zeros(sample_size, 2, 1),
                            "action_is_pad": torch.zeros(
                                sample_size, 2, dtype=torch.bool
                            ),
                            "ods_episode_id": ids,
                        },
                        batch_size=[sample_size],
                    )
                    yield sample
                    commits.extend(sample["ods_episode_id"].tolist())

                source = draw if layout == "per-rank" or rank == 0 else None
                aborted = False
                try:
                    with training_batch(ranks, source, transfer, {}) as batch:
                        with ranks.stage("optimizer"):
                            assert batch["ods_episode_id"].tolist() == [
                                rank * 2,
                                rank * 2 + 1,
                            ]
                            assert batch[RGB].flatten().tolist() == [
                                rank * 2 + 10 * step,
                                rank * 2 + 1 + 10 * step,
                            ]
                            if rank == 1 and transfer is not None:
                                pointer = batch[RGB].data_ptr()
                                if storage is None:
                                    storage = pointer
                                assert storage == pointer
                            if step == 2 and rank == 1:
                                raise ValueError("injected update error")
                except RankFailure as error:
                    assert step == 2 and "injected update error" in str(error)
                    aborted = True
                assert aborted == (step == 2)
            assert commits == (
                list(range(start, start + sample_size)) * 2 if source else []
            )

        native = _TinyACT()
        model = DistributedDataParallel(MaskedACTLoss(native, 2))
        feature = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float64)
        valid = torch.tensor(
            [[True, True], [True, False], [True, False], [True, False]]
        )
        part = slice(2 * rank, 2 * (rank + 1))
        local = {
            "feature": feature[part],
            "action": torch.zeros(2, 2, 1, dtype=torch.float64),
            "action_is_pad": ~valid[part],
        }
        model(local, ranks.sum_int(int(valid[part].sum()))).backward()

        weight = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        prediction = weight * feature
        reference = (
            prediction.abs() * valid
        ).sum() / valid.sum() + 0.1 * prediction.square().mean()
        reference.backward()
        torch.testing.assert_close(native.weight.grad, weight.grad, atol=1e-12, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="The rank test requires Gloo")
def test_rank_delivery_commit_and_weighted_loss(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    ranks = [
        ctx.Process(target=_peer, args=(rank, tmp_path / "rendezvous"))
        for rank in range(2)
    ]
    try:
        for process in ranks:
            process.start()
        for process in ranks:
            process.join(45)
        assert [process.exitcode for process in ranks] == [0, 0]
    finally:
        for process in ranks:
            if process.is_alive():
                process.terminate()
                process.join(5)


@pytest.mark.parametrize("layout", ["per-rank", "shared"])
def test_audit_requires_release_for_each_authorized_generation(layout: str) -> None:
    count = 2 if layout == "per-rank" else 1
    width = 10 // count
    devices = [dict(rank=i, pid=100 + i, gpu_uuid=f"GPU-{i}") for i in range(2)]
    recipe = dict(world_size=2, producer_layout=layout, ranks=devices)
    steps = 400
    spec = dict(
        producer_envs=10, shared_slots=20, episodes=100, batch_size=32, draw_floor=192
    )
    producers = [
        dict(
            rank=i,
            pid=200 + i,
            gpu_uuid=devices[i]["gpu_uuid"],
            width=width,
            slots=2 * width,
            cap=10 * width,
            engine=dict(
                run_id=1000 + i,
                successful_episodes=3 * width,
                completed_generation=3,
                requested_generation=4,
                sampled_chunks=steps * 32 // count,
            ),
            releases=[
                dict(requested_generation=2, episode_ids=[], committed_draws=[]),
                dict(
                    requested_generation=3,
                    episode_ids=list(range(width)),
                    committed_draws=[192] * width,
                ),
                dict(
                    requested_generation=4,
                    episode_ids=list(range(width, 2 * width)),
                    committed_draws=[220] * width,
                ),
            ],
        )
        for i in range(count)
    ]
    supply = dict(
        step=steps,
        ranks=[dict(rank=i, committed_draws=steps * 16) for i in range(2)],
        producers=producers,
    )
    assert (
        _audit_supply(recipe, supply, spec) == ([192] * width + [220] * width) * count
    )
    missing = producers[-1]["releases"].pop()
    with pytest.raises(ValueError, match="incomplete release chain"):
        _audit_supply(recipe, supply, spec)
    producers[-1]["releases"].append(missing)
    producers[-1]["releases"][1]["committed_draws"][-1] = 191
    with pytest.raises(ValueError, match="committed-draw floor"):
        _audit_supply(recipe, supply, spec)


def test_calibration_merges_rank_frames_and_episode_lengths() -> None:
    pytest.importorskip("lerobot")
    first = {
        "state": torch.tensor([[0.0], [2.0]]),
        "action": torch.tensor([[1.0], [3.0]]),
        "landmarks": torch.tensor([[0.1], [0.3]]),
        "frames": torch.arange(2),
        "lengths": torch.tensor([2]),
    }
    second = {
        "state": torch.tensor([[6.0]]),
        "action": torch.tensor([[8.0]]),
        "landmarks": torch.tensor([[0.5]]),
        "frames": torch.arange(1),
        "lengths": torch.tensor([1]),
    }
    stats, scale = _calibration_stats([first, second], elapsed=True)
    assert scale == 1
    torch.testing.assert_close(
        stats["observation.state"]["mean"], torch.tensor([8 / 3, 0.3, 1 / 3])
    )
    torch.testing.assert_close(stats["action"]["mean"], torch.tensor([4.0]))


def test_scene_shards_reconstruct_the_original_attempt() -> None:
    seen = set()
    for seed in range(100, 500, 10):
        first = _scene_shard(seed, 10, 0, 5, True)
        second = _scene_shard(seed, 10, 5, 5, True)
        assert first + second == _scene_shard(seed, 10, 0, 10, True)
        assert {stratum for _, stratum in first + second} == set(range(10))
        ids = {scene for scene, _ in first + second}
        assert not ids & seen
        seen.update(ids)
