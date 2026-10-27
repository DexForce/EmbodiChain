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

"""Real NCCL/SyncBatchNorm weighted updates and model-recipe recovery."""

from __future__ import annotations

from datetime import timedelta
import ctypes
import faulthandler
import json
from pathlib import Path
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from embodichain.learning.sl import SupervisedLearner
from embodichain_models.dynamics.rigidformer import RigidFormerModel
from embodichain_models.dynamics.rigidformer.sl_recipe import build_rigidformer_trainer
from embodichain_models.dynamics.rigidformer.supervised import (
    RigidFormerObjective,
    rigidformer_batch,
)
from ._rigidformer_fixtures import tiny_cfg, geometry_batch
from ._rigidformer_training import assert_exact, make_packed, recovery_config

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _disable_dropout(model: torch.nn.Module) -> None:
    # Encoder regularization is hard-coded in author code. Disable stochastic
    # masks for the concatenated-batch oracle; the recovery case keeps them.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def _weighted_batch(device: str) -> dict:
    values = geometry_batch(device)
    for key in ("obj_positions", "valid_vtx_mask"):
        values[key] = torch.cat((values[key], values[key][:, :, :1].clone()), dim=2)
    for key in ("ref_obj_positions", "anchor_idx", "phys_params", "vtx_rank"):
        values[key] = torch.cat((values[key], values[key][:, :1].clone()), dim=1)
    values["obj_positions"][:, :, 2, :, 0] += 1
    values["ref_obj_positions"][:, 2, :, 0] += 1
    values["valid_obj_mask"] = torch.tensor(
        [[True, True, True], [True, True, False]], device=device
    )
    return values


def _oracle(path: Path) -> None:
    # Compute outside any process group. The author's grouped encoder otherwise
    # all-reduces its moments even with ordinary BatchNorm modules.
    torch.manual_seed(73)
    reference = RigidFormerModel(tiny_cfg()).cuda().train()
    _disable_dropout(reference)
    initial = {n: v.detach().cpu().clone() for n, v in reference.state_dict().items()}
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    expected = RigidFormerObjective().compute_loss(
        reference, rigidformer_batch(_weighted_batch("cuda:0"))
    )
    expected.loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    optimizer.step()
    torch.save(
        {
            "initial": initial,
            "loss": float(expected.loss),
            "model": {n: v.detach().cpu() for n, v in reference.state_dict().items()},
            "gradients": {
                n: None if v.grad is None else v.grad.detach().cpu()
                for n, v in reference.named_parameters()
            },
        },
        path / "oracle.pt",
    )


def _weighted_update(rank: int, path: Path) -> dict:
    torch.manual_seed(73)
    model = RigidFormerModel(tiny_cfg()).cuda(rank)
    _disable_dropout(model)
    model.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.network)
    expected = torch.load(
        path / "oracle.pt", weights_only=True, map_location=f"cuda:{rank}"
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected["initial"][name], rtol=0, atol=0)
    values = _weighted_batch(f"cuda:{rank}")
    learner = SupervisedLearner(
        model,
        RigidFormerObjective(),
        torch.optim.SGD(model.parameters(), lr=0.001),
        max_grad_norm=1.0,
        ddp_find_unused_parameters=True,
        ddp_broadcast_buffers=True,
    )
    local = {key: value[rank : rank + 1] for key, value in values.items()}
    metrics = learner.update(rigidformer_batch(local))
    assert metrics["loss"] == pytest.approx(expected["loss"], rel=1e-4, abs=1e-5), (
        metrics["loss"],
        expected["loss"],
    )
    error = 0.0
    for name, value in model.state_dict().items():
        target = expected["model"][name]
        torch.testing.assert_close(value, target, rtol=1e-4, atol=1e-5)
        if value.numel():
            error = max(error, float((value - target).abs().max()))
    for name, actual in model.named_parameters():
        target = expected["gradients"][name]
        if actual.grad is None:
            assert target is None
        else:
            torch.testing.assert_close(actual.grad, target, rtol=1e-4, atol=1e-5)
    return {
        "global_loss": metrics["loss"],
        "state_max_abs": error,
        "valid_objects_by_rank": [3, 2],
    }


def _worker(rank: int, path: str) -> None:
    faulthandler.dump_traceback_later(30, repeat=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    root = Path(path)
    dist.init_process_group(
        "nccl",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        weighted = _weighted_update(rank, root)
        config = recovery_config(root / "data", root / "full", f"cuda:{rank}")
        full = build_rigidformer_trainer(config)
        assert full.signature["steps_per_epoch"] == 2
        full.fit()
        config["trainer"]["output_dir"] = str(root / "split")
        split = build_rigidformer_trainer(config)
        split.fit(stop_after=3)
        config["trainer"]["output_dir"] = str(root / "resumed")
        resumed = build_rigidformer_trainer(config)
        resumed.resume(root / "split/last.pt")
        resumed.fit()
        a = torch.load(root / "full/last.pt", weights_only=False)
        b = torch.load(root / "resumed/last.pt", weights_only=False)
        for key in ("learner", "training_states", "sampler", "rng", "distributed"):
            assert_exact(a[key], b[key])
        assert resumed.training_states["epoch_schedule"].scheduler.last_epoch == 3
        if rank == 0:
            runtime = ctypes.c_int()
            ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(runtime))
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "backend": "nccl",
                        "world_size": 2,
                        "sync_batch_norm": True,
                        "loaded_nccl_version": runtime.value,
                        "weighted_update": weighted,
                        "recovery": "exact",
                        "updates": 6,
                        "split_after": 3,
                        "epochs": 3,
                        "warmup_epochs": 2,
                        "compared": [
                            "model",
                            "optimizer",
                            "EMA",
                            "schedule",
                            "rank RNG",
                            "rank source cursor",
                            "rank buffers",
                        ],
                    },
                    indent=2,
                )
            )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="Requires two CUDA devices and NCCL",
)
def test_nccl_weighted_syncbn_and_exact_rank_recipe_recovery(tmp_path):
    make_packed(tmp_path / "data", frames_count=13)
    _oracle(tmp_path)
    context = mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Two-rank RigidFormer qualification exceeded 120 seconds"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join()
    assert json.loads((tmp_path / "summary.json").read_text())["recovery"] == "exact"
