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

"""Rank coordination, CPU batch delivery and ACT loss for the grouped study."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import timedelta
import time
from typing import Any, ContextManager, TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch import distributed as dist
from torch.nn import functional as F

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy

from embodichain.learning.sl.distributed import distributed_run

RGB = ("obs", "sensor", "cam_high", "color")
QPOS = ("obs", "robot", "qpos")
IMAGE = "observation.images.cam_high"


class RankFailure(RuntimeError):
    """A named stage failed on at least one participating rank."""


class Ranks:
    """Exchange small control messages and CPU tensors on a local Gloo group."""

    def __init__(self, device: torch.device, group: dist.ProcessGroup | None) -> None:
        self.device = device
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.wait_seconds = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Have all ranks observe a local failure before the next work phase."""
        error = None
        try:
            yield
        except RankFailure:
            raise
        except Exception as caught:
            if self.group is None:
                raise RankFailure(f"{name}: {caught}") from caught
            error = caught
        if self.group is None:
            return
        clock = time.perf_counter()
        failed = torch.tensor(int(error is not None), dtype=torch.int64)
        dist.all_reduce(failed, group=self.group)
        self.wait_seconds += time.perf_counter() - clock
        if failed:
            details: list[str | None] = [None] * self.size
            dist.all_gather_object(
                details,
                None if error is None else f"{type(error).__name__}: {error}",
                group=self.group,
            )
            failures = "; ".join(
                f"rank {rank}: {detail}"
                for rank, detail in enumerate(details)
                if detail is not None
            )
            raise RankFailure(f"{name}: {failures}") from error

    def sum_int(self, value: int) -> int:
        """Sum counts for equally sized rank batches."""
        if self.group is None:
            return value
        total = torch.tensor(value, dtype=torch.int64)
        clock = time.perf_counter()
        dist.all_reduce(total, group=self.group)
        self.wait_seconds += time.perf_counter() - clock
        return int(total)

    def decision(self, value: bool) -> bool:
        """Return the rank-zero decision at the next common loop boundary."""
        if self.group is None:
            return value
        answer = torch.tensor(value, dtype=torch.bool)
        dist.broadcast(answer, 0, group=self.group)
        return bool(answer)

    def gather(self, value: Any) -> list[Any] | None:
        """Give rank zero the ordered rank values, for setup and end-of-run reports."""
        if self.group is None:
            return [value]
        received = [None] * self.size if self.rank == 0 else None
        dist.gather_object(value, received, dst=0, group=self.group)
        return received

    def broadcast(self, value: Any) -> Any:
        """Broadcast rank-zero metadata or calibration to all ranks."""
        if self.group is None:
            return value
        message = [value]
        dist.broadcast_object_list(message, src=0, group=self.group)
        return message[0]


@contextmanager
def grouped_ranks(device: str = "cuda:0") -> Iterator[Ranks]:
    """Use NCCL for CUDA model gradients and Gloo for same-host ODS delivery."""
    with distributed_run(device, timeout_seconds=60) as local_device:
        if not dist.is_initialized():
            yield Ranks(local_device, None)
            return
        if dist.get_world_size() != 2:
            raise ValueError("Grouped ACT supports one process or two local DDP ranks")
        if dist.get_backend() == "gloo":
            yield Ranks(local_device, dist.group.WORLD)
            return
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=360))
        try:
            yield Ranks(local_device, group)
        finally:
            dist.destroy_process_group(group)


class BatchTransfer:
    """Send the second half of an owned ODS batch into persistent CPU storage."""

    def __init__(
        self,
        ranks: Ranks,
        local_size: int,
        anchor_fields: Sequence[str | tuple[str, ...]],
    ) -> None:
        self.ranks = ranks
        self.local_size = local_size
        self.fields = (*anchor_fields, "actions", "action_is_pad", "ods_episode_id")
        self.receive: TensorDict | None = None
        self.initialized = False

    def __call__(self, batch: TensorDict | None) -> TensorDict:
        ranks = self.ranks
        if not self.initialized:
            shapes = None
            with ranks.stage("batch transfer schema"):
                if ranks.rank == 0:
                    assert batch is not None
                    shapes = [
                        ((self.local_size, *batch[key].shape[1:]), batch[key].dtype)
                        for key in self.fields
                    ]
            shapes = ranks.broadcast(shapes)
            with ranks.stage("batch receive allocation"):
                if ranks.rank == 1:
                    self.receive = TensorDict({}, batch_size=[self.local_size])
                    for key, (shape, dtype) in zip(self.fields, shapes, strict=True):
                        self.receive[key] = torch.empty(shape, dtype=dtype)
            self.initialized = True

        if ranks.rank == 0:
            assert batch is not None
            payload = [
                batch[key][self.local_size :].contiguous() for key in self.fields
            ]
            operation = dist.isend
        else:
            assert self.receive is not None
            payload = [self.receive[key] for key in self.fields]
            operation = dist.irecv
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(operation, tensor, 1 - ranks.rank, group=ranks.group)
                for tensor in payload
            ]
        )
        for request in requests:
            request.wait(timedelta(seconds=60))
        return batch[: self.local_size] if ranks.rank == 0 else self.receive


@contextmanager
def training_batch(
    ranks: Ranks,
    draw: Callable[[], ContextManager[TensorDict]] | None,
    transfer: BatchTransfer | None,
    timings: dict[str, float],
) -> Iterator[TensorDict]:
    """Commit ODS draws on a successful update across all participating ranks."""
    with ranks.stage("ODS draw commit"):
        with ExitStack() as stack:
            batch = None
            with ranks.stage("ODS sampling"):
                clock = time.perf_counter()
                if draw is not None:
                    batch = stack.enter_context(draw())
                timings["sample_s"] = time.perf_counter() - clock
            timings["transfer_s"] = 0.0
            if transfer is not None:
                clock = time.perf_counter()
                batch = transfer(batch)
                timings["transfer_s"] = time.perf_counter() - clock
            assert batch is not None
            yield batch


class MaskedACTLoss(torch.nn.Module):
    """Use the native ACT network with globally normalized masked action loss."""

    def __init__(self, policy: ACTPolicy, world_size: int) -> None:
        super().__init__()
        self.policy = policy
        self.world_size = world_size

    def forward(
        self, batch: dict[str, torch.Tensor], global_valid: int
    ) -> torch.Tensor:
        """Scale only the L1 numerator before DDP averages rank gradients."""
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        cfg = self.policy.config
        if cfg.image_features:
            batch = {**batch, OBS_IMAGES: [batch[key] for key in cfg.image_features]}
        predicted, (mean, log_variance) = self.policy.model(batch)
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        l1_sum = (F.l1_loss(batch[ACTION], predicted, reduction="none") * valid).sum()
        loss = l1_sum * (self.world_size / max(1, global_valid))
        if cfg.use_vae and log_variance is not None:
            kl = -0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp())
            loss = loss + cfg.kl_weight * kl.sum(-1).mean()
        return loss
