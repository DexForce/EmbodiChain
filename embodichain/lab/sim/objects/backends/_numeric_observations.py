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
"""Bounded shared native reads during a single World-owner planning boundary.

Only Default CPU full-batch numeric reads are shared. Each consumer still
receives its own output buffer. Checked native writes invalidate all samples;
physics/topology revisions and scope exit prevent reuse across boundaries.
Raw SDK mutations and concurrent World writers are outside this internal
read-boundary contract.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
from typing import Any, Iterator, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects.robot import Robot

__all__: list[str] = []
_READS = frozenset(
    (
        "fetch_pose",
        "fetch_root_pose",
        "fetch_link_pose",
        "fetch_joint_position",
        "fetch_joint_velocity",
        "fetch_joint_force",
        "fetch_linear_velocity",
        "fetch_angular_velocity",
        "fetch_root_linear_velocity",
        "fetch_root_angular_velocity",
    )
)
_ACTIVE: ContextVar[_NumericReads | None] = ContextVar(
    "native_numeric_reads", default=None
)


class _NumericReads:
    def __init__(
        self, batches: tuple[object, ...], revision: Callable[[], object]
    ) -> None:
        self.batches = {id(batch): batch for batch in batches}
        self.revision: Callable[[], object] | None = revision
        self.stamp = revision()
        self.owner = get_ident()
        self.closed = False
        self.values: dict[tuple, tuple[torch.Tensor, Any]] = {}

    def refresh(self) -> bool:
        """Check the owner/revision before accessing protected samples."""
        if self.closed or self.owner != get_ident() or self.revision is None:
            return False
        stamp = self.revision()
        if stamp != self.stamp:
            self.values.clear()
            self.stamp = stamp
        return True

    def key(
        self, batch: object, method: str, args: tuple, kwargs: dict
    ) -> tuple | None:
        if not self.refresh():
            return None
        if not method.startswith("fetch_"):
            # Invalidate before a write, including partially failing writes.
            self.values.clear()
            return None
        if (
            method not in _READS
            or id(batch) not in self.batches
            or len(args) != 1
            or kwargs
        ):
            return None
        out = args[0]
        if (
            type(out) is not torch.Tensor
            or out.device.type != "cpu"
            or out.requires_grad
            or not out.is_contiguous()
        ):
            return None
        return (id(batch), method, out.shape, out.dtype)


def shared_pose(
    batch: object, to_matrix: bool, read: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """Own one derived pose representation and clone it for every consumer."""
    reads = _ACTIVE.get()
    if reads is None or not reads.refresh() or id(batch) not in reads.batches:
        return read()
    key = (
        id(batch),
        "derived_pose",
        to_matrix,
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cpu"),
    )
    cached = reads.values.get(key)
    if cached is not None:
        return cached[0].clone()
    value = read()
    if (
        type(value) is torch.Tensor
        and value.device.type == "cpu"
        and not value.requires_grad
    ):
        reads.values[key] = (value.detach().clone(), None)
    return value


@contextmanager
def numeric_observation_scope(sim: SimulationManager, robot: Robot) -> Iterator[None]:
    """Share numeric samples only while this owner is gathering row commands."""
    if sim.physics_backend != "default" or sim.device.type != "cpu":
        yield
        return
    batches = [robot._data.articulation_view.batch]
    for uid in sim.get_rigid_object_uid_list():
        data = sim.get_rigid_object(uid)._data
        if data is not None:
            batches.append(data.body_view.batch)
    reads = _NumericReads(
        tuple(batches),
        lambda: (
            sim.physics_step_index,
            id(robot._data.articulation_view.scene),
            robot._data.articulation_view.scene.topology_revision,
        ),
    )
    previous = _ACTIVE.get()
    if previous is not None:
        previous.values.clear()
    token = _ACTIVE.set(reads)
    try:
        yield
    finally:
        reads.values.clear()
        reads.closed = True
        reads.batches.clear()
        reads.revision = None
        _ACTIVE.reset(token)
        if previous is not None:
            previous.values.clear()
