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

"""Candidate-owned RNG streams around the canonical command handshake."""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
from .program import _Cursor

__all__: list[str] = []


class _WorkloadCursor(_Cursor):
    def __init__(self, bridge: TaskProgramDemoBridge, robot: Any, seed: int) -> None:
        super().__init__(bridge)
        self.robot = robot
        self.rejection = None
        self.last_command = None
        self.devices = (
            [
                (
                    robot.device.index
                    if robot.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if robot.device.type == "cuda"
            else []
        )
        self.cpu_rng = torch.Generator().manual_seed(seed).get_state()
        self.gpu_rng = (
            torch.Generator(device=robot.device).manual_seed(seed).get_state()
            if self.devices
            else None
        )
        self.numpy_rng = np.random.RandomState(seed % (2**32)).get_state()
        self.python_rng = random.Random(seed).getstate()

    def next(self) -> ControllerAction | None:
        before_numpy, before_python = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu_rng)
            if self.devices:
                torch.cuda.set_rng_state(self.gpu_rng, self.devices[0])
            np.random.set_state(self.numpy_rng)
            random.setstate(self.python_rng)
            try:
                action = super().next()
                if action is not None:
                    self.last_command = action
                if self.finished and self.rejection is not None:
                    self.accepted, self.reason = False, self.rejection
                return action
            finally:
                self.cpu_rng = torch.get_rng_state()
                if self.devices:
                    self.gpu_rng = torch.cuda.get_rng_state(self.devices[0])
                self.numpy_rng, self.python_rng = (
                    np.random.get_state(),
                    random.getstate(),
                )
                np.random.set_state(before_numpy)
                random.setstate(before_python)

    def abort(self, reason: str) -> None:
        if self.finished or self.rejection is not None or self.emergency is not None:
            return
        if not self.pending_consumed:
            # An unconsumed command must go through the original safe-stop
            # iterator; acknowledging it here would manufacture a World step.
            super().abort(reason)
            return
        ids = tuple(self.bridge.env_ids.tolist())
        hold = self.robot.get_qpos(target=True)[list(ids)].clone()
        previous = self.last_command.value
        if isinstance(previous, torch.Tensor):
            value = hold
        else:
            fields = {"qpos": hold}
            if "qvel" in previous:
                fields["qvel"] = torch.zeros_like(hold)
            value = TensorDict(fields, batch_size=[len(ids)])
        self.bridge.request_row_rejection(ids, ControllerAction(value), reason=reason)
        self.rejection = reason
