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

"""Deterministic segment motion callbacks for source control-flow comparisons."""

from __future__ import annotations
import hashlib, random
from types import SimpleNamespace as NS
import numpy as np
import torch
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanResult, PlanState


def segment_fixture(case):
    calls = []
    robot = NS(device=torch.device("cpu"))

    def fk(qpos, **kwargs):
        result = torch.eye(4).expand(qpos.shape[0], 4, 4).clone()
        result[:, 0, 3] = qpos[:, 0]
        return result

    robot.compute_fk = fk

    def generate(targets, *, options):
        draw = (float(torch.rand(())), float(np.random.rand()), random.random())
        calls.append(
            {
                "count": options.sample_count,
                "targets_sha256": hashlib.sha256(
                    torch.stack([t.xpos for t in targets]).numpy().tobytes()
                ).hexdigest(),
                "draws": list(draw),
                "preserve": options.preserve_cartesian_samples,
            }
        )
        valid = case["outcome"] == "nominal" or (
            case["outcome"] == "dense" and options.sample_count >= 260
        )
        positions = torch.linspace(0, 0.2, options.sample_count)[None, :, None]
        dt = torch.full(positions.shape[:2], 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([valid]), positions=positions, dt=dt)

    continuous = []

    def recover(generator, targets, options, **kwargs):
        continuous.append(
            {
                "count": options.sample_count,
                "targets_sha256": hashlib.sha256(
                    torch.stack([t.xpos for t in targets]).numpy().tobytes()
                ).hexdigest(),
                "torch_rng_sha256": hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
            }
        )
        if case["outcome"] != "continuous":
            return None
        positions = torch.linspace(0, 0.2, options.sample_count)[None, :, None]
        dt = torch.full(positions.shape[:2], 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([True]), positions=positions, dt=dt)

    target = torch.eye(4)[None]
    target[:, 0, 3] = 0.2
    return (
        NS(robot=robot, generate=generate, _generate_checked=generate),
        [PlanState(MoveType.EEF_MOVE, xpos=target)],
        calls,
        continuous,
        recover,
    )
