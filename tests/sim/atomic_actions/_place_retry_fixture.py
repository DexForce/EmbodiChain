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

"""Independent inner/outer Place retry counters with deterministic motion outputs."""

from __future__ import annotations
from dataclasses import replace
import random
import numpy as np
import torch
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.atomic_actions import PlaceGoal
from ._ordinary_pick_fixture import place_fixture


def retry_fixture(directory, mode):
    generator, context, goal, _, _ = place_fixture(directory)
    goal = PlaceGoal(context.scene.entities["target"].pose.clone())
    calls = []

    def generate(targets, options):
        count = options.sample_count
        calls.append(
            {
                "count": count,
                "preserve": options.preserve_cartesian_samples,
                "draws": [
                    float(torch.rand(())),
                    float(np.random.rand()),
                    random.random(),
                ],
            }
        )
        if options.preserve_cartesian_samples:
            positions = torch.cat(
                (
                    options.start_qpos[:, None],
                    torch.stack([t.xpos[:, :3, 3] for t in targets], 1),
                ),
                1,
            )
        else:
            points = torch.cat(
                (
                    options.start_qpos[:, None],
                    torch.stack([t.xpos[:, :3, 3] for t in targets], 1),
                ),
                1,
            )
            visits = (
                torch.linspace(0, count - 1, len(targets) + 1).round().to(torch.long)
            )
            positions = torch.empty(1, count, 3)
            for i in range(len(targets)):
                lo, hi = int(visits[i]), int(visits[i + 1])
                positions[:, lo : hi + 1] = torch.lerp(
                    points[:, i : i + 1],
                    points[:, i + 1 : i + 2],
                    torch.linspace(0, 1, hi - lo + 1)[None, :, None],
                )
        valid = mode == "outer" or (mode != "exhaust" and count >= 260)
        dt = torch.full(positions.shape[:2], 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([valid]), positions=positions, dt=dt)

    generator._generate_once = generate
    return generator, context, goal, calls
