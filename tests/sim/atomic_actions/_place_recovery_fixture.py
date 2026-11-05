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

"""Controlled motion and real input envelopes for ordinary Place recovery."""

from __future__ import annotations
from dataclasses import replace
import random
import numpy as np
import torch
import trimesh
from embodichain.lab.sim.atomic_actions import EntityState, SceneSnapshot
from embodichain.lab.sim.motion.planners.utils import PlanResult
from ._ordinary_pick_fixture import place_fixture


def recovery_fixture(directory, mode):
    generator, context, goal, _, ik = place_fixture(directory)
    floor = torch.eye(4)[None]
    floor[:, 2, 3] = -1
    context = replace(
        context,
        scene=SceneSnapshot(
            0.0, 0, {**context.scene.entities, "table": EntityState(floor)}
        ),
    )
    geometry = {
        "object": context.task.held_objects["arm"]
        .semantics.affordance.mesh_vertices[None]
        .clone(),
        "table": torch.tensor(
            trimesh.creation.box((4.0, 4.0, 0.2)).vertices, dtype=torch.float32
        )[None],
    }
    calls = []

    def generate(targets, options):
        count = options.sample_count
        sparse = not options.preserve_cartesian_samples
        calls.append(
            {
                "count": count,
                "sparse": sparse,
                "target_count": len(targets),
                "max_target_z": float(
                    torch.stack([t.xpos[:, 2, 3] for t in targets]).max()
                ),
                "draws": [
                    float(torch.rand(())),
                    float(np.random.rand()),
                    random.random(),
                ],
            }
        )
        if not sparse:
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
        valid = (
            mode == "nominal"
            or (mode in ("waypoint", "early") and sparse)
            or (mode == "via" and not sparse and calls[-1]["max_target_z"] > 0.575)
        )
        dt = torch.full(positions.shape[:2], 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([valid]), positions=positions, dt=dt)

    return generator, context, goal, geometry, calls, ik, generate


def early_geometry_oracle():
    count = 0

    def inspect(*args):
        nonlocal count
        count += 1
        angle = 0.9 if count <= 3 else 1.2
        return {
            "valid": True,
            "rotation_max_rad": angle,
            "rotation_path_rad": angle,
            "orientation_preserved": False,
            "scope": "controlled_geometry_oracle_not_physical",
            "physical_evidence": False,
        }

    return inspect
