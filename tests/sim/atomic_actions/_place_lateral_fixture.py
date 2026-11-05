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

"""Real phase composition with controlled IK/collision/native-prefix adapters."""

from __future__ import annotations
from types import SimpleNamespace
from contextlib import contextmanager
import random
import numpy as np
import torch
from embodichain.lab.sim.motion.planners.utils import PlanResult
from ._place_recovery_fixture import recovery_fixture


def lateral_fixture(
    directory, *, prefix_valid=True, initial_ik_failures=127, self_collision_probes=3
):
    generator, context, goal, geometry, _, batches, _ = recovery_fixture(
        directory, "nominal"
    )
    del generator._generate_once
    generator.robot.uid = "fixture"
    generator.planner.cfg.sim_instance_id = 0
    calls = []

    def inverse(pose, joint_seed, **kw):
        calls.append({"pose": pose.tolist(), "seed": joint_seed.tolist()})
        torch.rand(())
        np.random.rand()
        random.random()
        return torch.tensor([len(calls) > initial_ik_failures]), pose[:, :3, 3].clone()

    generator.robot.compute_ik.side_effect = inverse
    collision_calls = []

    def collision(robot, positions, part):
        collision_calls.append(positions.clone())
        if len(collision_calls) <= self_collision_probes:
            return {"valid": False, "kind": "self"}
        return {
            "valid": True,
            "frames": len(positions),
            "scope": "discrete_robot_meshes",
        }

    payload_calls = []

    def generate_prefix(targets, options):
        payload_calls.append(
            {
                "start": options.start_qpos.tolist(),
                "target": targets[0].qpos.tolist(),
                "velocity_limit": options.velocity_limit,
                "acceleration_limit": options.acceleration_limit,
                "sample_count": options.sample_count,
            }
        )
        torch.rand(())
        np.random.rand()
        random.random()
        q = torch.lerp(
            options.start_qpos[:, None],
            targets[0].qpos[:, None],
            torch.linspace(0, 1, 20)[None, :, None],
        )
        dt = q.new_full(q.shape[:2], 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([prefix_valid]), positions=q, dt=dt)

    return (
        generator,
        context,
        goal,
        geometry,
        calls,
        batches,
        collision,
        collision_calls,
        generate_prefix,
        payload_calls,
    )


def payload_query_fixture(robot, generate_prefix):
    from embodichain.lab.sim.motion.planners.base_planner import CollisionWorldInfo

    observed, closed = [], []

    class Queries:
        def planning_device(self):
            return torch.device("cpu")

        def plan(self, **kwargs):
            raise AssertionError("payload transport must use the typed query lifetime")

        @contextmanager
        def open(self, *, context, control_part, full_qpos, payload):
            observed.append(
                {
                    "object": payload.object_id,
                    "spheres": payload.spheres.tolist(),
                    "pose": payload.object_pose.tolist(),
                    "full_qpos": full_qpos.tolist(),
                }
            )
            try:
                yield SimpleNamespace(
                    generate=generate_prefix,
                    collision_world_info=CollisionWorldInfo(
                        entity_ids=("table", "target")
                    ),
                )
            finally:
                closed.append(True)

    query = Queries()
    query.robot = robot
    return query, observed, closed
