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

"""Full canonical drawer skill plans with live pose readers and analytic motion."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionCommand,
    Place,
    PlaceOptions,
    MotionPolicy,
    SceneSnapshot,
    EntityState,
)
from embodichain.lab.sim.atomic_actions.drawer_planning import DrawerPlanningInputs
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.scene.drawer_observation import DrawerPayloadObservation
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from ._ordinary_pick_fixture import place_fixture


def drawer_fixture(directory, mode="nominal", *, upright=False):
    generator, context, goal, _, _ = place_fixture(directory)
    calls = []
    poses = torch.eye(4)[None]
    poses[:, 2, 3] = 0.04
    link = torch.eye(4)[None]
    vertices = context.task.held_objects["arm"].semantics.affordance.mesh_vertices[None]
    obj = SimpleNamespace(
        get_vertices=lambda **kw: vertices, get_local_pose=lambda **kw: poses
    )
    art = SimpleNamespace(
        joint_names=["rail"],
        get_link_pose=lambda *a, **kw: link,
        get_qpos=lambda: torch.tensor([[0.4]]),
    )
    sim = SimpleNamespace(
        get_articulation=lambda _: art, get_rigid_object=lambda _: obj
    )
    binding = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/grip",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.4),
    )
    observation = DrawerPayloadObservation(
        sim, binding, "object", (-0.2, -0.2, 0.0), (0.2, 0.2, 0.2)
    )
    queries = []

    class Query:
        def __enter__(self):
            queries.append("open")
            return self

        def __exit__(self, *args):
            queries.append("close")

        def inspect(self, positions):
            clear = mode != "articulation_blocked"
            return SimpleNamespace(
                metadata=lambda: {
                    "valid": clear,
                    **(
                        {"frames": len(positions)}
                        if clear
                        else {
                            "frame": 0,
                            "kind": "scene",
                            "pairs": [["arm", "cabinet"]],
                        }
                    ),
                }
            )

    provider = SimpleNamespace(
        robot=generator.robot,
        inputs=lambda request, current: (
            DrawerPlanningInputs("target", "cabinet__drawer", observation),
        ),
        prepare_articulation_query=lambda *a: Query(),
    )

    def generate(targets, options):
        points = torch.cat(
            (
                options.start_qpos[:, None],
                torch.stack([value.xpos[:, :3, 3] for value in targets], 1),
            ),
            1,
        )
        count = options.sample_count or 200
        indices = torch.linspace(0, count - 1, len(points[0])).round().to(torch.long)
        path = points.new_empty((1, count, 3))
        for index in range(len(indices) - 1):
            first, last = int(indices[index]), int(indices[index + 1])
            path[:, first : last + 1] = torch.lerp(
                points[:, index : index + 1],
                points[:, index + 1 : index + 2],
                torch.linspace(0, 1, last - first + 1)[None, :, None],
            )
        calls.append(
            {
                "targets": torch.stack([value.xpos for value in targets], 1).tolist(),
                "count": count,
            }
        )
        success = mode not in ("exhausted", "articulation_blocked") and not (
            mode == "second_candidate" and len(calls) == 1
        )
        dt = torch.full((1, count), 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([success]), positions=path, dt=dt)

    generator._generate_once = generate
    context = replace(
        context,
        scene=SceneSnapshot(
            0.0, 0, {**context.scene.entities, "cabinet__drawer": EntityState(link)}
        ),
    )
    engine = AtomicActionEngine(
        generator,
        load_builtins=False,
        drawer_planning_provider=provider,
        control_profiles={
            "hand": ControlPartCommandProfile(
                commands={
                    "open": JointPositionCommand(torch.tensor([0.04, 0.04])),
                    "grasp": JointPositionCommand(torch.zeros(2)),
                }
            )
        },
    )
    engine.register(Place())
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=260,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            dynamic_collision_mode="off",
            acceleration_limit=None,
        ),
        skill_options=PlaceOptions(
            drawer_placement=True,
            preserve_vertical=upright,
            hand_interp_steps=5,
            release_settle_steps=3,
            monitor_scene_until_approach=True,
        ),
        invocation_id="drawer-place",
    )
    return engine, generator, context, invocation, calls, queries, observation


def drawer_transport_fixture(directory, *, upright=False, fail=False):
    from embodichain.lab.sim.atomic_actions import (
        MoveHeldObject,
        MoveHeldObjectOptions,
        HeldObjectPoseGoal,
        SceneEntityPose,
    )

    old_engine, generator, context, _, calls, queries, observation = drawer_fixture(
        directory, "exhausted" if fail else "nominal", upright=upright
    )
    lifecycle = []
    payloads = []
    original = generator._generate_once

    class PayloadQueries:
        robot = generator.robot

        def planning_device(self):
            return torch.device("cpu")

        def plan(self, **kwargs):
            raise AssertionError("Drawer must use payload query lifetime")

        @contextmanager
        def open(self, *, payload, full_qpos, world_metadata=None, **kwargs):
            lifecycle.append("open")
            payloads.append((payload, full_qpos.clone()))
            if world_metadata is not None:
                world_metadata["world_ids"] = ["table", "cabinet__drawer"]
            try:
                yield SimpleNamespace(
                    generate=lambda targets, options: original(targets, options)
                )
            finally:
                lifecycle.append("close")

    engine = AtomicActionEngine(
        generator,
        load_builtins=False,
        drawer_planning_provider=old_engine.planning_services.drawer_planning_provider,
        joint_prefix_planner=PayloadQueries(),
        control_profiles={
            "hand": ControlPartCommandProfile(
                commands={
                    "open": JointPositionCommand(torch.tensor([0.04, 0.04])),
                    "grasp": JointPositionCommand(torch.zeros(2)),
                }
            )
        },
    )
    engine.register(MoveHeldObject())
    invocation = engine.make_invocation(
        "move_held_object",
        HeldObjectPoseGoal(SceneEntityPose("target")),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="motion_gen",
            sample_count=260,
            acceleration_limit=None,
            dynamic_collision_mode="off",
        ),
        skill_options=MoveHeldObjectOptions(
            drawer_transport=True, preserve_vertical=upright
        ),
        invocation_id="drawer-transport",
    )
    return (
        engine,
        generator,
        context,
        invocation,
        calls,
        queries,
        observation,
        lifecycle,
        payloads,
    )
