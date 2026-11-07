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

"""Complete Atomic hosts with deterministic native facts for measured source parity."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import torch
import trimesh
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    PlanningBudget,
    RecoveryPolicy,
)
from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeContactPart,
    NativeClosingQuery,
    NativeQueryScope,
)
from embodichain.lab.sim.scene.native_motion_scene import (
    NativeRestCollisionPart,
    NativeRestSceneBody,
    NativeRestSceneQuery,
)
from .test_ordinary_pick import _inputs


class MeasuredSession:
    subject_id = "object"
    source_uid = "source"
    container_id = None
    physics_backend = "default"

    def __init__(self, robot, context, mode):
        self.robot, self.context, self.mode = robot, context, mode
        self.queries = []
        self.scenes = []
        self.closed = False
        self.target = trimesh.creation.box((0.03, 0.03, 0.06))
        self.events = []

    def validate_observation(self):
        if self.closed:
            raise ValueError("controlled measured observation closed")

    def closing_query(self, request, context, semantics):
        self.events.append("closing")
        endpoint = request.binding.endpoint("primary", "grasp")
        opened = endpoint.joint_positions(
            "open", num_envs=1, device="cpu", dtype=torch.float32
        )
        parts = []
        for index, sign in enumerate((-1, 1)):
            poses = np.repeat(np.eye(4)[None], 65, axis=0)
            opening = float(opened[0, index])
            poses[:, 0, 3] = sign * np.linspace(
                0.012 + 0.038 * opening / 0.04, 0.012, 65
            )
            parts.append(
                NativeContactPart(
                    "pad" + str(index),
                    (trimesh.creation.box((0.01, 0.04, 0.08)),),
                    poses,
                    0.0,
                    True,
                    0.002,
                )
            )
        query = NativeClosingQuery(
            tuple(parts),
            (self.target,),
            0.0,
            scope=NativeQueryScope("measured-" + str(len(self.queries)), 0.0, 0),
            revision=4,
            collision_backend="fcl",
            target_contact_offset=0.002,
        )
        if self.mode == "closure_rejected" and self.queries:
            query.mask = lambda tcp, pose: torch.tensor([False])
        self.queries.append(query)
        return query

    def scene_query(self):
        self.events.append("scene")
        if self.mode == "unavailable":
            raise ValueError("controlled native world unavailable")
        bodies = [
            NativeRestSceneBody(
                NativeRestCollisionPart("source", (self.target,), 0.0),
                self.context.scene.entities["object"].pose[0].numpy(),
            )
        ]
        if self.mode == "blocked":
            bodies.append(
                NativeRestSceneBody(
                    NativeRestCollisionPart(
                        "wall", (trimesh.creation.box((0.8, 0.8, 0.8)),), 0.0
                    ),
                    np.eye(4),
                )
            )
        scene = NativeRestSceneQuery(
            tuple(bodies),
            scope=NativeQueryScope("measured-scene", 0.0, 0),
            validate_observation=self.validate_observation,
        )
        self.scenes.append(scene)
        return scene

    def arm_parts(self, *args):
        raise AssertionError("Pick must not capture Place arm geometry")

    def inside_inputs(self, *args):
        raise AssertionError("Pick must not read an inside route")

    def close(self):
        for query in self.queries:
            query.close()
        for scene in self.scenes:
            scene.close()
        self.closed = True


class MeasuredProvider:
    def __init__(self, robot, context, mode):
        self.robot, self.context, self.mode = robot, context, mode
        self.sessions = []

    def prepare(self, request, context):
        value = MeasuredSession(self.robot, context, self.mode)
        self.sessions.append(value)
        return value


def fixture(directory, monkeypatch, mode="accepted"):
    base, invocation, context, calls, ik = _inputs(
        directory, monkeypatch, ik_failure=mode == "base_failure"
    )
    robot = base.robot
    root = torch.eye(4)[None]
    robot.get_local_pose = lambda **kw: root
    original_fk = robot.compute_fk.side_effect

    def fk(qpos, **kwargs):
        if "link_names" not in kwargs:
            return original_fk(qpos, **kwargs)
        tcp = original_fk(qpos)
        result = []
        for name in kwargs["link_names"]:
            part = int(name[-1])
            pose = torch.eye(4).repeat(len(qpos), 1, 1)
            pose[:, 0, 3] = (-1 if part == 0 else 1) * (
                0.012 + 0.038 * qpos[:, 3 + part] / 0.04
            )
            result.append(tcp @ pose)
        return torch.stack(result, 1)

    robot.compute_fk.side_effect = fk
    provider = MeasuredProvider(robot, context, mode)
    engine = AtomicActionEngine(
        base.motion_generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={
            "hand": base.planning_services.grasp_pose_generator("hand")
        },
        free_motion_query=base.planning_services.free_motion_query,
        measured_rigid_provider=provider,
    )
    invocation = engine.make_invocation(
        "pick_up",
        invocation.goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=invocation.motion_policy,
        skill_options=replace(invocation.skill_options, measured_preshape=True),
        recovery_policy=replace(
            invocation.recovery_policy,
            planning_budget=PlanningBudget(wall_seconds=90.0),
        ),
        invocation_id="pick",
    )
    return engine, invocation, context, provider, calls, ik


class MeasuredPlaceSession(MeasuredSession):
    container_id = "target"

    def __init__(self, robot, context, mode, geometry, contents):
        super().__init__(robot, context, mode)
        self.geometry, self.contents = geometry, contents

    def inside_inputs(self, context):
        from embodichain.lab.sim.atomic_actions.measured_rigid import (
            MeasuredInsidePlanningInputs,
        )

        self.events.append("inside")
        return MeasuredInsidePlanningInputs(
            "object", "target", "tray", 0.12, self.geometry, self.contents
        )

    def arm_parts(self, control_part, hand):
        self.events.append("arm")
        return (
            NativeRestCollisionPart(
                "arm_part", (trimesh.creation.box((0.005, 0.005, 0.005)),), 0.0
            ),
        )

    def scene_query(self):
        self.events.append("scene")
        if self.mode == "unavailable":
            raise ValueError("controlled native world unavailable")
        bodies = [
            NativeRestSceneBody(
                NativeRestCollisionPart("source", (self.target,), 0.0),
                self.context.scene.entities["object"].pose[0].numpy(),
            )
        ]
        floor = np.eye(4)
        floor[2, 3] = -0.01
        bodies.append(
            NativeRestSceneBody(
                NativeRestCollisionPart(
                    "tray", (trimesh.creation.box((0.6, 0.4, 0.02)),), 0.0
                ),
                floor,
            )
        )
        if self.mode == "blocked":
            bodies.append(
                NativeRestSceneBody(
                    NativeRestCollisionPart(
                        "wall", (trimesh.creation.box((0.8, 0.8, 0.8)),), 0.0
                    ),
                    np.eye(4),
                )
            )
        scene = NativeRestSceneQuery(
            tuple(bodies),
            scope=NativeQueryScope("measured-scene", 0.0, 0),
            validate_observation=self.validate_observation,
        )
        self.scenes.append(scene)
        return scene


def place_fixture(directory, monkeypatch, mode="accepted"):
    from embodichain.lab.sim.atomic_actions import (
        PlaceOptions,
        MotionPolicy,
        EntityState,
        SceneSnapshot,
    )
    from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement
    from embodichain.lab.sim.atomic_actions.primitives import (
        _measured_rigid,
        _manipulation_checks,
    )
    from ._ordinary_pick_fixture import place_fixture as ordinary_place_fixture

    generator, context, goal, calls, ik = ordinary_place_fixture(directory)
    robot = generator.robot
    root = torch.eye(4)[None]
    robot.get_local_pose = lambda **kw: root
    original_fk = robot.compute_fk.side_effect

    def fk(qpos, **kwargs):
        if "link_names" not in kwargs:
            return original_fk(qpos, **kwargs)
        tcp = original_fk(qpos)
        result = []
        for name in kwargs["link_names"]:
            offset = torch.eye(4).repeat(len(qpos), 1, 1)
            if name == "arm_part":
                offset[:, 2, 3] = -0.2
            else:
                part = int(name[-1])
                offset[:, 0, 3] = (-1 if part == 0 else 1) * (
                    0.012 + 0.038 * qpos[:, 3 + part] / 0.04
                )
            result.append(tcp @ offset)
        return torch.stack(result, 1)

    robot.compute_fk.side_effect = fk
    target = context.scene.entities["target"].pose.clone()
    target[0, :3, 3] = torch.tensor([0.15, 0.05, 0.15])
    object_pose = original_fk(context.robot.qpos[:, :3])
    tray = torch.eye(4)[None]
    tray[0, 2, 3] = -0.01
    entities = {
        "target": replace(context.scene.entities["target"], pose=target),
        "tray": EntityState(tray),
        "object": EntityState(object_pose),
    }
    object_mesh = trimesh.creation.box((0.03, 0.03, 0.06))
    tray_mesh = trimesh.creation.box((0.6, 0.4, 0.02))

    def physical(mesh):
        return NS(
            get_vertices=lambda **kw: torch.tensor(mesh.vertices.copy()).float()[None],
            get_triangles=lambda **kw: torch.tensor(mesh.faces.copy()).long()[None],
        )

    objects = {"source": physical(object_mesh), "tray": physical(tray_mesh)}
    simulation = NS(get_rigid_object=objects.get)
    geometry = NativePlanarPlacement(simulation, "source", "tray", batch_size=1)
    contents = []
    if mode == "occupied":
        cargo = trimesh.creation.box((0.06, 0.06, 0.04))
        pose = torch.eye(4)[None]
        pose[0, :3, 3] = torch.tensor([0.15, 0.05, 0.02])
        entities["content"] = EntityState(pose)
        contents.append(("content", torch.tensor(cargo.vertices.copy()).float()))
    context = replace(context, scene=SceneSnapshot(0.0, 0, entities))

    class Provider:
        def __init__(self):
            self.robot = robot
            self.sessions = []

        def prepare(self, request, current):
            session = MeasuredPlaceSession(
                robot, current, mode, geometry, tuple(contents)
            )
            self.sessions.append(session)
            return session

    provider = Provider()
    proxy = lambda robot, positions, part: {
        "valid": True,
        "frames": len(positions),
        "scope": "discrete_robot_meshes",
    }
    monkeypatch.setattr(_measured_rigid, "inspect_robot_motion", proxy)
    monkeypatch.setattr(_manipulation_checks, "inspect_robot_motion", proxy)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        measured_rigid_provider=provider,
    )
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PlaceOptions(
            measured_inside=True,
            linear_segments=True,
            hand_interp_steps=12,
            release_settle_steps=0,
            lift_height=0.05,
            cartesian_waypoint_count=2,
            preserve_current_object_orientation=True,
        ),
        recovery_policy=RecoveryPolicy(
            planning_budget=PlanningBudget(wall_seconds=90.0)
        ),
        invocation_id="place",
    )
    return engine, invocation, context, provider, calls, ik
