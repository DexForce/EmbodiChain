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
"""Place the source robot beside a fixed reconstructed tabletop."""

from __future__ import annotations
from copy import deepcopy
from itertools import product
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces import (
    sample_urdf_collision_surfaces,
)
from .inputs import ReconstructionInput, RegisteredScene
from ._kinematics import Robot

__all__ = ["place_robot_base"]


def _blocked_intervals(points, vertices, direction, clearance, maximum):
    # Convex envelopes are conservative for initial placement, not grasp planning.
    low, high = vertices.min(0) - clearance, vertices.max(0) + clearance
    finish = points + direction * maximum
    select = np.all(np.maximum(points, finish) >= low, axis=1) & np.all(
        np.minimum(points, finish) <= high, axis=1
    )
    points = points[select]
    if not len(points):
        return np.empty((0, 2))
    equations = ConvexHull(vertices).equations
    normals, offsets = equations[:, :3], equations[:, 3]
    slope = normals @ direction
    positive, negative = slope > 1e-9, slope < -1e-9
    parallel = ~(positive | negative)
    intervals = []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points) / 512)))):
        signed = batch @ normals.T + offsets - clearance
        enter = np.maximum(0, np.max(-signed[:, negative] / slope[negative], axis=1))
        leave = np.minimum(
            maximum, np.min(-signed[:, positive] / slope[positive], axis=1)
        )
        valid = (enter <= leave) & np.all(signed[:, parallel] <= 0, axis=1)
        intervals.extend(zip(enter[valid], leave[valid]))
    intervals = np.asarray(intervals).reshape(-1, 2)
    if not len(intervals):
        return intervals
    intervals = intervals[np.argsort(intervals[:, 0])]
    ends = np.maximum.accumulate(intervals[:, 1])
    starts = np.r_[0, np.flatnonzero(intervals[1:, 0] > ends[:-1]) + 1]
    stops = np.r_[starts[1:] - 1, len(intervals) - 1]
    return np.column_stack((intervals[starts, 0], ends[stops]))


def place_robot_base(
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    *,
    clearance_m: float = 0.004,
    max_translation_m: float = 0.20,
    replace_existing: bool = False,
) -> RegisteredScene:
    """Place the base outside initial objects and the arm's tabletop sweep.

    The one-dimensional search follows the horizontal table-to-base direction.
    It preserves mounting height, orientation, both arms' relative transforms,
    source joints, cameras and every scene asset. Collision surfaces use the
    existing URDF sampler. Generated object convex envelopes are conservative;
    Non-hand arm links are checked over the recorded motion against the tabletop;
    initial-pose clearance alone does not protect a turning shoulder. Native
    playback remains the check of physical stability and subsequent corrections.

    ``replace_existing`` recomputes from the recording's original mounting pose;
    it does not accumulate a new translation on the previous placement.
    """
    if not 0 < clearance_m < max_translation_m <= 0.3:
        raise ValueError("Invalid initial placement search bounds")
    if "robot_base_pose_world" in scene.description and not replace_existing:
        raise ValueError("Place an original registered scene only once")
    robot = Robot(inputs.robot)
    nodes = ET.parse(robot.path).getroot().findall("link")
    names = [node.get("name") for node in nodes if node.findall("collision")]
    if not names or "table" not in scene.vertices:
        raise ValueError("Initial placement needs robot collision geometry and table")
    surfaces = sample_urdf_collision_surfaces(
        robot.path,
        link_names=names,
        max_edge_by_link={name: 0.015 for name in names},
        max_subdivisions=8,
    )
    fk = robot.links(inputs.demonstration.qpos[:1])
    clouds = {
        name: points.numpy() @ fk[name][0, :3, :3].T + fk[name][0, :3, 3]
        for name, points in surfaces.items()
        if len(points)
    }
    points = np.concatenate(list(clouds.values()))
    center = np.asarray(scene.vertices["table"]).mean(0)
    direction = robot.base_pose[:3, 3] - center
    direction[2] = 0
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError("Table-to-base direction is undefined")
    direction /= np.linalg.norm(direction)
    envelopes = dict(scene.vertices)
    table = np.asarray(envelopes["table"])
    # The native convex cook can bridge the irregular tabletop edge. Keep the
    # mounting assembly outside its bounds, rather than fitting into small edge
    # recesses of the visual mesh. This envelope is never rendered or simulated.
    envelopes["table"] = np.array(list(product(*zip(table.min(0), table.max(0)))))
    intervals = [
        _blocked_intervals(
            points, np.asarray(vertices), direction, clearance_m, max_translation_m
        )
        for vertices in envelopes.values()
    ]
    joints = {
        joint.get("name"): joint.find("child").get("link")
        for joint in ET.parse(robot.path).getroot().findall("joint")
    }
    arm_links = {
        joints[robot.names[i]] for columns in robot.arm_columns for i in columns
    }
    # Stream the pose/mesh products. Do not allocate frames x all robot vertices,
    # and do not exclude intended finger/object contact by moving the base away.
    for start in range(0, len(inputs.demonstration.qpos), 32):
        poses = robot.links(inputs.demonstration.qpos[start : start + 32])
        for name in sorted(arm_links & surfaces.keys()):
            local = surfaces[name].numpy()
            world = (
                local[None] @ poses[name][:, :3, :3].transpose(0, 2, 1)
                + poses[name][:, None, :3, 3]
            ).reshape(-1, 3)
            intervals.append(
                _blocked_intervals(
                    world, envelopes["table"], direction, clearance_m, max_translation_m
                )
            )
    intervals = np.concatenate(intervals)
    distance = 0.0
    for begin, end in sorted(intervals.tolist()):
        if begin > distance:
            break
        distance = max(distance, end + 0.0001)
    if distance > max_translation_m:
        raise ValueError("No initial base placement inside the source-near bound")
    pose = robot.base_pose.copy()
    pose[:3, 3] += distance * direction
    description = deepcopy(scene.description)
    description["robot_base_pose_world"] = pose.tolist()
    description["robot_placement"] = dict(
        method="minimal horizontal table-to-base translation",
        translation_m=(distance * direction).tolist(),
        clearance_m=clearance_m,
        sampled_collision_points=len(points),
        motion_frames_checked=len(inputs.demonstration.qpos),
        motion_links_checked=sorted(arm_links & surfaces.keys()),
        blocked_intervals=int(len(intervals)),
        scene_modified=False,
        initial_joints_modified=False,
        validation="sampled URDF surfaces against table bounds and object convex envelopes",
    )
    return RegisteredScene(description, scene.vertices)
