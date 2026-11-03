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

"""Analytic inputs and actual-FCL instrumentation for native query regression."""

from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

CASES = [
    {"id": "bilateral", "revision": 1},
    {"id": "halo_veto", "revision": 2, "body": "halo"},
    {"id": "halo_diagnostic", "revision": 4, "body": "halo"},
    {"id": "body_first", "revision": 4, "body": "first"},
    {"id": "ambiguous_order", "revision": 4, "body": "ambiguous"},
    {"id": "opening_allowed_r4", "revision": 4, "stroke": "already_touching"},
    {"id": "opening_rejected_r5", "revision": 5, "stroke": "already_touching"},
    {"id": "contact_not_reached", "revision": 6, "stroke": "never_touching"},
    {"id": "depth_population", "revision": 6},
    {"id": "deferred_source", "revision": 8},
    {"id": "quality_level", "revision": 9},
    {"id": "quality_under_limit", "revision": 9, "angle": 1.9},
    {"id": "quality_over_limit", "revision": 9, "angle": 2.1},
    {"id": "quality_gravity", "revision": 9, "gravity": [1, 0, 0]},
    {"id": "quality_actual_axis", "revision": 9, "axis": 1, "angle": 20},
]


def query_inputs(case, part_type):
    parts = []
    distances = {
        "already_touching": [0.105, 0.08, 0.06],
        "never_touching": [0.15, 0.14, 0.13],
    }.get(case.get("stroke"), [0.15, 0.105, 0.08])
    for sign in (-1, 1):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, case.get("axis", 0), 3] = sign * np.asarray(distances)
        parts.append(
            part_type(
                f"pad_{sign}",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                True,
                0.002,
            )
        )
    if case.get("body"):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = {
            "halo": [0.112] * 3,
            "first": [0.105] * 3,
            "ambiguous": [0.15, 0.105, 0.105],
        }[case["body"]]
        parts.append(
            part_type(
                "housing",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                False,
                0.002,
            )
        )
    options = {"revision": case["revision"], "target_contact_offset": 0.002}
    if case["revision"] == 9:
        options["gravity_in_object"] = np.asarray(
            case.get("gravity", [0, 0, -1]), dtype=np.float64
        )
    return tuple(parts), (trimesh.creation.box((0.2, 0.2, 0.2)),), options


def query_pose(case):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "y", case.get("angle", 0), degrees=True
    ).as_matrix()
    return pose


def exercise(query, case):
    pose = query_pose(case)
    inspected = query.inspect(pose)
    repeated = query.inspect(pose.copy())
    candidates = torch.as_tensor(pose, dtype=torch.float64).repeat(1, 2, 1, 1, 1)
    mask = query.mask(candidates, torch.eye(4, dtype=torch.float64)[None])
    if case["revision"] == 8:
        if hasattr(query, "record_deferred_proposal"):
            query.record_deferred_proposal()
        else:
            query.deferred_source_proposal_calls += 1
    return {
        "inspect": inspected,
        "repeated": repeated,
        "mask": mask.tolist(),
        "depth_offsets": list(query.depth_offsets()),
        "report": stable_report(query.report()),
    }


def stable_report(value):
    if isinstance(value, dict):
        return {
            key: stable_report(item)
            for key, item in value.items()
            if key
            not in {
                "query_seconds",
                "build_seconds",
                "query_scope",
                "object_pose_at_capture",
            }
        }
    if isinstance(value, (tuple, list)):
        return [stable_report(item) for item in value]
    return deepcopy(value)


def recording_manager(original):
    trace = []
    created = []

    class Manager:
        def __init__(self):
            self.index = len(created)
            self.inner = original()
            created.append(self)

        def add_object(self, name, mesh, transform=None):
            trace.append(
                {
                    "op": "add",
                    "manager": self.index,
                    "name": name,
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                }
            )
            return self.inner.add_object(name, mesh, transform=transform)

        def set_transform(self, name, matrix):
            trace.append(
                {
                    "op": "transform",
                    "manager": self.index,
                    "name": name,
                    "matrix": np.asarray(matrix).tolist(),
                }
            )
            return self.inner.set_transform(name, matrix)

        def min_distance_other(self, other):
            distance = float(self.inner.min_distance_other(other.inner))
            trace.append(
                {
                    "op": "distance",
                    "manager": self.index,
                    "other": other.index,
                    "distance": distance,
                }
            )
            return distance

    return Manager, trace
