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

"""Analytic inputs and Newton query instrumentation for native query regression."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from embodichain.toolkits.graspkit.pg_grasp.native_contact import ClosingContactPolicy
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection


def source_policy(revision):
    """Decode only the pinned source fixture's historical experiment labels."""
    return ClosingContactPolicy(
        contact_halo=(
            "ignore" if revision == 1 else "reject" if revision < 4 else "report"
        ),
        reject_open_pad_contact=revision >= 5,
        depth_alternatives=revision >= 6,
        opening_check=revision >= 3,
        defer_source_proposals=revision >= 8,
        horizontal_pad_line=revision == 9,
    )


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
    options = {
        "policy": source_policy(case["revision"]),
        "report_metadata": {"revision": case["revision"]},
        "target_contact_offset": 0.002,
        "collision_factory": _ConvexCollection,
        "collision_metadata": _ConvexCollection.metadata(),
    }
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
        metadata = staticmethod(original.metadata)

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

        def distance_interval(self, other, limit):
            interval = self.inner.distance_interval(other.inner, limit)
            distance = interval.lower
            trace.append(
                {
                    "op": "distance",
                    "manager": self.index,
                    "other": other.index,
                    "distance": distance,
                }
            )
            return interval

        def close(self):
            self.inner.close()

    return Manager, trace


def decisions(value):
    """Compare source decisions/order separately from backend distance diagnostics."""
    if isinstance(value, dict):
        return {
            key: decisions(item)
            for key, item in value.items()
            if not key.endswith(("minimum_distance_m", "clearance_lower_bound_m"))
            and key
            not in {
                "distance",
                "minimum_rest_clearance_m",
                "mesh_distance_m",
                "collision_backend",
                "collision_backend_version",
                "distance_semantics",
            }
        }
    if isinstance(value, (tuple, list)):
        return [decisions(item) for item in value]
    return value


def decision_digest(value):
    """Hash the same source decision/order contract without backend diagnostics."""
    encoded = json.dumps(
        decisions(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
