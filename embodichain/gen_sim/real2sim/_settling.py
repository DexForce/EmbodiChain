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
"""Persist the existing initial-settling stage without owning a simulator."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from embodichain.data_pipeline.real2sim.reconstruction.inputs import RegisteredScene
from embodichain.data_pipeline.real2sim.reconstruction.settling import (
    apply_settled_poses,
)
from ._artifacts import digest, write_json

__all__: list[str] = []


def settle_registered_scene(
    scene: RegisteredScene, rollout: Path, output: Path
) -> RegisteredScene:
    contacts = json.loads((rollout / "contacts.json").read_text())
    robot_actors = {
        int(key)
        for key, value in contacts["actors"].items()
        if "path='arena_0/robot'" in value
    }
    robot_contact = np.asarray(
        [
            any(robot_actors.intersection(pair) for pair in sample["user_ids"])
            for sample in contacts["samples"]
        ]
    )
    poses_path = rollout / "physics_rollout.npz"
    with np.load(poses_path, allow_pickle=False) as data:
        result, report = apply_settled_poses(
            scene,
            data["timestamps"],
            tuple(data["object_names"].tolist()),
            data["object_poses"],
            robot_contact,
        )
    write_json(output / "scene_before_settling.json", scene.description)
    np.savez_compressed(output / "scene_before_settling_vertices.npz", **scene.vertices)
    report["scene_before_sha256"] = digest(output / "scene.json")
    write_json(output / "scene.json", result.description)
    np.savez_compressed(output / "scene_vertices.npz", **result.vertices)
    report.update(
        source_rollout=str(poses_path.resolve()),
        source_rollout_sha256=digest(poses_path),
        scene_after_sha256=digest(output / "scene.json"),
    )
    write_json(output / "settling.json", report)
    return result
