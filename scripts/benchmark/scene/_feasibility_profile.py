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

"""Optional native workload capture for the collision benchmark only."""

from __future__ import annotations

from pathlib import Path
import json
import os
import random
import time


def shape_hints(env: object) -> dict[str, str]:
    """Preserve SDK physical convex identity rather than guessing from triangles."""
    result = {}
    for name in env.robot.link_names:
        for i, shape in enumerate(env.robot.get_link_collision_shapes(name)):
            result[f"{env.robot.uid}/{name}/{i}"] = shape.shape_type.name
    for uid in env.sim.get_rigid_object_uid_list():
        for i, shape in enumerate(env.sim.get_rigid_object(uid).get_collision_shapes()):
            result[f"{uid}/{i}"] = shape.shape_type.name
    return result


def capture_workload(source: Path, output: Path) -> Path:
    """Profile the canonical workflow and reservoir-sample exact geometry queries.

    Timer scopes exclude reservoir serialization. This temporary instrumentation
    is process-local and restored in finally; it does not alter acceptance or
    own another execution lifecycle. Existing output paths are never reused.
    """
    from embodichain.utils.utility import load_config
    from embodichain.lab.task_program.integrations import feasibility as workflow
    from embodichain.lab.task_program.integrations.simulation import (
        feasibility_geometry,
    )
    from embodichain.lab.sim.scene.feasibility import FCLCollisionChecker
    from embodichain.lab.sim.scene.feasibility._values import plain
    from embodichain.lab.sim.scene.feasibility.planning import RobotFeasibilityBackend

    output.mkdir(parents=True, exist_ok=False)
    config = load_config(source)
    for key in ("deployment", "program"):
        if key in config["task_program"]:
            config["task_program"][key] = str(
                (source.parent / config["task_program"][key]).resolve()
            )
    config["search"].update(
        max_candidates=1, max_regions=1, validation_samples=1, max_wall_time_s=180.0
    )
    path = output / "workflow.json"
    path.write_text(json.dumps(config))
    original_init, original_check = (
        FCLCollisionChecker.__init__,
        FCLCollisionChecker.check,
    )
    original_frames = RobotFeasibilityBackend.frame_poses
    original_extract = feasibility_geometry.collision_bodies_from_simulation
    previous_exit = os.environ.get("EMBODICHAIN_SIM_EXIT_PROCESS")
    totals = {
        "geometry_setup_s": 0.0,
        "check_s": 0.0,
        "check_calls": 0,
        "fk_s": 0.0,
        "fk_calls": 0,
    }
    fixture = {"bodies": None, "ignored_pairs": None, "samples": [], "shape_hints": {}}
    rng = random.Random(1701)

    def extract(env):
        value = original_extract(env)
        if not fixture["shape_hints"]:
            fixture["shape_hints"] = shape_hints(env)
        return value

    def initialize(self, *args, **kwargs):
        started = time.perf_counter()
        original_init(self, *args, **kwargs)
        totals["geometry_setup_s"] += time.perf_counter() - started
        if fixture["bodies"] is None:
            fixture["bodies"] = plain(self.bodies)
            fixture["ignored_pairs"] = [list(pair) for pair in self.ignored_pairs]

    def check(self, frame_poses, **kwargs):
        started = time.perf_counter()
        value = original_check(self, frame_poses, **kwargs)
        totals["check_s"] += time.perf_counter() - started
        totals["check_calls"] += 1
        count = totals["check_calls"]
        slot = count - 1 if count <= 256 else rng.randrange(count)
        if slot < 256:
            sample = {
                "frames": plain(frame_poses),
                "options": plain(kwargs),
                "status": value[0].status,
                "clearance": value[1],
            }
            if count <= 256:
                fixture["samples"].append(sample)
            else:
                fixture["samples"][slot] = sample
        return value

    def frames(self, *args, **kwargs):
        started = time.perf_counter()
        value = original_frames(self, *args, **kwargs)
        totals["fk_s"] += time.perf_counter() - started
        totals["fk_calls"] += 1
        return value

    try:
        os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
        FCLCollisionChecker.__init__, FCLCollisionChecker.check = initialize, check
        RobotFeasibilityBackend.frame_poses = frames
        feasibility_geometry.collision_bodies_from_simulation = extract
        started = time.perf_counter()
        atlas = workflow.run_feasibility(path, output / "run")
        totals.update(
            wall_s=time.perf_counter() - started, metrics=plain(atlas.metrics)
        )
        (output / "profile.json").write_text(json.dumps(totals, indent=2))
        target = output / (
            "fixture-" + source.stem.removeprefix("search.").replace(".", "-") + ".json"
        )
        target.write_text(json.dumps(fixture))
        return target
    finally:
        FCLCollisionChecker.__init__, FCLCollisionChecker.check = (
            original_init,
            original_check,
        )
        RobotFeasibilityBackend.frame_poses = original_frames
        feasibility_geometry.collision_bodies_from_simulation = original_extract
        if previous_exit is None:
            os.environ.pop("EMBODICHAIN_SIM_EXIT_PROCESS", None)
        else:
            os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = previous_exit
