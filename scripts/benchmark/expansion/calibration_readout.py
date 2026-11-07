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

"""Measure native camera calibration readout at B=1,8,32 without stepping physics.

Run: python scripts/benchmark/expansion/calibration_readout.py --job JOB.yaml --output OUTPUT
The reference reproduces the former per-row full-batch getter calls. Results are
for calibration readout only, not whole-task throughput. Native probes use one
isolated process each; returned calibration values must match exactly.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import time
import statistics


def _reference(env):
    rows = []
    for row in range(env.num_envs):
        result = {}
        for uid in env.sim.get_sensor_uid_list():
            camera = env.sim.get_sensor(uid)
            if not callable(getattr(camera, "get_intrinsics", None)):
                continue
            intrinsic = camera.get_intrinsics()
            if isinstance(intrinsic, tuple):
                poses = camera.get_left_right_arena_pose()
                result[uid] = {
                    "intrinsics_left": intrinsic[0][row].cpu().tolist(),
                    "intrinsics_right": intrinsic[1][row].cpu().tolist(),
                    "arena_pose_left": poses[0][row].cpu().tolist(),
                    "arena_pose_right": poses[1][row].cpu().tolist(),
                }
            else:
                result[uid] = {
                    "intrinsics": intrinsic[row].cpu().tolist(),
                    "arena_pose": camera.get_arena_pose(to_matrix=True)[row]
                    .cpu()
                    .tolist(),
                }
            result[uid].update(
                width=camera.cfg.width,
                height=camera.cfg.height,
                camera_axes="opengl",
                pose_convention="camera_to_arena_z_up",
            )
        rows.append(result)
    return tuple(rows)


def _probe(args):
    import torch
    from embodichain.lab.expansion.configured import (
        load_expansion_job,
        preflight_expansion_job,
        resolve_expansion_job,
    )
    from embodichain.lab.expansion.host import GymExpansionHost
    from embodichain.lab.expansion.journal import ExpansionJournal
    from embodichain.lab.expansion.visual import _calibration_batch
    from embodichain.lab.sim.scene import SceneView, capture_state, compare_states

    cfg = resolve_expansion_job(load_expansion_job(args.job))
    cfg.batch_size = args.probe_size
    cfg.capture_media = True
    cfg.output_dir = str(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    host = GymExpansionHost(
        cfg,
        preflight_expansion_job(cfg),
        args.output,
        ExpansionJournal(args.output),
        {},
    )
    setup = time.perf_counter() - started
    env = host.env
    view = SceneView(env.sim, env_ids=tuple(range(env.num_envs)))
    before = capture_state(view)
    clock = env.sim._visualization_sim_step
    try:
        expected = _reference(env)
        assert _calibration_batch(env) == expected
        for _ in range(3):
            _reference(env)
            _calibration_batch(env)
        counters = {"intrinsics": 0, "poses": 0}
        cameras = []
        for uid in env.sim.get_sensor_uid_list():
            camera = env.sim.get_sensor(uid)
            if not callable(getattr(camera, "get_intrinsics", None)):
                continue
            cameras.append(camera)
            for method, kind in (
                ("get_intrinsics", "intrinsics"),
                ("get_arena_pose", "poses"),
                ("get_left_right_arena_pose", "poses"),
            ):
                if not hasattr(camera, method):
                    continue
                original = getattr(camera, method)

                def counted(*values, _original=original, _kind=kind, **kwargs):
                    counters[_kind] += 1
                    return _original(*values, **kwargs)

                setattr(camera, method, counted)
        results = {}
        for name, read in (
            ("per_row_reference", _reference),
            ("batched", _calibration_batch),
        ):
            counters.update(intrinsics=0, poses=0)
            samples = []
            for _ in range(args.iterations):
                if env.device.type == "cuda":
                    torch.cuda.synchronize(env.device)
                start = time.perf_counter()
                actual = read(env)
                if env.device.type == "cuda":
                    torch.cuda.synchronize(env.device)
                samples.append(time.perf_counter() - start)
                assert actual == expected
            results[name] = {
                "mean_seconds": statistics.mean(samples),
                "minimum_seconds": min(samples),
                "maximum_seconds": max(samples),
                "native_getter_calls": dict(counters),
            }
        assert env.sim._visualization_sim_step == clock
        assert compare_states(before, capture_state(view)).passed
        report = {
            "batch_size": args.probe_size,
            "device": str(env.device),
            "setup_seconds": setup,
            "iterations": args.iterations,
            "warmup": 3,
            "exact_calibration_match": True,
            "physics_steps": 0,
            "scope": "native calibration getters, packing and transfer; not task throughput",
            "results": results,
            "python": sys.version,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
    finally:
        host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", default="1,8,32")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--probe-size", type=int)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.probe_size:
        _probe(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for size in map(int, args.sizes.split(",")):
        path = args.output / f"b{size}"
        with (args.output / f"b{size}.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--job",
                    str(args.job.resolve()),
                    "--output",
                    str(path.resolve()),
                    "--iterations",
                    str(args.iterations),
                    "--probe-size",
                    str(size),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        results.append(json.loads((path / "result.json").read_text()))
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(f"B={size}: calibration matched, no physics steps", flush=True)


if __name__ == "__main__":
    main()
