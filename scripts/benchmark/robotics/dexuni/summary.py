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
"""Summarize completed solver qualification JSON files under outputs/dexuni.

Run: python -m scripts.benchmark.robotics.dexuni.summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(directory: Path) -> None:
    """Preserve mode, task, size and certification denominator in every result."""
    load = []
    paired = []
    piles = []
    pushes = []
    experts = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        d = json.loads(path.read_text())
        if path.name.startswith("load_"):
            trials = d["repeats"]
            mass = np.array([row[0] for row in d["rows"]])
            weights = 9.81 * mass
            force = []
            slip = []
            grip = []
            passed = []
            rates = []
            for steps in trials:
                by = {x["phase"]: x for x in steps if x["frame"] == 90}
                force.extend(
                    abs(np.array(by["weigh"]["scale_force_N"]) - weights) / weights
                )
                close = np.array(by["close"]["q"])
                grip.extend(np.max(abs(close[:, :2] - 0.045), axis=1) * 1000)
                lift = np.array(by["lift"]["delta"])
                carry = np.array(by["carry"]["delta"])
                release = np.array(by["release"]["delta"])
                slip.extend(1000 * (lift[:, 2] - carry[:, 2]))
                OK = (
                    (lift[:, 2] >= 0.06)
                    & (carry[:, 2] >= 0.06)
                    & (carry[:, 1] >= 0.06)
                    & (release[:, 1] >= 0.06)
                    & (abs(release[:, 2]) <= 0.005)
                )
                passed.extend(OK)
                rates.append(len(mass) * 450 / sum(x["seconds"] for x in steps))
            load.append(
                dict(
                    mode=d["mode"],
                    material=d.get("common_material_friction"),
                    n=len(passed),
                    passed=int(sum(passed)),
                    median_scale_pct=100 * np.median(force),
                    worst_scale_pct=100 * np.max(force),
                    median_grip_mm=np.median(grip),
                    median_slip_mm=np.median(slip),
                    env_steps_s=np.median(rates),
                )
            )
        elif path.name.startswith("deploy_"):
            for e in d["episodes"]:
                experts[(d["task"], d["mode"])].append(e)
        elif path.name.startswith("paired_"):
            for idx, t in enumerate(d["trials"]):
                stages = {label: np.array(mask) for label, mask in t["stages"].items()}
                ok = np.logical_and.reduce(list(stages.values()), axis=0)
                timings = np.asarray(t["elapsed"])
                paired.append(
                    dict(
                        task=d["task"],
                        mode=d["mode"],
                        n=len(ok),
                        passed=int(ok.sum()),
                        rep=idx,
                        stage_passed={k: int(v.sum()) for k, v in stages.items()},
                        env_steps_s=d["batch"] * d["steps"] / sum(timings[:, 1]),
                        p95_block_ms_per_step=1000
                        * np.percentile(timings[:, 1] / timings[:, 0], 95),
                        reference=d["source_sha256"],
                    )
                )
        elif path.name.startswith("push_"):
            pushes.append(
                dict(
                    mode=d["mode"],
                    count=d["count"],
                    batch=d["batch"],
                    env_steps_s=d["batch"] / np.median(d["seconds"]),
                    blade_mm=1000 * np.median(d["blade_travel_m"]),
                    mean_payload_dx_mm=1000 * d["mean_payload_dx_m"],
                    snapshot_contacts=d["snapshot_contact_count"],
                    snapshot_dropped=d["snapshot_contact_dropped"],
                )
            )
        elif path.name.startswith("pile_"):
            q = d["quality"]
            piles.append(
                dict(
                    mode=d["mode"],
                    count=d["count"],
                    batch=d["batch"],
                    env_steps_s=d["batch"] / np.median(d["seconds"]),
                    p95_ms=np.percentile(d["seconds"], 95) * 1000,
                    iterations=d.get("iterations"),
                    cg_sparse=d.get("cg_sparse", False),
                    qualified=q["finite"]
                    and float(q["max_floor_penetration_mm"]) <= 2
                    and float(q["max_adjacent_z_overlap_mm"]) <= 2
                    and float(q["max_drift_mm"]) <= 2
                    and float(q["max_tilt_deg"]) <= 5,
                    **q,
                    process_resident_mib=d["process_gpu_resident_MiB"],
                    snapshot_contacts=d.get("snapshot_contact_count"),
                    snapshot_dropped=d.get("snapshot_contact_dropped"),
                )
            )
    full = []
    for (task, mode), episodes in sorted(experts.items()):
        if len({e["seed"] for e in episodes}) != len(episodes):
            raise ValueError(f"Duplicate seed in {task}/{mode}")
        passed = [
            e["episode"]["completed"] and all(e["episode"]["success"]) for e in episodes
        ]
        full.append(
            dict(
                task=task,
                mode=mode,
                n=len(passed),
                passed=sum(passed),
                seconds_median=np.median([e["seconds"] for e in episodes]),
                world_steps=sum(e["world_steps"] for e in episodes),
            )
        )
    rows = dict(load=load, experts=full, paired=paired, piles=piles, pushes=pushes)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    for name, items in rows.items():
        print(name)
        for item in items:
            print(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("outputs/dexuni"))
    summarize(parser.parse_args().directory)
