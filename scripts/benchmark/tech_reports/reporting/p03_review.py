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
"""Recompute P-03 metrics from immutable run arrays, without a simulator.

Run: python -m scripts.benchmark.tech_reports.reporting.p03_review
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from .p03_metrics import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmarks/P-03"))
    parser.add_argument("--runs", nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "analysis/grasp-v1"
    out.mkdir(parents=True, exist_ok=True)
    geometry = root / "prepared/collision-v1/fingers.npz"
    fingers = dict(np.load(geometry))
    metrics_file = Path(__file__).with_name("p03_metrics.py")
    analyzer_hash = hashlib.sha256(
        metrics_file.read_bytes() + geometry.read_bytes()
    ).hexdigest()
    index = []
    all_rows = []
    for run in sorted((root / "runs").iterdir()):
        if args.runs and run.name not in args.runs:
            continue
        if not (run / "run.json").exists():
            continue
        meta = json.loads((run / "run.json").read_text())
        if "state_layout" not in meta:
            continue
        row = dict(
            run_id=run.name,
            platform=meta["platform"],
            status=meta["status"],
            config_version=meta["config"]["version"],
            **{
                k: meta["arguments"].get(k)
                for k in ["shape", "num_envs", "mode", "record", "calibrate", "latency"]
            },
        )
        row["mode"] = row["mode"] or "physics"
        row["failure"] = meta.get("failure")
        row["gpu_competitor_pids"] = []
        if (run / "gpu_memory.csv").exists():
            pid = json.loads((run / "launch.json").read_text()).get("pid")
            others = set()
            samples = []
            for line in (run / "gpu_memory.csv").read_text().splitlines():
                try:
                    ts, pid_text, memory = [s.strip() for s in line.split(",")]
                    measured_pid = int(pid_text)
                    epoch = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                    if measured_pid == pid:
                        samples.append((epoch, float(memory)))
                    else:
                        others.add(measured_pid)
                except ValueError:
                    continue
            row["gpu_competitor_pids"] = sorted(others)
            row["gpu_peak_process_mib"] = max((x[1] for x in samples), default=None)
            ram = np.genfromtxt(run / "process_memory.csv", delimiter=",", names=True)
            row["ram_peak_process_mib"] = float(ram["rss_bytes"].max() / 2**20)
            row["timing_windows"] = []
            for ep in meta.get("episodes", []):
                begin, end = ep.get("epoch_start"), ep.get("epoch_end")
                if begin is None:
                    continue
                gpu = np.array([x for x in samples if begin <= x[0] <= end]).reshape(
                    -1, 2
                )
                prior = [x for x in samples if x[0] <= begin]
                rmask = (ram["epoch_s"] >= begin) & (ram["epoch_s"] <= end)
                baseline_ram = (
                    ram["rss_bytes"][np.argmin(abs(ram["epoch_s"] - begin))] / 2**20
                )
                row["timing_windows"].append(
                    dict(
                        initial=ep["initial"],
                        wall_s=ep["wall_s"],
                        gpu_baseline_mib=prior[-1][1] if prior else None,
                        gpu_peak_mib=float(gpu[:, 1].max()) if len(gpu) else None,
                        ram_baseline_mib=float(baseline_ram),
                        ram_peak_mib=(
                            float(ram["rss_bytes"][rmask].max() / 2**20)
                            if rmask.any()
                            else None
                        ),
                    )
                )
        target = out / run.name
        target.mkdir(exist_ok=True)
        for path in sorted(run.glob("episode_*_state.npy")):
            initial = int(path.name.split("_")[1])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cached = target / f"episode_{initial:02d}.json"
            previous = json.loads(cached.read_text()) if cached.exists() else {}
            if (
                previous.get("raw_sha256") == digest
                and previous.get("analyzer_hash") == analyzer_hash
            ):
                measured = previous
            else:
                x = np.load(path, mmap_mode="r")
                t = np.load(path.with_name(path.name.replace("_state", "_time")))
                values = []
                for env in range(x.shape[1]):
                    result, trace = metrics(x[:, env], t, row["shape"], fingers)
                    values.append(dict(env=env, **result))
                    if env == 0:
                        np.savez_compressed(
                            target / f"episode_{initial:02d}_traces.npz", **trace
                        )
                measured = dict(
                    raw_sha256=digest,
                    analyzer_hash=analyzer_hash,
                    raw_file=str(path.relative_to(root)),
                    initial=initial,
                    values=values,
                )
                cached.write_text(json.dumps(measured, indent=2) + "\n")
            for value in measured["values"]:
                all_rows.append(
                    dict(
                        run_id=run.name,
                        platform=row["platform"],
                        shape=row["shape"],
                        num_envs=row["num_envs"],
                        mode=row["mode"],
                        initial=initial,
                        **value,
                    )
                )
        if (run / "force_calibration.npz").exists():
            data = np.load(run / "force_calibration.npz")
            x = data["state"]
            dt = data["t"][1] - data["t"][0]
            expected = data["force"][:20, :, 0].sum(0) * dt / 0.1
            row["force_calibration"] = dict(
                expected_delta_vx_m_s=expected.tolist(),
                measured_delta_vx_m_s=(x[20, :, 46] - x[0, :, 46]).tolist(),
                relative_error_percent=(
                    (x[20, :, 46] - x[0, :, 46] - expected) / expected * 100
                ).tolist(),
                velocity_change_after_stop_m_s=(x[40, :, 46] - x[20, :, 46]).tolist(),
                raw_sha256=hashlib.sha256(
                    (run / "force_calibration.npz").read_bytes()
                ).hexdigest(),
            )
        row["episodes"] = meta.get("episodes", [])
        index.append(row)
        print(run.name, row["status"], flush=True)
    (out / "runs.json").write_text(json.dumps(index, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    if all_rows:
        fields = list(dict.fromkeys(k for r in all_rows for k in r))
        with (out / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    (out / "method.json").write_text(
        json.dumps(
            dict(
                analyzer_hash=analyzer_hash,
                metric_source=str(metrics_file),
                geometry_file=str(geometry),
                contact_neighborhood_m=0.002,
                penetration_duration_resolution_m=1e-5,
                event_sampling_s=0.005,
                negative_sat_gap="Convex polyhedron minimum translation penetration; collision margins excluded",
                positive_sat_gap="Separating projection gap, not Euclidean distance",
                cycle="Continuous table clearance > combined contact offset and bilateral finger contact throughout 3-7 s; table neighborhood and both fingers separated at 10 s. Geometry is measured from original source convex hulls.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
