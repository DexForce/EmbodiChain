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

"""Adaptively locate finite-window stack boundaries, retaining every trial.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.p02_stack_sweep
"""

from __future__ import annotations

import os

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.benchmark.tech_reports.reporting.p02_stack_metrics import measure_stack

__all__ = ["PROFILES", "main"]
PROFILES = {
    "dexuni10": ("embodichain", "DexUni / VBD 10 iterations", ["--iterations", "10"]),
    "physx4": ("isaaclab", "PhysX TGS 4/1", ["--position-iterations", "4"]),
    "dexuni100group": (
        "embodichain",
        "DexUni / VBD 100 iterations / contact grouping",
        ["--iterations", "100", "--contact-grouping"],
    ),
    "physx100": ("isaaclab", "PhysX TGS 100/1", ["--position-iterations", "100"]),
    "dexuni100group_no_snap": (
        "embodichain",
        "DexUni / VBD 100 iterations / contact grouping / no contact snap",
        ["--iterations", "100", "--contact-grouping", "--no-contact-snap"],
    ),
}


def main() -> None:
    """Screen three initials, then verify ten, always for the requested 10 s."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=[p for p in PROFILES if p != "dexuni100group_no_snap"],
    )
    parser.add_argument("--max-layers", type=int, default=128)
    args = parser.parse_args()
    project = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    config = (
        project
        / "scripts/benchmark/tech_reports/assets/physics/P-02/stack-limit-10s-sleep.json"
    )
    out = project / "outputs/benchmarks/P-02/stack-limit-10s"
    duration = 10
    (out / "analysis/by-run").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    ledger = out / "search-ledger.jsonl"
    cache = {}
    for p in sorted((out / "runs").glob("*/run.json")):
        m = json.loads(p.read_text())
        if m["status"] == "completed" and m["phase"] != "capture":
            cache[
                (m["solver"], m["scene"]["layers"], m["duration_s"], m["episode_start"])
            ] = p

    def probe(
        profile: str, layers: int, duration: int, count: int, stop_on_event: bool
    ) -> list[dict]:
        platform, solver, options = PROFILES[profile]
        rows = []
        order = list(range(count))
        if count == 10 and stop_on_event:
            failures = {i: 0 for i in order}
            known_failure = set()
            for metric in (out / "analysis/by-run").glob("*.json"):
                previous = json.loads(metric.read_text())
                if (
                    previous["profile"] == profile
                    and not previous["intact_full_window"]
                ):
                    failures[previous["initial"]] += 1
                    if previous["layers"] == layers:
                        known_failure.add(previous["initial"])
            order.sort(key=lambda i: (i not in known_failure, -failures[i], i))
        for initial in order:
            key = (solver, layers, duration, initial)
            if key not in cache:
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.benchmark.tech_reports.physics.experiments.p02_batch",
                    "--phase",
                    "diagnostic",
                    "--config",
                    str(config),
                    "--output-root",
                    str(out),
                    "--scenes",
                    f"stack_{layers:03d}",
                    "--platforms",
                    platform,
                    "--count",
                    "1",
                    "--episode-start",
                    str(initial),
                    "--duration",
                    str(duration),
                    "--tag",
                    f"_{profile}_{duration}s",
                    *options,
                ]
                subprocess.run(cmd, cwd=project, check=True)
                candidates = sorted(
                    (out / "runs").glob(
                        f"*_{profile}_{duration}s_*stack_{layers:03d}*initial{initial:02d}/run.json"
                    )
                )
                cache[key] = candidates[-1]
            p = cache[key]
            m = json.loads(p.read_text())
            metric_path = out / "analysis/by-run" / (p.parent.name + ".json")
            if metric_path.exists():
                row = json.loads(metric_path.read_text())
            else:
                row, curves = measure_stack(p.parent / f"episode_{initial:02d}.npz", m)
                row["profile"] = profile
                metric_path.write_text(json.dumps(row, indent=2) + "\n")
                np.savez_compressed(metric_path.with_suffix(".npz"), **curves)
            rows.append(row)
            print(
                "OBSERVED",
                profile,
                layers,
                duration,
                initial,
                "first_event",
                row["first_geometry_event_s"],
                "max_speed",
                round(row["max_speed_m_s"], 4),
                flush=True,
            )
            if stop_on_event and not row["intact_full_window"]:
                break
        entry = dict(
            time=datetime.now().isoformat(),
            profile=profile,
            layers=layers,
            duration_s=duration,
            trials=len(rows),
            intact=sum(r["intact_full_window"] for r in rows),
            runs=[r["run_id"] for r in rows],
        )
        with ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return rows

    result_path = out / "boundary-results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    for profile in args.profiles:
        if profile in results:
            continue
        minimum = 5 if "group" in profile else 1
        low, high = minimum - 1, minimum
        while high <= args.max_layers:
            r = probe(profile, high, duration, 3, True)
            if not all(x["intact_full_window"] for x in r):
                break
            low, high = high, high * 2
        if high > args.max_layers:
            high = args.max_layers
            r = probe(profile, high, duration, 3, True)
            if all(x["intact_full_window"] for x in r):
                low = high
        while high - low > 1:
            mid = (low + high) // 2
            r = probe(profile, mid, duration, 3, True)
            if all(x["intact_full_window"] for x in r):
                low = mid
            else:
                high = mid
        candidate = max(minimum, low)
        verified_low = minimum - 1
        rejected_high = high if low < high else args.max_layers + 1
        r = probe(profile, candidate, duration, 10, True)
        if len(r) == 10 and all(x["intact_full_window"] for x in r):
            verified_low = candidate
        else:
            rejected_high = candidate
        # Both screening and ten-initial verification use binary subdivision.
        # A rejection stops scheduling new initials; each launched run lasts 10 s.
        while rejected_high - verified_low > 1:
            mid = (verified_low + rejected_high) // 2
            r = probe(profile, mid, duration, 10, True)
            if len(r) == 10 and all(x["intact_full_window"] for x in r):
                verified_low = mid
            else:
                rejected_high = mid
        candidate = verified_low
        if candidate < minimum:
            minimum_rows = probe(profile, minimum, duration, 10, False)
            results[profile] = dict(
                highest_verified_layers=None,
                verification_seconds=duration,
                initial_count=10,
                minimum_supported_layers=minimum,
                minimum_layer_intact=sum(r["intact_full_window"] for r in minimum_rows),
                search_ceiling=args.max_layers,
                interpretation="No ten-initial intact boundary found in the supported tested range; not evidence that every greater layer count fails.",
            )
            result_path.write_text(json.dumps(results, indent=2) + "\n")
            print("BOUNDARY", profile, results[profile], flush=True)
            continue
        if candidate < args.max_layers:
            next_rows = probe(profile, candidate + 1, duration, 10, False)
            assert any(not r["intact_full_window"] for r in next_rows)
        results[profile] = dict(
            highest_verified_layers=candidate,
            verification_seconds=duration,
            initial_count=10,
            next_layer_intact=(
                sum(r["intact_full_window"] for r in next_rows)
                if candidate < args.max_layers
                else None
            ),
            search_ceiling=args.max_layers,
            minimum_supported_layers=minimum,
            search_method="doubling, three-initial binary screening, ten-initial binary verification, adjacent-layer completion",
            interpretation="Highest tested intact ten-initial layer count; finite-window observed boundary, not an absolute deterministic solver limit.",
        )
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print("BOUNDARY", profile, results[profile], flush=True)


if __name__ == "__main__":
    main()
