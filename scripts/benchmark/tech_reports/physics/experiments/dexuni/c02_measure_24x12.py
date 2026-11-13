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
"""Measure the user-selected 24x12 budget with advisory geometry checks.

Run: Isaac Python measure_24x12.py --iterations 24 --substeps 12 --until 12 ...
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_pilot import _Timers
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab import c02

__all__ = ["main"]
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.c02_record_24x12 import (
    _record as _ORIGINAL_RECORD,
)


def _record(
    platform: str,
    inputs: dict,
    out: Path,
    step: Any,
    readings: Any,
    capture: Any,
    diagnostics: Any,
    audit: dict,
    until: float | None = None,
) -> None:
    """Retain short-window raw timings even when original quality gates fail."""
    import numpy as np

    assert capture is None and until is not None
    timers = _Timers("isaaclab", step, readings)
    source = Path(sys.modules[_Timers.__module__].__file__)
    for path in (Path(__file__), source):
        shutil.copy2(path, out / "source" / path.name)
    protocol = dict(
        scope="Short initial continuous episode window; no full packing/lift claim",
        argv=sys.argv,
        until_s=until,
        warmup_steps=30,
        primary="sim.step plus scene.update; owned Warp device synchronized before/after; excludes IK/commands/readback",
        memory="Sampled resident process memory outside timers; no transient peak measurement",
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), source, Path(c02.__file__))
        },
    )
    (out / "probe_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    error = None
    try:
        _ORIGINAL_RECORD(
            platform,
            inputs,
            out,
            timers.advance,
            timers.observe,
            None,
            diagnostics,
            audit,
            until,
        )
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        timers.restore()
        fields = [
            "frame",
            "steady",
            "physics_s",
            "outer_s",
            "readback_s",
            "actual_time_s",
        ]
        with (out / "probe_steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(timers.rows)
        values = np.asarray(
            [
                row["physics_s"] * 1000
                for row in timers.rows
                if row["steady"] and "physics_s" in row
            ]
        )
        summary = dict(
            error=error,
            complete_steps=len(timers.rows),
            steady_steps=len(values),
            warmup_steps=30,
            actual_seconds=timers.rows[-1].get("actual_time_s") if timers.rows else 0,
            mean_ms=float(values.mean()) if len(values) else None,
            median_ms=float(np.median(values)) if len(values) else None,
            p95_ms=float(np.percentile(values, 95)) if len(values) else None,
            memory_samples=timers.memory,
        )
        (out / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report = f"""# C02 short budget probe

Continuous initial window only; user-authorized shape/upright failures recorded without early stop. Full packing/lift untested.

Time & Memory

| cost_time_ms | median_ms | p95_ms | steady_samples | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
| --- | --- | --- | --- | --- | --- | --- |
| {summary['mean_ms']} | {summary['median_ms']} | {summary['p95_ms']} | {len(values)} | unmeasured | unmeasured | unmeasured |

Success & Other Metrics

| initial window completed | actual_seconds | original gate error |
| --- | --- | --- |
| {error is None} | {summary['actual_seconds']} | {error} |

Leaderboard

| candidate | full-task result | ranking |
| --- | --- | --- |
| See probe_protocol.json argv | untested | none: preliminary window |
"""
        (out / "PROBE_REPORT.md").write_text(report)
        print(
            "Probe timing summary:",
            {k: v for k, v in summary.items() if k != "memory_samples"},
            flush=True,
        )


def main() -> None:
    """Run the original CLI with only recording instrumentation replaced."""
    c02.record = _record
    c02.main()


if __name__ == "__main__":
    main()
