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
"""Legacy S01C diagnostic comparison, not a formal qualification gate.

Run: <numpy-python> scripts/benchmark/tech_reports/reporting/dexuni/s01c/parity_embodichain.py
Keep every parity attempt. This helper still pairs by state rather than frozen
attempt IDs; use only for pilot diagnosis (see SCRIPT_CONTRACT.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import analyze  # noqa: E402
import compare_models  # noqa: E402

SUBTASK_DIR = analyze.SUBTASK_DIR
PARITY = SUBTASK_DIR / "parity_embodichain"
METRICS = (
    "h0_mm",
    "hold_compression_pct",
    "peak_compression_pct",
    "press_duration_s",
    "recovery_error_pct",
    "grip_slip_episode_max_mm",
    "plate_tilt_max_deg",
    "block_xy_drift_mm",
    "tcp_tracking_error_hold_mm",
    "finger_hold_mean_mm",
    "step_ms_mean",
    "init_s",
    "success",
)


def to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def runs(platform: str, root: Path) -> dict:
    found = {}
    for path in sorted((root / platform).glob("s01c_*")):
        run = analyze.load_run(path)
        if run["manifest"].get("status") == "completed":
            found[run["manifest"]["state"]["id"]] = run
    return found


def frame_differences(a, b) -> dict:
    sa, sb = a["states"], b["states"]
    n = min(len(sa["time"]), len(sb["time"]))
    out = analyze.paired_difference(a, b)
    plate = np.linalg.norm(
        sa["plate_pose"][:n, :3, 3] - sb["plate_pose"][:n, :3, 3], axis=1
    )
    tcp = np.linalg.norm(sa["tcp"][:n] - sb["tcp"][:n], axis=1)
    qpos = np.abs(sa["qpos"][:n] - sb["qpos"][:n])
    out.update(
        {
            "frames_a": int(len(sa["time"])),
            "frames_b": int(len(sb["time"])),
            "plate_position_max_mm": 1e3 * float(plate.max()),
            "tcp_max_mm": 1e3 * float(tcp.max()),
            "arm_qpos_max_rad": float(qpos[:, :7].max()),
            "finger_qpos_max_mm": 1e3 * float(qpos[:, 7:].max()),
            "phase_mismatch_frames": int((sa["phase"][:n] != sb["phase"][:n]).sum()),
        }
    )
    ea = a["manifest"]["controller"]["events"]
    eb = b["manifest"]["controller"]["events"]
    out["phase_start_diff_s"] = {k: round(eb[k] - ea[k], 6) for k in ea if k in eb}
    return out


def metric_rows(a, b) -> dict:
    ma, mb = analyze.task_metrics(a), analyze.task_metrics(b)
    rows = {}
    for key in METRICS:
        va, vb = ma.get(key), mb.get(key)
        diff = (
            vb - va
            if isinstance(va, (int, float))
            and isinstance(vb, (int, float))
            and not isinstance(va, bool)
            else None
        )
        rows[key] = {"dexsim": va, "embodichain": vb, "diff": diff}
    return rows


def main() -> None:
    dexsim = runs("dexsim", PARITY / "raw_data")
    embodichain = runs("embodichain", PARITY / "raw_data")
    summary = {"pairs": {}}
    for state in sorted(set(dexsim) & set(embodichain)):
        a, b = dexsim[state], embodichain[state]
        summary["pairs"][state] = {
            "dexsim_run": a["path"].name,
            "embodichain_run": b["path"].name,
            "metrics": metric_rows(a, b),
            "frame_differences": frame_differences(a, b),
            "segmentation_check_embodichain": analyze.segmentation_check(b),
        }
        if (a["path"] / "newton_model.npz").exists() and (
            b["path"] / "newton_model.npz"
        ).exists():
            report = compare_models.compare(
                compare_models.load(a["path"]),
                compare_models.load(b["path"]),
                1e-5,
                1e-7,
            )
            summary["pairs"][state]["model_diff"] = {
                "identical_arrays": report["identical"],
                "differing_arrays": sorted(report["differences"]),
                "label_mismatches": report["labels"],
            }
    baseline = [
        analyze.load_run(path)
        for path in sorted((analyze.RAW / "dexsim").glob("s01c_*"))
    ]
    baseline = [
        run
        for run in baseline
        if run["manifest"].get("status") == "completed"
        and run["manifest"]["state"]["id"] == "s00"
        and run["manifest"]["variant"] == "nominal"
        and run["manifest"]["kind"] == "full"
        and run["manifest"].get("scenario", "task") == "task"
    ]
    if baseline and "s00" in dexsim:
        summary["dexsim_run_to_run_s00"] = {
            "phase1_run": baseline[0]["path"].name,
            "parity_run": dexsim["s00"]["path"].name,
            "frame_differences": frame_differences(baseline[0], dexsim["s00"]),
        }
    out = PARITY / "parity_summary.json"
    out.write_text(json.dumps(summary, indent=1, default=to_json) + "\n")
    for state, pair in summary["pairs"].items():
        print(f"== {state}")
        for key, row in pair["metrics"].items():
            print(
                f"  {key:28s} dexsim={analyze.fmt(row['dexsim'])!s:>10} "
                f"embodichain={analyze.fmt(row['embodichain'])!s:>10} "
                f"diff={analyze.fmt(row['diff'])}"
            )
        print("  frames:", pair["frame_differences"])
        if "model_diff" in pair:
            print("  model:", pair["model_diff"])
    if "dexsim_run_to_run_s00" in summary:
        print(
            "== DexSim run-to-run (s00):",
            summary["dexsim_run_to_run_s00"]["frame_differences"],
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
