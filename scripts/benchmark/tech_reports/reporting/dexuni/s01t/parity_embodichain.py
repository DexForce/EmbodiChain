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
"""Legacy S01T diagnostic comparison, not a formal qualification gate.

Run: <numpy-python> scripts/benchmark/tech_reports/reporting/dexuni/s01t/parity_embodichain.py --root <parity-run-root> --out <new-summary.json>
Keep every parity attempt. This helper still pairs by state rather than frozen
attempt IDs; use only for pilot diagnosis (see SCRIPT_CONTRACT.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import analyze  # noqa: E402


def _load_compare_models():
    """S01-C's model diff tool (loaded by path; both tasks have analyze.py)."""
    import importlib.util

    path = TOOLS.parent / "s01c" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("compare_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_models = _load_compare_models()

METRICS = (
    "l0_mm",
    "hold_strain_pct",
    "peak_strain_pct",
    "pull_duration_s",
    "tcp_pull_mm",
    "recovery_error_pct",
    "clamp_drift_mm",
    "grip_drift_mm",
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


def runs(root: Path, platform: str) -> list:
    found = []
    for path in sorted((root / platform).glob("s01t_*")):
        run = analyze.load_run(path)
        if run["manifest"].get("status") == "completed":
            found.append(run)
    return found


def frame_differences(a, b) -> dict:
    sa, sb = a["states"], b["states"]
    n = min(len(sa["time"]), len(sb["time"]))
    out = analyze.paired_difference(a, b)
    tcp = np.linalg.norm(sa["tcp"][:n] - sb["tcp"][:n], axis=1)
    qpos = np.abs(sa["qpos"][:n] - sb["qpos"][:n])
    out.update(
        {
            "frames_a": int(len(sa["time"])),
            "frames_b": int(len(sb["time"])),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="directory holding raw_data/{dexsim,embodichain}",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = args.root / "raw_data"
    dexsim = {r["manifest"]["state"]["id"]: r for r in runs(raw, "dexsim")}
    embodichain = {r["manifest"]["state"]["id"]: r for r in runs(raw, "embodichain")}
    summary = {"pairs": {}}
    for state in sorted(set(dexsim) & set(embodichain)):
        a, b = dexsim[state], embodichain[state]
        ma, mb = analyze.task_metrics(a), analyze.task_metrics(b)
        metrics = {}
        for key in METRICS:
            va, vb = ma.get(key), mb.get(key)
            numeric = (
                isinstance(va, (int, float))
                and isinstance(vb, (int, float))
                and not isinstance(va, bool)
            )
            metrics[key] = {
                "dexsim": va,
                "embodichain": vb,
                "diff": vb - va if numeric else None,
            }
        pair = {
            "dexsim_run": a["path"].name,
            "embodichain_run": b["path"].name,
            "metrics": metrics,
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
            pair["model_diff"] = {
                "identical_arrays": report["identical"],
                "differing_arrays": sorted(report["differences"]),
                "label_mismatches": report["labels"],
            }
        summary["pairs"][state] = pair
    baseline = [
        r
        for r in runs(analyze.RAW, "dexsim")
        if r["manifest"]["state"]["id"] == "s00"
        and r["manifest"]["variant"] == "nominal"
        and r["manifest"]["kind"] == "full"
    ]
    if baseline and "s00" in dexsim:
        summary["dexsim_run_to_run_s00"] = {
            "phase1_run": baseline[0]["path"].name,
            "parity_run": dexsim["s00"]["path"].name,
            "frame_differences": frame_differences(baseline[0], dexsim["s00"]),
        }
    text = json.dumps(summary, indent=1, default=to_json)
    if args.out:
        args.out.write_text(text + "\n")
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
            print("  model differing:", pair["model_diff"]["differing_arrays"])
    if "dexsim_run_to_run_s00" in summary:
        print(
            "== DexSim run-to-run (s00):",
            summary["dexsim_run_to_run_s00"]["frame_differences"],
        )


if __name__ == "__main__":
    main()
