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

"""Rebuild L-02 timing, content and memory tables without importing simulators."""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/benchmarks/L-02"))
    args = p.parse_args()
    root = args.root
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rows = []
    for folder in sorted((root / "runs").iterdir()):
        if not (folder / "config.json").exists():
            continue
        cfg = json.loads((folder / "config.json").read_text())
        meta = (
            json.loads((folder / "run.json").read_text())
            if (folder / "run.json").exists()
            else {}
        )
        launch = (
            json.loads((folder / "launch.json").read_text())
            if (folder / "launch.json").exists()
            else {}
        )
        exit_info = (
            json.loads((folder / "exit.json").read_text())
            if (folder / "exit.json").exists()
            else {}
        )
        row = {
            k: cfg[k]
            for k in [
                "platform",
                "shape",
                "object_count",
                "asset_mode",
                "cache_mode",
                "purpose",
                "repeat",
            ]
        }
        row.update(
            run_id=folder.name,
            protocol_version=cfg["version"],
            status=meta.get("status", "not_started"),
            exit_code=exit_info.get("code"),
            timeout=exit_info.get("timeout", False),
            failure_reason=meta.get("failure_reason"),
            T_ready_s=None,
            T_frame_s=None,
            ram_peak_mib=None,
            vram_peak_mib=None,
            cpu_delta_mb=None,
            gpu_delta_mb=None,
            gpu_competitor_pids=[],
            timing_eligible=False,
        )
        if exit_info.get("timeout"):
            row.update(status="timeout", failure_reason="180 s process budget exceeded")
        elif exit_info and exit_info["code"] != 0:
            row.update(
                status="failed",
                failure_reason=row["failure_reason"]
                or f"Process exit {exit_info['code']}",
            )
        elif not exit_info and row["status"] == "completed":
            row.update(
                status="unmonitored",
                failure_reason="No collected exit code or memory record",
            )
        times = meta.get("stage_times_s", {})
        # On externally terminated runs the console retains stages not flushed to run.json.
        if (folder / "console.log").exists():
            for line in (
                (folder / "console.log").read_text(errors="replace").splitlines()
            ):
                match = re.search(r"L02_STAGE (\w+) ([0-9.]+)", line)
                if match:
                    times[match[1]] = float(match[2])
        row.update(
            T_ready_s=times.get("ready"),
            T_frame_s=times.get("frame"),
            last_stage=next(reversed(times), None),
        )
        row.update({f"{key}_s": value for key, value in times.items()})
        if row["timeout"]:
            row["timeout_phase"] = (
                "after_first_frame"
                if "frame" in times
                else "before_first_frame" if "ready" in times else "before_first_state"
            )
            row["failure_reason"] = (
                f"{cfg['timeout_s']} s process budget exceeded: {row['timeout_phase']}"
            )
        for metric, start, finish in [
            ("scene_declaration_s", "engine_initialized", "declared"),
            ("scene_preparation_s", "declared", "prepared"),
            ("first_step_readback_s", "prepared", "ready"),
            ("first_rgb_readback_s", "ready", "frame"),
        ]:
            if start in times and finish in times:
                row[metric] = times[finish] - times[start]
        if times.get("ready") is not None:
            row["scene_to_ready_s"] = times["ready"] - times["engine_initialized"]
        unique = {x["path"]: x for x in cfg["objects"]}
        row.update(
            unique_source_files=len(unique),
            input_bytes=sum(x["bytes"] for x in unique.values()),
            visual_triangles=cfg["object_count"] * cfg["mesh_triangles"][cfg["shape"]],
            source_geometry_count=len({x["geometry_sha256"] for x in unique.values()}),
        )
        records = []
        if (folder / "memory.jsonl").exists():
            with (folder / "memory.jsonl").open() as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Interrupted final sample only.
        start = meta.get("entry_epoch", launch.get("start_epoch", 0))
        end = start + times.get("frame", float("inf"))
        memory = [x for x in records if start <= x["epoch"] <= end]
        if memory:
            ram = [x["rss_bytes"] / 2**20 for x in memory if x["rss_bytes"] is not None]
            total = launch.get("hardware", {}).get("gpu_total_bytes", float("inf"))
            process_gpu = []
            others = set()
            for x in memory:
                ids = set(x["tree_pids"])
                others.update(set(x["compute_pids"]) - ids)
                allocations = [
                    value
                    for pid, value in x["gpu_process_bytes"].items()
                    if int(pid) in ids and 0 <= value <= total
                ]
                process_gpu.append(sum(allocations) / 2**20)
            row.update(
                ram_peak_mib=max(ram) if ram else None,
                vram_peak_mib=max(process_gpu),
                cpu_delta_mb=ram[-1] - ram[0] if ram else None,
                gpu_delta_mb=process_gpu[-1] - process_gpu[0],
                gpu_competitor_pids=sorted(others),
                memory_sample_count=len(memory),
                memory_sample_interval_max_s=max(
                    np.diff([x["epoch"] for x in memory]), default=0
                ),
                card_delta_peak_mib=(
                    max(x["gpu_total_bytes"] for x in memory)
                    - launch["gpu_total_before_bytes"]
                )
                / 2**20,
            )
        if (folder / "model.npz").exists():
            a = json.loads((folder / "model-audit.json").read_text())
            m = dict(np.load(folder / "model.npz"))
            state = np.load(folder / "first_state.npy")
            inertia = m["inertia"]
            inertia = (
                inertia
                if inertia.ndim == 2
                else np.diagonal(inertia, axis1=-2, axis2=-1)
            )
            expect = np.array([o["inertia_kg_m2"] for o in cfg["objects"]])
            dimensions = np.array([a["extent"] for a in a["meshes"]])
            sizes = np.array([o["scale"] * 0.05 for o in cfg["objects"]])
            ncollisions = len(
                a.get("collisions", a.get("native", {}).get("shape_type", []))
            )
            row.update(
                observed_bodies=len(state),
                observed_collisions=ncollisions,
                observed_visual_meshes=len(a["meshes"]),
                all_finite=bool(np.isfinite(state).all()),
                position_error_max_m=float(
                    np.linalg.norm(state[:, :3] - cfg["positions"], axis=1).max()
                ),
                speed_max_mps=float(np.linalg.norm(state[:, 7:10], axis=1).max()),
                mass_error_max_kg=float(abs(m["mass"].astype(float) - 0.1).max()),
                inertia_error_max_kg_m2=float(abs(inertia - expect).max()),
                visual_extent_error_max_m=float(abs(dimensions - sizes[:, None]).max()),
            )
            row["content_complete"] = (
                len(state) == len(a["meshes"]) == cfg["object_count"]
                and ncollisions == cfg["object_count"] + 1
                and row["all_finite"]
            )
            materials = a.get("materials", [])
            if materials:
                colors = np.array([x["color_linear"][:3] for x in materials])
                expected = np.array([o["color"] for o in cfg["objects"]])
                row.update(
                    material_color_error_max=float(abs(colors - expected).max()),
                    unique_color_readback_count=len(np.unique(colors, axis=0)),
                    material_binding_count=len(materials),
                )
                if cfg["platform"] == "embodichain":
                    row["native_material_id_count"] = len({x["id"] for x in materials})
            for file in [
                "first_state.npy",
                "model.npz",
                "model-audit.json",
                "config.json",
                "run.json",
                "first_frame.png",
            ]:
                row[file + "_sha256"] = hashlib.sha256(
                    (folder / file).read_bytes()
                ).hexdigest()
        row["timing_eligible"] = (
            row["status"] == "completed"
            and bool(row.get("content_complete"))
            and not row["gpu_competitor_pids"]
            and row["T_frame_s"] is not None
            and cfg["purpose"] in ["main", "capacity"]
            and cfg["version"] == 2
        )
        rows.append(row)
    for row in rows:
        if row["cache_mode"] == "H" and row["purpose"] in ["main", "capacity"]:
            warmups = [
                x
                for x in rows
                if x["purpose"] == "prepare"
                and x["status"] == "completed"
                and x["run_id"] < row["run_id"]
                and all(
                    x[k] == row[k]
                    for k in [
                        "platform",
                        "shape",
                        "object_count",
                        "asset_mode",
                        "cache_mode",
                        "protocol_version",
                    ]
                )
            ]
            row["warmup_run_id"] = warmups[-1]["run_id"] if warmups else None
            row["timing_eligible"] = row["timing_eligible"] and bool(warmups)
    (out / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "runs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["purpose"] in ["main", "capacity"] and row["protocol_version"] == 2:
            groups[
                tuple(
                    row[k]
                    for k in [
                        "purpose",
                        "shape",
                        "object_count",
                        "asset_mode",
                        "cache_mode",
                        "platform",
                    ]
                )
            ].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        r = dict(
            zip(
                [
                    "purpose",
                    "shape",
                    "object_count",
                    "asset_mode",
                    "cache_mode",
                    "platform",
                ],
                key,
            )
        )
        valid = [x for x in items if x["timing_eligible"]]
        r.update(
            attempts=len(items),
            completed=sum(x["status"] == "completed" for x in items),
            eligible=len(valid),
            run_ids=[x["run_id"] for x in items],
        )
        for metric in [
            "T_ready_s",
            "T_frame_s",
            "scene_to_ready_s",
            "engine_initialized_s",
            "scene_declaration_s",
            "scene_preparation_s",
            "first_step_readback_s",
            "first_rgb_readback_s",
            "ram_peak_mib",
            "vram_peak_mib",
            "cpu_delta_mb",
            "gpu_delta_mb",
            "position_error_max_m",
            "mass_error_max_kg",
            "inertia_error_max_kg_m2",
            "visual_extent_error_max_m",
            "material_color_error_max",
            "native_material_id_count",
        ]:
            values = [x[metric] for x in valid if x.get(metric) is not None]
            if values:
                r[metric] = {
                    "median": float(np.median(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                }
        summary.append(r)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "method.json").write_text(
        json.dumps(
            {
                "T_ready": "perf_counter at module entry to first explicit physics step, synchronized full state and finiteness/shape check",
                "T_frame": "module entry to first scene-colored RGB acquisition including owning renderer synchronization; image export/audit excluded",
                "cache": "C is partially controlled new-process loading, not a fully cold cache; H follows completed same-case preparation. Analytic colliders avoid mesh cooking. Shader/kernel/OS and uncontrolled native renderer caches retained.",
                "memory": "20 ms requested NVML plus psutil sampling; actual cadence retained in JSONL; peak only until first frame; process GPU includes native graphics, not Torch allocator alone.",
                "comparison": "Only completed content-verified protocol-v2 main/capacity records without foreign compute pids in timed window. All failures/contended records remain in runs.csv. Ratios are Isaac median / EmbodiChain median.",
                "ground": "One hidden static analytic plane on each platform; count excludes ground unless collision_count reported.",
                "quality": "Full states, mass, inertia, visual/collision counts, triangle counts and material readbacks; continuous residuals, no arbitrary accuracy pass/fail line.",
                "aggregation": "Independent processes; median [min,max], no significance claims from 3 repeats; missing repetitions/attempts retained.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "runs": len(rows),
                "completed": sum(x["status"] == "completed" for x in rows),
                "eligible": sum(x["timing_eligible"] for x in rows),
                "groups": len(summary),
            }
        )
    )


if __name__ == "__main__":
    main()
