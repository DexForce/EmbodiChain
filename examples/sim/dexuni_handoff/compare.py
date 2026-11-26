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

"""Independently verify two native snapshots and task receipts for each demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import numpy as np


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def physical(value):
    # DexSim's scene_revision also counts MCP requests and is scoped to one
    # Session. Native topology and step identity are checked per Session.
    if isinstance(value, dict):
        return {k: physical(v) for k, v in value.items() if k != "scene_revision"}
    if isinstance(value, list):
        return [physical(v) for v in value]
    return value


def audit(root: Path) -> list[dict[str, Any]]:
    rows = []
    for profile in ("ur5", "rp1", "anymal"):
        reference = root / (profile + "-external")
        expected = physical(read(reference / "ready.json"))
        for source in ("external", "dexpilot"):
            folder = root / (profile + "-" + source)
            setup = read(folder / "scene.json")
            ready = physical(read(folder / "ready.json"))
            assert ready == expected, f"{profile}: physically different starting scenes"
            assert ready["simulation"]["solver"]["solver_type"] == "dexuni"
            result = read(folder / "result.json")
            assert (
                setup["session_id"] == result["session_id"]
                and setup["run_id"] == result["run_id"]
            )
            journal = folder / "sessions" / (result["session_id"] + ".jsonl")
            events = [json.loads(row) for row in journal.read_text().splitlines()]
            acquired = [
                e
                for e in events
                if e["status"] == "ok"
                and e.get("params", {})
                .get("arguments", {})
                .get("arguments", {})
                .get("operation")
                == "acquire"
            ]
            released = [
                e
                for e in events
                if e["status"] == "ok"
                and e.get("params", {})
                .get("arguments", {})
                .get("arguments", {})
                .get("operation")
                == "release"
            ]
            assert len(acquired) == 1 and len(released) == 1
            assert all(
                e["client"].get("run_id") == setup["run_id"]
                for e in acquired + released
            )
            floor_only = next(
                e["result"]
                for e in events
                if e.get("params", {}).get("arguments", {}).get("method")
                == "scene.list_objects"
            )
            assert (
                floor_only["count"] == 1
                and floor_only["objects"][0]["object_ref"] == "floor"
            )
            assert (
                acquired[0]["result"]["session_id"]
                == released[0]["result"]["session_id"]
                == setup["session_id"]
            )
            assert (
                acquired[0]["result"]["phase"] == "controlled"
                and released[0]["result"]["phase"] == "authoring"
            )
            assert (
                acquired[0]["result"]["topology_revision"]
                == released[0]["result"]["topology_revision"]
            )
            if source == "dexpilot":
                authored = [
                    e["params"]["arguments"]
                    for e in events
                    if e["status"] == "ok"
                    and e["client"].get("kind") == "mcp"
                    and e["params"].get("module") == "dexsim.kit.mcp.dispatch"
                ]
                ordered = iter(authored)
                for op, args in setup["authoring"]:
                    assert any(
                        a["method"] == op
                        and all(a["params"].get(k) == v for k, v in args.items())
                        for a in ordered
                    ), f"Missing DexPilot author operation {op}"
            if profile == "ur5":
                receipt = next(
                    r for r in result["receipts"] if r["command_id"] == "place"
                )
                sample = receipt["result"]
                assert (
                    all(sample["checks"].values()) and sample["center_error_m"] <= 0.01
                )
                metrics = {
                    "center_error_m": sample["center_error_m"],
                    "lift_m": sample["lift_m"],
                    "stable_s": sample["verified_stability_window_s"],
                }
            else:
                receipt = result["receipt"]
                sample = receipt["result"]
                assert sample["passed"] and receipt["status"] == "succeeded"
                job_dir = Path(receipt["receipt_path"]).parent
                if profile == "rp1":
                    video = next(job_dir.glob("*.mp4"))
                    samples = [
                        json.loads(l)
                        for l in (job_dir / "trajectory.jsonl").read_text().splitlines()
                    ]
                    streak = 0
                    for row in samples:
                        streak = (
                            streak + 1 if row["platform_support"] == [True, True] else 0
                        )
                    assert len(samples) == 400 and np.isclose(
                        streak * 0.02, sample["final_two_foot_support_s"]
                    )
                    assert all(
                        row["crate_pose"] is not None
                        and row["weld_max_error_m"] <= 0.03
                        for row in samples
                    )
                    probe = subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-count_frames",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=nb_read_frames",
                            "-of",
                            "csv=p=0",
                            str(video),
                        ],
                        text=True,
                    )
                    assert int(probe.strip()) == 400
                    metrics = {
                        "two_foot_support_s": sample["final_two_foot_support_s"],
                        "mass_kg": sample["payload_kg"],
                        "welded": sample["welded"],
                        "frames": 400,
                        "video": str(video),
                    }
                    assert metrics["welded"] and metrics["two_foot_support_s"] >= 0.5
                else:
                    trace = Path(sample["trajectory"])
                    samples = [json.loads(l) for l in trace.read_text().splitlines()]
                    first = ready["robot"]["root_pose"]
                    from scipy.spatial.transform import Rotation

                    heading = Rotation.from_quat(first[:4]).as_matrix()[:2, 0]
                    last = np.asarray(samples[-1]["root_pose"])
                    distance = float(
                        (last[:2, 3] - first[4:6]) @ (heading / np.linalg.norm(heading))
                    )
                    assert (
                        len(samples) == 100
                        and distance >= 0.4
                        and abs(distance - sample["measurements"]["displacement_x"])
                        < 1e-5
                    )
                    metrics = dict(sample["measurements"], frames=len(samples))
            rows.append(
                dict(
                    profile=profile,
                    source=source,
                    session=result["session_id"],
                    run=result["run_id"],
                    metrics=metrics,
                    journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root.resolve()
    rows = audit(root)
    (root / "qualification-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Verified {len(rows)} real DexPilot runs")
