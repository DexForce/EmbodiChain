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

"""Recompute the L-03 Go2 initialization diagnosis without a simulation runtime.

Run: python -m scripts.benchmark.tech_reports.reporting.l03_initialization_review
    --root PREFLIGHT_ROOT --manifest SHARED_MANIFEST --output NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

__all__ = ["main"]


def main() -> None:
    """Save every stage's measurements, paired results and input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "analysis-source.py")
    cfg = json.loads(args.manifest.read_text())["robots"]["go2"]
    hashes = {}
    for entry in cfg["files"]:
        digest = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"]
        hashes[entry["path"]] = digest
    stage = Usd.Stage.Open(cfg["usd_path"])
    cache = UsdGeom.XformCache()
    source_joints = []
    for prim in stage.Traverse():
        if not prim.GetName().endswith("calf_joint"):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        rotations = []
        for body, local in (
            (joint.GetBody0Rel().GetTargets()[0], joint.GetLocalRot0Attr().Get()),
            (joint.GetBody1Rel().GetTargets()[0], joint.GetLocalRot1Attr().Get()),
        ):
            matrix = Gf.Matrix4d(1)
            matrix.SetRotate(Gf.Quatd(local))
            world = matrix * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body))
            rotations.append(Rotation.from_matrix(np.asarray(world)[:3, :3].T))
        attr = prim.GetAttribute("state:angular:physics:position")
        source_joints.append(
            {
                "joint": prim.GetName(),
                "geometry_relative_angle_deg": float(
                    np.rad2deg((rotations[0].inv() * rotations[1]).magnitude())
                ),
                "lower_deg": joint.GetLowerLimitAttr().Get(),
                "upper_deg": joint.GetUpperLimitAttr().Get(),
                "authored_joint_position": bool(attr and attr.HasAuthoredValue()),
            }
        )
    rows, summary = [], []
    for case in ("joints", "authored-state", "pose-only", "full"):
        batch = args.root / case
        records = json.loads((batch / "execution.json").read_text())
        for entry in records:
            assert entry["status"] == "completed" and entry["returncode"] == 0
            run = batch / entry["name"]
            meta = json.loads((run / "run.json").read_text())
            assert meta["status"] == "completed"
            assert (
                hashlib.sha256((run / "source.py").read_bytes()).hexdigest()
                == meta["source_snapshot_sha256"]
            )
            with np.load(run / "initialization-trace.npz") as trace:
                run_rows = []
                for label in meta["trace_stages"]:
                    root = trace[label + "_root_pose"]
                    velocity = trace[label + "_root_velocity"]
                    assert np.isfinite(root).all() and np.isfinite(velocity).all()
                    delta = root[:, :3] - cfg["initial_position"]
                    row = {
                        "case": case,
                        "platform": meta["platform"],
                        "num_envs": meta["num_envs"],
                        "stage": label,
                        "root_distance_mm": float(
                            np.linalg.norm(delta, axis=-1).max() * 1000
                        ),
                        "root_rotation_deg": float(
                            np.rad2deg(
                                Rotation.from_quat(root[:, 3:]).magnitude()
                            ).max()
                        ),
                        "root_linear_speed_mm_s": float(
                            np.linalg.norm(velocity[:, :3], axis=-1).max() * 1000
                        ),
                        "root_angular_speed_rad_s": float(
                            np.linalg.norm(velocity[:, 3:], axis=-1).max()
                        ),
                    }
                    rows.append(row)
                    run_rows.append(row)
                summary.append(
                    {
                        "case": case,
                        "platform": meta["platform"],
                        "num_envs": meta["num_envs"],
                        "prepared": run_rows[0],
                        "last": run_rows[-1],
                        "maximum_root_distance_after_init_mm": max(
                            r["root_distance_mm"]
                            for r in run_rows
                            if r["stage"].startswith("step_")
                        ),
                        "run": str(run.resolve()),
                        "steps": meta["recorded_physics_steps"],
                    }
                )
            for path in [
                run / name
                for name in (
                    "run.json",
                    "input.json",
                    "source.py",
                    "states.npz",
                    "model.npz",
                    "initialization-trace.npz",
                )
            ] + [batch / (entry["name"] + ".log"), batch / "execution.json"]:
                hashes[str(path.resolve())] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    pairs = []
    for count in (1, 4):
        paths = [
            args.root / "full" / f"go2_{count:04d}_{p}"
            for p in ("isaaclab", "embodichain")
        ]
        metadata = [json.loads((p / "run.json").read_text()) for p in paths]
        traces = [np.load(p / "initialization-trace.npz") for p in paths]
        a, b = metadata
        links = [b["link_names"].index(name) for name in a["link_names"]]
        joints = [b["joint_names"].index(name) for name in a["joint_names"]]
        distances, angles, qdeltas, root_deltas = [], [], [], []
        for label in ["root_velocity_written"] + [
            f"step_{n:04d}" for n in range(1, 101)
        ]:
            ta, tb = traces
            pa, pb = ta[label + "_link_pose"], tb[label + "_link_pose"][:, links]
            distances.append(
                float(np.linalg.norm(pa[..., :3] - pb[..., :3], axis=-1).max() * 1000)
            )
            rotations = [
                Rotation.from_quat(p[..., 3:].reshape(-1, 4)) for p in (pa, pb)
            ]
            angles.append(
                float(np.rad2deg((rotations[0].inv() * rotations[1]).magnitude()).max())
            )
            qdeltas.append(
                float(
                    np.abs(ta[label + "_qpos"] - tb[label + "_qpos"][:, joints]).max()
                )
            )
            root_deltas.append(
                float(
                    np.linalg.norm(
                        ta[label + "_root_pose"][:, :3]
                        - tb[label + "_root_pose"][:, :3],
                        axis=-1,
                    ).max()
                    * 1000
                )
            )
        pairs.append(
            {
                "num_envs": count,
                "samples": 101,
                "duration_s": 0.5,
                "initial_root_distance_mm": root_deltas[0],
                "max_root_distance_mm": max(root_deltas),
                "max_link_distance_mm": max(distances),
                "max_link_rotation_deg": max(angles),
                "max_joint_position_difference_rad": max(qdeltas),
            }
        )
        for trace in traces:
            trace.close()
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_joints": source_joints,
        "runs": summary,
        "full_reset_pairs": pairs,
        "sha256": hashes,
        "scope": "Initialization diagnosis only; no loading-performance measurement. No resets after time zero.",
    }
    (args.output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "stages.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": summary, "full_reset_pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
