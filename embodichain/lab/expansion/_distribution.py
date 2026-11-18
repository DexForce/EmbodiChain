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

"""Offline confirmed-episode distribution analysis and linked HTML report.

Run with ``python -m embodichain.lab.expansion._distribution --help``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

__all__: list[str] = []


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _cell(pose: list[float], cell_size: float) -> str:
    return ",".join(str(int(x)) for x in np.floor(np.asarray(pose[:2]) / cell_size))


def _motion(points: np.ndarray, times: np.ndarray) -> dict:
    if points.ndim != 2 or points.shape[0] != len(times) or len(times) < 2:
        raise ValueError("Motion requires aligned T+1 points and timestamps")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Motion must be finite with increasing timestamps")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    grid = np.linspace(0, arc[-1], 32)
    path = np.stack(
        [np.interp(grid, arc[keep], points[keep, i]) for i in range(points.shape[1])],
        axis=1,
    )
    return {
        "path_length": float(arc[-1]),
        "duration_s": float(times[-1] - times[0]),
        "speed_p50_p95": np.quantile(lengths / np.diff(times), [0.5, 0.95]).tolist(),
        "path": path.tolist(),
    }


def _summarize(rows: list[dict], size: float, minimum: int) -> dict:
    unique = {}
    for row in rows:
        key = row["commit_id"]
        if key in unique and unique[key] != row:
            raise ValueError("Conflicting commit identity")
        unique[key] = deepcopy(row)
    rows = list(unique.values())
    pick, place, joint = Counter(), Counter(), Counter()
    for row in rows:
        if row["object_motion"] is None:
            continue
        a, b = _cell(row["initial_pose"], size), _cell(row["final_pose"], size)
        pick[a] += 1
        place[b] += 1
        joint[a + "|" + b] += 1
        row["pick_cell"], row["place_cell"] = a, b
    distributions = {}
    for family in ("object_motion", "joint_motion", "tcp_motion"):
        for metric in ("path_length", "duration_s"):
            values = [
                r[family][metric] for r in rows if r.get(family) and metric in r[family]
            ]
            distributions[family + "." + metric] = {
                "valid_count": len(values),
                "missing_count": len(rows) - len(values),
                "p0_p25_p50_p75_p100": (
                    np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
                    if values
                    else None
                ),
            }
    orientation_counts = {}
    for phase in ("initial", "final"):
        orientation_counts[phase] = dict(
            Counter(
                r["orientation"]["rotation_bin_" + phase]
                for r in rows
                if r.get("orientation")
            )
        )
    return {
        "orientation": orientation_counts,
        "tcp_valid": sum(r.get("tcp_motion") is not None for r in rows),
        "motion_distributions": distributions,
        "episodes": rows,
        "count": len(rows),
        "layout_valid": sum(pick.values()),
        "layout_missing": len(rows) - sum(pick.values()),
        "pick": dict(pick),
        "place": dict(place),
        "joint": dict(joint),
        "joint_at_minimum": sum(n >= minimum for n in joint.values()),
        "unique_geometry_buckets": len(
            {r["geometry_bucket"] for r in rows if r["geometry_bucket"]}
        ),
        "geometry_missing": sum(r["geometry_bucket"] is None for r in rows),
    }


def _read(paths: list[Path], entity: str) -> tuple[list[dict], list[str]]:
    from .persistence import ArtifactEpisodeSink, load_expansion_episode

    from ._distribution_pose import RecordedTCP, orientation

    rows, domains = [], []
    for directory in dict.fromkeys(p.resolve() for p in paths):
        if not (directory / "job.json").is_file():
            raise ValueError(f"Expected confirmed job directory: {directory}")
        job = json.loads((directory / "job.json").read_text())
        deployment = job.get("resolved_deployment", {})
        tcp = RecordedTCP(deployment)
        for receipt in ArtifactEpisodeSink(directory).inspect():
            if not receipt.confirmed:
                continue
            episode = load_expansion_episode(receipt.storage_id)
            if not episode.validation.accepted:
                raise ValueError("Confirmed episode lacks acceptance")
            meta = episode.metadata
            # Fixed topology, installation and task identity; sampled object poses are features.
            robot = [
                dict(uid=s["uid"], joints=list(s["joint_names"]), pose=list(s["pose"]))
                for s in meta.get("initial_state", ())
                if s["kind"] == "robot"
            ]
            domain = _digest(
                {
                    "robot": robot,
                    "instruction": meta.get("instruction"),
                    "backend": meta.get("physics_backend"),
                    "deployment_robot": deployment.get("robot"),
                    "scene_topology": [
                        (v["uid"], v["kind"], v["body_type"])
                        for v in meta.get("initial_state", ())
                    ],
                    "scene_configuration": {
                        k: deployment.get(k)
                        for k in ("rigid_object", "background", "physics_config")
                    },
                }
            )
            domains.append(domain)
            times = episode.timestamps.cpu().numpy()
            state = episode.observations.get(f"rigid/{entity}/state")
            motion = (
                None if state is None else _motion(state.cpu().numpy()[:, :3], times)
            )
            q = episode.observations.get("full_joint_positions")
            joint_motion = None if q is None else _motion(q.cpu().numpy(), times)
            descriptor = meta.get("coverage_descriptor")
            bucket = None
            if descriptor:
                bucket = _digest(
                    [
                        list(descriptor["phase_ids"]),
                        np.rint(np.asarray(descriptor["geometry"]) / 0.001)
                        .astype(int)
                        .tolist(),
                    ]
                )
            tcp_poses = tcp.poses(episode)
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "commit_id": episode.commit_id,
                    "artifact": receipt.storage_id,
                    "job": str(directory),
                    "domain_id": domain,
                    "source_episode_uuid": meta.get("source_episode_uuid"),
                    "initial_pose": None if state is None else state[0, :7].tolist(),
                    "final_pose": None if state is None else state[-1, :7].tolist(),
                    "orientation": (
                        None
                        if state is None
                        else orientation(state.cpu().numpy(), times)
                    ),
                    "tcp_motion": (
                        None if tcp_poses is None else _motion(tcp_poses[:, :3], times)
                    ),
                    "tcp_orientation": (
                        None if tcp_poses is None else orientation(tcp_poses, times)
                    ),
                    "tcp_evidence": (
                        {"unavailable": tcp.reason} if tcp.reason else tcp.evidence
                    ),
                    "object_motion": motion,
                    "joint_motion": joint_motion,
                    "geometry_bucket": bucket,
                }
            )
    return rows, sorted(set(domains))


def _run(
    before: list[Path],
    after: list[Path],
    output: Path,
    entity: str,
    size: float,
    minimum: int,
) -> None:
    if not np.isfinite(size) or size <= 0 or minimum < 1:
        raise ValueError("Positive cell size and minimum count required")
    if output.exists():
        raise FileExistsError("Use a new output directory")
    b, bd = _read(before, entity)
    a, ad = _read(after, entity)
    if not b or not a or bd != ad or len(bd) != 1:
        raise ValueError(
            "Comparison requires nonempty snapshots of one matching domain"
        )
    report = {
        "schema_version": 2,
        "feature_version": "se3-world-object-joint-tcp-v2",
        "entity": entity,
        "coordinate_frame": "world",
        "cell_size_m": size,
        "minimum": minimum,
        "domain_id": bd[0],
        "before": _summarize(b, size, minimum),
        "after": _summarize(a, size, minimum),
    }
    report["comparison_kind"] = (
        "cumulative"
        if {r["commit_id"] for r in b} <= {r["commit_id"] for r in a}
        else "independent_cohorts"
    )
    keys = sorted(set(report["before"]["joint"]) | set(report["after"]["joint"]))
    report["comparison"] = [
        {
            "cell": k,
            "before": report["before"]["joint"].get(k, 0),
            "after": report["after"]["joint"].get(k, 0),
            "deficit": max(0, minimum - report["after"]["joint"].get(k, 0)),
        }
        for k in keys
    ]
    report["limitations"] = [
        "XY bins cover observed domain only; no feasible-volume denominator.",
        "TCP is recorded-URDF FK from measured qpos/root plus explicit gripper TCP calibration.",
        "Orientation bins: canonical SO(3) rotation vectors, 30-degree Cartesian bins, not equal-volume; no symmetry folding.",
        "Raw terminal pose bins include tracking variation; receipt-confirmed fixed task categories are a separate coverage metric.",
        "Joint path length mixes joint coordinates; compare only matching domains.",
        "Geometry buckets reuse existing 1e-3 descriptor quantization, not clustering.",
    ]
    report["snapshot_id"] = _digest(report)
    output.mkdir(parents=True)
    # Detail pages are shared by the web viewer and Agent; summaries retain only an index.
    pages = output / "pages"
    pages.mkdir()
    for name in ("before", "after"):
        rows = report[name]["episodes"]
        for start in range(0, len(rows), 100):
            chunk = rows[start : start + 100]
            payload = json.dumps(chunk, ensure_ascii=False, allow_nan=False).replace(
                "<", "\\u003c"
            )
            key = f"{name}-{start//100}"
            (pages / f"{key}.json").write_text(payload)
            (pages / f"{key}.js").write_text(
                f"window.distributionPage({json.dumps(key)},{payload});"
            )
        with (output / f"{name}-episodes.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report[name]["episodes"] = [
            dict(
                episode_id=row["episode_id"],
                commit_id=row["commit_id"],
                initial_pose=row["initial_pose"],
                final_pose=row["final_pose"],
                object_motion=row["object_motion"] is not None,
                pick_cell=row.get("pick_cell"),
                place_cell=row.get("place_cell"),
                page=f"{name}-{i//100}",
            )
            for i, row in enumerate(rows)
        ]
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    template = Path(__file__).with_name("_distribution.html").read_text()
    (output / "report.html").write_text(
        template.replace(
            "__REPORT__",
            json.dumps(report, ensure_ascii=False, allow_nan=False).replace(
                "<", "\\u003c"
            ),
        )
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity", default="cube")
    parser.add_argument("--cell-size", type=float, default=0.1)
    parser.add_argument("--minimum", type=int, default=5)
    args = parser.parse_args()
    _run(
        args.before, args.after, args.output, args.entity, args.cell_size, args.minimum
    )


if __name__ == "__main__":
    _main()
