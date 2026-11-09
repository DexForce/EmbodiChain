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

"""Compare isolated physics traces in physical units using only NumPy.

This experimental gate measures motion closeness. It does not require identical
internal model trees, physical-array bytes, velocities or solver implementations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__: list[str] = []


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _orientation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result: dict[str, Any] = {
        "count": int(values.size),
        "nonfinite_count": int((~finite).sum()),
        "per_trajectory": [float(x) if np.isfinite(x) else None for x in values],
    }
    for key in ("mean", "median", "p90", "p95", "max"):
        result[key] = None
    if bool(finite.all()):
        result.update(
            mean=float(values.mean()),
            median=float(np.median(values)),
            p90=float(np.quantile(values, 0.9)),
            p95=float(np.quantile(values, 0.95)),
            max=float(values.max()),
        )
    return result


def _first(mask: np.ndarray) -> list[int | None]:
    return [int(np.flatnonzero(row)[0]) if bool(row.any()) else None for row in mask.T]


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    receipt = json.loads(path.with_suffix(".json").read_text())
    if _hash(path) != receipt["trace_sha256"]:
        raise ValueError(f"Trace hash differs from receipt: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        data = {name: arrays[name].astype(np.float64) for name in arrays.files}
    if data["q"].shape[:2] != data["qd"].shape[:2]:
        raise ValueError("Position/velocity time and trajectory dimensions differ")
    if data["q"].ndim != 3 or data["q"].shape[0] < 101:
        raise ValueError("Expected time-major traces with at least 100 transitions")
    return data, receipt


def _metrics(
    robot: str,
    ref: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    metadata: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rq, q = ref["q"], actual["q"]
    rv, v = ref["qd"], actual["qd"]
    if rq.shape != q.shape or rv.shape != v.shape:
        raise ValueError("Trace state shapes differ")
    metrics = {}
    if robot == "cartpole":
        delta = q[..., 1] - rq[..., 1]
        metrics.update(
            slider_position_abs_m=np.abs(q[..., 0] - rq[..., 0]),
            pole_angle_abs_rad=np.abs(np.arctan2(np.sin(delta), np.cos(delta))),
            slider_velocity_abs_m_per_s=np.abs(v[..., 0] - rv[..., 0]),
            pole_velocity_abs_rad_per_s=np.abs(v[..., 1] - rv[..., 1]),
        )
        # A fixed massless root is not an independently moving free base.
        interpretation = {"root_pose": "not applicable: fixed Cartpole root"}
    else:
        metrics.update(
            joint_angle_MAE_rad=np.abs(q[..., 7:] - rq[..., 7:]).mean(-1),
            root_position_L2_m=np.linalg.norm(q[..., :3] - rq[..., :3], axis=-1),
            root_orientation_geodesic_rad=_orientation(q[..., 3:7], rq[..., 3:7]),
            joint_velocity_MAE_rad_per_s=np.abs(v[..., 6:] - rv[..., 6:]).mean(-1),
            root_linear_velocity_L2_m_per_s=np.linalg.norm(
                v[..., :3] - rv[..., :3], axis=-1
            ),
            root_angular_velocity_L2_rad_per_s=np.linalg.norm(
                v[..., 3:6] - rv[..., 3:6], axis=-1
            ),
        )
        for index in range(8):
            metrics[f"joint_{index}_angle_abs_rad"] = np.abs(
                q[..., 7 + index] - rq[..., 7 + index]
            )
        interpretation = {"joint_angles": "eight limited scalar joints; no wrapping"}
    common_names = []
    if "body_q" in ref and "body_q" in actual:
        names_r, names_a = (row["body_names"] for row in metadata)
        common_names = [name for name in names_r if name in names_a]
        if not common_names:
            raise ValueError("No common named links in body traces")
        errors = []
        for name in common_names:
            rbody = ref["body_q"][..., names_r.index(name), :]
            body = actual["body_q"][..., names_a.index(name), :]
            error = np.linalg.norm(body[..., :3] - rbody[..., :3], axis=-1)
            metrics[f"link_{name}_position_L2_m"] = error
            errors.append(error)
            if robot == "cartpole" and name == "pole":
                tip = body[..., :3] + _rotate(body[..., 3:], np.array([0.0, 0.0, 1.0]))
                rtip = rbody[..., :3] + _rotate(
                    rbody[..., 3:], np.array([0.0, 0.0, 1.0])
                )
                metrics["pole_tip_position_L2_m"] = np.linalg.norm(tip - rtip, axis=-1)
                errors.append(metrics["pole_tip_position_L2_m"])
        metrics["key_link_max_position_L2_m"] = np.stack(errors).max(0)
    interpretation.update(
        common_named_links=common_names,
        link_matching="Semantic name intersection; extra fixed or collapsed bodies do not require identical trees",
        prefix="Within each trajectory mean over steps 1..H; distributions across all requested trajectories",
        endpoint="At step H; distributions across all requested trajectories",
        selection="No failed or nonfinite trajectories are dropped",
        velocity="Descriptive diagnostics; not an automatic motion gate",
    )
    return metrics, interpretation


def _compare(reference: Path, actual: Path, thresholds: Path) -> dict[str, Any]:
    ref, rm = _load(reference)
    data, am = _load(actual)
    if rm["robot"] != am["robot"] or rm["inputs_sha256"] != am["inputs_sha256"]:
        raise ValueError("Traces must have the same robot and input SHA256")
    robot = rm["robot"]
    limits = json.loads(thresholds.read_text())
    metrics, interpretation = _metrics(robot, ref, data, (rm, am))
    initial_keys = (
        ["slider_position_abs_m", "pole_angle_abs_rad"]
        if robot == "cartpole"
        else [
            "joint_angle_MAE_rad",
            "root_position_L2_m",
            "root_orientation_geodesic_rad",
        ]
    )
    if any(np.max(metrics[name][0]) > 1e-5 for name in initial_keys):
        raise ValueError("Canonical initial states differ beyond 1e-5 physical units")
    if not np.allclose(data["qd"][0], ref["qd"][0], rtol=1e-5, atol=1e-6):
        raise ValueError("Canonical initial velocities differ")
    finite_masks = []
    for trace in (ref, data):
        mask = np.isfinite(trace["q"]).all(-1) & np.isfinite(trace["qd"]).all(-1)
        if robot == "ant":
            mask &= np.linalg.norm(trace["q"][..., 3:7], axis=-1) > 0
        if "body_q" in trace:
            mask &= np.isfinite(trace["body_q"]).all((-1, -2))
            mask &= (np.linalg.norm(trace["body_q"][..., 3:], axis=-1) > 0).all(-1)
        finite_masks.append(mask)
    horizon_results, checks = {}, []
    for h in (1, 10, 100):
        valid = finite_masks[0][: h + 1].all(0) & finite_masks[1][: h + 1].all(0)
        row = {
            "finite_trajectories": int(valid.sum()),
            "nonfinite_trajectories": int((~valid).sum()),
            "endpoint": {key: _summary(value[h]) for key, value in metrics.items()},
            "prefix_mean": {
                key: _summary(value[1 : h + 1].mean(0))
                for key, value in metrics.items()
            },
        }
        checks.append(
            {
                "horizon": h,
                "check": "all requested trajectories finite",
                "passed": bool(valid.all()),
            }
        )
        for key, bounds in limits["robots"][robot]["horizons"][str(h)].items():
            for kind in ("endpoint", "prefix_mean"):
                for statistic, bound in bounds.items():
                    value = row[kind].get(key, {}).get(statistic)
                    checks.append(
                        {
                            "horizon": h,
                            "metric": key,
                            "aggregation": kind,
                            "statistic": statistic,
                            "value": value,
                            "upper_bound": bound,
                            "passed": value is not None and value <= bound,
                        }
                    )
        horizon_results[str(h)] = row
    failures = {}
    for label, trace in (("reference", ref), ("actual", data)):
        if robot == "cartpole":
            task = (np.abs(trace["q"][..., 0]) > 4) | (np.abs(trace["qd"]) > 10).any(-1)
            criterion = "official Cartpole: abs(slider)>4m or abs(either velocity)>10 (respective units)"
        else:
            task = trace["q"][..., 2] < 0.2
            criterion = "descriptive torso/root height below 0.2m; falling is retained, not automatic rejection"
        failures[label] = {
            "first_nonfinite_step": _first(
                ~finite_masks[0 if label == "reference" else 1]
            ),
            "first_task_failure_step": _first(task),
            "task_failure_criterion": criterion,
            "max_abs_q": (
                float(np.nanmax(np.abs(trace["q"])))
                if np.isfinite(trace["q"]).all()
                else None
            ),
            "max_abs_qd": (
                float(np.nanmax(np.abs(trace["qd"])))
                if np.isfinite(trace["qd"]).all()
                else None
            ),
        }
    passed = all(row["passed"] for row in checks)
    return {
        "schema": "embodichain.physics_trajectory_closeness.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "motion_closeness_passed"
            if passed
            else "motion_closeness_requires_analysis"
        ),
        "motion_gate_passed": passed,
        "robot": robot,
        "requested_trajectories": int(data["q"].shape[1]),
        "horizons": horizon_results,
        "checks": checks,
        "failures": failures,
        "interpretation": interpretation,
        "reference_receipt": rm,
        "actual_receipt": am,
        "bound_files_sha256": {
            str(path): _hash(path)
            for path in (
                reference,
                reference.with_suffix(".json"),
                actual,
                actual.with_suffix(".json"),
                thresholds,
                Path(__file__),
            )
        },
        "qualification": "Small fixed-input teacher motion gate only; not a neural accuracy, same-data convergence or native inference qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = _compare(args.reference, args.actual, args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "sha256": _hash(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
