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

"""Recompute P-02 metrics from original poses without importing a simulator.

Run: python -m scripts.benchmark.tech_reports.reporting.p02_analysis --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["box_penetration", "projected_overlap", "analyze_episode"]


def box_penetration(
    a: np.ndarray, sa: np.ndarray, b: np.ndarray, sb: np.ndarray
) -> np.ndarray:
    """Return the minimum separating translation for intersecting oriented boxes.

    Test all 15 SAT axes. Parallel edge cross products carry no constraint.
    Poses are (T,7) xyz+xyzw; extents are full SI side lengths.
    """
    ra = Rotation.from_quat(np.array(a[:, 3:7], dtype=float, copy=True)).as_matrix()
    rb = Rotation.from_quat(np.array(b[:, 3:7], dtype=float, copy=True)).as_matrix()
    aa, bb = ra.swapaxes(1, 2), rb.swapaxes(1, 2)
    cross = np.cross(aa[:, :, None, :], bb[:, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate((aa, bb, cross), axis=1)
    lengths = np.linalg.norm(axes, axis=-1)
    valid = lengths > 1e-10
    axes = axes / np.where(valid, lengths, 1)[..., None]
    ar = np.abs(np.einsum("tki,tij->tkj", axes, ra)) @ (sa / 2)
    br = np.abs(np.einsum("tki,tij->tkj", axes, rb)) @ (sb / 2)
    distance = np.abs(np.einsum("tki,ti->tk", axes, b[:, :3] - a[:, :3]))
    overlap = np.where(valid, ar + br - distance, np.inf)
    return np.maximum(overlap.min(axis=1), 0)


def projected_overlap(
    a: np.ndarray, sa: np.ndarray, b: np.ndarray, sb: np.ndarray
) -> np.ndarray:
    """Test overlap of the exact horizontal projections of two 3D boxes."""
    ra = Rotation.from_quat(np.array(a[:, 3:7], dtype=float, copy=True)).as_matrix()
    rb = Rotation.from_quat(np.array(b[:, 3:7], dtype=float, copy=True)).as_matrix()
    edges = np.concatenate(
        (ra[:, :2, :].swapaxes(1, 2), rb[:, :2, :].swapaxes(1, 2)), axis=1
    )
    axes = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
    length = np.linalg.norm(axes, axis=-1)
    valid = length > 1e-10
    axes /= np.where(valid, length, 1)[..., None]
    ar = np.abs(np.einsum("tki,tij->tkj", axes, ra[:, :2, :])) @ (sa / 2)
    br = np.abs(np.einsum("tki,tij->tkj", axes, rb[:, :2, :])) @ (sb / 2)
    distance = np.abs(np.einsum("tki,ti->tk", axes, b[:, :2] - a[:, :2]))
    return np.all(~valid | (ar + br >= distance), axis=1)


def analyze_episode(path: Path, meta: dict) -> tuple[dict, dict]:
    """Calculate reference errors, independent penetration and geometric events."""
    with np.load(path) as raw:
        data = {k: raw[k] for k in raw.files}
    t, state = data["t"], data["state"]
    assert np.isfinite(state).all()
    np.testing.assert_allclose(np.linalg.norm(state[..., 3:7], axis=-1), 1, atol=2e-5)
    np.testing.assert_allclose(
        state[0, :, :3], data["requested_initial_poses"][:, :3], atol=1e-6
    )
    rotations = (
        Rotation.from_quat(state[..., 3:7].reshape(-1, 4))
        .as_matrix()
        .reshape(*state.shape[:2], 3, 3)
    )
    size = data["body_size"]
    surface = np.broadcast_to(data["support_pose"], (len(t), 7))
    depth = np.zeros((len(t), state.shape[1]))
    for i in range(state.shape[1]):
        depth[:, i] = box_penetration(
            state[:, i, :7], size, surface, data["support_size"]
        )
        for j in range(i):
            pair = box_penetration(state[:, i, :7], size, state[:, j, :7], size)
            depth[:, i] = np.maximum(depth[:, i], pair)
            depth[:, j] = np.maximum(depth[:, j], pair)
    metric = dict(
        run_id=meta["run_id"],
        platform=meta["platform"],
        scene=meta["scene"]["id"],
        solver=meta["solver"],
        phase=meta["phase"],
        evaluation_role=(
            "primary"
            if meta["phase"] == "independent"
            or (
                meta["phase"] == "main"
                and meta["scene"]["id"] not in ("stack_05", "stack_10")
            )
            else (
                "independent_sensitivity"
                if meta["phase"] == "independent_supplement"
                else (
                    "reset_sequence_study"
                    if meta["phase"] in ("main", "supplement")
                    else "diagnostic"
                )
            )
        ),
        dt_s=meta["dt_s"],
        initial=int(path.stem.split("_")[-1]),
        duration_s=float(t[-1]),
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    curves = dict(t=t, penetration_m=depth)
    if meta["scene"]["kind"] == "slope":
        scene = meta["scene"]
        theta = np.deg2rad(scene["angle_deg"])
        # The installed DexUni material has one Coulomb coefficient.
        static_mu = (
            scene["mu_dynamic"]
            if meta["platform"] == "embodichain"
            else scene["mu_static"]
        )
        static = bool(np.tan(theta) <= static_mu)
        a_ref = (
            0.0
            if static
            else 9.81 * (np.sin(theta) - scene["mu_dynamic"] * np.cos(theta))
        )
        downhill, normal = data["downhill"], data["surface_normal"]
        s = (state[:, 0, :3] - state[0, 0, :3]) @ downhill
        v = state[:, 0, 7:10] @ downhill
        fit = (t >= 0.1 - 1e-12) & (t <= 1 + 1e-12)
        short = t <= 1 + 1e-12
        fit_coefficients = np.polyfit(t[fit], v[fit], 1)
        accel = float(fit_coefficients[0])
        s_ref, v_ref = 0.5 * a_ref * t**2, a_ref * t
        radius_normal = np.abs(np.einsum("i,tij->tj", normal, rotations[:, 0])) @ (
            size / 2
        )
        radius_along = np.abs(np.einsum("i,tij->tj", downhill, rotations[:, 0])) @ (
            size / 2
        )
        plane_gap = (state[:, 0, :3] - data["surface_origin"]) @ normal - radius_normal
        along = (state[:, 0, :3] - data["surface_origin"]) @ downhill
        leaves = (along - radius_along > data["support_size"][0] / 2) | (
            along + radius_along < -data["support_size"][0] / 2
        )
        starts_exit = (along + radius_along > data["support_size"][0] / 2) | (
            along - radius_along < -data["support_size"][0] / 2
        )
        assert not starts_exit[short].any(), "Analytic window reaches the ramp boundary"
        tilt = np.rad2deg(np.arccos(np.clip(rotations[:, 0, :, 2] @ normal, -1, 1)))
        window = np.ones(len(t), dtype=bool) if static else short
        metric.update(
            reference_static=static,
            reference_acceleration_m_s2=a_ref,
            fitted_acceleration_m_s2=accel,
            fitted_velocity_intercept_mm_s=float(fit_coefficients[1] * 1000),
            acceleration_error_m_s2=abs(accel - a_ref),
            acceleration_relative_error_pct=(
                None if static else abs(accel - a_ref) / a_ref * 100
            ),
            inferred_dynamic_mu=(
                None
                if static
                else (9.81 * np.sin(theta) - accel) / (9.81 * np.cos(theta))
            ),
            position_max_error_1s_mm=float(
                np.max(np.abs(s[short] - s_ref[short])) * 1000
            ),
            velocity_max_error_1s_mm_s=float(
                np.max(np.abs(v[short] - v_ref[short])) * 1000
            ),
            along_displacement_final_mm=float(s[-1] * 1000),
            drift_max_mm=float(
                np.max(np.linalg.norm(state[window, 0, :3] - state[0, 0, :3], axis=-1))
                * 1000
            ),
            lateral_drift_max_mm=float(
                np.max(np.abs(state[window, 0, 1] - state[0, 0, 1])) * 1000
            ),
            tilt_max_deg=float(tilt[window].max()),
            penetration_max_mm=float(depth[window].max() * 1000),
            penetration_p95_mm=float(
                np.percentile(depth[window].max(axis=1), 95) * 1000
            ),
            signed_plane_gap_min_mm=float(plane_gap[window].min() * 1000),
            geometry_window_s=float(t[window][-1]),
            ramp_edge_first_s=float(t[starts_exit][0]) if starts_exit.any() else None,
            ramp_clear_first_s=float(t[leaves][0]) if leaves.any() else None,
        )
        curves.update(
            along_s_m=s,
            along_v_m_s=v,
            reference_s_m=s_ref,
            reference_v_m_s=v_ref,
            plane_gap_m=plane_gap,
            tilt_deg=tilt,
            ramp_edge=starts_exit,
            ramp_clear=leaves,
        )
    else:
        settle = meta["protocol"]["stack_settle_s"]
        i2 = np.flatnonzero(t >= settle - 1e-12)[0]
        xy = np.linalg.norm(state[:, :, :2] - state[0, :, :2], axis=-1)
        xy_after = np.linalg.norm(state[i2:, :, :2] - state[i2, :, :2], axis=-1)
        tilt = np.rad2deg(np.arccos(np.clip(rotations[..., 2, 2], -1, 1)))
        # These are objective geometric events, not measured force-support edges.
        pairs = []
        for i in range(1, state.shape[1]):
            overlap = projected_overlap(
                state[:, i, :7], size, state[:, i - 1, :7], size
            )
            pairs.append(~overlap | (state[:, i, 2] <= state[:, i - 1, 2]))
        events = np.stack(pairs, axis=1) if pairs else np.zeros((len(t), 0), dtype=bool)
        any_event = events.any(axis=1)
        metric.update(
            layers=state.shape[1],
            xy_drift_full_max_mm=float(xy.max() * 1000),
            xy_drift_observation_max_mm=float(xy_after.max() * 1000),
            tilt_full_max_deg=float(tilt.max()),
            tilt_observation_max_deg=float(tilt[i2:].max()),
            penetration_full_max_mm=float(depth.max() * 1000),
            penetration_max_mm=float(depth[i2:].max() * 1000),
            penetration_p95_mm=float(np.percentile(depth[i2:].max(axis=1), 95) * 1000),
            top_height_loss_final_mm=float((state[0, -1, 2] - state[-1, -1, 2]) * 1000),
            geometric_event_first_s=float(t[any_event][0]) if any_event.any() else None,
            geometric_event_samples=int(any_event.sum()),
            geometric_event_present_final=bool(any_event[-1]),
            geometric_event_during_settle=bool(any_event[: i2 + 1].any()),
            observation_start_s=float(t[i2]),
        )
        curves.update(
            xy_drift_m=xy,
            tilt_deg=tilt,
            adjacent_geometry_events=events,
            center_height_m=state[:, :, 2],
        )
    if "normal_forces" in data:
        force_window = (
            (t >= 0.1) & (t <= 1)
            if meta["scene"]["kind"] == "slope"
            else t >= meta["protocol"]["stack_settle_s"]
        )
        metric["contact_count_max_per_body"] = int(data["contact_counts"].max())
        metric["normal_force_z_mean_sum_N"] = float(
            data["normal_forces"][force_window, :, 2].sum(axis=1).mean()
        )
        metric["friction_force_mean_sum_N"] = np.mean(
            data["friction_forces"][force_window].sum(axis=1), axis=0
        ).tolist()
    return metric, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["main", "supplement", "independent", "independent_supplement"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, manifest = [], []
    for p in sorted(args.runs.glob("*/run.json")):
        meta = json.loads(p.read_text())
        if (
            meta["phase"] not in args.phases
            or meta["status"] != "completed"
            or meta["duration_s"] < args.min_duration
        ):
            continue
        manifest.append(
            dict(
                run_id=meta["run_id"],
                run_json=str(p.resolve()),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
        folder = args.output / "curves" / meta["run_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for ep in sorted(p.parent.glob("episode_??.npz")):
            metric, curves = analyze_episode(ep, meta)
            metrics.append(metric)
            np.savez_compressed(folder / ep.name, **curves)
    (args.output / "episode-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    keys = list(dict.fromkeys(k for m in metrics for k in m))
    with (args.output / "episode-metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(metrics)
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                analysis_source=str(Path(__file__).resolve()),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                runs=manifest,
            ),
            indent=2,
        )
        + "\n"
    )
    for key in sorted(
        {(m["platform"], m["scene"], m["dt_s"], m["solver"]) for m in metrics}
    ):
        group = [
            m
            for m in metrics
            if (m["platform"], m["scene"], m["dt_s"], m["solver"]) == key
        ]
        fields = [
            "acceleration_relative_error_pct",
            "position_max_error_1s_mm",
            "drift_max_mm",
            "xy_drift_full_max_mm",
            "tilt_full_max_deg",
            "penetration_max_mm",
            "geometric_event_first_s",
        ]
        print(
            key,
            len(group),
            {
                k: (
                    None
                    if not any(g.get(k) is not None for g in group)
                    else round(
                        float(np.mean([g[k] for g in group if g.get(k) is not None])), 5
                    )
                )
                for k in fields
            },
        )


if __name__ == "__main__":
    main()
