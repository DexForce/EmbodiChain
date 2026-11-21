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

"""Compare the PGS baseline with preserved TGS data using independent references."""

from __future__ import annotations

import os

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp

from scripts.benchmark.tech_reports.reporting.p01_analysis import measure, reference

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/pgs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plan_path = ROOT / "configs/P-01/pgs-baseline.json"
    plan = json.loads(plan_path.read_text())
    cfg = json.loads((ROOT / "configs/P-01/isaaclab.json").read_text())
    entries = [
        (plan["run_prefix"] + r["id"], r["group"])
        for r in plan["runs"]
        if r["group"] != "capture"
    ]
    entries += [
        (f"20261014_main_{case}_dt{tag}", "baseline")
        for case in ["freefall", "pendulum"]
        for tag in ["010", "005", "0025"]
    ]
    entries += [
        (f"20261014_study_{case}_dt001_p1", "common_configuration")
        for case in ["freefall", "pendulum"]
    ]
    run_root = ROOT / "results/P-01"
    theta0 = np.array(
        [
            np.load(
                run_root / f"{plan['run_prefix']}pendulum_dt005/episode_{i:02d}.npz"
            )["state"][0, 0]
            for i in range(10)
        ]
    )
    m, g, length, radius = (
        cfg[k] for k in ["mass_kg", "gravity_m_s2", "pendulum_length_m", "bob_radius_m"]
    )
    inertia = m * (length**2 + 0.4 * radius**2)
    coefficient = m * g * length / inertia
    sample_dt = 0.0005
    expected = reference(theta0, 10.0, cfg["reference_dt_s"], sample_dt, coefficient)
    refined = reference(
        theta0, 10.0, cfg["reference_refined_dt_s"], sample_dt, coefficient
    )
    times = np.arange(len(expected)) * sample_dt
    independent = solve_ivp(
        lambda _, y: [y[1], -coefficient * np.sin(y[0])],
        [0, 10],
        [theta0[0], 0.0],
        t_eval=times,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert independent.success
    check = {
        "refinement_max_angle_rad": float(
            np.max(np.abs(expected[:, :, 0] - refined[:, :, 0]))
        ),
        "refinement_max_velocity_rad_s": float(
            np.max(np.abs(expected[:, :, 1] - refined[:, :, 1]))
        ),
        "dop853_max_angle_rad": float(
            np.max(np.abs(expected[:, 0, 0] - independent.y[0]))
        ),
    }
    assert max(check.values()) < 1e-8, check
    np.savez_compressed(
        OUT / "reference.npz", t=times, state=expected, refined_state=refined
    )
    rows, summaries, manifests, traces = [], [], {}, {}
    metadata = {}
    for run_id, group in entries:
        run = run_root / run_id
        meta = json.loads((run / "run.json").read_text())
        metadata[run_id] = meta
        assert meta["status"] == "completed" and meta["completed_episodes"] == 10
        assert meta["duration_s"] == 10 and meta["device"] == "cuda:0"
        assert meta["velocity_iterations"] == 1
        case, dt = meta["case"], meta["dt_s"]
        solver = meta["solver"].split()[-1]
        if solver == "PGS":
            scene = meta["physics_scene_attributes"]["/physicsScene"]
            assert scene["physxScene:solverType"] == "PGS"
            assert scene["physxScene:enableGPUDynamics"]
            assert scene["physxScene:minPositionIterationCount"] == 4
            assert scene["physxScene:maxPositionIterationCount"] == 4
            assert scene["physxScene:minVelocityIterationCount"] == 1
            assert scene["physxScene:maxVelocityIterationCount"] == 1
        manifest = {
            "run_json_sha256": hashlib.sha256(
                (run / "run.json").read_bytes()
            ).hexdigest(),
            "episodes": {},
        }
        subset = []
        for i in range(10):
            file = run / f"episode_{i:02d}.npz"
            a = np.load(file)
            t, state = a["t"], a["state"]
            assert state.shape == (round(10 / dt) + 1, 2) and np.isfinite(state).all()
            np.testing.assert_allclose(t, np.arange(len(t)) * dt, rtol=0, atol=1e-12)
            assert state[0, 1] == 0
            if case == "pendulum":
                assert state[0, 0] == theta0[i]
                ref = expected[:: round(dt / sample_dt), i]
                geometric = np.column_stack(
                    (
                        -length * np.sin(state[:, 0]),
                        np.zeros(len(t)),
                        1 - length * np.cos(state[:, 0]),
                    )
                )
                assert (
                    np.max(np.linalg.norm(a["com_position"] - geometric, axis=1)) < 1e-5
                )
            else:
                ref = np.column_stack((state[0, 0] - g * t**2 / 2, -g * t))
            row = dict(
                run_id=run_id,
                group=group,
                solver=solver,
                case=case,
                dt_s=dt,
                position_iterations=meta["position_iterations"],
                velocity_iterations=1,
                initial_id=i,
                initial=float(state[0, 0]),
                **measure(case, t, state, ref, cfg),
            )
            if case == "freefall":
                n = meta["position_iterations"] if solver == "TGS" else 1
                predicted_bias = -g * dt * t / (2 * n)
                row["fall_bias_model_max_residual_0_1s_m"] = float(
                    np.max(np.abs((state[:, 0] - ref[:, 0] - predicted_bias)[t <= 1]))
                )
            rows.append(row)
            subset.append(row)
            manifest["episodes"][file.name] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            if i == 0:
                traces[solver, case, dt] = (t, state, ref)
        summary = {
            k: subset[0][k]
            for k in [
                "run_id",
                "group",
                "solver",
                "case",
                "dt_s",
                "position_iterations",
                "velocity_iterations",
            ]
        }
        summary.update(
            trials=10, candidate_pass_count=sum(r["candidate_pass"] for r in subset)
        )
        for key, value in subset[0].items():
            if key in summary or key in [
                "initial_id",
                "initial",
                "candidate_pass",
                "first_pointwise_threshold_failure_s",
            ]:
                continue
            if isinstance(value, (float, int)):
                values = [r[key] for r in subset]
                summary[key] = dict(
                    mean=float(np.mean(values)),
                    min=float(min(values)),
                    max=float(max(values)),
                )
        summaries.append(summary)
        manifests[run_id] = manifest

    # Same models and initial states; only solver selection differs in the primary comparison.
    paired_checks = []
    for case in ["freefall", "pendulum"]:
        for tag in ["010", "005", "0025"]:
            pgs_id, tgs_id = (
                f"{plan['run_prefix']}{case}_dt{tag}",
                f"20261014_main_{case}_dt{tag}",
            )
            a, b = metadata[pgs_id], metadata[tgs_id]
            keys = [
                k
                for k in a
                if k.startswith("effective_body_") or k.startswith("effective_joint_")
            ]
            keys += [
                "dt_s",
                "duration_s",
                "position_iterations",
                "velocity_iterations",
                "device",
                "packages",
                "isaaclab_commit",
                "gpu_inventory",
            ]
            for key in keys:
                assert a[key] == b[key], (pgs_id, key)
            for path in [
                "assets/P-01/initial-states.json",
                "configs/P-01/isaaclab.json",
            ]:
                assert a["hashes"][path] == b["hashes"][path]
            ar = [r for r in rows if r["run_id"] == pgs_id]
            br = [r for r in rows if r["run_id"] == tgs_id]
            assert [r["initial"] for r in ar] == [r["initial"] for r in br]
            paired_checks.append(
                dict(
                    pgs=pgs_id,
                    tgs=tgs_id,
                    matching_fields=keys,
                    exact_initial_match=True,
                )
            )
    old = np.load(run_root / "20261014_study_pgs_dt005_p4/episode_00.npz")
    new = np.load(run_root / f"{plan['run_prefix']}pendulum_dt005/episode_00.npz")
    assert all(np.array_equal(old[k], new[k]) for k in old.files)
    with (OUT / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                metric_script_sha256=hashlib.sha256(
                    (ROOT / "code/analysis/p01_analysis.py").read_bytes()
                ).hexdigest(),
                input_runs=manifests,
                reference_check=check,
                paired_checks=paired_checks,
                old_pgs_initial0_exact_reproduction=True,
                pgs_baseline_trajectories=60,
                pgs_common_configuration_trajectories=20,
                tgs_reused_trajectories=80,
                performance_comparison=False,
            ),
            indent=2,
        )
        + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(ROOT / "assets/fonts/NotoSansSC-Regular.ttf"))
    plt.rcParams.update(
        {
            "font.family": "Noto Sans SC",
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    colors = {"PGS": "#168575", "TGS": "#BA543B"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, case, key, scale, threshold, title, ylabel in [
        (
            axes[0],
            "freefall",
            "position_max_0_1s_m",
            1000,
            5,
            "A  自由落体：前 1 s",
            "最大位置误差（mm）",
        ),
        (
            axes[1],
            "pendulum",
            "energy_max_relative_drift",
            100,
            2,
            "B  单摆：完整 10 s",
            "最大相对能量漂移（%）",
        ),
    ]:
        for solver in ["PGS", "TGS"]:
            groups = sorted(
                [
                    s
                    for s in summaries
                    if s["solver"] == solver
                    and s["case"] == case
                    and s["position_iterations"] == 4
                ],
                key=lambda s: s["dt_s"],
            )
            y = np.array([s[key]["mean"] * scale for s in groups])
            low, high = (
                np.array([s[key][v] * scale for s in groups]) for v in ["min", "max"]
            )
            ax.errorbar(
                [s["dt_s"] * 1000 for s in groups],
                y,
                yerr=[y - low, high - y],
                fmt="o-",
                capsize=3,
                color=colors[solver],
                label=f"{solver} 4／1",
            )
        ax.axhline(threshold, color="#68737E", ls=":", label=f"判据 {threshold:g}")
        ax.set(
            xlabel="物理步长（ms）",
            ylabel=ylabel,
            title=title,
            yscale="log",
            xticks=[1, 2.5, 5, 10],
        )
    for solver in ["PGS", "TGS"]:
        t, state, _ = traces[solver, "pendulum", 0.005]
        e = inertia * state[:, 1] ** 2 / 2 + m * g * length * (1 - np.cos(state[:, 0]))
        axes[2].plot(
            t, (e / e[0] - 1) * 100, color=colors[solver], label=f"{solver} 4／1"
        )
    axes[2].axhline(2, color="#68737E", ls=":", label="+2% 判据")
    axes[2].axhline(-2, color="#68737E", ls=":")
    axes[2].set(
        xlabel="物理时间（s）",
        ylabel="相对初始能量变化（%）",
        title="C  单摆：5 ms，初态 0",
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"solver_comparison.{ext}")
    plt.close(fig)
    print(
        json.dumps(
            [
                dict(
                    solver=s["solver"],
                    case=s["case"],
                    dt_s=s["dt_s"],
                    position_iterations=s["position_iterations"],
                    passes=s["candidate_pass_count"],
                    metric=s.get(
                        "energy_max_relative_drift", s.get("position_max_0_1s_m")
                    ),
                )
                for s in summaries
            ],
            indent=2,
        ),
        flush=True,
    )
    capture_media(plan)


def capture_media(plan: dict) -> None:
    """Verify native PGS recording against the baseline, then copy and encode it."""
    run = ROOT / "results/P-01" / (plan["run_prefix"] + "capture_pendulum_dt005")
    baseline = ROOT / "results/P-01" / (plan["run_prefix"] + "pendulum_dt005")
    recorded, original = (
        json.loads((p / "run.json").read_text()) for p in [run, baseline]
    )
    assert recorded["status"] == "completed"
    keys = [
        k
        for k in original
        if k.startswith("effective_body_") or k.startswith("effective_joint_")
    ]
    keys += [
        "dt_s",
        "duration_s",
        "solver",
        "position_iterations",
        "velocity_iterations",
        "packages",
        "isaaclab_commit",
    ]
    assert all(recorded[k] == original[k] for k in keys)
    a, b = (np.load(p / "episode_00.npz") for p in [run, baseline])
    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    source = run / "native_camera"
    capture = json.loads((source / "capture.json").read_text())
    assert len(capture["frames"]) == 400
    frame_hashes, center_errors = {}, []
    for frame in capture["frames"]:
        assert np.array_equal(frame["state"], a["state"][frame["step"]])
        assert abs(frame["time_s"] - a["t"][frame["step"]]) < 1e-12
        path = source / frame["file"]
        rgb = np.asarray(Image.open(path)).astype(float)
        mask = (
            (rgb[..., 0] < 100)
            & (rgb[..., 2] > rgb[..., 0] + 50)
            & (rgb[..., 1] > rgb[..., 0] + 30)
        )
        y, x = np.nonzero(mask)
        assert len(x) > 100
        eye, target = np.array(frame["camera_eye"]), np.array(frame["camera_target"])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        rel = a["com_position"][frame["step"]] - eye
        projected = np.array([640, 360]) + (1280 * 40 / 36) * np.array(
            [right @ rel, -up @ rel]
        ) / (forward @ rel)
        error = float(np.linalg.norm(projected - [x.mean(), y.mean()]))
        assert error < 8, (frame["file"], error)
        center_errors.append(error)
        frame_hashes[frame["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(source / "native_1x.mp4", OUT / "pendulum_pgs_1x.mp4")
    shot = capture["frames"][0]
    shutil.copy2(source / shot["file"], OUT / "pendulum_pgs_t0005.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(OUT / "pendulum_pgs_1x.mp4"),
            "-filter_complex",
            "fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
            "-loop",
            "0",
            str(OUT / "pendulum_pgs_1x.gif"),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(OUT / "pendulum_pgs_1x.mp4"),
            ]
        )
    )
    manifest = dict(
        run=str(run.relative_to(ROOT)),
        baseline=str(baseline.relative_to(ROOT)),
        exact_array_equality=True,
        matching_config_keys=keys,
        screenshot=shot,
        capture={k: v for k, v in capture.items() if k != "frames"},
        raw_frame_sha256=frame_hashes,
        max_projected_center_error_px=max(center_errors),
        video_probe=probe,
        input_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "run.json",
                run / "episode_00.npz",
                source / "capture.json",
                baseline / "episode_00.npz",
            ]
        },
        outputs_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.glob("pendulum_pgs*")
        },
    )
    (OUT / "native-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
